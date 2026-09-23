#!/usr/bin/env python3
"""Ego 数据二元质检流程：抽帧 -> 提示词 -> 本地千问推理 -> 反证审计 -> 聚合 -> 评测"""
import argparse, base64, collections, hashlib, itertools, json, os, re, subprocess, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

# ★ v23：数据集参数化。用 QC_DATASET 选 set01~set05，或用 QC_DATASET_ROOT 直接给路径
_D5X100 = '/mnt/checkpoint/zhn/cus000001_ego_qc_balanced_5x100_v2'
DATASETS = {f'set{i:02d}': f'{_D5X100}/cus000001_ego_qc_set_{i:02d}_100' for i in range(1, 6)}
DATASET = os.environ.get('QC_DATASET', 'set01')
ROOT = os.environ.get('QC_DATASET_ROOT') or DATASETS.get(DATASET, DATASETS['set01'])
OUT = '/data1/review/egoqc'
FRAMES = os.path.join(OUT, 'frames')
# ★ 豆包（火山方舟）版本：推理走公网 API，不再是本地 vLLM。
# 抽帧方案、提示词、口径、聚合规则与 v21 完全一致，只换后端，保证可比。
ARK_BASE = os.environ.get('ARK_BASE', 'https://ark.cn-beijing.volces.com/api/coding/v3')
API_KEY = os.environ.get('ARK_API_KEY', '')
MODEL = os.environ.get('ARK_MODEL', 'doubao-seed-2.0-pro')
SEND_DETAIL = os.environ.get('ARK_DETAIL', '0') == '1'   # 方舟不一定认 detail 字段
BACKEND = 'ark:' + MODEL
# 思考模式：auto=不传 thinking 字段（对齐 v21 默认行为）；'1'=enabled；'0'=disabled
THINK_MODE = 'auto'

ENDPOINTS = [ARK_BASE.rstrip('/') + '/chat/completions'] * 8
_ep_cycle = itertools.cycle(ENDPOINTS)
_ep_lock = threading.Lock()

def next_endpoint():
    with _ep_lock:
        return next(_ep_cycle)
DETAILS = '/tmp/p7/invalid_details_dedup.tsv'
# 每次改提示词/聚合规则都必须递增此版本号；结果文件会记录版本号与提示词哈希，便于追溯
PROMPT_VERSION = 'v23'
SAMPLE_VER = 's2'
# v20 口径参数
TRIM_ACCEPT_HEAD = 2.0   # 开头这 N 秒内的露出算通过
TRIM_ACCEPT_TAIL = 2.0   # 结尾这 N 秒内的露出算通过
MIN_DUR = 5.0            # 短于此时长直接判删（用户要求，已知零代价）

# ---------------- 提示词 ----------------
SYSTEM = """你是第一人称（Ego）采集数据的质检审核员。你的判断决定一条数据被保留还是删除。

【铁律】
1. 只报告画面中直接可见的事实，不得从文件名、目录名、任务名推断结论。
2. 任何"存在问题"的判定必须给出可直接核对的帧 ID；帧号合法不代表帧内容支持你的主张，必须亲自核对画面语义。
3. "没有看到问题"不能作为"没问题"的证据。只有帧覆盖满足要求且画面确实清晰可判时才可判"未出现"，否则必须选"无法判断"。
4. 判"无法判断"不会被惩罚，编造证据会被追溯。证据不足时一律选"无法判断"。
5. 你只回答被问到的这几条问题，不做总体结论。总体结论由确定性代码生成。
6. 任务描述中明确要求操作的物品（平板、手机、积木、碗等）属于任务道具，不得判为问题。
7. 只输出一个合法 JSON 对象，不要输出 Markdown。"""

def user_prompt(task_text, nframes_desc):
    if task_text:
        q2_block = f"""【问题2｜任务是否按上面的任务描述完成】
**只能在"完成 / 部分完成 / 未完成"三者中选一个，没有其他选项。**
判定要点：
- 关键动作必须确实发生；准备动作（伸手、对准、靠近）不算完成。
- 终态必须符合描述。
- 含"放入/归位/放下"的任务，必须看到物体被放置并停稳；仅手持不算完成。
- **只看到任务的一部分**（例如只看到拿起、没看到放入；或只看到打开、没看到扣上）→ 判"部分完成"。
- 若描述本身写着"尝试但未成功"，则该任务就是尝试，不得据此判失败。
- 不要因为视频结尾突然停止就判未完成，除非终态明显与描述不符。
必须给出三选一；如选"部分完成"或"未完成"，给出至少 3 个帧编号说明差在哪一步，并给反证。"""
    else:
        q2_block = """【问题2】本题跳过：该数据集确实没有任务描述。请在 q2_task_completion.verdict 填 "跳过"。"""
    return f"""【判定倾向】本次以**召回优先**：宁可多判为问题，也不要把可能的问题放过去。当你在"没问题"和"有问题"之间犹豫时，选择"有问题"。
漏判（把有问题的数据放过去）的代价远高于误判（把好数据送去人工复核）。

【任务描述】（本数据集的真实任务，**已经提供**；不得声称"未提供任务描述"）
{task_text or '（该数据集确实没有任务描述，此时问题2 才可跳过）'}
【视频信息】{nframes_desc}

【图像说明】帧按组顺序给出：每组前有组标题，**每张图前都标了「帧标签 @ 该帧在视频中的秒数」**。
 H 组 = 视频**开头**的密集帧（覆盖架设/开机阶段）
 A 组 = 全片均匀采样帧
 T 组 = 视频**结尾**的密集帧（覆盖收拾/终态）
 E 组 = 视频的**最后一帧**（判断终态是否收拾干净、画面是否停在异常状态）
 B 组 = 画面下部 35% 区域裁切放大帧（看清桌面下缘细节）
 C 组 = 中段连续帧（间隔约 0.2 秒，判断抖动/静止）；长视频无此组
引用帧时必须使用标签（如 H_00 / A_03 / T_05 / B_02 / C_01）。
若某物体**只在开头或结尾短暂出现**，请务必在理由中写出它出现的秒数区间，例如 [1.2s-3.4s]。

【问题1｜画面中是否出现手机/平板/采集设备/无关屏幕（最重要）】
需要判定的对象（出现任一类即算）：
 (a) 手机、平板电脑 —— 无论屏幕是否点亮，无论被手持、放在桌上、放在凳子上
 (b) 发亮的屏幕 —— 显示器、采集设备的显示屏
 (c) 采集设备本体 —— 操作员身上佩戴的录制装置（深色方形或带屏幕的小型装置）
 (d) 腰包、挎包等采集员随身包具
判定要点：
- **不要求"佩戴"**：手机/平板只要在任务过程中出现在画面里就算，放在旁边桌上、凳子上也算。
- **明确排除：桌面上的黑色/深色圆形或椭圆形盖片、嵌入桌面且与桌面齐平的薄片、螺丝盖、桌面固定件** —— 这些是标准办公桌结构（穿线孔盖等），**不是采集设备**，不得判为问题，也不得列为"疑似"。
- **包具只在被操作员佩戴在身上时才算**（挂在腰间/胸前/肩上）；放在桌面、椅子、地上、推车里的包不算问题。
- 只露出一角、被手或物体遮挡一部分也算；任意一帧出现即算。
- 唯一豁免：任务描述中**明确要求操作的那一个具体物品**才是任务道具。
  例：任务是"更换手机壳" → **手机**是道具，属正常；但画面中若出现**平板**，仍算问题。
- 手表、手环、项链等个人穿戴物不算。
- **首尾各 2 秒内的露出算通过（重要）**：采集流程会把视频**最开始 2 秒**与**最末尾 2 秒**
  整段截断删除、只保留中间数据，因此只出现在这两段里的内容是可接受的。
  · ★ **本豁免对"屏幕/采集设备"和"腰包/随身包具"同样适用**：只要这两类**只**出现在
    开头 2 秒内、或**只**出现在结尾 2 秒内，就判"未出现"，**该条数据视为合格（保留）**。
  · 某物体**只出现在开头 2 秒内**、或**只出现在结尾 2 秒内** → 判"未出现"，**不计为问题**。
  · 该物体若在**中间区间**（约 2 秒 到 时长减 2 秒）**出现过** → 照常判"出现"，按问题处理。
  · 判据是帧标签里 @ 后面的秒数，本片的边界已在【视频信息】里给出：
    秒数 ≤ 开头窗口 属开头区间；秒数 ≥ (时长 − 结尾窗口) 属结尾区间；两者之间属中间区间。
  · 例：H_00(0.0s)、H_03(1.5s)、E_00(末帧) 属首尾区间；A_02(30.0s) 属中间区间。
  · **H 组与 T 组的帧本来就落在豁免区间内**（它们就是首尾那几秒），引用它们不构成"中间出现"。
  · 若你判断不了它的出现是否也落在中间区间，仍按"出现"处理（召回优先）。
- **屏幕没点亮、一直放在桌面上的手机或平板 → 算任务道具，不计为问题**：
  · 同时满足"屏幕全程未点亮（熄屏/黑屏）"且"全程只是平放在桌面或台面上、没被手持操作"
    → 视为任务道具，判"未出现"。
  · 反之只要满足任一条 → 仍算问题：屏幕点亮/发亮；被手持或触摸操作；
    出现在椅子、凳、推车、地面等非桌面位置。
- **桌面上的黑色/深色圆形或椭圆形盖片、嵌入桌面且与桌面齐平的薄片、螺丝盖**
  —— 这些是桌子的一部分（穿线孔盖等），**不是设备**，不得判为问题，也不得列为"疑似"。
- 不得判为问题的：积木、碗、收纳盒等任务道具；桌面、桌腿、隔断等固定结构；操作员的手和衣物。
- **不要因为"它可能是别的东西"就选无法判断**：如果你在清晰的帧上看到了一个符合上述特征的物体，请判为"明确出现"或"疑似出现但看不清"，并说明理由。倾向于判有问题，但**上面明确排除的桌面结构不在此列**。
回答：未出现 / 疑似出现但看不清 / 明确出现 / 无法判断
若选"疑似"或"明确出现"：给出至少 2 个帧编号，说明是什么（手机/平板/屏幕/设备本体/包具）、在画面哪个位置（下边缘/左下/右下/左侧/右侧/中央）、大约占画面多大比例。
再给反证：有没有帧显示画面中完全没有这些物品？给出帧编号。

{q2_block}

【问题3｜画面模糊】
回答：清晰 / 轻度模糊 / 严重模糊 / 中心区域模糊 / 画面发白过曝 / 光线过暗
若选后四项：给出至少 3 个帧编号，说明模糊区域在画面哪个部位、占多大面积，是全程还是局部时段。
再给反证：有没有帧显示画面清晰？

【问题4｜抖动 / 长时间静止 / 无意义动作】
对照第3组连续帧判断。回答（可多选）：都没有 / 有抖动 / 有长时间静止 / 有无意义动作 / 疑似测试片段 / 光照剧烈变化
"光照剧烈变化"指画面亮度在短时间内反复大幅跳变（如灯管频闪、强光扫过、忽明忽暗），导致画面时亮时暗、难以稳定判读；仅整体偏亮或偏暗不算，那属于问题3。
"长时间静止"指连续超过10秒画面几乎无变化；"无意义动作"指摸鼻子、抓痒、聊天、休息等与任务无关且超过10秒。
若选中任一项：给出帧编号并估计持续时长。

【问题5｜人脸 / 头部遮挡】
画面中是否出现人脸（含镜面反射、照片、屏幕中的人脸）？
注意：**手指、手臂、腿部、头发、后脑勺、模糊的肤色色块都不算人脸**；必须是能辨认出五官的人脸才判"出现"。
另需判断**头发是否大面积遮挡画面**：第一人称视角下，若操作员的头发长时间占据画面较大面积
（约 ≥20%）、遮挡住任务操作区域、导致看不清桌面或双手动作，判"头发大面积遮挡"。
只是画面边缘偶尔出现发丝、或占比很小，不算。
回答：出现 / 头发大面积遮挡 / 未出现 / 无法判断
若选"出现"或"头发大面积遮挡"：给出至少 3 个帧编号，并说明占画面比例、遮挡了哪个区域。

【问题6｜手部可见度】
任务动作主要靠双手完成，因此需要判断操作员的手在画面里是否看得清。
分三档：
- 手部全程可见：绝大多数帧里能看到手或手臂，且能看清在做什么
- 手部经常不可见：有较多帧看不到手，或手被身体/头发/物体遮住，只能靠画面变化推测动作
- 手部几乎不可见：绝大部分帧里看不到手，无法判断操作细节
注意：手只露出边缘、或只在画面角落一闪而过，都算"不可见"。
回答：手部全程可见 / 手部经常不可见 / 手部几乎不可见 / 无法判断
若选后两项：给出至少 3 个帧编号，并估计手部不可见的时间占比。

【输出】严格输出以下 JSON，不要任何额外文字：
{{"q1_device_exposed":{{"verdict":"","frame_ids":[],"what":"","position":"","area_pct":"","counter_evidence":[]}},
"q2_task_completion":{{"verdict":"","frame_ids":[],"missing_step":"","counter_evidence":[]}},
"q3_blur":{{"verdict":"","frame_ids":[],"region":"","span":"","counter_evidence":[]}},
"q4_temporal":{{"flags":[],"frame_ids":[],"duration_s":null}},
"q5_face":{{"verdict":"","frame_ids":[],"occlusion_ratio":""}},
"q6_hand_visibility":{{"verdict":"","frame_ids":[],"invisible_pct":""}},
"coverage":{{"covers_whole_video":true,"notes":""}},
"confidence":0.0}}"""

AUDIT_PROMPT = """上一位审核员对同一条视频给出了以下判定与证据：
{prev}

请你**主动寻找能推翻该判定的画面证据**。
- 若你能指出他引用的帧其实不支持其主张（例如那不是采集设备、或该物品属于任务道具、或模糊程度被夸大），
  请给出帧编号与理由，并给出你建议的判定。
- 若你核对后认为他的判定成立，**必须明确写"未找到反驳证据"，不得为了反驳而编造**。
- 不得凭空质疑；每条反驳必须绑定帧编号。
只输出 JSON：{{"overturn": false, "frame_ids": [], "reason": "", "suggested_verdict": ""}}"""

# ---------------- 抽帧 ----------------
SAMPLE_VER = 's2'      # s1=固定16帧均匀; s2=时长自适应 + 首尾密集帧 + 帧级时间戳

GROUP_DESC = {
    'H': '视频开头密集帧',
    'A': '全片均匀帧',
    'T': '视频结尾密集帧',
    'E': '视频最后一帧',
    'B': '画面下部35%裁切放大帧',
    'C': '中段连续帧(间隔0.2秒)',
}

def run(cmd, stdin_devnull=True):
    return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                          stdin=subprocess.DEVNULL if stdin_devnull else None)

def probe_duration(video):
    r = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                        '-of', 'default=nw=1:nk=1', video], capture_output=True, text=True)
    try: return float(r.stdout.strip())
    except Exception: return 0.0

def plan_sampling(dur):
    """按时长规划抽帧组，返回 [(组名, 帧数, 起始秒, 窗口秒, 是否底部裁切), ...]。

    s2 相对 s1 的三处改动：
      1) 均匀帧数随时长增长 —— s1 固定 16 帧，长视频盲区极大
         （736s 的片子相邻帧间隔 46 秒，最后 46 秒完全没有覆盖）；
      2) 新增开头/结尾两组密集帧 —— 采集设备/屏幕露出集中在架机阶段（开头），
         终态是否收拾干净集中在结尾，s1 在这两处只能靠均匀帧碰运气；
      3) 分辨率按组分配 —— 只有需要看细节的 H/T/B 组用 896px，
         负责全局覆盖的 A 组降到 704px，用省下的 token 把长视频的帧数提上去。

    各组的图像 token 开销差异很大（约与像素面积成正比），因此"帧数"不是
    真正的预算，"token"才是：896px≈576tok，704px≈355tok，640px≈299tok。
    """
    dur = max(float(dur or 0.0), 0.5)
    # ★ v21：帧组窗口与【口径豁免窗口】严格对齐 —— 消除 v20 里 dur-2.5~dur-2.0 的 0.5 秒夹缝
    head, tail = accept_windows(dur)
    nA = min(max(int(round(dur / 40.0)) + 14, 16), 24)
    if dur <= 180:
        nH = nT = 6; nB, nC = 4, 4
    else:
        nH = nT = 5; nB, nC = 5, 0
    spec = [('H', nH, 0.0, head, False),                           # 开头密集（豁免区）
            ('A', nA, head, max(dur - head - tail, 0.5), False),   # 严格中间
            ('T', nT, max(dur - tail, 0.0), tail, False),          # 结尾密集（豁免区）
            ('E', 1, max(dur - 0.1, 0.0), 0.0, False),             # 真正的末帧（豁免区）
            ('B', nB, 0.0, dur, True)]                             # 底部裁切，覆盖全片
    if nC:
        spec.append(('C', nC, max(dur * 0.5, 0.0), 0.0, False))
    return spec

def accept_windows(dur):
    """豁免窗口。口径是首尾各 2 秒；对极短片做保护性收窄，避免中间区间被吃光。"""
    dur = max(float(dur or 0.0), 0.5)
    head = min(TRIM_ACCEPT_HEAD, max(dur * 0.20, 0.5))
    tail = min(TRIM_ACCEPT_TAIL, max(dur * 0.20, 0.5))
    return head, tail

# 各组出图宽度：只有要看细节的组用高分辨率
GROUP_PX = {'H': 896, 'T': 896, 'E': 896, 'B': 896, 'A': 704, 'C': 640}

def frame_times(spec, dur):
    """展开出每一帧的时间戳，用于算覆盖盲区与给模型标时间。"""
    ts = []
    for grp, n, start, span, _crop in spec:
        if n <= 0 or grp == 'C': continue
        start = max(0.0, min(start, max(dur - 0.05, 0.0)))
        span = max(span, 0.1)
        ts += [start + (span * k / (n - 1) if n > 1 else 0.0) for k in range(n)]
    return sorted(ts)

def coverage_gap(spec, dur):
    """相邻采样帧的最大时间间隔（= 最坏盲区，秒）。用于区分"漏检"是采样问题还是判断问题。"""
    ts = [0.0] + frame_times(spec, dur) + [max(dur, 0.1)]
    return max(b - a for a, b in zip(ts, ts[1:])) if len(ts) > 1 else float(dur)


def extract(video, outdir, dur):
    """抽帧，返回 (帧标签列表, 各帧在视频中的秒数 {组名: [t,...]})。

    时间戳不按"计划帧数"算，而是**按实际出图数反推**：fps 滤镜第 k 帧落在
    start + k/fr，ffmpeg 实际出几张就以几张为准（不同版本会差 1 帧）。
    """
    os.makedirs(outdir, exist_ok=True)
    for f in os.listdir(outdir):
        os.remove(os.path.join(outdir, f))
    stamps = {}
    for grp, n, start, span, crop in plan_sampling(dur):
        if n <= 0: continue
        start = max(0.0, min(start, max(dur - 0.05, 0.0)))
        cropv = 'crop=iw:ih*0.35:0:ih*0.65,' if crop else ''
        px = GROUP_PX[grp]
        if grp == 'E':
            # 真正的最后一帧。不能用 fps 去"请求" t=dur —— 那一帧不存在，
            # 用 -sseof 从末尾回退才能稳定拿到，而末帧正是"有没有收拾干净"的关键证据。
            run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-sseof', '-0.1', '-i', video,
                 '-vf', f'scale={px}:-2', '-q:v', '5', '-frames:v', '1',
                 '-start_number', '0', os.path.join(outdir, 'E_%02d.jpg')])
            fr = None
        elif grp == 'C':                                 # 中段连续帧：固定 5fps
            run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-ss', f'{start:.2f}', '-i', video,
                 '-vf', f'{cropv}fps=5,scale={px}:-2', '-q:v', '5', '-frames:v', str(n),
                 '-start_number', '0', os.path.join(outdir, 'C_%02d.jpg')])
            fr = 5.0
        else:
            span = max(span, 0.1)
            # fps=(n-1)/span 时理想落点是 0, span/(n-1), ..., span。
            # -t 多给 2 个帧间隔：ffmpeg 实际会比请求少出 1 帧，给足余量才能铺到窗口末端。
            fr = (n - 1) / span if n > 1 else 1.0 / span
            run(['ffmpeg', '-nostdin', '-v', 'error', '-y',
                 '-ss', f'{start:.2f}', '-t', f'{span + 2.0 / fr:.3f}', '-i', video,
                 '-vf', f'{cropv}fps={fr:.6f},scale={px}:-2', '-q:v', '5', '-frames:v', str(n),
                 '-start_number', '0', os.path.join(outdir, f'{grp}_%02d.jpg')])
        m = len([x for x in os.listdir(outdir) if x.startswith(grp + '_')])
        if grp == 'E':
            stamps[grp] = [max(dur - 0.1, 0.0)]
        else:
            stamps[grp] = [start + k / fr for k in range(m)]
    names = sorted(os.listdir(outdir))
    labels = []
    for grp in GROUP_DESC:                               # 按 H,A,T,E,B,C 固定顺序
        labels += [x[:-4] for x in names if x.startswith(grp + '_')]
    return labels, stamps

def build_blocks(outdir, labels, stamps):
    """把帧组织成交错的内容块：每组一个小标题 + 每帧前标注「标签 @ 秒数」再放图。

    模型此前只能靠数位置去猜 A1/B2/C3，5 组之后必然错位；改为帧前显式标注，
    同时给出真实时间戳，模型才能说出「设备只在 0.0-2.0s 出现」这类定位。
    """
    blocks = []
    for grp in GROUP_DESC:
        g = [x for x in labels if x.startswith(grp + '_')]
        if not g: continue
        ts = stamps.get(grp, [])
        blocks.append(f'—— {grp} 组：{GROUP_DESC[grp]}，共 {len(g)} 帧 ——')
        for i, lab in enumerate(g):
            t = ts[i] if i < len(ts) else None
            blocks.append(f'[{lab} @ {t:.1f}s]' if t is not None else f'[{lab}]')
            blocks.append((lab, os.path.join(outdir, lab + '.jpg')))
    return blocks

def data_url(p):
    return 'data:image/jpeg;base64,' + base64.b64encode(open(p, 'rb').read()).decode()

# ---------------- 调用 ----------------
def call_vlm(blocks, prompt, max_tokens=16000):
    """blocks: 字符串=文字块（组标题/帧标签），(label, path)=图片块。"""
    content = [{'type': 'text', 'text': prompt}]
    for item in blocks:
        if isinstance(item, str):
            content.append({'type': 'text', 'text': item})
        else:
            _name, path = item
            _iu = {'url': data_url(path)}
            if SEND_DETAIL:
                _iu['detail'] = 'high'
            content.append({'type': 'image_url', 'image_url': _iu})
    body = {'model': MODEL, 'temperature': 0, 'max_tokens': max_tokens,
            'stream': True, 'stream_options': {'include_usage': True},
            'messages': [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': content}]}
    if THINK_MODE == '1':
        body['thinking'] = {'type': 'enabled'}
    elif THINK_MODE == '0':
        body['thinking'] = {'type': 'disabled'}
    payload = json.dumps(body).encode()
    hdr = {'Content-Type': 'application/json'}
    if API_KEY:
        hdr['Authorization'] = 'Bearer ' + API_KEY
    t0 = time.time()
    last = None
    for attempt in range(6):        # 公网 API：网络抖动/限流都要退避重试
        try:
            req = urllib.request.Request(next_endpoint(), data=payload, headers=hdr)
            cparts, rparts, usage, finish = [], [], {}, None
            with urllib.request.urlopen(req, timeout=900) as r:
                for raw in r:
                    line = raw.decode('utf-8', 'ignore').strip()
                    if not line.startswith('data:'):
                        continue
                    data = line[5:].strip()
                    if data == '[DONE]':
                        break
                    try:
                        ev = json.loads(data)
                    except Exception:
                        continue
                    if ev.get('usage'):
                        usage = ev['usage']
                    for ch in ev.get('choices') or []:
                        if ch.get('finish_reason'):
                            finish = ch['finish_reason']
                        d = ch.get('delta') or {}
                        if d.get('content'):
                            cparts.append(d['content'])
                        if d.get('reasoning_content'):
                            rparts.append(d['reasoning_content'])
            # 拼回成与 vLLM 一致的结构，下游代码无需改动
            resp = {'choices': [{'message': {'content': ''.join(cparts),
                                             'reasoning': ''.join(rparts)},
                                 'finish_reason': finish}],
                    'usage': usage}
            return resp, time.time() - t0
        except urllib.error.HTTPError as e:
            txt = ''
            try:
                txt = e.read().decode('utf-8', 'ignore')[:300]
            except Exception:
                pass
            last = f'HTTP {e.code}: {txt}'
            if e.code in (400, 401, 403, 404):   # 参数/鉴权类错误重试无意义
                raise RuntimeError(last)
            # 429 = 账号级限流（实测 16 并发跑全量会触发 AccountRateLimitExceeded）。
            # 退避必须够长，并尊重服务端 Retry-After；否则会在同一波里反复撞限流。
            wait = 45 if e.code == 429 else 2 ** attempt * 2
            try:
                ra = e.headers.get('Retry-After')
                if ra:
                    wait = max(wait, float(ra))
            except Exception:
                pass
            time.sleep(min(wait, 90))
        except Exception as e:
            last = f'{type(e).__name__}: {e}'
            time.sleep(2 ** attempt * 2)
    raise RuntimeError(f'重试 6 次仍失败: {last}')

def parse_obj(text):
    if not text: return None
    m = re.search(r'\{.*\}', text, re.S)
    if m:
        raw = m.group(0)
        for cand in (raw, raw.rstrip().rstrip(',') + '}', raw + '}' * 3):
            try: return json.loads(cand)
            except Exception:
                try: return json.loads(cand.replace("'", '"'))
                except Exception: pass
    return salvage(text)

def salvage(text):
    """JSON 被截断时，用正则把关键 verdict 抠出来"""
    if not text: return None
    out = {}
    for key in ('q1_device_exposed', 'q2_task_completion', 'q3_blur', 'q5_face'):
        mm = re.search(key + r'"?\s*:\s*\{[^{}]*?"?verdict"?\s*:\s*"([^"]{1,20})"', text, re.S)
        if mm: out[key] = {'verdict': mm.group(1)}
    mm = re.search(r'"flags"\s*:\s*\[([^\]]{0,200})\]', text, re.S)
    if mm:
        out['q4_temporal'] = {'flags': [s.strip().strip('"') for s in mm.group(1).split(',') if s.strip()],
                              'duration_s': None}
    mm = re.search(r'"covers_whole_video"\s*:\s*(true|false)', text)
    if mm: out['coverage'] = {'covers_whole_video': mm.group(1) == 'true', 'notes': 'salvaged'}
    return out if out else None

# ---------------- 聚合（代码定结论） ----------------
def decide(r):
    """召回优先口径：犹豫即判删除；只有设备完全判不了才转人工"""
    if not isinstance(r, dict): return 'needs_human', 'parse_fail'
    g = lambda k, d='': (r.get(k) or {}).get('verdict', d) if isinstance(r.get(k), dict) else d
    q1, q2, q3 = g('q1_device_exposed'), g('q2_task_completion'), g('q3_blur')
    q5 = g('q5_face')
    q6 = g('q6_hand_visibility')
    q4 = r.get('q4_temporal') or {}
    flags = q4.get('flags') or []
    if isinstance(flags, str): flags = [flags]
    if q1 in ('明确出现', '疑似出现但看不清'): return 'delete', 'device_or_suspect'
    if q5 == '出现': return 'delete', 'face'
    # ★ v23 新增：头发大面积遮挡
    if q5 == '头发大面积遮挡': return 'delete', 'hair_occlusion'
    if q2 in ('未完成', '部分完成'): return 'delete', 'task_incomplete'
    # ★ v23 新增：光线过暗
    if q3 in ('严重模糊', '画面发白过曝', '中心区域模糊', '轻度模糊'): return 'delete', 'blur'
    if q3 == '光线过暗': return 'delete', 'too_dark'
    # ★ v23 新增：光照剧烈变化
    if '光照剧烈变化' in flags: return 'delete', 'lighting_change'
    if any(x in ('有抖动', '有长时间静止', '有无意义动作', '疑似测试片段') for x in flags):
        return 'delete', 'temporal'
    # ★ v23 新增：手部可见度
    if q6 in ('手部经常不可见', '手部几乎不可见'): return 'delete', 'hand_visibility'
    if q1 == '无法判断': return 'needs_human', 'device_unknown'
    return 'keep', ''

# ---------------- 选择 100 条 ----------------
V3_INDEX = 'EVAL_INDEX.csv'


def build_selection():
    """v23：读 balanced_5x100_v2 的一个 set。

    输入（模型允许看到的）：inference_inputs.pending.jsonl
        dataset_id / episode_index / collection_task_id / collection_task_name /
        task_description(=平台 SOP) / local_dir / files
    标签与原因（仅用于评测，不进模型）：dataset_index.csv 的「人工审核」「无效原因」

    ★ 明确不读 segment_index.csv：实测「有无保留片段」与标签共线 97.8%，
      属标签泄漏，且该字段只在 manifest 里，不在 inference_inputs 里。
    """
    import csv as _csv
    lab = {}
    idx = os.path.join(ROOT, 'dataset_index.csv')
    if os.path.isfile(idx):
        for r in _csv.DictReader(open(idx, encoding='utf-8-sig')):
            lab[(str(r['数据集ID']).strip(), str(r['Episode']).strip())] = r
    out = []
    with open(os.path.join(ROOT, 'inference_inputs.pending.jsonl'), encoding='utf-8') as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            k = (str(d['dataset_id']).strip(), str(d['episode_index']).strip())
            r = lab.get(k, {})
            review = (r.get('人工审核') or '').strip()
            reason = (r.get('无效原因') or '').strip()
            if not review:                     # 没读到标签就跳过，避免误算
                review, reason = 'unknown', ''
            files = d.get('files') or []
            left = next((f for f in files if 'left_rgb' in f), files[0] if files else '')
            out.append({
                'dataset_id': int(d['dataset_id']),
                'episode_index': int(d['episode_index']),
                'name': d.get('collection_task_name') or '',
                'local_dir': d.get('local_dir') or '',
                'left_video': left,
                'task_text_v3': (d.get('task_description') or '').strip(),   # 平台 SOP
                'task_desc_source': 'platform_sop',
                'eval_usable': 'yes' if (d.get('task_description') or '').strip() else 'no',
                'audit_status': 'valid' if review == '有效' else ('invalid' if review == '无效' else 'unknown'),
                'invalid_reason': reason,
                '_dev': reason == '采集设备屏幕露出',
            })
    return out


# ---------------- 单条处理 ----------------
def process(m):
    t_start = time.time()
    ep_dir = os.path.join(ROOT, m['local_dir'])
    # ★ v3：任务文本直接用补全好的 task_description，不再去 ../meta/tasks.jsonl 抓
    task_text = m.get('task_text_v3') or ''
    # ★ v23：files 里的路径是相对 local_dir 的（balanced 5x100 的结构）
    video = os.path.join(ROOT, m['local_dir'], m['left_video'])
    if not video or not os.path.isfile(video):
        return {'dataset_id': m['dataset_id'], 'episode_index': m['episode_index'], 'error': 'no_video',
                'gt': m['audit_status'], 'reason': m.get('invalid_reason')}
    fd = os.path.join(FRAMES, f"{m['dataset_id']}_{m['episode_index']}")
    dur = probe_duration(video)
    if dur < MIN_DUR:                       # 太短的视频直接判删，不抽帧不推理
        return {'dataset_id': m['dataset_id'], 'episode_index': m['episode_index'],
                'name': m.get('collection_task_name'), 'gt': m['audit_status'],
                'gt_reason': m.get('invalid_reason'), 'gt_is_device': m['_dev'],
                'verdict': 'delete', 'rule': 'too_short', 'audit': None, 'raw': None,
                'prompt_version': PROMPT_VERSION, 'sample_ver': SAMPLE_VER,
                'dur_s': round(dur, 1), 'n_frames': 0,
                'groups': f'时长 {dur:.1f}s < {MIN_DUR}s，未抽帧',
                'coverage_gap_s': round(dur, 1), 'prompt_sha': '', 'task_text': task_text,
                'finish': 'skipped_too_short', 'reasoning_len': 0, 'raw_text': '',
                't_vlm_s': 0.0, 't_audit_s': 0.0,
                't_total_s': round(time.time() - t_start, 1), 'usage': {},
                'task_desc_source': m.get('task_desc_source'), 'eval_usable': m.get('eval_usable'),
                'accept_head_s': TRIM_ACCEPT_HEAD, 'accept_tail_s': TRIM_ACCEPT_TAIL}
    labels, stamps = extract(video, fd, dur)
    blocks = build_blocks(fd, labels, stamps)
    n_img = sum(1 for b in blocks if not isinstance(b, str))
    cnt = collections.Counter(x.split('_')[0] for x in labels)
    grp_desc = '，'.join(f'{g}{cnt[g]}帧' for g in GROUP_DESC if cnt.get(g))
    gap_s = coverage_gap(plan_sampling(dur), dur)
    _head, _tail = accept_windows(dur)
    prompt = user_prompt(task_text,
        f"时长约 {dur:.1f} 秒，30fps；共提供 {n_img} 帧（{grp_desc}）；"
        f"首尾豁免窗口 = 开头 {_head:.1f}s 与 结尾 {_tail:.1f}s，"
        f"即 [0, {_head:.1f}s] 与 [{max(dur-_tail,0):.1f}s, {dur:.1f}s] 属豁免区间，"
        f"({_head:.1f}s, {max(dur-_tail,0):.1f}s) 属中间区间")
    t_vlm = 0; t_audit = 0; usage = {}
    try:
        resp, dt = call_vlm(blocks, prompt); t_vlm = dt
        usage = resp.get('usage', {}) or {}
        ch = resp['choices'][0]
        msg = ch['message']
        content = msg.get('content') or ''
        reasoning = msg.get('reasoning') or msg.get('reasoning_content') or ''
        finish = ch.get('finish_reason')
        parsed = parse_obj(content)
        if parsed is None:            # 解析失败 → 重试一次，明确要求纯 JSON
            resp2, dt2 = call_vlm(blocks, prompt + '\n\n【重要】上一次回复无法解析为 JSON。请**只输出一个 JSON 对象**，不要任何解释文字，不要 Markdown 代码块。', max_tokens=16000)
            t_vlm += dt2
            c2 = (resp2['choices'][0]['message'].get('content') or '')
            p2 = parse_obj(c2)
            if p2:
                parsed = p2
                content = c2
                usage = resp2.get('usage', {}) or usage
    except Exception as e:
        return {'dataset_id': m['dataset_id'], 'episode_index': m['episode_index'], 'error': f'vlm:{e}',
                'gt': m['audit_status'], 'reason': m.get('invalid_reason')}
    verdict, rule = decide(parsed)
    audit = None
    if verdict == 'delete':
        try:
            aresp, adt = call_vlm(blocks, AUDIT_PROMPT.format(prev=json.dumps(parsed, ensure_ascii=False)), max_tokens=500)
            t_audit = adt
            audit = parse_obj(aresp['choices'][0]['message']['content'])
        except Exception as e:
            audit = {'error': str(e)}
    return {'dataset_id': m['dataset_id'], 'episode_index': m['episode_index'],
            'name': m.get('collection_task_name'), 'gt': m['audit_status'],
            'gt_reason': m.get('invalid_reason'), 'gt_is_device': m['_dev'],
            'verdict': verdict, 'rule': rule, 'audit': audit, 'raw': parsed,
            'prompt_version': PROMPT_VERSION,
            'backend': BACKEND,
            'think': THINK_MODE,
            'task_desc_source': m.get('task_desc_source'), 'eval_usable': m.get('eval_usable'),
            'sample_ver': SAMPLE_VER,
            'accept_head_s': TRIM_ACCEPT_HEAD, 'accept_tail_s': TRIM_ACCEPT_TAIL,
            'root': ROOT,
            'dur_s': round(dur, 1), 'n_frames': n_img, 'groups': grp_desc,
            'coverage_gap_s': round(gap_s, 1),
            'prompt_sha': hashlib.sha256((SYSTEM + prompt).encode('utf-8')).hexdigest()[:16],
            'task_text': task_text,
            'finish': finish, 'reasoning_len': len(reasoning), 'raw_text': content[:1500],
            't_vlm_s': round(t_vlm, 1), 't_audit_s': round(t_audit, 1),
            't_total_s': round(time.time() - t_start, 1), 'usage': usage}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=100)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--tag', default='run1')
    ap.add_argument('--only-usable', type=int, default=0,
                    help='1=剔除没有任务描述的样本（eval_usable=no）')
    ap.add_argument('--half', type=int, default=0,
                    help='1=在选样列表上隔一条取一条（保持 valid/invalid 比例，用于半量试跑）')
    ap.add_argument('--think', default='auto', choices=['auto', '1', '0'],
                    help='思考模式: auto=不传该字段(对齐 v21 默认); 1=enabled; 0=disabled(快数倍, 可能掉准确率)')
    a = ap.parse_args()
    global THINK_MODE
    THINK_MODE = a.think
    os.makedirs(OUT, exist_ok=True)
    # 落盘本次使用的提示词，保证结果可追溯（结果文件里也记录 prompt_version / prompt_sha）
    pdir = os.path.join(OUT, 'prompts'); os.makedirs(pdir, exist_ok=True)
    with open(os.path.join(pdir, f'{PROMPT_VERSION}.txt'), 'w', encoding='utf-8') as fh:
        fh.write('===== SYSTEM =====\n' + SYSTEM + '\n\n===== USER TEMPLATE =====\n' + user_prompt('{task_text}', '{frames_desc}'))
    print(f'提示词已存档: prompts/{PROMPT_VERSION}.txt  (提示词哈希示例见结果文件 prompt_sha)', flush=True)
    print(f'后端 backend={BACKEND} base={ARK_BASE} detail={SEND_DETAIL} '
          f'key={"已设置" if API_KEY else "缺失!"} workers={a.workers} half={a.half} think={THINK_MODE}', flush=True)
    sel = build_selection()
    if a.only_usable:
        sel = [m for m in sel if m.get('eval_usable') != 'no']
    if a.half:
        sel = sel[::2]
    sel = sel[:a.limit]
    print(f'选中 {len(sel)} 条：valid={sum(1 for m in sel if m["audit_status"]=="valid")} '
          f'invalid={sum(1 for m in sel if m["audit_status"]=="invalid")} '
          f'device相关={sum(1 for m in sel if m["_dev"])}', flush=True)
    res = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(process, m): m for m in sel}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result(); res.append(r)
            print(f'  [{i}/{len(sel)}] ds{r.get("dataset_id")} ep{r.get("episode_index")} '
                  f'gt={r.get("gt")} -> {r.get("verdict","ERR")} ({r.get("rule","")}) {r.get("t_total_s","")}s', flush=True)
    wall = time.time() - t0
    out = os.path.join(OUT, f'preds_{a.tag}.jsonl')
    with open(out, 'w', encoding='utf-8') as fh:
        for r in res: fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    # 指标
    ok = [r for r in res if 'verdict' in r]
    tp = sum(1 for r in ok if r['gt'] == 'invalid' and r['verdict'] == 'delete')
    fn = sum(1 for r in ok if r['gt'] == 'invalid' and r['verdict'] != 'delete')
    fp = sum(1 for r in ok if r['gt'] == 'valid' and r['verdict'] == 'delete')
    tn = sum(1 for r in ok if r['gt'] == 'valid' and r['verdict'] != 'delete')
    per = collections.defaultdict(lambda: [0, 0])
    for r in ok:
        if r['gt'] == 'invalid':
            k = 'device相关' if r['gt_is_device'] else (r['gt_reason'] or 'none')
            per[k][1] += 1
            if r['verdict'] == 'delete': per[k][0] += 1
    metrics = {'n': len(ok), 'errors': len(res) - len(ok), 'wall_s': round(wall, 1),
               'per_item_s': round(wall / max(len(ok), 1), 1),
               'delete_TP': tp, 'delete_FN': fn, 'delete_FP': fp, 'keep_TN': tn,
               'recall': round(tp / max(tp + fn, 1), 4), 'precision': round(tp / max(tp + fp, 1), 4),
               'fp_rate_on_valid': round(fp / max(fp + tn, 1), 4),
               'needs_human': sum(1 for r in ok if r['verdict'] == 'needs_human'),
               'per_reason_recall': {k: f'{v[0]}/{v[1]}' for k, v in sorted(per.items(), key=lambda x: -x[1][1])},
               'avg_vlm_s': round(sum(r.get('t_vlm_s', 0) for r in ok) / max(len(ok), 1), 1),
               'avg_total_s': round(sum(r.get('t_total_s', 0) for r in ok) / max(len(ok), 1), 1),
               'total_prompt_tokens': sum((r.get('usage') or {}).get('prompt_tokens', 0) for r in ok),
               'total_completion_tokens': sum((r.get('usage') or {}).get('completion_tokens', 0) for r in ok)}
    with open(os.path.join(OUT, f'metrics_{a.tag}.json'), 'w', encoding='utf-8') as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)
    print('\n=== 指标 ===')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
