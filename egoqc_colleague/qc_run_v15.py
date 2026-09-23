#!/usr/bin/env python3
"""Ego 数据二元质检流程：抽帧 -> 提示词 -> 本地千问推理 -> 反证审计 -> 聚合 -> 评测"""
import argparse, base64, collections, hashlib, itertools, json, os, re, subprocess, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = '/data1/cus000001_mixed150'
OUT = '/data1/review/egoqc'
FRAMES = os.path.join(OUT, 'frames')
ENDPOINTS = [f'http://127.0.0.1:{18090 + i}/v1/chat/completions' for i in range(8)]
_ep_cycle = itertools.cycle(ENDPOINTS)
_ep_lock = threading.Lock()

def next_endpoint():
    with _ep_lock:
        return next(_ep_cycle)

MODEL = 'qwen3.8-27b-fp8'
DETAILS = '/tmp/p7/invalid_details_dedup.tsv'
# 每次改提示词/聚合规则都必须递增此版本号；结果文件会记录版本号与提示词哈希，便于追溯
PROMPT_VERSION = 'v15'
# v15 新增抽帧：首尾密集帧（与主轴 A/B/C 分开落盘）
SAMPLE_VER = 'v15_ht'
# 首尾检测到「亮屏手机」时"无法判断"要不要也算问题（默认不算，只认明确的"有"）
HT_FIRE_ON_UNKNOWN = False

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

【图像说明】按顺序给出三组：
 第1组 A1..A16：全片均匀采样帧
 第2组 B1..B6：画面下部 35% 区域裁切放大帧（用于看清下缘细节）
 第3组 C1..C6：中段连续帧（间隔约 0.2 秒，用于判断抖动/静止）
引用帧时请用 A1/B2/C3 这样的编号。

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
- 不得判为问题的：积木、碗、收纳盒等任务道具；桌面、桌腿、隔断等固定结构；操作员的手和衣物。
- **不要因为"它可能是别的东西"就选无法判断**：如果你在清晰的帧上看到了一个符合上述特征的物体，请判为"明确出现"或"疑似出现但看不清"，并说明理由。倾向于判有问题，但**上面明确排除的桌面结构不在此列**。
回答：未出现 / 疑似出现但看不清 / 明确出现 / 无法判断
若选"疑似"或"明确出现"：给出至少 2 个帧编号，说明是什么（手机/平板/屏幕/设备本体/包具）、在画面哪个位置（下边缘/左下/右下/左侧/右侧/中央）、大约占画面多大比例。
再给反证：有没有帧显示画面中完全没有这些物品？给出帧编号。

{q2_block}

【问题3｜画面模糊】
回答：清晰 / 轻度模糊 / 严重模糊 / 中心区域模糊 / 画面发白过曝
若选后四项：给出至少 3 个帧编号，说明模糊区域在画面哪个部位、占多大面积，是全程还是局部时段。
再给反证：有没有帧显示画面清晰？

【问题4｜抖动 / 长时间静止 / 无意义动作】
对照第3组连续帧判断。回答（可多选）：都没有 / 有抖动 / 有长时间静止 / 有无意义动作 / 疑似测试片段
"长时间静止"指连续超过10秒画面几乎无变化；"无意义动作"指摸鼻子、抓痒、聊天、休息等与任务无关且超过10秒。
若选中任一项：给出帧编号并估计持续时长。

【问题5｜人脸】
画面中是否出现人脸（含镜面反射、照片、屏幕中的人脸）？
注意：**手指、手臂、腿部、头发、后脑勺、模糊的肤色色块都不算人脸**；必须是能辨认出五官的人脸才判"出现"。
回答：出现 / 未出现 / 无法判断

【输出】严格输出以下 JSON，不要任何额外文字：
{{"q1_device_exposed":{{"verdict":"","frame_ids":[],"what":"","position":"","area_pct":"","counter_evidence":[]}},
"q2_task_completion":{{"verdict":"","frame_ids":[],"missing_step":"","counter_evidence":[]}},
"q3_blur":{{"verdict":"","frame_ids":[],"region":"","span":"","counter_evidence":[]}},
"q4_temporal":{{"flags":[],"frame_ids":[],"duration_s":null}},
"q5_face":{{"verdict":"","frame_ids":[]}},
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
def run(cmd, stdin_devnull=True):
    return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                          stdin=subprocess.DEVNULL if stdin_devnull else None)

def probe_duration(video):
    r = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                        '-of', 'default=nw=1:nk=1', video], capture_output=True, text=True)
    try: return float(r.stdout.strip())
    except Exception: return 0.0

def extract(video, outdir, dur):
    os.makedirs(outdir, exist_ok=True)
    for f in os.listdir(outdir):
        os.remove(os.path.join(outdir, f))
    A = 16; B = 6; C = 6
    # 组A：全片均匀
    run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', video,
         '-vf', f'fps={A/max(dur,1):.6f},scale=896:-2', '-q:v', '5',
         os.path.join(outdir, 'A_%02d.jpg')])
    # 组B：底部35%裁切放大
    run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', video,
         '-vf', f'fps={B/max(dur,1):.6f},crop=iw:ih*0.35:0:ih*0.65,scale=896:-2', '-q:v', '5',
         os.path.join(outdir, 'B_%02d.jpg')])
    # 组C：中段连续帧（0.2s 间隔）
    mid = max(dur * 0.5, 1.0)
    run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-ss', f'{mid:.2f}', '-i', video,
         '-vf', 'fps=5,scale=640:-2', '-frames:v', str(C), '-q:v', '5',
         os.path.join(outdir, 'C_%02d.jpg')])
    return sorted(os.listdir(outdir))

# ---------------- 首尾密集帧 + 亮屏手机独立判定（v15 新增） ----------------
HT_PX = 896
HT_GROUP_DESC = {'H': '视频开头密集帧', 'T': '视频结尾密集帧', 'E': '视频最后一帧'}

def plan_headtail(dur):
    """首尾窗口规划：开头 win 秒 + 结尾 win 秒 + 真正的末帧。"""
    dur = max(float(dur or 0.0), 0.5)
    win = min(2.5, max(0.4, dur * 0.10))       # 绝对 2.5s 与 10% 时长取小
    if dur < 2 * win + 0.5:                    # 极短片，避免首尾窗口重叠
        win = max(dur * 0.25, 0.2)
    return [('H', 6, 0.0, win),
            ('T', 6, max(dur - win, 0.0), win),
            ('E', 1, max(dur - 0.1, 0.0), 0.0)]

def headtail_win(dur):
    dur = max(float(dur or 0.0), 0.5)
    win = min(2.5, max(0.4, dur * 0.10))
    if dur < 2 * win + 0.5:
        win = max(dur * 0.25, 0.2)
    return win

def extract_headtail(video, outdir, dur):
    """只抽开头/结尾的密集帧。时间戳按实际出图数反推（fps 滤镜实测会少出一帧）。"""
    os.makedirs(outdir, exist_ok=True)
    for f in os.listdir(outdir):
        os.remove(os.path.join(outdir, f))
    stamps = {}
    for grp, n, start, span in plan_headtail(dur):
        start = max(0.0, min(start, max(dur - 0.05, 0.0)))
        if grp == 'E':
            # 用 fps 请求 t=时长 的那一帧永远拿不到，必须用 -sseof 从末尾回退
            run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-sseof', '-0.1', '-i', video,
                 '-vf', f'scale={HT_PX}:-2', '-q:v', '5', '-frames:v', '1',
                 '-start_number', '0', os.path.join(outdir, 'E_%02d.jpg')])
            fr = None
        else:
            span = max(span, 0.1)
            fr = (n - 1) / span if n > 1 else 1.0 / span
            run(['ffmpeg', '-nostdin', '-v', 'error', '-y',
                 '-ss', f'{start:.2f}', '-t', f'{span + 2.0 / fr:.3f}', '-i', video,
                 '-vf', f'fps={fr:.6f},scale={HT_PX}:-2', '-q:v', '5', '-frames:v', str(n),
                 '-start_number', '0', os.path.join(outdir, f'{grp}_%02d.jpg')])
        m = len([x for x in os.listdir(outdir) if x.startswith(grp + '_')])
        stamps[grp] = [max(dur - 0.1, 0.0)] if grp == 'E' else [start + k / fr for k in range(m)]
    names = sorted(os.listdir(outdir))
    labels = []
    for grp in ('H', 'T', 'E'):
        labels += [x[:-4] for x in names if x.startswith(grp + '_')]
    return labels, stamps

def build_ht_blocks(outdir, labels, stamps):
    """与 s2 一样交错排布：组标题 + 「[标签 @ 秒数]」+ 图，让模型能定位到具体秒数。"""
    blocks = []
    for grp in ('H', 'T', 'E'):
        g = [x for x in labels if x.startswith(grp + '_')]
        if not g: continue
        ts = stamps.get(grp, [])
        blocks.append(f'—— {grp} 组：{HT_GROUP_DESC[grp]}，共 {len(g)} 帧 ——')
        for i, lab in enumerate(g):
            t = ts[i] if i < len(ts) else None
            blocks.append(f'[{lab} @ {t:.1f}s]' if t is not None else f'[{lab}]')
            blocks.append((lab, os.path.join(outdir, lab + '.jpg')))
    return blocks

HT_PROMPT = """你只看同一条采集视频的【开头】和【结尾】两小段画面。
每张图前标了「[帧标签 @ 秒数]」，@ 后面是该帧在视频中的时间。

【只回答一个问题】这些画面里，是否出现**亮着屏幕的手机**？

定义（必须同时满足）：
- 必须是**手机**：能看出是手持电话的形态（长方形、手掌大小、有屏幕区域）。
- 必须**屏幕亮着**：屏幕点亮并显示内容（白色/蓝色/彩色界面、明显亮光都算）。
  熄屏的黑屏手机、屏幕朝下扣着的手机 → **不算**。

以下都**不算**（不要误判）：
- 平板电脑（明显比手机大、或双手捧着看）
- 显示器、笔记本电脑、电视
- 采集设备本体（佩戴在身上或连接线缆的小型装置）
- 充电宝、遥控器、耳机盒、纸张、书本、桌面固定件（穿线孔盖等）

判定规则：
- 只看下面给出的这些帧。**任意一帧出现即算「有」**。
- 只露出一部分、被手遮挡一部分也算。
- 若所有相关帧都太暗或太糊、无法辨认，选「无法判断」。

【输出】严格输出以下 JSON，不要任何额外文字：
{"lit_phone_in_headtail":"有/没有/无法判断","frame_ids":[],"what":"","position":"","confidence":0.0}"""

def data_url(p):
    return 'data:image/jpeg;base64,' + base64.b64encode(open(p, 'rb').read()).decode()

# ---------------- 调用 ----------------
def merge_usage(dst, src):
    """把一次调用的 usage 累加进 dst。vLLM 的 usage 里可能有 None 值（如 details 字段），要过滤。"""
    for k, v in (src or {}).items():
        if isinstance(v, (int, float)):
            dst[k] = (dst.get(k) or 0) + v
    return dst

def call_vlm(blocks, prompt, max_tokens=8000):
    """blocks: 字符串=文字块（组标题/帧标签），(label, path)=图片块。"""
    content = [{'type': 'text', 'text': prompt}]
    for item in blocks:
        if isinstance(item, str):
            content.append({'type': 'text', 'text': item})
        else:
            _name, path = item
            content.append({'type': 'image_url',
                            'image_url': {'url': data_url(path), 'detail': 'high'}})
    body = {'model': MODEL, 'temperature': 0, 'max_tokens': max_tokens,
            'messages': [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': content}]}
    req = urllib.request.Request(next_endpoint(), data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        resp = json.load(r)
    return resp, time.time() - t0

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
    q4 = r.get('q4_temporal') or {}
    flags = q4.get('flags') or []
    if isinstance(flags, str): flags = [flags]
    if q1 in ('明确出现', '疑似出现但看不清'): return 'delete', 'device_or_suspect'
    if q5 == '出现': return 'delete', 'face'
    if q2 in ('未完成', '部分完成'): return 'delete', 'task_incomplete'
    if q3 in ('严重模糊', '画面发白过曝', '中心区域模糊', '轻度模糊'): return 'delete', 'blur'
    if any(x in ('有抖动', '有长时间静止', '有无意义动作', '疑似测试片段') for x in flags):
        return 'delete', 'temporal'
    if q1 == '无法判断': return 'needs_human', 'device_unknown'
    return 'keep', ''

# ---------------- 选择 100 条 ----------------
def build_selection():
    man = [json.loads(l) for l in open(os.path.join(ROOT, 'manifest.jsonl'), encoding='utf-8') if l.strip()]
    dev_re = re.compile(r'平板|手机|设备|腰包|屏幕|漏板')
    dev_text = {}
    if os.path.isfile(DETAILS):
        for line in open(DETAILS, encoding='utf-8'):
            f = line.rstrip('\n').split('\t')
            if len(f) >= 6: dev_text[(f[0], f[2])] = f[5]
    for m in man:
        key = (str(m['dataset_id']), str(m['episode_index']))
        m['_dev'] = (m['audit_status'] == 'invalid') and (
            m['invalid_reason'] == '采集设备屏幕露出' or bool(dev_re.search(dev_text.get(key, ''))))
    inv = [m for m in man if m['audit_status'] == 'invalid']
    val = [m for m in man if m['audit_status'] == 'valid']
    buckets = collections.defaultdict(list)
    for m in inv:
        if m['_dev']: buckets['device'].append(m)
        else: buckets[m['invalid_reason'] or 'none'].append(m)
    quota = {'device': 30, '未按要求完成任务': 12, '全程静止无动作': 6, '发生意外干扰': 3,
             '包含多余动作': 4, '视频模糊不清': 2, '手部露出少': 1, '其他': 2}
    sel = []
    for k, n in quota.items():
        sel += buckets.get(k, [])[:n]
    sel += val[:40]
    return sel

# ---------------- 单条处理 ----------------
def process(m):
    t_start = time.time()
    ep_dir = os.path.join(ROOT, m['local_dir'])
    ds_meta = os.path.join(ep_dir, '..', '..', 'meta', 'tasks.jsonl')
    task_text = ''
    try:
        cands = []
        for line in open(ds_meta, encoding='utf-8'):
            j = json.loads(line); t = (j.get('task') or '').strip()
            if not t or t in ('bad', 'null', 'unlabeled') or t.startswith('bad'): continue
            if 'object spatial relation' in t: continue
            if '://' in t or t.startswith('http'): continue
            if len(t) < 6: continue
            if re.match(r'^b[\u4e00-\u9fff]', t): continue
            cands.append(t)
        if cands: task_text = max(cands, key=len)
    except Exception:
        pass
    vids = [f for f in m['files'] if f.startswith('videos/')]
    video = os.path.join(ep_dir, sorted(vids)[0]) if vids else None
    if not video or not os.path.isfile(video):
        return {'dataset_id': m['dataset_id'], 'episode_index': m['episode_index'], 'error': 'no_video',
                'gt': m['audit_status'], 'reason': m.get('invalid_reason')}
    fd = os.path.join(FRAMES, f"{m['dataset_id']}_{m['episode_index']}")
    dur = probe_duration(video)
    names = extract(video, fd, dur)
    ordered = []
    for pre in ('A', 'B', 'C'):
        for n in sorted(names):
            if n.startswith(pre + '_'): ordered.append((n[:-4], os.path.join(fd, n)))
    prompt = user_prompt(task_text, f"时长约 {dur:.0f} 秒，30fps；共提供 {len(ordered)} 帧")

    # ---- v15 新增：首尾密集帧（单独落盘，不干扰主轴）----
    htd = os.path.join(FRAMES, f"{m['dataset_id']}_{m['episode_index']}_ht")
    ht_labels, ht_stamps = extract_headtail(video, htd, dur)
    ht_blocks = build_ht_blocks(htd, ht_labels, ht_stamps)
    n_ht = sum(1 for b in ht_blocks if not isinstance(b, str))
    win = headtail_win(dur)
    ht_prompt = (HT_PROMPT + f"\n\n【本片信息】时长约 {dur:.0f} 秒；"
                 f"共 {n_ht} 帧，取自视频开头 0~{win:.1f}s 与结尾 {max(dur-win,0):.1f}~{dur:.1f}s。")
    t_vlm = 0; t_audit = 0; t_ht = 0.0; usage = {}
    ht = None; ht_verdict = ''
    try:
        hresp, hdt = call_vlm(ht_blocks, ht_prompt, max_tokens=3000); t_ht += hdt
        merge_usage(usage, hresp.get('usage'))
        ht = parse_obj(hresp['choices'][0]['message'].get('content') or '')
        if ht is None:                     # 解析失败 → 重试一次
            hresp2, hdt2 = call_vlm(ht_blocks, ht_prompt + '\n\n【重要】只输出一个 JSON 对象。', max_tokens=3000)
            t_ht += hdt2
            merge_usage(usage, hresp2.get('usage'))
            ht = parse_obj(hresp2['choices'][0]['message'].get('content') or '')
        if isinstance(ht, dict):
            ht_verdict = str(ht.get('lit_phone_in_headtail') or '')
    except Exception as e:
        ht = {'error': str(e)}
    try:
        resp, dt = call_vlm(ordered, prompt); t_vlm = dt
        merge_usage(usage, resp.get('usage'))   # ★ 累加，不覆盖首尾检测的用量
        ch = resp['choices'][0]
        msg = ch['message']
        content = msg.get('content') or ''
        reasoning = msg.get('reasoning') or ''
        finish = ch.get('finish_reason')
        parsed = parse_obj(content)
        if parsed is None:            # 解析失败 → 重试一次，明确要求纯 JSON
            resp2, dt2 = call_vlm(ordered, prompt + '\n\n【重要】上一次回复无法解析为 JSON。请**只输出一个 JSON 对象**，不要任何解释文字，不要 Markdown 代码块。', max_tokens=8000)
            t_vlm += dt2
            merge_usage(usage, resp2.get('usage'))
            c2 = (resp2['choices'][0]['message'].get('content') or '')
            p2 = parse_obj(c2)
            if p2:
                parsed = p2
                content = c2
    except Exception as e:
        return {'dataset_id': m['dataset_id'], 'episode_index': m['episode_index'], 'error': f'vlm:{e}',
                'gt': m['audit_status'], 'reason': m.get('invalid_reason')}
    verdict, rule = decide(parsed)
    audit = None
    if verdict == 'delete':          # v11 行为：仅主判定为删除时留档反证审计（结果不采纳）
        try:
            aresp, adt = call_vlm(ordered, AUDIT_PROMPT.format(prev=json.dumps(parsed, ensure_ascii=False)), max_tokens=500)
            t_audit = adt
            audit = parse_obj(aresp['choices'][0]['message'].get('content'))
        except Exception as e:
            audit = {'error': str(e)}

    # ---- v15 聚合：主判定没判删时，看首尾"亮屏手机"检测 ----
    ht_fired = (ht_verdict == '有') or (HT_FIRE_ON_UNKNOWN and ht_verdict == '无法判断')
    if verdict != 'delete' and ht_fired:
        verdict, rule = 'delete', 'headtail_lit_phone'
    return {'dataset_id': m['dataset_id'], 'episode_index': m['episode_index'],
            'name': m.get('collection_task_name'), 'gt': m['audit_status'],
            'gt_reason': m.get('invalid_reason'), 'gt_is_device': m['_dev'],
            'verdict': verdict, 'rule': rule, 'audit': audit, 'raw': parsed,
            'prompt_version': PROMPT_VERSION,
            'prompt_sha': hashlib.sha256((SYSTEM + prompt).encode('utf-8')).hexdigest()[:16],
            'task_text': task_text,
            'finish': finish, 'reasoning_len': len(reasoning), 'raw_text': content[:1500],
            't_vlm_s': round(t_vlm, 1), 't_audit_s': round(t_audit, 1),
            't_total_s': round(time.time() - t_start, 1), 'usage': usage,
            'sample_ver': SAMPLE_VER, 'dur_s': round(dur, 1),
            'ht': ht, 'ht_verdict': ht_verdict, 'ht_fired': ht_fired,
            'ht_n_frames': n_ht, 't_ht_s': round(t_ht, 1),
            'ht_prompt_sha': hashlib.sha256(ht_prompt.encode('utf-8')).hexdigest()[:16]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=100)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--tag', default='run1')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    # 落盘本次使用的提示词，保证结果可追溯（结果文件里也记录 prompt_version / prompt_sha）
    pdir = os.path.join(OUT, 'prompts'); os.makedirs(pdir, exist_ok=True)
    with open(os.path.join(pdir, f'{PROMPT_VERSION}.txt'), 'w', encoding='utf-8') as fh:
        fh.write('===== SYSTEM =====\n' + SYSTEM + '\n\n===== USER TEMPLATE =====\n' + user_prompt('{task_text}', '{frames_desc}'))
        fh.write('\n\n===== v15 首尾"亮屏手机"独立检测提示词 =====\n' + HT_PROMPT)
        fh.write('\n\n===== 首尾抽帧参数 =====\n'
                 'plan_headtail(dur) = H(6帧, [0,win]) + T(6帧, [dur-win,dur]) + E(1帧, -sseof -0.1)\n'
                 f'win = min(2.5, max(0.4, dur*0.10))，HT_PX={HT_PX}\n'
                 f'HT_FIRE_ON_UNKNOWN = {HT_FIRE_ON_UNKNOWN}\n'
                 f'SAMPLE_VER = {SAMPLE_VER}')
    print(f'提示词已存档: prompts/{PROMPT_VERSION}.txt  (提示词哈希示例见结果文件 prompt_sha)', flush=True)
    sel = build_selection()[:a.limit]
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
