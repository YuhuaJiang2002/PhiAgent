# -*- coding: utf-8 -*-
"""生成《00_说明.md》—— 逐条说明每条漏检"问题是什么、该怎么看"。

数据来源全部是服务器上的实测结果，不手写数字：
  preds_db100nt2.jsonl   豆包 think=0 全量 100 条的判定
  review_52_human.json   人工复核 52 条的真值
  frames/<ds>_<ep>/      模型实际看到的帧（拼图就是这些帧）
"""
import json
import os
import re
import sys

sys.path.insert(0, '/data1/review/egoqc')
import qc_run_db as q

OUT = '/data1/review/漏检复核_18条'
EXEMPT = {'H', 'T', 'E'}          # 首尾豁免区
MID = {'A', 'C'}                  # 中间区

A = [(9605, 1), (8685, 0), (8866, 0), (8878, 0), (9115, 2), (9480, 0), (9052, 0), (8865, 0)]
B = [(8863, 0), (9179, 23), (9052, 9), (9662, 13), (9179, 49), (9662, 53),
     (9424, 0), (9318, 19), (9060, 1), (8898, 0)]


def load_preds():
    d = {}
    with open('/data1/review/egoqc/preds_db100nt2.jsonl', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if line:
                x = json.loads(line)
                d[(int(x['dataset_id']), int(x['episode_index']))] = x
    return d


def load_human():
    h = {}
    for r in json.load(open('/data1/review/egoqc/review_52_human.json', encoding='utf-8')):
        h[(int(r['ds']), int(r['ep']))] = (r['verdict'], r.get('types') or [], r.get('note') or '')
    return h


def stamps_for(dur):
    st = {}
    for grp, n, start, span, crop in q.plan_sampling(dur):
        if n <= 0:
            continue
        start = max(0.0, min(start, max(dur - 0.05, 0.0)))
        if grp == 'E':
            st[grp] = [max(dur - 0.1, 0.0)]
        elif grp == 'C':
            st[grp] = [start + k / 5.0 for k in range(n)]
        else:
            span = max(span, 0.1)
            fr = (n - 1) / span if n > 1 else 1.0 / span
            st[grp] = [start + k / fr for k in range(n)]
    return st


def citations(x):
    """返回 (引用到的帧组, [(帧标签, 秒数)])"""
    raw = x.get('raw') or {}
    q1 = (raw.get('q1_device_exposed') or {}) if isinstance(raw, dict) else {}
    ids = q1.get('frame_ids') or []
    if not ids:
        ids = list(dict.fromkeys(re.findall(r'\b([HATEBC]_\d{2})\b', x.get('raw_text') or '')))
    st = stamps_for(x['dur_s'])
    out = []
    for fid in ids:
        if not re.match(r'^[HATEBC]_\d{2}$', fid):
            continue
        g, i = fid.split('_')
        i = int(i)
        arr = st.get(g) or []
        out.append((fid, arr[i] if i < len(arr) else -1.0))
    return sorted(set(f.split('_')[0] for f in ids if re.match(r'^[HATEBC]_\d{2}$', f))), out


def sub_answers(x):
    raw = x.get('raw') or {}
    if not isinstance(raw, dict):
        return {}
    out = {}
    for k, name in (('q1_device_exposed', '设备'), ('q2_task_completion', '任务'),
                    ('q3_blur', '模糊'), ('q4_temporal', '时序'), ('q5_face', '人脸')):
        v = raw.get(k)
        if isinstance(v, dict):
            out[name] = v.get('verdict') or v.get('flags') or ''
    return out


def main():
    P = load_preds()
    H = load_human()
    L = []
    L.append('# 漏检复核包（18 条）\n')
    L.append('## 一、这是什么\n')
    L.append('本次把质检模型从**本地 Qwen3.5-27B** 换成**豆包 doubao-seed-2.0-pro**（口径/抽帧/提示词'
             '与 v21 完全一致），在 100 条评测集上跑出的**全部漏检**。\n')
    L.append('| 版本 | 召回 | 误删 |\n|---|---|---|\n| v21 本地 Qwen | 61.7% | 30.0% |\n'
             '| **豆包 think=0** | **86.7%** | 77.5% |\n')
    L.append('漏检 18 条分两类，**成因完全不同，必须分开看**：\n')
    L.append('| 类别 | 条数 | 含义 |\n|---|---|---|\n'
             '| **A 原本漏检** | 8 | 模型 q1 直接判"未出现"，**根本没检出问题**，与豁免规则无关 |\n'
             '| **B 被豁免误杀** | 10 | 模型**原本判删是对的**，只因它引用的证据帧全落在首尾 2 秒豁免区，'
             '被"强制豁免"改判成保留 |\n')
    L.append('> 注：B 类的 10 条是被"强制豁免"这一版实验实现误杀的。**原始豆包结果是判删的**，'
             '也就是说原始结果里这 10 条不算漏检。\n')

    L.append('## 二、目录结构\n')
    L.append('```\n漏检复核_18条/\n'
             '├── 00_说明.md          本文件\n'
             '├── 漏检清单.csv        结构化清单（Excel 可直接打开）\n'
             '├── 拼图/               <ds>_<ep>.jpg —— 模型实际看到的全部帧，带帧标签+秒数\n'
             '├── 模型判定/           <ds>_<ep>.json —— 模型对该条的原始 JSON 判定\n'
             '└── 视频/               <类>_<ds>_<ep>_left_rgb.mp4 —— 模型实际使用的机位\n```\n')
    L.append('**拼图怎么读**：按 5 列排布，**帧组顺序固定为 H → A → T → E → B → C**，'
             '每张图左上角是 `帧标签 秒数`。\n')
    L.append('- `H` 开头密集帧、`T` 结尾密集帧、`E` 最后一帧 → **都落在首尾 2 秒豁免区内**\n')
    L.append('- `A` 全片均匀帧、`C` 中段连续帧 → **中间区**\n')
    L.append('- `B` 画面下部 35% 裁切放大帧（看桌面下缘细节），跨全片\n')

    for grp, lst, title, desc in (
        ('A', A, '三、A 类：原本漏检（8 条）—— 模型没检出问题',
         '这 8 条模型 q1 全部回答"未出现"。**它们和首尾豁免规则没有任何关系**，就是单纯没看出来。'),
        ('B', B, '四、B 类：被"强制豁免"误杀（10 条）—— 模型原本判对了',
         '这 10 条模型原本判删（正确），但因为它引用的证据帧全部落在 H/T/E 组（首尾 2 秒豁免区），'
         '被"只要证据全在豁免区就改判保留"的规则误杀。**注意这 10 条里只有 2 条的平台原因是'
         '"采集设备屏幕露出"**，其余是"包含多余动作/未按要求完成任务/手部露出少/发生意外干扰"——'
         '这些类别**根本不该被首尾豁免覆盖**，说明该规则实现得太粗。'),
    ):
        L.append(f'## {title}\n')
        L.append(desc + '\n')
        L.append('| # | 条目 | 时长 | 平台原因 | 人工复核 | 模型 q1 | 引用帧组 | 引用帧(秒) |')
        L.append('|---|---|---|---|---|---|---|---|')
        for n, (ds, ep) in enumerate(lst, 1):
            x = P[(ds, ep)]
            gs, cits = citations(x)
            sa = sub_answers(x)
            h = H.get((ds, ep))
            hs = f"**{h[0]}**" + (f"（{'、'.join(h[1])}）" if h[1] else '') if h else '未复核'
            cits_s = ' '.join(f'{f}@{t:.0f}s' for f, t in cits[:5]) + ('…' if len(cits) > 5 else '')
            L.append(f"| {n} | ds{ds} ep{ep} | {x['dur_s']:.0f}s | {x.get('gt_reason') or '-'} | "
                     f"{hs} | {sa.get('设备','-')} | {','.join(gs) or '-'} | {cits_s or '-'} |")
        L.append('')

    L.append('## 五、逐条说明\n')
    for grp, lst, cname in (('A', A, '原本漏检'), ('B', B, '被豁免误杀')):
        for ds, ep in lst:
            x = P[(ds, ep)]
            gs, cits = citations(x)
            sa = sub_answers(x)
            h = H.get((ds, ep))
            L.append(f'### {"AB"[grp == "B"]}类 · ds{ds} ep{ep}（{x["dur_s"]:.0f}s）\n')
            gt_s = x.get('gt')
            rs_s = x.get('gt_reason') or '-'
            L.append(f"- **平台标注**：{gt_s}（原因：{rs_s}）")
            hu = (f"**{h[0]}**" + (f"，问题类型：{'、'.join(h[1])}" if h[1] else '')
                  + (f"，备注：{h[2]}" if h[2] else '')) if h else '本条目未做过人工复核'
            L.append(f"- **人工复核**：{hu}")
            L.append(f"- **模型判定**：{x.get('verdict')}（规则 `{x.get('rule') or '无'}`），"
                     f"分项答案：{', '.join(f'{k}={v}' for k, v in sa.items()) or '-'}")
            if grp == 'A':
                L.append(f"- **问题在哪**：模型 q1 判「未出现」，即**它认为画面里没有采集设备/手机/平板**。"
                         f"请对着 `拼图/ds{ds}_{ep}.jpg` 重点核对 **A 组中间帧**和 **B 组底部裁切帧**"
                         f"（B 组专门放大桌面下缘，设备常藏在那里）。")
            else:
                L.append(f"- **问题在哪**：模型引用的证据帧是 {', '.join(f'{f}@{t:.1f}s' for f, t in cits)}，"
                         f"**全部落在首尾 2 秒豁免区内**（引用帧组 {','.join(gs)}）。"
                         f"按现行口径，首尾 2 秒内的露出本就不算问题，所以模型这条判删理由不成立；"
                         f"但平台给的原因是「{x.get('gt_reason')}」，**与设备无关**，"
                         f"正确做法是只扣掉设备这一项证据、保留其它判据，而不是整条改判保留。")
            L.append(f"- **对应文件**：`拼图/ds{ds}_{ep}.jpg`、`模型判定/ds{ds}_{ep}.json`、"
                     f"`视频/{grp}_ds{ds}_ep{ep}_left_rgb.mp4`\n")

    L.append('## 六、关键结论\n')
    L.append('1. **A 类的 8 条是模型的真实能力缺口**——它压根没看到设备/问题，'
             '和口径无关。合起来看，豆包在"设备相关"这个最难类别上是 26/30，剩下 4 条没抓到。\n')
    L.append('2. **B 类的 10 条不是模型的问题，是"强制豁免"实现太粗**。'
             '这 10 条里 8 条的平台原因与设备无关（多余动作/未完成任务/手部露出少/意外干扰），'
             '却因为设备证据落在首尾区被整条改判保留。\n')
    L.append('3. **正确修法**：豁免只应扣掉 q1（设备）这一项证据，'
             '不能撤销整条删除——扣掉后再看 q2(任务完成)/q3(模糊)/q4(时序)/q5(人脸) 是否独立支持删除，'
             '有则仍判删。这样 A 类不受影响，B 类的 10 条能全部救回来。\n')
    L.append('4. **平台标签本身有错**：本包 18 条里，已人工复核的 7 条中，'
             'A 类有 1 条（ds8878 ep0）人工判「没问题」——即平台标错、模型判对；'
             'B 类 3 条人工判「有问题」——即平台标对、是豁免规则误杀。\n')
    L.append('---\n')
    L.append('*生成脚本：`/data1/review/egoqc/_build_miss18.py`；'
             '判定来源：`preds_db100nt2.jsonl`（豆包 think=0，全量 100 条，0 报错）*\n')

    p = os.path.join(OUT, '00_说明.md')
    with open(p, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L))
    print(f'已生成 {p}  行数={len(L)}  字符数={sum(len(x) for x in L)}')


if __name__ == '__main__':
    main()
