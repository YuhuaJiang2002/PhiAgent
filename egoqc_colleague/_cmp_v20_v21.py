# -*- coding: utf-8 -*-
"""v20 vs v21 逐条对比：确认「采样窗口 / 豁免窗口对齐」是否解决了 A_15·T_00 缝隙误删。

官方口径 = 平台 audit_status 标签（gt 字段）。
"""
import json
import os
import sys

ROOT = '/data1/review/egoqc'
A = sys.argv[1] if len(sys.argv) > 1 else 'v20'
B = sys.argv[2] if len(sys.argv) > 2 else 'v21'


def load(tag):
    p = os.path.join(ROOT, f'preds_{tag}.jsonl')
    out = {}
    with open(p, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out[(int(d['dataset_id']), int(d['episode_index']))] = d
    return out


def stat(items):
    tp = fn = fp = tn = 0
    for d in items.values():
        dele = d['verdict'] == 'delete'
        inv = d['gt'] == 'invalid'
        if inv and dele:
            tp += 1
        elif inv and not dele:
            fn += 1
        elif (not inv) and dele:
            fp += 1
        else:
            tn += 1
    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return dict(n=len(items), TP=tp, FN=fn, FP=fp, TN=tn,
                recall=round(recall, 4), fp_rate=round(fpr, 4))


a, b = load(A), load(B)
common = sorted(set(a) & set(b))
print(f'{A}: {len(a)} 条   {B}: {len(b)} 条   共有 {len(common)} 条\n')

sa, sb = stat(a), stat(b)
print(f"{'':>10} {'n':>5} {'TP':>4} {'FN':>4} {'FP':>4} {'TN':>4} {'召回':>8} {'误删':>8}")
for tag, s in ((A, sa), (B, sb)):
    print(f"{tag:>10} {s['n']:>5} {s['TP']:>4} {s['FN']:>4} {s['FP']:>4} {s['TN']:>4} "
          f"{s['recall']*100:>7.1f}% {s['fp_rate']*100:>7.1f}%")

print('\n' + '=' * 78)
print(f'仅看两版共有的 {len(common)} 条（严格可比）')
print('=' * 78)
sa2, sb2 = stat({k: a[k] for k in common}), stat({k: b[k] for k in common})
print(f"{'':>10} {'n':>5} {'TP':>4} {'FN':>4} {'FP':>4} {'TN':>4} {'召回':>8} {'误删':>8}")
for tag, s in ((A, sa2), (B, sb2)):
    print(f"{tag:>10} {s['n']:>5} {s['TP']:>4} {s['FN']:>4} {s['FP']:>4} {s['TN']:>4} "
          f"{s['recall']*100:>7.1f}% {s['fp_rate']*100:>7.1f}%")


def flip(k):
    da, db = a[k], b[k]
    return (da['verdict'] == 'delete', db['verdict'] == 'delete')


print('\n' + '=' * 78)
print('翻转明细（delete -> keep）')
print('=' * 78)
fixed_fp, broke_tp, other = [], [], []
for k in common:
    va, vb = flip(k)
    if va and not vb:
        (broke_tp if a[k]['gt'] == 'invalid' else fixed_fp).append(k)
    elif (not va) and vb:
        other.append(('keep->delete', k))

print(f'\n【修复：v20 误删 -> v21 保留】{len(fixed_fp)} 条  (gt=valid，方向正确)')
for k in fixed_fp:
    print(f"  ds{k[0]} ep{k[1]}  {(b[k]['name'] or '')[:22]:<24} dur={b[k]['dur_s']:>6.1f}s  "
          f"gt_reason={(b[k].get('gt_reason') or '')[:18]}  v21_rule={b[k]['rule']}")

print(f'\n【代价：v20 正确删除 -> v21 漏检】{len(broke_tp)} 条  (gt=invalid，方向错误)')
for k in broke_tp:
    print(f"  ds{k[0]} ep{k[1]}  {(b[k]['name'] or '')[:22]:<24} dur={b[k]['dur_s']:>6.1f}s  "
          f"gt_reason={(b[k].get('gt_reason') or '')[:18]}  v21_rule={b[k]['rule']}")

print(f'\n【新增删除：v20 保留 -> v21 删除】{len(other)} 条')
for kind, k in other:
    tag = '★正确' if b[k]['gt'] == 'invalid' else '⚠新增误删'
    print(f"  {tag} ds{k[0]} ep{k[1]}  {b[k]['name'][:22]:<24} dur={b[k]['dur_s']:>6.1f}s  "
          f"v21_rule={b[k]['rule']}")

# ---- 核心假设检验：v20 误删里有多少条其证据帧落在 dur-2.5 ~ dur-2.0 缝隙 ----
print('\n' + '=' * 78)
print('核心假设检验：缝隙帧（dur-2.5s ~ dur-2.0s 之间）作为唯一证据的误删')
print('=' * 78)
SLIVER_LO, SLIVER_HI = -2.5, -2.0


def sliver_evidence(d):
    """从 audit 的 cited_ts / 文本里找是否有落在缝隙区间的时间戳引用。"""
    au = d.get('audit')
    if not isinstance(au, dict):
        return None
    txt = json.dumps(au, ensure_ascii=False)
    dur = d.get('dur_s') or 0
    lo, hi = dur + SLIVER_LO, dur + SLIVER_HI
    hits = []
    for key in ('evidence', 'cited', 'frames', 'notes', 'reason'):
        v = au.get(key)
        if v is None:
            continue
        s = json.dumps(v, ensure_ascii=False)
        # 抓形如 77.8s / 77.8 的数字
        import re
        for m in re.finditer(r'(\d+(?:\.\d+)?)\s*s?\b', s):
            t = float(m.group(1))
            if lo <= t <= hi:
                hits.append((key, t, m.group(0)))
    return hits or None


sliver_fp_a = [k for k in common
               if a[k]['gt'] != 'invalid' and a[k]['verdict'] == 'delete'
               and sliver_evidence(a[k])]
print(f'\nv20 误删总数 = {sa2["FP"]}，其中证据时间戳命中缝隙的 = {len(sliver_fp_a)}')
for k in sliver_fp_a:
    da, db = a[k], b[k]
    print(f"  ds{k[0]} ep{k[1]} dur={da['dur_s']:.1f}s  缝隙=[{da['dur_s']-2.5:.1f},{da['dur_s']-2.0:.1f}]  "
          f"v20={da['verdict']}  v21={db['verdict']}  "
          f"{'✅已修复' if db['verdict'] != 'delete' else '❌仍误删'}")

print('\n' + '=' * 78)
print('耗时对比')
print('=' * 78)
for tag, d in ((A, a), (B, b)):
    tot = sum(x.get('t_total_s') or 0 for x in d.values())
    vlm = sum(x.get('t_vlm_s') or 0 for x in d.values())
    pt = sum((x.get('usage') or {}).get('prompt_tokens') or 0
             for x in d.values() if isinstance(x.get('usage'), dict))
    ct = sum((x.get('usage') or {}).get('completion_tokens') or 0
             for x in d.values() if isinstance(x.get('usage'), dict))
    print(f'{tag}: 单条均 {tot/len(d):.1f}s (VLM {vlm/len(d):.1f}s)  '
          f'prompt_tok={pt} completion_tok={ct}')
