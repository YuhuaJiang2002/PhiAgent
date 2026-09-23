# -*- coding: utf-8 -*-
"""合并 db50t(50) + v3rest(58) => v3 全量 108 条，出完整指标与分层分析。"""
import json
import csv
import os
import collections

D = '/data1/zqy/cus000001_ego_qc_142_v3'
OUT = '/data1/review/egoqc'


def load(tag):
    d = {}
    p = f'{OUT}/preds_{tag}.jsonl'
    if not os.path.exists(p):
        return d
    for line in open(p, encoding='utf-8'):
        if line.strip():
            x = json.loads(line)
            if 'verdict' in x:
                d[(int(x['dataset_id']), int(x['episode_index']))] = x
    return d


def main():
    a, b = load('db50t'), load('v3rest')
    M = dict(a)
    dup = [k for k in b if k in M]
    M.update(b)
    print(f'db50t={len(a)}  v3rest={len(b)}  重叠={len(dup)}  合计={len(M)}')

    rows = {(int(r['dataset_id']), int(r['episode_index'])): r
            for r in csv.DictReader(open(f'{D}/EVAL_INDEX.csv', encoding='utf-8-sig'))}
    hum = {}
    for r in json.load(open(f'{OUT}/review_52_human.json', encoding='utf-8')):
        hum[(int(r['ds']), int(r['ep']))] = (r['verdict'], r.get('types') or [])
    VERD = {(9367, 0): 'task_mismatch', (9367, 1): 'task_mismatch', (9367, 2): 'task_mismatch',
            (9648, 0): 'task_mismatch', (9648, 1): 'task_mismatch', (8927, 0): 'task_name_wrong'}

    keys = sorted(M)
    missing = [k for k in rows if k not in M]
    print(f'清单 108 条中未覆盖: {len(missing)} {missing}')

    def st(ks):
        tp = fn = fp = tn = 0
        for k in ks:
            x = M[k]
            de = x['verdict'] == 'delete'
            iv = x['gt'] == 'invalid'
            if iv and de: tp += 1
            elif iv: fn += 1
            elif de: fp += 1
            else: tn += 1
        rec = tp / (tp + fn) if tp + fn else 0
        fpr = fp / (fp + tn) if fp + tn else 0
        return tp, fn, fp, tn, rec, fpr

    print('\n' + '=' * 74)
    print('v3 全量 108 条 · 平台口径')
    print('=' * 74)
    tp, fn, fp, tn, rec, fpr = st(keys)
    print(f'  n={len(keys)}  TP{tp} FN{fn} FP{fp} TN{tn}')
    print(f'  召回 {rec*100:.1f}%   误删 {fpr*100:.1f}%')

    # 分层
    print('\n--- 按任务描述来源分档 ---')
    for src in ('episode_observed', 'sibling_episode', 'source_tasks_jsonl',
                'platform_sop', 'task_name_majority', 'unavailable'):
        ks = [k for k in keys if M[k].get('task_desc_source') == src]
        if ks:
            t2, f2, p2, n2, r2, fr2 = st(ks)
            print(f'  {src:<20} n={len(ks):>3}  召回 {r2*100:5.1f}%  误删 {fr2*100:5.1f}%')

    print('\n--- 按评测可用性分档 ---')
    for u in ('yes', 'limited', 'no'):
        ks = [k for k in keys if M[k].get('eval_usable') == u]
        if ks:
            t2, f2, p2, n2, r2, fr2 = st(ks)
            print(f'  {u:<9} n={len(ks):>3}  召回 {r2*100:5.1f}%  误删 {fr2*100:5.1f}%')

    # 误删真伪
    print('\n' + '=' * 74)
    print('误删真伪核验')
    print('=' * 74)
    fps = [k for k in keys if M[k]['gt'] != 'invalid' and M[k]['verdict'] == 'delete']
    cw = cr = 0
    for k in sorted(fps):
        h = hum.get(k)
        if h:
            if h[0] == '有问题': cw += 1
            else: cr += 1
    print(f'  误删 {len(fps)} 条（平台口径 {len(fps)}/42 = {len(fps)/42*100:.1f}%）')
    print(f'  已人工复核且判「有问题」(平台标错): {cw}')
    print(f'  已人工复核且判「没问题」(确认误删): {cr}')
    print(f'  未复核: {len(fps)-cw-cr}')
    print(f'  => 已核实部分真实误删率 = {cr}/{cw+cr} = {cr/max(cw+cr,1)*100:.1f}%')
    print(f'  => 最坏情况（未复核全算误删） = {len(fps)-cw}/42 = {(len(fps)-cw)/42*100:.1f}%')

    # 漏检
    fns = [k for k in keys if M[k]['gt'] == 'invalid' and M[k]['verdict'] != 'delete']
    print(f'\n  漏检 {len(fns)} 条（{len(fns)}/66 = {len(fns)/66*100:.1f}%）:')
    for k in sorted(fns):
        x = M[k]
        print(f'    ds{k[0]:<5} ep{k[1]:<3} {x["dur_s"]:>6.0f}s 原因={x.get("gt_reason"):<12} '
              f'描述来源={x.get("task_desc_source")}')

    # 分原因召回
    print('\n--- 分原因召回 ---')
    bt = collections.defaultdict(lambda: [0, 0])
    for k in keys:
        x = M[k]
        if x['gt'] == 'invalid':
            r = x.get('gt_reason') or '其他'
            bt[r][1] += 1
            if x['verdict'] == 'delete': bt[r][0] += 1
    for r, (a2, b2) in sorted(bt.items(), key=lambda x: -x[1][1]):
        print(f'  {r:<16} {a2}/{b2}')

    # 历史对比
    print('\n' + '=' * 74)
    print('历史版本对比（口径各不相同，仅作参考）')
    print('=' * 74)
    hist = [('v20 本地Qwen/旧集合100', 45, 15, 24, 16), ('v21 本地Qwen/旧集合100', 37, 23, 12, 28),
            ('豆包think=0/旧集合100', 52, 8, 31, 9), ('豆包think=auto/旧集合100', 51, 9, 17, 23),
            ('v3 50条分层/思考', 28, 2, 11, 9)]
    print(f"{'版本':<28}{'召回':>9}{'误删':>9}")
    for nm, t2, f2, p2, n2 in hist:
        print(f'{nm:<28}{t2/(t2+f2)*100:>8.1f}%{p2/(p2+n2)*100:>8.1f}%')
    print(f'{"★ v3 全量108/思考":<28}{rec*100:>8.1f}%{fpr*100:>8.1f}%')

    # 落盘
    out = dict(n=len(keys), TP=tp, FN=fn, FP=fp, TN=tn, recall=round(rec, 4),
               fp_rate_on_valid=round(fpr, 4),
               fp_confirmed=cr, fp_platform_wrong=cw, fp_unreviewed=len(fps)-cw-cr,
               fp_worst_case_rate=round((len(fps)-cw)/42, 4))
    with open(f'{OUT}/metrics_v3full_merged.json', 'w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    with open(f'{OUT}/preds_v3full_merged.jsonl', 'w', encoding='utf-8') as fh:
        for k in keys:
            fh.write(json.dumps(M[k], ensure_ascii=False) + '\n')
    print(f'\n已写 metrics_v3full_merged.json / preds_v3full_merged.jsonl')


if __name__ == '__main__':
    main()
