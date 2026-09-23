# -*- coding: utf-8 -*-
"""构建「漏检复核 18 条」交付包。

产物结构（服务器 /data1/review/漏检复核_18条/）：
  00_说明.md              总说明（生成后由外部覆盖为正式版）
  漏检清单.csv            结构化清单
  拼图/<ds>_<ep>.jpg      带帧标签+秒数的拼图（模型实际看到的帧，按模型顺序 H,A,T,E,B,C）
  模型判定/<ds>_<ep>.json 模型对该条的原始 JSON 判定
  视频/<类>_<ds>_<ep>_left_rgb.mp4   模型实际使用的机位（left_rgb）
  漏检复核_18条.zip
"""
import csv
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, '/data1/review/egoqc')
import qc_run_db as q

OUT = '/data1/review/漏检复核_18条'
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
GRID = '5x8'          # 5 列，最多 40 帧
COLS = 5
PAD = 5
TILE_W = 320
TILE_H = 180          # B 组是底部裁切帧，长宽比与其它组不同，拼图时统一缩放到该尺寸内居中

# A 类 = 原本就漏检（模型 q1 判"未出现"，与豁免无关）
# B 类 = 被"强制豁免"误杀（原本判删正确，只因证据帧全在首尾区被改判保留）
A = [(9605, 1), (8685, 0), (8866, 0), (8878, 0), (9115, 2), (9480, 0), (9052, 0), (8865, 0)]
B = [(8863, 0), (9179, 23), (9052, 9), (9662, 13), (9179, 49), (9662, 53),
     (9424, 0), (9318, 19), (9060, 1), (8898, 0)]


def load_preds(tag='db100nt2'):
    d = {}
    with open(f'/data1/review/egoqc/preds_{tag}.jsonl', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if line:
                x = json.loads(line)
                d[(int(x['dataset_id']), int(x['episode_index']))] = x
    return d


def stamps_for(dur):
    """复现 extract() 的帧 -> 秒数映射"""
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


def ordered_frames(ds, ep, dur):
    src = f'/data1/review/egoqc/frames/{ds}_{ep}'
    st = stamps_for(dur)
    files = os.listdir(src)
    out = []
    for grp in q.GROUP_DESC:                     # H,A,T,E,B,C 固定顺序
        idxs = sorted(int(f.split('_')[1][:-4]) for f in files
                      if f.startswith(grp + '_') and f.endswith('.jpg'))
        for i in idxs:
            arr = st.get(grp) or []
            t = arr[i] if i < len(arr) else -1.0
            out.append((f'{grp}_{i:02d}', t, os.path.join(src, f'{grp}_{i:02d}.jpg')))
    return out


def make_sheet(ds, ep, dur, dest):
    """用 PIL 拼图并逐帧烧标签。

    不用 ffmpeg tile：实测在该输入下 tile 只填了 17/35 格（不同组的帧宽高/pix_fmt 不一致），
    且 drawtext 的 source_basename 元数据在 glob 输入下为空、标签不会渲染。
    PIL 完全可控，避免这两类坑。需要 /data1/Video-Depth-Anything-Large/env/bin/python3。
    """
    from PIL import Image, ImageDraw, ImageFont
    frames = ordered_frames(ds, ep, dur)
    cols = COLS
    rows = (len(frames) + cols - 1) // cols
    label_h = 20
    cw, ch = TILE_W + PAD, TILE_H + label_h + PAD
    canvas = Image.new('RGB', (cols * cw + PAD, rows * ch + PAD), (32, 32, 32))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(FONT, 15)
    for i, (label, t, p) in enumerate(frames):
        im = Image.open(p).convert('RGB')
        im.thumbnail((TILE_W, TILE_H), Image.LANCZOS)
        cell = Image.new('RGB', (TILE_W, TILE_H), (0, 0, 0))
        cell.paste(im, ((TILE_W - im.width) // 2, (TILE_H - im.height) // 2))
        r, c = divmod(i, cols)
        x = PAD + c * cw
        y = PAD + r * ch
        draw.text((x + 6, y + 2), f'{label} {t:.1f}s', font=font, fill=(255, 220, 60))
        canvas.paste(cell, (x, y + label_h))
    canvas.save(dest, quality=88)
    return 0, ''


def find_video(ds, ep):
    import glob
    c = glob.glob(f'/mnt/checkpoint/zhn/cus000001_mixed150/task-*/dataset-{ds}_*/episode-{ep:06d}')
    if not c:
        return None
    v = glob.glob(c[0] + '/videos/observation.images.left_rgb/chunk-*/*.mp4')
    return v[0] if v else None


def main():
    preds = load_preds()
    for sub in ('拼图', '模型判定', '视频'):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    rows = []
    for grp, lst in (('A', A), ('B', B)):
        for ds, ep in lst:
            x = preds.get((ds, ep))
            if not x:
                print(f'!! 缺 preds: ds{ds} ep{ep}', flush=True)
                continue
            dur = x['dur_s']
            key = f'{ds}_{ep}'
            rc, err = make_sheet(ds, ep, dur, os.path.join(OUT, '拼图', key + '.jpg'))
            print(f'拼图 ds{ds} ep{ep}: rc={rc} {("ERR " + err) if rc else "OK"}', flush=True)
            with open(os.path.join(OUT, '模型判定', key + '.json'), 'w', encoding='utf-8') as fh:
                json.dump({k: x.get(k) for k in
                           ('dataset_id', 'episode_index', 'gt', 'gt_reason', 'verdict', 'rule',
                            'dur_s', 'n_frames', 'groups', 'raw', 'raw_text', 'usage', 'think')},
                          fh, ensure_ascii=False, indent=1)
            v = find_video(ds, ep)
            if v:
                shutil.copy(v, os.path.join(OUT, '视频', f'{grp}_{key}_left_rgb.mp4'))
            rows.append(dict(类别=('A 原本漏检' if grp == 'A' else 'B 被豁免误杀'),
                             dataset_id=ds, episode_index=ep, 时长秒=round(dur, 1),
                             平台标注=x.get('gt'), 平台原因=x.get('gt_reason'),
                             模型判定=x.get('verdict'), 命中规则=x.get('rule') or '',
                             帧数=x.get('n_frames')))
    with open(os.path.join(OUT, '漏检清单.csv'), 'w', encoding='utf-8-sig', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'\n完成 {len(rows)} 条 -> {OUT}', flush=True)
    for sub in ('拼图', '模型判定', '视频'):
        p = os.path.join(OUT, sub)
        print(f'  {sub}: {len(os.listdir(p))} 个文件', flush=True)


if __name__ == '__main__':
    main()
