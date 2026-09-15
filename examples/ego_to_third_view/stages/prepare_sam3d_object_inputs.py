"""Export ranked real RGB/mask pairs, without using generated views as evidence."""
from runtime import require_launcher
require_launcher()

import argparse
import json
from pathlib import Path
import cv2
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--run', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
frames = sorted((a.run/'input/ego_action_4s_16s/extracted_images').glob('*.jpg'))
report = {}
for name in ['alarm_clock_v2','small_cylinder_v2','tall_cylinder']:
    candidates = []
    for i in range(0,len(frames),6):
        m = cv2.imread(str(a.run/'reconstruction/sam2'/name/'masks'/f'{i:04d}.png'),0)
        rgb = cv2.imread(str(frames[i]))
        if m is None or rgb is None:
            continue
        ys,xs = np.where(m>127)
        if len(xs)<500:
            continue
        edge = (xs.min()<3 or ys.min()<3 or xs.max()>m.shape[1]-4 or ys.max()>m.shape[0]-4)
        sharpness = float(cv2.Laplacian(cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY),cv2.CV_32F)[m>127].var())
        score = len(xs)*np.log1p(sharpness)*(0.1 if edge else 1)
        candidates.append((float(score),i,len(xs),sharpness))
    chosen=[]
    for candidate in sorted(candidates,reverse=True):
        if all(abs(candidate[1]-other[1])>=24 for other in chosen):
            chosen.append(candidate)
        if len(chosen)==3:
            break
    output=a.output/name
    output.mkdir(parents=True,exist_ok=True)
    report[name]=[]
    for rank,(score,i,area,sharpness) in enumerate(chosen):
        rgb=cv2.imread(str(frames[i]))
        mask=cv2.imread(str(a.run/'reconstruction/sam2'/name/'masks'/f'{i:04d}.png'),0)
        cv2.imwrite(str(output/f'rank{rank}_frame{i:04d}_rgb.png'),rgb)
        cv2.imwrite(str(output/f'rank{rank}_frame{i:04d}_mask.png'),mask)
        report[name].append(dict(frame=i,score=score,area_px=area,sharpness=sharpness))
(a.output/'selection.json').write_text(json.dumps({'objects':report,'limitations':'Area/sharpness ranking only; inspect hand occlusion and cross-view mesh reprojection before accepting a generated asset.'},indent=2))
print(json.dumps(report,indent=2))
