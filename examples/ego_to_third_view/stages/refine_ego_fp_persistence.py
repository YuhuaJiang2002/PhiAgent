"""Piecewise rigid scene persistence with explicitly reviewed contact phases.

Never freeze interaction frames. Preserve raw FoundationPose tracks separately.
"""
from runtime import require_launcher
require_launcher()

import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--input-root',default='foundationpose-v1');p.add_argument('--output-root',default='foundationpose-persistent-v1');a=p.parse_args()
phases={'alarm_clock':(0,82),'small_cylinder':(105,218),'tall_cylinder':(258,359)}
reports={}
for key,(begin,end) in phases.items():
    src=a.run/a.input_root/key/'foundationpose_tracks.npz'
    with np.load(src) as d: data={k:d[k] for k in d.files}
    raw=data['world_from_object'];out=raw.copy();n=len(raw)
    segment_reports=[]
    for lo,hi in [(0,begin),(end+1,n)]:
        if hi-lo<3: continue
        good=np.flatnonzero(data['valid'][lo:hi]&np.isfinite(raw[lo:hi]).all((1,2)))+lo
        if len(good)<3: continue
        w=np.maximum(data['silhouette_iou'][good],.01)**2
        pos=np.median(raw[good,:3,3],axis=0)
        rot=Rotation.from_matrix(raw[good,:3,:3]).mean(weights=w)
        anchor=np.eye(4);anchor[:3,:3]=rot.as_matrix();anchor[:3,3]=pos
        out[lo:hi]=anchor
        rms=float(np.sqrt(np.mean(np.sum((raw[good,:3,3]-pos)**2,axis=1))))
        segment_reports.append({'start':lo,'stop_exclusive':hi,'raw_translation_rms_m':rms,'refined_translation_rms_m':0.0})
        # Blend correction into the adjacent interaction, not the action time.
        boundary=hi if lo==0 else lo-1
        if not 0<=boundary<n: continue
        delta=rot*Rotation.from_matrix(raw[boundary,:3,:3]).inv()
        interp=Slerp([0,1],Rotation.concatenate([Rotation.identity(),delta]))
        dp=pos-raw[boundary,:3,3]
        for j in range(12):
            i=boundary+j if lo==0 else boundary-j
            if not begin<=i<=end: continue
            weight=(1-j/12)**2
            out[i,:3,3]+=dp*weight
            out[i,:3,:3]=(interp(weight)*Rotation.from_matrix(out[i,:3,:3])).as_matrix()
    data['world_from_object_raw']=raw;data['world_from_object']=out
    # camera_from_object must correspond to the refined world trajectory too.
    aligned=np.load(a.run/'vggt-omega-512-allframes-v1/table-aligned/vggt_omega_table_aligned.npz')
    C=np.broadcast_to(np.eye(4),(n,4,4)).copy();C[:,:3,:3]=aligned['R_table_world_to_camera_m'][:n];C[:,:3,3]=aligned['t_table_world_to_camera_m'][:n]
    data['camera_from_object_raw']=data['camera_from_object'];data['camera_from_object']=C@out
    dest=a.run/a.output_root/key;dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/'foundationpose_tracks.npz',**data)
    reports[key]={'reviewed_interaction_frames_inclusive':[begin,end],'static_segments':segment_reports,'max_translation_correction_m':float(np.linalg.norm(out[:,:3,3]-raw[:,:3,3],axis=1).max()),'metrics_note':'Stored IoU/depth residuals describe RAW FP estimates, not refined reprojections.'}
(a.run/a.output_root/'manifest.json').write_text(json.dumps({'method':'Reviewed static spans + robust SE3 anchor; 12-frame boundary correction. No action time resampling.','claim':'Visualization prior, not independent evidence of metric accuracy.','objects':reports},indent=2))
print(json.dumps(reports,indent=2))
