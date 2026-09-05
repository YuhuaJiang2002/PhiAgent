"""Fit one SE3 per reviewed static span against multiple observed masks.

Bounded optimization, fixed camera estimates and fixed meshes. Never infer GT.
"""
from runtime import require_launcher
require_launcher()

import sys,json
from pathlib import Path
import numpy as np
root=Path(__import__('os').environ['PHI_EGO_ROOT']);run=Path(__import__('os').environ['PHI_EGO_RUN'])
sys.path[:0]=[str(root/'envs/sam3d-container-cu128/phiagent-native'),str(root/'third_party/FoundationPose')]
import cv2,torch,trimesh,nvdiffrast.torch as dr
from Utils import nvdiffrast_render,make_mesh_tensors
from scipy.spatial.transform import Rotation,Slerp
from scipy.optimize import minimize
base=run/'foundationpose-layout-v2';dest=run/'foundationpose-multiview-persistent-v3';dest.mkdir(exist_ok=True)
v=np.load(run/'vggt-omega-512-allframes-v1/vggt_omega_predictions_compact.npz');K=v['intrinsic_processed_pixels'];h,w=v['processed_images'].shape[2:]
c=np.load(run/'vggt-omega-512-allframes-v1/table-aligned/vggt_omega_table_aligned.npz')
C=np.broadcast_to(np.eye(4),(360,4,4)).copy();C[:,:3,:3]=c['R_table_world_to_camera_m'];C[:,:3,3]=c['t_table_world_to_camera_m']
context=dr.RasterizeCudaContext();report={}
for key,maskkey,begin,end in [('alarm_clock','alarm_clock_v2',0,82),('small_cylinder','small_cylinder_v2',105,218),('tall_cylinder','tall_cylinder',258,359)]:
    with np.load(base/key/'foundationpose_tracks.npz') as z:data={k:z[k] for k in z.files}
    raw=data['world_from_object'];out=raw.copy();mesh=trimesh.load(base/key/'tracking_mesh_metric.obj',force='mesh');mt=make_mesh_tensors(mesh);report[key]=[]
    for lo,hi in [(0,begin),(end+1,360)]:
        if hi-lo<3:continue
        indices=np.linspace(lo,hi-1,12).astype(int)
        masks=[]
        for i in indices:
            m=cv2.imread(str(run/'reconstruction/sam2'/maskkey/'masks'/f'{i:04d}.png'),0)
            masks.append(cv2.resize(m,(w,h),interpolation=cv2.INTER_NEAREST)>127)
        masks=np.asarray(masks)
        anchor=np.eye(4);anchor[:3,3]=np.median(raw[lo:hi,:3,3],axis=0);anchor[:3,:3]=Rotation.from_matrix(raw[lo:hi,:3,:3]).mean().as_matrix()
        def pose(x):
            T=anchor.copy();T[:3,3]+=x[:3];T[:3,:3]=Rotation.from_rotvec(x[3:]).as_matrix()@anchor[:3,:3];return T
        def loss(x):
            T=pose(x);scores=[]
            for j,i in enumerate(indices):
                with torch.no_grad():
                    _,z,_=nvdiffrast_render(K=K[i],H=h,W=w,ob_in_cams=torch.tensor((C[i]@T)[None],device='cuda',dtype=torch.float32),mesh_tensors=mt,glctx=context)
                pred=z[0].cpu().numpy()>0;obs=masks[j]
                scores.append((pred&obs).sum()/max((pred|obs).sum(),1))
            return 1-float(np.mean(scores))+.002*float(np.sum((x[:3]/.03)**2))+ .001*float(np.sum((x[3:]/.25)**2))
        before=loss(np.zeros(6));res=minimize(loss,np.zeros(6),method='Powell',bounds=[(-.03,.03)]*3+[(-.25,.25)]*3,options={'maxfev':350,'xtol':.0003,'ftol':.0003})
        accepted=bool(res.fun<before);T=pose(res.x) if accepted else anchor;out[lo:hi]=T
        boundary=hi if lo==0 else lo-1
        if 0<=boundary<360:
            delta=Rotation.from_matrix(T[:3,:3]) * Rotation.from_matrix(raw[boundary,:3,:3]).inv()
            interp=Slerp([0,1],Rotation.concatenate([Rotation.identity(),delta]));dp=T[:3,3]-raw[boundary,:3,3]
            for j in range(12):
                i=boundary+j if lo==0 else boundary-j
                if begin<=i<=end:
                    weight=(1-j/12)**2;out[i,:3,3]+=dp*weight;out[i,:3,:3]=(interp(weight)*Rotation.from_matrix(out[i,:3,:3])).as_matrix()
        report[key].append({'start':lo,'stop':hi,'sampled_frames':indices.tolist(),'objective_before':before,'objective_after':float(res.fun),'accepted':accepted,'offset_m_rotvec':res.x.tolist(),'evaluations':int(res.nfev)})
        print(key,lo,hi,report[key][-1],flush=True)
    data['world_from_object_raw']=raw;data['world_from_object']=out;data['camera_from_object_raw']=data['camera_from_object'];data['camera_from_object']=C@out
    (dest/key).mkdir(exist_ok=True);np.savez_compressed(dest/key/'foundationpose_tracks.npz',**data)
(dest/'manifest.json').write_text(json.dumps({'method':'Bounded multi-frame silhouette SE3 fit on reviewed static spans, fixed estimated cameras. Sampled-frame objective is training fit, not held-out evaluation.','objects':report},indent=2))
