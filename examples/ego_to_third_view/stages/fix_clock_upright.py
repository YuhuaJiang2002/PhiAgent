"""Explicit upright-support prior, scored against observed RGB and masks."""
from runtime import require_launcher
require_launcher()

import sys,json
from pathlib import Path
import numpy as np
root=Path(__import__('os').environ['PHI_EGO_ROOT']);r=Path(__import__('os').environ['PHI_EGO_RUN']);sys.path[:0]=[str(root/'third_party/FoundationPose'),str(root/'envs/sam3d-container-cu128/phiagent-native')]
import cv2,trimesh,torch,nvdiffrast.torch as dr
from Utils import make_mesh_tensors,nvdiffrast_render
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
mesh=trimesh.load(r/'foundationpose-layout-v2/alarm_clock/tracking_mesh_metric.obj',force='mesh');mt=make_mesh_tensors(mesh);ctx=dr.RasterizeCudaContext()
data=np.load(r/'vggt-omega-512-allframes-v1/vggt_omega_predictions_compact.npz');K=data['intrinsic_processed_pixels'];rgb=(data['processed_images'].transpose(0,2,3,1)*255).astype(np.uint8);h,w=rgb.shape[1:3]
cal=np.load(r/'vggt-omega-512-allframes-v1/table-aligned/vggt_omega_table_aligned.npz');C=np.broadcast_to(np.eye(4),(360,4,4)).copy();C[:,:3,:3]=cal['R_table_world_to_camera_m'];C[:,:3,3]=cal['t_table_world_to_camera_m']
path=r/'foundationpose-multiview-persistent-v3/alarm_clock/foundationpose_tracks.npz'
with np.load(path) as d:track={k:d[k] for k in d.files}
raw=track['world_from_object'];anchor=raw[120].copy();indices=[90,120,180,240,300,350];masks=[]
for i in indices:
    m=cv2.imread(str(r/'reconstruction/sam2/alarm_clock_v2/masks'/f'{i:04d}.png'),0);masks.append(cv2.resize(m,(w,h),interpolation=cv2.INTER_NEAREST)>127)
def pose(x,up_sign=1):
    yaw,dx,dy=x;front=np.array([np.cos(yaw),np.sin(yaw),0.]);up=np.array([0.,0.,float(up_sign)]);right=np.cross(up,front)
    T=np.eye(4);T[:3,:3]=np.stack([right,up,front],axis=1);T[:2,3]=anchor[:2,3]+[dx,dy];T[2,3]=-.009-(mesh.vertices@T[:3,:3].T)[:,2].min();return T
def score(x,sign):
    T=pose(x,sign);values=[]
    for j,i in enumerate(indices):
        with torch.no_grad():c,z,_=nvdiffrast_render(K=K[i],H=h,W=w,ob_in_cams=torch.tensor((C[i]@T)[None],device='cuda',dtype=torch.float32),mesh_tensors=mt,glctx=ctx)
        pred=z[0].cpu().numpy()>0;obs=masks[j];over=pred&obs;iou=(over).sum()/max((pred|obs).sum(),1)
        color=c[0].cpu().numpy()*255;phot=float(np.abs(color[over]-rgb[i][over]).mean()/255) if over.any() else 1.
        values.append(1-iou+.35*phot)
    return float(np.mean(values))+.1*float(np.sum(np.asarray(x[1:])**2))
ranked=sorted((score([yaw,0,0],sign),yaw,sign) for sign in [1,-1] for yaw in np.linspace(-np.pi,np.pi,24,endpoint=False))
best=None
for _,yaw,sign in ranked[:4]:
    opt=minimize(lambda x:score(x,sign),[yaw,0,0],method='Powell',bounds=[(yaw-.4,yaw+.4),(-.045,.045),(-.045,.045)],options={'maxfev':200,'xtol':.0005,'ftol':.0005})
    if best is None or opt.fun<best[0]:best=(float(opt.fun),opt.x,sign)
new=pose(best[1],best[2]);out=raw.copy();delta=anchor[:3,:3].T@new[:3,:3];shift=new[:3,3]-anchor[:3,3]
out[:,:3,:3]=raw[:,:3,:3]@delta;out[:,:3,3]+=shift
# Source shows an upright handheld clock; reject large inherited roll/pitch.
for i in range(83):
    u=out[i,:3,1];angle=np.arccos(np.clip(u[2],-1,1));bound=np.deg2rad(20)
    if angle>bound:
        horizontal=u[:2]/max(np.linalg.norm(u[:2]),1e-9)
        target=np.r_[horizontal*np.sin(bound),np.cos(bound)]
        axis=np.cross(u,target);norm=np.linalg.norm(axis)
        if norm>1e-8:out[i,:3,:3]=Rotation.from_rotvec(axis/norm*np.arccos(np.clip(u@target,-1,1))).as_matrix()@out[i,:3,:3]
track['world_from_object_before_upright']=raw;track['world_from_object']=out;track['camera_from_object_before_upright']=track['camera_from_object'];track['camera_from_object']=C@out
dest=r/'foundationpose-clock-upright-v4';(dest/'alarm_clock').mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/'alarm_clock/foundationpose_tracks.npz',**track)
for key in ['small_cylinder','tall_cylinder']:
    if not (dest/key).exists():(dest/key).symlink_to(r/'foundationpose-multiview-persistent-v3'/key,target_is_directory=True)
report={'prior':'Clock standing upright with mesh Y axis vertical and support on tabletop, source RGB/mask yaw selection. Handheld tilt capped at 20 degrees based on reviewed source. Visualization correction requested by user, not calibrated GT.','fit_frames':indices,'objective':best[0],'yaw_dx_dy':best[1].tolist(),'up_axis_sign':best[2],'old_anchor':anchor.tolist(),'new_anchor':new.tolist(),'translation_shift_m':shift.tolist(),'motion':'Same frame index, no time resampling; clock-only orientation/support correction; original tracks retained.'}
(dest/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
