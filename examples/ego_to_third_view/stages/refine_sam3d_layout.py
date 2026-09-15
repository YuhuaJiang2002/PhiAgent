"""Single-view metric layout refinement, with original assets preserved."""
from runtime import require_launcher
require_launcher()

import argparse,json,sys,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--object',required=True);a=p.parse_args()
root=Path(__import__('os').environ['PHI_EGO_ROOT']);sys.path[:0]=[str(root/'envs/sam3d-container-cu128/phiagent-native'),str(root/'third_party/FoundationPose')]
import cv2,torch,trimesh,nvdiffrast.torch as dr
from Utils import make_mesh_tensors,nvdiffrast_render
src=a.run/'sam3d-meshes-v1'/a.object;out=a.run/'sam3d-layout-v2'/a.object;out.mkdir(parents=True,exist_ok=True)
with np.load(src/'sam3d_state.npz') as z: data={k:z[k] for k in z.files}
info=json.loads((src/'manifest.json').read_text());frame=info['frame']
fp=np.load(a.run/'foundationpose-v1'/a.object/'foundationpose_tracks.npz')
T0=fp['camera_from_object'][frame].copy();scene=trimesh.load(src/'object.glb');mesh=scene.to_geometry() if isinstance(scene,trimesh.Scene) else scene
if hasattr(mesh.visual.material,'to_simple'): mesh.visual.material=mesh.visual.material.to_simple()
mesh.apply_scale(data['object_scale']);mt=make_mesh_tensors(mesh);base=mt['pos'].clone();ctx=dr.RasterizeCudaContext()
depth=data['depth_m'];obs=data['mask'].astype(bool);h,w=depth.shape;K=data['K'];count=0;start=time.monotonic()
def evaluate(x,save=False):
    global count
    T=T0.copy();T[:3,:3]=Rotation.from_rotvec(x[:3]).as_matrix()@T0[:3,:3];T[:3,3]+=x[3:6]
    mt['pos']=base*float(np.exp(x[6]))
    with torch.no_grad():
        c,d,_=nvdiffrast_render(K=K,H=h,W=w,ob_in_cams=torch.as_tensor(T[None],device='cuda',dtype=torch.float32),mesh_tensors=mt,glctx=ctx)
    d=d[0].cpu().numpy();valid=d>0;overlap=valid&obs&(depth>0)
    iou=float((valid&obs).sum()/max((valid|obs).sum(),1));err=float(np.median(np.abs(d[overlap]-depth[overlap]))) if overlap.any() else .5
    loss=1-iou+.2*min(err/.02,10)+.005*np.sum(x[:3]**2)
    count+=1
    if save: return T,iou,err,c[0].cpu().numpy()
    return loss
x0=np.zeros(7);_,initial_iou,initial_depth,_=evaluate(x0,True)
fit=minimize(evaluate,x0,method='Powell',bounds=[(-.5,.5)]*3+[(-.05,.05),(-.05,.05),(-.15,.15),(np.log(.55),np.log(1.4))],options={'maxfev':700,'xtol':.001,'ftol':.0001})
T,iou,err,color=evaluate(fit.x,True)
accepted=bool(iou>=initial_iou and err<=initial_depth+.005)
if not accepted: T,iou,err,color=evaluate(x0,True)
factor=float(np.exp(fit.x[6])) if accepted else 1.
data['camera_from_object']=T;data['object_scale']=data['object_scale']*factor
np.savez(out/'sam3d_state.npz',**data)
if not (out/'object.glb').exists(): (out/'object.glb').symlink_to(src/'object.glb')
for name in ['observed_rgb.png','observed_mask.png']:
    if not (out/name).exists(): (out/name).symlink_to(src/name)
rgb=cv2.imread(str(src/'observed_rgb.png'));c=(color[...,::-1]*255).astype(np.uint8)
cv2.imwrite(str(out/'layout_comparison.jpg'),np.concatenate([rgb,c],axis=1))
report={'frame':frame,'object':a.object,'initial_iou':initial_iou,'refined_iou':iou,'initial_depth_residual_m':initial_depth,'refined_depth_residual_m':err,'scale_factor':factor,'accepted':accepted,'evaluations':count,'elapsed_s':time.monotonic()-start,'claim':'Fit against one observed mask and predicted VGGT depth; requires multi-frame validation, not GT metric accuracy.'}
(out/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
