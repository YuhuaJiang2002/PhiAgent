"""Track a SAM3D mesh against observed RGB/depth, retaining camera/world poses."""
from runtime import require_launcher
require_launcher()

import argparse,json,sys,time
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--mesh-dir',type=Path,required=True)
    p.add_argument('--object',required=True)
    p.add_argument('--foundationpose-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int,default=360)
    p.add_argument('--native-dir',type=Path,default=None)
    p.add_argument('--sam-anchor-frame',type=int,default=None)
    a=p.parse_args()
    import cv2,trimesh,torch
    sys.path[:0]=[str(a.foundationpose_root),str(a.foundationpose_root/'mycpp/build')]
    if a.native_dir: sys.path.insert(0,str(a.native_dir))
    from estimater import FoundationPose
    from learning.training.predict_score import ScorePredictor
    from learning.training.predict_pose_refine import PoseRefinePredictor
    import nvdiffrast.torch as dr
    from Utils import nvdiffrast_render,make_mesh_tensors
    scene=trimesh.load(a.mesh_dir/'object.glb')
    mesh=scene.to_geometry() if isinstance(scene,trimesh.Scene) else scene
    if hasattr(mesh.visual,'material') and hasattr(mesh.visual.material,'to_simple'):
        mesh.visual.material=mesh.visual.material.to_simple()
    with np.load(a.mesh_dir/'sam3d_state.npz') as d:
        mesh.apply_scale(d['object_scale'])
        sam_pose=d['camera_from_object'].copy()
    if not np.all(np.isfinite(mesh.vertices)): raise RuntimeError('Nonfinite mesh')
    a.output.mkdir(parents=True,exist_ok=True)
    mesh.export(a.output/'tracking_mesh_metric.obj')
    b=a.run/'vggt-omega-512-allframes-v1'
    d=np.load(b/'vggt_omega_predictions_compact.npz')
    aligned=np.load(b/'table-aligned/vggt_omega_table_aligned.npz')
    depth=d['depth'][...,0].astype(np.float32)*float(aligned['omega_to_table_scale'])
    images=np.clip(d['processed_images'].transpose(0,2,3,1)*255,0,255).astype(np.uint8)
    intrinsics=d['intrinsic_processed_pixels']
    n=min(a.limit,len(images)); h,w=images.shape[1:3]
    model=FoundationPose(model_pts=mesh.vertices,model_normals=mesh.vertex_normals,mesh=mesh,scorer=ScorePredictor(),refiner=PoseRefinePredictor(),glctx=dr.RasterizeCudaContext(),debug=0,debug_dir=str(a.output/'debug'))
    # NumPy scalar diameters promote crop offsets to float64 in newer torch.
    model.diameter=float(model.diameter)
    import logging
    logging.getLogger().setLevel(logging.WARNING)
    mesh_tensors=make_mesh_tensors(mesh)
    poses=[];world=[];valid=[];errors=[];ious=[];depth_errors=[];start=time.monotonic()
    order=list(range(n)) if a.sam_anchor_frame is None else list(range(a.sam_anchor_frame,-1,-1))+list(range(a.sam_anchor_frame+1,n))
    anchor_centered=None
    for step,i in enumerate(order):
        m=cv2.imread(str(a.run/'reconstruction/sam2'/a.object/'masks'/f'{i:04d}.png'),0)
        mask=np.zeros((h,w),np.uint8) if m is None else (cv2.resize(m,(w,h),interpolation=cv2.INTER_NEAREST)>127).astype(np.uint8)
        ok=bool(mask.sum()>60)
        try:
            if not ok: raise RuntimeError('insufficient observed mask')
            if a.sam_anchor_frame is not None and (step==0 or i==a.sam_anchor_frame+1):
                if step==0:
                    model.pose_last=torch.as_tensor(sam_pose,device='cuda',dtype=torch.float32)@torch.linalg.inv(model.get_tf_to_centered_mesh())
                else:
                    model.pose_last=anchor_centered.clone()
                T=model.track_one(rgb=images[i],depth=depth[i],K=intrinsics[i],iteration=3)
                if step==0: anchor_centered=model.pose_last.clone()
            elif not poses or not valid[-1]:
                T=model.register(K=intrinsics[i],rgb=images[i],depth=depth[i],ob_mask=mask,iteration=5)
            else:
                T=model.track_one(rgb=images[i],depth=depth[i],K=intrinsics[i],iteration=2)
            if not np.all(np.isfinite(T)): raise RuntimeError('nonfinite pose')
        except Exception as e:
            ok=False; errors.append({'frame':i,'error':str(e)})
            if step==0:
                (a.output/'initialization_error.json').write_text(json.dumps(errors[-1],indent=2))
                raise RuntimeError('Initial pose registration failed; refusing an empty trajectory') from e
            T=poses[-1].copy() if poses else np.full((4,4),np.nan)
        iou=float('nan');depth_error=float('nan')
        if ok:
            with torch.no_grad():
                _,rd,_=nvdiffrast_render(K=intrinsics[i],H=h,W=w,ob_in_cams=torch.as_tensor(T[None],device='cuda',dtype=torch.float32),mesh_tensors=mesh_tensors,glctx=model.glctx)
            rd=rd[0].cpu().numpy(); pred=rd>0; obs=mask>0
            iou=float(np.logical_and(pred,obs).sum()/max(np.logical_or(pred,obs).sum(),1))
            overlap=pred&obs&(depth[i]>0)
            if overlap.any(): depth_error=float(np.median(np.abs(rd[overlap]-depth[i][overlap])))
        ious.append(iou);depth_errors.append(depth_error)
        C=np.eye(4);C[:3,:3]=aligned['R_table_world_to_camera_m'][i];C[:3,3]=aligned['t_table_world_to_camera_m'][i]
        poses.append(T);world.append(np.linalg.inv(C)@T);valid.append(ok)
        if i%30==0:
            print('frame',i,'valid',ok,'IoU',iou,'elapsed',time.monotonic()-start,flush=True)
            np.savez_compressed(a.output/'foundationpose_tracks_partial.npz',camera_from_object=np.asarray(poses),world_from_object=np.asarray(world),valid=np.asarray(valid),silhouette_iou=ious,predicted_depth_residual_m=depth_errors,frame_index=np.asarray(order[:step+1]),fps=30.)
    sort=np.argsort(order)
    np.savez_compressed(a.output/'foundationpose_tracks.npz',camera_from_object=np.asarray(poses)[sort],world_from_object=np.asarray(world)[sort],valid=np.asarray(valid)[sort],silhouette_iou=np.asarray(ious)[sort],predicted_depth_residual_m=np.asarray(depth_errors)[sort],frame_index=np.arange(n),fps=30.)
    report={'frames':n,'valid_estimates':sum(valid),'errors':errors,'elapsed_s':time.monotonic()-start,'mesh_source':str(a.mesh_dir),'sam_anchor_frame':a.sam_anchor_frame,'status':'raw model estimates; held poses on failures are explicitly invalid; no physics claim'}
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
