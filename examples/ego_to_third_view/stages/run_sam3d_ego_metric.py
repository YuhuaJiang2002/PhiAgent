"""SAM3D reconstruction conditioned on the existing aligned VGGT pointmap."""
from runtime import require_launcher
require_launcher()

import argparse,json,os,time
from pathlib import Path
os.environ.setdefault('ATTN_BACKEND','sdpa')
os.environ.setdefault('SPARSE_ATTN_BACKEND','sdpa')
os.environ.setdefault('LIDRA_SKIP_INIT','true')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--dino-source',type=Path,required=True)
    p.add_argument('--object',required=True)
    p.add_argument('--frame',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    import cv2,numpy as np,torch
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    from scipy.spatial.transform import Rotation
    # Both released generator checkpoints contain the complete DINO
    # condition_embedder weights; instantiate its architecture offline.
    hub_load=torch.hub.load
    def local_hub(repo_or_dir,model,*args,**kwargs):
        if repo_or_dir=='facebookresearch/dinov2':
            kwargs.update(source='local',pretrained=False)
            repo_or_dir=str(a.dino_source)
        return hub_load(repo_or_dir,model,*args,**kwargs)
    torch.hub.load=local_hub
    from sam3d_objects.model.backbone.tdfy_dit.utils import render_utils
    from sam3d_objects.model.backbone.tdfy_dit.representations import Gaussian
    original_render=render_utils.render_frames
    def gs_render(sample,extrinsics,intrinsics,options=None,**kwargs):
        opts=dict(options or {})
        if isinstance(sample,Gaussian): opts['backend']='gsplat'
        return original_render(sample,extrinsics,intrinsics,options=opts,**kwargs)
    render_utils.render_frames=gs_render
    # Resolve all runtime imports before expensive shared-disk data reads.
    from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap
    cache=a.run/'sam3d-inputs-v1'/a.object/f'metric_{a.frame:04d}.npz'
    if cache.exists():
        with np.load(cache) as d: rgb,depth,K,mask=[d[k] for k in ['rgb','depth','K','mask']]
    else:
        b=a.run/'vggt-omega-512-allframes-v1'
        with np.load(b/'vggt_omega_predictions_compact.npz') as d:
            rgb=np.clip(d['processed_images'][a.frame].transpose(1,2,0)*255,0,255).astype(np.uint8)
            depth=d['depth'][a.frame,...,0].astype(np.float32)
            K=d['intrinsic_processed_pixels'][a.frame].astype(np.float32)
        with np.load(b/'table-aligned/vggt_omega_table_aligned.npz') as d:
            depth*=float(d['omega_to_table_scale'])
        mask=cv2.imread(str(a.run/'reconstruction/sam2'/a.object/'masks'/f'{a.frame:04d}.png'),0)
        if mask is None: raise RuntimeError('Missing observed mask')
        mask=cv2.resize(mask,(depth.shape[1],depth.shape[0]),interpolation=cv2.INTER_NEAREST)>127
        cache.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(cache,rgb=rgb,depth=depth,K=K,mask=mask)
    h,w=depth.shape
    yy,xx=np.mgrid[:h,:w]
    # PyTorch3D camera convention used by SAM3D: x left, y up, z forward.
    pointmap=np.stack([-(xx-K[0,2])*depth/K[0,0],-(yy-K[1,2])*depth/K[1,1],depth],axis=-1).astype(np.float32)
    cfg=OmegaConf.load(a.weights/'checkpoints/pipeline.yaml')
    cfg.workspace_dir=str(a.weights/'checkpoints')
    cfg.compile_model=False
    cfg.depth_model=None
    cfg.rendering_engine='pytorch3d'
    start=time.monotonic()
    pipeline=instantiate(cfg)
    if not mask.any(): raise RuntimeError('Empty observed object mask')
    result=pipeline.run(rgb,mask.astype(np.uint8)*255,seed=a.seed,pointmap=torch.from_numpy(pointmap),with_mesh_postprocess=True,with_texture_baking=True,with_layout_postprocess=False)
    a.output.mkdir(parents=True,exist_ok=True)
    result['glb'].export(str(a.output/'object.glb'))
    for key in ['gaussian','gs']:
        if key in result and hasattr(result[key],'save_ply'):
            result[key].save_ply(str(a.output/'object_gaussians.ply'))
            break
    q=result['rotation'][0].detach().cpu().numpy()
    t=result['translation'][0].detach().cpu().numpy()
    s=result['scale'][0].detach().cpu().numpy()
    F_T=np.array([[1,0,0],[0,0,-1],[0,1,0]],dtype=float)
    N=np.diag([-1.,-1.,1.])
    T=np.eye(4);T[:3,:3]=N@Rotation.from_quat(q[[1,2,3,0]]).as_matrix().T@F_T;T[:3,3]=N@t
    np.savez(a.output/'sam3d_state.npz',camera_from_object=T,object_scale=s,K=K,depth_m=depth,mask=mask)
    cv2.imwrite(str(a.output/'observed_rgb.png'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(a.output/'observed_mask.png'),mask.astype(np.uint8)*255)
    report={'object':a.object,'frame':a.frame,'seed':a.seed,'elapsed_s':time.monotonic()-start,'mesh':str(a.output/'object.glb'),'camera_from_object':T.tolist(),'object_scale':s.tolist(),'depth_source':'VGGT-Omega, table-scale aligned; predicted depth','status':'generated_mesh_and_pose_candidate; requires reprojection validation'}
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__': main()
