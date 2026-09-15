"""Export inspectable object poses and independently reproject display tracks."""
from runtime import require_launcher
require_launcher()

import os,sys,json,csv,argparse
from pathlib import Path
import numpy as np
root=Path(__import__('os').environ['PHI_EGO_ROOT']); run=Path(__import__('os').environ['PHI_EGO_RUN'])
sys.path[:0]=[str(root/'envs/sam3d-container-cu128/phiagent-native'),str(root/'third_party/FoundationPose')]
import cv2,torch,trimesh,nvdiffrast.torch as dr
from Utils import nvdiffrast_render,make_mesh_tensors
from scipy.spatial.transform import Rotation
p=argparse.ArgumentParser();p.add_argument('--tracks-root',default='foundationpose-layout-persistent-v2');p.add_argument('--output',default='trajectory-audit-v2');a=p.parse_args()
out=run/a.output;out.mkdir(exist_ok=True)
data=np.load(run/'vggt-omega-512-allframes-v1/vggt_omega_predictions_compact.npz')
images=np.clip(data['processed_images'].transpose(0,2,3,1)*255,0,255).astype(np.uint8)
K=data['intrinsic_processed_pixels'];h,w=images.shape[1:3]
context=dr.RasterizeCudaContext();assets={};summary={}
for key,mask in [('alarm_clock','alarm_clock_v2'),('small_cylinder','small_cylinder_v2'),('tall_cylinder','tall_cylinder')]:
    mesh=trimesh.load(run/'foundationpose-layout-v2'/key/'tracking_mesh_metric.obj',force='mesh')
    z=np.load(run/a.tracks_root/key/'foundationpose_tracks.npz')
    poses=z['world_from_object'];q=Rotation.from_matrix(poses[:,:3,:3]).as_quat()
    with (out/f'{key}_world_poses.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['frame','time_s','x_m','y_m','z_m','qx','qy','qz','qw','finite_estimate'])
        for i in range(len(poses)):writer.writerow([i,i/30,*poses[i,:3,3],*q[i],bool(z['valid'][i])])
    assets[key]=(make_mesh_tensors(mesh),z['camera_from_object'],mask)
    summary[key]=[]
writer=cv2.VideoWriter(str(out/'source_camera_mesh_overlay.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(w,h))
for i,rgb in enumerate(images):
    vis=rgb.copy()
    for key,(mesh,poses,maskkey) in assets.items():
        with torch.no_grad():
            color,depth,_=nvdiffrast_render(K=K[i],H=h,W=w,ob_in_cams=torch.tensor(poses[i:i+1],device='cuda',dtype=torch.float32),mesh_tensors=mesh,glctx=context)
        z=depth[0].cpu().numpy();pred=z>0
        m=cv2.imread(str(run/'reconstruction/sam2'/maskkey/'masks'/f'{i:04d}.png'),0)
        obs=cv2.resize(m,(w,h),interpolation=cv2.INTER_NEAREST)>127
        summary[key].append(float((pred&obs).sum()/max((pred|obs).sum(),1)))
        c=(color[0].cpu().numpy()*255).astype(np.uint8)
        vis[pred]=(.55*vis[pred]+.45*c[pred]).astype(np.uint8)
        contours,_=cv2.findContours(pred.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis,contours,-1,(255,128,0),1)
    cv2.putText(vis,f'Frame {i:03d} | predicted display mesh reprojection',(10,24),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
    writer.write(vis[...,::-1])
    if i in [0,120,240,330]:cv2.imwrite(str(out/f'overlay_{i:04d}.jpg'),vis[...,::-1])
writer.release()
np.savez_compressed(out/'per_frame_iou.npz',**summary)
(out/'manifest.json').write_text(json.dumps({'tracks_root':a.tracks_root,'display_reprojection_median_iou':{k:float(np.median(v)) for k,v in summary.items()},'coordinate_frame':'Estimated table world, scale set from assumed table dimensions; not calibrated ground truth.','pose_semantics':'Object 6D poses, not robot EEF/joint action. xyzw quaternion.','reviewed_prior':'Display poses include static-span anchors; raw estimates preserved in foundationpose-layout-v2.'},indent=2))
print((out/'manifest.json').read_text())
