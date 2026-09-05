"""Rebuild the observed table texture in the same metric plane as the meshes."""
from runtime import require_launcher
require_launcher()

import argparse,json,warnings
from pathlib import Path
import cv2,numpy as np
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
out=a.run/'observed-table-texture-v2';out.mkdir(parents=True,exist_ok=True)
state=np.load(a.run/'fixed-third-view-v6-persistence/reconstruction/world_state_tracks.npz')
aligned=np.load(a.run/'vggt-omega-512-allframes-v1/table-aligned/vggt_omega_table_aligned.npz')
pred=np.load(a.run/'vggt-omega-512-allframes-v1/vggt_omega_predictions_compact.npz')
depths=pred['depth'][...,0].astype(np.float32)*float(aligned['omega_to_table_scale'])
frames=sorted((a.run/'input/ego_action_4s_16s/extracted_images').glob('*.jpg'))
height,width=cv2.imread(str(frames[0])).shape[:2]
plane=np.array([[-.52,-.36,-.012],[.52,-.36,-.012],[.52,.36,-.012],[-.52,.36,-.012]])
pts=plane@aligned['R_table_world_to_camera_m'][0].T+aligned['t_table_world_to_camera_m'][0]
uv=pts@aligned['intrinsic_processed_pixels'][0].T;uv=uv[:,:2]/uv[:,2:]
uv*=np.array([width/688,height/384])
tw,th=1040,720;dst=np.array([[0,0],[tw-1,0],[tw-1,th-1],[0,th-1]],np.float32)
samples=[]
for i in range(0,len(frames),6):
    rgb=cv2.imread(str(frames[i]));hsv=cv2.cvtColor(rgb,cv2.COLOR_BGR2HSV)
    b,g,r=cv2.split(rgb.astype(np.int16))
    skin=((hsv[...,0]<26)&(hsv[...,1]>25)&(hsv[...,1]<210)&(r>g+6)&(r>b+12)&(r>55))
    dyn=skin.astype(np.uint8)*255
    depth=depths[i];yy,xx=np.mgrid[:depth.shape[0],:depth.shape[1]];Ki=aligned['intrinsic_processed_pixels'][i]
    camera=np.stack([(xx-Ki[0,2])*depth/Ki[0,0],(yy-Ki[1,2])*depth/Ki[1,1],depth],axis=-1)
    world_z=((camera-aligned['t_table_world_to_camera_m'][i])@aligned['R_table_world_to_camera_m'][i])[...,2]
    plane_valid=np.isfinite(world_z)&(np.abs(world_z+.012)<.045)&(depth>0)
    plane_valid=cv2.resize(plane_valid.astype(np.uint8),(width,height),interpolation=cv2.INTER_NEAREST)>0
    dyn[~plane_valid]=255
    for key in ['alarm_clock_v2','small_cylinder_v2','tall_cylinder']:
        mask=cv2.imread(str(a.run/'reconstruction/sam2'/key/'masks'/f'{i:04d}.png'),0)
        if mask is not None: dyn|=mask
    dyn=cv2.dilate(dyn,np.ones((15,15),np.uint8))
    current=cv2.perspectiveTransform(uv.astype(np.float32)[None],state['table_homography_reference_to_frame'][i])[0]
    H=cv2.getPerspectiveTransform(current,dst)
    sample=cv2.warpPerspective(rgb,H,(tw,th)).astype(np.float32)
    valid=cv2.warpPerspective((dyn==0).astype(np.uint8),H,(tw,th),flags=cv2.INTER_NEAREST)>0
    sample[~valid]=np.nan;samples.append(sample)
with warnings.catch_warnings():
    warnings.simplefilter('ignore',RuntimeWarning);texture=np.nanmedian(np.stack(samples),axis=0)
valid=np.isfinite(texture).all(-1);texture[~valid]=232
cv2.imwrite(str(out/'table_texture.png'),np.clip(texture,0,255).astype(np.uint8));cv2.imwrite(str(out/'observed_mask.png'),valid.astype(np.uint8)*255)
sharp=np.full_like(texture,np.nan);source_index=np.full((th,tw),65535,np.uint16)
order=sorted(range(len(samples)),key=lambda j:np.isfinite(samples[j][...,0]).sum(),reverse=True)
for j in order:
    use=~np.isfinite(sharp[...,0])&np.isfinite(samples[j][...,0])
    sharp[use]=samples[j][use];source_index[use]=j*6
sharp[~np.isfinite(sharp)]=232
cv2.imwrite(str(out/'table_texture_sharp_observed.png'),np.clip(sharp,0,255).astype(np.uint8))
cv2.imwrite(str(out/'texture_source_frame_index.png'),source_index)
(out/'manifest.json').write_text(json.dumps({'method':'Metric table-plane projection + tracked planar homographies + object/skin exclusion + predicted-depth plane gate + multi-frame median','depth_plane_tolerance_m':.045,'samples':len(samples),'observed_fraction':float(valid.mean()),'unobserved_fill':'neutral gray, no generative inpainting','skin_mask':'color heuristic, not ground truth','table_bounds_m':plane.tolist()},indent=2))
print('TEXTURE_DONE',float(valid.mean()))
