"""Generic scene-world mesh replay, derived from the accepted v35 z-buffer renderer.

Requires the external FoundationPose rendering utilities and nvdiffrast runtime.
The scene supplies camera, objects and background; no table edge or object IDs
are encoded here. This is a kinematic visual replay, not contact dynamics.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from .contracts import read_json, validate_scene, write_json
from .media import probe, same_clock
from .stages import load_motion


def render_bundle(context,out):
    if not os.environ.get('CUDA_VISIBLE_DEVICES') or not context.get('gpu_selection'):
        raise RuntimeError('GPU renderer requires inventoried launcher selection')
    settings=context.get('settings',{}).get('render',{})
    dependency=Path(settings['foundationpose_root']).resolve()
    if not (dependency/'Utils.py').is_file():raise FileNotFoundError('External FoundationPose Utils.py')
    sys.path.insert(0,str(dependency))
    import cv2
    import numpy as np
    import torch
    import trimesh
    import nvdiffrast.torch as dr
    from Utils import make_mesh_tensors,nvdiffrast_render
    previous=context['artifacts']['stabilize']
    scene=validate_scene(read_json(previous['scene']))
    events=read_json(previous['events'])
    data,hands,poses=load_motion(previous['motion'],scene,events)
    rig=np.load(previous['rig'],allow_pickle=False)
    camera=scene['render_camera']
    width,height=camera.get('width',1024),camera.get('height',768)
    if width%2 or height%2 or width<=0 or height<=0:raise ValueError('Expected positive even render dimensions')
    eye,target=np.asarray(camera['eye_m']),np.asarray(camera['target_m'])
    forward=target-eye;forward/=np.linalg.norm(forward)
    right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right)
    down=np.cross(forward,right)
    rotation=np.stack([right,down,forward]);translation=-rotation@eye
    camera_from_world=np.eye(4);camera_from_world[:3,:3]=rotation;camera_from_world[:3,3]=translation
    focal=height/(2*np.tan(np.deg2rad(camera.get('fov_y_deg',55))/2))
    K=np.array([[focal,0,width/2],[0,focal,height/2],[0,0,1.]])
    ctx=dr.RasterizeCudaContext()
    def tensors(mesh):return make_mesh_tensors(mesh)
    def draw(mt,transform):
        with torch.no_grad():
            rgb,z,_=nvdiffrast_render(K=K,H=height,W=width,
                ob_in_cams=torch.as_tensor(transform[None],device='cuda',dtype=torch.float32),
                mesh_tensors=mt,glctx=ctx,use_light=True,light_dir=np.array([.3,.5,1.]),w_ambient=.8,w_diffuse=.2)
        return (rgb[0].cpu().numpy()*255).clip(0,255).astype(np.uint8),z[0].cpu().numpy()
    def color(mesh,rgb):
        mesh.visual.vertex_colors=np.tile([*rgb,255],(len(mesh.vertices),1));return mesh
    skin=scene['actor'].get('skin_rgb',[209,159,128]);shirt=scene['actor'].get('shirt_rgb',[69,92,118])
    def segment(a,b,radius,rgb):
        return color(trimesh.creation.cylinder(radius=radius,segment=[a,b],sections=20),rgb)
    static=[]
    for item in scene.get('static_meshes',[]):
        mesh=trimesh.load(item['path'],force='mesh')
        mesh.apply_transform(np.asarray(item.get('world_from_mesh',np.eye(4))))
        static.append(mesh)
    for item in scene.get('primitives',[]):
        if item['type']!='box':raise ValueError('Only explicit box primitives currently supported')
        mesh=trimesh.creation.box(extents=item['size_m']);mesh.apply_translation(item['center_m'])
        static.append(color(mesh,item['rgb']))
    static_rgb=np.full((height,width,3),235,np.uint8)
    static_z=np.full((height,width),np.inf,np.float32)
    for mesh in static:
        rgb,z=draw(tensors(mesh),camera_from_world);mask=(z>0)&(z<static_z)
        static_rgb[mask]=rgb[mask];static_z[mask]=z[mask]
    cv2.imwrite(str(out/'static_scene.png'),static_rgb[...,::-1])
    objects={o['id']:tensors(trimesh.load(o['mesh'],force='mesh')) for o in scene['objects']}
    # One torso shared by both shoulders. Shape stays constant; only source-root yaw moves.
    theta=np.linspace(0,2*np.pi,32,endpoint=False)
    half=scene['actor']['shoulder_width_m']/2
    torso_vertices=np.concatenate([np.stack([rx*np.cos(theta),ry*np.sin(theta),np.full(32,z)],axis=1)
        for z,rx,ry in [(-.48,half*.85,.105),(-.20,half*1.05,.13),(.015,half*1.025,.12)]])
    torso_faces=[]
    for ring in range(2):
        for j in range(32):
            k=(j+1)%32;a=ring*32+j;b=ring*32+k;c=(ring+1)*32+k;d=(ring+1)*32+j
            torso_faces.extend([[a,b,c],[a,c,d]])
    def writer(path,pixel_format='rgb24',codec='libx264'):
        return subprocess.Popen(['ffmpeg','-v','error','-n','-f','rawvideo','-pix_fmt',pixel_format,
            '-s',f'{width}x{height}','-r',str(events['fps']),'-i','-','-an','-c:v',codec,
            *(['-crf','17','-pix_fmt','yuv420p','-movflags','+faststart'] if codec=='libx264' else []),str(path)],stdin=subprocess.PIPE)
    video=writer(out/'sim.mp4');mask_video=writer(out/'foreground.mkv','gray','ffv1')
    try:
        for frame in range(events['frames']):
            rgb=static_rgb.copy();z=static_z.copy();foreground=np.zeros((height,width),np.uint8)
            dynamic=[]
            torso=trimesh.Trimesh(vertices=torso_vertices@rig['rotation'][frame].T+rig['root'][frame],faces=torso_faces,process=False)
            dynamic.append((tensors(color(torso,shirt)),camera_from_world))
            for side,vertices in hands.items():
                hand=color(trimesh.Trimesh(vertices=vertices[frame],faces=data[side+'_faces'],process=False),skin)
                shoulder,elbow,wrist=[rig[side+'_'+joint][frame] for joint in ['shoulder','elbow','wrist']]
                sleeve=shoulder+.42*(elbow-shoulder)
                for mesh in [segment(shoulder,sleeve,.055,shirt),segment(sleeve,elbow,.038,skin),segment(elbow,wrist,.027,skin),hand]:
                    dynamic.append((tensors(mesh),camera_from_world))
            dynamic.extend((objects[key],camera_from_world@pose[frame]) for key,pose in poses.items())
            for mt,transform in dynamic:
                rendered,depth=draw(mt,transform);mask=(depth>0)&(depth<z)
                rgb[mask]=rendered[mask];z[mask]=depth[mask];foreground[mask]=255
            video.stdin.write(rgb.tobytes());mask_video.stdin.write(foreground.tobytes())
        for process in [video,mask_video]:
            process.stdin.close()
            if process.wait()!=0:raise RuntimeError('Video encoding failure')
    finally:
        for process in [video,mask_video]:
            if process.poll() is None:process.terminate();process.wait()
    if not same_clock(probe(out/'sim.mp4'),probe(previous['ego'])):
        raise ValueError('Rendered SIM does not preserve the reviewed EGO clock')
    write_json(out/'render_report.json',{'scene_frame':'scene_world','camera_from_world':camera_from_world.tolist(),
        'frames':events['frames'],'fps':events['fps'],'claim':'kinematic mesh replay, not physically validated simulation'})
    return {**previous,'sim':str(out/'sim.mp4'),'foreground':str(out/'foreground.mkv')}
