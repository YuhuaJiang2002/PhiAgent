"""Built-in scene-bundle import and stabilization adapters; heavy imports stay local."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from .contracts import digest, read_json, validate_events, validate_scene, write_json


def load_motion(path,scene,events):
    import numpy as np
    data=np.load(path,allow_pickle=False)
    n=events['frames'];fps=events['fps']
    if n<2 or fps<=0:
        raise ValueError('Invalid motion clock')
    times=data['timestamps_s']
    if times.shape!=(n,) or not np.allclose(times,np.arange(n)/fps,atol=1e-5):
        raise ValueError('Motion requires a source-aligned CFR timeline; preserve VFR timestamp mapping in frontend')
    hands={s:data[s+'_hand_world_m'] for s in ['left','right'] if s+'_hand_world_m' in data}
    if not hands:
        raise ValueError('No observed hand geometry')
    poses={o['id']:data[o['id']+'__world_from_object'] for o in scene['objects']}
    for side,vertices in hands.items():
        if vertices.ndim!=3 or vertices.shape[0]!=n or vertices.shape[-1]!=3 or not np.isfinite(vertices).all():
            raise ValueError('Invalid hand geometry')
        faces=data[side+'_faces']
        if faces.ndim!=2 or faces.shape[1]!=3 or faces.min()<0 or faces.max()>=vertices.shape[1]:
            raise ValueError('Invalid hand topology')
    for pose in poses.values():
        if pose.shape!=(n,4,4) or not np.isfinite(pose).all():
            raise ValueError('Invalid object pose timeline')
    return data,hands,poses


def import_bundle(context,out):
    """Reuse an explicitly supplied reconstruction, not pretend it came from a new video."""
    bundle=Path(context['clip']['bundle']).resolve()
    scene=validate_scene(read_json(bundle/'scene.json'))
    if scene['source_sha256']!=context['source_sha256']:
        raise ValueError('Reconstruction belongs to another ego clip')
    events=read_json(bundle/'events.json')
    if 'events' not in events:
        from .event_detection import propose_events
        _,_,poses=load_motion(bundle/'motion.npz',scene,events)
        events['events']=propose_events(poses,events.get('contacts',[]),events['fps'])
    validate_events(events['events'],events['frames']/events['fps'])
    if events.get('source_sha256')!=scene['source_sha256']:
        raise ValueError('Events belong to another source')
    load_motion(bundle/'motion.npz',scene,events)
    # Resolve asset paths in the bundle's own namespace, never the batch cwd.
    for obj in scene['objects']:
        obj['mesh']=str((bundle/obj['mesh']).resolve())
        if not Path(obj['mesh']).is_file():raise FileNotFoundError(obj['mesh'])
    for mesh in scene.get('static_meshes',[]):
        mesh['path']=str((bundle/mesh['path']).resolve())
        if not Path(mesh['path']).is_file():raise FileNotFoundError(mesh['path'])
    write_json(out/'scene.json',scene)
    write_json(out/'events.json',events)
    shutil.copyfile(bundle/'motion.npz',out/'motion.npz')
    ego=bundle/'ego.mp4'
    if not ego.is_file():raise FileNotFoundError('Bundle must contain the reviewed ego.mp4 export')
    return {'scene':str(out/'scene.json'),'events':str(out/'events.json'),'motion':str(out/'motion.npz'),
            'ego':str(ego),'source':context['source']}


def stabilize_bundle(context,out):
    import numpy as np
    from .geometry import build_rig, stabilize
    previous=context['artifacts']['reconstruct']
    scene=validate_scene(read_json(previous['scene']))
    events=read_json(previous['events'])
    data,hands,poses=load_motion(previous['motion'],scene,events)
    settings=context.get('settings',{}).get('stabilize',{})
    if 'elbow_pole_speed_deg_s' in settings:
        scene['actor']['max_elbow_pole_speed_deg_s']=settings['elbow_pole_speed_deg_s']
        write_json(out/'scene.json',scene)
        previous={**previous,'scene':str(out/'scene.json')}
    hands,poses,contact_report=stabilize(hands,poses,events.get('contacts',[]),events['fps'],
        {o['id']:o.get('orientation_mode','object') for o in scene['objects']},
        smoothing_s=settings.get('smoothing_s',.05),ramp_s=settings.get('ramp_s',.20),
        max_correction_m=settings.get('max_correction_m',.06))
    wrists={};backwards={}
    for side,vertices in hands.items():
        faces=data[side+'_faces']
        edges=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)
        unique,count=np.unique(edges,axis=0,return_counts=True)
        boundary=np.unique(unique[count==1])
        if not len(boundary):raise ValueError('Hand mesh has no wrist boundary; frontend must export an open wrist mesh')
        wrists[side]=vertices[:,boundary].mean(1)
        backwards[side]=wrists[side]-vertices.mean(1)
    rig,rig_report=build_rig(wrists,backwards,scene,events['fps'])
    np.savez_compressed(out/'motion.npz',timestamps_s=data['timestamps_s'],
        **{s+'_hand_world_m':v for s,v in hands.items()},
        **{s+'_faces':data[s+'_faces'] for s in hands},
        **{key+'__world_from_object':pose for key,pose in poses.items()})
    np.savez_compressed(out/'rig.npz',**rig)
    write_json(out/'geometry_report.json',{'rig':rig_report,'stabilization':contact_report})
    return {**previous,'motion':str(out/'motion.npz'),'rig':str(out/'rig.npz'),
            'geometry_report':str(out/'geometry_report.json')}


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['import_bundle','stabilize','render'])
    parser.add_argument('--context',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    context=read_json(args.context)
    out=args.output.resolve()
    if not out.is_dir():raise ValueError('Use the batch launcher to create the stage output')
    if args.stage=='import_bundle':result=import_bundle(context,out)
    elif args.stage=='stabilize':result=stabilize_bundle(context,out)
    else:
        from .mesh_renderer import render_bundle
        result=render_bundle(context,out)
    write_json(out/'result.json',result)


if __name__=='__main__':main()
