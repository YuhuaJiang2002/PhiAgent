"""Fixed anthropometry and contact stabilization, independent of camera or object names.

NumPy/SciPy are optional stage dependencies. Coordinates: right-handed scene_world
metres, +z up; transforms are column-vector world_from_object matrices.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

from .contracts import validate_scene


def unit(value, label):
    length = np.linalg.norm(value, axis=-1, keepdims=True)
    if np.any(length < 1e-9) or not np.isfinite(value).all():
        raise ValueError(f'Degenerate {label}; review calibration/pose, do not extend limbs')
    return value / length


def build_rig(wrists, backward_axes, scene, fps):
    validate_scene(scene)
    actor = scene['actor']
    root = np.asarray(actor['root_m'], dtype=float)
    sides = sorted(wrists)
    if not sides or set(sides)-{'left','right'}:
        raise ValueError('Expected observed left and/or right wrists')
    n = len(wrists[sides[0]])
    for side in sides:
        if np.shape(wrists[side]) != (n,3) or np.shape(backward_axes[side]) != (n,3):
            raise ValueError('Mismatched wrist/axis timelines')
    target = np.mean([wrists[s] for s in sides], axis=0)
    heading = actor['heading_rad']
    # Heading 0 is local +y. It comes from ego/world evidence, never render eye.
    desired = np.arctan2(-(target[:,0]-root[0]),target[:,1]-root[1])
    relative = np.angle(np.exp(1j*(desired-heading)))
    limit = np.deg2rad(actor.get('yaw_limit_deg',20.))
    yaw = heading + gaussian_filter1d(np.clip(relative,-limit,limit),max(.4*fps,.01),mode='nearest')
    rotation = Rotation.from_euler('z',yaw).as_matrix()
    result = {'root':np.broadcast_to(root,(n,3)).copy(),'yaw':yaw,'rotation':rotation}
    upper, fore = actor['upper_arm_m'],actor['forearm_m']
    report = {'frame':'scene_world','units':'m','frames':n,'fps':fps,
              'actor_origin_authority':actor['origin_authority'],
              'upper_arm_m':upper,'forearm_m':fore,'root_translation_range_m':[0.,0.,0.],
              'roll_pitch_rad':[0.,0.], 'yaw_range_deg':np.rad2deg([yaw.min(),yaw.max()]).tolist()}
    for side in sides:
        offset = np.array([(-1 if side=='left' else 1)*actor['shoulder_width_m']/2,
                           actor.get('shoulder_forward_m',.035),0.])
        shoulder = root+np.einsum('nij,j->ni',rotation,offset)
        wrist = np.asarray(wrists[side],float)
        distance = np.linalg.norm(wrist-shoulder,axis=1)
        invalid=(distance>=upper+fore-1e-5)|(distance<=abs(upper-fore)+1e-5)
        if np.any(invalid):
            raise ValueError(f'unreachable_wrist {side} frames {np.where(invalid)[0].tolist()}; '
                             'correct source scale/origin/tracks; never lengthen arms or move camera to hide it')
        axis=unit(wrist-shoulder,'reach axis')
        along=(upper**2-fore**2+distance**2)/(2*distance)
        center=shoulder+along[:,None]*axis
        preferred=wrist+fore*unit(np.asarray(backward_axes[side]),'wrist backward axis')-center
        pole=unit(preferred-np.sum(preferred*axis,axis=1)[:,None]*axis,'elbow pole')
        pole=gaussian_filter1d(pole,max(fps/15,.01),axis=0,mode='nearest')
        pole=unit(pole-np.sum(pole*axis,axis=1)[:,None]*axis,'smoothed elbow pole')
        max_step=np.deg2rad(actor.get('max_elbow_pole_speed_deg_s',90.))/fps
        for i in range(1,n):
            previous=unit(pole[i-1]-np.dot(pole[i-1],axis[i])*axis[i],'transported elbow pole')
            turn=np.arctan2(np.dot(axis[i],np.cross(previous,pole[i])),np.dot(previous,pole[i]))
            turn=np.clip(turn,-max_step,max_step)
            pole[i]=previous*np.cos(turn)+np.cross(axis[i],previous)*np.sin(turn)
        elbow=center+np.sqrt(np.maximum(upper**2-along**2,0))[:,None]*pole
        result.update({f'{side}_shoulder':shoulder,f'{side}_elbow':elbow,f'{side}_wrist':wrist})
        report[side]={'max_upper_length_error_m':float(np.abs(np.linalg.norm(elbow-shoulder,axis=1)-upper).max()),
                      'max_forearm_length_error_m':float(np.abs(np.linalg.norm(elbow-wrist,axis=1)-fore).max()),
                      'max_elbow_step_m':float(np.linalg.norm(np.diff(elbow,axis=0),axis=1).max()) if n>1 else 0.}
    return result,report


def stabilize(hands, poses, contacts, fps, orientation_modes=None, smoothing_s=.05, ramp_s=.20,
              max_correction_m=.06):
    """Object-relative medoid grip. No object type heuristic or invented grasp interval.

    Contacts: object_id, side, start_frame, end_frame, confidence, evidence.
    World-locked modes must already be justified in the validated scene contract.
    Unlike the historical hull offset, no universal 1mm penetration target is used.
    """
    n = len(next(iter(hands.values())))
    if any(len(v)!=n for v in [*hands.values(),*poses.values()]):
        raise ValueError('Hand/object frame counts differ')
    if not all(np.isfinite(v).all() for v in [*hands.values(),*poses.values()]):
        raise ValueError('Nonfinite motion')
    filtered={k:gaussian_filter1d(v,max(smoothing_s*fps,.01),axis=0,mode='nearest') for k,v in hands.items()}
    smooth={}
    for key,raw in poses.items():
        raw=np.asarray(raw,float)
        if raw.shape!=(n,4,4) or not np.allclose(raw[:,3,:],[0,0,0,1]):
            raise ValueError('Expected homogeneous world_from_object matrices')
        if not np.allclose(raw[:,:3,:3].transpose(0,2,1)@raw[:,:3,:3],np.eye(3),atol=1e-4):
            raise ValueError('Nonrigid object rotation')
        pose=raw.copy()
        pose[:,:3,3]=gaussian_filter1d(raw[:,:3,3],max(smoothing_s*fps,.01),axis=0,mode='nearest')
        quat=Rotation.from_matrix(raw[:,:3,:3]).as_quat()
        for i in range(1,n):
            if quat[i]@quat[i-1]<0:
                quat[i]*=-1
        quat=unit(gaussian_filter1d(quat,max(smoothing_s*fps,.01),axis=0,mode='nearest'),'rotation quaternion')
        pose[:,:3,:3]=Rotation.from_quat(quat).as_matrix()
        smooth[key]=pose
    occupancy={side:np.zeros(n,dtype=bool) for side in hands}
    report=[]
    for contact in contacts:
        key,side=contact['object_id'],contact['side']
        lo,hi=contact['start_frame'],contact['end_frame']
        if key not in poses or side not in hands or not 0<=lo<=hi<n:
            raise ValueError('Invalid contact interval')
        if contact.get('confidence',0)<.7 or not contact.get('evidence'):
            raise ValueError('Contact stabilization needs reviewed source evidence')
        if occupancy[side][lo:hi+1].any():
            raise ValueError('Overlapping rigid grasps on one hand need a dedicated multi-contact adapter')
        occupancy[side][lo:hi+1]=True
        raw=poses[key]; hand=hands[side]
        local=np.einsum('nvi,nij->nvj',hand-raw[:,None,:3,3],raw[:,:3,:3])
        world_locked=(orientation_modes or {}).get(key,'object')=='world_locked'
        selection=hand-raw[:,None,:3,3] if world_locked else local
        block=selection[lo:hi+1]
        medoid=lo+int(np.argmin(np.linalg.norm(block-np.median(block,axis=0),axis=2).mean(1)))
        rotation=np.broadcast_to(raw[medoid,:3,:3],(n,3,3)) if world_locked else smooth[key][:,:3,:3]
        target=local[medoid][None]@rotation.transpose(0,2,1)+smooth[key][:,None,:3,3]
        w=np.zeros(n); w[lo:hi+1]=1
        ramp=min(max(1,round(ramp_s*fps)),max(1,(hi-lo+1)//2))
        for j in range(ramp):
            value=.5-.5*np.cos(np.pi*(j+1)/(ramp+1))
            if lo>0:w[lo+j]=min(w[lo+j],value)
            if hi<n-1:w[hi-j]=min(w[hi-j],value)
        candidate=filtered[side]*(1-w[:,None,None])+target*w[:,None,None]
        correction=float(np.linalg.norm(candidate-hand,axis=2).max())
        if correction>max_correction_m:
            raise ValueError(f'Contact correction {correction:.4f}m exceeds budget; inspect geometry/tracking')
        filtered[side]=candidate
        report.append({**contact,'medoid_frame':medoid,'max_correction_m':correction})
    return filtered,smooth,{'contacts':report,'smoothing_s':smoothing_s,'ramp_s':ramp_s,
                            'claim':'visual kinematic stabilization, not force closure or mesh collision validation'}


def audit_rig(rig,scene,frames):
    """Recompute hard gates from arrays instead of trusting an adapter's pass boolean."""
    actor=validate_scene(scene)['actor']
    root=np.asarray(rig['root'])
    rotation=np.asarray(rig['rotation'])
    if root.shape!=(frames,3) or not np.allclose(root,actor['root_m'],atol=1e-7):
        raise ValueError('Rig actor origin is not the source-derived stationary root')
    if rotation.shape!=(frames,3,3) or not np.isfinite(rotation).all():raise ValueError('Invalid torso rotations')
    if not np.allclose(rotation.transpose(0,2,1)@rotation,np.eye(3),atol=1e-6) or not np.allclose(np.linalg.det(rotation),1,atol=1e-6):
        raise ValueError('Nonrigid torso transform')
    if not np.allclose(rotation[:,:,2],[0,0,1],atol=1e-6):raise ValueError('Stationary torso must be yaw-only')
    observed=[]
    for side in ['left','right']:
        if side+'_wrist' not in rig:continue
        observed.append(side)
        shoulder,elbow,wrist=[np.asarray(rig[side+'_'+joint]) for joint in ['shoulder','elbow','wrist']]
        if any(v.shape!=(frames,3) or not np.isfinite(v).all() for v in [shoulder,elbow,wrist]):
            raise ValueError('Invalid limb arrays')
        offset=np.array([(-1 if side=='left' else 1)*actor['shoulder_width_m']/2,actor.get('shoulder_forward_m',.035),0.])
        if not np.allclose(shoulder,root+np.einsum('nij,j->ni',rotation,offset),atol=1e-6):
            raise ValueError('Shoulders are not attached to the same rigid torso')
        for delta,length in [(elbow-shoulder,actor['upper_arm_m']),(wrist-elbow,actor['forearm_m'])]:
            if not np.allclose(np.linalg.norm(delta,axis=1),length,atol=1e-5):raise ValueError('Limb length changed')
    if not observed:raise ValueError('No observed arms in rig')
    return {'hard_gates_passed':True,'observed_sides':observed,'frames':frames}
