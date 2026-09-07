"""Fixed-root yaw-only torso and constant-length two-link arm control rig."""
import json
import numpy as np
from scipy.ndimage import gaussian_filter1d

def build_rig(hands, mano, out):
    root=np.array([.025,-.52,.27])
    wrists={};backs={}
    for side,v in hands.items():
        faces=mano[side+'_faces']
        edges=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)
        unique,count=np.unique(edges,axis=0,return_counts=True)
        boundary=np.unique(unique[count==1])
        wrists[side]=v[:,boundary].mean(1)
        b=wrists[side]-v.mean(1)
        backs[side]=b/np.linalg.norm(b,axis=1,keepdims=True)
    target=(wrists['left']+wrists['right'])/2
    yaw=np.arctan2(-(target[:,0]-root[0]),target[:,1]-root[1])
    yaw=gaussian_filter1d(np.clip(yaw,-np.deg2rad(20),np.deg2rad(20)),12,mode='nearest')
    n=len(yaw);rot=np.zeros((n,3,3));rot[:,0,0]=rot[:,1,1]=np.cos(yaw)
    rot[:,1,0]=np.sin(yaw);rot[:,0,1]=-np.sin(yaw);rot[:,2,2]=1
    result={'root':root,'yaw':yaw,'rotation':rot}
    report={'root_xyz_m':root.tolist(),'root_translation_range_m':[0,0,0], 'roll_pitch_radians':[0,0], 'yaw_range_degrees':np.rad2deg([yaw.min(),yaw.max()]).tolist(), 'max_yaw_speed_degrees_per_second':float(np.rad2deg(np.abs(np.diff(yaw))).max()*30), 'upper_arm_m':.36,'forearm_m':.32, 'method':'Two-sphere elbow IK; pole chosen nearest existing wrist backward axis. Shared rigid torso; no hand-driven torso translation.'}
    for side in wrists:
        shoulder=root+np.einsum('nij,j->ni',rot,np.array([-.18 if side=='left' else .18,.035,0]))
        wrist=wrists[side];axis=wrist-shoulder;distance=np.linalg.norm(axis,axis=1)
        if distance.max()>=.68 or distance.min()<=.04:
            raise ValueError(f'Unreachable {side} wrist: {distance.min()}..{distance.max()}')
        axis/=distance[:,None]
        a=(.36**2-.32**2+distance**2)/(2*distance)
        center=shoulder+a[:,None]*axis
        preferred=wrist+.32*backs[side]-center
        pole=preferred-(preferred*axis).sum(1)[:,None]*axis
        norm=np.linalg.norm(pole,axis=1)
        if norm.min()<1e-5:raise ValueError('Degenerate elbow pole')
        pole/=norm[:,None]
        pole=gaussian_filter1d(pole,2,axis=0,mode='nearest')
        pole-=np.sum(pole*axis,axis=1)[:,None]*axis
        pole/=np.linalg.norm(pole,axis=1,keepdims=True)
        # Parallel-transport the elbow bend direction as the reach axis moves.
        # A wrist rotation must not flip the whole elbow around the arm axis.
        max_step=np.deg2rad(3)
        for i in range(1,n):
            previous=pole[i-1]-np.dot(pole[i-1],axis[i])*axis[i]
            previous/=np.linalg.norm(previous)
            turn=np.arctan2(np.dot(axis[i],np.cross(previous,pole[i])),np.dot(previous,pole[i]))
            turn=np.clip(turn,-max_step,max_step)
            pole[i]=previous*np.cos(turn)+np.cross(axis[i],previous)*np.sin(turn)
        elbow=center+np.sqrt(.36**2-a**2)[:,None]*pole
        fore=(elbow-wrist)/.32
        angle=np.rad2deg(np.arccos(np.clip(np.sum(fore*backs[side],axis=1),-1,1)))
        result[side+'_shoulder']=shoulder;result[side+'_elbow']=elbow;result[side+'_wrist']=wrist
        report['elbow_bend_direction_max_relative_speed_degrees_per_second']=90
        report[side]={'shoulder_y_range_m':[float(shoulder[:,1].min()),float(shoulder[:,1].max())], 'shoulder_outside_long_edge_all_frames':bool((shoulder[:,1]<-.36).all()),'reach_distance_range_m':[float(distance.min()),float(distance.max())], 'max_upper_length_error_m':float(np.abs(np.linalg.norm(elbow-shoulder,axis=1)-.36).max()),'max_forearm_length_error_m':float(np.abs(np.linalg.norm(elbow-wrist,axis=1)-.32).max()),'forearm_to_hand_backward_angle_degrees':np.quantile(angle,[0,.5,.95,1]).tolist()}
    np.savez_compressed(out/'body_rig.npz',**result)
    (out/'body_rig_report.json').write_text(json.dumps(report,indent=2))
    return result
