"""Propose motion events from calibrated tracks and observed contact intervals.

These are proposals for pre-DiT visual review, not contact ground truth. No RGB
appearance/color threshold or scene-specific object name is used.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d


def intervals(mask,min_frames=1):
    edges=np.diff(np.r_[False,np.asarray(mask,bool),False].astype(int))
    return [(int(lo),int(hi-1)) for lo,hi in zip(np.where(edges==1)[0],np.where(edges==-1)[0]) if hi-lo>=min_frames]


def propose_events(poses,contacts,fps,*,speed_threshold_m_s=.015,min_motion_s=.12):
    events=[]
    for key,pose in poses.items():
        pose=np.asarray(pose)
        if pose.ndim!=3 or pose.shape[1:]!=(4,4) or not np.isfinite(pose).all():raise ValueError('Invalid world poses')
        n=len(pose)
        observed=[c for c in contacts if c['object_id']==key]
        held=np.zeros(n,bool)
        for contact in observed:
            if contact.get('confidence',0)<.7:continue
            lo,hi=contact['start_frame'],contact['end_frame']
            if not 0<=lo<=hi<n:raise ValueError('Invalid contact interval')
            held[lo:hi+1]=True
        position=gaussian_filter1d(pose[:,:3,3],max(fps*.05,.01),axis=0,mode='nearest')
        speed=np.linalg.norm(np.gradient(position,axis=0)*fps,axis=1)
        for segment,(lo,hi) in enumerate(intervals(held)):
            def event(label,frame):
                events.append({'id':f'{key}.{segment}.{label}','object_id':key,'action':label,
                    'time_s':frame/fps,'confidence':.7,
                    'evidence':'Calibrated world motion plus observed contact interval; requires visual confirmation',
                    'annotation':'automatic_proposal'})
            event('already_held' if lo==0 else 'grasp',lo)
            for motion,(start,end) in enumerate(intervals((speed>speed_threshold_m_s)&(np.arange(n)>=lo)&(np.arange(n)<=hi),max(1,round(min_motion_s*fps)))):
                event(f'carry_{motion}_start',start)
                if end<n-1:event(f'carry_{motion}_stop',end)
            # Never invent release when a held interval reaches the final frame.
            if hi<n-1:event('release',hi+1)
    return sorted(events,key=lambda event:(event['time_s'],event['id']))
