"""Versioned visual stabilization; approximate hull contact, not a physics solve."""
import json
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

def stabilize(state, objects, out):
    hands = {s: gaussian_filter1d(state[f'{s}_mano_vertices_table_world'], 1.5, axis=0, mode='nearest') for s in ['left', 'right']}
    poses = {}
    report = {'method': '50ms Gaussian smoothing; object-relative medoid grip with 200ms cosine transition; bounded rigid radial hull fit', 'limitations': 'Estimated geometry; hull distances are not exact mesh collision or force validation.', 'grips': []}
    spans = {'alarm_clock': (0, 84), 'small_cylinder': (114, 233), 'tall_cylinder': (270, 359)}
    for key, (mesh, _, data) in objects.items():
        raw = data['world_from_object']
        pose = raw.copy()
        pose[:, :3, 3] = gaussian_filter1d(raw[:, :3, 3], 1.5, axis=0, mode='nearest')
        quat = Rotation.from_matrix(raw[:, :3, :3]).as_quat()
        for i in range(1, len(quat)):
            if quat[i] @ quat[i-1] < 0: quat[i] *= -1
        quat = gaussian_filter1d(quat, 1.5, axis=0, mode='nearest')
        quat /= np.linalg.norm(quat, axis=1, keepdims=True)
        pose[:, :3, :3] = Rotation.from_quat(quat).as_matrix()
        poses[key] = pose
        hull = ConvexHull(mesh.vertices)
        A, b = hull.equations[:, :3], hull.equations[:, 3]
        lo, hi = spans[key]
        for side in hands:
            rawhand = state[f'{side}_mano_vertices_table_world']
            local = np.einsum('nvi,nij->nvj', rawhand - raw[:, None, :3, 3], raw[:, :3, :3])
            selection = local if key == 'alarm_clock' else rawhand - raw[:, None, :3, 3]
            block = selection[lo:hi+1]
            idx = lo + int(np.argmin(np.mean(np.linalg.norm(block-np.median(block, axis=0), axis=2), axis=1)))
            template = local[idx].copy()
            direction = template.mean(0) - np.asarray(mesh.vertices).mean(0)
            direction /= max(np.linalg.norm(direction), 1e-9)
            offsets = np.linspace(-.025, .025, 101)
            distances = [np.quantile(((template+d*direction)@A.T+b).max(1), .05) for d in offsets]
            offset = float(offsets[np.argmin(np.abs(np.asarray(distances)+.001))])
            template += offset*direction
            # A textureless cylinder's axial pose is unobservable: following
            # estimated spin/axis flips rotates the arms around it spuriously.
            # These upright carrying actions constrain grip orientation in world.
            grip_rotation = pose[:, :3, :3] if key == 'alarm_clock' else np.broadcast_to(raw[idx,:3,:3], pose[:,:3,:3].shape)
            target = template[None] @ grip_rotation.transpose(0,2,1) + pose[:, None, :3, 3]
            weights = np.zeros(len(raw))
            weights[lo:hi+1] = 1.
            for j in range(6):
                w = .5-.5*np.cos(np.pi*(j+1)/7)
                if lo > 0: weights[lo+j] = w
                if hi < len(raw)-1: weights[hi-j] = w
            hands[side] = hands[side]*(1-weights[:,None,None])+target*weights[:,None,None]
            report['grips'].append({'object':key, 'side':side, 'frames':[lo,hi], 'template_frame':idx, 'rigid_contact_offset_m':offset, 'q05_hull_distance_after_m':float(np.quantile((template@A.T+b).max(1),.05)), 'max_vertex_correction_m':float(np.linalg.norm(hands[side][lo:hi+1]-rawhand[lo:hi+1],axis=2).max())})
    for side in hands:
        original = state[f'{side}_mano_vertices_table_world']
        report[side+'_centroid_acceleration_rms_m_per_frame2'] = {'before':float(np.sqrt(np.mean(np.diff(original.mean(1),n=2,axis=0)**2))), 'after':float(np.sqrt(np.mean(np.diff(hands[side].mean(1),n=2,axis=0)**2)))}
    np.savez_compressed(out/'stabilized_tracks.npz', **{s+'_hands':v for s,v in hands.items()}, **poses)
    (out/'stabilization_report.json').write_text(json.dumps(report,indent=2))
    return hands, poses
