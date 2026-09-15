"""Validation video profiles: bounded sampling, source cuts and camera candidates."""
import copy
import json
import math
from pathlib import Path

DEFAULT_PROFILE = 'configs/plenoptic/video_pipeline.json'
SAMPLING_FIELDS = {'k', 'denoising_steps', 'guidance', 'shift'}


def sampling_plan(base, cases):
    """Custom-video overrides never enter the fixed validation loss protocol."""
    overrides = [case.get('generation_parameters', {}) for case in cases]
    if any(overrides) and any(not case.get('qualitative_only') for case in cases):
        raise ValueError('Fixed validation cases cannot override sampling parameters')
    if len({json.dumps(value, sort_keys=True) for value in overrides}) > 1:
        raise ValueError('Different video sampling parameters cannot share a batch')
    value = dict(overrides[0]) if overrides else {}
    if set(value) - SAMPLING_FIELDS:
        raise ValueError('Unknown custom video sampling override')
    plan = dict(base, **value)
    if (type(plan['k']) is not int or not 1 <= plan['k'] <= 4
            or type(plan['denoising_steps']) is not int or not 1 <= plan['denoising_steps'] <= 100
            or not math.isfinite(plan['guidance']) or not 0 <= plan['guidance'] <= 6
            or not math.isfinite(plan['shift']) or not 0 < plan['shift'] <= 10):
        raise ValueError('Invalid custom video sampling parameters')
    return plan



def per_video_profile(profile, entry):
    """Allow evidence-based source-specific sampling without changing fixed validation."""
    result = dict(profile)
    result['generation'] = dict(profile.get('generation', {}), **entry.get('generation', {}))
    sampling_plan(dict(k=1, denoising_steps=35, guidance=1.5, shift=5.),
                  [dict(qualitative_only=True, generation_parameters=result['generation'])])
    scale = entry.get('camera_translation_scale', profile.get('camera_translation_scale', 1.))
    if type(scale) not in (int,float) or not math.isfinite(scale) or not 0 < scale <= 16:
        raise ValueError('Per-video camera scale must be finite and in (0,16]')
    result['camera_translation_scale'] = scale
    return result

def model_timeline(valid_frames, method='pad'):
    """Map short real clips to model frames and exactly back to their original timeline."""
    import numpy as np
    if type(valid_frames) is not int or not 1 <= valid_frames <= 81 or method not in ('pad','stretch'):
        raise ValueError('Invalid video temporal sampling')
    if method == 'pad':
        source=np.minimum(np.arange(81),valid_frames-1)
        output=np.arange(valid_frames)
    else:
        source=np.rint(np.linspace(0,valid_frames-1,81)).astype(int)
        output=np.rint(np.linspace(0,80,valid_frames)).astype(int) if valid_frames>1 else np.array([0])
    if not np.array_equal(source[output],np.arange(valid_frames)):
        raise ValueError('Model temporal map cannot recover the original frame timeline')
    return source,output


def compatible_batches(batches):
    result = []
    for batch in batches:
        keys = {json.dumps(case.get('generation_parameters', {}), sort_keys=True) for case, _ in batch}
        result.extend([[pair] for pair in batch] if len(keys) > 1 else [batch])
    return result


def load_profile(path):
    import plenoptic_paths as layout
    profile = path if isinstance(path, dict) else json.loads(Path(path).read_text())
    if profile.get('schema') != 1 or not profile.get('videos'):
        raise ValueError('Video pipeline profile requires schema 1 and input videos')
    result = copy.deepcopy(profile)
    pinned = {'checkpoint', 'checkpoint_sha256', 'checkpoint_step'} & set(result)
    if pinned:
        raise ValueError('Video profiles cannot pin a checkpoint; stage-one validation always uses synchronized latest.pt')
    if result.get('skip_standard_videos'):
        raise ValueError('Video profiles cannot skip standard videos; ./validate.sh start runs the complete validation')
    identifiers = set()
    from validation_artifacts import CASE_ID
    for entry in result['videos']:
        name = entry.get('id')
        if not isinstance(name, str) or not CASE_ID.fullmatch(name) or name in identifiers:
            raise ValueError('Video profile IDs must be unique safe names')
        identifiers.add(name)
        per_video_profile(result, entry)
        if not isinstance(entry.get('prompt'), str) or not entry['prompt'].strip():
            raise ValueError('Every input video needs a prompt')
        entry['source'] = str(layout.rooted(entry['source']))
        source = Path(entry['source']).resolve()
        if not source.is_relative_to(layout.INPUTS_ROOT / 'user_videos'):
            raise ValueError('Pipeline input videos must be below DATASETS/inputs/user_videos')
        if not source.is_file():
            raise FileNotFoundError(source)
        sampling_plan(dict(k=1, denoising_steps=35, guidance=1.5, shift=5.),
                      [dict(qualitative_only=True, generation_parameters=result['generation'])])
    scale = result.get('camera_translation_scale', 1.)
    if type(scale) not in (int,float) or not math.isfinite(scale) or not 0 < scale <= 16:
        raise ValueError('Camera translation scale must be finite and in (0,16]')
    segment = result['segmentation']
    if (type(segment['min_frames']) is not int or type(segment['max_frames']) is not int
            or not 1 <= segment['min_frames'] <= segment['max_frames'] <= 80
            or not 0 < segment['max_rotation_degrees'] <= 30):
        raise ValueError('Invalid segmentation limits; at most 80 new frames leave room for refinement history')
    if not result['camera_candidates']:
        raise ValueError('Provide explicit camera candidates')
    for candidate in result['camera_candidates']:
        if (set(candidate) != {'azimuth', 'distance_scale', 'source_pitch', 'target_pitch'}
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in candidate.values())
                or not 0 < candidate['azimuth'] < 90
                or not 1 <= candidate['distance_scale'] <= 4
                or not 0 < candidate['source_pitch'] < 85
                or not 0 < candidate['target_pitch'] < 85):
            raise ValueError('Invalid external-camera candidate')
    if result.get('temporal_sampling','pad') not in ('pad','stretch'):
        raise ValueError('Unknown temporal sampling method')
    selected = result.get('preview_segments',[])
    if (not isinstance(selected,list) or any(type(value) is not int or value < 1 for value in selected)
            or len(selected) != len(set(selected))):
        raise ValueError('Preview segments must be distinct positive integers')
    variants = result.get('tuning_variants',[])
    names = set()
    if variants and not result.get('preview_segments'):
        raise ValueError('Tuning variants require explicit preview segments')
    for variant in variants:
        name = variant.get('id')
        if not isinstance(name,str) or not CASE_ID.fullmatch(name) or name in names:
            raise ValueError('Tuning variant IDs must be safe and unique')
        names.add(name)
        if variant.get('temporal_sampling',result.get('temporal_sampling','pad')) not in ('pad','stretch'):
            raise ValueError('Unknown variant temporal sampling method')
        if set(variant)-{'id','generation','camera','camera_translation_scale','temporal_sampling'}:
            raise ValueError('Unknown tuning variant parameter')
        sampling_plan(dict(k=1,denoising_steps=35,guidance=1.5,shift=5.),
                      [dict(qualitative_only=True,generation_parameters=variant.get('generation',{}))])
        scale = variant.get('camera_translation_scale',result.get('camera_translation_scale',1.))
        if type(scale) not in (int,float) or not math.isfinite(scale) or not 0 < scale <=16:
            raise ValueError('Invalid tuning camera scale')
        if variant.get('camera') is not None:
            trial = copy.deepcopy(result)
            trial.pop('tuning_variants',None)
            trial['camera_candidates'] = [variant['camera']]
            load_profile(trial)
    return result


def rotation_distance_degrees(left, right):
    import numpy as np
    cosine = (np.trace(left.T @ right) - 1) / 2
    return math.degrees(math.acos(float(np.clip(cosine, -1., 1.))))


def stable_segments(rotations, *, min_frames=25, max_frames=80, max_rotation_degrees=8., boundaries=()):
    """Keep every real frame exactly once; mark source motion that cannot be split further."""
    import numpy as np
    rotations = np.asarray(rotations)
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3) or len(rotations) < 1:
        raise ValueError('Expected a nonempty sequence of camera rotations')
    if not np.isfinite(rotations).all() or not 1 <= min_frames <= max_frames <= 80:
        raise ValueError('Invalid source rotations or clip-length limits')
    # Optimize the whole timeline: penalize rotation spread and excessive cuts.
    traces = np.einsum('aij,bij->ab', rotations, rotations)
    angles = np.degrees(np.arccos(np.clip((traces-1)/2, -1., 1.)))
    n = len(rotations)
    costs = np.full(n+1, np.inf)
    costs[0] = 0.
    previous = {}
    preferred = {int(b) for b in boundaries}
    for end in range(1, n+1):
        for start in range(max(0, end-max_frames), end):
            length = end-start
            if length < min_frames and not (start == 0 and end == n):
                continue
            if not np.isfinite(costs[start]):
                continue
            # A medoid anchor is robust to a brief fast turn at either clip edge.
            block = angles[start:end, start:end]
            anchor = int(np.argmin(np.mean(np.minimum(block, 60.)**2, axis=1)))
            distances = block[anchor]
            cost = costs[start] + float(np.sum((np.minimum(distances, 60.) / max_rotation_degrees)**2))
            cost += 8. - (.5 if end in preferred else 0.)
            if cost < costs[end]:
                costs[end] = cost
                previous[end] = (start, start+anchor, float(distances.max()))
    if n not in previous:
        raise ValueError('Timeline cannot satisfy the chosen minimum/maximum clip length')
    result = []
    end = n
    while end:
        start, anchor, spread = previous[end]
        result.append(dict(start_frame=start, valid_frames=end-start, anchor_frame=anchor,
                           cut_reason='optimized_rotation_spread_and_clip_count',
                           maximum_rotation_from_anchor_degrees=round(spread, 4),
                           stable_proxy=spread <= max_rotation_degrees))
        end = start
    result.reverse()
    assert sum(item['valid_frames'] for item in result) == n
    return result


def workspace_roi(frame):
    """Locate the olive desk mat in these explicitly configured sample videos."""
    import cv2
    import numpy as np
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array([22, 25, 40]), np.array([82, 220, 245]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        fraction = cv2.contourArea(contour) / (width * height)
        if .10 <= fraction <= .85:
            x, y, w, h = cv2.boundingRect(contour)
            return dict(bounds=[x/width, y/height, (x+w)/width, (y+h)/height],
                        method='olive_mat_color_region_heuristic', detected_fraction=fraction)
    return dict(bounds=[.12, .22, .88, .90], method='fallback_nominal_workspace', detected_fraction=None)


def camera_candidate_scores(intrinsic, depth, candidates, *, width=768, height=432, bounds=(.12, .22, .88, .90)):
    """Score visibility of a nominal workspace plane; this is not a video-quality metric."""
    import numpy as np
    from prepare_custom_exo import external_pose
    # A fixed image-space workspace approximation, not recovered metric geometry.
    u, v = np.meshgrid(np.linspace(bounds[0] * width, bounds[2] * width, 9),
                       np.linspace(bounds[1] * height, bounds[3] * height, 7))
    rays = np.linalg.inv(intrinsic) @ np.stack([u.ravel(), v.ravel(), np.ones(u.size)])
    points = np.concatenate([rays * depth, np.ones((1, u.size))])
    result = []
    for candidate in candidates:
        pose, focus, up = external_pose(depth, candidate['source_pitch'], candidate['target_pitch'],
                                        candidate['azimuth'], candidate['distance_scale'])
        workspace_center = (rays * depth).mean(axis=1)
        pose[:3, 3] += workspace_center - focus
        focus = workspace_center
        target = np.linalg.inv(pose) @ points
        projected = intrinsic @ target[:3]
        uv = projected[:2] / np.maximum(projected[2:], 1e-8)
        visible = ((target[2] > .05) & (uv[0] >= 0) & (uv[0] < width)
                   & (uv[1] >= 0) & (uv[1] < height))
        occupancy = max(0., float((uv[0].max()-uv[0].min()) * (uv[1].max()-uv[1].min()) / (width*height)))
        coverage = float(visible.mean())
        # Prefer a visible, sufficiently large workspace and moderate view change.
        score = 4 * coverage - abs(occupancy - .30) - .003 * abs(candidate['azimuth'] - 45.)
        result.append(dict(parameters=candidate, score=round(score, 6),
                           visible_fraction=coverage, projected_frame_fraction=occupancy,
                           relative_c2w=pose.tolist(), focus=focus.tolist(), up=up.tolist()))
    return sorted(result, key=lambda row: row['score'], reverse=True)
