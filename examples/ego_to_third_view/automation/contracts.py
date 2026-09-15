"""Fail-closed, frame-explicit contracts shared by batch stages and visual review."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

VERSION = 1
INVARIANTS = {
    'actor_origin': 'source_ego_world',
    'arm_lengths': 'constant_per_actor',
    'one_shared_torso': True,
    'render_camera_controls_actor': False,
    'ego_timeline_authority': True,
    'preserve_object_identity': True,
    'require_pre_dit_review': True,
    'require_post_dit_review': True,
    'allow_cross_numa': False,
    'allow_unverified_acceleration': False,
}
ISSUE_STAGE = {
    'wrong_actor_origin': 'reconstruct', 'wrong_metric_scale': 'reconstruct',
    'wrong_object_identity': 'reconstruct', 'missing_geometry': 'reconstruct',
    'unreachable_wrist': 'stabilize', 'limb_length': 'stabilize',
    'elbow_flip': 'stabilize', 'torso_translation': 'stabilize',
    'contact_slip': 'stabilize', 'object_jitter': 'stabilize',
    'camera_drift': 'render', 'occlusion': 'render', 'background': 'render',
    'sim_timing': 'reconstruct', 'dit_timing': 'align',
    'dit_anatomy': 'generate', 'dit_identity': 'generate', 'dit_camera_cut': 'generate',
    'insufficient_evidence': 'review',
}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def number(value, name, low=None, high=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name}: expected finite number')
    if low is not None and value < low or high is not None and value > high:
        raise ValueError(f'{name}: outside [{low}, {high}]')
    return value


def vector(value, name, length=3):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f'{name}: expected {length} numbers')
    return [number(v, name) for v in value]


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', value):
        raise ValueError('IDs must be safe, non-path identifiers')
    return value


def validate_scene(scene):
    if scene.get('schema_version') != VERSION:
        raise ValueError('Unsupported scene schema')
    if scene.get('coordinate_system') != {'frame': 'scene_world', 'unit': 'm', 'up': '+z', 'handedness': 'right'}:
        raise ValueError('All geometry must use explicit right-handed scene_world metres, +z up')
    actor = scene['actor']
    if actor.get('origin_authority') != 'source_ego_world' or not actor.get('origin_evidence'):
        raise ValueError('Actor placement must be estimated from source ego evidence, not render camera')
    number(actor.get('origin_confidence'), 'origin_confidence', .7, 1)
    vector(actor['root_m'], 'root_m')
    number(actor['heading_rad'], 'heading_rad')
    if actor.get('root_motion', 'stationary') != 'stationary':
        raise ValueError('This rig currently supports stationary manipulation only; use a validated locomotion adapter')
    # Anthropometry is chosen once per person/clip, never adjusted to fit a reach.
    number(actor['upper_arm_m'], 'upper_arm_m', .22, .40)
    number(actor['forearm_m'], 'forearm_m', .20, .36)
    number(actor['shoulder_width_m'], 'shoulder_width_m', .28, .52)
    if actor.get('length_policy') != 'constant_per_actor':
        raise ValueError('Telescoping or reach-dependent limbs are forbidden')
    camera = scene['render_camera']
    vector(camera['eye_m'], 'render_camera.eye_m')
    vector(camera['target_m'], 'render_camera.target_m')
    delta=[a-b for a,b in zip(camera['eye_m'],camera['target_m'])]
    if math.hypot(delta[0],delta[1])<1e-6:
        raise ValueError('Camera eye/target needs a nonvertical viewing direction for this renderer')
    if camera.get('actor_placement_input', False):
        raise ValueError('Render camera must not drive actor placement')
    if not scene.get('source_sha256') or len(scene['source_sha256']) != 64:
        raise ValueError('Scene must be bound to its ego source')
    objects = scene.get('objects', [])
    ids = [identifier(o['id']) for o in objects]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate object IDs')
    for obj in objects:
        if obj.get('orientation_mode', 'object') not in ('object', 'world_locked'):
            raise ValueError('Invalid object orientation mode')
        if obj.get('orientation_mode') == 'world_locked' and not obj.get('symmetry_evidence'):
            raise ValueError('World-locked contact requires explicit symmetry/observability evidence')
    return scene


def validate_events(events, duration):
    ids = set()
    previous = -1.
    for event in events:
        key = identifier(event['id'])
        if key in ids:
            raise ValueError('Duplicate event ID')
        ids.add(key)
        t = number(event['time_s'], 'event time', 0, duration)
        if t < previous:
            raise ValueError('Events must be chronological')
        previous = t
        number(event['confidence'], 'event confidence', 0, 1)
        if not event.get('evidence'):
            raise ValueError('Event needs source evidence, not only a prompt timestamp')
    return events


def validate_manifest(manifest):
    if manifest.get('schema_version') != VERSION:
        raise ValueError('Unsupported batch schema')
    if not manifest.get('clips'):
        raise ValueError('No clips')
    seen = set()
    for clip in manifest['clips']:
        key = identifier(clip['id'])
        if key in seen:
            raise ValueError('Duplicate clip ID')
        seen.add(key)
        if not isinstance(clip.get('source'), str):
            raise ValueError('Every clip requires its own source path')
    for name, expected in INVARIANTS.items():
        if manifest.get('invariants', {}).get(name, expected) != expected:
            raise ValueError(f'Cannot relax invariant {name}')
    retries = manifest.get('max_repairs', 2)
    if isinstance(retries, bool) or not isinstance(retries, int) or not 0 <= retries <= 10:
        raise ValueError('max_repairs must be 0..10')
    for name, adapter in manifest.get('adapters', {}).items():
        if name not in ('reconstruct','stabilize','render','review_pre','review_post'):
            raise ValueError(f'Unknown adapter stage {name}')
        argv = adapter.get('argv')
        if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
            raise ValueError(f'{name}: adapter argv must be a nonempty string array; no shell interpolation')
    h3=manifest.get('h3',{})
    supported={'python','model_path','sol_root','num_gpus','steps','min_free_mib','max_utilization',
               'allowed_gpus','lease_dir','master_port','startup_timeout_s','request_timeout_s'}
    if set(h3)-supported:raise ValueError(f'Unsupported/unverified H3 options: {sorted(set(h3)-supported)}')
    if h3.get('num_gpus',4)!=4:raise ValueError('Current integrated H3 profile is four-GPU FSDP+Ulysses4; benchmark other topologies separately')
    if h3.get('steps',24)<14:raise ValueError('Few-step sampling needs a validated Ref2VA distillation adapter, not a smaller integer')
    return manifest


def validate_review(review, expected_binding, phase):
    if review.get('binding') != expected_binding:
        raise ValueError('Review is stale or belongs to another clip/attempt')
    if review.get('phase') != phase or review.get('reviewer', {}).get('kind') not in ('human', 'codex', 'vlm'):
        raise ValueError('Visual review must identify phase and actual visual reviewer')
    if not review.get('reviewer', {}).get('name') or not review.get('inspected_frames'):
        raise ValueError('Review must identify reviewer and inspected frames')
    inspected=review['inspected_frames']
    if isinstance(inspected,dict):
        coverage=set()
        for lo,hi in inspected.get('ranges',[]):
            if not isinstance(lo,int) or not isinstance(hi,int) or not 0<=lo<=hi<expected_binding['frames']:
                raise ValueError('Invalid reviewed frame range')
            coverage.update(range(lo,hi+1))
    elif isinstance(inspected,list) and all(isinstance(i,int) and 0<=i<expected_binding['frames'] for i in inspected):
        coverage=set(inspected)
    else:raise ValueError('inspected_frames must be indices or inclusive ranges')
    if review.get('decision') not in ('accept', 'repair', 'needs_input'):
        raise ValueError('Invalid review decision')
    required = {'actor_origin', 'limb_lengths', 'torso', 'contacts', 'identity', 'occlusion', 'camera', 'timing', 'background'}
    checks = review.get('checks', {})
    if set(checks) != required or any(v not in ('pass', 'fail', 'unknown') for v in checks.values()):
        raise ValueError('All visual checks must be explicitly assessed')
    for issue in review.get('issues', []):
        if issue.get('code') not in ISSUE_STAGE or not issue.get('evidence'):
            raise ValueError('Issue requires a known repair code and visual evidence')
    if review['decision'] == 'accept' and (set(checks.values()) != {'pass'} or review.get('issues')):
        raise ValueError('Failed/unknown checks cannot be promoted')
    if review['decision']=='accept' and coverage!=set(range(expected_binding['frames'])):
        raise ValueError('Acceptance requires full-timeline visual coverage, including transitions')
    if review['decision'] == 'repair' and not review.get('issues'):
        raise ValueError('Repair decision requires actionable issues')
    return review
