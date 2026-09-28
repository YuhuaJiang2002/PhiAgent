"""Provisional phase planning and instruction-blind visual evidence checks.

These image-space observations are neither an inverse dynamics model nor a
physical execution certificate. No benchmark reference or score enters here.
"""
import hashlib
import json

from .backend import GATES, GATE_SCHEMA, TEXT, obj


FRAME_IDS = {'type': 'array', 'minItems': 1, 'maxItems': 16,
             'items': {'type': 'integer', 'minimum': 1, 'maximum': 16}}
TRACE_GATE_SCHEMA = obj({'terminal_goal': TEXT, 'gates': obj({g: obj({
    'status': {'type': 'string', 'enum': ['PASS', 'FAIL', 'UNKNOWN']},
    'reason': TEXT, 'frame_ids': FRAME_IDS}) for g in GATES})})
PHASE_SCHEMA = obj({
    'initial_observation': TEXT, 'initial_uncertainties': TEXT,
    'literal_goal': TEXT, 'entities_and_arm_identity': TEXT,
    'phases': {'type': 'array', 'minItems': 1, 'maxItems': 4, 'items': obj({
        'action': TEXT, 'visible_precondition': TEXT, 'visible_postcondition': TEXT})},
    'prompt': TEXT,
})
WINDOWS = {'frames_01_04': list(range(1, 5)), 'frames_05_08': list(range(5, 9)),
           'frames_09_12': list(range(9, 13)), 'frames_13_16': list(range(13, 17))}
STATE_SCHEMA = obj({'object_state': TEXT, 'gripper_state': TEXT,
    'visible_contact': {'type': 'string', 'enum': ['YES', 'NO', 'UNKNOWN']},
    'camera_frame_relation': TEXT, 'uncertainty': TEXT})
OBSERVATION_SCHEMA = obj({
    'observed_entities': TEXT, 'observed_action': TEXT,
    'initial_state': TEXT, 'final_state': TEXT,
    'windows': obj({name: STATE_SCHEMA for name in WINDOWS}),
    'visible_discontinuities': TEXT, 'camera_changes': TEXT,
    'unobserved_or_ambiguous_intervals': TEXT,
})

PLAN_PROMPT = '''Build a minimal state-transition plan for the literal robot task
from the REAL INITIAL IMAGE only. Do not read or predict a reference video.
First record the visible initial object/gripper state and uncertainty. Then give
1-4 ordered phases with visible preconditions and postconditions. Omit phases
already completed in the initial image. Keep only actions required by the task:
do not add a new grasp if already held, return, regrasp, flourish or extra release.
Preserve the named robot arm; viewer-left/right are camera relations, not robot
left/right. If arm identity or a grasp is hidden, record uncertainty rather than
invent it. Preserve each object's visible identity and all stationary objects.
Motion is produced by the requested gripper-object interaction, with smooth
transitions between the existing start and required end state. No teleport,
morph, cut, camera shift, unexplained object motion or invented hidden forces.
Do not impose exact timings, new waypoints or an unrequested arm freeze/retract.
Write a concise positive generation prompt of 20-130 words for ONE continuous
eight-second video, grounded in the phases. No review scores or case identifiers.
Return JSON only. TASK: '''

OBSERVE_PROMPT = '''Record what is visibly happening in this robot video WITHOUT
guessing what someone wanted it to do. No task instruction or desired outcome is
provided. Image0 is the real initial image; images1..16 are uniformly spaced
generated frames including endpoints. Describe the same manipulated object(s)
and gripper(s) in EXACTLY four fixed windows: generated frames1-4, 5-8, 9-12,
13-16. Each window's record describes only its own frames and any change within
that window. These windows are fixed by the caller, not inferred event times.
Describe initial and final support/hold state, observed actions,
contact visibility, motion between states, object identity changes, camera
changes, extra actions, and ambiguous intervals. Object motion alone does not
prove contact. An occluded grasp/support relation is UNKNOWN. Viewer-left/right
are camera-frame relations only, never robot-arm identity. Do not assume a
trajectory is continuous merely because its endpoints are plausible. Report
only visible discontinuities; missing evidence remains uncertainty. Return JSON.
'''

CHECK_PROMPT = '''Audit the literal task against an independently recorded visual
trace. The observer did not see the task. You have no reference video, generating
prompt, method identity or benchmark score. Do not repair or reinterpret the
observed trace to match the task. Assess the original five visible gates:
task_completion, entity_identity, camera_preservation, motion_continuity,
visible_action_order. A pickup must end held and a placement supported, only when
requested. Check contact-before-object-motion when visible and necessary; do not
invent an initial grasp or unseen action. Correct nouns and a plausible final
frame alone do not establish action order or continuity. A concrete visible
contradiction is FAIL; insufficient or ambiguous evidence is UNKNOWN, never PASS.
PASS/FAIL need nonempty frame_ids drawn from the supplied trace. Judge only
camera-space visible evidence; no calibrated geometry, force or joint claims.
Return terminal_goal and five gates as JSON.
'''


def validate_plan(plan):
    if not 20 <= len(plan['prompt'].split()) <= 130:
        raise ValueError('Phase prompt outside frozen word budget')
    if any(not p[k].strip() for p in plan['phases'] for k in p):
        raise ValueError('Empty phase condition')
    text = json.dumps(plan).lower()
    if any(token in text for token in ('ndtw', 'bleuscore', 'clipscore', 'gt_dataset/')):
        raise ValueError('Evaluation identity in generation plan')


def validate_trace(trace):
    ids = [s['frame_ids'] for s in trace['states']]
    if not all(values == sorted(set(values)) for values in ids):
        raise ValueError('Unordered observation evidence')
    if any(max(a) > min(b) for a, b in zip(ids, ids[1:])):
        raise ValueError('Temporal state order is inconsistent')
    if 1 not in ids[0] or 16 not in ids[-1]:
        raise ValueError('Initial or terminal observation missing')


def intersect_audits(original, independent):
    """An independent check may veto, but cannot erase an existing non-pass."""
    result = {'terminal_goal': independent['terminal_goal'], 'gates': {}}
    for gate in GATES:
        a, b = original['gates'][gate], independent['gates'][gate]
        status = ('FAIL' if 'FAIL' in (a['status'], b['status']) else
                  'UNKNOWN' if 'UNKNOWN' in (a['status'], b['status']) else 'PASS')
        result['gates'][gate] = {'status': status,
            'reason': 'Original: ' + a['reason'] + ' Independent trace: ' + b['reason'],
            'frame_ids': sorted(set(a['frame_ids'] + b['frame_ids']))}
    return result


def validate_trace_verdict(checked, trace):
    supported = {i for state in trace['states'] for i in state['frame_ids']}
    for value in checked['gates'].values():
        if value['status'] != 'UNKNOWN' and (not value['frame_ids'] or not set(value['frame_ids']) <= supported):
            raise ValueError('Trace verdict cites unobserved frame evidence')


def phase_plan(backend, row, folder):
    from .backend import save
    plan, usage = backend.visual_query(PLAN_PROMPT + row['instruction'], [row['initial']],
        PHASE_SCHEMA, 'phase-plan', 20260916)
    validate_plan(plan)
    save(folder / 'plan.json', plan)
    save(folder / 'plan-binding.json', {'initial_sha256': row['initial_sha256'],
        'instruction_sha256': hashlib.sha256(row['instruction'].encode()).hexdigest(),
        'coordinate_frame': 'camera; qualitative visible relations only',
        'physical_certification': False, 'usage': usage})
    return plan


def trace_audit(backend, row, video, folder, original, motion_context=None):
    from .backend import save
    images, positions = backend.frames(video, folder / 'uniform')
    # Instruction blindness is enforced by the fixed prompt and API boundary.
    prompt = OBSERVE_PROMPT
    if motion_context is not None:
        prompt += '''\nAdditional independent pixel measurements follow. They use ALL native
frames. Initial spatial cells contain image features, not identified objects.
Use the supplied native-to-observer-frame mapping; native frame numbers are NOT
the frame_ids for your visual verdict. Pixels increase right/down in the resized
camera image. Border common motion is not calibrated camera pose. Missing/lost
tracks mean unavailable evidence, not stillness, occlusion success, or teleport.
Numbers alone cannot prove identity, task completion, contact or continuity.
Use them to locate visible changes and uncertainties in the supplied images.\n'''
        prompt += json.dumps(motion_context, separators=(',', ':'))
        save(folder / 'motion-context.json', motion_context)
    raw, usage = backend.visual_query(prompt, [row['initial']] + images,
        OBSERVATION_SCHEMA, 'instruction-blind-observation', row['seed'], positions)
    save(folder / 'raw-observation.json', raw)
    trace = {k: v for k, v in raw.items() if k != 'windows'}
    # IDs name the fixed supplied observation window, not invented per-frame
    # tracking annotations. The raw model response and window mapping are saved.
    trace['states'] = [{**raw['windows'][name], 'frame_ids': ids,
                       'evidence_scope': 'entire fixed observed window'}
                      for name, ids in WINDOWS.items()]
    validate_trace(trace)
    save(folder / 'observation.json', trace)
    checked, usage2 = backend.query([{'role': 'user', 'content': CHECK_PROMPT +
        '\nLITERAL TASK:\n' + row['instruction'] + '\nOBSERVED TRACE:\n' + json.dumps(trace)}],
        TRACE_GATE_SCHEMA, 'trace-contract-check', row['seed'], 2048)
    validate_trace_verdict(checked, trace)
    save(folder / 'independent-audit.json', checked)
    combined = intersect_audits(original, checked)
    save(folder / 'combined-audit.json', combined)
    save(folder / 'usage.json', [usage, usage2])
    return combined
