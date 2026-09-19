"""Bounded video-feedback prompt repair; never receives reference or score data."""
import hashlib
import json

from .backend import TEXT, obj, save

CONTROL_SUFFIX = (' One continuous eight-second shot. Preserve the provided initial '
                  'image, the existing camera viewpoint, and the identity of every '
                  'visible object and robot throughout the requested action.')
PRIORITY = ('task_completion', 'entity_identity', 'visible_action_order',
            'motion_continuity', 'camera_preservation')
REPAIR_SCHEMA = obj({
    'decision': {'type': 'string', 'enum': ['APPLY', 'ABSTAIN']},
    'reason': TEXT,
    'residual': {'type': 'string', 'maxLength': 450},
    'evidence_frame_ids': {'type': 'array', 'maxItems': 16,
                           'items': {'type': 'integer', 'minimum': 1, 'maximum': 16}},
})
CHECKS = ('literal_task_preserved', 'necessary_actions_only',
          'initial_image_consistent', 'no_extra_spatial_or_timing_constraints')
GUARD_SCHEMA = obj({k: obj({
    'status': {'type': 'string', 'enum': ['PASS', 'FAIL', 'UNKNOWN']},
    'reason': TEXT}) for k in CHECKS})

REPAIR_PROMPT = '''Propose at most ONE short corrective addition to a robot-video
generation instruction, based on the reported failure of its first generated
video. The real initial image is supplied. The report can be mistaken: ABSTAIN
when the proposed correction is not supported by the literal task and visible
initial state. Correct only the supplied failed criterion. Do not rewrite the
instruction or narrate the whole scene. Return a positive residual of 1-45 words
and the supplied evidence frame IDs if APPLY; otherwise an empty residual.
Keep the exact requested arm, object, action and terminal state. Do not add colors,
backgrounds, waypoints, viewer-left/right directions, precise timing, camera moves,
an arm freeze, regrasp, retraction or release unless explicitly required by the
task. A placement needs support before release; a pickup ends held. Do not impose
a new grasp if the initial image already shows a held object. No inferred forces,
joint angles, reference-video motion, scores or benchmark names. The complete
literal task will remain an immutable prefix; your addition must not contradict
or expand it. Return JSON only.
'''
GUARD_PROMPT = '''Check a proposed prompt addition against the literal task and
REAL INITIAL IMAGE. You do not see a reference video or any evaluation score.
PASS only when the addition preserves the named arm/object/action/endpoint,
contains only necessary actions, is consistent with visible initial state, and
adds no arbitrary direction, waypoint, timing, arm freeze, or background detail.
Uncertain identity, contact or hidden state is UNKNOWN, never PASS. You are
validating instruction fidelity, not judging physical success of a video.
Return the four JSON checks. Do not fix or rewrite the addition.
'''


def compose_prompt(instruction, residual=''):
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError('Missing literal task')
    if len(residual.split()) > 45:
        raise ValueError('Residual exceeds frozen 45-word budget')
    if any(x in residual.lower() for x in ('ndtw', 'bleuscore', 'clipscore', 'gt_dataset/')):
        raise ValueError('Evaluation identity is not permitted')
    return instruction + CONTROL_SUFFIX + (' ' + residual.strip() if residual.strip() else '')


def guard_accepts(checks):
    return set(checks) == set(CHECKS) and all(checks[k]['status'] == 'PASS' for k in CHECKS)


def prepare_repair(backend, row, combined_base_audit, folder):
    failed = next((g for g in PRIORITY if combined_base_audit['gates'][g]['status'] == 'FAIL'), None)
    record = {'target_gate': failed, 'residual': '', 'status': 'NO_REPORTED_FAILURE',
              'instruction_sha256': hashlib.sha256(row['instruction'].encode()).hexdigest(),
              'initial_sha256': row['initial_sha256'], 'usage': []}
    if failed:
        evidence = combined_base_audit['gates'][failed]
        context = '\nLITERAL TASK:\n' + row['instruction'] + '\nREPORTED FAILURE:\n' + json.dumps({failed: evidence})
        proposal, usage = backend.visual_query(REPAIR_PROMPT + context, [row['initial']],
                                               REPAIR_SCHEMA, 'bounded-residual', row['seed'])
        record['usage'].append(usage); record['proposal'] = proposal
        record['status'] = 'PROPOSER_ABSTAINED'
        if proposal['decision'] == 'APPLY':
            residual = proposal['residual'].strip()
            # Invalid proposals fail the run, never silently become approved prompts.
            if not residual or not proposal['evidence_frame_ids']:
                raise ValueError('APPLY requires a residual and cited observed frames')
            if not set(proposal['evidence_frame_ids']) <= set(evidence['frame_ids']):
                raise ValueError('Repair cites evidence outside the supplied failure')
            compose_prompt(row['instruction'], residual)
            checks, usage = backend.visual_query(GUARD_PROMPT + '\nLITERAL TASK:\n' + row['instruction']
                + '\nPROPOSED ADDITION:\n' + residual, [row['initial']], GUARD_SCHEMA,
                'instruction-fidelity-guard', row['seed'])
            record['usage'].append(usage); record['guard'] = checks
            record['status'] = 'GUARD_REJECTED'
            if guard_accepts(checks):
                record.update(status='APPLIED', residual=residual)
    record['control_prompt'] = compose_prompt(row['instruction'])
    record['repair_prompt'] = compose_prompt(row['instruction'], record['residual'])
    record['physical_success_established'] = False
    save(folder / 'repair-plan.json', record)
    return record
