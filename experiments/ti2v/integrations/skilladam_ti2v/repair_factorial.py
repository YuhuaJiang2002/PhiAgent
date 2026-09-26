"""Frozen proposal-factorial controls; no changes to deployment evaluation."""
import json
import re

from .backend import TEXT, obj, save
from .relational_repair import GROUNDING_CHECKS, TEMPLATES, check_schema, compile_repair, text_sha256
from .residual_repair import PRIORITY


EXTENDED_TEMPLATES = {
    **TEMPLATES,
    'support_before_release': (
        'When placing an object, establish support at the task-specified destination '
        'before releasing the grasp; do not add a withdrawal action.'
    ),
    'receiver_grasp_before_giver_release': (
        'When handing an object between grippers, establish the receiving grasp '
        'before the giving gripper releases the object.'
    ),
    'maintain_contact_during_push': (
        'When pushing the task-specified object, maintain visible contact while '
        'the gripper drives its motion; do not add grasping or lifting.'
    ),
}
ARMS = ('combined_original', 'factored_original', 'combined_extended', 'factored_extended')
STATUS = {'type': 'string', 'enum': ['PASS', 'FAIL', 'UNKNOWN']}
FRAME_IDS = {'type': 'array', 'maxItems': 16,
             'items': {'type': 'integer', 'minimum': 1, 'maximum': 16}}
ENTITY_FIELDS = {
    'actor_quote': TEXT,
    'actor_binding': {'type': 'string', 'enum': ['LITERAL_SPAN', 'TASK_UNSPECIFIED']},
    'object_quote': TEXT,
    'binding_observation': TEXT,
    'binding_status': STATUS,
}
EVIDENCE_FIELDS = {
    'target_gate': {'type': 'string', 'enum': list(PRIORITY)},
    'evidence_frame_ids': FRAME_IDS,
    'visible_observation': TEXT,
    'failure_visible': STATUS,
}
GROUND_SCHEMA = obj({**ENTITY_FIELDS, 'assessments': {
    'type': 'array', 'maxItems': 5, 'items': obj(EVIDENCE_FIELDS)}})
OBSERVATION_RULES = (
    'Describe only visible evidence in the real initial image and sixteen generated frames. '
    'The literal task alone defines the required action. Bind object_quote to an exact '
    'task substring and identify that object in the initial image. Bind actor_quote to '
    'an exact task substring when the task specifies the acting arm or gripper. Otherwise '
    'use actor_binding TASK_UNSPECIFIED and an empty actor_quote; never invent an arm. '
    'binding_status is PASS only if the intended entities are identifiable without guessing. '
    'Return exactly one assessment for each supplied failed gate in the supplied order. '
    'failure_visible is PASS only when the stated failure is visibly supported. '
    'Cite only generated frame IDs already cited by that gate. Occluded contact, unresolved '
    'identity or hidden geometry remains UNKNOWN. Do not infer forces, depth or future references. '
    'Keep observations concise and factual. Do not require a new grasp for an already held '
    'object, release after a pickup, or withdrawal absent from a placement instruction. '
)
PROPOSAL_RULES = (
    'Choose at most one listed relation for each failed gate, or ABSTAIN. '
    'Apply only when the literal task requires the relation and the visible failure is '
    'addressed by that relation. General topic overlap is insufficient: a grasp template '
    'does not repair a camera change, a wrong object, or an unsupported action. '
    'Do not add attributes, actions, directions, timing constraints or reference paths. '
    'A relation is a generic skill constraint, not a case-specific rewritten task. '
    'State a short relation-applicability reason before the final decision. '
)


def actor_fields(instruction=None):
    fields = dict(ENTITY_FIELDS)
    if instruction is not None:
        actors = list(dict.fromkeys(match.group(0) for match in re.finditer(
            r'\b(?:left|right)\s+(?:arm|gripper)\b', instruction, flags=re.IGNORECASE)))
        fields['actor_quote'] = {'type': 'string', 'enum': actors or ['']}
        fields['actor_binding'] = {'type': 'string', 'enum': ['LITERAL_SPAN' if actors else 'TASK_UNSPECIFIED']}
    return fields


def grounding_schema(instruction):
    return obj({**actor_fields(instruction), 'assessments': {
        'type': 'array', 'maxItems': 5, 'items': obj(EVIDENCE_FIELDS)}})


def proposal_schema(templates, instruction=None):
    return obj({**actor_fields(instruction), 'assessments': {
        'type': 'array', 'maxItems': 5,
        'items': obj({**EVIDENCE_FIELDS, 'relation_reason': TEXT,
                      'relation_id': {'type': 'string', 'enum': ['none', *templates]},
                      'decision': {'type': 'string', 'enum': ['APPLY', 'ABSTAIN']}})}})


def literal_relation_precondition(instruction, relation_id):
    match = re.match(r'\s*(pick\s+up|grab|place|put|pass|push)\b', instruction, re.IGNORECASE)
    if not match:
        return 'UNKNOWN'
    action = match.group(1).lower()
    families = {
        'pick up': 'pickup', 'grab': 'pickup', 'place': 'placement',
        'put': 'placement', 'pass': 'handover', 'push': 'pushing',
    }
    family = families.get(re.sub(r'\s+', ' ', action))
    allowed = {
        'contact_before_transport': {'pickup'},
        'maintain_grasp': {'pickup', 'placement', 'handover'},
        'release_before_withdrawal': {'placement'},
        'support_before_release': {'placement'},
        'receiver_grasp_before_giver_release': {'handover'},
        'maintain_contact_during_push': {'pushing'},
    }
    if relation_id not in allowed:
        return 'UNKNOWN'
    if family not in allowed[relation_id]:
        return 'FAIL'
    if relation_id == 'release_before_withdrawal':
        if re.search(r'\b(?:no|not|never|without)\b[^.;]*\b(?:withdraw\w*|retract\w*)\b', instruction, re.IGNORECASE):
            return 'FAIL'
        if not re.search(r'\b(?:then|and)\s+(?:withdraw|retract)\b', instruction, re.IGNORECASE):
            return 'UNKNOWN'
    return 'PASS'


def validate_grounding(record, instruction, failed):
    actor = record['actor_quote']
    if record['actor_binding'] == 'TASK_UNSPECIFIED':
        if actor != '':
            raise ValueError('Unspecified actor must not introduce an actor quote')
    elif record['actor_binding'] != 'LITERAL_SPAN' or not actor.strip() or actor not in instruction:
        raise ValueError('Actor must be unspecified or an exact literal span')
    target = record['object_quote']
    if not target.strip() or target not in instruction:
        raise ValueError('Object must bind an exact literal span')
    assessments = record['assessments']
    if [entry['target_gate'] for entry in assessments] != list(failed):
        raise ValueError('Missing, duplicated or reordered failed-gate assessments')
    for entry in assessments:
        frames = entry['evidence_frame_ids']
        if any(type(frame) is not int or not 1 <= frame <= 16 for frame in frames):
            raise ValueError('Invalid generated-frame identity')
        if len(set(frames)) != len(frames) or not set(frames) <= set(failed[entry['target_gate']]['frame_ids']):
            raise ValueError('Evidence must use distinct inherited failure witnesses')
        if entry['failure_visible'] == 'PASS' and not frames:
            raise ValueError('Visible failure needs generated witnesses')


def select_proposal(first, second, instruction, failed, templates):
    validate_grounding(first, instruction, failed)
    validate_grounding(second, instruction, failed)
    if first['binding_status'] != 'PASS' or second['binding_status'] != 'PASS':
        return None, 'ENTITY_UNRESOLVED'
    if any(first[key] != second[key] for key in ('actor_quote', 'actor_binding', 'object_quote')):
        return None, 'ENTITY_BINDING_CHANGED'
    original = {entry['target_gate']: entry for entry in first['assessments']}
    for entry in second['assessments']:
        if entry['decision'] != 'APPLY':
            continue
        if entry['relation_id'] not in templates:
            raise ValueError('Unknown relation template')
        inherited = original[entry['target_gate']]
        if inherited['failure_visible'] != 'PASS' or entry['failure_visible'] != 'PASS':
            return None, 'FAILURE_NOT_GROUNDED'
        if not set(entry['evidence_frame_ids']) <= set(inherited['evidence_frame_ids']):
            return None, 'EVIDENCE_CHANGED'
        return {**entry, **{key: second[key] for key in ('actor_quote', 'actor_binding', 'object_quote')}}, 'PROPOSED'
    return None, 'PROPOSER_ABSTAINED'


def prepare_factorial_repair(backend, row, base_audit, skill, folder, arm, images, positions):
    if arm not in ARMS:
        raise ValueError('Unknown frozen arm')
    templates = EXTENDED_TEMPLATES if arm.endswith('_extended') else TEMPLATES
    binding = {'video_sha256': row['base_sha256'], 'initial_sha256': row['initial_sha256'],
               'instruction_sha256': text_sha256(row['instruction'])}
    result = {'arm': arm, 'status': 'ABSTAINED', 'reason': 'NO_REPORTED_FAILURE',
              'skill': skill, 'residual': '', 'edits': [], 'usage': [],
              'source_binding': binding, 'physical_success_established': False,
              'original_audit_changed': False, 'stage_reached': 'SOURCE_SCREEN'}
    failed = {gate: base_audit['gates'][gate] for gate in PRIORITY
              if base_audit['gates'][gate]['status'] == 'FAIL'}
    if not failed:
        save(folder / 'plan.json', result)
        return result
    context = '\nFAILED GATES:\n' + json.dumps(failed) + '\nLITERAL TASK:\n' + row['instruction']
    vocabulary = '\nFROZEN RELATIONS:\n' + json.dumps(templates)
    combined = arm.startswith('combined_')
    first_prompt = OBSERVATION_RULES
    if combined:
        first_prompt += PROPOSAL_RULES + vocabulary
    else:
        first_prompt += 'Do not choose a repair. Record entity bindings and visible failures only. '
    first, usage = backend.visual_query(first_prompt + context, [row['initial']] + images,
        proposal_schema(templates, row['instruction']) if combined else grounding_schema(row['instruction']),
        arm + '-stage1', row['seed'], positions)
    result['usage'].append(usage)
    result['first_stage'] = first
    result['stage_reached'] = 'OBSERVATION_RECORDED'
    second_prompt = (
        OBSERVATION_RULES + PROPOSAL_RULES
        + 'The first-stage record is supplied below. Preserve its entity bindings and do not '
        'turn its FAIL or UNKNOWN evidence/binding statuses into PASS. '
        + ('Complete one predeclared refinement of the joint proposal; reassess relation choices only. '
           if combined else 'Now map the recorded observations to an applicable relation. ')
        + vocabulary + context + '\nFIRST-STAGE RECORD:\n' + json.dumps(first))
    second, usage = backend.visual_query(second_prompt, [row['initial']] + images,
        proposal_schema(templates, row['instruction']), arm + '-stage2', row['seed'], positions)
    result['usage'].append(usage)
    result['second_stage'] = second
    result['stage_reached'] = 'PROPOSAL_RECORDED'
    try:
        proposal, reason = select_proposal(first, second, row['instruction'], failed, templates)
        result['reason'] = reason
    except (KeyError, TypeError, ValueError) as error:
        proposal = None
        result.update(reason='INVALID_PROPOSAL', error=str(error))
    if proposal is not None:
        result['proposal'] = proposal
        result['stage_reached'] = 'GROUNDING_GUARD'
        guard_prompt = (
            OBSERVATION_RULES + 'Verify the proposed relation; do not revise it. '
            'Every listed check must PASS independently; uncertainty rejects the edit. '
            'Verify task fidelity, necessary actions only, initial-image consistency, absence '
            'of invented spatial or timing constraints, task-required relation, visibly '
            'supported source failure, and correct entity binding. TASK_UNSPECIFIED means '
            'the task did not specify an acting arm and the relation must not introduce one. '
            'Do not infer an actual grasp or contact from a desired task outcome. '
            + context + '\nPROPOSAL:\n' + json.dumps(proposal)
            + '\nRELATION:\n' + templates[proposal['relation_id']])
        checks, usage = backend.visual_query(guard_prompt, [row['initial']] + images,
            check_schema(GROUNDING_CHECKS), arm + '-guard', row['seed'], positions)
        result['usage'].append(usage)
        result['guard'] = checks
        try:
            result.update(compile_repair(skill, proposal, instruction=row['instruction'],
                source_binding=binding, expected_binding=binding, base_audit=base_audit,
                positions=positions, checks=checks, templates=templates,
                actor_unspecified=proposal['actor_binding'] == 'TASK_UNSPECIFIED'))
        except (KeyError, TypeError, ValueError) as error:
            result.update(reason='INVALID_PROPOSAL', error=str(error))
    save(folder / 'plan.json', result)
    return result