"""Method-side evidence diagnostics; never replace an audit or select a video."""
from copy import deepcopy

from .backend import GATES, obj
from .relational_repair import TEMPLATES


TEMPLATE_GATE_TOPICS = {
    'contact_before_transport': ('task_completion', 'visible_action_order'),
    'maintain_grasp': ('task_completion', 'visible_action_order', 'motion_continuity'),
    'release_before_withdrawal': ('task_completion', 'visible_action_order'),
}
APPLICABILITY_CHECKS = ('required_by_literal_task', 'initial_state_compatible')
ACTION_REQUIREMENTS = {
    'pickup': ('contact_before_transport', 'maintain_grasp'),
    'placement': ('maintain_grasp', 'support_before_release'),
    'handover': ('maintain_grasp', 'receiver_grasp_before_giver_release'),
    'push_open': ('maintain_contact_during_push',),
    'other': (),
    'unknown': (),
}
SHORT_TEXT = {'type': 'string', 'maxLength': 600}
TERNARY = {'type': 'string', 'enum': ['YES', 'NO', 'UNKNOWN']}
TASK_SCHEMA = obj({
    'reason': SHORT_TEXT,
    'action_quote': SHORT_TEXT,
    'actor_quote': SHORT_TEXT,
    'object_quote': SHORT_TEXT,
    'action_kind': {'type': 'string', 'enum': list(ACTION_REQUIREMENTS)},
    'initial_object_state': {'type': 'string', 'enum': ['HELD', 'SUPPORTED', 'NOT_IDENTIFIABLE', 'UNKNOWN']},
    'object_identifiable': TERNARY,
    'withdrawal_explicitly_requested': TERNARY,
    'templates': obj({name: obj({'reason': SHORT_TEXT, **{check: TERNARY for check in APPLICABILITY_CHECKS}})
                      for name in TEMPLATES}),
})
TASK_PROMPT = '''Diagnose only the applicability of the listed repair templates.
Use the literal task and REAL INITIAL IMAGE; no generated video or failure report
is available or needed to decide what the task requires. This is not a decision
to repair, and you must not propose a prompt or judge a generated action.
Report an exact task substring for the main action and target object. Quote the
named arm only when explicitly named; otherwise actor_quote is empty, not an
invented arm. If target identity is unclear, report it as such without guessing.
Assess EVERY template independently. required_by_literal_task asks whether that
relation belongs to the requested action; initial_state_compatible asks whether
its preconditions fit the visible initial state. YES, NO and UNKNOWN are distinct.
An existing grasp rules out requiring a new pickup. Maintaining a grasp can be
required while carrying an already held object to a placement destination; it
does not require observing a generated video to be task-applicable. Placement
requires support before release, but does not imply gripper withdrawal. The full
release_before_withdrawal template requires explicitly requested withdrawal.
A handover requires the receiver to hold before the giver releases; maintaining
a grasp alone does not encode that transfer. A push is not a grasp-and-carry task.
No hidden forces, future references, metric scores, new actions, or timing.
'''


def applicability_status(checks):
    if set(checks) != set(APPLICABILITY_CHECKS):
        raise ValueError('Incomplete task applicability checks')
    values = [checks[name] for name in APPLICABILITY_CHECKS]
    if any(value not in ('YES', 'NO', 'UNKNOWN') for value in values):
        raise ValueError('Invalid task applicability verdict')
    if 'NO' in values:
        return 'NOT_APPLICABLE'
    if 'UNKNOWN' in values:
        return 'UNKNOWN'
    return 'APPLICABLE'


def validate_task_diagnosis(value, instruction):
    if value['action_kind'] not in ACTION_REQUIREMENTS or set(value['templates']) != set(TEMPLATES):
        raise ValueError('Incomplete action or template vocabulary')
    binding_errors = []
    for name in ('action_quote', 'actor_quote', 'object_quote'):
        quote = value[name]
        if not isinstance(quote, str) or (quote and quote not in instruction):
            binding_errors.append(name + ':NOT_AN_EXACT_TASK_QUOTE')
    if value['action_kind'] not in ('unknown', 'other') and not value['action_quote'].strip():
        binding_errors.append('action_quote:MISSING')
    if value['object_identifiable'] == 'YES' and not value['object_quote'].strip():
        binding_errors.append('object_quote:MISSING')
    states = {name: applicability_status({check: row[check] for check in APPLICABILITY_CHECKS})
              for name, row in value['templates'].items()}
    contradictions = []
    if value['withdrawal_explicitly_requested'] != 'YES' and states['release_before_withdrawal'] == 'APPLICABLE':
        contradictions.append('WITHDRAWAL_TEMPLATE_WITHOUT_REQUESTED_WITHDRAWAL')
    if value['initial_object_state'] == 'HELD' and states['contact_before_transport'] == 'APPLICABLE':
        contradictions.append('NEW_PICKUP_TEMPLATE_FOR_ALREADY_HELD_OBJECT')
    required = ACTION_REQUIREMENTS[value['action_kind']]
    return {'template_applicability': dict.fromkeys(TEMPLATES, 'UNKNOWN') if binding_errors else states,
            'raw_template_applicability': states, 'binding_errors': binding_errors,
            'task_binding_valid': not binding_errors,
            'action_catalog_requirements': list(required),
            'missing_dedicated_templates': [name for name in required if name not in TEMPLATES],
            'contradictions': contradictions,
            'catalog_is_action_level_only': True,
            'video_failure_repairability_established': False}


def witness_quality(verdict):
    if verdict['status'] not in ('PASS', 'FAIL', 'UNKNOWN'):
        raise ValueError('Invalid original gate status')
    frames = verdict['frame_ids']
    valid = (isinstance(frames, list) and all(type(frame) is int and 0 <= frame <= 16
                                            for frame in frames))
    if not valid:
        return {'status': 'INVALID_FRAME_REFERENCE', 'generated_frame_count': 0}
    if len(frames) != len(set(frames)):
        return {'status': 'DUPLICATE_FRAME_REFERENCE', 'generated_frame_count': len(set(frames) - {0})}
    if verdict['status'] != 'UNKNOWN' and (not frames or not verdict.get('reason', '').strip()):
        return {'status': 'MISSING_EVIDENCE', 'generated_frame_count': len(set(frames) - {0})}
    generated = len(set(frames) - {0})
    return {'status': 'GENERATED_FRAMES_CITED' if generated else 'NO_GENERATED_FRAME_CITED',
            'generated_frame_count': generated,
            'initial_frame_cited': 0 in frames,
            'semantic_validity_established': False}


def diagnose_input(original, replay, applicability):
    if set(original['gates']) != set(GATES) or set(replay['gates']) != set(GATES):
        raise ValueError('Both observations must retain all five original gates')
    if set(applicability) != set(TEMPLATES):
        raise ValueError('Every frozen relation template must be assessed')
    allowed = ('APPLICABLE', 'NOT_APPLICABLE', 'UNKNOWN')
    if any(value not in allowed for value in applicability.values()):
        raise ValueError('Invalid template applicability state')
    comparisons = {}
    for gate in GATES:
        old = original['gates'][gate]
        new = replay['gates'][gate]
        old_witness = witness_quality(old)
        new_witness = witness_quality(new)
        applicable = [name for name, topics in TEMPLATE_GATE_TOPICS.items()
                      if gate in topics and applicability[name] == 'APPLICABLE']
        uncertain = [name for name, topics in TEMPLATE_GATE_TOPICS.items()
                     if gate in topics and applicability[name] == 'UNKNOWN']
        if old['status'] != 'FAIL':
            diagnosis = 'NEW_REPORTED_FAILURE' if new['status'] == 'FAIL' else 'NO_OLD_FAILURE'
        elif old_witness['status'] != 'GENERATED_FRAMES_CITED':
            diagnosis = 'OLD_FAILURE_WITHOUT_USABLE_GENERATED_WITNESS'
        elif new['status'] != 'FAIL':
            diagnosis = 'OLD_FAILURE_NOT_REPRODUCED'
        elif new_witness['status'] != 'GENERATED_FRAMES_CITED':
            diagnosis = 'REPEATED_FAILURE_WITHOUT_USABLE_GENERATED_WITNESS'
        elif applicable:
            diagnosis = 'REPEATED_FAILURE_WITH_TEMPLATE_TOPIC'
        elif uncertain:
            diagnosis = 'REPEATED_FAILURE_APPLICABILITY_UNKNOWN'
        else:
            diagnosis = 'REPEATED_FAILURE_OUTSIDE_TEMPLATE_SCOPE'
        comparisons[gate] = {
            'original_status': old['status'], 'replay_status': new['status'],
            'status_agrees': old['status'] == new['status'],
            'original_witness': old_witness, 'replay_witness': new_witness,
            'diagnosis': diagnosis, 'applicable_template_topics': applicable,
            'uncertain_template_topics': uncertain,
            'exact_failure_repairability_established': False,
        }
    return {'gates': comparisons, 'retained_original_audit': deepcopy(original),
            'human_ground_truth_available': False, 'population_accuracy_established': False,
            'deployment_decision_changed': False, 'calibrated_physical_success': False}