"""Bounded relation-to-skill compilation without model or benchmark execution."""
import hashlib
import json
import math

from .backend import TEXT, obj, save
from .residual_repair import CHECKS, PRIORITY, guard_accepts


TEMPLATES = {
    'contact_before_transport': (
        'When picking up an initially supported object, establish gripper contact '
        'before lifting or transporting the object.'
    ),
    'maintain_grasp': (
        'When carrying a grasped object, maintain the grasp so the object moves '
        'together with its holding gripper.'
    ),
    'release_before_withdrawal': (
        'Only when placement, release and gripper withdrawal are required, keep '
        'the object supported while releasing it before withdrawing the gripper.'
    ),
}
EDIT_SLOT = 'Follow the literal task without adding another action.'
GROUNDING_CHECKS = CHECKS + ('relation_applicable', 'failure_visible', 'entity_binding_valid')
INSTRUCTION_CHECKS = CHECKS + ('relation_applicable', 'entity_binding_valid')
NEUTRAL_TEMPLATES = {
    'contact_before_transport': (
        'Follow the literal task while preserving the given initial image, camera, '
        'robot and existing object identities.'
    ),
    'maintain_grasp': (
        'Follow the literal task while preserving the original given initial image, '
        'camera, robot and existing object identities.'
    ),
    'release_before_withdrawal': (
        'Follow the literal task while preserving the given initial image, camera, '
        'robot and existing object identities throughout the entire video.'
    ),
}
RELATION_SCHEMA = obj({
    'reason': TEXT,
    'actor_quote': TEXT,
    'object_quote': TEXT,
    'relation_id': {'type': 'string', 'enum': ['none', *TEMPLATES]},
    'decision': {'type': 'string', 'enum': ['APPLY', 'ABSTAIN']},
})
EVIDENCE_SCHEMA = obj({
    'assessments': {
        'type': 'array', 'maxItems': 5,
        'items': obj({
            'target_gate': {'type': 'string', 'enum': list(PRIORITY)},
            'evidence_frame_ids': {'type': 'array', 'maxItems': 16,
                                   'items': {'type': 'integer', 'minimum': 1, 'maximum': 16}},
            **RELATION_SCHEMA['properties'],
        }),
    },
})


def check_schema(names):
    return obj({name: obj({'status': {'type': 'string', 'enum': ['PASS', 'FAIL', 'UNKNOWN']},
                           'reason': TEXT}) for name in names})


def replace_slot(skill, replacement):
    if len(replacement.split()) > 45 or '\n' in replacement or '\r' in replacement:
        raise ValueError('Replacement must be one line of at most 45 words')
    lines = skill.splitlines()
    if lines.count(EDIT_SLOT) != 1:
        raise ValueError('Require one predeclared skill edit slot')
    lines[lines.index(EDIT_SLOT)] = replacement
    return '\n'.join(lines) + ('\n' if skill.endswith('\n') else '')


def compile_instruction_repair(skill, proposal, *, instruction, checks):
    result = {'status': 'ABSTAINED', 'skill': skill, 'residual': '', 'edits': [],
              'reason': 'PROPOSER_ABSTAINED', 'video_evidence_used': False,
              'physical_success_established': False}
    if proposal['decision'] == 'ABSTAIN':
        return result
    if proposal['decision'] != 'APPLY' or proposal['relation_id'] not in TEMPLATES:
        raise ValueError('Unknown instruction-only relation')
    for role in ('actor_quote', 'object_quote'):
        quote = proposal[role]
        if not quote.strip() or quote not in instruction:
            raise ValueError('Instruction-only entity is not grounded in the task')
    if set(checks) != set(INSTRUCTION_CHECKS) or any(
            checks[name]['status'] != 'PASS' for name in INSTRUCTION_CHECKS):
        result['reason'] = 'GROUNDING_REJECTED'
        return result
    replacement = TEMPLATES[proposal['relation_id']]
    result.update(status='APPLIED', reason='INSTRUCTION_GROUNDING_PASSED',
                  skill=replace_slot(skill, replacement), residual=replacement,
                  relation_id=proposal['relation_id'],
                  edits=[{'old': EDIT_SLOT, 'new': replacement}])
    return result


def text_sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()


def compile_repair(skill, proposal, *, instruction, source_binding,
                   expected_binding, base_audit, positions, checks,
                   templates=None, actor_unspecified=False):
    """Compile one frozen template or abstain; invalid evidence raises ValueError."""
    templates = TEMPLATES if templates is None else templates
    if not instruction.strip():
        raise ValueError('Missing literal instruction')
    keys = {'video_sha256', 'initial_sha256', 'instruction_sha256'}
    if set(source_binding) != keys or source_binding != expected_binding:
        raise ValueError('Repair evidence does not bind the expected source')
    if source_binding['instruction_sha256'] != text_sha256(instruction):
        raise ValueError('Repair evidence belongs to a different instruction')
    for digest in source_binding.values():
        if not isinstance(digest, str) or len(digest) != 64 or any(
                character not in '0123456789abcdef' for character in digest):
            raise ValueError('Invalid source SHA-256')
    result = {'status': 'ABSTAINED', 'residual': '', 'skill': skill,
              'edits': [], 'source_binding': dict(source_binding),
              'physical_success_established': False}
    if proposal['decision'] == 'ABSTAIN':
        result['reason'] = 'PROPOSER_ABSTAINED'
        return result
    if proposal['decision'] != 'APPLY' or proposal['relation_id'] not in templates:
        raise ValueError('Unknown decision or relation template')
    gate = proposal['target_gate']
    if gate not in PRIORITY or base_audit['gates'][gate]['status'] != 'FAIL':
        raise ValueError('A relational repair requires an explicit source failure')
    if len(positions) != 16 or any(not math.isfinite(position) for position in positions):
        raise ValueError('Require sixteen finite normalized frame positions')
    if positions[0] != 0 or positions[-1] != 1 or any(
            left >= right for left, right in zip(positions, positions[1:])):
        raise ValueError('Frame positions must be strictly increasing with endpoints')
    frames = proposal['evidence_frame_ids']
    if not frames or any(type(frame) is not int or not 1 <= frame <= 16 for frame in frames):
        raise ValueError('Evidence must cite generated frames 1 through 16')
    if len(frames) != len(set(frames)):
        raise ValueError('Duplicate evidence frame identifiers')
    if not set(frames) <= set(base_audit['gates'][gate]['frame_ids']):
        raise ValueError('Evidence is outside the supplied failure')
    for role in ('actor_quote', 'object_quote'):
        quote = proposal[role]
        if role == 'actor_quote' and actor_unspecified and quote == '':
            continue
        if not isinstance(quote, str) or not quote.strip() or quote not in instruction:
            raise ValueError('Entity roles must bind exact literal-task spans')
    if set(checks) != set(GROUNDING_CHECKS) or not guard_accepts(
            {key: checks[key] for key in CHECKS}):
        result['reason'] = 'GROUNDING_REJECTED'
        return result
    if any(checks[key]['status'] != 'PASS' for key in GROUNDING_CHECKS):
        result['reason'] = 'GROUNDING_REJECTED'
        return result
    lines = skill.splitlines()
    if lines.count(EDIT_SLOT) != 1:
        result['reason'] = 'NO_UNIQUE_EDIT_SLOT'
        return result
    replacement = templates[proposal['relation_id']]
    if len(replacement.split()) > 45:
        raise ValueError('Frozen template exceeds residual budget')
    patched = replace_slot(skill, replacement)
    result.update(status='APPLIED', reason='EVIDENCE_AND_GROUNDING_PASSED',
                  residual=replacement, skill=patched,
                  relation_id=proposal['relation_id'], target_gate=gate,
                  evidence_frame_ids=sorted(frames),
                  normalized_interval=[positions[min(frames) - 1], positions[max(frames) - 1]],
                  edits=[{'old': EDIT_SLOT, 'new': replacement}])
    return result


def prepare_relational_repair(backend, row, base_audit, skill, folder):
    """Two calls at most; the gate order, templates and selector are fixed."""
    binding = {'video_sha256': row['base_sha256'], 'initial_sha256': row['initial_sha256'],
               'instruction_sha256': text_sha256(row['instruction'])}
    result = {'status': 'ABSTAINED', 'reason': 'NO_REPORTED_FAILURE', 'skill': skill,
              'residual': '', 'usage': [], 'source_binding': binding, 'edits': [],
              'physical_success_established': False}
    failed = {gate: base_audit['gates'][gate] for gate in PRIORITY
              if base_audit['gates'][gate]['status'] == 'FAIL'}
    if failed:
        images, positions = backend.frames(row['base'], folder / 'evidence-frames')
        prompt = (
            'Inspect the real initial image and sixteen generated frames for the literal task. '
            'For EACH supplied failed gate, in the supplied order, return one assessment. '
            'A reported failure can be mistaken: ABSTAIN unless its violation is visibly supported '
            'and one listed relation is required by the literal task. Choose only a listed relation. '
            'actor_quote and object_quote must be exact task substrings. Cite only generated frame '
            'IDs already cited by that gate. Hidden or occluded contact is UNKNOWN: abstain. '
            'Do not infer forces or reference motion. An initially held object does not need a new '
            'grasp. Carrying alone does not require release or withdrawal. Do not add actions, '
            'attributes, directions, timings, or camera changes. No score is available.\n'
            'TEMPLATES:\n' + json.dumps(TEMPLATES) + '\nFAILED GATES:\n' + json.dumps(failed)
            + '\nLITERAL TASK:\n' + row['instruction'])
        assessments, usage = backend.visual_query(prompt, [row['initial']] + images,
            EVIDENCE_SCHEMA, 'relation-evidence', row['seed'], positions)
        result['usage'].append(usage)
        result['assessments'] = assessments
        proposals = assessments['assessments']
        if [proposal['target_gate'] for proposal in proposals] != list(failed):
            result.update(reason='INVALID_PROPOSAL', error='Incomplete or reordered failed-gate assessments')
        else:
            proposal = next((proposal for proposal in proposals if proposal['decision'] == 'APPLY'), None)
            result['reason'] = 'PROPOSER_ABSTAINED'
            if proposal is not None and proposal['relation_id'] not in TEMPLATES:
                result.update(reason='INVALID_PROPOSAL', error='Missing relation template')
            elif proposal is not None:
                relation = TEMPLATES[proposal['relation_id']]
                guard_prompt = (
                    'Verify this proposed relation against the literal task, initial image and '
                    'generated frames. Do not repair or rewrite it. Each check is PASS only with '
                    'visible support; FAIL or UNKNOWN rejects the edit. Check task preservation, '
                    'necessary actions only, initial-image consistency, absence of extra spatial '
                    'or timing constraints, relation applicability, whether the cited source failure '
                    'is actually visible, and the actor/object binding. Pickup ends held; release '
                    'and withdrawal must be explicitly required, never added by a template. '
                    'The action interval is evidence, not a target timing.\nPROPOSAL:\n'
                    + json.dumps(proposal) + '\nRELATION:\n' + relation
                    + '\nLITERAL TASK:\n' + row['instruction'])
                checks, usage = backend.visual_query(guard_prompt, [row['initial']] + images,
                    check_schema(GROUNDING_CHECKS), 'relation-grounding', row['seed'], positions)
                result['usage'].append(usage)
                result['guard'] = checks
                try:
                    result.update(compile_repair(skill, proposal, instruction=row['instruction'],
                        source_binding=binding, expected_binding=binding, base_audit=base_audit,
                        positions=positions, checks=checks))
                except ValueError as error:
                    result.update(reason='INVALID_PROPOSAL', error=str(error))
    save(folder / 'relation-plan.json', result)
    return result


def prepare_instruction_repair(backend, row, skill, folder):
    """A task-only control with no generated video, gate report, or score input."""
    prompt = (
        'Choose at most one listed relation required by the literal task and real initial image. '
        'Return ABSTAIN when none is applicable or initial identity/contact is unclear. '
        'You have no generated video or failure report. Do not guess them. actor_quote and '
        'object_quote must be exact task substrings. Pickup ends holding. Do not add a new '
        'grasp when already held, or release/withdrawal absent from the task.\nTEMPLATES:\n'
        + json.dumps(TEMPLATES) + '\nLITERAL TASK:\n' + row['instruction'])
    proposal, usage = backend.visual_query(prompt, [row['initial']], RELATION_SCHEMA,
                                           'instruction-relation', row['seed'])
    result = {'status': 'ABSTAINED', 'reason': 'PROPOSER_ABSTAINED', 'skill': skill,
              'residual': '', 'edits': [], 'usage': [usage], 'proposal': proposal,
              'video_evidence_used': False, 'physical_success_established': False}
    if proposal['decision'] == 'APPLY':
        if proposal['relation_id'] not in TEMPLATES:
            result.update(reason='INVALID_PROPOSAL', error='Missing relation template')
        else:
            prompt = (
                'Check instruction fidelity and entity binding for the proposed relation. '
                'Only the literal task and real initial image are available. PASS each check '
                'only if supported; otherwise FAIL or UNKNOWN. No new actions, attributes, '
                'direction or timing. A pickup ends held; carrying does not imply release.\n'
                'PROPOSAL:\n' + json.dumps(proposal) + '\nRELATION:\n'
                + TEMPLATES[proposal['relation_id']] + '\nLITERAL TASK:\n' + row['instruction'])
            checks, usage = backend.visual_query(prompt, [row['initial']],
                check_schema(INSTRUCTION_CHECKS), 'instruction-grounding', row['seed'])
            result['usage'].append(usage)
            result['guard'] = checks
            try:
                result.update(compile_instruction_repair(skill, proposal,
                    instruction=row['instruction'], checks=checks))
            except ValueError as error:
                result.update(reason='INVALID_PROPOSAL', error=str(error))
    save(folder / 'instruction-plan.json', result)
    return result