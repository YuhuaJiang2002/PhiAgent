"""Synthetic compiler contracts only; no benchmark data, media, or services."""
import copy
import base64
import hashlib
import io
from pathlib import Path
import tempfile
import unittest

from integrations.skilladam_ti2v.relational_repair import (
    EDIT_SLOT, EVIDENCE_SCHEMA, GROUNDING_CHECKS, INSTRUCTION_CHECKS, NEUTRAL_TEMPLATES,
    RELATION_SCHEMA, TEMPLATES,
    compile_instruction_repair, compile_repair, replace_slot, text_sha256,
    prepare_instruction_repair, prepare_relational_repair,
)
from integrations.skilladam_ti2v.residual_repair import PRIORITY
from scripts.run_ti2v_relational_repair import ARMS, distinct_prompt_groups, prompt_from_skill
from integrations.skilladam_ti2v.repair_diagnostics import (
    ACTION_REQUIREMENTS, applicability_status, diagnose_input, validate_task_diagnosis, witness_quality,
)
from scripts.verify_ti2v_repair_diagnostics import image_hashes, initial_binding_barriers
from scripts.verify_ti2v_repair_factorial import factorial_contrasts
from integrations.skilladam_ti2v.repair_factorial import (
    EXTENDED_TEMPLATES, GROUND_SCHEMA, actor_fields, grounding_schema, proposal_schema, select_proposal,
    prepare_factorial_repair, literal_relation_precondition,
)
from scripts.analyze_ti2v_fixed_comparison import load_rows


class RelationalRepairTests(unittest.TestCase):
    def setUp(self):
        self.instruction = 'The left arm picks up the cube and holds it.'
        self.skill = '# Reusable skill\nPreserve the literal task and camera.\n' + EDIT_SLOT + '\n'
        self.binding = {'video_sha256': 'a' * 64, 'initial_sha256': 'b' * 64,
                        'instruction_sha256': text_sha256(self.instruction)}
        self.audit = {'gates': {gate: {'status': 'PASS', 'frame_ids': [1, 16]}
                                for gate in PRIORITY}}
        self.audit['gates']['visible_action_order'] = {'status': 'FAIL', 'frame_ids': [4, 5]}
        self.proposal = {'decision': 'APPLY', 'relation_id': 'contact_before_transport',
                         'target_gate': 'visible_action_order', 'evidence_frame_ids': [4, 5],
                         'actor_quote': 'left arm', 'object_quote': 'cube'}
        self.checks = {key: {'status': 'PASS', 'reason': 'Synthetic explicit observation'}
                       for key in GROUNDING_CHECKS}

    def compile(self, **overrides):
        arguments = dict(instruction=self.instruction, source_binding=self.binding,
                         expected_binding=dict(self.binding), base_audit=self.audit,
                         positions=[index / 15 for index in range(16)], checks=self.checks)
        arguments.update(overrides)
        return compile_repair(self.skill, self.proposal, **arguments)

    def test_exactly_one_existing_line_changes(self):
        result = self.compile()
        self.assertEqual(result['status'], 'APPLIED')
        self.assertEqual(len(result['edits']), 1)
        self.assertEqual(result['skill'], self.skill.replace(EDIT_SLOT, result['residual']))
        self.assertTrue(result['skill'].endswith('\n'))
        self.assertEqual(result['normalized_interval'], [3 / 15, 4 / 15])
        self.assertFalse(result['physical_success_established'])

    def test_all_templates_are_bounded_and_deterministic(self):
        for relation in TEMPLATES:
            with self.subTest(relation=relation):
                self.proposal['relation_id'] = relation
                result = self.compile()
                self.assertEqual(result['residual'], TEMPLATES[relation])
                self.assertLessEqual(len(result['residual'].split()), 45)

    def test_unknown_or_failed_semantic_check_abstains(self):
        for check in GROUNDING_CHECKS:
            for status in ('UNKNOWN', 'FAIL'):
                with self.subTest(check=check, status=status):
                    checks = copy.deepcopy(self.checks)
                    checks[check]['status'] = status
                    result = self.compile(checks=checks)
                    self.assertEqual(result['status'], 'ABSTAINED')
                    self.assertEqual(result['skill'], self.skill)

    def test_incomplete_checks_abstain(self):
        checks = dict(self.checks)
        checks.pop('failure_visible')
        self.assertEqual(self.compile(checks=checks)['status'], 'ABSTAINED')

    def test_new_vocabulary_requires_explicit_binding_and_keeps_all_guards(self):
        self.proposal['relation_id'] = 'synthetic_relation'
        templates = {'synthetic_relation': 'Preserve the relation required by the literal task.'}
        with self.assertRaises(ValueError):
            self.compile()
        result = self.compile(templates=templates)
        self.assertEqual(result['status'], 'APPLIED')
        self.assertEqual(result['residual'], templates['synthetic_relation'])
        self.assertNotIn('synthetic_relation', TEMPLATES)
        for name in GROUNDING_CHECKS:
            for verdict in ('UNKNOWN', 'FAIL'):
                with self.subTest(name=name, verdict=verdict):
                    checks = copy.deepcopy(self.checks)
                    checks[name]['status'] = verdict
                    self.assertEqual(self.compile(templates=templates, checks=checks)['status'],
                                     'ABSTAINED')

    def test_source_or_instruction_mismatch_rejected(self):
        for key in self.binding:
            with self.subTest(key=key):
                binding = dict(self.binding)
                binding[key] = 'c' * 64
                with self.assertRaises(ValueError):
                    self.compile(source_binding=binding)
        with self.assertRaises(ValueError):
            self.compile(instruction='A different literal task.')

    def test_unspecified_actor_requires_explicit_new_protocol_and_all_checks(self):
        self.proposal['actor_quote'] = ''
        with self.assertRaises(ValueError):
            self.compile()
        self.assertEqual(self.compile(actor_unspecified=True)['status'], 'APPLIED')
        self.checks['entity_binding_valid']['status'] = 'UNKNOWN'
        self.assertEqual(self.compile(actor_unspecified=True)['status'], 'ABSTAINED')

    def test_unknown_or_passing_gate_cannot_be_repaired(self):
        for status in ('UNKNOWN', 'PASS'):
            self.audit['gates']['visible_action_order']['status'] = status
            with self.assertRaises(ValueError):
                self.compile()

    def test_unobserved_initial_duplicate_and_noninteger_frames_rejected(self):
        for frames in ([], [0], [17], [4, 4], [4.0], [True], [1, 16]):
            with self.subTest(frames=frames):
                self.proposal['evidence_frame_ids'] = frames
                with self.assertRaises(ValueError):
                    self.compile()

    def test_invalid_sampling_rejected(self):
        for positions in ([0.0] * 16, [index / 15 for index in range(15)],
                          [float('nan')] + [index / 15 for index in range(1, 16)]):
            with self.assertRaises(ValueError):
                self.compile(positions=positions)

    def test_unbound_entity_or_unknown_template_rejected(self):
        for field, value in (('actor_quote', 'right arm'), ('object_quote', 'red sphere'),
                             ('relation_id', 'invent_release')):
            with self.subTest(field=field):
                original = self.proposal[field]
                self.proposal[field] = value
                with self.assertRaises(ValueError):
                    self.compile()
                self.proposal[field] = original

    def test_missing_or_duplicate_edit_slot_abstains(self):
        for skill in ('# Skill\nPreserve the camera.', EDIT_SLOT + '\n' + EDIT_SLOT):
            self.skill = skill
            result = self.compile()
            self.assertEqual(result['reason'], 'NO_UNIQUE_EDIT_SLOT')
            self.assertEqual(result['skill'], skill)

    def test_proposer_abstention_never_changes_skill(self):
        self.proposal = {'decision': 'ABSTAIN'}
        result = self.compile()
        self.assertEqual(result['reason'], 'PROPOSER_ABSTAINED')
        self.assertEqual(result['skill'], self.skill)

    def test_neutral_controls_have_exact_template_word_lengths(self):
        self.assertEqual(set(NEUTRAL_TEMPLATES), set(TEMPLATES))
        for relation, template in TEMPLATES.items():
            with self.subTest(relation=relation):
                self.assertEqual(len(template.split()), len(NEUTRAL_TEMPLATES[relation].split()))

    def test_decision_is_emitted_after_evidence_and_reason(self):
        for schema in (RELATION_SCHEMA, EVIDENCE_SCHEMA['properties']['assessments']['items']):
            self.assertEqual(list(schema['properties'])[-1], 'decision')
            self.assertEqual(schema['required'][-1], 'decision')
            self.assertEqual(schema['properties']['decision']['enum'], ['APPLY', 'ABSTAIN'])
            self.assertEqual(schema['properties']['relation_id']['enum'], ['none', *TEMPLATES])

    def test_instruction_control_never_fabricates_failure_evidence(self):
        checks = {key: self.checks[key] for key in INSTRUCTION_CHECKS}
        result = compile_instruction_repair(self.skill, self.proposal,
                                           instruction=self.instruction, checks=checks)
        self.assertEqual(result['status'], 'APPLIED')
        self.assertFalse(result['video_evidence_used'])
        self.assertNotIn('evidence_frame_ids', result)
        checks['relation_applicable'] = {'status': 'UNKNOWN'}
        result = compile_instruction_repair(self.skill, self.proposal,
                                           instruction=self.instruction, checks=checks)
        self.assertEqual(result['status'], 'ABSTAINED')

    def test_multiline_or_oversized_replacement_rejected(self):
        for replacement in ('first\nsecond', 'first\rsecond', 'word ' * 46):
            with self.assertRaises(ValueError):
                replace_slot(self.skill, replacement)

    def test_prompt_groups_reuse_only_exact_prompt_identity(self):
        prompts = dict.fromkeys(ARMS, prompt_from_skill(self.instruction, self.skill))
        self.assertEqual(len(distinct_prompt_groups(prompts)), 1)
        prompts['structured'] += ' '
        self.assertEqual(len(distinct_prompt_groups(prompts)), 2)
        self.assertTrue(prompts['parent'].startswith(self.instruction + '\n'))
        prompts.pop('neutral')
        with self.assertRaises(ValueError):
            distinct_prompt_groups(prompts)

    def test_instruction_adapter_never_receives_video_or_audit(self):
        calls = []

        class SyntheticBackend:
            def visual_query(self, prompt, images, schema, stage, seed):
                calls.append((prompt, images))
                return {'decision': 'ABSTAIN', 'relation_id': 'none', 'actor_quote': '',
                        'object_quote': '', 'reason': 'Synthetic abstention'}, {}

        with tempfile.TemporaryDirectory() as folder:
            row = {'instruction': self.instruction, 'initial': 'synthetic-initial', 'seed': 7}
            result = prepare_instruction_repair(SyntheticBackend(), row, self.skill, Path(folder))
        self.assertEqual(result['status'], 'ABSTAINED')
        self.assertEqual(calls[0][1], ['synthetic-initial'])
        self.assertEqual(len(calls), 1)

    def test_missing_template_is_invalid_without_a_guard_retry(self):
        proposal = {**self.proposal, 'relation_id': 'none'}
        calls = []

        class SyntheticBackend:
            def frames(self, video, folder):
                return ['synthetic-frame'] * 16, [index / 15 for index in range(16)]

            def visual_query(self, prompt, images, schema, stage, seed, positions):
                calls.append(stage)
                return {'assessments': [proposal]}, {}

        row = {'instruction': self.instruction, 'initial': 'synthetic-initial',
               'base': 'synthetic-video', 'initial_sha256': 'b' * 64,
               'base_sha256': 'a' * 64, 'seed': 7}
        with tempfile.TemporaryDirectory() as folder:
            result = prepare_relational_repair(SyntheticBackend(), row, self.audit,
                                               self.skill, Path(folder))
        self.assertEqual(result['reason'], 'INVALID_PROPOSAL')
        self.assertEqual(calls, ['relation-evidence'])


class RepairFactorialTests(unittest.TestCase):
    def setUp(self):
        self.instruction = 'Place the cube on the table.'
        self.first = {'actor_quote': '', 'actor_binding': 'TASK_UNSPECIFIED',
                      'object_quote': 'cube', 'binding_observation': 'Synthetic observation',
                      'binding_status': 'PASS', 'assessments': [{
                          'target_gate': 'visible_action_order', 'evidence_frame_ids': [4, 5],
                          'visible_observation': 'Synthetic release before support', 'failure_visible': 'PASS'}]}
        self.second = copy.deepcopy(self.first)
        self.second['assessments'][0].update(relation_id='support_before_release',
                                            relation_reason='Synthetic applicable relation', decision='APPLY')
        self.failed = {'visible_action_order': {'status': 'FAIL', 'frame_ids': [4, 5]}}

    def select(self):
        return select_proposal(self.first, self.second, self.instruction, self.failed, EXTENDED_TEMPLATES)

    def test_new_templates_do_not_mutate_historical_vocabulary(self):
        self.assertEqual(len(TEMPLATES), 3)
        self.assertEqual(len(EXTENDED_TEMPLATES), 6)
        self.assertLessEqual(max(len(template.split()) for template in EXTENDED_TEMPLATES.values()), 45)
        self.assertNotIn('relation_id', GROUND_SCHEMA['properties']['assessments']['items']['properties'])
        for vocabulary in (TEMPLATES, EXTENDED_TEMPLATES):
            fields = proposal_schema(vocabulary)['properties']['assessments']['items']['properties']
            self.assertEqual(fields['relation_id']['enum'], ['none', *vocabulary])
            self.assertEqual(list(fields)[-1], 'decision')

    def test_unnamed_actor_is_preserved_without_adding_an_arm(self):
        proposal, reason = self.select()
        self.assertEqual(reason, 'PROPOSED')
        self.assertEqual(proposal['actor_quote'], '')
        self.second['actor_quote'] = 'left arm'
        with self.assertRaises(ValueError):
            self.select()

    def test_actor_schema_cannot_emit_contradictory_binding(self):
        for instruction, quotes, binding in (
            ('Place the cube on the table.', [''], 'TASK_UNSPECIFIED'),
            ('Pick up the cube with right arm.', ['right arm'], 'LITERAL_SPAN'),
            ('Pass the cube from left arm to right arm.', ['left arm', 'right arm'], 'LITERAL_SPAN'),
        ):
            with self.subTest(instruction=instruction):
                for schema in (grounding_schema(instruction), proposal_schema(TEMPLATES, instruction),
                               proposal_schema(EXTENDED_TEMPLATES, instruction)):
                    self.assertEqual(schema['properties']['actor_quote']['enum'], quotes)
                    self.assertEqual(schema['properties']['actor_binding']['enum'], [binding])
                    self.assertEqual(schema['properties']['binding_status'],
                                     {'type': 'string', 'enum': ['PASS', 'FAIL', 'UNKNOWN']})
        self.assertNotIn('enum', actor_fields()['actor_quote'])

    def test_later_answer_cannot_override_missing_first_stage_evidence(self):
        for verdict in ('FAIL', 'UNKNOWN'):
            self.first['assessments'][0]['failure_visible'] = verdict
            self.assertEqual(self.select(), (None, 'FAILURE_NOT_GROUNDED'))
        self.first['binding_status'] = 'UNKNOWN'
        self.assertEqual(self.select(), (None, 'ENTITY_UNRESOLVED'))

    def test_duplicate_or_new_witness_is_not_accepted(self):
        for frames in ([4, 4], [1], [True]):
            self.second['assessments'][0]['evidence_frame_ids'] = frames
            with self.assertRaises(ValueError):
                self.select()

    def test_original_arm_cannot_use_an_extended_relation(self):
        with self.assertRaises(ValueError):
            select_proposal(self.first, self.second, self.instruction, self.failed, TEMPLATES)

    def test_literal_action_preconditions_keep_unrequested_actions_out(self):
        self.assertEqual(literal_relation_precondition('Push open the door.', 'contact_before_transport'), 'FAIL')
        self.assertEqual(literal_relation_precondition('Push open the door.', 'maintain_contact_during_push'), 'PASS')
        self.assertEqual(literal_relation_precondition('Place the cube on the table.', 'release_before_withdrawal'), 'UNKNOWN')
        self.assertEqual(literal_relation_precondition('Place the cube on the table and withdraw the gripper.', 'release_before_withdrawal'), 'PASS')
        self.assertEqual(literal_relation_precondition('Place the cube; do not retract the gripper.', 'release_before_withdrawal'), 'FAIL')
        self.assertEqual(literal_relation_precondition('Place the cube on the table.', 'support_before_release'), 'PASS')
        self.assertEqual(literal_relation_precondition('Pick up the cube.', 'support_before_release'), 'FAIL')
        self.assertEqual(literal_relation_precondition('Pass the object to the right arm.', 'receiver_grasp_before_giver_release'), 'PASS')
        self.assertEqual(literal_relation_precondition('Turn the object.', 'maintain_grasp'), 'UNKNOWN')

    def test_factored_adapter_uses_two_frozen_stages_and_unchanged_guard(self):
        calls = []
        values = [self.first, self.second,
                  {name: {'status': 'PASS', 'reason': 'Synthetic'} for name in GROUNDING_CHECKS}]

        class SyntheticBackend:
            def visual_query(self, prompt, images, schema, stage, seed, positions):
                calls.append((stage, prompt, schema))
                return values[len(calls) - 1], {'total_tokens': 1}

        row = {'instruction': self.instruction, 'initial': 'synthetic-initial', 'seed': 7,
               'base_sha256': 'a' * 64, 'initial_sha256': 'b' * 64}
        audit = {'gates': {gate: {'status': 'PASS', 'frame_ids': [1, 16]} for gate in PRIORITY}}
        audit['gates'].update(self.failed)
        before = copy.deepcopy(audit)
        with tempfile.TemporaryDirectory() as folder:
            result = prepare_factorial_repair(SyntheticBackend(), row, audit, EDIT_SLOT,
                Path(folder), 'factored_extended', ['synthetic-frame'] * 16,
                [index / 15 for index in range(16)])
        self.assertEqual(result['status'], 'APPLIED')
        self.assertEqual(len(calls), 3)
        self.assertNotIn('FROZEN RELATIONS:', calls[0][1])
        self.assertEqual(set(calls[-1][2]['properties']), set(GROUNDING_CHECKS))
        self.assertEqual(audit, before)
        self.assertFalse(result['physical_success_established'])

    def test_paired_csv_parser_keeps_synthetic_case_trial_identity(self):
        header = 'task_id,episode_id,trial_id,BLEUScore,CLIPScore,hsd,dyn,ndtw\n'
        rows = [f'{case},1,{trial},0.1,0.2,0.3,0.4,0.5\n'
                for case in range(20) for trial in range(3)]
        parsed = load_rows(io.StringIO(header + ''.join(rows) + 'MEAN,,,,,,,\n'))
        self.assertEqual(len(parsed), 60)
        with self.assertRaises(ValueError):
            load_rows(io.StringIO(header + ''.join(rows + rows[:1])))
        with self.assertRaises(ValueError):
            load_rows(io.StringIO(header + ''.join(rows[:-1])))


class RepairDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.original = {'gates': {gate: {'status': 'PASS', 'frame_ids': [1, 16],
                                          'reason': 'Synthetic visible evidence'} for gate in PRIORITY}}
        self.original['gates']['visible_action_order']['status'] = 'FAIL'
        self.replay = copy.deepcopy(self.original)
        self.applicability = dict.fromkeys(TEMPLATES, 'NOT_APPLICABLE')

    def test_task_applicability_requires_both_supported_conditions(self):
        self.assertEqual(applicability_status({'required_by_literal_task': 'YES',
                                              'initial_state_compatible': 'YES'}), 'APPLICABLE')
        self.assertEqual(applicability_status({'required_by_literal_task': 'YES',
                                              'initial_state_compatible': 'UNKNOWN'}), 'UNKNOWN')
        self.assertEqual(applicability_status({'required_by_literal_task': 'NO',
                                              'initial_state_compatible': 'UNKNOWN'}), 'NOT_APPLICABLE')
        with self.assertRaises(ValueError):
            applicability_status({'required_by_literal_task': 'YES'})

    def test_repeatable_failure_does_not_imply_template_coverage(self):
        result = diagnose_input(self.original, self.replay, self.applicability)
        self.assertEqual(result['gates']['visible_action_order']['diagnosis'],
                         'REPEATED_FAILURE_OUTSIDE_TEMPLATE_SCOPE')
        self.assertFalse(result['population_accuracy_established'])

    def test_template_topic_is_not_a_claim_of_repairability(self):
        self.applicability['maintain_grasp'] = 'APPLICABLE'
        result = diagnose_input(self.original, self.replay, self.applicability)
        self.assertEqual(result['gates']['visible_action_order']['diagnosis'],
                         'REPEATED_FAILURE_WITH_TEMPLATE_TOPIC')
        self.assertFalse(result['gates']['visible_action_order']['exact_failure_repairability_established'])

    def test_camera_failure_remains_outside_grasp_templates(self):
        self.original['gates']['camera_preservation']['status'] = 'FAIL'
        self.replay = copy.deepcopy(self.original)
        result = diagnose_input(self.original, self.replay, dict.fromkeys(TEMPLATES, 'APPLICABLE'))
        self.assertEqual(result['gates']['camera_preservation']['diagnosis'],
                         'REPEATED_FAILURE_OUTSIDE_TEMPLATE_SCOPE')

    def test_shadow_pass_never_clears_old_failure(self):
        before = copy.deepcopy(self.original)
        self.replay['gates']['visible_action_order']['status'] = 'PASS'
        result = diagnose_input(self.original, self.replay, self.applicability)
        self.assertEqual(result['gates']['visible_action_order']['diagnosis'], 'OLD_FAILURE_NOT_REPRODUCED')
        self.assertEqual(result['retained_original_audit'], before)
        self.assertEqual(self.original, before)
        self.assertFalse(result['deployment_decision_changed'])

    def test_initial_only_evidence_is_preserved_but_not_video_motion_evidence(self):
        self.original['gates']['visible_action_order']['frame_ids'] = [0]
        result = diagnose_input(self.original, self.replay, self.applicability)
        self.assertEqual(result['gates']['visible_action_order']['diagnosis'],
                         'OLD_FAILURE_WITHOUT_USABLE_GENERATED_WITNESS')
        self.assertEqual(result['retained_original_audit']['gates']['visible_action_order']['frame_ids'], [0])

    def test_invalid_witness_references_are_explicit(self):
        for frames in ([17], [-1], [True]):
            verdict = {'status': 'FAIL', 'reason': 'Synthetic', 'frame_ids': frames}
            self.assertEqual(witness_quality(verdict)['status'], 'INVALID_FRAME_REFERENCE')
        verdict = {'status': 'FAIL', 'reason': 'Synthetic', 'frame_ids': [1, 1]}
        self.assertEqual(witness_quality(verdict)['status'], 'DUPLICATE_FRAME_REFERENCE')

    def test_action_catalog_does_not_hide_missing_contact_relations(self):
        self.assertIn('support_before_release', ACTION_REQUIREMENTS['placement'])
        self.assertIn('receiver_grasp_before_giver_release', ACTION_REQUIREMENTS['handover'])
        self.assertIn('maintain_contact_during_push', ACTION_REQUIREMENTS['push_open'])
        for missing in ('support_before_release', 'receiver_grasp_before_giver_release', 'maintain_contact_during_push'):
            self.assertNotIn(missing, TEMPLATES)

    def test_withdrawal_contradiction_is_recorded_not_silently_fixed(self):
        value = {'action_kind': 'placement', 'action_quote': 'Place', 'actor_quote': '',
                 'object_quote': 'cube', 'initial_object_state': 'HELD', 'object_identifiable': 'YES',
                 'withdrawal_explicitly_requested': 'NO', 'templates': {
                     name: {'required_by_literal_task': 'YES', 'initial_state_compatible': 'YES'} for name in TEMPLATES}}
        validated = validate_task_diagnosis(value, 'Place the cube on the table.')
        self.assertIn('WITHDRAWAL_TEMPLATE_WITHOUT_REQUESTED_WITHDRAWAL', validated['contradictions'])
        self.assertIn('NEW_PICKUP_TEMPLATE_FOR_ALREADY_HELD_OBJECT', validated['contradictions'])
        self.assertEqual(validated['template_applicability']['release_before_withdrawal'], 'APPLICABLE')
        self.assertIn('support_before_release', validated['missing_dedicated_templates'])
        self.assertFalse(validated['video_failure_repairability_established'])

    def test_hallucinated_actor_is_retained_as_invalid_not_admitted(self):
        value = {'action_kind': 'placement', 'action_quote': 'Place', 'actor_quote': 'right',
                 'object_quote': 'toast', 'initial_object_state': 'HELD', 'object_identifiable': 'YES',
                 'withdrawal_explicitly_requested': 'NO', 'templates': {
                     name: {'required_by_literal_task': 'YES', 'initial_state_compatible': 'YES'} for name in TEMPLATES}}
        before = copy.deepcopy(value)
        validated = validate_task_diagnosis(value, 'Place the toast on the plate.')
        self.assertFalse(validated['task_binding_valid'])
        self.assertIn('actor_quote:NOT_AN_EXACT_TASK_QUOTE', validated['binding_errors'])
        self.assertEqual(set(validated['template_applicability'].values()), {'UNKNOWN'})
        self.assertEqual(value, before)

    def test_transport_hashes_keep_image_order_and_reject_external_urls(self):
        payloads = [b'synthetic first image', b'synthetic generated image']
        content = [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,'
                    + base64.b64encode(payload).decode()}} for payload in payloads]
        self.assertEqual(image_hashes(content), [hashlib.sha256(payload).hexdigest() for payload in payloads])
        content[0]['image_url']['url'] = 'https://example.invalid/image.png'
        with self.assertRaises(ValueError):
            image_hashes(content)

    def test_topic_applicability_does_not_override_unidentified_target(self):
        task = {'value': {'object_identifiable': 'NO', 'initial_object_state': 'NOT_IDENTIFIABLE'},
                'validated': {'task_binding_valid': True, 'contradictions': []}}
        self.assertEqual(initial_binding_barriers(task),
                         ['TARGET_NOT_IDENTIFIED', 'INITIAL_OBJECT_STATE_UNRESOLVED'])
        task['value'].update(object_identifiable='YES', initial_object_state='HELD')
        self.assertEqual(initial_binding_barriers(task), [])
        task['validated']['task_binding_valid'] = False
        self.assertEqual(initial_binding_barriers(task), ['INVALID_LITERAL_BINDING'])


class FactorialPriorityTests(unittest.TestCase):
    def test_paired_factorial_contrasts_separate_both_changes(self):
        values = {'combined_original': 1, 'factored_original': 3,
                  'combined_extended': 4, 'factored_extended': 8}
        self.assertEqual(factorial_contrasts(values), {
            'separation_with_original': 2, 'separation_with_extended': 4,
            'extension_with_combined': 3, 'extension_with_factored': 5,
            'average_separation': 3, 'average_extension': 4, 'interaction': 2,
        })
        with self.assertRaises(ValueError):
            factorial_contrasts({'combined_original': 1})

    def test_frozen_priority_keeps_first_proposal_unknown_as_a_veto(self):
        first = {
            'actor_quote': '', 'actor_binding': 'TASK_UNSPECIFIED', 'object_quote': 'cube',
            'binding_observation': 'Synthetic visible cube', 'binding_status': 'PASS',
            'assessments': [
                {'target_gate': 'task_completion', 'evidence_frame_ids': [4, 5],
                 'visible_observation': 'Synthetic occlusion', 'failure_visible': 'UNKNOWN'},
                {'target_gate': 'visible_action_order', 'evidence_frame_ids': [4, 5],
                 'visible_observation': 'Synthetic early release', 'failure_visible': 'PASS'},
            ],
        }
        second = copy.deepcopy(first)
        for entry in second['assessments']:
            entry.update(failure_visible='PASS', relation_reason='Synthetic task-required relation',
                         relation_id='support_before_release', decision='APPLY')
        failed = {entry['target_gate']: {'status': 'FAIL', 'frame_ids': [4, 5]}
                  for entry in first['assessments']}
        before = copy.deepcopy(first)
        proposal, reason = select_proposal(first, second, 'Place the cube on the table.', failed,
                                          EXTENDED_TEMPLATES)
        self.assertEqual(reason, 'FAILURE_NOT_GROUNDED')
        self.assertIsNone(proposal)
        self.assertEqual(first, before)
        second['assessments'][0]['decision'] = 'ABSTAIN'
        proposal, reason = select_proposal(first, second, 'Place the cube on the table.', failed,
                                          EXTENDED_TEMPLATES)
        self.assertEqual(reason, 'PROPOSED')
        self.assertEqual(proposal['target_gate'], 'visible_action_order')


if __name__ == '__main__':
    unittest.main()