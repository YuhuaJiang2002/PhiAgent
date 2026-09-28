"""Synthetic state-machine tests, with no datasets, models, or evaluator calls."""
import unittest
from dataclasses import replace
from phiagent.harness.video_meta_rsi import (
    MetaRSI, Partition, ScoreCard, GATES, text_hash, validate_patch, validate_policy,
)
from scripts.run_ti2v_skill_optimization import bounded_edit

BASE = '# Skill\nKeep identity.\nUse the given instruction.\n'
CONTROL = BASE.replace('Use the given instruction.', 'Preserve visible action order.')
PROPOSED = BASE.replace('Use the given instruction.', 'Ground each task phase in visible evidence.')
POLICY = 'Propose a small task-general skill correction.'
NEW_POLICY = 'Diagnose the recurring failure before proposing a task-general correction.'
HASH = 'a' * 64


def proposal(**fields):
    return dict(hypothesis='A recurring visible failure.', expected_effect='A reusable correction.',
                risk='The change may not transfer.', **fields)


class Fixture:
    def __init__(self, scores=None):
        self.parts = [Partition(name, ((name + '-case', 7),))
                      for name in ('train', 'validation', 'meta_validation', 'test')]
        self.scores = scores or {BASE: (1,) * 5, CONTROL: (2,) * 5, PROPOSED: (3,) * 5}
        self.log, self.proposer_inputs, self.measured = {}, [], []
        self.observed = []

    def observe(self, skill, part, stage):
        self.observed.append(skill)
        return [{'case_id': c, 'seed': s, 'skill_sha256': text_hash(skill), 'video_sha256': HASH, 'trace': 'Synthetic action-order observation.',
                 'gates': {g: 'FAIL' for g in GATES}} for c, s in part.members]

    def propose_policy(self, policy, signal, stage):
        self.proposer_inputs.append(signal)
        return proposal(policy=NEW_POLICY)

    def propose_skill(self, policy, skill, signal, stage):
        self.proposer_inputs.append(signal)
        line = 'Preserve visible action order.' if policy == POLICY else 'Ground each task phase in visible evidence.'
        return proposal(edits=[{'old': 'Use the given instruction.', 'new': line}])

    def evaluate(self, skill, part, stage):
        self.measured.append(part.name)
        values = self.scores.get((skill, part.name), self.scores.get(skill, (1,) * 5))
        return ScoreCard(text_hash(skill), part.sha256, HASH, HASH,
                         tuple((c, s, values) for c, s in part.members))

    def controller(self, **kw):
        args = dict(observe=self.observe, propose_policy=self.propose_policy, propose_skill=self.propose_skill,
                    apply_patch=bounded_edit, evaluate=self.evaluate, record=lambda k,v: self.log.__setitem__(k,v))
        args.update(kw)
        return MetaRSI(self.parts, HASH, **args)


class MetaRSIContracts(unittest.TestCase):
    def test_promotes_better_policy_and_child(self):
        f=Fixture(); r=f.controller().run(BASE, POLICY, terms=1)
        self.assertEqual((r['skill'], r['policy']), (PROPOSED, NEW_POLICY))
        self.assertFalse(r['final_test_evaluated'])
        self.assertNotIn('test', f.measured)

    def test_policy_must_beat_optimized_control_not_just_parent(self):
        f=Fixture({BASE:(1,)*5, CONTROL:(3,)*5, PROPOSED:(2,)*5})
        r=f.controller().run(BASE, POLICY, terms=1)
        self.assertEqual((r['skill'], r['policy']), (CONTROL, POLICY))

    def test_metric_tradeoff_rolls_back(self):
        f=Fixture({BASE:(1,)*5, CONTROL:(1,)*5, PROPOSED:(2,2,2,2,0)})
        r=f.controller().run(BASE, POLICY, terms=1)
        self.assertEqual((r['skill'], r['policy']), (BASE, POLICY))

    def test_meta_validation_must_also_beat_preterm_parent(self):
        f=Fixture();f.scores.update({(CONTROL,'meta_validation'):(0,)*5,
                                    (PROPOSED,'meta_validation'):(.5,)*5})
        r=f.controller().run(BASE, POLICY, terms=1)
        self.assertEqual((r['skill'], r['policy']), (BASE, POLICY))

    def test_all_seeds_of_case_cannot_cross_splits(self):
        f=Fixture();f.parts[2]=Partition('meta_validation',(('train-case',99),))
        with self.assertRaises(ValueError):f.controller()

    def test_missing_partition_rejected(self):
        f=Fixture();f.parts=f.parts[:3]
        with self.assertRaises(ValueError):f.controller()

    def test_score_card_binding_and_coverage_fail_closed(self):
        f=Fixture();p=f.parts[1];card=f.evaluate(BASE,p,'synthetic')
        broken=[replace(card,skill_sha256='b'*64),replace(card,protocol_sha256='b'*64),
                replace(card,partition_sha256='b'*64),replace(card,rows=()),
                replace(card,rows=card.rows+card.rows),
                replace(card,rows=(('validation-case',7,(float('nan'),)*5),)),
                replace(card,rows=(('validation-case',7,(1,)*4),))]
        for c in broken:
            with self.subTest(card=c):
                with self.assertRaises(ValueError):c.means(BASE,p,HASH)

    def test_stale_training_signal_rejected(self):
        f=Fixture()
        def observe(skill,p,stage):
            r=f.observe(skill,p,stage);r[0]['skill_sha256']='b'*64;return r
        with self.assertRaises(ValueError):f.controller(observe=observe).run(BASE,POLICY,terms=1)

    def test_validation_not_exposed_and_training_refreshed(self):
        f=Fixture();f.controller().run(BASE,POLICY,terms=2)
        self.assertEqual(f.observed,[BASE,PROPOSED])
        import json
        inputs=json.dumps(f.proposer_inputs)
        for forbidden in ('validation-case','meta_validation-case','test-case','BLEUScore','means'):
            self.assertNotIn(forbidden,inputs)

    def test_invalid_patch_does_not_receive_an_unbounded_retry(self):
        f=Fixture();calls=[]
        def propose(*args):
            calls.append(args);return proposal(edits=[{'old':'# Skill','new':'Changed'}])
        r=f.controller(propose_skill=propose).run(BASE,POLICY,terms=1)
        self.assertEqual(r['skill'],BASE);self.assertEqual(len(calls),2)

    def test_protected_fields_cannot_be_written(self):
        for validator,p in [(validate_policy,proposal(policy=NEW_POLICY,gate='always pass')),
                            (validate_patch,proposal(edits=[],evaluator='new'))]:
            with self.assertRaises(ValueError):validator(p)

    def test_case_identity_cannot_be_memorized_in_policy(self):
        f=Fixture()
        r=f.controller(propose_policy=lambda *args: proposal(policy='Special-case train-case.')).run(BASE,POLICY,terms=1)
        self.assertEqual(r['policy'],POLICY)
        self.assertIn('term-01-policy-rejected',f.log)

    def test_policy_length_and_budget_are_bounded(self):
        with self.assertRaises(ValueError):validate_policy(proposal(policy='x '*221))
        with self.assertRaises(ValueError):Fixture().controller().run(BASE,POLICY,terms=0)


if __name__=='__main__':unittest.main()
