"""Synthetic-only tests of information boundaries, search and promotion."""
import json
import unittest
from dataclasses import replace
from phiagent.harness.video_meta_rsi import Partition, ScoreCard, GATES, text_hash
from phiagent.harness.video_pareto_rsi import (Assessment, Frontier, ParetoRSI, stratified_cases)
from scripts.run_ti2v_skill_optimization import bounded_edit

BASE = '# Skill\nFollow the task.\n'
POLICY = 'Diagnose failures conservatively.'
HASH = 'a'*64


def entry(name, vector, phenotype=None):
    return dict(skill=name, sha256=text_hash(name), inner=vector, meta=vector,
                phenotype=phenotype or text_hash('output:'+name))


class Fixture:
    def __init__(self):
        self.parts = [Partition(n, ((n+'-synthetic', 7),)) for n in ('train','validation','meta_validation','test')]
        self.log = {}; self.inputs = []; self.evals = []; self.proposals = 0
        self.vector = lambda skill, part: (1,)*5

    def observe(self, skill, part, stage):
        return [dict(case_id=c, seed=s, skill_sha256=text_hash(skill), video_sha256=HASH,
                     gates={g:'FAIL' for g in GATES}, trace='Synthetic visible failure.') for c,s in part.members]

    def propose_policy(self, policy, context, stage):
        self.inputs.append(context)
        return dict(hypothesis='Repair recurring failures.', expected_effect='Improve transfer.',
                    risk='May overconstrain motion.', policy='Use a different supported repair.')

    def propose_skill(self, policy, skill, context, stage):
        self.inputs.append(context); self.proposals += 1
        return dict(hypothesis='Visible repair.', expected_effect='Improve.', risk='Transfer.',
                    edits=[dict(old=skill.splitlines()[1],new='Follow repair '+str(self.proposals)+'.')])

    def evaluate(self, skill, part, stage):
        self.evals.append((skill, part.name))
        card = ScoreCard(text_hash(skill), part.sha256, HASH, HASH,
                         tuple((c,s,self.vector(skill,part.name)) for c,s in part.members))
        return Assessment(card,tuple((c,s,text_hash('output:'+skill)) for c,s in part.members))

    def controller(self, **kw):
        args = dict(observe=self.observe,propose_policy=self.propose_policy,propose_skill=self.propose_skill,
                    apply_patch=bounded_edit,evaluate=self.evaluate,record=lambda k,v:self.log.__setitem__(k,v))
        return ParetoRSI(self.parts,HASH,**(args|kw))


class ParetoContracts(unittest.TestCase):
    def test_split_balanced_and_order_independent(self):
        status={f'fake-{i}':'FAIL' if i<10 else 'PASS' for i in range(20)}
        family={c:str(i%5) for i,c in enumerate(status)}
        first=stratified_cases(status,family)
        self.assertEqual(first,stratified_cases(dict(reversed(list(status.items()))),family))
        for name in ('train','validation','meta_validation'):
            cases=[c for c in first if first[c]==name]
            self.assertEqual(len(cases),4)
            self.assertEqual(sum(status[c]=='FAIL' for c in cases),2)
        self.assertEqual(sum(v=='test' for v in first.values()),8)

    def test_uninformative_split_fails_closed(self):
        status={str(i):'PASS' for i in range(20)}
        with self.assertRaises(ValueError):stratified_cases(status,status)

    def test_frontier_explores_complementary_tradeoff(self):
        base=entry('base',(1,)*5); trade=entry('trade',(2,2,2,2,.5))
        f=Frontier(base);self.assertEqual(f.parent(),base)
        self.assertEqual(f.add(trade),'retained');self.assertEqual(f.parent(),trade)
        self.assertEqual(f.add(entry('copy',(3,)*5,trade['phenotype'])),'duplicate_selected_outputs')
        self.assertEqual(f.add(entry('bad',(0,)*5)),'dominated_or_equal_vector')

    def test_tradeoff_is_explored_but_never_promoted(self):
        for arm in ('rsi','pareto_rsi','pareto_meta_rsi'):
            f=Fixture();f.vector=lambda skill,part:(1,)*5 if skill==BASE else (2,2,2,2,.5)
            r=f.controller().run(BASE,POLICY,arm=arm)
            self.assertEqual(r['skill'],BASE);self.assertEqual(r['proposal_slots_consumed'],4)
            parent=f.log['term-02-export']['search_parent_sha256']
            self.assertEqual(parent==text_hash(BASE),arm=='rsi')

    def test_meta_requires_beating_matched_control(self):
        f=Fixture();f.vector=lambda skill,part:(1,)*5 if skill==BASE else (2,)*5
        r=f.controller().run(BASE,POLICY,arm='pareto_meta_rsi')
        self.assertEqual(r['policy'],POLICY)
        self.assertFalse(f.log['term-01-export']['policy_accepted'])
        self.assertNotEqual(r['skill'],BASE)

    def test_meta_promotes_when_both_levels_and_control_pass(self):
        f=Fixture();f.vector=lambda skill,part:((1,)*5 if skill==BASE else
                                               (2+int(skill.split('repair ')[1].split('.')[0]),)*5)
        r=f.controller().run(BASE,POLICY,arm='pareto_meta_rsi')
        self.assertNotEqual(r['policy'],POLICY)
        self.assertTrue(f.log['term-01-export']['policy_accepted'])

    def test_no_validation_feedback_or_final_access(self):
        f=Fixture();f.controller().run(BASE,POLICY,arm='pareto_meta_rsi')
        serialized=json.dumps(f.inputs)
        for term in ('validation-synthetic','test-synthetic','BLEUScore','means','frontier'):
            self.assertNotIn(term,serialized)
        self.assertNotIn('test',[p for _,p in f.evals])

    def test_invalid_proposals_consume_all_four_slots(self):
        f=Fixture()
        def bad(*args):
            f.proposals+=1
            return dict(hypothesis='x',expected_effect='x',risk='x',edits=[dict(old='# Skill',new='x')])
        r=f.controller(propose_skill=bad).run(BASE,POLICY,arm='pareto_rsi')
        self.assertEqual(f.proposals,4);self.assertEqual(r['skill'],BASE)
        self.assertEqual(r['unique_skills_evaluated'],1)

    def test_runtime_score_failures_are_not_swallowed(self):
        f=Fixture()
        def evaluate(skill,part,stage):
            if skill!=BASE:raise ValueError('Incomplete official metric coverage')
            return f.evaluate(skill,part,stage)
        with self.assertRaisesRegex(ValueError,'Incomplete official'):
            f.controller(evaluate=evaluate).run(BASE,POLICY,arm='pareto_rsi')

    def test_phenotype_coverage_binding(self):
        f=Fixture();part=f.parts[1];a=f.evaluate(BASE,part,'fake')
        with self.assertRaises(ValueError):replace(a,outputs=()).verify(BASE,part,HASH)


if __name__=='__main__':unittest.main()
