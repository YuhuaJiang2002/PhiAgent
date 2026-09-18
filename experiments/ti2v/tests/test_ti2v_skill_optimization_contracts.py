"""Synthetic contract tests only; no data, model, or benchmark execution."""
import unittest
import hashlib
import importlib.util
from pathlib import Path
import tempfile
from integrations.skilladam_ti2v.backend import choose, GATES, stage_initial_image
from scripts.run_ti2v_skill_optimization import bounded_edit


def audit(default='PASS', **overrides):
    return {'gates':{g:{'status':overrides.get(g,default)} for g in GATES}}


class OptimizationContracts(unittest.TestCase):
    def test_unknown_cannot_pass_even_with_repaired_goal(self):
        old=audit(task_completion='FAIL')
        new=audit(visible_action_order='UNKNOWN')
        self.assertEqual(choose(old,new)['selected'],'base')

    def test_all_pass_without_strict_repair_retains_base(self):
        self.assertEqual(choose(audit(),audit())['selected'],'base')

    def test_all_pass_repair_is_candidate(self):
        self.assertEqual(choose(audit(task_completion='FAIL'),audit())['selected'],'candidate')

    def test_edits_preserve_unrelated_content(self):
        skill='# Skill\nKeep identities.\nAvoid jumps.\n'
        result=bounded_edit(skill,[{'old':'Avoid jumps.','new':'Maintain continuous motion.'}])
        self.assertEqual(result,'# Skill\nKeep identities.\nMaintain continuous motion.\n')

    def test_unknown_or_duplicate_target_is_rejected(self):
        for skill,old in [('# Skill\nKeep identities.\n','Missing.'),('# Skill\nSame.\nSame.\n','Same.')]:
            with self.assertRaises(ValueError):bounded_edit(skill,[{'old':old,'new':'replacement'}])

    def test_heading_and_newline_injection_are_rejected(self):
        for old,new in [('# Skill','New heading'),('Keep identities.','first\nsecond')]:
            with self.assertRaises(ValueError):bounded_edit('# Skill\nKeep identities.\n',[{'old':old,'new':new}])

    def test_external_reference_is_copied_inside_native_input_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'external.png';source.write_bytes(b'synthetic file boundary test, not a real image')
            pool=root/'pool';pool.mkdir();expected=hashlib.sha256(source.read_bytes()).hexdigest()
            target=stage_initial_image(pool,source,expected)
            self.assertTrue(target.resolve().is_relative_to((pool/'inputs').resolve()))
            self.assertEqual(target.read_bytes(),source.read_bytes())
            self.assertNotEqual(target,source)

    def test_cached_reference_with_wrong_bytes_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'input.png';source.write_bytes(b'synthetic correct bytes')
            pool=root/'pool';(pool/'inputs').mkdir(parents=True)
            expected=hashlib.sha256(source.read_bytes()).hexdigest();(pool/'inputs'/(expected+'.png')).write_bytes(b'corrupt')
            with self.assertRaises(AssertionError):stage_initial_image(pool,source,expected)

    @unittest.skipUnless(importlib.util.find_spec('skilladam'), 'Optional pinned upstream is not on PYTHONPATH')
    def test_official_gate_resume_signature_serializes_all_five_rules(self):
        # Pure interface/configuration check: no cases, evaluator or model calls.
        from integrations.skilladam_ti2v.adapter import register, METRICS
        from skilladam.methods.skilladam import _SerializableAdapterGate
        adapter=register()(Path('.'))
        gate=_SerializableAdapterGate(adapter=adapter,benchmark='ewm_ti2v',scope=None,profile='default')
        self.assertEqual({r['metric'] for r in gate.to_dict()['configuration']['non_regressions']},set(METRICS))


if __name__=='__main__':unittest.main()
