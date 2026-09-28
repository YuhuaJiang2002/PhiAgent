"""Synthetic contracts only: no model, video or benchmark data."""
import unittest

from integrations.skilladam_ti2v.backend import GATES, choose
from integrations.skilladam_ti2v.phase_trace import intersect_audits, validate_trace, validate_plan, validate_trace_verdict


def audit(status='PASS'):
    return {'terminal_goal': 'synthetic', 'gates': {g: {'status': status,
        'reason': 'synthetic evidence', 'frame_ids': [1, 16]} for g in GATES}}


class PhaseTraceContracts(unittest.TestCase):
    def test_independent_pass_cannot_clear_original_fail(self):
        a, b = audit(), audit(); a['gates']['motion_continuity']['status'] = 'FAIL'
        c = intersect_audits(a, b)
        self.assertEqual(c['gates']['motion_continuity']['status'], 'FAIL')
        self.assertEqual(choose(audit('FAIL'), c)['selected'], 'base')

    def test_unknown_is_not_a_pass(self):
        self.assertTrue(all(g['status'] == 'UNKNOWN' for g in intersect_audits(audit(), audit('UNKNOWN'))['gates'].values()))

    def test_trace_must_cover_endpoints_in_order(self):
        validate_trace({'states': [{'frame_ids': [1, 2]}, {'frame_ids': [3, 7]}, {'frame_ids': [8, 12]}, {'frame_ids': [13, 16]}]})
        for seq in [[[2], [4], [8], [16]], [[1], [8], [7], [16]], [[1], [4], [8], [15]]]:
            with self.assertRaises(ValueError): validate_trace({'states': [{'frame_ids': ids} for ids in seq]})

    def test_plan_rejects_benchmark_identity(self):
        p = {'prompt': 'word ' * 25, 'phases': [{'action': 'move', 'visible_precondition': 'held', 'visible_postcondition': 'held'}]}
        validate_plan(p)
        p['prompt'] += 'gt_dataset/000'
        with self.assertRaises(ValueError): validate_plan(p)

    def test_trace_verdict_cannot_cite_unprovided_initial_frame(self):
        value = audit(); value['gates']['task_completion']['frame_ids'] = [0]
        with self.assertRaises(ValueError):
            validate_trace_verdict(value, {'states': [{'frame_ids': list(range(1, 17))}]})


if __name__ == '__main__': unittest.main()
