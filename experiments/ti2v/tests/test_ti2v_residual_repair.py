"""Synthetic prompt and guard tests; no benchmark data or model calls."""
import unittest
from integrations.skilladam_ti2v.residual_repair import CHECKS, compose_prompt, guard_accepts


class ResidualTests(unittest.TestCase):
    def test_literal_prefix_is_exact(self):
        task = 'Pick up the cube. Keep the first frame unchanged.'
        self.assertTrue(compose_prompt(task, 'End with the cube held.').startswith(task))

    def test_failed_or_unknown_guard_does_not_admit(self):
        checks = {k: {'status': 'PASS'} for k in CHECKS}
        self.assertTrue(guard_accepts(checks))
        for status in ('FAIL', 'UNKNOWN'):
            checks[CHECKS[0]] = {'status': status}
            self.assertFalse(guard_accepts(checks))

    def test_word_budget_and_metric_leak_fail_closed(self):
        for residual in ('word ' * 46, 'Improve nDTW'):
            with self.assertRaises(ValueError): compose_prompt('Move cube.', residual)


if __name__ == '__main__': unittest.main()
