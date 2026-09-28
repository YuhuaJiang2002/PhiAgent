"""Synthetic grouping checks; no model, media or benchmark data."""
import unittest
from scripts.run_ti2v_extended_generation import ARMS, distinct_prompt_groups

class ExtendedGenerationTests(unittest.TestCase):
    def test_exact_prompt_duplicates_share_one_allocation(self):
        prompts = dict.fromkeys(ARMS, 'Task and unchanged parent')
        prompts['combined_extended'] = 'Task and a supported edit'
        groups = distinct_prompt_groups(prompts)
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups['Task and a supported edit'], ['combined_extended'])
        self.assertEqual(sum(map(len, groups.values())), 3)

    def test_missing_arm_fails(self):
        with self.assertRaises(ValueError):
            distinct_prompt_groups({'parent': 'Task'})

    def test_no_cross_prompt_normalization(self):
        prompts = dict.fromkeys(ARMS, 'Task')
        prompts['factored_extended'] = 'Task '
        self.assertEqual(len(distinct_prompt_groups(prompts)), 2)

if __name__ == '__main__':
    unittest.main()
