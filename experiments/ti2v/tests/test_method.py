"""Synthetic public-entry contracts; no benchmark/model access."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from method import propose

class MethodTests(unittest.TestCase):
    def call(self, folder, relation='support_before_release', instruction='Place the cube on the table.'):
        row = dict(instruction=instruction, initial='synthetic.png', initial_sha256='a'*64,
                   base_sha256='b'*64, seed=7, official_score=999, reference_future='forbidden')
        raw = dict(status='APPLIED', relation_id=relation, skill='edited', edits=['synthetic'])
        with patch('method.prepare_factorial_repair', return_value=raw) as backend:
            value = propose(None, row, {'gates': {}}, 'parent', folder, [], [])
            self.assertNotIn('official_score', backend.call_args.args[1])
            self.assertNotIn('reference_future', backend.call_args.args[1])
        self.assertEqual(raw['status'], 'APPLIED')
        return value

    def test_supported_placement_edit(self):
        with tempfile.TemporaryDirectory() as folder:
            value = self.call(folder)
            self.assertEqual(value['status'], 'APPLIED')
            self.assertEqual(value['skill'], 'edited')
            self.assertFalse(value['video_quality_improvement_established'])

    def test_unrequested_withdrawal_retains_parent(self):
        with tempfile.TemporaryDirectory() as folder:
            value = self.call(folder, 'release_before_withdrawal')
            self.assertEqual(value['status'], 'ABSTAINED')
            self.assertEqual(value['skill'], 'parent')
            self.assertEqual(value['edits'], [])
            self.assertEqual(json.loads((Path(folder)/'method-decision.json').read_text()), value)

    def test_unknown_task_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(self.call(folder, instruction='Arrange the cube.')['status'], 'ABSTAINED')

    def test_prior_decision_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            self.call(folder)
            with self.assertRaises(FileExistsError): self.call(folder)
