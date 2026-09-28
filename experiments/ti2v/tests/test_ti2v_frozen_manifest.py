"""Synthetic file fixtures only; no media, evaluator or benchmark inputs."""
from pathlib import Path
import tempfile
import unittest
from scripts.ti2v_frozen_manifest import REQUIRED, freeze_manifest

class FrozenManifestTests(unittest.TestCase):
    def populate(self, root):
        for name in (*REQUIRED, 'source/controller.py'):
            p = root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('{}')

    def test_live_logs_do_not_invalidate_frozen_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); self.populate(root)
            log = root / 'preparation.log'; log.write_text('starting')
            before = freeze_manifest(root)
            log.write_text('completed')
            (root / 'supervisor-state.json').write_text('running')
            self.assertEqual(before, freeze_manifest(root))
            (root / 'inputs.json').write_text('changed')
            self.assertNotEqual(before, freeze_manifest(root))

    def test_missing_required_input_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); self.populate(root)
            (root / 'protocol.json').unlink()
            with self.assertRaises(ValueError): freeze_manifest(root)

    def test_source_symlink_cannot_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'run'; self.populate(root)
            outside = Path(directory) / 'outside.py'; outside.write_text('x = 1')
            (root / 'source/outside.py').symlink_to(outside)
            with self.assertRaises(ValueError): freeze_manifest(root)

if __name__ == '__main__': unittest.main()
