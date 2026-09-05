import ast
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('ego_pipeline', ROOT/'pipeline.py')
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


class PipelineTests(unittest.TestCase):
    def test_all_sources_compile_without_model_imports(self):
        for path in ROOT.rglob('*.py'):
            compile(path.read_text(), str(path), 'exec')

    def test_gpu_uuid_survives_container_renumbering(self):
        with patch.object(pipeline, 'capture', return_value='0, GPU-demo, NVIDIA, 20, 72000'):
            self.assertEqual(pipeline.select_gpu('GPU-demo')[0], 'GPU-demo')
            with self.assertRaises(ValueError):
                pipeline.select_gpu('2')

    def test_gpu_stage_requires_explicit_selection(self):
        with self.assertRaises(SystemExit):
            pipeline.main(['render_lab_demo', '--root', '.', '--run', '.', '--output', '/unused'])

    def test_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                pipeline.main(['prepare_sam3d_object_inputs', '--root', directory,
                               '--run', directory, '--output', directory])

    def test_gpu_imports_are_classified(self):
        for path in (ROOT/'stages').glob('*.py'):
            tree = ast.parse(path.read_text())
            imports = [alias.name for n in ast.walk(tree) if isinstance(n, ast.Import) for alias in n.names]
            if 'torch' in imports:
                self.assertIn(path.stem, pipeline.GPU_STAGES)

    def test_sources_have_no_private_machine_paths(self):
        for path in (ROOT/'stages').glob('*.py'):
            self.assertNotIn('/mnt/checkpoint/guoyijun', path.read_text())


if __name__ == '__main__':
    unittest.main()
