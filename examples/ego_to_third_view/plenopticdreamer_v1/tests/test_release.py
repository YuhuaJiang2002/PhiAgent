from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReleaseContractTests(unittest.TestCase):
    def test_progressive_stage_one_recipe(self):
        config = json.loads((ROOT / "configs/basic_stage1_24gpu.json").read_text())
        self.assertEqual(config["world_size"], 24)
        self.assertEqual(config["context_parallel_size"], 8)
        self.assertEqual(config["max_steps"], 16000)
        self.assertEqual(config["data"]["frames"], 81)
        self.assertEqual(
            config["context_schedule"],
            [
                {"until_step": 10000, "k": 1},
                {"until_step": 14000, "k": 2},
                {"until_step": 15000, "k": 3},
                {"until_step": 16000, "k": 4},
            ],
        )

    def test_asset_pins_match_downloader(self):
        pins = json.loads((ROOT / "third_party.json").read_text())
        prepare = load_module("v1_prepare_plenoptic", ROOT / "tools/prepare_plenoptic.py")
        for name in ("base", "vae", "reason"):
            repository, _, revision, _, _ = prepare.JOBS[name]
            self.assertEqual(
                (repository, revision),
                (pins["models"][name]["repository"], pins["models"][name]["revision"]),
            )
        for name in ("syncam", "multicam"):
            repository, _, revision, _, _ = prepare.JOBS[name]
            self.assertEqual(
                (repository, revision),
                (pins["datasets"][name]["repository"], pins["datasets"][name]["revision"]),
            )

    def test_physical_gpu_selection_is_explicit(self):
        launch = load_module("v1_launch", ROOT / "tools/launch.py")
        inventory = [
            {"index": "0", "uuid": "GPU-a", "name": "H20", "memory_free_mib": 90000},
            {"index": "1", "uuid": "GPU-b", "name": "H20", "memory_free_mib": 91000},
        ]
        selected = launch.select_gpus(inventory, "1,GPU-a")
        self.assertEqual([gpu["uuid"] for gpu in selected], ["GPU-b", "GPU-a"])
        with self.assertRaises(ValueError):
            launch.select_gpus(inventory, "0,0")
        with self.assertRaises(ValueError):
            launch.select_gpus(inventory, "2")

    def test_release_paths_do_not_embed_cluster_mounts(self):
        checked = list((ROOT / "configs").rglob("*.json")) + list((ROOT / "tools").rglob("*.py"))
        for path in checked:
            text = path.read_text()
            self.assertNotIn("/mnt/datasets-livsyn", text, path)
            self.assertNotIn("/data3/llq", text, path)

    def test_final_snapshot_payload_and_evidence_are_hash_bound(self):
        verifier = load_module(
            "v1_verify_final_release", ROOT / "tools/verify_final_release.py"
        )
        report = verifier.verify_release()
        self.assertEqual(report["status"], "verified")
        self.assertEqual(report["payload_files"], 61)
        self.assertEqual(
            report["checkpoint_sha256"],
            "f7ad29d7e91e1e9674335b2368b079f63c8051e9d97a122b02bce97cac93d6b3",
        )

    def test_final_checkpoint_verifier_rejects_wrong_size(self):
        verifier = load_module(
            "v1_verify_final_release_checkpoint", ROOT / "tools/verify_final_release.py"
        )
        contract = json.loads(
            (ROOT / "releases/step-016000/release.json").read_text()
        )
        with tempfile.TemporaryDirectory() as directory:
            wrong = Path(directory) / "step-016000.pt"
            wrong.write_bytes(b"not the checkpoint")
            with self.assertRaisesRegex(verifier.VerificationError, "size mismatch"):
                verifier.verify_checkpoint(wrong, contract)


if __name__ == "__main__":
    unittest.main()
