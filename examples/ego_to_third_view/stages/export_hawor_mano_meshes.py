
from runtime import require_launcher
require_launcher()
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

for _name, _value in {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "unicode": str,
    "str": str,
}.items():
    if _name not in np.__dict__:
        setattr(np, _name, _value)

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hawor.utils.process import get_mano_faces, run_mano, run_mano_left
from hawor.utils.rotation import rotation_matrix_to_angle_axis


def load_side(path: Path, left: bool) -> np.ndarray:
    data = json.loads(path.read_text())
    trans = torch.tensor(data["init_trans"], dtype=torch.float32)
    root = rotation_matrix_to_angle_axis(
        torch.tensor(data["init_root_orient"], dtype=torch.float32)
    )
    pose = rotation_matrix_to_angle_axis(
        torch.tensor(data["init_hand_pose"], dtype=torch.float32)
    )
    betas = torch.tensor(data["init_betas"], dtype=torch.float32)
    fn = run_mano_left if left else run_mano
    with torch.inference_mode():
        result = fn(trans, root, pose, betas=betas)
    return result["vertices"][0].detach().cpu().numpy().astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    left_path = next((args.sequence_dir / "cam_space" / "0").glob("*.json"))
    right_path = next((args.sequence_dir / "cam_space" / "1").glob("*.json"))
    left = load_side(left_path, left=True)
    right = load_side(right_path, left=False)
    faces = get_mano_faces().astype(np.int32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        left_vertices_camera=left,
        right_vertices_camera=right,
        right_faces=faces,
        left_faces=faces[:, [0, 2, 1]],
        units="meters",
        camera_convention="OpenCV-like x-right y-down z-forward",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "frames": int(left.shape[0]),
                "vertices_per_hand": int(left.shape[1]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
