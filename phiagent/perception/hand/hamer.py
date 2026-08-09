"""Optional in-process adapter for the official HaMeR right-hand teacher."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from phiagent.perception.geometry import rotation_matrix_to_quaternion
from phiagent.perception.schema import HandObservation
from phiagent.physical_language.schema import FrameKind, FrameRef, Point3D, PoseSE3

HAMER_COMMIT = "3a01849f4148352e9260b69bf28b65d1671a4905"


@dataclass(frozen=True)
class HamerConfig:
    repository: Path
    checkpoint: Path | None = None
    frame_stride: int = 1
    detection_threshold: float = 0.5
    rescale_factor: float = 2.0

    def __post_init__(self) -> None:
        if self.frame_stride < 1:
            raise ValueError("HaMeR frame_stride must be positive")
        if not 0 < self.detection_threshold <= 1:
            raise ValueError("HaMeR detection threshold must be in (0, 1]")
        if self.rescale_factor <= 0:
            raise ValueError("HaMeR rescale factor must be positive")


class HamerHandTracker:
    """Run the pinned official model and export metric camera-frame joints."""

    def __init__(self, config: HamerConfig) -> None:
        self.config = config

    def preflight(self) -> None:
        repository = self.config.repository.resolve()
        if not (repository / "hamer").is_dir():
            raise RuntimeError(f"HaMeR repository is missing: {repository}")
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        if commit != HAMER_COMMIT:
            raise RuntimeError(f"HaMeR commit mismatch: {commit} != {HAMER_COMMIT}")
        mano = repository / "_DATA" / "data" / "mano" / "MANO_RIGHT.pkl"
        if not mano.is_file():
            raise RuntimeError(
                "HaMeR requires the separately licensed MANO_RIGHT.pkl at "
                f"{mano}"
            )
        if self.config.checkpoint is not None and not self.config.checkpoint.is_file():
            raise RuntimeError(f"HaMeR checkpoint is missing: {self.config.checkpoint}")

    @staticmethod
    def _imports(repository: Path) -> dict[str, Any]:
        if str(repository) not in sys.path:
            sys.path.insert(0, str(repository))
        try:
            import cv2
            import numpy as np
            import torch
            from detectron2 import model_zoo
            from hamer.configs import CACHE_DIR_HAMER
            from hamer.datasets.vitdet_dataset import ViTDetDataset
            from hamer.models import DEFAULT_CHECKPOINT, download_models, load_hamer
            from hamer.utils import recursive_to
            from hamer.utils.renderer import cam_crop_to_full
            from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
            from vitpose_model import ViTPoseModel
        except ImportError as exc:
            raise RuntimeError(
                "HaMeR runtime is unavailable; use its dedicated optional environment"
            ) from exc
        return {
            "cv2": cv2,
            "np": np,
            "torch": torch,
            "model_zoo": model_zoo,
            "CACHE_DIR_HAMER": CACHE_DIR_HAMER,
            "ViTDetDataset": ViTDetDataset,
            "DEFAULT_CHECKPOINT": DEFAULT_CHECKPOINT,
            "download_models": download_models,
            "load_hamer": load_hamer,
            "recursive_to": recursive_to,
            "cam_crop_to_full": cam_crop_to_full,
            "DefaultPredictor_Lazy": DefaultPredictor_Lazy,
            "ViTPoseModel": ViTPoseModel,
        }

    def track(
        self, video_path: Path, camera_frame: FrameRef
    ) -> tuple[HandObservation, ...]:
        self.preflight()
        if camera_frame.kind is not FrameKind.CAMERA:
            raise ValueError("HaMeR observations require a camera target frame")
        if not video_path.is_file():
            raise ValueError(f"HaMeR video does not exist: {video_path}")
        runtime = self._imports(self.config.repository.resolve())
        cv2, np, torch = runtime["cv2"], runtime["np"], runtime["torch"]
        cache_dir = runtime["CACHE_DIR_HAMER"]
        runtime["download_models"](cache_dir)
        checkpoint = (
            str(self.config.checkpoint)
            if self.config.checkpoint is not None
            else runtime["DEFAULT_CHECKPOINT"]
        )
        model, model_cfg = runtime["load_hamer"](checkpoint)
        if not torch.cuda.is_available():
            raise RuntimeError("HaMeR teacher requires CUDA in the selected environment")
        device = torch.device("cuda")
        model = model.to(device).eval()

        detectron_cfg = runtime["model_zoo"].get_config(
            "new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py",
            trained=True,
        )
        detectron_cfg.model.roi_heads.box_predictor.test_score_thresh = (
            self.config.detection_threshold
        )
        detectron_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
        detector = runtime["DefaultPredictor_Lazy"](detectron_cfg)
        pose_detector = runtime["ViTPoseModel"](device)

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"could not open HaMeR input video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            capture.release()
            raise ValueError("HaMeR input video has invalid FPS")
        observations: list[HandObservation] = []
        wrist_frame = FrameRef(FrameKind.HUMAN_WRIST, "right")
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index % self.config.frame_stride:
                    frame_index += 1
                    continue
                detections = detector(frame)["instances"]
                valid = (detections.pred_classes == 0) & (
                    detections.scores > self.config.detection_threshold
                )
                boxes = detections.pred_boxes.tensor[valid].cpu().numpy()
                scores = detections.scores[valid].cpu().numpy()
                if len(boxes) == 0:
                    frame_index += 1
                    continue
                poses = pose_detector.predict_pose(
                    frame[:, :, ::-1],
                    [np.concatenate([boxes, scores[:, None]], axis=1)],
                )
                candidates = []
                for pose in poses:
                    keypoints = pose["keypoints"][-21:]
                    confident = keypoints[:, 2] > self.config.detection_threshold
                    if int(confident.sum()) > 3:
                        selected = keypoints[confident]
                        bbox = np.array(
                            [
                                selected[:, 0].min(),
                                selected[:, 1].min(),
                                selected[:, 0].max(),
                                selected[:, 1].max(),
                            ]
                        )
                        candidates.append((float(selected[:, 2].mean()), bbox))
                if not candidates:
                    frame_index += 1
                    continue
                confidence, bbox = max(candidates, key=lambda item: item[0])
                dataset = runtime["ViTDetDataset"](
                    model_cfg,
                    frame,
                    np.stack([bbox]),
                    np.ones(1),
                    rescale_factor=self.config.rescale_factor,
                )
                batch = next(
                    iter(
                        torch.utils.data.DataLoader(
                            dataset, batch_size=1, shuffle=False, num_workers=0
                        )
                    )
                )
                batch = runtime["recursive_to"](batch, device)
                with torch.no_grad():
                    output = model(batch)
                pred_cam = output["pred_cam"].clone()
                box_center = batch["box_center"].float()
                box_size = batch["box_size"].float()
                image_size = batch["img_size"].float()
                scaled_focal = (
                    model_cfg.EXTRA.FOCAL_LENGTH
                    / model_cfg.MODEL.IMAGE_SIZE
                    * image_size.max()
                )
                camera_translation = runtime["cam_crop_to_full"](
                    pred_cam, box_center, box_size, image_size, scaled_focal
                )[0].detach().cpu().numpy()
                joints = output["pred_keypoints_3d"][0, :21].detach().cpu().numpy()
                joints = joints + camera_translation[None, :]
                rotation = (
                    output["pred_mano_params"]["global_orient"][0, 0]
                    .detach()
                    .cpu()
                    .numpy()
                )
                quaternion = rotation_matrix_to_quaternion(rotation.tolist())
                articulation = tuple(
                    float(value)
                    for value in output["pred_mano_params"]["hand_pose"][0]
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )
                points = tuple(
                    Point3D(
                        camera_frame,
                        tuple(float(value) for value in point),
                        confidence,
                    )
                    for point in joints
                )
                observations.append(
                    HandObservation(
                        timestamp_s=frame_index / fps,
                        wrist_pose=PoseSE3(
                            wrist_frame,
                            camera_frame,
                            points[0].xyz_m,
                            quaternion,
                            confidence,
                        ),
                        keypoints_3d=points,
                        articulation=articulation,
                        confidence=confidence,
                    )
                )
                frame_index += 1
        finally:
            capture.release()
        if len(observations) < 2:
            raise RuntimeError(
                "HaMeR found fewer than two right-hand observations; no EPL sequence can be built"
            )
        return tuple(observations)
