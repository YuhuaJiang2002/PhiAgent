#!/usr/bin/env python3
"""Metric-align VGGT-Omega's arbitrary gauge to a reviewed tabletop frame."""

from __future__ import annotations

from runtime import require_launcher
require_launcher()

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--frames-dir", type=Path, required=True)
    p.add_argument("--mano-npz", type=Path, required=True)
    p.add_argument("--sam2-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def umeyama(source: np.ndarray, target: np.ndarray):
    """Least-squares similarity X_target = scale * R @ X_source + t."""
    mu_s, mu_t = source.mean(0), target.mean(0)
    xs, xt = source - mu_s, target - mu_t
    cov = xt.T @ xs / len(source)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    var = np.mean(np.sum(xs * xs, axis=1))
    scale = float(np.trace(np.diag(D) @ S) / max(var, 1e-12))
    t = mu_t - scale * R @ mu_s
    return scale, R, t


def transform_similarity(points: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray):
    return scale * np.einsum("ij,...j->...i", R, points) + t


def unproject_pixels(depth: np.ndarray, uv: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray):
    u, v = uv[:, 0], uv[:, 1]
    cam = np.c_[(u - intrinsic[0, 2]) / intrinsic[0, 0] * depth,
                (v - intrinsic[1, 2]) / intrinsic[1, 1] * depth,
                depth]
    R, t = extrinsic[:3, :3], extrinsic[:3, 3]
    return (R.T @ (cam - t).T).T


def mask_for_frame(index: int, source_hw: tuple[int, int], mano: np.lib.npyio.NpzFile,
                   sam2_root: Path, K_source: np.ndarray):
    h, w = source_hw
    mask = np.zeros((h, w), np.uint8)
    paths = [sam2_root / "alarm_clock_v2" / "masks" / f"{index:04d}.png",
             sam2_root / "small_cylinder_v2" / "masks" / f"{index:04d}.png",
             sam2_root / "tall_cylinder" / "masks" / f"{index:04d}.png"]
    for path in paths:
        m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if m is not None:
            mask |= (m > 127).astype(np.uint8) * 255
    for side in ("left", "right"):
        vcam = mano[f"{side}_vertices_camera"][index]
        uv = np.c_[K_source[0, 0] * vcam[:, 0] / np.maximum(vcam[:, 2], 1e-5) + K_source[0, 2],
                   K_source[1, 1] * vcam[:, 1] / np.maximum(vcam[:, 2], 1e-5) + K_source[1, 2]].astype(np.int32)
        cv2.fillConvexPoly(mask, cv2.convexHull(uv), 255)
    return cv2.dilate(mask, np.ones((21, 21), np.uint8))


def write_binary_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray):
    vertex = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                       ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vertex["x"], vertex["y"], vertex["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = ("ply\nformat binary_little_endian 1.0\n" + f"element vertex {len(vertex)}\n" +
              "property float x\nproperty float y\nproperty float z\n" +
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with path.open("wb") as f:
        f.write(header.encode("ascii")); vertex.tofile(f)


def main() -> None:
    cfg = parse_args()
    pred = np.load(cfg.predictions)
    ext = pred["extrinsic_camera_from_omega_world"].astype(np.float64)
    intr = pred["intrinsic_processed_pixels"].astype(np.float64)
    depth = pred["depth"].astype(np.float32)
    conf = pred["depth_confidence"].astype(np.float32)
    if depth.ndim == 4:
        depth = depth[..., 0]
    if conf.ndim == 4:
        conf = conf[..., 0]
    images = pred["processed_images"].astype(np.float32)
    if images.shape[1] == 3:
        images = np.transpose(images, (0, 2, 3, 1))
    images = np.clip(images * 255.0, 0, 255).astype(np.uint8)
    names = sorted(cfg.frames_dir.glob("*.jpg"))
    source0 = cv2.imread(str(names[0]))
    sh, sw = source0.shape[:2]
    ph, pw = depth.shape[1:3]
    mano = np.load(cfg.mano_npz)
    K_source = np.array([[600.0, 0.0, sw/2], [0.0, 600.0, sh/2], [0.0, 0.0, 1.0]])
    ref_px = np.array([[307,207],[744,226],[829,525],[115,535]], np.float32)
    ref_xy = np.array([[-.35,-.225],[.35,-.225],[.35,.225],[-.35,.225]], np.float32)
    H_xy_to_src = cv2.getPerspectiveTransform(ref_xy.astype(np.float32), ref_px)
    dyn0 = mask_for_frame(0, (sh, sw), mano, cfg.sam2_root, K_source)

    source_points, target_points, weights = [], [], []
    for y in np.linspace(-.195, .195, 11):
        for x in np.linspace(-.31, .31, 15):
            uv_src = cv2.perspectiveTransform(np.array([[[x, y]]], np.float32), H_xy_to_src)[0, 0]
            sx, sy = int(round(uv_src[0])), int(round(uv_src[1]))
            if not (4 <= sx < sw-4 and 4 <= sy < sh-4) or dyn0[sy, sx] > 0:
                continue
            u, v = uv_src[0] * pw / sw, uv_src[1] * ph / sh
            ui, vi = int(round(u)), int(round(v))
            patch_d = depth[0, max(0,vi-2):vi+3, max(0,ui-2):ui+3]
            patch_c = conf[0, max(0,vi-2):vi+3, max(0,ui-2):ui+3]
            good = np.isfinite(patch_d) & (patch_d > 0)
            if not good.any():
                continue
            d = float(np.median(patch_d[good]))
            c = float(np.median(patch_c[good]))
            p = unproject_pixels(np.array([d]), np.array([[u, v]]), ext[0], intr[0])[0]
            # Choose the planar normal branch for which the head camera is at
            # positive z.  Flipping y together with z is a proper 180-degree
            # rotation (not an invalid reflection), and preserves handedness.
            source_points.append(p); target_points.append([x, -y, 0.0]); weights.append(c)
    source_points = np.asarray(source_points); target_points = np.asarray(target_points); weights = np.asarray(weights)
    if len(source_points) < 12:
        raise RuntimeError(f"only {len(source_points)} unmasked table samples")

    # Robustly reject pixels that actually see a forearm/paper/object instead of the table plane.
    keep = np.ones(len(source_points), bool)
    for _ in range(5):
        scale, Ra, ta = umeyama(source_points[keep], target_points[keep])
        residual = np.linalg.norm(transform_similarity(source_points, scale, Ra, ta) - target_points, axis=1)
        med = float(np.median(residual[keep])); mad = float(np.median(np.abs(residual[keep] - med)))
        threshold = max(0.018, med + 2.8 * max(mad, 1e-4))
        new_keep = residual < threshold
        if new_keep.sum() == keep.sum():
            break
        keep = new_keep
    scale, Ra, ta = umeyama(source_points[keep], target_points[keep])
    residual = np.linalg.norm(transform_similarity(source_points, scale, Ra, ta) - target_points, axis=1)

    # Convert the arbitrary-gauge camera model into metric table-world -> metric camera.
    aligned_R, aligned_t = [], []
    for E in ext:
        Ri, ti = E[:3, :3], E[:3, 3]
        Rp = Ri @ Ra.T
        tp = scale * ti - Rp @ ta
        aligned_R.append(Rp); aligned_t.append(tp)
    aligned_R, aligned_t = np.asarray(aligned_R), np.asarray(aligned_t)
    centers = np.asarray([-aligned_R[i].T @ aligned_t[i] for i in range(len(aligned_R))])

    # Fuse a conservative, observed-only static point cloud from 2 FPS frames.
    points_all, colors_all, conf_all = [], [], []
    for i in range(0, len(ext), 15):
        ys, xs = np.mgrid[0:ph:8, 0:pw:8]
        uv = np.c_[xs.ravel(), ys.ravel()]
        d = depth[i, ys, xs].reshape(-1)
        c = conf[i, ys, xs].reshape(-1)
        dyn = mask_for_frame(i, (sh, sw), mano, cfg.sam2_root, K_source)
        dyn_small = cv2.resize(dyn, (pw, ph), interpolation=cv2.INTER_NEAREST)
        dynamic = dyn_small[ys, xs].reshape(-1) > 0
        valid = np.isfinite(d) & (d > 0) & (~dynamic) & (c >= np.percentile(c[np.isfinite(c)], 55))
        if not valid.any():
            continue
        wo = unproject_pixels(d[valid], uv[valid], ext[i], intr[i])
        wp = transform_similarity(wo, scale, Ra, ta)
        rgb = images[i, ys, xs].reshape(-1, 3)[valid]
        bounded = ((wp[:,0] > -1.5) & (wp[:,0] < 1.5) & (wp[:,1] > -1.5) & (wp[:,1] < 1.5) &
                   (wp[:,2] > -0.15) & (wp[:,2] < 2.0))
        points_all.append(wp[bounded]); colors_all.append(rgb[bounded]); conf_all.append(c[valid][bounded])
    points = np.concatenate(points_all); colors = np.concatenate(colors_all); point_conf = np.concatenate(conf_all)
    # One representative per 8 mm voxel, preferring higher-confidence samples.
    vox = np.floor(points / 0.008).astype(np.int64)
    order = np.argsort(point_conf)[::-1]
    _, first = np.unique(vox[order], axis=0, return_index=True)
    chosen = order[first]
    points, colors = points[chosen], colors[chosen]

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    aligned_path = cfg.output_dir / "vggt_omega_table_aligned.npz"
    np.savez_compressed(aligned_path,
                        R_table_world_to_camera_m=aligned_R.astype(np.float32),
                        t_table_world_to_camera_m=aligned_t.astype(np.float32),
                        intrinsic_processed_pixels=intr.astype(np.float32),
                        omega_to_table_scale=np.asarray(scale, np.float32),
                        omega_to_table_R=Ra.astype(np.float32), omega_to_table_t=ta.astype(np.float32),
                        mat_corners_source_order_table_world=np.array([[-.35,.225,0],[.35,.225,0],[.35,-.225,0],[-.35,-.225,0]], np.float32),
                        static_points_table_world=points.astype(np.float32), static_colors_rgb=colors.astype(np.uint8))
    write_binary_ply(cfg.output_dir / "observed_static_scene_table_world.ply", points.astype(np.float32), colors.astype(np.uint8))
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    report = {
        "schema_version": "phiagent-vggt-omega-table-alignment/0.1",
        "correspondences": int(len(source_points)), "inliers": int(keep.sum()),
        "scale_m_per_omega_unit": scale,
        "alignment_rmse_m": float(np.sqrt(np.mean(residual[keep] ** 2))),
        "alignment_p95_m": float(np.percentile(residual[keep], 95)),
        "camera_center_bounds_m": {"min": centers.min(0).tolist(), "max": centers.max(0).tolist()},
        "camera_step_m": {"median": float(np.median(steps)), "p99": float(np.percentile(steps,99)), "max": float(steps.max())},
        "static_observed_points": int(len(points)),
        "authority": "Metric gauge comes from a manually reviewed 0.70 x 0.45 m mat; point cloud is observed-only VGGT-Omega depth, not completed geometry.",
        "output": str(aligned_path.resolve()),
    }
    (cfg.output_dir / "alignment_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
