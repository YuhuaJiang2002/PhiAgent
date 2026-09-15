#!/usr/bin/env python3
"""Build an auditable fixed-third-view replay from an egocentric RGB clip.

This first-pass demo deliberately separates measured/predicted content from
unobserved content:

* camera motion: planar visual tracking of the tabletop, anchored by one
  manually reviewed metric quadrilateral;
* hands: HaWoR camera-space MANO meshes transformed into the table frame;
* rigid objects: SAM2 masks converted to coarse metric proxy trajectories;
* background: a multi-frame median texture of only the observed table plane;
* outside the observed plane: neutral background, never generative fill.

It is a visualization/reconstruction demo, not action ground truth.  All
heuristics and coordinate conventions are recorded in the output manifest.
"""

from __future__ import annotations

from runtime import require_launcher
require_launcher()

import argparse
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class ObjectSpec:
    key: str
    label: str
    kind: str
    size: tuple[float, float, float]
    color: tuple[int, int, int]
    mask_dir: Path
    held_range: tuple[int, int] | None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames-dir", type=Path, required=True)
    p.add_argument("--mano-npz", type=Path, required=True)
    p.add_argument("--sam2-root", type=Path, required=True)
    p.add_argument("--vggt-aligned-npz", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--fps", type=float, default=30.0)
    return p.parse_args()


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / max(n, 1e-9)


def smooth_1d(values: np.ndarray, radius: int = 4) -> np.ndarray:
    values = np.asarray(values, np.float64)
    if radius <= 0:
        return values.copy()
    pad = np.pad(values, (radius, radius), mode="edge")
    kernel = np.ones(2 * radius + 1, np.float64) / (2 * radius + 1)
    return np.convolve(pad, kernel, mode="valid")


def smooth_array(values: np.ndarray, radius: int = 4) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float64)
    flat = values.reshape(values.shape[0], -1)
    out_flat = out.reshape(values.shape[0], -1)
    for j in range(flat.shape[1]):
        out_flat[:, j] = smooth_1d(flat[:, j], radius)
    return out


def interpolate_missing(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.asarray(values, np.float64).copy()
    x = np.arange(len(out))
    good = np.flatnonzero(valid & np.all(np.isfinite(out), axis=1))
    if len(good) == 0:
        return np.zeros_like(out)
    for d in range(out.shape[1]):
        out[:, d] = np.interp(x, good, out[good, d])
    return out


def polygon_area(points: np.ndarray) -> float:
    return float(abs(cv2.contourArea(points.astype(np.float32))))


def track_table_homographies(frames: list[np.ndarray], ref_corners: np.ndarray):
    """Track reference tabletop coordinates into every frame using ORB+RANSAC."""
    orb = cv2.ORB_create(nfeatures=6000, scaleFactor=1.2, nlevels=8, edgeThreshold=15)
    ref_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    ref_mask = np.zeros_like(ref_gray)
    # Include the table and a little wall border, while excluding the lower body.
    table_roi = np.array([[20, 115], [1000, 120], [1160, 650], [20, 650]], np.int32)
    cv2.fillConvexPoly(ref_mask, table_roi, 255)
    kp_ref, des_ref = orb.detectAndCompute(ref_gray, ref_mask)
    if des_ref is None or len(kp_ref) < 30:
        raise RuntimeError("reference frame has too few ORB features")
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    homographies: list[np.ndarray] = []
    corners_raw: list[np.ndarray] = []
    inlier_counts: list[int] = []
    prev_gray = ref_gray
    kp_prev, des_prev = orb.detectAndCompute(prev_gray, None)
    h_ref_prev = np.eye(3, dtype=np.float64)
    for i, frame in enumerate(frames):
        if i == 0:
            H = np.eye(3, dtype=np.float64)
            inliers = len(kp_ref)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kp_cur, des_cur = orb.detectAndCompute(gray, None)
            H = None
            inliers = 0
            if des_cur is not None:
                pairs = matcher.knnMatch(des_ref, des_cur, k=2)
                good = [m for m, n in pairs if m.distance < 0.74 * n.distance]
                if len(good) >= 18:
                    src = np.float32([kp_ref[m.queryIdx].pt for m in good])
                    dst = np.float32([kp_cur[m.trainIdx].pt for m in good])
                    cand, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
                    if cand is not None:
                        q = cv2.perspectiveTransform(ref_corners[None], cand)[0]
                        a = polygon_area(q)
                        if 7.0e4 < a < 5.5e5 and np.all(np.isfinite(q)):
                            H = cand
                            inliers = int(mask.sum()) if mask is not None else 0
            if H is None and des_cur is not None and des_prev is not None:
                pairs = matcher.knnMatch(des_prev, des_cur, k=2)
                good = [m for m, n in pairs if m.distance < 0.72 * n.distance]
                if len(good) >= 18:
                    src = np.float32([kp_prev[m.queryIdx].pt for m in good])
                    dst = np.float32([kp_cur[m.trainIdx].pt for m in good])
                    step, mask = cv2.findHomography(src, dst, cv2.RANSAC, 2.5)
                    if step is not None:
                        cand = step @ h_ref_prev
                        q = cv2.perspectiveTransform(ref_corners[None], cand)[0]
                        if 7.0e4 < polygon_area(q) < 5.5e5:
                            H = cand
                            inliers = int(mask.sum()) if mask is not None else 0
            if H is None:
                H = h_ref_prev.copy()
            prev_gray, kp_prev, des_prev = gray, kp_cur, des_cur
        h_ref_prev = H / H[2, 2]
        q = cv2.perspectiveTransform(ref_corners[None], h_ref_prev)[0]
        homographies.append(h_ref_prev.copy())
        corners_raw.append(q)
        inlier_counts.append(inliers)
    corners_raw_arr = np.asarray(corners_raw, np.float64)
    inlier_arr = np.asarray(inlier_counts)
    # Homography estimates can remain algebraically valid while sending one
    # table corner far outside the image.  Such frames caused catastrophic
    # planar-PnP flips in v1, so gate them before temporal interpolation.
    bounds_lo = np.array([[40, 120], [430, 110], [430, 380], [-20, 330]], np.float64)
    bounds_hi = np.array([[720, 390], [1180, 450], [1240, 710], [760, 710]], np.float64)
    in_bounds = np.all((corners_raw_arr >= bounds_lo[None]) & (corners_raw_arr <= bounds_hi[None]), axis=(1, 2))
    areas = np.asarray([polygon_area(q) for q in corners_raw_arr])
    valid = (inlier_arr >= 18) & in_bounds & (areas > 7.0e4) & (areas < 4.5e5)
    # A direct-reference estimate is independent frame-to-frame, making linear
    # interpolation across rejected spans safer than retaining a bad fallback.
    flat = corners_raw_arr.reshape(len(frames), -1)
    flat = interpolate_missing(flat, valid)
    corners_smoothed = smooth_array(flat.reshape(-1, 4, 2), radius=4)
    homographies_smoothed = []
    for q in corners_smoothed:
        homographies_smoothed.append(cv2.getPerspectiveTransform(ref_corners.astype(np.float32), q.astype(np.float32)))
    return np.asarray(homographies_smoothed), corners_smoothed, inlier_arr, valid


def solve_camera_poses(corners: np.ndarray, K: np.ndarray):
    mat_world = np.array(
        [[-0.35, -0.225, 0.0], [0.35, -0.225, 0.0], [0.35, 0.225, 0.0], [-0.35, 0.225, 0.0]],
        np.float32,
    )
    rvecs, tvecs, ok_flags = [], [], []
    previous_center = None
    for q in corners:
        candidates = cv2.solvePnPGeneric(mat_world, q.astype(np.float32), K, None, flags=cv2.SOLVEPNP_IPPE)
        ok = bool(candidates[0])
        rv_list, tv_list = candidates[1], candidates[2]
        options = []
        for rv, tv in zip(rv_list, tv_list):
            R = cv2.Rodrigues(rv)[0]
            c = -R.T @ tv.reshape(3)
            c[2] *= -1.0
            # The head camera must stay above and on the operator side of the table.
            if 0.35 < c[2] < 1.15 and -0.65 < c[0] < 0.75 and -0.10 < c[1] < 1.10:
                continuity = 0.0 if previous_center is None else float(np.linalg.norm(c - previous_center))
                options.append((continuity, rv.reshape(3), tv.reshape(3), c))
        if options:
            _, rvec, tvec, previous_center = min(options, key=lambda x: x[0])
        elif rvecs:
            rvec, tvec = rvecs[-1].copy(), tvecs[-1].copy()
            ok = False
        else:
            ok0, rv, tv = cv2.solvePnP(mat_world, q.astype(np.float32), K, None, flags=cv2.SOLVEPNP_ITERATIVE)
            rvec, tvec, ok = rv.reshape(3), tv.reshape(3), bool(ok0)
            R = cv2.Rodrigues(rvec)[0]
            previous_center = camera_center_physical(R, tvec)
        rvecs.append(np.asarray(rvec).reshape(3))
        tvecs.append(np.asarray(tvec).reshape(3))
        ok_flags.append(bool(ok))
    rvecs = smooth_array(np.asarray(rvecs), radius=3)
    tvecs = smooth_array(np.asarray(tvecs), radius=3)
    rotations = np.asarray([cv2.Rodrigues(r)[0] for r in rvecs])
    return rotations, tvecs, np.asarray(ok_flags)


def stabilize_mesh_global_translation(meshes_camera: np.ndarray, max_step_m: float = 0.065) -> np.ndarray:
    """Remove single-frame HaWoR translation spikes without altering hand articulation."""
    centers = np.median(meshes_camera, axis=1)
    local = meshes_camera - centers[:, None]
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    valid = np.ones(len(centers), bool)
    valid[1:] &= steps < max_step_m
    valid[:-1] &= steps < max_step_m
    centers = interpolate_missing(centers, valid)
    centers = smooth_array(centers, radius=2)
    return local + centers[:, None]


def enforce_observed_table_support(meshes_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Prevent a predicted MANO mesh from crossing the observed tabletop.

    Monocular hand depth and VGGT camera depth have independent gauges.  After
    the camera is metric-aligned, their residual vertical offset is resolved by
    the weakest defensible contact prior: the lower hand surface cannot be
    below the table.  Positive lift motion is preserved.
    """
    out = meshes_world.copy()
    lower_surface = np.percentile(out[:, :, 2], 5, axis=1)
    correction = np.maximum(0.0, 0.004 - lower_surface)
    correction = smooth_1d(correction, radius=3)
    out[:, :, 2] += correction[:, None]
    return out, correction


def camera_to_physical_world(vertices_camera: np.ndarray, R: np.ndarray, t: np.ndarray, flip_world_z: bool) -> np.ndarray:
    raw = (R.T @ (vertices_camera - t[None]).T).T
    if flip_world_z:
        raw[:, 2] *= -1.0  # Planar PnP chose the plane normal opposite physical z-up.
    return raw


def camera_center_physical(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    c = -R.T @ t
    c[2] *= -1.0
    return c


def pixel_ray_to_height(u: float, v: float, height: float, K: np.ndarray, R: np.ndarray, t: np.ndarray,
                        flip_world_z: bool) -> np.ndarray:
    d_cam = np.linalg.inv(K) @ np.array([u, v, 1.0], np.float64)
    d_raw = R.T @ d_cam
    c_raw = -R.T @ t
    d = d_raw.copy(); c = c_raw.copy()
    if flip_world_z:
        d[2] *= -1.0; c[2] *= -1.0
    if abs(d[2]) < 1e-8:
        return np.array([np.nan, np.nan, height])
    lam = (height - c[2]) / d[2]
    return c + lam * d


def mask_centroid(path: Path) -> tuple[float, float, int]:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return math.nan, math.nan, 0
    m = (m > 127).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return math.nan, math.nan, 0
    j = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return float(cents[j, 0]), float(cents[j, 1]), int(stats[j, cv2.CC_STAT_AREA])


def object_tracks(specs: list[ObjectSpec], n: int, K: np.ndarray, rotations: np.ndarray, tvecs: np.ndarray,
                  hands_world: dict[str, np.ndarray], flip_world_z: bool):
    tracks = {}
    diagnostics = {}
    hand_center = 0.5 * (np.median(hands_world["left"], axis=1) + np.median(hands_world["right"], axis=1))
    for spec in specs:
        uv = np.full((n, 2), np.nan, np.float64)
        area = np.zeros(n, np.int64)
        valid = np.zeros(n, bool)
        for i in range(n):
            u, v, a = mask_centroid(spec.mask_dir / f"{i:04d}.png")
            uv[i] = (u, v); area[i] = a
            valid[i] = a > 350 and a < 80000 and np.isfinite(u + v)
        uv = smooth_array(interpolate_missing(uv, valid), radius=3)
        half_h = 0.5 * spec.size[2]
        z = np.full(n, half_h, np.float64)
        if spec.held_range is not None:
            lo, hi = spec.held_range
            held_z = hand_center[:, 2] + (0.01 if spec.key != "tall_cylinder" else 0.10)
            held_z = np.maximum(half_h, held_z)
            z[lo:hi + 1] = held_z[lo:hi + 1]
            # Ease in/out instead of introducing a synthetic discontinuity.
            z = smooth_1d(z, radius=8)
        xyz = np.asarray([pixel_ray_to_height(uv[i, 0], uv[i, 1], z[i], K, rotations[i], tvecs[i], flip_world_z) for i in range(n)])
        xyz = smooth_array(xyz, radius=4)
        tracks[spec.key] = xyz
        diagnostics[spec.key] = {"valid_masks": int(valid.sum()), "median_component_area_px": float(np.median(area[valid])) if valid.any() else 0.0}
    return tracks, diagnostics


def build_plane_texture(frames: list[np.ndarray], homographies: np.ndarray, specs: list[ObjectSpec], K: np.ndarray,
                        left_cam: np.ndarray, right_cam: np.ndarray) -> np.ndarray:
    tw, th = 700, 450
    dst = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]], np.float32)
    samples = []
    for i in range(0, len(frames), 12):
        frame = frames[i]
        current = cv2.perspectiveTransform(
            np.array([[[307, 207], [744, 226], [829, 525], [115, 535]]], np.float32), homographies[i]
        )[0]
        W = cv2.getPerspectiveTransform(current.astype(np.float32), dst)
        warped = cv2.warpPerspective(frame, W, (tw, th), flags=cv2.INTER_LINEAR)
        dyn = np.zeros(frame.shape[:2], np.uint8)
        for spec in specs:
            m = cv2.imread(str(spec.mask_dir / f"{i:04d}.png"), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                dyn |= (m > 127).astype(np.uint8) * 255
        for vcam in (left_cam[i], right_cam[i]):
            p = np.c_[K[0, 0] * vcam[:, 0] / np.maximum(vcam[:, 2], 1e-5) + K[0, 2],
                      K[1, 1] * vcam[:, 1] / np.maximum(vcam[:, 2], 1e-5) + K[1, 2]].astype(np.int32)
            if len(p):
                hull = cv2.convexHull(p)
                cv2.fillConvexPoly(dyn, hull, 255)
        dyn = cv2.dilate(dyn, np.ones((25, 25), np.uint8))
        dyn_w = cv2.warpPerspective(dyn, W, (tw, th), flags=cv2.INTER_NEAREST)
        a = warped.astype(np.float32)
        a[dyn_w > 0] = np.nan
        samples.append(a)
    stack = np.stack(samples)
    with np.errstate(all="ignore"):
        texture = np.nanmedian(stack, axis=0)
    missing = ~np.all(np.isfinite(texture), axis=2)
    texture[~np.isfinite(texture)] = 0
    texture = np.clip(texture, 0, 255).astype(np.uint8)
    if missing.any():
        texture = cv2.inpaint(texture, missing.astype(np.uint8) * 255, 7, cv2.INPAINT_TELEA)
    return texture


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0.0, 0.0, 1.0])):
    forward = normalize(target - eye)
    right = normalize(np.cross(forward, up))
    true_up = normalize(np.cross(right, forward))
    R = np.stack([right, -true_up, forward], axis=0)
    t = -R @ eye
    return R, t


def project(points: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray):
    c = (R @ points.T).T + t
    z = c[:, 2]
    uv = np.c_[K[0, 0] * c[:, 0] / np.maximum(z, 1e-6) + K[0, 2],
               K[1, 1] * c[:, 1] / np.maximum(z, 1e-6) + K[1, 2]]
    return uv, z


def box_mesh(size: tuple[float, float, float]):
    sx, sy, sz = size
    v = np.array([[x, y, z] for z in (-sz/2, sz/2) for y in (-sy/2, sy/2) for x in (-sx/2, sx/2)], np.float64)
    f = np.array([[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],[2,3,7],[2,7,6],
                  [0,2,6],[0,6,4],[1,5,7],[1,7,3]], np.int32)
    return v, f


def cylinder_mesh(size: tuple[float, float, float], segments: int = 20):
    rx, ry, h = size[0] / 2, size[1] / 2, size[2]
    verts = []
    for z in (-h/2, h/2):
        for j in range(segments):
            a = 2 * math.pi * j / segments
            verts.append([rx * math.cos(a), ry * math.sin(a), z])
    verts += [[0, 0, -h/2], [0, 0, h/2]]
    faces = []
    for j in range(segments):
        k = (j + 1) % segments
        faces += [[j, k, segments+k], [j, segments+k, segments+j], [2*segments, k, j], [2*segments+1, segments+j, segments+k]]
    return np.asarray(verts, np.float64), np.asarray(faces, np.int32)


def add_mesh_primitives(primitives: list, vertices: np.ndarray, faces: np.ndarray, color: tuple[int, int, int],
                        Rv: np.ndarray, tv: np.ndarray, Kv: np.ndarray, stride: int = 1):
    uv, z = project(vertices, Rv, tv, Kv)
    for face in faces[::stride]:
        if np.any(z[face] <= 0.01):
            continue
        pts = np.rint(uv[face]).astype(np.int32)
        normal = np.cross(vertices[face[1]] - vertices[face[0]], vertices[face[2]] - vertices[face[0]])
        shade = 0.70 + 0.30 * abs(float(normal[2])) / max(float(np.linalg.norm(normal)), 1e-9)
        col = tuple(int(np.clip(x * shade, 0, 255)) for x in color)
        primitives.append((float(np.mean(z[face])), pts, col))


def draw_polyline_3d(img: np.ndarray, points: np.ndarray, Rv: np.ndarray, tv: np.ndarray, Kv: np.ndarray,
                     color: tuple[int, int, int], thickness: int = 2):
    if len(points) < 2:
        return
    uv, z = project(points, Rv, tv, Kv)
    for i in range(1, len(points)):
        if z[i-1] > 0.01 and z[i] > 0.01:
            cv2.line(img, tuple(np.rint(uv[i-1]).astype(int)), tuple(np.rint(uv[i]).astype(int)), color, thickness, cv2.LINE_AA)


def render_static_point_cloud(points: np.ndarray, colors_rgb: np.ndarray, Rv: np.ndarray, tv: np.ndarray,
                              Kv: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize an observed-only point cloud once for the fixed virtual camera."""
    layer = np.zeros((height, width, 3), np.uint8)
    support = np.zeros((height, width), np.uint8)
    uv, z = project(points, Rv, tv, Kv)
    pix = np.rint(uv).astype(np.int32)
    valid = (z > 0.01) & (pix[:,0] >= 1) & (pix[:,0] < width-1) & (pix[:,1] >= 1) & (pix[:,1] < height-1)
    pix, z, bgr = pix[valid], z[valid], colors_rgb[valid][:, ::-1]
    # Far to near, so later writes implement a compact z-buffer.
    order = np.argsort(z)[::-1]
    pix, bgr = pix[order], bgr[order]
    for dx, dy in ((0,0),(1,0),(0,1),(1,1)):
        layer[pix[:,1] + dy, pix[:,0] + dx] = bgr
        support[pix[:,1] + dy, pix[:,0] + dx] = 255
    return layer, support


def open_ffmpeg(path: Path, width: int, height: int, fps: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
           "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def main() -> None:
    cfg = parse_args()
    frame_files = sorted(cfg.frames_dir.glob("*.jpg"))
    frames = [cv2.imread(str(p)) for p in frame_files]
    if not frames or any(f is None for f in frames):
        raise SystemExit("missing or unreadable input frames")
    mano = np.load(cfg.mano_npz)
    left_cam = stabilize_mesh_global_translation(mano["left_vertices_camera"].astype(np.float64))
    right_cam = stabilize_mesh_global_translation(mano["right_vertices_camera"].astype(np.float64))
    n = min(len(frames), len(left_cam), len(right_cam))
    frames, left_cam, right_cam = frames[:n], left_cam[:n], right_cam[:n]
    H, W = frames[0].shape[:2]
    K = np.array([[600.0, 0.0, W/2], [0.0, 600.0, H/2], [0.0, 0.0, 1.0]], np.float64)
    K_object_ray = K.copy()
    ref_corners = np.array([[307, 207], [744, 226], [829, 525], [115, 535]], np.float32)

    homographies, tracked_corners, inliers, homography_valid = track_table_homographies(frames, ref_corners)
    rotations, tvecs, pnp_ok = solve_camera_poses(tracked_corners, K)
    flip_world_z = True
    camera_authority = "planar ORB/RANSAC visual tracking + one manually reviewed metric mat quadrilateral; not calibrated SLAM"
    background_authority = "observed table plane only, multi-frame median texture; unseen space intentionally neutral"
    camera_limitation = "Planar tracking cannot recover non-planar scene geometry or globally drift-free camera motion."
    static_points = static_colors = None
    virtual_eye = np.array([0.72, 0.66, 0.52])
    aligned_mat_world = None
    if cfg.vggt_aligned_npz is not None:
        aligned = np.load(cfg.vggt_aligned_npz)
        rotations = aligned["R_table_world_to_camera_m"][:n].astype(np.float64)
        tvecs = aligned["t_table_world_to_camera_m"][:n].astype(np.float64)
        static_points = aligned["static_points_table_world"].astype(np.float64)
        static_colors = aligned["static_colors_rgb"].astype(np.uint8)
        aligned_mat_world = aligned["mat_corners_source_order_table_world"].astype(np.float64)
        virtual_eye = np.array([0.72, -0.66, 0.52])
        kp = aligned["intrinsic_processed_pixels"][0].astype(np.float64)
        # VGGT-Omega predicts intrinsics in its resized image coordinates.
        # Scale them back to source pixels for SAM2 centroid ray casting.
        ph = float(kp[1, 2] * 2.0); pw = float(kp[0, 2] * 2.0)
        K_object_ray = kp.copy()
        K_object_ray[0, :] *= W / pw
        K_object_ray[1, :] *= H / ph
        flip_world_z = False
        camera_authority = "VGGT-Omega camera/depth prediction, similarity-aligned to the reviewed 0.70 x 0.45 m mat; not calibrated SLAM"
        background_authority = "observed-only VGGT-Omega static point cloud plus multi-frame table texture; dynamic masks excluded and unseen space left neutral"
        camera_limitation = "VGGT-Omega camera/depth are predictions in a table-aligned metric gauge, not calibrated sensor ground truth."
    hands_world = {"left": [], "right": []}
    for i in range(n):
        hands_world["left"].append(camera_to_physical_world(left_cam[i], rotations[i], tvecs[i], flip_world_z))
        hands_world["right"].append(camera_to_physical_world(right_cam[i], rotations[i], tvecs[i], flip_world_z))
    hands_world = {k: np.asarray(v) for k, v in hands_world.items()}
    support_corrections = {"left": np.zeros(n), "right": np.zeros(n)}
    if cfg.vggt_aligned_npz is not None:
        for side in ("left", "right"):
            hands_world[side], support_corrections[side] = enforce_observed_table_support(hands_world[side])

    specs = [
        ObjectSpec("alarm_clock", "alarm clock", "box", (0.105, 0.055, 0.105), (210, 215, 218), cfg.sam2_root / "alarm_clock_v2" / "masks", (0, 82)),
        ObjectSpec("small_cylinder", "small cylinder", "cylinder", (0.085, 0.085, 0.135), (214, 184, 164), cfg.sam2_root / "small_cylinder_v2" / "masks", (105, 218)),
        ObjectSpec("tall_cylinder", "tall cylinder", "cylinder", (0.095, 0.095, 0.31), (205, 212, 209), cfg.sam2_root / "tall_cylinder" / "masks", (258, n-1)),
    ]
    tracks, track_diag = object_tracks(specs, n, K_object_ray, rotations, tvecs, hands_world, flip_world_z)
    texture = build_plane_texture(frames, homographies, specs, K, left_cam, right_cam)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    recon_dir = cfg.output_dir / "reconstruction"
    render_dir = cfg.output_dir / "render"
    key_dir = render_dir / "keyframes"
    recon_dir.mkdir(exist_ok=True); render_dir.mkdir(exist_ok=True); key_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(recon_dir / "observed_table_median_texture.png"), texture)

    out_w, out_h = 960, 720
    Kv = np.array([[720.0, 0.0, out_w/2], [0.0, 720.0, out_h/2], [0.0, 0.0, 1.0]], np.float64)
    # A fixed side-oblique observer, independent of the moving ego camera.
    Rv, tv = look_at(virtual_eye, np.array([0.02, -0.01, 0.07]))
    mat_world = aligned_mat_world if aligned_mat_world is not None else np.array([[-.35,-.225,0],[.35,-.225,0],[.35,.225,0],[-.35,.225,0]], np.float64)
    table_world = np.array([[-.52,-.36,-.012],[.52,-.36,-.012],[.52,.36,-.012],[-.52,.36,-.012]], np.float64)
    uv_mat, _ = project(mat_world, Rv, tv, Kv)
    uv_table, _ = project(table_world, Rv, tv, Kv)
    tex_src = np.array([[0,0],[texture.shape[1]-1,0],[texture.shape[1]-1,texture.shape[0]-1],[0,texture.shape[0]-1]], np.float32)
    tex_to_view = cv2.getPerspectiveTransform(tex_src, uv_mat.astype(np.float32))
    warped_texture = cv2.warpPerspective(texture, tex_to_view, (out_w, out_h))
    texture_mask = cv2.warpPerspective(np.full(texture.shape[:2], 255, np.uint8), tex_to_view, (out_w, out_h))
    point_layer = point_support = None
    if static_points is not None:
        point_layer, point_support = render_static_point_cloud(static_points, static_colors, Rv, tv, Kv, out_w, out_h)

    third_path = render_dir / "fixed_third_view.mp4"
    compare_path = render_dir / "ego_vs_fixed_third_view.mp4"
    third_writer = open_ffmpeg(third_path, out_w, out_h, cfg.fps)
    compare_writer = open_ffmpeg(compare_path, 1920, 720, cfg.fps)
    left_centers = np.median(hands_world["left"], axis=1)
    right_centers = np.median(hands_world["right"], axis=1)

    for i in range(n):
        canvas = np.full((out_h, out_w, 3), (38, 40, 44), np.uint8)
        if point_layer is not None:
            canvas[point_support > 0] = point_layer[point_support > 0]
        # Only observed geometry is textured; everything else stays neutral.
        cv2.fillConvexPoly(canvas, np.rint(uv_table).astype(np.int32), (218, 216, 208))
        canvas[texture_mask > 0] = warped_texture[texture_mask > 0]
        primitives = []
        add_mesh_primitives(primitives, hands_world["left"][i], mano["left_faces"], (106, 154, 224), Rv, tv, Kv)
        add_mesh_primitives(primitives, hands_world["right"][i], mano["right_faces"], (104, 189, 139), Rv, tv, Kv)
        for spec in specs:
            v, f = box_mesh(spec.size) if spec.kind == "box" else cylinder_mesh(spec.size)
            v = v + tracks[spec.key][i]
            add_mesh_primitives(primitives, v, f, spec.color, Rv, tv, Kv)
        for _, pts, color in sorted(primitives, key=lambda x: x[0], reverse=True):
            cv2.fillConvexPoly(canvas, pts, color, lineType=cv2.LINE_AA)
        begin = max(0, i - 24)
        draw_polyline_3d(canvas, left_centers[begin:i+1], Rv, tv, Kv, (255, 110, 80), 2)
        draw_polyline_3d(canvas, right_centers[begin:i+1], Rv, tv, Kv, (80, 180, 255), 2)
        for spec in specs:
            draw_polyline_3d(canvas, tracks[spec.key][begin:i+1], Rv, tv, Kv, spec.color, 2)
        cv2.putText(canvas, "FIXED THIRD VIEW", (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (245,245,245), 2, cv2.LINE_AA)
        camera_label = "VGGT-Omega metric-aligned camera/depth" if cfg.vggt_aligned_npz is not None else "planar camera track"
        cv2.putText(canvas, f"HaWoR metric MANO + {camera_label} + SAM2 rigid proxies", (24, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230,230,230), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Unobserved volume is neutral (no DiT hallucination)", (24, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (190,196,205), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"frame {i:04d} | table-track inliers {int(inliers[i])}", (24, out_h-22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230,230,230), 1, cv2.LINE_AA)

        src = cv2.resize(frames[i], (1280, 720), interpolation=cv2.INTER_AREA)
        src_panel = src[:, 160:1120]
        cv2.putText(src_panel, "INPUT EGO RGB (t=4-16s)", (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,255), 2, cv2.LINE_AA)
        compare = np.hstack([src_panel, canvas])
        assert third_writer.stdin is not None and compare_writer.stdin is not None
        third_writer.stdin.write(canvas.tobytes())
        compare_writer.stdin.write(compare.tobytes())
        if i % 60 == 0 or i == n - 1:
            cv2.imwrite(str(key_dir / f"{i:04d}.jpg"), compare)
    for proc in (third_writer, compare_writer):
        assert proc.stdin is not None
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg encoding failed")

    np.savez_compressed(
        recon_dir / "world_state_tracks.npz",
        camera_R_world_to_camera=rotations,
        camera_t_world_to_camera=tvecs,
        table_homography_reference_to_frame=homographies,
        table_corners_px=tracked_corners,
        left_mano_vertices_table_world=hands_world["left"],
        right_mano_vertices_table_world=hands_world["right"],
        **{f"{k}_center_table_world": v for k, v in tracks.items()},
    )
    manifest = {
        "schema_version": "phiagent-ego-to-fixed-third-view/0.1",
        "input": {"frames_dir": str(cfg.frames_dir.resolve()), "frames": n, "fps": cfg.fps, "cropped_source_seconds": [4.0, 16.0]},
        "coordinate_frames": {
            "ha_wor_camera": "OpenCV-like x-right, y-down, z-forward; meters",
            "table_world": "origin=reviewed mat center; x=mat long edge, y=mat short edge, z=up; meters",
            "virtual_camera": {"eye_table_world_m": virtual_eye.tolist(), "target_table_world_m": [0.02, -0.01, 0.07]},
        },
        "authority": {
            "hands": "HaWoR prediction reconstructed with licensed MANO assets; not ground truth",
            "camera": camera_authority,
            "objects": "SAM2 temporal masks + metric ray/height proxy; identity/mesh/pose are not CAD-ground-truth",
            "background": background_authority,
            "interaction": "visually reviewed heuristic held intervals; no contact force or physics simulation",
        },
        "object_specs": [{"key": s.key, "label": s.label, "kind": s.kind, "size_m": list(s.size), "held_frame_range": list(s.held_range) if s.held_range else None} for s in specs],
        "diagnostics": {
            "planar_texture_registration_inliers": {"median": float(np.median(inliers)), "min": int(np.min(inliers)), "frames_below_15": int(np.sum(inliers < 15)), "accepted_frames": int(homography_valid.sum()), "interpolated_frames": int((~homography_valid).sum())},
            "planar_pnp_success_frames_diagnostic_only": int(pnp_ok.sum()),
            "object_masks": track_diag,
            "mano_table_support_correction_m": {side: {"median": float(np.median(delta)), "p95": float(np.percentile(delta, 95)), "max": float(np.max(delta))} for side, delta in support_corrections.items()},
        },
        "outputs": {"third_view": str(third_path.resolve()), "comparison": str(compare_path.resolve()), "world_state": str((recon_dir / "world_state_tracks.npz").resolve())},
        "limitations": [
            camera_limitation,
            "Proxy objects do not represent unseen backsides, exact CAD geometry, material, contact, or articulation.",
            "This clip provides no camera calibration, action labels, force/torque, or task state ground truth.",
        ],
    }
    (cfg.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"comparison": str(compare_path), "third_view": str(third_path), "diagnostics": manifest["diagnostics"]}, indent=2))


if __name__ == "__main__":
    main()
