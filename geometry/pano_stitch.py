"""
pano_stitch.py — Stitch PTZ frames into an equirectangular panorama.

Two strategies, tried in order:

  1. OpenCV Stitcher (SCANS/PANORAMA mode) — feature-based auto-stitching
     with proven exposure compensation, seam finding, and multi-band
     blending. Best quality WHEN frames have enough texture overlap.

  2. Pose-based equirectangular projection — uses the KNOWN pan/tilt
     angles from the PTZ to place each frame on a sphere directly.
     Never fails on blank walls / low texture (common indoors) because
     it doesn't rely on feature matching. Accuracy depends on the
     camera angle + FOV calibration in config.

For fall-risk room scans we default to trying OpenCV first (cleaner seams),
then fall back to pose-based, which always produces *something* usable.
You can force pose-based with config: processing.stitch_method = "pose".

Pipeline stage: STITCH (server, CPU-bound).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ───────────────────────── pose-based projection ─────────────────────────

def _normalized_to_degrees(value: float, min_deg: float,
                           max_deg: float) -> float:
    """Map ONVIF normalized -1..1 to a physical degree range."""
    frac = (value + 1.0) / 2.0
    return min_deg + frac * (max_deg - min_deg)


def _rotation_matrix(pan_deg: float, tilt_deg: float) -> np.ndarray:
    """Rotation matrix: pan about Y (yaw), tilt about X (pitch)."""
    p = math.radians(pan_deg)
    t = math.radians(tilt_deg)
    Ry = np.array([
        [ math.cos(p), 0, math.sin(p)],
        [ 0,           1, 0          ],
        [-math.sin(p), 0, math.cos(p)],
    ])
    Rx = np.array([
        [1, 0,            0           ],
        [0, math.cos(t), -math.sin(t) ],
        [0, math.sin(t),  math.cos(t) ],
    ])
    return Ry @ Rx


class PanoramaStitcher:
    """Pose-based equirectangular projector (the reliable fallback)."""

    def __init__(self, config: dict):
        cam = config["camera"]
        self.pan_min_deg  = cam.get("pan_min_deg", -170.0)
        self.pan_max_deg  = cam.get("pan_max_deg",  170.0)
        self.tilt_min_deg = cam.get("tilt_min_deg", -30.0)
        self.tilt_max_deg = cam.get("tilt_max_deg",  90.0)
        self.hfov_deg     = cam.get("hfov_deg", 65.0)
        self.vfov_deg     = cam.get("vfov_deg", 40.0)

        proc = config.get("processing", {})
        self.pano_width  = proc.get("pano_width", 4096)
        self.pano_height = self.pano_width // 2

    def _frame_pose_degrees(self, fm: dict) -> Tuple[float, float]:
        return (
            _normalized_to_degrees(fm["pan"], self.pan_min_deg, self.pan_max_deg),
            _normalized_to_degrees(fm["tilt"], self.tilt_min_deg, self.tilt_max_deg),
        )

    def stitch(self, frames_dir: Path, frame_metas: List[dict],
               progress=True) -> Tuple[np.ndarray, np.ndarray]:
        W, H = self.pano_width, self.pano_height
        accum  = np.zeros((H, W, 3), dtype=np.float64)
        weight = np.zeros((H, W),    dtype=np.float64)

        lon = (np.linspace(0, W - 1, W) / W) * 2 * np.pi - np.pi
        lat = np.pi/2 - (np.linspace(0, H - 1, H) / H) * np.pi
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        dx = np.cos(lat_grid) * np.sin(lon_grid)
        dy = np.sin(lat_grid)
        dz = np.cos(lat_grid) * np.cos(lon_grid)
        rays = np.stack([dx, dy, dz], axis=-1)

        hfov = math.radians(self.hfov_deg)
        vfov = math.radians(self.vfov_deg)
        fx = 0.5 / math.tan(hfov / 2)
        fy = 0.5 / math.tan(vfov / 2)

        n = len(frame_metas)
        for idx, fm in enumerate(frame_metas):
            if not fm.get("captured"):
                continue
            img_path = frames_dir / fm["filename"]
            if not img_path.exists():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            ih, iw = img.shape[:2]

            pan_deg, tilt_deg = self._frame_pose_degrees(fm)
            R = _rotation_matrix(pan_deg, tilt_deg)
            cam_rays = rays @ R

            cz = cam_rays[..., 2]
            valid = cz > 1e-6
            with np.errstate(divide="ignore", invalid="ignore"):
                u = (cam_rays[..., 0] / cz) * fx + 0.5
                v = 0.5 - (cam_rays[..., 1] / cz) * fy

            in_fov = valid & (u >= 0) & (u < 1) & (v >= 0) & (v < 1)
            if not np.any(in_fov):
                continue

            sx = (u * (iw - 1)).astype(np.float32)
            sy = (v * (ih - 1)).astype(np.float32)
            edge = np.maximum(np.abs(u - 0.5), np.abs(v - 0.5)) * 2
            w = np.clip(1.0 - edge, 0.0, 1.0) ** 2 * in_fov

            map_x = np.where(in_fov, sx, 0).astype(np.float32)
            map_y = np.where(in_fov, sy, 0).astype(np.float32)
            sampled = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT)
            accum  += sampled.astype(np.float64) * w[..., None]
            weight += w

            if progress:
                print(f"  [pose-stitch] {idx+1}/{n} "
                      f"(pan={pan_deg:+.0f} tilt={tilt_deg:+.0f})", flush=True)

        safe = weight[..., None].copy()
        safe[safe == 0] = 1.0
        pano = (accum / safe).clip(0, 255).astype(np.uint8)
        return pano, weight.astype(np.float32)


# ───────────────────────── OpenCV auto-stitch ────────────────────────────

def opencv_stitch(image_paths: List[Path]) -> Optional[np.ndarray]:
    """
    Try OpenCV's high-level Stitcher in SCANS mode (good for rotational
    panoramas). Returns the stitched image, or None if it fails (which
    happens on low-overlap / low-texture indoor scenes).
    """
    imgs = []
    for p in image_paths:
        im = cv2.imread(str(p))
        if im is not None:
            # Downscale very large frames a bit — faster + better matching
            h, w = im.shape[:2]
            if w > 1600:
                scale = 1600 / w
                im = cv2.resize(im, (int(w*scale), int(h*scale)))
            imgs.append(im)
    if len(imgs) < 2:
        return None

    # SCANS mode assumes affine/rotational capture (our PTZ case)
    try:
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    except AttributeError:
        stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)

    status, pano = stitcher.stitch(imgs)
    if status == cv2.Stitcher_OK:
        return pano

    # Retry in SCANS mode (sometimes succeeds where PANORAMA fails)
    try:
        stitcher2 = cv2.Stitcher_create(cv2.Stitcher_SCANS)
        status2, pano2 = stitcher2.stitch(imgs)
        if status2 == cv2.Stitcher_OK:
            return pano2
    except Exception:
        pass

    status_names = {
        1: "ERR_NEED_MORE_IMGS",
        2: "ERR_HOMOGRAPHY_EST_FAIL",
        3: "ERR_CAMERA_PARAMS_ADJUST_FAIL",
    }
    print(f"  [opencv-stitch] failed: {status_names.get(status, status)}")
    return None


# ───────────────────────── pipeline entry ────────────────────────────────

def run_stitch(session, config):
    """
    Pipeline entry point. Tries OpenCV auto-stitch first (unless config
    forces pose mode), falls back to pose-based projection.
    """
    print("\n[Stitch] Building panorama...")
    session.set_status("stitch", "running")

    m = session.manifest
    captured = [f for f in m.frames if f.captured]
    if not captured:
        print("[Stitch] No captured frames!")
        session.set_status("stitch", "failed")
        return None

    method = config.get("processing", {}).get("stitch_method", "auto")
    pano = None
    used = ""

    # Try OpenCV auto-stitch
    if method in ("auto", "opencv"):
        print(f"[Stitch] Trying OpenCV auto-stitch on {len(captured)} frames...")
        paths = [session.frames_dir / f.filename for f in captured]
        pano = opencv_stitch(paths)
        if pano is not None:
            used = "opencv"
            print("[Stitch] OpenCV auto-stitch succeeded.")
        elif method == "opencv":
            print("[Stitch] OpenCV failed and method=opencv (no fallback).")

    # Fall back to (or force) pose-based projection
    if pano is None and method in ("auto", "pose"):
        print("[Stitch] Using pose-based equirectangular projection...")
        frame_metas = [{
            "filename": f.filename, "pan": f.pan, "tilt": f.tilt,
            "zoom": f.zoom, "captured": f.captured,
        } for f in m.frames]
        stitcher = PanoramaStitcher(config)
        pano, coverage = stitcher.stitch(session.frames_dir, frame_metas)
        used = "pose"

        cov_vis = (coverage / max(coverage.max(), 1e-6) * 255).astype(np.uint8)
        cov_path = session.outputs_dir / "panorama_coverage.jpg"
        cv2.imwrite(str(cov_path), cov_vis)
        session.set_output("panorama_coverage", str(cov_path))
        gaps = float((coverage == 0).mean() * 100)
        print(f"[Stitch] Coverage gaps: {gaps:.1f}%")
        if gaps > 30:
            print("[Stitch] ⚠ Large gaps — add tilt rows or widen FOV in config")

    if pano is None:
        print("[Stitch] FAILED — no panorama produced.")
        session.set_status("stitch", "failed")
        return None

    out_path = session.outputs_dir / "panorama.jpg"
    cv2.imwrite(str(out_path), pano, [cv2.IMWRITE_JPEG_QUALITY, 92])
    session.set_output("panorama", str(out_path))
    session.set_output("stitch_method", used)
    session.set_status("stitch", "done")

    h, w = pano.shape[:2]
    print(f"[Stitch] Panorama {w}x{h} via '{used}' -> {out_path.name}")
    return pano

