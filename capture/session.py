"""
session.py — Session folder management for room scans.

Each scan is stored in sessions/<session_name>/ with a manifest.json
tracking metadata, captured frames, and processing status.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

SESSION_VERSION = "1.0"


@dataclass
class FrameMeta:
    """Metadata for a single captured frame."""
    frame_id: int
    filename: str
    pan: float          # ONVIF normalized pan  (-1.0 to 1.0)
    tilt: float         # ONVIF normalized tilt (-1.0 to 1.0)
    zoom: float         # ONVIF normalized zoom (0.0 to 1.0)
    pan_deg: float      # Estimated degrees (for projection geometry)
    tilt_deg: float     # Estimated degrees
    timestamp: str
    width: int
    height: int
    captured: bool = False


@dataclass
class SessionManifest:
    """Top-level manifest for a scan session."""
    session_name: str
    version: str = SESSION_VERSION
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Camera config snapshot
    camera_ip: str = ""
    camera_port: int = 2000
    rtsp_url: str = ""

    # Sweep config snapshot
    pan_steps: int = 8
    tilt_steps: int = 4
    pan_min: float = -1.0
    pan_max: float = 1.0
    tilt_min: float = -0.3
    tilt_max: float = 0.5
    zoom: float = 0.0

    # Frames
    frames: List[FrameMeta] = field(default_factory=list)

    # Processing status
    status: Dict[str, str] = field(default_factory=lambda: {
        "capture":   "pending",
        "depth":     "pending",
        "stitch":    "pending",
        "segment":   "pending",
        "pointcloud":"pending",
        "analysis":  "pending",
        "splat":     "pending",
    })

    # Output files
    outputs: Dict[str, str] = field(default_factory=dict)


class Session:
    """Manages a scan session on disk."""

    def __init__(self, sessions_dir: str, session_name: str):
        self.root = Path(sessions_dir) / session_name
        self.manifest_path = self.root / "manifest.json"
        self.frames_dir = self.root / "frames"
        self.depth_dir = self.root / "depth"
        self.masks_dir = self.root / "masks"
        self.outputs_dir = self.root / "outputs"
        self._manifest: Optional[SessionManifest] = None

    # ── Lifecycle ────────────────────────────────────────────────────────

    def create(self, config: dict) -> SessionManifest:
        """Create a new session from config dict."""
        if self.root.exists():
            raise FileExistsError(f"Session already exists: {self.root}")

        for d in [self.root, self.frames_dir, self.depth_dir,
                  self.masks_dir, self.outputs_dir]:
            d.mkdir(parents=True, exist_ok=True)

        m = SessionManifest(
            session_name=self.root.name,
            camera_ip=config.get("camera", {}).get("ip", ""),
            camera_port=config.get("camera", {}).get("port", 2000),
            rtsp_url=config.get("camera", {}).get("rtsp_url", ""),
            pan_steps=config["sweep"]["pan_steps"],
            tilt_steps=config["sweep"]["tilt_steps"],
            pan_min=config["sweep"]["pan_min"],
            pan_max=config["sweep"]["pan_max"],
            tilt_min=config["sweep"]["tilt_min"],
            tilt_max=config["sweep"]["tilt_max"],
            zoom=config["sweep"]["zoom"],
        )
        self._manifest = m
        self.save()
        print(f"[Session] Created: {self.root}")
        return m

    def load(self) -> SessionManifest:
        """Load an existing session from disk."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"No manifest at {self.manifest_path}")
        data = json.loads(self.manifest_path.read_text())
        # Reconstruct dataclass
        frames = [FrameMeta(**f) for f in data.pop("frames", [])]
        m = SessionManifest(**data)
        m.frames = frames
        self._manifest = m
        return m

    def save(self):
        """Save manifest to disk."""
        if self._manifest is None:
            raise RuntimeError("No manifest loaded")
        self._manifest.updated_at = datetime.now().isoformat()
        self.manifest_path.write_text(
            json.dumps(asdict(self._manifest), indent=2)
        )

    def delete(self):
        """Remove the session directory entirely."""
        if self.root.exists():
            shutil.rmtree(self.root)
            print(f"[Session] Deleted: {self.root}")

    # ── Convenience ──────────────────────────────────────────────────────

    @property
    def manifest(self) -> SessionManifest:
        if self._manifest is None:
            self.load()
        return self._manifest

    def frame_path(self, frame_id: int, fmt: str = "png") -> Path:
        return self.frames_dir / f"frame_{frame_id:04d}.{fmt}"

    def depth_path(self, frame_id: int) -> Path:
        return self.depth_dir / f"depth_{frame_id:04d}.npy"

    def mask_path(self, frame_id: int) -> Path:
        return self.masks_dir / f"masks_{frame_id:04d}.json"

    def set_status(self, stage: str, status: str):
        """Update a processing stage status and save."""
        self.manifest.status[stage] = status
        self.save()

    def set_output(self, key: str, path: str):
        """Record an output file path and save."""
        self.manifest.outputs[key] = path
        self.save()

    def add_frame(self, frame: FrameMeta):
        """Add a frame to the manifest and save."""
        self.manifest.frames.append(frame)
        self.save()

    def captured_frames(self) -> List[FrameMeta]:
        return [f for f in self.manifest.frames if f.captured]

    def summary(self) -> str:
        m = self.manifest
        n_captured = len(self.captured_frames())
        n_total = len(m.frames)
        lines = [
            f"Session:  {m.session_name}",
            f"Created:  {m.created_at[:19]}",
            f"Grid:     {m.pan_steps} pan × {m.tilt_steps} tilt = {n_total} views",
            f"Captured: {n_captured}/{n_total} frames",
            f"Status:",
        ]
        for stage, status in m.status.items():
            icon = {"done": "✓", "running": "⟳", "failed": "✗",
                    "pending": "·"}.get(status, "?")
            lines.append(f"  {icon} {stage:<12} {status}")
        return "\n".join(lines)


def list_sessions(sessions_dir: str) -> List[str]:
    """Return names of all sessions in the sessions directory."""
    p = Path(sessions_dir)
    if not p.exists():
        return []
    return sorted([
        d.name for d in p.iterdir()
        if d.is_dir() and (d / "manifest.json").exists()
    ])