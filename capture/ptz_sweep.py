"""
ptz_sweep.py — Sweep orchestrator (coordinates the two nodes).

Two-node architecture:
  - StreamNode (capture/stream_node.py): owns the video stream, always has
    the freshest clean decoded frame. Runs continuously in background.
  - PTZNode   (capture/ptz_node.py):    owns camera motion only.

The sweep loop:
  1. PTZNode.move_absolute(pan, tilt)   -> camera starts moving
  2. sleep(settle_time)                  -> camera physically reaches & stops
  3. StreamNode.wait_for_fresh_frame()   -> get a frame decoded AFTER the move
  4. corruption check (std-dev)          -> re-request if artifact slipped in
  5. save frame + pose metadata

Because the stream runs in its own node with a leaky queue and proper
keyframe boundaries, we never grab a half-decoded buffer -- no HEVC
cu_qp_delta corruption. Because we wait for frames newer than the move,
we never save duplicates of the previous position.
"""
from __future__ import annotations

import math
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from .ptz_node import PTZNode
from .session import FrameMeta, Session
from .stream_node import StreamNode

PAN_DEGREES_PER_UNIT  = 90.0
TILT_DEGREES_PER_UNIT = 45.0

# Frames whose std-dev is below this are corrupt (solid grey / HEVC artifact)
CORRUPTION_STD_THRESHOLD = 3.0


class PTZSweep:
    """Coordinates PTZNode + StreamNode to capture a grid of room views."""

    def __init__(self, config: dict, session: Session):
        self.cfg = config
        self.session = session
        self._ptz: Optional[PTZNode] = None
        self._stream: Optional[StreamNode] = None
        self._motion_threshold = config["sweep"].get("motion_threshold", 1.5)

    # -- Grid -------------------------------------------------------------

    def _build_grid(self):
        s = self.cfg["sweep"]
        pan_min, pan_max   = s["pan_min"],  s["pan_max"]
        tilt_min, tilt_max = s["tilt_min"], s["tilt_max"]
        zoom  = s["zoom"]
        snake = s.get("snake_pattern", True)
        invert_pan = s.get("invert_pan", False)

        if invert_pan:
            pans = np.linspace(pan_max, pan_min, s["pan_steps"])
        else:
            pans = np.linspace(pan_min, pan_max, s["pan_steps"])
        tilts = np.linspace(tilt_max, tilt_min, s["tilt_steps"])  # top->bottom

        grid = []
        for row_idx, tilt in enumerate(tilts):
            row_pans = pans if (not snake or row_idx % 2 == 0) else pans[::-1]
            for pan in row_pans:
                grid.append((float(pan), float(tilt), float(zoom)))
        return grid

    def _to_degrees(self, pan: float, tilt: float) -> Tuple[float, float]:
        return pan * PAN_DEGREES_PER_UNIT, tilt * TILT_DEGREES_PER_UNIT

    # -- Connect both nodes ----------------------------------------------

    def connect(self) -> bool:
        cam = self.cfg["camera"]

        # Node 2: PTZ control
        self._ptz = PTZNode(cam["ip"], cam["port"], cam["user"], cam["password"])
        print(f"[Sweep] Connecting PTZ control to {cam['ip']}:{cam['port']} ...")
        if not self._ptz.connect():
            return False
        self._ptz.setup(rtsp_port=cam.get("rtsp_port", 554))

        # Node 1: video stream
        rtsp_url = cam.get("rtsp_url") or self._ptz.stream_uri
        self._stream = StreamNode(
            uri=rtsp_url,
            codec=cam.get("codec", "h265"),
            width=self.cfg["sweep"]["capture_width"],
            height=self.cfg["sweep"]["capture_height"],
            use_tcp=True,
            prefer_hw=True,
        )
        print(f"[Sweep] Opening video stream node...")
        if not self._stream.start():
            print("[Sweep] Stream node failed to open")
            return False

        print(f"[Sweep] Both nodes up. Stream via {self._stream.opened_with()}")
        return True

    # -- Capture one settled, clean frame --------------------------------

    def _capture_settled_frame(self, settle_time: float,
                                min_new_frames: int,
                                max_corrupt_retries: int = 4
                                ) -> Optional[np.ndarray]:
        """
        Capture a frame ONLY after the camera has come to a complete stop.

        Sequence:
          1. sleep(settle_time)              -- let the bulk of the move finish
          2. ptz.stop()                      -- force zero velocity
          3. stream.wait_until_static()      -- confirm scene stopped changing
          4. corruption check (std-dev)      -- reject HEVC artifacts
        """
        # Initial settle — let the camera do most of its travel
        time.sleep(settle_time)

        # Force a hard stop so there is zero residual velocity / drift
        self._ptz.stop()

        frame = None
        for attempt in range(max_corrupt_retries):
            # Wait until the image literally stops changing (camera static)
            frame = self._stream.wait_until_static(
                motion_threshold=self._motion_threshold,
                stable_checks=3,
                timeout=8.0,
            )
            if frame is None:
                if not self._stream.is_running():
                    print("[stream died]", end=" ", flush=True)
                    return None
                continue

            # Reject corrupt frames (solid grey / HEVC artifact)
            if float(frame.std()) >= CORRUPTION_STD_THRESHOLD:
                return frame

            print(f"[corrupt std={frame.std():.1f}]", end=" ", flush=True)

        return frame

    # -- Full sweep -------------------------------------------------------

    def run(self, resume: bool = True) -> bool:
        if not self.connect():
            print("[Sweep] Failed to connect")
            return False

        s = self.cfg["sweep"]
        fmt = s.get("capture_format", "png")
        settle = s.get("settle_time", 1.2)
        min_new = s.get("min_new_frames", 8)
        speed = s.get("ptz_speed", 0.5)
        rotate180 = s.get("rotate_image_180", True)
        grid = self._build_grid()
        total = len(grid)

        print(f"\n[Sweep] {s['pan_steps']}x{s['tilt_steps']} grid = {total} views")
        print(f"[Sweep] settle={settle}s  min_new_frames={min_new}  "
              f"speed={speed}  rotate180={rotate180}\n")

        self.session.set_status("capture", "running")

        m = self.session.manifest
        if not m.frames:
            for i, (pan, tilt, zoom) in enumerate(grid):
                pan_deg, tilt_deg = self._to_degrees(pan, tilt)
                m.frames.append(FrameMeta(
                    frame_id=i, filename=f"frame_{i:04d}.{fmt}",
                    pan=pan, tilt=tilt, zoom=zoom,
                    pan_deg=pan_deg, tilt_deg=tilt_deg, timestamp="",
                    width=s["capture_width"], height=s["capture_height"],
                    captured=False,
                ))
            self.session.save()

        failed = []
        try:
            for i, (pan, tilt, zoom) in enumerate(grid):
                fm = m.frames[i]
                if resume and fm.captured:
                    print(f"[Sweep] {i+1:3d}/{total}  SKIP (done)")
                    continue

                print(f"[Sweep] {i+1:3d}/{total}  pan={pan:+.2f} tilt={tilt:+.2f}  "
                      f"move...", end=" ", flush=True)
                self._ptz.move_absolute(pan, tilt, zoom, speed=speed)

                print("settle...", end=" ", flush=True)
                frame = self._capture_settled_frame(settle, min_new)

                if frame is None:
                    print("FAILED")
                    failed.append(i)
                    continue

                if rotate180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)

                out = self.session.frame_path(i, fmt)
                if fmt == "png":
                    cv2.imwrite(str(out), frame)
                else:
                    cv2.imwrite(str(out), frame,
                                [cv2.IMWRITE_JPEG_QUALITY, 95])

                fm.captured = True
                fm.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
                self.session.save()
                print(f"OK  {out.name}")
        except KeyboardInterrupt:
            print("\n[Sweep] Interrupted by user")

        print("\n[Sweep] Returning home...")
        self._ptz.go_home()
        self._stream.stop()

        n = len(self.session.captured_frames())
        if n == total:
            self.session.set_status("capture", "done")
            print(f"\n[Sweep] Complete! {n}/{total} captured.")
            return True
        else:
            self.session.set_status("capture", "failed")
            print(f"\n[Sweep] Partial: {n}/{total}. Failed: {failed}")
            print("[Sweep] Re-run to retry failed frames.")
            return False

    # -- Quick preview ----------------------------------------------------

    def preview(self, n_frames: int = 5):
        if not self.connect():
            return
        s = self.cfg["sweep"]
        settle = s.get("settle_time", 1.2)
        min_new = s.get("min_new_frames", 8)
        speed = s.get("ptz_speed", 0.5)
        rotate180 = s.get("rotate_image_180", True)

        pans = np.linspace(s["pan_min"], s["pan_max"], n_frames)
        tilt = (s["tilt_min"] + s["tilt_max"]) / 2.0

        print(f"\n[Preview] {n_frames} frames at tilt={tilt:.2f}")
        cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Preview", 960, 540)

        try:
            for i, pan in enumerate(pans):
                print(f"[Preview] {i+1}/{n_frames}  pan={pan:+.2f}  move...",
                      end=" ", flush=True)
                self._ptz.move_absolute(pan, tilt, 0.0, speed=speed)
                frame = self._capture_settled_frame(settle, min_new)
                if frame is not None:
                    if rotate180:
                        frame = cv2.rotate(frame, cv2.ROTATE_180)
                    cv2.putText(frame, f"Preview {i+1}/{n_frames} pan={pan:+.2f}",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                (0, 255, 0), 2)
                    cv2.imshow("Preview", frame)
                    print("shown")
                    if cv2.waitKey(800) in (ord('q'), 27):
                        break
                else:
                    print("FAILED")
        except KeyboardInterrupt:
            pass

        self._ptz.go_home()
        self._stream.stop()
        cv2.destroyAllWindows()
        print("[Preview] Done.")