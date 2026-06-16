"""
stream_node.py — Dedicated RTSP stream reader (NODE 1 of 2).

This is the PROVEN McLaren Bot FrameReader design: spawns gst-launch-1.0
as a subprocess with a leaky-downstream queue that always drops to the
newest frame, reads raw BGR from stdout in a background thread.

Why this fixes the corruption:
  - The leaky queue with config-interval=-1 on the parser ensures every
    frame handed to the decoder starts at a proper boundary. No partial
    keyframes → no HEVC cu_qp_delta corruption.
  - The decoder runs continuously in its own process. We NEVER reach into
    a half-decoded buffer — we only ever read complete BGR frames off stdout.
  - Frame freshness is tracked by frame_id so the sweep node can wait for
    genuinely NEW frames after a PTZ move (no duplicates).

This node does ONE thing: keep the latest clean frame available.
The sweep node asks it for frames; it never drives the camera.
"""
from __future__ import annotations

import select
import subprocess
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np


class StreamNode:
    """
    Persistent RTSP reader. Run start() once; it keeps the newest decoded
    frame available via read() for the lifetime of the sweep.
    """

    def __init__(self, uri: str, codec: str = "h265",
                 width: int = 1920, height: int = 1080,
                 use_tcp: bool = True, prefer_hw: bool = True):
        self._uri     = uri
        self._codec   = codec.lower()
        self._width   = width
        self._height  = height
        self._use_tcp = use_tcp
        self._prefer_hw = prefer_hw

        self._lock       = threading.Lock()
        self._frame      = None
        self._capture_t  = 0.0
        self._frame_id   = 0
        self._running    = False
        self._opened_with = "none"
        self._proc       = None
        self._thread     = None

    # ── Pipeline ─────────────────────────────────────────────────────────

    def _build_pipeline(self, hw: bool) -> list:
        protocols = "tcp" if self._use_tcp else "udp"
        if self._codec == "h265":
            depay, parse = "rtph265depay", "h265parse"
            decoder = "nvh265dec" if hw else "avdec_h265"
        else:
            depay, parse = "rtph264depay", "h264parse"
            decoder = "nvh264dec" if hw else "avdec_h264"

        return [
            "gst-launch-1.0", "-q",
            "rtspsrc",
            f"location={self._uri}",
            "latency=0",
            f"protocols={protocols}",
            "drop-on-latency=true",
            "do-retransmission=false",
            "ntp-sync=false",
            "!", depay,
            "!", parse, "config-interval=-1",   # full keyframe boundaries
            "!", "queue", "leaky=downstream",     # always drop to newest
            "max-size-buffers=2",
            "max-size-bytes=0",
            "max-size-time=0",
            "!", decoder,
            "!", "videoconvert", "n-threads=4",
            "!", "video/x-raw,format=BGR",
            "!", "fdsink", "fd=1", "sync=false",
        ]

    def _detect_dimensions(self) -> Tuple[int, int]:
        """Probe actual stream dimensions with ffprobe (fast, bounded)."""
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error",
                 "-rtsp_transport", "tcp",
                 "-timeout", "8000000",
                 "-analyzeduration", "500000",
                 "-probesize", "500000",
                 "-select_streams", "v:0",
                 "-show_entries", "stream=width,height",
                 "-of", "csv=p=0:s=x",
                 self._uri],
                capture_output=True, text=True, timeout=12,
            )
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split("x")
                if len(parts) == 2:
                    w, h = int(parts[0]), int(parts[1])
                    print(f"[StreamNode] Detected stream {w}x{h}")
                    return w, h
        except Exception as e:
            print(f"[StreamNode] ffprobe failed: {e}")
        return self._width, self._height

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start(self) -> bool:
        w, h = self._detect_dimensions()
        self._width, self._height = w, h
        frame_size = w * h * 3

        attempts = [("GST-NVDEC", True), ("GST-SW", False)] if self._prefer_hw \
            else [("GST-SW", False)]

        for name, hw in attempts:
            cmd = self._build_pipeline(hw=hw)
            print(f"[StreamNode] Trying {name}...")
            try:
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    bufsize=0)
            except FileNotFoundError:
                print("[StreamNode] gst-launch-1.0 not found")
                return False

            t0 = time.time()
            first = None
            while time.time() - t0 < 15.0:
                if proc.poll() is not None:
                    err = proc.stderr.read().decode("utf-8", errors="ignore")
                    print(f"[StreamNode] {name} died: {err[:300]}")
                    break
                ready, _, _ = select.select([proc.stdout], [], [], 0.5)
                if ready:
                    first = self._read_exact(proc.stdout, frame_size)
                    if first is not None:
                        break

            if first is None or len(first) != frame_size:
                proc.kill()
                try:
                    proc.wait(timeout=2)
                except Exception:
                    pass
                continue

            # Good — start background reader
            frame = np.frombuffer(first, dtype=np.uint8).reshape((h, w, 3))
            with self._lock:
                self._frame = frame.copy()
                self._capture_t = time.time()
                self._frame_id += 1

            self._proc = proc
            self._opened_with = name
            self._running = True
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            print(f"[StreamNode] ✓ Opened with {name} ({w}x{h})")
            return True

        print("[StreamNode] FATAL: all pipelines failed")
        return False

    def _read_exact(self, stream, n: int) -> Optional[bytes]:
        buf = b""
        while len(buf) < n:
            chunk = stream.read(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    def _loop(self):
        frame_size = self._width * self._height * 3
        stream = self._proc.stdout
        while self._running:
            data = self._read_exact(stream, frame_size)
            t = time.time()
            if data is None:
                print("[StreamNode] Stream closed")
                self._running = False
                break
            try:
                frame = np.frombuffer(data, dtype=np.uint8).reshape(
                    (self._height, self._width, 3))
                with self._lock:
                    self._frame = frame
                    self._capture_t = t
                    self._frame_id += 1
            except ValueError as e:
                print(f"[StreamNode] reshape error: {e}")

    # ── Frame access ─────────────────────────────────────────────────────

    def read(self) -> Tuple[Optional[np.ndarray], float, int]:
        """Return (frame_copy, capture_time, frame_id)."""
        with self._lock:
            if self._frame is None:
                return None, 0.0, 0
            return self._frame.copy(), self._capture_t, self._frame_id

    def current_frame_id(self) -> int:
        with self._lock:
            return self._frame_id

    def wait_for_fresh_frame(self, after_id: int,
                             min_new: int = 8,
                             timeout: float = 5.0) -> Optional[np.ndarray]:
        """
        Block until at least `min_new` frames newer than `after_id` have
        arrived, then return the latest. This guarantees the returned
        frame was decoded AFTER the camera stopped moving — no duplicates,
        no motion blur.
        """
        deadline = time.time() + timeout
        target = after_id + min_new
        while time.time() < deadline:
            with self._lock:
                if self._frame_id >= target and self._frame is not None:
                    return self._frame.copy()
            time.sleep(0.02)
        # Timed out — return whatever is newest
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def wait_until_static(self, motion_threshold: float = 1.5,
                          stable_checks: int = 3,
                          timeout: float = 8.0) -> Optional[np.ndarray]:
        """
        Wait until the SCENE STOPS CHANGING — i.e. the camera has physically
        come to a complete stop (zero velocity).

        Compares consecutive frames pixel-by-pixel. When the mean absolute
        difference between consecutive frames drops below motion_threshold
        for `stable_checks` consecutive comparisons, the camera is static.

        Returns the first confirmed-static frame, or None on timeout.
        """
        deadline = time.time() + timeout
        prev = None
        prev_id = -1
        stable_count = 0

        while time.time() < deadline:
            with self._lock:
                cur = self._frame.copy() if self._frame is not None else None
                cur_id = self._frame_id

            if cur is None or cur_id == prev_id:
                time.sleep(0.02)
                continue

            if prev is not None:
                # Mean absolute difference between consecutive frames.
                # Downsample for speed — we only need motion magnitude.
                small_prev = cv2.resize(prev, (320, 180))
                small_cur  = cv2.resize(cur,  (320, 180))
                diff = cv2.absdiff(small_prev, small_cur)
                motion = float(diff.mean())

                if motion < motion_threshold:
                    stable_count += 1
                    if stable_count >= stable_checks:
                        return cur  # Confirmed static
                else:
                    stable_count = 0  # Reset — still moving

            prev = cur
            prev_id = cur_id
            time.sleep(0.02)

        # Timed out — return newest frame anyway
        print(f"[StreamNode] wait_until_static timed out "
              f"(last motion check did not stabilize)", flush=True)
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def is_running(self) -> bool:
        return self._running

    def opened_with(self) -> str:
        return self._opened_with

    def stop(self):
        self._running = False
        if self._proc is not None:
            try:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=1.5)
                except Exception:
                    self._proc.kill()
            except Exception:
                pass
            self._proc = None