"""
ptz_node.py — PTZ camera control (NODE 2a of 2).

This node ONLY drives the camera. It does not touch the video stream.
It sends AbsoluteMove commands and waits a fixed settle time for the
camera to physically stop. We deliberately do NOT poll GetStatus because:

  - This camera's GetStatus either hangs (10s timeout per move) or isn't
    implemented, which was causing the duplicate-frame / out-of-sync issue.
  - Frame freshness from the StreamNode is a more reliable "did we capture
    after the move" signal than ONVIF status polling.

So the sequence is: move → sleep(settle) → ask StreamNode for a frame
that is newer than the move. Clean separation of concerns.
"""
from __future__ import annotations

import re
import threading
from urllib.parse import urlsplit, urlunsplit

try:
    from onvif import ONVIFCamera
    from zeep.transports import Transport as _ZeepTransport
    _zeep_orig = _ZeepTransport.__init__
    def _zeep_patched(self, *a, **k):
        k.setdefault("timeout", 5)
        k.setdefault("operation_timeout", 5)
        _zeep_orig(self, *a, **k)
    _ZeepTransport.__init__ = _zeep_patched
except ImportError:
    raise SystemExit("pip install onvif-zeep")


class PTZNode:
    """ONVIF PTZ control — connect, AbsoluteMove, go_home. Nothing else."""

    def __init__(self, ip: str, port: int, user: str, password: str):
        self.ip = ip
        self.port = port
        self.user = user
        self.password = password
        self.cam = None
        self.media_svc = None
        self.ptz_svc = None
        self.ptz_token = ""
        self.stream_uri = ""
        self.ptz_ok = False
        self._lock = threading.Lock()

    def connect(self) -> bool:
        try:
            cam = ONVIFCamera(self.ip, self.port, self.user, self.password,
                              no_cache=True)
            base = f"http://{self.ip}:{self.port}"
            cam.xaddrs = {
                "http://www.onvif.org/ver10/device/wsdl":
                    base + "/onvif/device_service",
                "http://www.onvif.org/ver10/media/wsdl":
                    base + "/onvif/Media",
                "http://www.onvif.org/ver20/media/wsdl":
                    base + "/onvif/Media",
                "http://www.onvif.org/ver20/ptz/wsdl":
                    base + "/onvif/PTZ",
                "http://www.onvif.org/ver20/imaging/wsdl":
                    base + "/onvif/Imaging",
            }
            self.cam = cam
            return True
        except Exception as e:
            print(f"[PTZNode] Connect failed: {e}")
            return False

    def setup(self, rtsp_port: int = 554):
        self.media_svc = self.cam.create_media_service()
        try:
            self.ptz_svc = self.cam.create_ptz_service()
            self.ptz_ok = True
        except Exception:
            self.ptz_ok = False
            print("[PTZNode] No PTZ service found")

        profiles = self.media_svc.GetProfiles()
        self.profile = next(
            (p for p in profiles if getattr(p, "PTZConfiguration", None)),
            profiles[0])
        self.ptz_token = self.profile.token

        req = self.media_svc.create_type("GetStreamUri")
        req.ProfileToken = self.profile.token
        req.StreamSetup = {"Stream": "RTP-Unicast",
                           "Transport": {"Protocol": "RTSP"}}
        uri = self.media_svc.GetStreamUri(req).Uri

        # Rewrite host to external IP, force RTSP port (not ONVIF port)
        parts = urlsplit(str(uri))
        uri = urlunsplit((parts.scheme, f"{self.ip}:{rtsp_port}",
                          parts.path, parts.query, parts.fragment))
        if self.user:
            uri = re.sub(r"^rtsp://",
                         f"rtsp://{self.user}:{self.password}@",
                         uri, count=1)
        self.stream_uri = uri
        print(f"[PTZNode] Stream URI: {self.stream_uri}")

    def move_absolute(self, pan: float, tilt: float, zoom: float = 0.0,
                      speed: float = 0.5):
        """Command an absolute move. Returns immediately; camera moves async."""
        if not self.ptz_ok:
            return
        with self._lock:
            try:
                req = self.ptz_svc.create_type("AbsoluteMove")
                req.ProfileToken = self.ptz_token
                req.Position = {
                    "PanTilt": {"x": float(pan), "y": float(tilt)},
                    "Zoom":    {"x": float(zoom)},
                }
                req.Speed = {
                    "PanTilt": {"x": float(speed), "y": float(speed)},
                    "Zoom":    {"x": float(speed)},
                }
                self.ptz_svc.AbsoluteMove(req)
            except Exception as e:
                print(f"[PTZNode] AbsoluteMove failed: {e}")

    def stop(self):
        """Explicitly halt all PTZ motion — forces zero velocity."""
        if not self.ptz_ok:
            return
        with self._lock:
            try:
                req = self.ptz_svc.create_type("Stop")
                req.ProfileToken = self.ptz_token
                req.PanTilt = True
                req.Zoom = True
                self.ptz_svc.Stop(req)
            except Exception as e:
                print(f"[PTZNode] Stop failed: {e}")

    def go_home(self):
        self.move_absolute(0.0, 0.0, 0.0, speed=0.6)