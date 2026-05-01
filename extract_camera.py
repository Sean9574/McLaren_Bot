#!/usr/bin/env python3
"""
onvif_camera_node.py  —  ROS 2 Humble
======================================
Bare-bones ONVIF PTZ camera node with GStreamer subprocess RTSP pipeline.
Designed to be a base layer for downstream follower/tracker nodes.

Key design decisions:
  - GStreamer subprocess for RTSP (NOT cv2.VideoCapture or PyGObject):
    * Avoids Qt/GLib event loop conflicts with OpenCV display
    * Hardware decode via NVDEC if available (nvh265dec / nvh264dec)
    * Software decode fallback (avdec_h265 / avdec_h264)
    * Drops late frames at the source for minimum latency
  - ONVIF for PTZ control only — never for video stream
  - Auto-detects stream resolution and codec from ONVIF profile
  - Threading: GStreamer subprocess feeds a numpy frame slot under a lock,
    ROS 2 executor spins in a background thread so OpenCV display can run
    on the main thread without conflict.

Published topics:
  /camera/image_raw        (sensor_msgs/Image)
  /camera/camera_info      (sensor_msgs/CameraInfo)
  /camera/ptz_state        (geometry_msgs/Pose)   x=pan, y=tilt, z=zoom

Subscribed topics:
  /camera/cmd_vel          (geometry_msgs/Twist)
    angular.z = pan
    angular.y = tilt
    linear.x  = zoom

Action server:
  /camera/goto_position    (ros2_ptz_camera_msgs/action/PtzGoto)

OpenCV window keyboard:
  Arrow keys / WASD : Pan / Tilt
  + / -             : Zoom
  H                 : Go to home (0, 0, 0)
  F                 : Toggle 180-degree image rotation
  P                 : Toggle pan invert
  T                 : Toggle tilt invert
  I                 : Print camera info
  [/]               : Decrease/increase manual PTZ speed
  Q / ESC           : Shutdown

Parameters:
  camera_ip, camera_port, camera_user, camera_password
  rotate_image_180  (bool, default true)   — display + publish rotated
  invert_pan        (bool, default false)
  invert_tilt       (bool, default false)
  use_tcp           (bool, default false)
  codec             (string, "h264" or "h265", default "auto")
  publish_rate      (double, default 30.0) — image/info publish rate
  ptz_speed         (double, default 1.0)
  zoom_speed        (double, default 0.6)
  show_window       (bool, default true)

Install:
  pip install onvif-zeep opencv-python "numpy<2" requests
  sudo apt install ros-humble-cv-bridge \\
      gstreamer1.0-tools gstreamer1.0-plugins-{base,good,bad,ugly} \\
      gstreamer1.0-libav
"""

# Suppress Qt warnings caused by GStreamer subprocess running on its own thread
import os as _early_os

_early_os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.*=false")

import os
import re
import select
import subprocess
import threading
import time
from datetime import datetime
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, Twist
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image

try:
    from ros2_ptz_camera_msgs.action import PtzGoto
    HAS_ACTION_MSG = True
except ImportError:
    HAS_ACTION_MSG = False

try:
    from onvif import ONVIFCamera
except ImportError:
    raise SystemExit("ERROR: pip install onvif-zeep")


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def rewrite_url_host(url, new_host):
    if not url or not new_host:
        return url
    try:
        parts    = urlsplit(url)
        username = parts.username or ""
        password = parts.password or ""
        port     = parts.port
        userinfo = f"{username}:{password}@" if username else ""
        netloc   = f"{userinfo}{new_host}"
        if port:
            netloc += f":{port}"
        return urlunsplit((parts.scheme, netloc, parts.path,
                           parts.query, parts.fragment))
    except Exception:
        return url


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ---------------------------------------------------------------------------
#  GStreamer subprocess frame reader
# ---------------------------------------------------------------------------

class FrameReader:
    """
    RTSP reader that spawns gst-launch-1.0 as a subprocess and reads raw
    BGR frames from its stdout. Lifted exactly from follow_person.py.
    """
    def __init__(self, uri, use_tcp=False, codec="h264",
                 width=1920, height=1080):
        self._uri      = uri
        self._use_tcp  = use_tcp
        self._codec    = codec.lower()
        self._width    = width
        self._height   = height
        self._lock     = threading.Lock()
        self._frame    = None
        self._capture_t = 0.0
        self._frame_id = 0
        self._running  = False
        self._opened_with = "none"
        self._proc     = None
        self._reader_thread = None

    def opened_with(self):
        return self._opened_with

    def _build_pipeline_str(self, hw=True):
        """Build a gst-launch-1.0 command list for the configured codec."""
        protocols = "tcp" if self._use_tcp else "udp"

        if self._codec == "h265":
            depay = "rtph265depay"
            parse = "h265parse"
            decoder = "nvh265dec" if hw else "avdec_h265"
        else:
            depay = "rtph264depay"
            parse = "h264parse"
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
            "!", parse, "config-interval=-1",
            "!", "queue", "leaky=downstream",
            "max-size-buffers=2",
            "max-size-bytes=0",
            "max-size-time=0",
            "!", decoder,
            "!", "videoconvert", "n-threads=4",
            "!", "video/x-raw,format=BGR",
            "!", "fdsink", "fd=1", "sync=false",
        ]

    def _detect_dimensions(self):
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error",
                 "-select_streams", "v:0",
                 "-show_entries", "stream=width,height",
                 "-of", "csv=p=0:s=x",
                 "-rtsp_transport", "tcp",
                 self._uri],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split("x")
                if len(parts) == 2:
                    w = int(parts[0])
                    h = int(parts[1])
                    print(f"[FrameReader] Detected stream {w}x{h}")
                    return w, h
        except Exception as e:
            print(f"[FrameReader] ffprobe failed: {e}")
        return self._width, self._height

    def start(self):
        w, h = self._detect_dimensions()
        self._width = w
        self._height = h

        for name, hw in [("GST-NVDEC", True), ("GST-SW", False)]:
            cmd = self._build_pipeline_str(hw=hw)
            print(f"[FrameReader] Trying {name}...")
            print(f"  Command: {' '.join(cmd)}")

            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0,
                )
            except FileNotFoundError:
                print("[FrameReader] gst-launch-1.0 not found in PATH")
                return False

            frame_size = self._width * self._height * 3
            t0 = time.time()
            first_frame = None

            try:
                while time.time() - t0 < 5.0:
                    if proc.poll() is not None:
                        err = proc.stderr.read().decode("utf-8", errors="ignore")
                        print(f"[FrameReader] {name} died:")
                        print(f"  {err[:500]}")
                        break

                    ready, _, _ = select.select([proc.stdout], [], [], 0.5)
                    if ready:
                        first_frame = self._read_exact(proc.stdout, frame_size)
                        if first_frame is not None:
                            break
            except Exception as e:
                print(f"[FrameReader] {name} read error: {e}")

            if first_frame is None or len(first_frame) != frame_size:
                proc.kill()
                proc.wait(timeout=2)
                continue

            frame = np.frombuffer(first_frame, dtype=np.uint8).reshape(
                (self._height, self._width, 3))
            with self._lock:
                self._frame = frame.copy()
                self._capture_t = time.time()
                self._frame_id += 1

            self._proc = proc
            self._opened_with = name
            self._running = True
            self._reader_thread = threading.Thread(
                target=self._reader_loop, daemon=True)
            self._reader_thread.start()
            print(f"[FrameReader] ✓ Opened with {name}")
            return True

        print("[FrameReader] FATAL: All GStreamer pipelines failed")
        return False

    def _read_exact(self, stream, n):
        buf = b""
        while len(buf) < n:
            chunk = stream.read(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    def _reader_loop(self):
        frame_size = self._width * self._height * 3
        stream = self._proc.stdout
        while self._running:
            data = self._read_exact(stream, frame_size)
            t = time.time()
            if data is None:
                print("[FrameReader] Stream closed")
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
                print(f"[FrameReader] reshape error: {e}")

    def read(self):
        """Returns (ok, frame). Frame is a copy."""
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def stop(self):
        self._running = False
        if self._proc is not None:
            try:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=1.0)
                except Exception:
                    self._proc.kill()
            except Exception:
                pass


# ---------------------------------------------------------------------------
#  ONVIF session
# ---------------------------------------------------------------------------

ONVIF_PORTS = [2000, 80, 8080, 8000, 8899, 554]


class ONVIFSession:
    def __init__(self, ip, port, user, password,
                 invert_pan=False, invert_tilt=False):
        self.ip       = ip
        self.port     = port
        self.user     = user
        self.password = password
        self.invert_pan  = bool(invert_pan)
        self.invert_tilt = bool(invert_tilt)

        self.cam        = None
        self.device_svc = None
        self.media_svc  = None
        self.ptz_svc    = None
        self.profile    = None
        self.ptz_token  = ""
        self.stream_uri = ""
        self.codec      = "h264"
        self.ptz_ok     = False
        self._lock      = threading.Lock()

        self.ptz_speed  = 1.0
        self.zoom_speed = 0.6

    def connect(self):
        ports = [self.port] + [p for p in ONVIF_PORTS if p != self.port]
        for port in ports:
            try:
                cam = ONVIFCamera(self.ip, port, self.user, self.password)
                cam.update_xaddrs()
                self.cam = cam
                self.port = port
                return True
            except Exception:
                pass
        return False

    def setup(self):
        self.device_svc = self.cam.create_devicemgmt_service()
        self.media_svc  = self.cam.create_media_service()
        try:
            self.ptz_svc = self.cam.create_ptz_service()
            self.ptz_ok  = True
        except Exception:
            self.ptz_ok = False

        profiles = self.media_svc.GetProfiles()
        self.profile = next(
            (p for p in profiles if getattr(p, "PTZConfiguration", None)),
            profiles[0])
        self.ptz_token = self.profile.token

        # Detect codec from profile — check Encoding field plus
        # H264/H265 sub-config objects since some cameras lie about Encoding
        self.codec = "h264"  # default
        try:
            ve = self.profile.VideoEncoderConfiguration
            enc = str(getattr(ve, "Encoding", "")).lower()
            print(f"[ONVIF] VideoEncoderConfiguration.Encoding = {enc!r}")

            if "265" in enc or "hevc" in enc:
                self.codec = "h265"
            elif "264" in enc:
                self.codec = "h264"
            else:
                # Some cameras use generic "JPEG" or empty — check sub-configs
                if getattr(ve, "H265", None) is not None:
                    self.codec = "h265"
                elif getattr(ve, "H264", None) is not None:
                    self.codec = "h264"
            print(f"[ONVIF] Detected codec: {self.codec}")
        except Exception as e:
            print(f"[ONVIF] Codec detection failed: {e}, defaulting to h264")

        req = self.media_svc.create_type("GetStreamUri")
        req.ProfileToken = self.profile.token
        req.StreamSetup  = {"Stream": "RTP-Unicast",
                            "Transport": {"Protocol": "RTSP"}}
        uri = self.media_svc.GetStreamUri(req).Uri
        uri = rewrite_url_host(uri, self.ip)
        if self.user:
            uri = re.sub(r"^rtsp://",
                         f"rtsp://{self.user}:{self.password}@", uri, count=1)
        self.stream_uri = uri

    def _apply_axis_inversion(self, pan, tilt):
        if self.invert_pan:  pan  = -pan
        if self.invert_tilt: tilt = -tilt
        return pan, tilt

    def move(self, pan=0.0, tilt=0.0, zoom=0.0):
        if not self.ptz_ok:
            return
        pan, tilt = self._apply_axis_inversion(pan, tilt)
        with self._lock:
            try:
                req = self.ptz_svc.create_type("ContinuousMove")
                req.ProfileToken = self.ptz_token
                req.Velocity = {"PanTilt": {"x": pan, "y": tilt},
                                "Zoom":    {"x": zoom}}
                self.ptz_svc.ContinuousMove(req)
            except Exception as e:
                print(f"[PTZ] {e}")

    def stop(self):
        if not self.ptz_ok:
            return
        with self._lock:
            try:
                req = self.ptz_svc.create_type("Stop")
                req.ProfileToken = self.ptz_token
                req.PanTilt = True
                req.Zoom    = True
                self.ptz_svc.Stop(req)
            except Exception:
                pass

    def go_home(self):
        if not self.ptz_ok:
            return True, "PTZ not available"
        with self._lock:
            try:
                req = self.ptz_svc.create_type("AbsoluteMove")
                req.ProfileToken = self.ptz_token
                req.Position = {"PanTilt": {"x": 0.0, "y": 0.0},
                                "Zoom":    {"x": 0.0}}
                req.Speed    = {"PanTilt": {"x": 0.5, "y": 0.5},
                                "Zoom":    {"x": 0.5}}
                self.ptz_svc.AbsoluteMove(req)
                return True, "OK"
            except Exception as e:
                try:
                    self.ptz_svc.GotoHomePosition(
                        {"ProfileToken": self.ptz_token})
                    return True, "OK"
                except Exception as e2:
                    return False, str(e2)

    def absolute_move(self, pan, tilt, zoom):
        if not self.ptz_ok:
            return False, "PTZ not available"
        pan_s, tilt_s = self._apply_axis_inversion(pan, tilt)
        with self._lock:
            try:
                req = self.ptz_svc.create_type("AbsoluteMove")
                req.ProfileToken = self.ptz_token
                req.Position = {
                    "PanTilt": {"x": clamp(pan_s, -1.0, 1.0),
                                "y": clamp(tilt_s, -1.0, 1.0)},
                    "Zoom":    {"x": clamp(zoom,   0.0,  1.0)},
                }
                req.Speed = {"PanTilt": {"x": 0.5, "y": 0.5},
                             "Zoom":    {"x": 0.5}}
                self.ptz_svc.AbsoluteMove(req)
                return True, "OK"
            except Exception as e:
                return False, str(e)

    def get_status(self):
        if not self.ptz_ok:
            return 0.0, 0.0, 0.0
        with self._lock:
            try:
                s    = self.ptz_svc.GetStatus({"ProfileToken": self.ptz_token})
                pos  = s.Position
                pan  = float(getattr(getattr(pos, "PanTilt", None), "x", 0.0))
                tilt = float(getattr(getattr(pos, "PanTilt", None), "y", 0.0))
                zoom = float(getattr(getattr(pos, "Zoom",    None), "x", 0.0))
                return pan, tilt, zoom
            except Exception:
                return 0.0, 0.0, 0.0

    def print_info(self, logger):
        sep = "-" * 56
        logger.info(sep)
        logger.info("  ONVIF CAMERA INFORMATION")
        logger.info(sep)
        try:
            info = self.device_svc.GetDeviceInformation()
            logger.info(f"  Manufacturer : {info.Manufacturer}")
            logger.info(f"  Model        : {info.Model}")
            logger.info(f"  Firmware     : {info.FirmwareVersion}")
        except Exception:
            pass
        logger.info(f"  IP           : {self.ip}:{self.port}")
        logger.info(f"  PTZ          : {'Yes' if self.ptz_ok else 'No'}")
        logger.info(f"  Codec        : {self.codec}")
        logger.info(f"  Invert pan   : {self.invert_pan}")
        logger.info(f"  Invert tilt  : {self.invert_tilt}")
        logger.info(f"  Stream       : {self.stream_uri}")
        try:
            ve = self.profile.VideoEncoderConfiguration
            logger.info(f"  Encoding     : {ve.Encoding}")
            logger.info(
                f"  Resolution   : {ve.Resolution.Width}x{ve.Resolution.Height}")
            logger.info(
                f"  Frame rate   : {ve.RateControl.FrameRateLimit} fps")
        except Exception:
            pass
        logger.info(sep)


# ---------------------------------------------------------------------------
#  HUD overlay
# ---------------------------------------------------------------------------

def overlay_hud(frame, info_lines, ptz_available, rotated, src):
    h, w     = frame.shape[:2]
    rot_tag  = "  [ROT-180]" if rotated else ""
    ptz_tag  = "ON" if ptz_available else "OFF"

    lines = info_lines + [
        f"PTZ:{ptz_tag}{rot_tag}  Source: {src}",
        f"Keys: Arrows/WASD=Pan/Tilt  +/-=Zoom  H=Home  F=Rotate  P/T=Invert  I=Info  Q=Quit",
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]

    bar_h   = 22 * len(lines) + 14
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    y = h - bar_h + 18
    for line in lines:
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255),
                    1, cv2.LINE_AA)
        y += 22

    cv2.circle(frame, (w - 20, 20), 8, (0, 0, 255), -1)
    cv2.putText(frame, "LIVE", (w - 60, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 255), 1, cv2.LINE_AA)
    return frame


# ---------------------------------------------------------------------------
#  ROS 2 node
# ---------------------------------------------------------------------------

class ONVIFCameraNode(Node):

    def __init__(self):
        super().__init__("onvif_camera_node")

        # Parameters
        self.declare_parameter("camera_ip",       "192.168.8.195")
        self.declare_parameter("camera_port",     2000)
        self.declare_parameter("camera_user",     "admin")
        self.declare_parameter("camera_password", "admin")
        self.declare_parameter("rotate_image_180", True)
        self.declare_parameter("invert_pan",      True)
        self.declare_parameter("invert_tilt",     True)
        self.declare_parameter("use_tcp",         False)
        self.declare_parameter("codec",           "h265")
        self.declare_parameter("publish_rate",    30.0)
        self.declare_parameter("frame_id",        "camera_optical_frame")
        self.declare_parameter("ptz_speed",       1.0)
        self.declare_parameter("zoom_speed",      0.6)
        self.declare_parameter("show_window",     True)

        p = lambda n: self.get_parameter(n).value

        ip            = p("camera_ip")
        port          = p("camera_port")
        user          = p("camera_user")
        password      = p("camera_password")
        self._rotated = p("rotate_image_180")
        invert_pan    = p("invert_pan")
        invert_tilt   = p("invert_tilt")
        use_tcp       = p("use_tcp")
        codec_param   = str(p("codec")).lower()
        rate          = p("publish_rate")
        self._frame_id     = p("frame_id")
        self._show_window  = p("show_window")

        # ONVIF
        self.get_logger().info(f"Connecting to {ip}:{port} ...")
        self._session = ONVIFSession(ip, port, user, password,
                                     invert_pan=invert_pan,
                                     invert_tilt=invert_tilt)
        self._session.ptz_speed  = p("ptz_speed")
        self._session.zoom_speed = p("zoom_speed")

        if not self._session.connect():
            self.get_logger().error("ONVIF connection failed")
            raise SystemExit(1)
        self._session.setup()
        self.get_logger().info(
            f"Connected. Codec: {self._session.codec}  "
            f"Stream: {self._session.stream_uri}")

        # Codec auto-detect or override
        if codec_param == "auto":
            codec = self._session.codec
        else:
            codec = codec_param

        # Frame reader (GStreamer subprocess)
        self._reader = FrameReader(
            self._session.stream_uri,
            use_tcp=use_tcp,
            codec=codec,
        )
        if not self._reader.start():
            self.get_logger().error(
                "Could not open RTSP stream via GStreamer")
            raise SystemExit(1)
        self.get_logger().info(
            f"Stream open via {self._reader.opened_with()}")

        self._bridge   = CvBridge()
        self._cam_info = None
        self._img_w    = 0
        self._img_h    = 0

        # QoS
        qos_be = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                            history=HistoryPolicy.KEEP_LAST, depth=1)
        qos_re = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                            history=HistoryPolicy.KEEP_LAST, depth=1)

        # Publishers
        self._img_pub  = self.create_publisher(Image,      "/camera/image_raw",   qos_be)
        self._info_pub = self.create_publisher(CameraInfo, "/camera/camera_info", qos_re)
        self._ptz_pub  = self.create_publisher(Pose,       "/camera/ptz_state",   qos_re)

        # Subscriber
        self._cmd_sub = self.create_subscription(
            Twist, "/camera/cmd_vel", self._cmd_vel_cb, qos_re)

        # Action server
        self._cb_group = ReentrantCallbackGroup()
        if HAS_ACTION_MSG:
            self._action_server = ActionServer(
                self, PtzGoto, "/camera/goto_position",
                execute_callback=self._goto_execute,
                goal_callback=lambda g: GoalResponse.ACCEPT,
                cancel_callback=lambda g: CancelResponse.ACCEPT,
                callback_group=self._cb_group,
            )

        # Timers
        self._img_timer = self.create_timer(1.0 / rate, self._publish_frame)
        self._ptz_timer = self.create_timer(0.5, self._publish_ptz_state)

        # cmd_vel watchdog
        self._cmd_lock      = threading.Lock()
        self._last_cmd_t    = 0.0
        self._cmd_active    = False
        self._CMD_TIMEOUT   = 0.5
        self._watchdog_tmr  = self.create_timer(0.1, self._watchdog)

        # Display state
        self._key_active    = False
        self._display_lock  = threading.Lock()
        self._display_frame = None
        self._win           = "ONVIF Camera  [ROS 2]"

        self.get_logger().info(
            "ONVIFCameraNode ready — "
            "Arrows/WASD=Pan/Tilt  +/-=Zoom  H=Home  F=Rotate  P=PanInv  T=TiltInv  I=Info  Q=Quit")

    # -- Frame publisher ------------------------------------------------------

    def _publish_frame(self):
        ok, frame = self._reader.read()
        if not ok or frame is None:
            return

        if self._rotated:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        fh, fw = frame.shape[:2]
        now    = self.get_clock().now().to_msg()

        # Camera info
        if self._cam_info is None or self._img_w != fw or self._img_h != fh:
            self._img_w    = fw
            self._img_h    = fh
            self._cam_info = self._make_cam_info(fw, fh)

        img_msg = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        img_msg.header.stamp    = now
        img_msg.header.frame_id = self._frame_id
        self._img_pub.publish(img_msg)

        self._cam_info.header.stamp = now
        self._info_pub.publish(self._cam_info)

        # Build display frame (with HUD)
        if self._show_window:
            disp = frame.copy()
            disp = overlay_hud(
                disp,
                [f"Camera: {self._session.ip}:{self._session.port}  "
                 f"Stream: {fw}x{fh}  Codec: {self._session.codec}"],
                self._session.ptz_ok,
                self._rotated,
                self._reader.opened_with(),
            )
            with self._display_lock:
                self._display_frame = disp

    def _make_cam_info(self, w, h):
        info = CameraInfo()
        info.header.frame_id = self._frame_id
        info.width  = w
        info.height = h
        f = w * 1.2
        info.k = [f, 0., w/2, 0., f, h/2, 0., 0., 1.]
        info.p = [f, 0., w/2, 0., 0., f, h/2, 0., 0., 0., 1., 0.]
        info.distortion_model = "plumb_bob"
        info.d = [0., 0., 0., 0., 0.]
        return info

    # -- PTZ state publisher --------------------------------------------------

    def _publish_ptz_state(self):
        pan, tilt, zoom = self._session.get_status()
        pose = Pose()
        pose.position.x = pan
        pose.position.y = tilt
        pose.position.z = zoom
        self._ptz_pub.publish(pose)

    # -- cmd_vel subscriber ---------------------------------------------------

    def _cmd_vel_cb(self, msg: Twist):
        pan  = float(msg.angular.z)
        tilt = float(msg.angular.y)
        zoom = float(msg.linear.x)
        with self._cmd_lock:
            self._last_cmd_t = time.time()
            self._cmd_active = True
        if abs(pan) < 0.01 and abs(tilt) < 0.01 and abs(zoom) < 0.01:
            self._session.stop()
            with self._cmd_lock:
                self._cmd_active = False
        else:
            self._session.move(pan, tilt, zoom)

    def _watchdog(self):
        with self._cmd_lock:
            active  = self._cmd_active
            elapsed = time.time() - self._last_cmd_t
        if active and elapsed > self._CMD_TIMEOUT and not self._key_active:
            self._session.stop()
            with self._cmd_lock:
                self._cmd_active = False

    # -- Action server --------------------------------------------------------

    def _goto_execute(self, goal_handle):
        goal   = goal_handle.request
        result = PtzGoto.Result()

        self.get_logger().info(
            f"goto_position: pan={goal.pan:.2f} "
            f"tilt={goal.tilt:.2f} zoom={goal.zoom:.2f}")

        ok, msg = self._session.absolute_move(goal.pan, goal.tilt, goal.zoom)
        if not ok:
            result.success = False
            result.message = msg
            goal_handle.abort()
            return result

        feedback = PtzGoto.Feedback()
        deadline = time.time() + 10.0
        tol      = 0.05

        while time.time() < deadline:
            if goal_handle.is_cancel_requested:
                self._session.stop()
                goal_handle.canceled()
                result.success = False
                result.message = "Cancelled"
                return result

            pan, tilt, zoom = self._session.get_status()
            feedback.current_pan  = pan
            feedback.current_tilt = tilt
            feedback.current_zoom = zoom
            goal_handle.publish_feedback(feedback)

            if (abs(pan  - goal.pan)  < tol and
                    abs(tilt - goal.tilt) < tol and
                    abs(zoom - goal.zoom) < tol):
                break
            time.sleep(0.2)

        result.success = True
        result.message = "Done"
        goal_handle.succeed()
        return result

    # -- Cleanup --------------------------------------------------------------

    def destroy_node(self):
        self._reader.stop()
        self._session.stop()
        if self._show_window:
            cv2.destroyAllWindows()
        super().destroy_node()


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    executor = MultiThreadedExecutor(num_threads=4)
    node     = ONVIFCameraNode()
    executor.add_node(node)

    if node._show_window:
        cv2.namedWindow(node._win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(node._win, 1280, 720)
    else:
        try:
            executor.spin()
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()
        return

    # Spin ROS 2 in background — OpenCV must be on main thread
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        while rclpy.ok():
            with node._display_lock:
                frame = node._display_frame
            if frame is not None:
                cv2.imshow(node._win, frame)

            key = cv2.waitKeyEx(1)
            if key == -1:
                if node._key_active:
                    node._session.stop()
                    node._key_active = False
                continue

            spd  = node._session.ptz_speed
            zspd = node._session.zoom_speed

            if key in (ord("q"), ord("Q"), 27):
                node.get_logger().info("Quit")
                node._session.stop()
                break
            elif key in (65361, 2424832, ord("a"), ord("A")):
                node._session.move(pan=-spd);  node._key_active = True
            elif key in (65363, 2555904, ord("d"), ord("D")):
                node._session.move(pan= spd);  node._key_active = True
            elif key in (65362, 2490368, ord("w"), ord("W")):
                node._session.move(tilt= spd); node._key_active = True
            elif key in (65364, 2621440, ord("s"), ord("S")):
                node._session.move(tilt=-spd); node._key_active = True
            elif key in (ord("+"), ord("=")):
                node._session.move(zoom= zspd); node._key_active = True
            elif key in (ord("-"), ord("_")):
                node._session.move(zoom=-zspd); node._key_active = True
            elif key in (ord("h"), ord("H")):
                node._session.go_home();        node._key_active = False
            elif key in (ord("f"), ord("F")):
                node._rotated = not node._rotated
                node.get_logger().info(
                    f"Image rotation: {'ROT-180' if node._rotated else 'NORMAL'}")
            elif key in (ord("p"), ord("P")):
                node._session.invert_pan = not node._session.invert_pan
                node._session.stop()
                node.get_logger().info(
                    f"Invert pan: {'ON' if node._session.invert_pan else 'OFF'}")
            elif key in (ord("t"), ord("T")):
                node._session.invert_tilt = not node._session.invert_tilt
                node._session.stop()
                node.get_logger().info(
                    f"Invert tilt: {'ON' if node._session.invert_tilt else 'OFF'}")
            elif key in (ord("i"), ord("I")):
                node._session.print_info(node.get_logger())
            elif key == ord("["):
                node._session.ptz_speed = max(0.05, node._session.ptz_speed - 0.05)
                node.get_logger().info(f"PTZ speed: {node._session.ptz_speed:.2f}")
            elif key == ord("]"):
                node._session.ptz_speed = min(1.0, node._session.ptz_speed + 0.05)
                node.get_logger().info(f"PTZ speed: {node._session.ptz_speed:.2f}")
            else:
                if node._key_active:
                    node._session.stop()
                    node._key_active = False

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()