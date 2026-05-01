# McLaren Bot — PTZ Person Follower

A self-contained ROS 2 / Python system for tracking people with a Minrray (or any ONVIF-compatible) PTZ camera. Combines YOLO11 pose detection, TensorRT-accelerated inference, GStreamer low-latency video, and a custom alpha-beta filter + image-center servo controller for smooth automatic following.

![status](https://img.shields.io/badge/status-working-brightgreen)
![ros2](https://img.shields.io/badge/ROS_2-Humble-blue)
![python](https://img.shields.io/badge/Python-3.10-blue)

## Features

- **Self-contained `follow_person.py`** — connects to the camera directly, no separate node required. Optional `--ros` flag publishes ROS 2 topics for downstream consumers.
- **Bare-bones `extract_camera.py`** — ROS 2 base node with low-latency video + ONVIF PTZ control, ready for any custom follower built on top.
- **TensorRT FP16 auto-export** — first run builds an optimized GPU engine; subsequent runs load instantly with ~3-5x faster inference (~7ms instead of ~30ms on RTX 3070).
- **GStreamer subprocess RTSP pipeline** — bypasses OpenCV's FFmpeg buffering for sub-50ms stream latency. Hardware decode (NVDEC) when available, software fallback (avdec).
- **Alpha-beta filter** — 1957-vintage radar tracking algorithm for smoothed target position + velocity. Simpler than Kalman, perfect for PTZ tracking.
- **Image-center servo controller** — bounded velocity command with dead zone, square-root response curve, and slew-rate limiting. Stable, no oscillation.
- **Fire-and-forget ONVIF** — PTZ commands run on a background queue with coalescing; control loop never blocks waiting for camera response.
- **Live tuning panel** — drag sliders to adjust dead zone, gain, smoothing, etc. without restarting.
- **Performance HUD** — real-time FPS, YOLO inference time, RTSP age, and stream source for diagnostics.
- **Auto re-acquire and home** — locks back onto target after brief occlusion; returns home if target is lost for >5 seconds.

## Hardware

- **Camera:** Minrray PTZ ONVIF (or compatible)
- **GPU:** NVIDIA RTX 3070 tested; any CUDA-capable card works
- **OS:** Ubuntu 22.04 LTS
- **ROS:** ROS 2 Humble (optional, only needed for the camera node)

## System architecture

```
              ┌──────────────────────┐
              │  Minrray PTZ Camera  │
              └──────┬───────┬───────┘
                     │       │
              RTSP   │       │   ONVIF SOAP
            (H.265)  │       │   (PTZ control)
                     ▼       ▼
              ┌──────────────────────────┐
              │  follow_person.py        │
              │  ┌─────────────────────┐ │
              │  │ GStreamer subproc   │ │
              │  │  → BGR frames       │ │
              │  └─────────┬───────────┘ │
              │            ▼             │
              │  ┌──────────────────┐    │
              │  │ YOLO11s-pose     │    │
              │  │ (TensorRT FP16)  │    │
              │  └─────────┬────────┘    │
              │            ▼             │
              │  ┌──────────────────┐    │
              │  │ Alpha-beta filter│    │
              │  └─────────┬────────┘    │
              │            ▼             │
              │  ┌──────────────────┐    │
              │  │ Center-servo ctrl│    │
              │  └─────────┬────────┘    │
              │            │             │
              │            ▼ ONVIF cmd   │
              └────────────┼─────────────┘
                           │
                           └──→ back to camera
```

## Installation

### 1. System dependencies

```bash
sudo apt update
sudo apt install -y \
    python3-pip \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    ffmpeg
```

### 2. Conda environment (recommended for ROS 2)

```bash
conda create -n ros_humble python=3.10 -y
conda activate ros_humble
source /opt/ros/humble/setup.bash
```

### 3. Python dependencies

```bash
pip install \
    onvif-zeep \
    ultralytics \
    opencv-python \
    "numpy==1.26.4" \
    requests
```

### 4. NVIDIA TensorRT (optional, for faster inference)

```bash
pip install tensorrt
```

If TensorRT export fails, the script will fall back to PyTorch FP32 automatically.

### 5. ROS 2 packages (only needed for `extract_camera.py`)

```bash
sudo apt install -y \
    ros-humble-cv-bridge \
    ros-humble-vision-msgs
```

## Quick start

### Standalone follower (no ROS 2 needed)

```bash
python follow_person.py
```

That's it. The script auto-detects the camera, exports a TensorRT engine on first run (~3-5 minutes one time), and starts following.

### ROS 2 base camera node

For building your own tracker on top of the camera node:

```bash
# Terminal 1 — camera node publishes /camera/image_raw etc.
python extract_camera.py

# Terminal 2 — your custom follower subscribes to /camera/image_raw
python my_custom_follower.py
```

## Keyboard controls

| Key | Action |
|---|---|
| Click on a person | Lock onto that target |
| `f` | Toggle follow on/off |
| `a` | Toggle auto-acquire (locks onto closest person) |
| `z` | Cycle aim mode (FACE / UPPER / BODY) |
| `c` | Clear locked target |
| `x` | Force-stop PTZ |
| `h` | Send camera home (0, 0, 0) |
| `F` | Toggle 180° image rotation |
| `p` | Toggle pan invert |
| `T` | Toggle tilt invert |
| `Arrow keys` / `WASD` | Manual pan/tilt at full speed |
| `+` / `-` | Manual zoom |
| `[` / `]` | Decrease/increase manual PTZ speed |
| `q` / `ESC` | Quit |

## Live tuning sliders

A separate "PTZ Tuning" window shows all controller parameters. Drag to adjust live:

| Slider | What it controls |
|---|---|
| **Dead Zone (px)** | Pixels of error before the camera starts moving |
| **Center Gain** | How aggressively the camera responds to position error |
| **Max Pan / Tilt Speed** | Upper limit on commanded ONVIF velocity (0-1.0) |
| **Min Speed** | Floor below which commands are zeroed out |
| **Accel Limit** | How fast velocity can change between commands |
| **Target Smooth (alpha)** | Alpha-beta filter position smoothing |
| **Velocity Smooth (beta)** | Alpha-beta filter velocity smoothing |
| **Cmd Interval ms** | Minimum time between PTZ commands |
| **Keypoint Conf %** | YOLO pose confidence threshold |

## Camera setup

For best latency, configure the camera via its web UI:

```
http://<camera-ip>
admin / admin
```

Recommended encoder settings:
- **Codec:** H.265
- **Bitrate:** 2-8 Mbps (CBR)
- **GOP / Keyframe interval:** 1 (every frame is a keyframe — minimum latency)
- **Profile:** Main or Baseline
- **Quality:** Lowest (favors speed over visual fidelity)
- **Frame rate:** 30 or 60 fps

## Performance benchmarks

Measured on RTX 3070, 1080p H.265 stream, GStreamer software decode:

| Stage | Time |
|---|---|
| RTSP stream age | ~20ms |
| YOLO11s-pose (TensorRT FP16) | ~9ms |
| ONVIF ContinuousMove | ~16ms |
| **Total system latency** | **~45ms** |
| Display FPS | 60-160 (limited by display refresh) |

## ROS 2 topics (when `--ros` flag enabled)

| Topic | Type | Direction |
|---|---|---|
| `/camera/image_raw` | `sensor_msgs/Image` | Published |
| `/camera/cmd_vel` | `geometry_msgs/Twist` | Published (current PTZ command) |
| `/follow/target` | `geometry_msgs/PointStamped` | Published (target pixel coords) |

For the bare-bones `extract_camera.py`:

| Topic | Type | Direction |
|---|---|---|
| `/camera/image_raw` | `sensor_msgs/Image` | Published |
| `/camera/camera_info` | `sensor_msgs/CameraInfo` | Published |
| `/camera/ptz_state` | `geometry_msgs/Pose` | Published |
| `/camera/cmd_vel` | `geometry_msgs/Twist` | Subscribed |
| `/camera/goto_position` | action `PtzGoto` | Action server |

## Troubleshooting

**"GStreamer pipeline failed"**
Make sure the camera codec parameter matches reality. If the camera streams H.265, run with `--codec h265`. If H.264, use `--codec h264`.

**"CUDA unknown error"**
Reload NVIDIA kernel modules:
```bash
sudo rmmod nvidia_uvm
sudo modprobe nvidia_uvm
```

**"cv_bridge / numpy 2.x error"**
Force numpy 1.x: `pip install "numpy==1.26.4" --force-reinstall`

**Video lag is high**
Open the camera web UI and set bitrate to 2-4 Mbps, GOP to 1, and shutter speed to manual 1/250 or faster.

**Camera goes opposite direction in auto follow**
Press `T` to toggle tilt invert or `p` to toggle pan invert. The defaults assume the camera is mounted upside down.

## Project structure

```
McLaren_Bot/
├── follow_person.py           # Self-contained follower (main script)
├── extract_camera.py          # ROS 2 base camera node
├── README.md                  # This file
└── ros2_ptz_camera/           # Optional ROS 2 package layout
    ├── package.xml
    ├── setup.py
    ├── ros2_ptz_camera/
    │   ├── onvif_camera_node.py
    │   └── person_follow_node.py
    ├── launch/
    │   └── ptz_camera.launch.py
    └── config/
        └── camera_params.yaml
```

## Credits

Built on:
- [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) for pose detection
- [onvif-zeep](https://github.com/FalkTannhaeuser/python-onvif-zeep) for ONVIF protocol
- [TensorRT](https://developer.nvidia.com/tensorrt) for GPU-accelerated inference
- [GStreamer](https://gstreamer.freedesktop.org/) for low-latency video

Tracking algorithms inspired by aerospace systems:
- **Alpha-beta filter** (Sklansky, 1957) — radar target tracking
- **Image-based visual servoing** — robotics literature, simplified for PTZ control

## License

MIT
