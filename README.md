# McLaren Bot — PTZ Person Follower

Tracks people with a Minrray (or any ONVIF) PTZ camera using YOLO11 pose detection, TensorRT-accelerated inference, and a GStreamer low-latency video pipeline.

## Hardware

- ONVIF-compatible PTZ camera (tested on Minrray)
- NVIDIA GPU (tested on RTX 3070)
- Ubuntu 22.04, Python 3.10

## Install

```bash
git clone https://github.com/Sean9574/McLaren_Bot.git
cd McLaren_Bot

# System packages
sudo apt install -y \
    gstreamer1.0-tools gstreamer1.0-plugins-{base,good,bad,ugly} \
    gstreamer1.0-libav ffmpeg ros-humble-cv-bridge

# Python env
conda create -n ros_humble python=3.10 -y
conda activate ros_humble
source /opt/ros/humble/setup.bash
pip install -r requirements.txt
```

## Run

```bash
python follow_person.py
```

First run auto-downloads YOLO11s-pose and exports a TensorRT engine (~3-5 min, one time).

## Controls

| Key | Action |
|---|---|
| Click | Lock target |
| `f` | Follow on/off |
| `a` | Auto-acquire toggle |
| `c` | Clear target |
| `h` | Home |
| `x` | Stop |
| `F` | Rotate 180° |
| `p` / `T` | Invert pan / tilt |
| Arrows / WASD | Manual move |
| `q` | Quit |

## Common flags

```bash
python follow_person.py --ip 192.168.1.100      # custom IP
python follow_person.py --codec h264            # H.264 instead of H.265
python follow_person.py --auto-acquire          # auto-lock nearest person
python follow_person.py --ros                   # also publish ROS 2 topics
python follow_person.py --no-trt --device cpu   # CPU fallback
```

## Performance (RTX 3070, 1080p H.265)

| Stage | Latency |
|---|---|
| RTSP stream | ~20ms |
| YOLO11s-pose (TensorRT FP16) | ~9ms |
| ONVIF command | ~16ms |
| **Total** | **~45ms** |

## Troubleshooting

**GStreamer fails:** mismatch between `--codec` flag and camera stream. Try `--codec h264`.

**CUDA unknown error:** `sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm`

**numpy / cv_bridge crash:** `pip install "numpy==1.26.4" --force-reinstall`

**Camera moves wrong direction in auto-follow:** press `p` (pan) or `T` (tilt) to invert.

**High lag:** in the camera web UI, set bitrate to 2-4 Mbps, GOP=1, shutter ≥ 1/250.

## License

MIT