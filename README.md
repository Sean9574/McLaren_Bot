# McLaren Bot — PTZ Person Follower

Self-contained Python tracker that follows a person with a Minrray (or any ONVIF) PTZ camera. Uses YOLO11 pose detection on the GPU, low-latency GStreamer video, and an alpha-beta filter + image-center servo controller.

## Install & run

```bash
git clone https://github.com/Sean9574/McLaren_Bot.git
cd McLaren_Bot
pip install -r requirements.txt
python follow_person.py
```

Default camera IP: `192.168.8.195` (admin / admin). Override with:

```bash
python follow_person.py --ip 192.168.1.100 --user admin --password yourpw
```

## Operation

| Key | Action |
|---|---|
| Click on person | Lock target |
| `f` | Toggle follow on/off |
| `a` | Toggle auto-acquire |
| `c` | Clear target |
| `h` | Send camera home |
| `x` | Force stop |
| Arrow keys / WASD | Manual move |
| `+` / `-` | Manual zoom |
| `q` | Quit |

When the target is lost for >5 seconds, the camera automatically returns home.

## Fine-tuning

A live "PTZ Tuning" slider window opens alongside the video. Adjust without restarting:

| Slider | Effect |
|---|---|
| **Dead Zone** | Pixels of error tolerated before camera moves |
| **Center Gain** | How aggressively camera reacts to error |
| **Max Pan / Tilt Speed** | Top ONVIF velocity (0-1.0) |
| **Min Speed** | Floor for tiny corrections |
| **Accel Limit** | How fast velocity can change between commands |
| **Target Smooth (α)** | Filter responsiveness to new measurements |
| **Velocity Smooth (β)** | Filter responsiveness to velocity changes |
| **Cmd Interval** | Minimum ms between PTZ commands |
| **Keypoint Conf** | YOLO pose confidence threshold |

If auto-follow goes the wrong direction, press `p` (pan invert) or `T` (tilt invert) and `F` to flip the image rotation.

## License

MIT