# McLaren Room Scanner — Fall Risk Assessment

A PTZ (pan-tilt-zoom) camera scans a room and the frames are stitched into a
360-degree panorama. SAM 3, Meta's text-prompted segmentation model, then
finds fall hazards from the HOME FAST (Home Falls and Accidents Screening
Tool) checklist, and an interactive 3D viewer shows the results.

Capture and viewing run on your computer. Processing runs in one of two places:

- **Lab server** (default). This is the lab's GPU server, and you need to be
  given an account on it. `process` uploads your frames over SSH, runs the
  stage there and downloads the results automatically.
- **Your own computer** (fallback if you have no server access). Add `--local`
  to `process`. Stitching runs on any machine; hazard detection needs an
  NVIDIA GPU.

This repo also contains a separate person-following bot. See
[Other scripts](#other-scripts).

## Quick Reference

Run every command from the repo root. Session data goes in `sessions/`.

```bash
python run_scan.py capture --session my_room_01                  # 1. scan the room
python run_scan.py process --session my_room_01 --stage stitch   # 2. build panorama
python run_scan.py process --session my_room_01 --stage segment  # 3. detect hazards
python run_scan.py view    --session my_room_01                  # 4. view results
```

**No lab server access?** Add `--local` to steps 2 and 3 to run everything on
your computer.

> **Always pass `--stage`.** If you leave it out, `process` runs `all`, which
> starts with the depth stage. That stage isn't built yet, so it fails before
> the panorama is made (see [Status](#status)).

| Command | Does |
|---|---|
| `capture --session NAME` | Sweeps the camera and saves frames. Re-running resumes; `--no-resume` starts over |
| `preview --session NAME` | Quick test sweep (5 frames; change with `--n`) |
| `process --session NAME --stage STAGE` | Uploads to the lab server, runs `STAGE` there and downloads the results. `STAGE` = `stitch` or `segment` |
| `process --session NAME --stage STAGE --local` | Runs `STAGE` on this computer instead. No server needed |
| `view --session NAME` | Opens the 3D viewer |
| `list` | Lists all sessions |
| `status --session NAME` | Shows one session's progress |
| `delete --session NAME` | Deletes a session (asks first) |
| `setup-server --stage STAGE` | Lab server only: one-time dependency install |
| `transfer --session NAME` / `pull --session NAME` | Lab server only: manual upload/download (rarely needed; `process` does both) |

The lab server commands also take `--server IP` to use a different server
than `server.primary` in the config.

## Requirements

- **Python 3.10.** The lab uses the `ros_humble` conda env, but the scanner
  itself doesn't need ROS.
- **Lab WiFi** to reach the camera during `capture` and `preview`.
- **Lab server path:** an account on the lab GPU server (ask the lab to set
  one up) and SSH key login.
- **Local path:** an NVIDIA GPU with CUDA for `segment`. `stitch` and `view`
  work on any computer.
- **For `segment` on either path:** a Hugging Face account approved for SAM 3
  (see setup step 2).

## Setup (one-time)

### 1. Your computer (everyone)

```bash
conda activate ros_humble        # or any Python 3.10 env
pip install onvif-zeep opencv-python "numpy<2" PyYAML open3d
sudo apt install rsync gstreamer1.0-tools gstreamer1.0-plugins-bad
```

Don't use `requirements.txt` for this. It belongs to the person-following bot.

### 2. SAM 3 weights (needed for `segment`)

The weights are gated and download automatically on first use. First
request access at [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3),
then make a token at
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), then:

```bash
echo 'export HF_TOKEN=hf_xxxxxxxxxxxx' >> ~/.bashrc && source ~/.bashrc
```

When processing runs on the lab server, `process` passes your token along
automatically.

### 3a. Lab server (if you have an account)

1. **Get an account** on the lab GPU server. Ask the lab; you can't sign up
   yourself.
2. **Set up SSH key login.** `process` connects without a password prompt, so a
   password-only login won't work. The server's address is `server.primary`
   in `config/sweep.yaml`.
   ```bash
   ssh-copy-id <your-user>@<server-ip>
   ssh <your-user>@<server-ip>       # should log in without asking for a password
   ```
3. **Edit the `server:` block in `config/sweep.yaml`** and replace every
   `sbrainard` value with your own (see [Configuration](#configuration)).
4. **Install server deps** once per stage:
   ```bash
   python run_scan.py setup-server --stage stitch
   python run_scan.py setup-server --stage segment
   ```

### 3b. Your own computer (no server account)

Skip the `server:` config entirely. `stitch` needs nothing beyond step 1.
For `segment` (NVIDIA GPU only), install:

```bash
pip install torch torchvision
pip install -U ultralytics huggingface_hub ftfy regex
pip uninstall -y clip openai-clip ultralytics-clip
pip install git+https://github.com/ultralytics/CLIP.git
pip install "numpy<2"            # must come last (see Troubleshooting)
```

Then add `--local` to every `process` command.

## Configuration

All settings live in `config/sweep.yaml`.

**Fixed, don't change:** the `camera:` block (`ip`, `port`, `user`/`password`,
`rtsp_url`, pan/tilt/FOV degrees). It describes the lab's shared camera, and
editing it breaks stitching.

**Change to your own** (lab server users only):

| Setting | What |
|---|---|
| `server.user` | Your SSH username on the lab server |
| `server.data_root` | Writable server path, e.g. `/home/<you>` |
| `server.remote_base` | Where sessions go on the server, e.g. `/home/<you>/mclaren_room_scanner/sessions` |
| `server.remote_code_dir` | Where code goes on the server, e.g. `/home/<you>/mclaren_room_scanner` |
| `HF_TOKEN` (env var, not in a file) | Your personal Hugging Face token (everyone who runs `segment`) |

`server.fallback` is a second lab server. It isn't used automatically; pass
`--server <its-ip>` to use it.

**Tune anytime:** `sweep:` (grid density, settle time),
`processing.hazard_concepts` (what SAM 3 looks for), `processing.sam_gpu`
(which GPU runs SAM 3), `viewer:` (colors, cosmetic).

## Troubleshooting

- **`Cannot reach <ip> as <user>`**: SSH key login isn't working. Check
  `server.user` and run `ssh-copy-id` (setup step 3a). If you don't have a
  server account, add `--local` instead.
- **`ImportError: cannot import name 'run_depth'`**: you left out `--stage`,
  so it ran `all`. Pass `--stage stitch` or `--stage segment`.
- **`No manifest at sessions/.../manifest.json`**: the upload didn't finish
  before processing started. Check that the rsync output completed.
- **`_ARRAY_API not found`**: NumPy version conflict. Re-run `setup-server`,
  or locally run `pip install "numpy<2"`.
- **`No module named 'clip'` / `SimpleTokenizer not callable`**: known `clip`
  package conflict. Re-run `setup-server --stage segment`, or locally re-run
  the CLIP lines from step 3b.
- **SAM 3 download 403/gated error**: your HF account isn't approved yet, or
  `HF_TOKEN` isn't set (check with `echo $HF_TOKEN`).
- **CUDA / device errors on `segment --local`**: SAM 3 needs an NVIDIA GPU.
  Use the lab server, or check `processing.sam_gpu`.
- **Distorted panorama**: check that nothing in `camera:` was edited by
  accident.

## Status

**Working:** capture, automated server upload/process/download, local
processing (`--local`), panorama stitching, SAM 3 segmentation, 3D viewer
(it shows placeholder demo hazards for now).

**Not built yet** (these files are empty stubs):
- [ ] `perception/depth_pro.py`: per-frame depth estimation (`--stage depth`)
- [ ] `geometry/point_cloud.py`: combines depth, pose and masks into a labeled 3D point cloud (`--stage pointcloud`)
- [ ] `analysis/home_fast.py`: real hazard scoring to replace the demo markers (`--stage analysis`)
- [ ] `splat/train_3dgs.py`: 3D Gaussian Splatting / NeRFStudio training (`--stage splat`)
- [ ] Viewer mode for the trained splat (today's viewer is a panorama with markers, not full 3D)
- [ ] `perception/vlm.py`: vision-language model; purpose TBD

## Other scripts

These are not part of the room scanner:

- **`follow_person.py`**: standalone PTZ person follower using YOLO, with
  optional TensorRT. Its dependencies are in `requirements.txt`, and
  `run_lab.sh` shows the launch command with the lab camera's settings.
- **`extract_camera.py`**: ROS 2 Humble camera node. It publishes the camera
  stream and PTZ state and takes PTZ commands on `/camera/cmd_vel`.
  Requires ROS 2 Humble.
