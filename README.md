# McLaren Room Scanner — Fall Risk Assessment System

PTZ camera sweep → panorama stitch → SAM 3 segmentation → (depth + point cloud
+ 3D Gaussian splat — in progress) → HOME FAST fall risk analysis → interactive
3D viewer.

Built for the McLaren lab fall-risk-assessment demo: a PTZ camera does a
rotation-only sweep of a room, frames are processed on a GPU server, and the
results (panorama + detected hazards) are viewed in an interactive 3D viewer
on the laptop.

## Architecture

```
Laptop (capture + view)  <-- rsync -->  GPU server (all processing)
```

One command on the laptop (`run_scan.py process`) pushes code + session data
to the server over SSH/rsync, runs the requested pipeline stage remotely,
and pulls the results back automatically. The server is treated as pure
compute — no manual SSH steps required day-to-day.

## Quick Start

### 1. Capture (laptop, on lab WiFi, camera reachable)
```bash
conda activate ros_humble
python run_scan.py capture --session my_room_01
```
Sweeps the PTZ camera through a calibrated grid (see `config/sweep.yaml`),
waiting for the camera to fully stop and checking for corruption before
saving each frame.

### 2. Process on the server (one command, fully automated)
```bash
python run_scan.py process --session my_room_01 --stage stitch
python run_scan.py process --session my_room_01 --stage segment
```
This pushes code + frames to the server, runs the stage, and pulls results
back — no manual SSH needed. Available stages: `stitch`, `depth` (pending),
`segment`, `full`.

First time using a stage, install its server-side dependencies:
```bash
python run_scan.py setup-server --stage segment
```

### 3. View results (laptop)
```bash
python run_scan.py view --session my_room_01
```
Opens an interactive 3D viewer: you're placed inside a curved screen
textured with the room panorama (drag to look around), with fall-risk
hazards shown as clickable colored markers.

## Configuration

All tunables live in `config/sweep.yaml` — no code changes needed for:
camera calibration (pan/tilt range, FOV), sweep grid density, server
connection details, SAM 3 settings, the HOME FAST hazard concept
vocabulary, and viewer colors.

## One-time setup: SAM 3 weights

SAM 3 weights (`sam3.pt`, ~3.2GB) are gated on Hugging Face and are **not**
committed to this repo (see `.gitignore`). To enable auto-download:

1. Request access at https://huggingface.co/facebook/sam3 (Meta approves
   these, usually within a day or two)
2. Create a read-access token at https://huggingface.co/settings/tokens
3. Set it in your shell (laptop side — it gets forwarded to the server
   automatically over SSH):
   ```bash
   echo 'export HF_TOKEN=hf_xxxxxxxxxxxx' >> ~/.bashrc
   source ~/.bashrc
   ```

The first `process --stage segment` run will download and cache `sam3.pt`
on the server automatically. Subsequent runs reuse the cached weights.

## Requirements
See `requirements_laptop.txt` and `requirements_server.txt`. Server-side
Python deps (torch, ultralytics, open3d, etc.) are also installed
automatically per-stage via `run_scan.py setup-server --stage <name>`.

---

## Status

### Done
- **Capture pipeline** — two-node architecture (stream + PTZ control),
  motion-stability detection, corruption checks. 32/32 frames captured
  cleanly in testing.
- **Server automation** — `process` does push → remote-run → pull in one
  command; per-stage dependency installation with disk-space and
  NumPy/CLIP ABI-conflict fixes baked in.
- **Panorama stitching** (`geometry/pano_stitch.py`) — OpenCV auto-stitch
  with a pose-based equirectangular fallback. Verified on real room data
  (9173x1797, clean seams, full coverage).
- **SAM 3 segmentation** (`perception/sam3.py`) — promptable concept
  segmentation across the HOME FAST hazard vocabulary (26 concepts:
  rugs, cords, grab bars, unstable furniture, etc.), batched per frame.
  Verified: 32/32 frames segmented successfully on real data.
- **3D panorama viewer** (`viewer/app.py`) — interactive curved-screen
  viewer sized to the camera's real FOV (not a generic equirectangular
  sphere, which distorted wide/short pano crops), with clickable hazard
  markers, severity legend, and a 2D panorama thumbnail. Currently shows
  placeholder `DEMO_HAZARDS` until real hazard data exists (see below).

### Remaining work
- [ ] **`perception/depth_pro.py`** — per-frame metric depth estimation
      (stub only). Needed because the capture is rotation-only (no
      parallax), so depth must come from a monocular depth model rather
      than triangulation/COLMAP.
- [ ] **`geometry/point_cloud.py`** — fuse depth + known PTZ pose + SAM 3
      masks into a single labeled 3D point cloud (PLY) and a
      `transforms.json` describing camera poses, for seeding the splat
      trainer below (stub only).
- [ ] **`analysis/home_fast.py`** — HOME FAST risk scoring over the
      labeled point cloud, writes the real
      `sessions/<name>/outputs/hazards.json` that the viewer is already
      wired to read (stub only — viewer currently falls back to demo
      data).
- [ ] **NeRFStudio / Splatfacto integration** (`splat/train_3dgs.py`) —
      train a 3D Gaussian Splat seeded with the point cloud above and the
      known PTZ poses (skipping COLMAP, since poses are already known).
      Stub only; server-side NeRFStudio install not yet done.
- [ ] **Viewer upgrade to full splat mode** — once the splat exists, add
      a second viewer mode that renders the trained 3D Gaussian Splat
      directly (current viewer is a 360 degree panorama + flat hazard
      markers, which is a fully working interim deliverable but not the
      final "true 3D" reconstruction).
- [ ] **`perception/vlm.py`** — currently an empty stub; original purpose
      (VLM-based hazard description/captioning vs. SAM 3 alone) needs
      revisiting once `home_fast.py` exists and we know what extra
      context, if any, the risk-scoring stage actually needs.
- [ ] **Torch/CUDA version drift** — a recent `pip install -U ultralytics`
      pulled in torch 2.12.0+cu130 as a transitive dependency (server was
      previously on a known-good 2.5.1+cu121). Works today on the A100,
      but if future installs hit CUDA driver mismatches, pin the torch
      version explicitly in `capture/transfer.py`'s install commands.
- [ ] **SAM 3 confidence tuning** — initial real-data run produced high
      per-frame instance counts (22-94) at `sam3_conf=0.25` in a visually
      cluttered lab. Plausible given the environment, but worth a pass
      once `home_fast.py` exists to see whether false positives are
      inflating hazard counts.
