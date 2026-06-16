# McLaren Room Scanner — Fall Risk Assessment

A PTZ camera sweeps a room; a GPU server stitches a panorama and runs SAM 3
segmentation to find fall hazards (HOME FAST framework); an interactive 3D
viewer shows the results.

```
Laptop (capture + view)  <--- rsync over SSH --->  GPU server (all compute)
```

`run_scan.py process` pushes code + session data to the server, runs the
pipeline stage there, and pulls results back — one command, no manual SSH.

## Setup (one-time)

**1. Laptop environment**
```bash
conda activate ros_humble
pip install -r requirements_laptop.txt
```

**2. Server access** — confirm you can SSH in (config defaults to
`sbrainard@10.0.11.14`, see `config/sweep.yaml` → `server:`):
```bash
ssh sbrainard@10.0.11.14
```

**3. SAM 3 weights** (gated, ~3.2GB, auto-downloaded on first use):
   - Request access: https://huggingface.co/facebook/sam3
   - Create a token: https://huggingface.co/settings/tokens
   - Add it to your laptop shell (forwarded to the server automatically):
     ```bash
     echo 'export HF_TOKEN=hf_xxxxxxxxxxxx' >> ~/.bashrc
     source ~/.bashrc
     ```

**4. Install server-side dependencies**, once per stage you plan to use:
```bash
python run_scan.py setup-server --stage stitch
python run_scan.py setup-server --stage segment
```

## Usage

**1. Capture** (laptop, on lab WiFi, camera reachable at `10.0.11.162`):
```bash
python run_scan.py capture --session my_room_01
```

**2. Process** (laptop — pushes, runs remotely, pulls back automatically):
```bash
python run_scan.py process --session my_room_01 --stage stitch
python run_scan.py process --session my_room_01 --stage segment
```

**3. View** (laptop):
```bash
python run_scan.py view --session my_room_01
```
Drag to look around the panorama; click a hazard marker for details.

## Configuration

Everything tunable lives in `config/sweep.yaml`: camera calibration
(pan/tilt range, FOV), sweep grid density, server connection, SAM 3
settings + hazard vocabulary, and viewer colors. No code changes needed.

---

## Status

**Done:** capture pipeline, automated push/process/pull, panorama stitching
(OpenCV auto-stitch, verified on real data), SAM 3 segmentation (32/32
frames, verified), 3D panorama viewer with clickable hazard markers
(currently shows placeholder demo hazards).

**Remaining:**
- [ ] `perception/depth_pro.py` — per-frame metric depth (stub)
- [ ] `geometry/point_cloud.py` — fuse depth + pose + SAM 3 masks into a
      labeled 3D point cloud (stub)
- [ ] `analysis/home_fast.py` — HOME FAST risk scoring → real
      `hazards.json` (stub; viewer already reads this once it exists)
- [ ] `splat/train_3dgs.py` — NeRFStudio/Splatfacto training seeded with
      the point cloud + known poses (stub)
- [ ] Viewer: add a full-splat render mode once the splat exists
- [ ] `perception/vlm.py` — purpose TBD pending `home_fast.py` needs (stub)
- [ ] Torch/CUDA drift — `ultralytics` upgrade pulled in torch
      2.12.0+cu130 (previously pinned 2.5.1+cu121); revisit if CUDA errors
      appear
- [ ] SAM 3 confidence tuning — `sam3_conf=0.25` produced high instance
      counts (22-94/frame) in testing; revisit once `home_fast.py` exists
