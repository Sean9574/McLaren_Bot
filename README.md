# McLaren Room Scanner — Fall Risk Assessment

This system uses a PTZ (pan-tilt-zoom) camera to scan a room, stitches the
captured images into a 360-degree panorama, detects fall-risk hazards in
the scene (loose rugs, cables, low furniture, missing grab bars, etc.)
using the SAM 3 segmentation model and the clinical HOME FAST framework,
and presents the results in an interactive 3D viewer you can look around
in and click on.

It's split into two halves that talk to each other over SSH:

- **A laptop**, which controls the camera and runs the viewer.
- **A GPU server**, which does the actual image processing (stitching,
  segmentation, depth, eventually 3D reconstruction). A modern NVIDIA GPU
  with at least ~16GB VRAM is recommended; the reference setup uses an
  A100.

You run one command on the laptop and it handles pushing data to the
server, running the work there, and pulling the results back. You never
need to manually log into the server for day-to-day use — only for the
one-time setup below.

## What you need before starting

- An ONVIF-compatible PTZ camera, reachable over the network, with RTSP
  streaming enabled.
- A GPU server reachable over SSH from the laptop (password or key auth
  both work; the scripts use plain `ssh`/`rsync` commands).
- Both machines on the same network as the camera, or otherwise able to
  reach its RTSP/ONVIF ports.

The included `config/sweep.yaml` ships with placeholder values from the
lab this was built in (camera IP `10.0.11.162`, server IP `10.0.11.14`,
server user `sbrainard`, etc.) — **you must replace these with your own**
before anything will work. See "Configuration" below for exactly which
fields to change.

## One-time setup

**1. Laptop dependencies**
```bash
conda activate ros_humble   # or any Python 3.10+ environment
pip install -r requirements_laptop.txt
sudo apt install gstreamer1.0-tools gstreamer1.0-plugins-bad
```

**2. Point the config at your hardware**

Open `config/sweep.yaml` and edit:
- `camera:` — your camera's IP, ONVIF port, RTSP credentials/URL, and its
  mechanical pan/tilt range in degrees (check your camera's spec sheet —
  these vary a lot between models).
- `server:` — your server's IP and SSH username.

**3. Confirm SSH access to the server** (key-based auth recommended so the
automated steps don't prompt for a password mid-pipeline):
```bash
ssh <your-server-user>@<your-server-ip>
```

**4. SAM 3 model weights** — these are gated on Hugging Face and are not
included in this repo. To let the pipeline download them automatically:
   - Request access at https://huggingface.co/facebook/sam3 (approval is
     usually fast but isn't instant)
   - Create a read-access token at https://huggingface.co/settings/tokens
   - Set it as an environment variable on your laptop — it gets forwarded
     to the server automatically when needed:
     ```bash
     echo 'export HF_TOKEN=hf_xxxxxxxxxxxx' >> ~/.bashrc
     source ~/.bashrc
     ```
   If you'd rather not wait on approval, you can instead download
   `sam3.pt` (~3.2GB) manually once approved and place it on the server at
   the path given by `processing.sam3_weights` in the config.

**5. Install server-side Python dependencies**, once per pipeline stage
you intend to use. This runs entirely over SSH from the laptop:
```bash
python run_scan.py setup-server --stage stitch
python run_scan.py setup-server --stage segment
```
This installs the right Python packages for that stage's libraries
(OpenCV, PyTorch with CUDA, Ultralytics/SAM 3, etc.) and works around a
couple of known environment issues automatically (a NumPy 1.x/2.x ABI
conflict with system `matplotlib`, and a package-name collision between
the `clip` library SAM 3 needs and an unrelated PyPI package also named
`clip`). You shouldn't need to think about either of these — they're
handled for you — but if `setup-server` fails, the printed output will
say exactly which step failed.

## Day-to-day usage

**1. Capture a scan** (run from the laptop, on the same network as the
camera):
```bash
python run_scan.py capture --session my_room_01
```
The camera sweeps through a grid of pan/tilt positions defined in
`config/sweep.yaml`, waiting for it to fully stop moving before saving
each frame (and rejecting blurry/corrupted frames automatically).

**2. Process the scan** (run from the laptop — this pushes your captured
frames to the server, runs the requested stage there, and pulls the
results back automatically; no manual server login needed):
```bash
python run_scan.py process --session my_room_01 --stage stitch
python run_scan.py process --session my_room_01 --stage segment
```
Available stages today: `stitch` (builds the panorama) and `segment`
(detects hazard concepts with SAM 3). More stages are planned — see
"Status" below.

**3. View the results** (laptop):
```bash
python run_scan.py view --session my_room_01
```
This opens a window where you're placed inside the room's panorama —
click and drag to look around in any direction, scroll to zoom. Detected
hazards appear as colored markers (red = high risk, orange = medium,
yellow = low, green = confirmed-safe feature like a grab bar) that you
can click for details.

## Configuration reference

Everything that varies between installations lives in
`config/sweep.yaml`, grouped into:
- `sweep:` — how densely the camera scans (grid size, settle time, motion
  detection sensitivity). Defaults are conservative; reduce settle time
  or grid density for faster scans once you trust your camera's behavior.
- `camera:` — network address, credentials, and mechanical calibration.
  **You must set this correctly for your hardware** — the pan/tilt degree
  ranges especially, since panorama stitching depends on them.
- `server:` — SSH connection details and remote file paths.
- `processing:` — SAM 3 settings (confidence threshold, the list of
  hazard concepts it looks for, which you can freely edit to add/remove
  hazard types) and HOME FAST scoring thresholds.
- `viewer:` — window title, marker colors.

## Troubleshooting

**`No manifest at sessions/<name>/manifest.json`** — the session directory
wasn't pushed to the server before processing started, usually because
the parent `sessions/` directory didn't exist remotely yet. This is
handled automatically as of the current version; if you still see it,
check that `python run_scan.py process` completed its upload step (look
for the rsync output) before the remote command ran.

**`AttributeError: _ARRAY_API not found` (NumPy-related crash during
segment/depth stages)** — a NumPy version conflict between newly-installed
packages and the system's pre-existing libraries. `setup-server` pins a
compatible NumPy version as its last step; re-run it if you see this.

**`No module named 'clip'` or `'SimpleTokenizer' object is not
callable`** — a known conflict between the `clip` package SAM 3 expects
and a different PyPI package of the same name. `setup-server --stage
segment` and `sam3.py` both work around this automatically; if it still
fails, the setup output will show exactly which install step failed.

**SAM 3 weight download fails with a 403 or "gated repo" error** — your
Hugging Face account hasn't been approved for `facebook/sam3` yet, or
`HF_TOKEN` isn't set/exported. Request access and wait for approval; in
the meantime you can still use the `stitch` stage.

## Status

**Working today:** camera capture (grid sweep with motion/corruption
checks), fully automated push/process/pull to the GPU server, panorama
stitching (OpenCV feature-based auto-stitch with a pose-based fallback),
SAM 3 hazard segmentation, and an interactive 3D panorama viewer with
clickable hazard markers. The viewer currently displays placeholder demo
hazards, since real hazard scoring isn't wired up yet (next item below).

**Not yet built:**
- [ ] Per-frame metric depth estimation (`perception/depth_pro.py`)
- [ ] Fusing depth + camera pose + SAM 3 masks into a labeled 3D point
      cloud (`geometry/point_cloud.py`)
- [ ] HOME FAST risk scoring that turns the point cloud into the real
      hazard list the viewer displays (`analysis/home_fast.py`)
- [ ] 3D Gaussian Splat training for a true 3D reconstruction, seeded
      with the point cloud and known camera poses (`splat/train_3dgs.py`)
- [ ] A viewer mode that renders the trained splat directly, once it
      exists (today's viewer shows a 360-degree photo, not a full 3D
      reconstruction)
- [ ] `perception/vlm.py` — placeholder; purpose to be decided once the
      risk-scoring stage is built and we know what extra context, if any,
      a vision-language model would add beyond SAM 3's labels
