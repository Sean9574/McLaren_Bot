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

## Quick Reference

Everything you need to run the full system, in order, assuming setup
(below) is already done:

```bash
# 1. Capture a scan (laptop, on the lab network)
python run_scan.py capture --session my_room_01

# 2. Build the panorama (laptop -> server -> laptop, automatic)
python run_scan.py process --session my_room_01 --stage stitch

# 3. Run SAM 3 segmentation (laptop -> server -> laptop, automatic)
python run_scan.py process --session my_room_01 --stage segment

# 4. View the results (laptop)
python run_scan.py view --session my_room_01
```

### All commands

| Command | What it does |
|---|---|
| `capture --session NAME` | Sweeps the PTZ camera through the configured grid and saves frames to `sessions/NAME/`. |
| `preview --session NAME [--n N]` | Quick low-effort sweep (default 5 frames) to sanity-check camera/positioning before a full capture. |
| `process --session NAME --stage STAGE` | Pushes code + session to the server, runs `STAGE` there, pulls results back. One command does the whole round trip. `STAGE` is one of `stitch`, `segment`, `depth`, `pointcloud`, `analysis`, `splat`, or `all` (default). Only `stitch` and `segment` are implemented today — see Status. |
| `view --session NAME` | Opens the interactive 3D viewer for a processed session. |
| `setup-server --stage STAGE` | One-time install of server-side Python dependencies for a given stage (`stitch`, `depth`, `segment`, or `full`). Run once per stage before using it. |
| `transfer --session NAME` | Manually push code + session to the server without running anything (rarely needed — `process` does this automatically). |
| `pull --session NAME` | Manually pull results back from the server without re-running anything (rarely needed — `process` does this automatically). |
| `list` | Lists all local sessions. |
| `status --session NAME` | Shows the processing status of a session (which stages have completed). |
| `delete --session NAME` | Deletes a session's local data. |

## What you need before starting

- Access to the lab's PTZ camera (already set up with a static IP — see
  "Fixed lab hardware" below) and the network it's on.
- Your own account on the GPU server, reachable over SSH from your
  laptop (password or key auth both work; the scripts use plain
  `ssh`/`rsync` commands).
- Your laptop on the same network as the camera, or otherwise able to
  reach its RTSP/ONVIF ports.

## Fixed lab hardware (do not need to change)

The PTZ camera in `config/sweep.yaml` is a static, lab-assigned device —
its IP, ONVIF port, credentials, RTSP URL, and mechanical pan/tilt/FOV
specs are fixed and shared by everyone using this system. You don't need
to touch the `camera:` section unless the camera itself is physically
replaced or its network config changes.

| Setting | What it is | Lab value |
|---|---|---|
| `camera.ip` | PTZ camera's static IP | `10.0.11.162` |
| `camera.port` | Camera's ONVIF control port | `2000` |
| `camera.user` / `camera.password` | Camera login credentials | `admin` / `admin` |
| `camera.rtsp_url` | Full RTSP stream URL | `rtsp://admin:admin@10.0.11.162:554/live/av0` |
| `camera.pan_min_deg` / `pan_max_deg` | Mechanical pan range | `-170.0` / `170.0` |
| `camera.tilt_min_deg` / `tilt_max_deg` | Mechanical tilt range | `-30.0` / `90.0` |
| `camera.hfov_deg` / `vfov_deg` | Field of view at zoom=0 | `65.0` / `40.0` |

## ⚠️ Values you must change before this will work

These are tied to *your own* SSH/server access and Hugging Face account,
not the lab's shared camera — every person running this needs their own:

| Setting | Where | What it is |
|---|---|---|
| `server.primary` | `config/sweep.yaml` | The GPU server's IP address (shared infra — confirm with whoever manages it, but it's still environment-specific, not hardcoded into this repo's design) |
| `server.user` | `config/sweep.yaml` | **Your own** SSH username on that server |
| `server.data_root` | `config/sweep.yaml` | A writable home/data directory on the server with enough free space (frames + model weights add up to several GB) — usually `/home/<your-username>` |
| `HF_TOKEN` | your shell environment, not in any file | **Your own** Hugging Face access token (see step 4 below) — every user needs their own, since it's tied to your personal HF account's approval for the gated SAM 3 weights |

If something fails to connect, recheck this table first — it's almost
always a leftover placeholder value for `server.user` or a missing
`HF_TOKEN`.

## One-time setup

**1. Laptop dependencies**
```bash
conda activate ros_humble   # or any Python 3.10+ environment
pip install -r requirements_laptop.txt
sudo apt install gstreamer1.0-tools gstreamer1.0-plugins-bad
```

**2. Edit `config/sweep.yaml`** — set your own `server.user` and
`server.data_root` under the `server:` section (see the table above).
The `camera:` section is already correct for the lab's camera and
shouldn't need changes.

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
   - Set it as an environment variable on your laptop (this is personal
     to your Hugging Face account — never commit it to git; it gets
     forwarded to the server automatically when needed):
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

See **Quick Reference** above for the exact commands in order. A bit more
detail on each step:

**Capture** — `my_room_01` is just a name you choose for this scan; use
anything descriptive (`living_room`, `test`, etc). The camera sweeps
through a grid of pan/tilt positions defined in `config/sweep.yaml`,
waiting for it to fully stop moving before saving each frame, and
rejecting blurry or corrupted frames automatically.

**Process** — uses the same session name you captured with. This pushes
your captured frames to the server, runs the requested stage there, and
pulls the results back automatically — no manual server login needed.
`stitch` builds the panorama; `segment` detects hazard concepts with
SAM 3. More stages are planned — see "Status" below.

**View** — opens a window where you're placed inside the room's
panorama. Click and drag to look around in any direction, scroll to
zoom. Detected hazards appear as colored markers (red = high risk,
orange = medium, yellow = low, green = confirmed-safe feature like a
grab bar) that you can click for details.

## Configuration reference

Everything that varies between installations lives in
`config/sweep.yaml`, grouped into:
- `sweep:` — how densely the camera scans (grid size, settle time, motion
  detection sensitivity). Defaults are conservative; reduce settle time
  or grid density for faster scans once you trust your camera's behavior.
  Generally safe to leave as-is when first trying this on new hardware.
- `camera:` — network address, credentials, and mechanical calibration
  for the lab's PTZ camera. **Already set correctly — see "Fixed lab
  hardware" above.** Only touch this if the physical camera changes.
- `server:` — SSH connection details and remote file paths. **Must be set
  to your own values** — see the "must change" table above.
- `processing:` — SAM 3 settings (confidence threshold, the list of
  hazard concepts it looks for, which you can freely edit to add/remove
  hazard types) and HOME FAST scoring thresholds. Safe defaults; edit the
  `hazard_concepts` list to tune what gets detected.
- `viewer:` — window title, marker colors. Cosmetic only.

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
`HF_TOKEN` isn't set/exported on your laptop (run `echo $HF_TOKEN` to
check). Request access and wait for approval; in the meantime you can
still use the `stitch` stage.

**Can't SSH / `rsync` fails with "connection refused" or similar** —
double check `server.primary` and `server.user` in `config/sweep.yaml`
match a server you can actually reach, and that you can manually
`ssh <user>@<ip>` from the laptop first.

**Panorama looks distorted or stitching fails entirely** — almost always
incorrect `pan_min_deg`/`pan_max_deg`/`hfov_deg`/`vfov_deg` values. These
are fixed mechanical specs for the lab's camera (see "Fixed lab
hardware") and shouldn't normally need editing — if you're seeing this,
double-check nothing in `config/sweep.yaml` was accidentally changed.

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
