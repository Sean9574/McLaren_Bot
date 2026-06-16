# McLaren Room Scanner — Fall Risk Assessment

A PTZ camera scans a room, frames get stitched into a 360-degree panorama,
SAM 3 detects fall hazards (HOME FAST framework), and an interactive 3D
viewer shows the results. Capture/viewing run on a laptop; processing runs
on a GPU server, talked to automatically over SSH.

## Quick Reference

```bash
python run_scan.py capture --session my_room_01                  # 1. scan the room
python run_scan.py process --session my_room_01 --stage stitch   # 2. build panorama
python run_scan.py process --session my_room_01 --stage segment  # 3. detect hazards
python run_scan.py view --session my_room_01                     # 4. view results
```

| Command | Does |
|---|---|
| `capture --session NAME` | Sweeps the camera, saves frames |
| `preview --session NAME` | Quick 5-frame test sweep |
| `process --session NAME --stage STAGE` | Push → run `STAGE` on server → pull results. `STAGE` = `stitch`, `segment` (others planned, see Status) |
| `view --session NAME` | Opens the 3D viewer |
| `setup-server --stage STAGE` | One-time server dependency install |
| `status` / `list` / `delete --session NAME` | Manage sessions |
| `transfer` / `pull` | Manual push/pull (rarely needed — `process` does both) |

## Setup (one-time)

1. **Laptop deps:**
   ```bash
   conda activate ros_humble
   pip install -r requirements_laptop.txt
   sudo apt install gstreamer1.0-tools gstreamer1.0-plugins-bad
   ```
2. **Edit `config/sweep.yaml`** — set `server.user` and `server.data_root`
   to your own. Leave `camera:` alone (see below).
3. **Confirm SSH access:** `ssh <your-user>@<server-ip>`
4. **SAM 3 weights** (gated, auto-downloaded): request access at
   [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3),
   make a token at
   [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens),
   then:
   ```bash
   echo 'export HF_TOKEN=hf_xxxxxxxxxxxx' >> ~/.bashrc && source ~/.bashrc
   ```
5. **Install server deps** (once per stage):
   ```bash
   python run_scan.py setup-server --stage stitch
   python run_scan.py setup-server --stage segment
   ```

## Configuration

All settings live in `config/sweep.yaml`.

**Fixed — don't change** (lab's shared camera):
`camera.ip`, `port`, `user`/`password`, `rtsp_url`, pan/tilt/FOV degrees.

**Change to your own:**

| Setting | What |
|---|---|
| `server.user` | Your SSH username |
| `server.data_root` | Writable server path, e.g. `/home/<you>` |
| `HF_TOKEN` (env var, not in a file) | Your personal Hugging Face token |

**Tune anytime:** `sweep:` (grid density, settle time), `processing.hazard_concepts`
(what SAM 3 looks for), `viewer:` (colors, cosmetic).

## Troubleshooting

- **`No manifest at sessions/.../manifest.json`** — upload step didn't finish before processing started; check the rsync output completed.
- **`_ARRAY_API not found`** — NumPy version conflict; re-run `setup-server`.
- **`No module named 'clip'` / `SimpleTokenizer not callable`** — known `clip` package conflict, handled by `setup-server --stage segment`; re-run if it persists.
- **SAM 3 download 403/gated error** — your HF account isn't approved yet, or `HF_TOKEN` isn't set (`echo $HF_TOKEN` to check).
- **Can't SSH/rsync** — check `server.user` in the config and that you can manually SSH in.
- **Distorted panorama** — check nothing in `camera:` got accidentally edited.

## Status

**Working:** capture, automated server push/process/pull, panorama stitching,
SAM 3 segmentation, 3D viewer (currently shows placeholder demo hazards).

**Not built yet:**
- [ ] `perception/depth_pro.py` — per-frame depth estimation
- [ ] `geometry/point_cloud.py` — depth + pose + masks → labeled 3D point cloud
- [ ] `analysis/home_fast.py` — real hazard scoring (replaces demo markers)
- [ ] `splat/train_3dgs.py` — Gaussian Splat / NeRFStudio training
- [ ] Viewer mode for the trained splat (today's viewer is panorama + markers, not full 3D)
- [ ] `perception/vlm.py` — purpose TBD
