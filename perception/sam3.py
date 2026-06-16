"""
perception/sam3.py — Promptable Concept Segmentation with SAM 3.

Runs SAM 3 on every captured frame using the HOME FAST hazard vocabulary
defined in config.processing.hazard_concepts (e.g. "throw rug", "electrical
cord", "grab bar", ...). For each frame, saves:

  masks/masks_XXXX.npz   - boolean mask stack, shape (N, H, W), uint8
  masks/masks_XXXX.json  - per-mask metadata: concept text, confidence, bbox

These feed geometry/point_cloud.py (project mask pixels into 3D using the
frame's depth map + PTZ pose) and analysis/home_fast.py (risk scoring).

──────────────────────────────────────────────────────────────────────────
SETUP REQUIRED (one-time, on the server):

SAM 3 weights (sam3.pt, ~3.45GB) are GATED and NOT auto-downloaded.
  1. Request access: https://huggingface.co/facebook/sam3
  2. Once approved, download sam3.pt
  3. Place it at the path in config.processing.sam3_weights
     (default: <project_root>/sam3.pt)

Known issue: a conflicting 'clip' package causes
  TypeError: 'SimpleTokenizer' object is not callable
Fix:
  pip uninstall clip -y
  pip install git+https://github.com/ultralytics/CLIP.git
(setup-server --stage segment runs this automatically)
──────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import numpy as np


def _ensure_weights(weights_path: str, repo_id: str = "facebook/sam3",
                    filename: str = "sam3.pt") -> bool:
    """
    If weights_path doesn't exist, try to auto-download the gated SAM 3
    checkpoint from Hugging Face using HF_TOKEN (or HUGGING_FACE_HUB_TOKEN).

    Your HF account must already be approved for facebook/sam3 — request
    access at https://huggingface.co/facebook/sam3, then create a token at
    https://huggingface.co/settings/tokens and set it as HF_TOKEN.

    Returns True if weights are present after this call.
    """
    if Path(weights_path).exists():
        return True

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        return False

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[SAM3] huggingface_hub not installed — "
              "pip install huggingface_hub")
        return False

    print(f"[SAM3] Downloading {filename} from {repo_id} "
          f"(HF_TOKEN set, ~3.2GB, this may take a while)...")
    try:
        dest_dir = str(Path(weights_path).parent)
        downloaded = hf_hub_download(
            repo_id=repo_id, filename=filename,
            token=token, local_dir=dest_dir,
        )
        # hf_hub_download may place it in a subfolder structure matching
        # the repo layout — move/symlink to the expected flat path.
        downloaded_path = Path(downloaded)
        target_path = Path(weights_path)
        if downloaded_path.resolve() != target_path.resolve():
            if target_path.exists() or target_path.is_symlink():
                target_path.unlink()
            target_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                target_path.symlink_to(downloaded_path.resolve())
            except OSError:
                import shutil
                shutil.copy2(downloaded_path, target_path)
        print(f"[SAM3] Downloaded -> {weights_path}")
        return True
    except Exception as e:
        msg = str(e)
        if "403" in msg or "gated" in msg.lower() or "access" in msg.lower():
            print(f"[SAM3] Download failed — access not yet approved for "
                  f"{repo_id}.")
            print(f"[SAM3] Request access at "
                  f"https://huggingface.co/{repo_id}, wait for approval, "
                  f"then retry.")
        else:
            print(f"[SAM3] Download failed: {e}")
        return False


def _load_predictor(weights_path: str, conf: float, device: int):
    """Create a SAM3SemanticPredictor. Raises a clear error if weights missing."""
    if not _ensure_weights(weights_path):
        token_hint = "" if (os.environ.get("HF_TOKEN")
                            or os.environ.get("HUGGING_FACE_HUB_TOKEN"))             else ("\n  No HF_TOKEN set in environment — auto-download "
                  "skipped. To enable it:\n"
                  "    1. Request access: https://huggingface.co/facebook/sam3\n"
                  "    2. Create a token: https://huggingface.co/settings/tokens\n"
                  "    3. export HF_TOKEN=hf_xxxxx  (on the server)\n")
        raise FileNotFoundError(
            f"SAM 3 weights not found at {weights_path}{token_hint}\n"
            f"  Or download sam3.pt manually and place it at {weights_path}\n"
            f"  (or set processing.sam3_weights in config/sweep.yaml)"
        )

    # Patch a known issue: SAM3's code calls its tokenizer instance
    # directly (tokenizer(text)), but the `clip` package's SimpleTokenizer
    # class only defines .encode()/.decode(), not __call__. Depending on
    # which `clip` variant ended up installed, __call__ may be missing,
    # producing: TypeError: 'SimpleTokenizer' object is not callable
    #
    # Fix: make SimpleTokenizer instances callable by delegating to the
    # module-level clip.tokenize() function (present in all variants),
    # which performs the same encode + pad + tensor-ify SAM3 expects.
    try:
        import clip
        from clip.simple_tokenizer import SimpleTokenizer

        # Always apply the patch (idempotent) — checking whether instances
        # are already callable via hasattr(SimpleTokenizer, "__call__")
        # doesn't work: that inspects the METACLASS's __call__ (used for
        # SimpleTokenizer(...) instantiation), which is always present
        # regardless of whether *instances* are callable.
        def _tokenizer_call(self, texts, *args, **kwargs):
            try:
                return clip.tokenize(texts, *args, **kwargs)
            except TypeError:
                return clip.tokenize(texts)
        SimpleTokenizer.__call__ = _tokenizer_call
    except ImportError as e:
        raise ImportError(
            f"Could not import clip package: {e}\n"
            f"Run: pip uninstall clip -y && "
            f"pip install git+https://github.com/ultralytics/CLIP.git"
        )

    try:
        from ultralytics.models.sam import SAM3SemanticPredictor
    except ImportError as e:
        raise ImportError(
            f"Could not import SAM3SemanticPredictor: {e}\n"
            f"Requires ultralytics>=8.3.237: pip install -U ultralytics"
        )

    overrides = dict(
        conf=conf,
        task="segment",
        mode="predict",
        model=str(weights_path),
        half=True,        # FP16 for speed on A100
        save=False,
        device=device,
        verbose=False,
    )
    return SAM3SemanticPredictor(overrides=overrides)


def _chunks(lst: List[str], n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def segment_frame(predictor, image_path: Path, concepts: List[str],
                  batch_size: int = 8) -> dict:
    """
    Run all concept prompts against one image. Concepts are queried in
    batches (predictor.set_image caches image features, so re-querying
    with a new text batch on the same image is cheap).

    Returns:
        {
          "masks": np.ndarray (N, H, W) bool,
          "meta": [ {"concept": str, "confidence": float,
                     "bbox": [x1,y1,x2,y2]}, ... ]
        }
    """
    predictor.set_image(str(image_path))

    all_masks = []
    all_meta = []

    for batch in _chunks(concepts, batch_size):
        try:
            results = predictor(text=batch)
        except Exception as e:
            print(f"    [SAM3] batch {batch} failed: {e}")
            continue

        if not results:
            continue
        r = results[0]
        if r.masks is None or len(r.masks.data) == 0:
            continue

        masks = r.masks.data.cpu().numpy().astype(bool)   # (n, H, W)

        if r.boxes is not None and len(r.boxes) == len(masks):
            cls_idx = r.boxes.cls.cpu().numpy().astype(int)
            confs   = r.boxes.conf.cpu().numpy()
            bboxes  = r.boxes.xyxy.cpu().numpy()
        else:
            cls_idx = np.zeros(len(masks), dtype=int)
            confs   = np.ones(len(masks))
            bboxes  = np.zeros((len(masks), 4))

        for i in range(len(masks)):
            all_masks.append(masks[i])
            # cls index corresponds to position in THIS call's text batch
            idx = int(cls_idx[i])
            concept_text = batch[idx] if 0 <= idx < len(batch) else str(idx)
            all_meta.append({
                "concept": concept_text,
                "confidence": float(confs[i]),
                "bbox": [float(x) for x in bboxes[i]],
            })

    if all_masks:
        mask_stack = np.stack(all_masks, axis=0)
    else:
        mask_stack = np.zeros((0, 0, 0), dtype=bool)

    return {"masks": mask_stack, "meta": all_meta}


def run_segmentation(session, config):
    """Pipeline entry point — segment every captured frame with SAM 3."""
    print("\n[SAM3] Promptable Concept Segmentation...")
    session.set_status("segment", "running")

    proc = config.get("processing", {})
    concepts = proc.get("hazard_concepts", [])
    if not concepts:
        print("[SAM3] No hazard_concepts in config — nothing to segment")
        session.set_status("segment", "failed")
        return

    weights = proc.get("sam3_weights", "sam3.pt")
    # Resolve relative to project root if not absolute
    if not Path(weights).is_absolute():
        weights = str(Path(__file__).resolve().parents[1] / weights)

    conf = proc.get("sam3_conf", 0.25)
    batch_size = proc.get("sam3_concepts_per_batch", 8)
    device = proc.get("sam_gpu", 0)

    print(f"[SAM3] Weights: {weights}")
    print(f"[SAM3] Concepts ({len(concepts)}): {', '.join(concepts[:5])}"
          f"{'...' if len(concepts) > 5 else ''}")
    print(f"[SAM3] conf={conf}  batch_size={batch_size}  device={device}")

    try:
        predictor = _load_predictor(weights, conf, device)
    except (FileNotFoundError, ImportError) as e:
        print(f"\n[SAM3] SETUP REQUIRED:\n{e}\n")
        session.set_status("segment", "failed")
        return

    m = session.manifest
    captured = [f for f in m.frames if f.captured]
    session.masks_dir.mkdir(parents=True, exist_ok=True)

    n_ok, n_fail = 0, 0
    for i, fm in enumerate(captured):
        img_path = session.frames_dir / fm.filename
        if not img_path.exists():
            continue

        print(f"  [{i+1}/{len(captured)}] {fm.filename} ...",
              end=" ", flush=True)
        try:
            result = segment_frame(predictor, img_path, concepts,
                                   batch_size=batch_size)
        except Exception as e:
            print(f"FAILED: {e}")
            n_fail += 1
            continue

        n_masks = len(result["meta"])
        out_npz = session.masks_dir / f"masks_{fm.frame_id:04d}.npz"
        out_json = session.masks_dir / f"masks_{fm.frame_id:04d}.json"

        np.savez_compressed(out_npz, masks=result["masks"])
        out_json.write_text(json.dumps({
            "frame_id": fm.frame_id,
            "filename": fm.filename,
            "pan": fm.pan, "tilt": fm.tilt,
            "masks": result["meta"],
        }, indent=2))

        concepts_found = sorted(set(mm["concept"] for mm in result["meta"]))
        print(f"OK  {n_masks} instances "
              f"({', '.join(concepts_found[:4])}"
              f"{'...' if len(concepts_found) > 4 else ''})")
        n_ok += 1

    session.set_output("masks_dir", str(session.masks_dir))
    if n_ok > 0:
        session.set_status("segment", "done")
        print(f"\n[SAM3] Done: {n_ok}/{len(captured)} frames segmented "
              f"({n_fail} failed)")
    else:
        session.set_status("segment", "failed")
        print(f"\n[SAM3] FAILED: 0/{len(captured)} frames segmented")