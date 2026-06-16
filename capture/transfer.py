"""
transfer.py — Sync session data between laptop and A100 server.

Uses rsync over SSH for efficient incremental transfers.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path


def rsync(src: str, dst: str, delete: bool = False,
          exclude: list = None, dry_run: bool = False,
          verbose: bool = True) -> bool:
    """
    Run rsync from src to dst.
    Supports local paths and remote paths like user@host:/path.
    """
    cmd = ["rsync", "-avz", "--progress"]
    if delete:
        cmd.append("--delete")
    for pat in (exclude or []):
        cmd += ["--exclude", pat]
    if dry_run:
        cmd.append("--dry-run")
    cmd += [src, dst]

    if verbose:
        print(f"[Transfer] {' '.join(cmd)}")

    result = subprocess.run(cmd)
    return result.returncode == 0


def push_code(project_root: str, server: str, user: str,
              remote_code_dir: str, dry_run: bool = False) -> bool:
    """
    Push code modules to the server so processing uses the same code you
    edit locally. Excludes sessions/, caches, git, and model weights.
    """
    src = str(Path(project_root)) + "/"
    dst = f"{user}@{server}:{remote_code_dir}/"
    print(f"\n[Transfer] Pushing code to {server}...")
    return rsync(src, dst, dry_run=dry_run,
                 exclude=["sessions/", "*.pyc", "__pycache__/",
                          ".git/", ".vscode/", "*.engine", "*.pt", "*.onnx"])


def push_session(session_dir: str, server: str, user: str,
                 remote_base: str, dry_run: bool = False) -> bool:
    """
    Push a local session directory to the server.

    session_dir: local path e.g. "sessions/my_room_01"
    server:      server IP e.g. "10.0.11.13"
    user:        SSH user e.g. "sbrainard"
    remote_base: remote parent dir e.g. "/home/sbrainard/mclaren_room_scanner/sessions"
    """
    # Ensure the remote sessions directory exists before rsync tries to
    # create a subdirectory inside it.
    if not dry_run:
        subprocess.run(
            ["ssh", f"{user}@{server}", f"mkdir -p {remote_base}"],
            check=False,
        )

    src = str(Path(session_dir)) + "/"
    dst = f"{user}@{server}:{remote_base}/{Path(session_dir).name}/"
    print(f"\n[Transfer] Pushing session to {server}...")
    return rsync(src, dst, dry_run=dry_run,
                 exclude=["*.pyc", "__pycache__"])


def pull_results(session_name: str, server: str, user: str,
                 remote_base: str, local_sessions: str,
                 dry_run: bool = False) -> bool:
    """
    Pull processed results back from the server to the laptop.
    Only syncs the outputs/ subdirectory (not frames/depth/masks
    which are large and already on the laptop).
    """
    remote_outputs = f"{user}@{server}:{remote_base}/{session_name}/outputs/"
    local_outputs  = str(Path(local_sessions) / session_name / "outputs") + "/"
    Path(local_outputs).mkdir(parents=True, exist_ok=True)

    print(f"\n[Transfer] Pulling results from {server}...")
    ok1 = rsync(remote_outputs, local_outputs, dry_run=dry_run)

    # Also sync the manifest to get updated processing status
    remote_manifest = f"{user}@{server}:{remote_base}/{session_name}/manifest.json"
    local_manifest  = str(Path(local_sessions) / session_name / "manifest.json")
    ok2 = rsync(remote_manifest, local_manifest, dry_run=dry_run)

    return ok1 and ok2


def check_server(server: str, user: str) -> bool:
    """Quick SSH connectivity check."""
    result = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
         f"{user}@{server}", "echo ok"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(f"[Transfer] ✓ Server {server} reachable as {user}")
        return True
    else:
        print(f"[Transfer] ✗ Cannot reach {server} as {user}")
        print(f"  Make sure you have SSH key auth set up:")
        print(f"  ssh-copy-id {user}@{server}")
        return False


def run_remote(server: str, user: str, remote_code_dir: str,
               command: str, stream: bool = True,
               data_root: str = "/data1",
               forward_env: list = None) -> bool:
    """
    Execute a command on the server over SSH and stream its output back
    to the local terminal live. Uses the SSH key (no password prompt).

    Also redirects HF/torch model caches and temp to the data partition
    so model downloads never fill the (small) home directory.

    command: the run_scan.py subcommand to run remotely, e.g.
             "process --session test --stage stitch --local"
    """
    import sys
    env_prefix = (
        f"export TMPDIR={data_root}/tmp && "
        f"export PIP_CACHE_DIR={data_root}/.pip_cache && "
        f"export HF_HOME={data_root}/.hf_cache && "
        f"export TORCH_HOME={data_root}/.torch_cache && "
        f"export XDG_CACHE_HOME={data_root}/.cache && "
        f"mkdir -p {data_root}/tmp {data_root}/.hf_cache "
        f"{data_root}/.torch_cache {data_root}/.cache && "
    )

    # Forward selected env vars from the LOCAL environment to the server
    # (SSH doesn't pass env vars by default). Used for HF_TOKEN so SAM 3
    # can auto-download its gated weights.
    for var in (forward_env or []):
        val = os.environ.get(var)
        if val:
            env_prefix += f"export {var}={shlex.quote(val)} && "
    remote_cmd = (
        f"{env_prefix}"
        f"cd {remote_code_dir} && "
        f"python3 -u run_scan.py {command}"
    )
    ssh_cmd = [
        "ssh", "-o", "ConnectTimeout=10",
        f"{user}@{server}",
        remote_cmd,
    ]
    print(f"[Remote] Running on {server}:")
    print(f"  {remote_cmd}\n")

    if not stream:
        result = subprocess.run(ssh_cmd)
        return result.returncode == 0

    # Stream stdout/stderr live so it feels like running locally
    proc = subprocess.Popen(
        ssh_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, text=True,
    )
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
    except KeyboardInterrupt:
        proc.terminate()
        print("\n[Remote] Interrupted — process on server may still be running")
        return False
    proc.wait()
    return proc.returncode == 0


def setup_server_env(server: str, user: str, remote_base: str,
                     remote_code_dir: str, data_root: str = "/data1",
                     stage: str = "stitch"):
    """
    One-time server setup. Creates all directories on the chosen data
    partition, redirects pip cache + temp there (home dir is often full),
    and installs ONLY the dependencies needed for the requested stage so
    we do not waste disk on models we are not using yet.

    stage options:
      "stitch"  -> numpy, opencv, pyyaml         (tiny, no GPU libs)
      "depth"   -> + torch/cuda, depth models
      "segment" -> + ultralytics (SAM 3)
      "full"    -> everything (torch, ultralytics, transformers, open3d)
    """
    # pip package sets per stage (cumulative)
    base_pkgs = "numpy PyYAML opencv-python-headless scipy"
    torch_cmd = ("pip install torch torchvision "
                 "--index-url https://download.pytorch.org/whl/cu121 -q")
    # SAM 3 requires ultralytics' CLIP fork, not a conflicting PyPI package
    # also named 'clip' (causes "TypeError: 'SimpleTokenizer' object is not
    # callable" or "ModuleNotFoundError: No module named 'clip'").
    #
    # We: (1) remove any clip-named packages that might shadow it,
    #     (2) force-reinstall the ultralytics fork with --user --no-cache-dir
    #         so a stale/cached broken build can't be silently reused,
    #     (3) VERIFY `import clip` works right here — if this fails, setup
    #         aborts with a clear error instead of failing 32x during
    #         segmentation (each failure costs ~5-10s of wasted AutoUpdate).
    clip_fix = (
        "pip uninstall -y clip openai-clip ultralytics-clip -q "
        "2>/dev/null; "
        "pip install --user ftfy regex -q; "
        "echo '--- Installing CLIP (ultralytics fork) ---'; "
        "pip install --user --force-reinstall --no-cache-dir "
        "git+https://github.com/ultralytics/CLIP.git || true; "
        "pip show clip 2>&1 | head -5; "
        "python3 -c 'import clip; print(\"[clip OK]\", clip.__file__)' "
        "|| ( echo '--- ultralytics fork failed, trying openai/CLIP ---'; "
        "pip install --user --force-reinstall --no-cache-dir "
        "git+https://github.com/openai/CLIP.git; "
        "python3 -c 'import clip; print(\"[clip OK]\", clip.__file__)' )"
    )

    # System matplotlib (imported transitively by ultralytics) is often
    # compiled against NumPy 1.x. If pip pulls in NumPy 2.x as a dependency
    # of torch/ultralytics, importing matplotlib crashes with:
    #   AttributeError: _ARRAY_API not found
    # Fix: re-pin numpy<2 as the LAST step so it wins regardless of what
    # earlier installs requested.
    numpy_pin = 'pip install "numpy<2" -q'

    stage_pkgs = {
        "stitch":  [f"pip install {base_pkgs} -q", numpy_pin],
        "depth":   [f"pip install {base_pkgs} -q", torch_cmd,
                    "pip install timm pillow -q", numpy_pin],
        "segment": [f"pip install {base_pkgs} -q", torch_cmd,
                    "pip install -U ultralytics huggingface_hub -q", clip_fix,
                    numpy_pin],
        "full":    [f"pip install {base_pkgs} open3d -q", torch_cmd,
                    "pip install -U ultralytics transformers accelerate timm "
                    "huggingface_hub -q", clip_fix, numpy_pin],
    }
    installs = stage_pkgs.get(stage, stage_pkgs["stitch"])

    pip_cache = f"{data_root}/.pip_cache"
    tmp_dir   = f"{data_root}/tmp"

    # Build one script that runs on the server
    script = f"""
set -e
echo '=== McLaren server setup (stage: {stage}) ==='

# 1. Verify the data partition exists and has space
if [ ! -d "{data_root}" ]; then
    echo "ERROR: {data_root} does not exist on this server."
    echo "Available partitions:"
    df -h
    exit 1
fi
echo "--- Disk space on {data_root}: ---"
df -h {data_root}

# 2. Create all project directories on the data partition
mkdir -p {remote_code_dir}
mkdir -p {remote_base}
mkdir -p {pip_cache}
mkdir -p {tmp_dir}
echo "Created project dirs under {data_root}"

# 3. Redirect pip cache + temp to the data partition (home may be full)
export PIP_CACHE_DIR={pip_cache}
export TMPDIR={tmp_dir}
export TEMP={tmp_dir}
export TMP={tmp_dir}

# 4. Clear any old pip cache in home to reclaim space
pip cache purge 2>/dev/null || true

# 5. Show which python/pip we are using
echo "--- Python environment: ---"
which python3
python3 --version

# 6. Install stage dependencies
echo "--- Installing dependencies for stage: {stage} ---"
"""
    for cmd in installs:
        # ensure each install also uses the redirected cache/tmp
        script += (f'PIP_CACHE_DIR={pip_cache} TMPDIR={tmp_dir} {cmd}\n')

    script += f"""
echo ""
echo "=== Setup complete for stage: {stage} ==="
echo "Code dir:    {remote_code_dir}"
echo "Sessions:    {remote_base}"
echo "Pip cache:   {pip_cache}"
df -h {data_root}
"""

    result = subprocess.run(
        ["ssh", f"{user}@{server}", "bash -s"],
        input=script, text=True,
    )
    return result.returncode == 0