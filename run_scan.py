#!/usr/bin/env python3 -u
"""
run_scan.py — Top-level CLI for the McLaren Room Scanner.

Usage:
    # On laptop (lab WiFi):
    python run_scan.py capture  --session living_room_01
    python run_scan.py preview  --session living_room_01
    python run_scan.py transfer --session living_room_01

    # On server (ssh sbrainard@10.0.11.13):
    python run_scan.py process  --session living_room_01
    python run_scan.py process  --session living_room_01 --stage depth
    python run_scan.py process  --session living_room_01 --stage segment

    # On laptop (to view results):
    python run_scan.py pull     --session living_room_01
    python run_scan.py view     --session living_room_01

    # Utilities:
    python run_scan.py list
    python run_scan.py status   --session living_room_01
    python run_scan.py delete   --session living_room_01
    python run_scan.py setup-server
"""

import argparse
import sys
from pathlib import Path

import yaml


def load_config(config_path: str = "config/sweep.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def cmd_capture(args, config):
    from capture.ptz_sweep import PTZSweep
    from capture.session import Session

    session = Session(config["local"]["sessions_dir"], args.session)
    if not (session.root / "manifest.json").exists():
        session.create(config)
    else:
        session.load()
        print(f"[Scan] Resuming session: {args.session}")

    sweep = PTZSweep(config, session)
    ok = sweep.run(resume=True)
    if ok:
        print(f"\n✓ Capture complete. Transfer with:")
        print(f"  python run_scan.py transfer --session {args.session}")
    else:
        print(f"\n⚠  Capture incomplete. Re-run to retry failed frames.")


def cmd_preview(args, config):
    from capture.ptz_sweep import PTZSweep
    from capture.session import Session

    session = Session(config["local"]["sessions_dir"], args.session)
    if not (session.root / "manifest.json").exists():
        session.create(config)

    sweep = PTZSweep(config, session)
    n = getattr(args, "n", 5)
    sweep.preview(n_frames=n)


def cmd_transfer(args, config):
    from capture.transfer import check_server, push_code, push_session

    srv = config["server"]
    server = getattr(args, "server", None) or srv["primary"]

    if not check_server(server, srv["user"]):
        sys.exit(1)

    # Push the code first (so the server runs the same code you edit locally)
    if not getattr(args, "no_code", False):
        project_root = str(Path(__file__).parent)
        push_code(project_root, server, srv["user"],
                  srv["remote_code_dir"],
                  dry_run=getattr(args, "dry_run", False))

    # Push the session data (frames + manifest)
    session_dir = str(Path(config["local"]["sessions_dir"]) / args.session)
    ok = push_session(session_dir, server, srv["user"],
                      srv["remote_base"],
                      dry_run=getattr(args, "dry_run", False))
    if ok:
        print(f"\n✓ Transfer complete. Process on server with:")
        print(f"  ssh {srv['user']}@{server}")
        print(f"  cd {srv['remote_code_dir']} && python run_scan.py process --session {args.session}")
    else:
        print("\n✗ Transfer failed.")
        sys.exit(1)


def cmd_process(args, config):
    stage = getattr(args, "stage", "all")
    local = getattr(args, "local", False)

    # --- LOCAL EXECUTION (we're already on the server, or --local forced) ---
    if local:
        print(f"[Process] Running LOCALLY. Stage: {stage}")
        from capture.session import Session
        session = Session(config["local"]["sessions_dir"], args.session)
        session.load()

        if stage in ("all", "depth"):
            from perception.depth_pro import run_depth
            run_depth(session, config)
        if stage in ("all", "stitch"):
            from geometry.pano_stitch import run_stitch
            run_stitch(session, config)
        if stage in ("all", "segment"):
            from perception.sam3 import run_segmentation
            run_segmentation(session, config)
        if stage in ("all", "pointcloud"):
            from geometry.point_cloud import run_pointcloud
            run_pointcloud(session, config)
        if stage in ("all", "analysis"):
            from analysis.home_fast import run_analysis
            run_analysis(session, config)
        if stage in ("all", "splat") and config["processing"].get("run_splatting"):
            from splat.train_3dgs import run_splatting
            run_splatting(session, config)
        print(f"\n✓ Local processing complete.")
        return

    # --- AUTOMATED REMOTE EXECUTION (default, from laptop) ---
    # 1. push code + frames  2. run on server  3. pull results
    from capture.transfer import check_server, pull_results, push_code, push_session, run_remote
    srv = config["server"]
    server = getattr(args, "server", None) or srv["primary"]

    print(f"[Process] Automated remote pipeline → {server}")
    print("=" * 60)

    if not check_server(server, srv["user"]):
        print("\n✗ Cannot reach server. Set up SSH keys with:")
        print(f"  ssh-copy-id {srv['user']}@{server}")
        sys.exit(1)

    # Step 1: push code + session data
    if not getattr(args, "no_upload", False):
        print("\n[1/3] Uploading code + frames...")
        project_root = str(Path(__file__).parent)
        push_code(project_root, server, srv["user"], srv["remote_code_dir"])
        session_dir = str(Path(config["local"]["sessions_dir"]) / args.session)
        push_session(session_dir, server, srv["user"], srv["remote_base"])
    else:
        print("\n[1/3] Skipping upload (--no-upload)")

    # Step 2: run processing on the server (streams output live)
    print(f"\n[2/3] Processing on server (stage={stage})...")
    print("-" * 60)
    ok = run_remote(server, srv["user"], srv["remote_code_dir"],
                    f"process --session {args.session} --stage {stage} --local",
                    data_root=srv.get("data_root", "/data1"),
                    forward_env=["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"])
    print("-" * 60)
    if not ok:
        print("\n✗ Remote processing failed. See output above.")
        sys.exit(1)

    # Step 3: pull results back
    print("\n[3/3] Downloading results...")
    pull_results(args.session, server, srv["user"],
                 srv["remote_base"], config["local"]["sessions_dir"])

    print("\n" + "=" * 60)
    print(f"✓ Done! Results in sessions/{args.session}/outputs/")
    print(f"  View with: python run_scan.py view --session {args.session}")


def cmd_pull(args, config):
    from capture.transfer import check_server, pull_results

    srv = config["server"]
    server = getattr(args, "server", None) or srv["primary"]

    if not check_server(server, srv["user"]):
        sys.exit(1)

    ok = pull_results(args.session, server, srv["user"],
                      srv["remote_base"],
                      config["local"]["sessions_dir"],
                      dry_run=getattr(args, "dry_run", False))
    if ok:
        print(f"\n✓ Results pulled. View with:")
        print(f"  python run_scan.py view --session {args.session}")
    else:
        print("\n✗ Pull failed.")
        sys.exit(1)


def cmd_view(args, config):
    from capture.session import Session
    session = Session(config["local"]["sessions_dir"], args.session)
    session.load()

    from viewer.app import launch_viewer
    launch_viewer(session, config)


def cmd_list(args, config):
    from capture.session import Session, list_sessions
    sessions = list_sessions(config["local"]["sessions_dir"])
    if not sessions:
        print("No sessions found.")
        return
    print(f"{'Session':<30} {'Frames':>8} {'Status'}")
    print("-" * 60)
    for name in sessions:
        try:
            s = Session(config["local"]["sessions_dir"], name)
            m = s.load()
            n_cap = len(s.captured_frames())
            n_tot = len(m.frames)
            stages = [v for v in m.status.values() if v == "done"]
            print(f"{name:<30} {n_cap:>4}/{n_tot:<3}  "
                  f"{len(stages)}/{len(m.status)} stages done")
        except Exception:
            print(f"{name:<30} (error reading manifest)")


def cmd_status(args, config):
    from capture.session import Session
    session = Session(config["local"]["sessions_dir"], args.session)
    session.load()
    print(session.summary())


def cmd_delete(args, config):
    from capture.session import Session
    session = Session(config["local"]["sessions_dir"], args.session)
    confirm = input(f"Delete session '{args.session}'? [y/N] ")
    if confirm.lower() == "y":
        session.delete()
    else:
        print("Cancelled.")


def cmd_setup_server(args, config):
    from capture.transfer import check_server, setup_server_env
    srv = config["server"]
    server = getattr(args, "server", None) or srv["primary"]
    stage = getattr(args, "stage", "stitch")
    data_root = getattr(args, "data_root", None) or srv.get("data_root", "/data1")

    if not check_server(server, srv["user"]):
        print(f"\n✗ Cannot reach server. Run: ssh-copy-id {srv['user']}@{server}")
        sys.exit(1)

    ok = setup_server_env(
        server, srv["user"], srv["remote_base"], srv["remote_code_dir"],
        data_root=data_root, stage=stage,
    )
    if ok:
        print(f"\n✓ Server ready for stage '{stage}'. "
              f"Now run: python run_scan.py process --session NAME --stage {stage}")
    else:
        print("\n✗ Setup failed — see output above.")
        sys.exit(1)


# =============================================================================
#  CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="McLaren Room Scanner — Fall Risk Assessment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--config", default="config/sweep.yaml",
                    help="Path to config YAML (default: config/sweep.yaml)")

    sub = ap.add_subparsers(dest="command", required=True)

    # capture
    p = sub.add_parser("capture", help="Run PTZ sweep and capture frames")
    p.add_argument("--session", required=True)
    p.add_argument("--no-resume", action="store_true",
                   help="Start fresh even if frames already captured")

    # preview
    p = sub.add_parser("preview", help="Quick preview sweep (no full capture)")
    p.add_argument("--session", required=True)
    p.add_argument("--n", type=int, default=5, help="Number of preview frames")

    # transfer
    p = sub.add_parser("transfer", help="Push code + session to server")
    p.add_argument("--session", required=True)
    p.add_argument("--server", default=None, help="Override server IP")
    p.add_argument("--no-code", action="store_true",
                   help="Skip pushing code (only push session data)")
    p.add_argument("--dry-run", action="store_true")

    # process
    p = sub.add_parser("process",
                       help="Process a session (auto: push->server->pull)")
    p.add_argument("--session", required=True)
    p.add_argument("--stage", default="all",
                   choices=["all", "depth", "stitch", "segment",
                             "pointcloud", "analysis", "splat"],
                   help="Which stage to run (default: all)")
    p.add_argument("--local", action="store_true",
                   help="Run on THIS machine (used internally on the server)")
    p.add_argument("--no-upload", action="store_true",
                   help="Skip uploading code+frames (use what is on server)")
    p.add_argument("--server", default=None, help="Override server IP")

    # pull
    p = sub.add_parser("pull", help="Pull results back from server")
    p.add_argument("--session", required=True)
    p.add_argument("--server", default=None)
    p.add_argument("--dry-run", action="store_true")

    # view
    p = sub.add_parser("view", help="Launch 3D viewer")
    p.add_argument("--session", required=True)

    # list
    sub.add_parser("list", help="List all sessions")

    # status
    p = sub.add_parser("status", help="Show session status")
    p.add_argument("--session", required=True)

    # delete
    p = sub.add_parser("delete", help="Delete a session")
    p.add_argument("--session", required=True)

    # setup-server
    p = sub.add_parser("setup-server", help="One-time server environment setup")
    p.add_argument("--server", default=None)
    p.add_argument("--stage", default="stitch",
                   choices=["stitch", "depth", "segment", "full"],
                   help="Install deps for this stage only (default: stitch)")
    p.add_argument("--data-root", default=None,
                   help="Server partition for installs+data (default: /data1)")

    args = ap.parse_args()
    config = load_config(args.config)

    dispatch = {
        "capture":      cmd_capture,
        "preview":      cmd_preview,
        "transfer":     cmd_transfer,
        "process":      cmd_process,
        "pull":         cmd_pull,
        "view":         cmd_view,
        "list":         cmd_list,
        "status":       cmd_status,
        "delete":       cmd_delete,
        "setup-server": cmd_setup_server,
    }
    dispatch[args.command](args, config)


if __name__ == "__main__":
    main()