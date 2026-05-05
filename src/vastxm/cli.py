from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from vastxm import vast
from vastxm._log import get_logger
from vastxm.config import merge_config
from vastxm.ssh import SshTarget, run_remote_streaming
from vastxm.vast import VastError
from vastxm.workflow import launch

log = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vastxm", description="Launch GPU training jobs on vast.ai.")
    sub = p.add_subparsers(dest="command", required=True)

    # launch
    pl = sub.add_parser("launch", help="provision an instance, ship code, run a command, then destroy.")
    pl.add_argument("--cmd", required=True, help="full command to run on the instance.")
    pl.add_argument("--gpu")
    pl.add_argument("--num-gpus", type=int, dest="num_gpus")
    pl.add_argument("--max-price", type=float, dest="max_price")
    pl.add_argument("--disk", type=int)
    pl.add_argument("--image")
    pl.add_argument("--bundle-root", dest="bundle_root")
    pl.add_argument("--exclude", action="append", default=None,
                    help="extra exclude pattern (repeatable). Adds to defaults.")
    pl.add_argument("--artifact-path", dest="artifact_path")
    pl.add_argument("--artifact-dest", dest="artifact_dest")
    pl.add_argument("--name")
    pl.add_argument("--keep", action="store_true", default=None,
                    help="leave the instance running after exit.")
    pl.add_argument("--dry-run", dest="dry_run", action="store_true", default=None)
    pl.add_argument("--config-file", default="vastxm.toml",
                    help="path to a vastxm.toml with [defaults]; missing file is fine.")

    # ls
    sub.add_parser("ls", help="list active vast.ai instances.")

    # logs <id>
    pg = sub.add_parser("logs", help="tail the remote train.log of a running instance.")
    pg.add_argument("instance_id", type=int)

    # ssh <id>
    ps = sub.add_parser("ssh", help="open an interactive SSH shell on the instance.")
    ps.add_argument("instance_id", type=int)

    # stop <id>
    pst = sub.add_parser("stop", help="destroy an instance.")
    pst.add_argument("instance_id", type=int)

    # pull
    pp = sub.add_parser("pull", help="vastai copy from instance to local.")
    pp.add_argument("instance_id", type=int)
    pp.add_argument("remote_path")
    pp.add_argument("local_path")

    return p


def _cmd_launch(args: argparse.Namespace) -> int:
    overrides = {
        k: getattr(args, k)
        for k in ("gpu", "num_gpus", "max_price", "disk", "image",
                  "bundle_root", "exclude", "artifact_path", "artifact_dest",
                  "name", "keep", "dry_run")
    }
    if isinstance(overrides.get("exclude"), list):
        overrides["exclude"] = tuple(overrides["exclude"])
    cfg = merge_config(cmd=args.cmd, toml_path=Path(args.config_file), cli_overrides=overrides)
    return launch(cfg)


def _cmd_ls(_: argparse.Namespace) -> int:
    rows = vast.show_instances()
    if not rows:
        print("(no instances)")
        return 0
    print(f"{'ID':>10}  {'STATUS':<10}  {'GPU':<14}  {'$/hr':>6}  IMAGE")
    for r in rows:
        print(f"{r.get('id'):>10}  "
              f"{(r.get('actual_status') or '?'):<10}  "
              f"{(r.get('gpu_name') or '?'):<14}  "
              f"{r.get('dph_total', 0.0):>6.3f}  "
              f"{r.get('image_uuid') or r.get('image', '')}")
    return 0


def _cmd_logs(args: argparse.Namespace) -> int:
    target = SshTarget.parse(vast.ssh_url(args.instance_id))
    return run_remote_streaming(target, "tail -F /workspace/train.log")


def _cmd_ssh(args: argparse.Namespace) -> int:
    target = SshTarget.parse(vast.ssh_url(args.instance_id))
    argv = [
        "ssh", "-p", str(target.port),
        "-o", "StrictHostKeyChecking=accept-new",
        f"{target.user}@{target.host}",
    ]
    os.execvp(argv[0], argv)  # never returns
    return 0  # unreachable


def _cmd_stop(args: argparse.Namespace) -> int:
    vast.destroy_instance(args.instance_id)
    log.info("destroyed instance %s", args.instance_id)
    return 0


def _cmd_pull(args: argparse.Namespace) -> int:
    vast.copy(f"{args.instance_id}:{args.remote_path}", f"local:{args.local_path}")
    return 0


_DISPATCH = {
    "launch": _cmd_launch,
    "ls": _cmd_ls,
    "logs": _cmd_logs,
    "ssh": _cmd_ssh,
    "stop": _cmd_stop,
    "pull": _cmd_pull,
}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return _DISPATCH[args.command](args)
    except VastError as e:
        log.error("vastai error: %s", e)
        return 2
    except (RuntimeError, FileNotFoundError, TimeoutError) as e:
        log.error("%s", e)
        return 1
    except KeyboardInterrupt:
        log.warning("interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
