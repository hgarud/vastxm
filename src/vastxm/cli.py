import argparse
import os
import sys
from dataclasses import fields
from pathlib import Path
from typing import NoReturn

from rich.console import Console
from rich.table import Table

from vastxm import instance, vast
from vastxm._log import get_logger
from vastxm.config import LaunchConfig, merge_config
from vastxm.ssh import SSH_COMMON_OPTS, run_remote_streaming
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
    pl.add_argument("--num-gpus", type=int)
    pl.add_argument("--max-price", type=float)
    pl.add_argument("--disk", type=int)
    pl.add_argument("--image")
    pl.add_argument("--bundle-root")
    pl.add_argument("--exclude", action="append", default=None,
                    help="extra exclude pattern (repeatable). Adds to defaults.")
    pl.add_argument("--artifact-path")
    pl.add_argument("--artifact-dest")
    pl.add_argument("--name")
    pl.add_argument("--keep", action="store_true", default=None,
                    help="leave the instance running after exit.")
    pl.add_argument("--dry-run", action="store_true", default=None)
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

    # auth
    pa = sub.add_parser("auth", help="set the vast.ai API key (one-time setup).")
    pa.add_argument("key", help="API key from https://cloud.vast.ai/manage-keys/")

    # vastai passthrough
    pv = sub.add_parser("vastai", help="forward args to the bundled vastai CLI.")
    pv.add_argument("vastai_args", nargs=argparse.REMAINDER,
                    help="arguments passed verbatim, e.g. `vastxm vastai show user`.")

    return p


_LAUNCH_OVERRIDE_FIELDS = tuple(
    f.name for f in fields(LaunchConfig) if f.name != "cmd"
)


def _cmd_launch(args: argparse.Namespace) -> int:
    overrides = {k: getattr(args, k) for k in _LAUNCH_OVERRIDE_FIELDS}
    cfg = merge_config(cmd=args.cmd, toml_path=Path(args.config_file), cli_overrides=overrides)
    return launch(cfg)


def _cmd_ls(_: argparse.Namespace) -> int:
    rows = vast.show_instances()
    if not rows:
        print("(no instances)")
        return 0
    table = Table(header_style="bold cyan", expand=False)
    table.add_column("ID", justify="right", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("GPU", no_wrap=True)
    table.add_column("$/hr", justify="right", no_wrap=True)
    table.add_column("Image")
    for r in rows:
        table.add_row(
            str(r.get("id") or "?"),
            r.get("actual_status") or "?",
            r.get("gpu_name") or "?",
            f"{r.get('dph_total', 0.0):.3f}",
            r.get("image_uuid") or r.get("image", ""),
        )
    Console().print(table)
    return 0


def _cmd_logs(args: argparse.Namespace) -> int:
    target = instance.resolve_ssh_target(args.instance_id)
    return run_remote_streaming(target, "tail -F /workspace/train.log")


def _cmd_ssh(args: argparse.Namespace) -> NoReturn:
    target = instance.resolve_ssh_target(args.instance_id)
    argv = [
        "ssh", "-p", str(target.port),
        *SSH_COMMON_OPTS,
        f"{target.user}@{target.host}",
    ]
    os.execvp(argv[0], argv)


def _cmd_stop(args: argparse.Namespace) -> int:
    vast.destroy_instance(args.instance_id)
    log.info("destroyed instance %s", args.instance_id)
    return 0


def _cmd_pull(args: argparse.Namespace) -> int:
    vast.copy(f"{args.instance_id}:{args.remote_path}", f"local:{args.local_path}")
    return 0


def _cmd_auth(args: argparse.Namespace) -> NoReturn:
    vast.exec_passthrough(["set", "api-key", args.key])


def _cmd_vastai(args: argparse.Namespace) -> NoReturn:
    vast.exec_passthrough(args.vastai_args or [])


_DISPATCH = {
    "launch": _cmd_launch,
    "ls": _cmd_ls,
    "logs": _cmd_logs,
    "ssh": _cmd_ssh,
    "stop": _cmd_stop,
    "pull": _cmd_pull,
    "auth": _cmd_auth,
    "vastai": _cmd_vastai,
}


_SKIP_AUTH = {"auth", "vastai"}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command not in _SKIP_AUTH and not getattr(args, "dry_run", False):
            vast.ensure_authenticated()
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
