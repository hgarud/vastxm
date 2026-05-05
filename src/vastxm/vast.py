from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any

from vastxm._log import get_logger

log = get_logger(__name__)


class VastError(RuntimeError):
    """Raised when a vastai CLI invocation fails or returns malformed output."""


def ensure_installed() -> None:
    if shutil.which("vastai") is None:
        raise VastError(
            "The `vastai` CLI was not found on PATH. Install it with `pip install vastai` "
            "and authenticate via `vastai set api-key <key>`."
        )


def _run(args: list[str], *, raw: bool = True, check: bool = True) -> str:
    """Invoke `vastai <args>` and return its stdout as a string."""
    cmd = ["vastai", *args]
    if raw and "--raw" not in cmd:
        cmd.append("--raw")
    log.debug("running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise VastError(
            f"vastai {' '.join(args)} failed (exit {proc.returncode}):\n"
            f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        )
    return proc.stdout


def _run_json(args: list[str]) -> Any:
    out = _run(args, raw=True)
    try:
        return json.loads(out)
    except json.JSONDecodeError as e:
        raise VastError(f"vastai {' '.join(args)} did not return valid JSON: {out[:500]}") from e


# ---- Public API ----

def search_offers(query: str, *, order: str = "dph_total") -> list[dict]:
    """Run `vastai search offers '<query>' -o '<order>' --raw`. Returns list of offer dicts."""
    return _run_json(["search", "offers", query, "-o", order])


def show_instance(instance_id: int) -> dict:
    return _run_json(["show", "instance", str(instance_id)])


def show_instances() -> list[dict]:
    return _run_json(["show", "instances"])


def show_ssh_keys() -> list[dict]:
    return _run_json(["show", "ssh-keys"])


def create_ssh_key(public_key: str) -> dict:
    """Register an SSH public key (the literal key contents, not a path)."""
    return _run_json(["create", "ssh-key", public_key])


def create_instance(
    offer_id: int,
    *,
    image: str,
    disk: int,
    onstart_cmd: str,
    ssh: bool = True,
    direct: bool = True,
) -> dict:
    args = [
        "create", "instance", str(offer_id),
        "--image", image,
        "--disk", str(disk),
        "--onstart-cmd", onstart_cmd,
    ]
    if ssh:
        args.append("--ssh")
    if direct:
        args.append("--direct")
    return _run_json(args)


def destroy_instance(instance_id: int) -> None:
    _run(["destroy", "instance", str(instance_id)], raw=False)


def copy(src: str, dst: str) -> None:
    """vastai copy. Endpoints look like `local:./foo` or `<INSTANCE_ID>:/path`."""
    _run(["copy", src, dst], raw=False)


def ssh_url(instance_id: int) -> str:
    """Return the ssh:// URL for an instance, e.g. ssh://root@host.example:12345"""
    out = _run(["ssh-url", str(instance_id)], raw=False).strip()
    if not out.startswith("ssh://"):
        raise VastError(f"unexpected ssh-url output: {out!r}")
    return out
