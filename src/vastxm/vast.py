from __future__ import annotations

import getpass
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, NoReturn

from vastxm._log import get_logger

log = get_logger(__name__)


class VastError(RuntimeError):
    """Raised when a vastai CLI invocation fails or returns malformed output."""


def _vastai_executable() -> str | None:
    # vastai is a declared dependency, so it lives next to our own python in the
    # tool venv. Prefer that bundled copy over whatever PATH happens to expose,
    # since `uv tool install` doesn't link dependency scripts into ~/.local/bin.
    bundled = Path(sys.executable).parent / "vastai"
    if bundled.exists():
        return str(bundled)
    return shutil.which("vastai")


def ensure_installed() -> None:
    if _vastai_executable() is None:
        raise VastError(
            "The bundled `vastai` CLI is missing from the vastxm tool venv. "
            "Reinstall vastxm with `uv tool install --reinstall --editable .` "
            "from the vastxm repo, then authenticate via `vastxm vastai set api-key <key>` "
            "(or call the bundled binary directly)."
        )


def _api_key_path() -> Path:
    config_dir = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(config_dir) / "vastai" / "vast_api_key"


def _has_api_key() -> bool:
    if os.environ.get("VAST_API_KEY"):
        return True
    if _api_key_path().exists():
        return True
    if (Path.home() / ".vast_api_key").exists():  # legacy
        return True
    return False


def ensure_authenticated() -> None:
    """If no key is configured, prompt for one (TTY) or fail with guidance (non-TTY)."""
    if _has_api_key():
        return
    if not sys.stdin.isatty():
        raise VastError(
            "vast.ai API key not configured. Set VAST_API_KEY or run "
            "`vastxm auth <key>` (key from https://cloud.vast.ai/manage-keys/)."
        )
    print("No vast.ai API key found.", file=sys.stderr)
    print("Get one from https://cloud.vast.ai/manage-keys/", file=sys.stderr)
    try:
        key = getpass.getpass("Paste API key (input hidden): ").strip()
    except EOFError:
        raise VastError("aborted: no API key provided.") from None
    if not key:
        raise VastError("aborted: empty API key.")
    exe = _vastai_executable()
    if exe is None:
        ensure_installed()  # raises
    proc = subprocess.run([exe, "set", "api-key", key], capture_output=True, text=True)
    if proc.returncode != 0:
        raise VastError(
            f"failed to store API key (exit {proc.returncode}):\n"
            f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        )
    print(f"API key saved to {_api_key_path()}.", file=sys.stderr)


def _run(args: list[str], *, raw: bool = True, check: bool = True) -> str:
    """Invoke `vastai <args>` and return its stdout as a string."""
    exe = _vastai_executable()
    if exe is None:
        ensure_installed()  # raises
    cmd = [exe, *args]
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


def exec_passthrough(args: list[str]) -> NoReturn:
    """Replace this process with the bundled vastai, forwarding args verbatim."""
    exe = _vastai_executable()
    if exe is None:
        ensure_installed()  # raises
    os.execv(exe, [exe, *args])


def ssh_url(instance_id: int) -> str:
    """Return the ssh:// URL for an instance, e.g. ssh://root@host.example:12345"""
    out = _run(["ssh-url", str(instance_id)], raw=False).strip()
    if not out.startswith("ssh://"):
        raise VastError(f"unexpected ssh-url output: {out!r}")
    return out
