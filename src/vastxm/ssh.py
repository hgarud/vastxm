import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from vastxm._log import get_logger

log = get_logger(__name__)

# vast.ai recycles host:port across rentals, so any host key we cached from a
# previous instance is stale. Don't read or write known_hosts for these probes —
# it just trips StrictHostKeyChecking and breaks BatchMode probes silently.
SSH_COMMON_OPTS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "GlobalKnownHostsFile=/dev/null",
    "-o", "LogLevel=ERROR",
]


@dataclass(frozen=True)
class SshTarget:
    user: str
    host: str
    port: int
    # Pinning the private key (with IdentitiesOnly) avoids two pitfalls:
    # (a) sshd's MaxAuthTries=6 cutting us off when an agent has many keys
    # before the registered one, (b) ambiguity about which default identity
    # OpenSSH picks across environments. Set when the workflow has resolved
    # the local key matching the user's vast.ai-registered pubkey.
    identity: Path | None = None


def _identity_opts(target: SshTarget) -> list[str]:
    if target.identity is None:
        return []
    return ["-i", str(target.identity), "-o", "IdentitiesOnly=yes"]


def probe(target: SshTarget, remote_cmd: str, *, connect_timeout: int = 5) -> tuple[int, str]:
    """Run a one-shot non-interactive ssh probe. Returns (returncode, stderr)."""
    argv = [
        "ssh", "-p", str(target.port),
        *SSH_COMMON_OPTS,
        *_identity_opts(target),
        "-o", f"ConnectTimeout={connect_timeout}",
        "-o", "BatchMode=yes",
        f"{target.user}@{target.host}",
        remote_cmd,
    ]
    proc = subprocess.run(argv, capture_output=True, text=True)
    return proc.returncode, proc.stderr.strip()


def _ssh_argv(target: SshTarget, remote_cmd: str) -> list[str]:
    return [
        "ssh",
        "-p", str(target.port),
        *SSH_COMMON_OPTS,
        *_identity_opts(target),
        "-o", "ServerAliveInterval=30",
        f"{target.user}@{target.host}",
        "bash", "-lc", remote_cmd,
    ]


def scp_upload(target: SshTarget, local_path: Path, remote_path: str) -> None:
    """Copy a local file to remote_path via scp, reusing the target's pinned identity.

    We avoid `vastai copy` because it rsyncs to a host-level account
    (vastai_kaalia@<host>:22) whose auth isn't tied to the SSH key vast.ai
    registers inside the *container*. The direct SSH target already has the
    right user, port, and key — scp through that.
    """
    argv = [
        "scp",
        "-P", str(target.port),
        *SSH_COMMON_OPTS,
        *_identity_opts(target),
        str(local_path),
        f"{target.user}@{target.host}:{remote_path}",
    ]
    proc = subprocess.run(argv, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"scp upload {local_path} -> {target.host}:{remote_path} failed "
            f"(rc={proc.returncode}): {proc.stderr.strip()}"
        )


def scp_download(target: SshTarget, remote_path: str, local_path: Path) -> None:
    """Recursively copy remote_path to local_path via scp (works for files too)."""
    argv = [
        "scp", "-r",
        "-P", str(target.port),
        *SSH_COMMON_OPTS,
        *_identity_opts(target),
        f"{target.user}@{target.host}:{remote_path}",
        str(local_path),
    ]
    proc = subprocess.run(argv, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"scp download {target.host}:{remote_path} -> {local_path} failed "
            f"(rc={proc.returncode}): {proc.stderr.strip()}"
        )


def run_remote_streaming(
    target: SshTarget,
    remote_cmd: str,
    *,
    log_file: Path | None = None,
) -> int:
    """Run `remote_cmd` over SSH, streaming output to stdout and (optionally) to log_file.
    Returns the remote exit code. Raises on local SSH plumbing errors only."""
    argv = _ssh_argv(target, remote_cmd)
    log.info("ssh %s@%s:%s — running remote command", target.user, target.host, target.port)
    log.debug("remote_cmd:\n%s", remote_cmd)

    log_handle = None
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_file.open("w", buffering=1)

    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if log_handle is not None:
                log_handle.write(line)
        rc = proc.wait()
    finally:
        if log_handle is not None:
            log_handle.close()
    return rc
