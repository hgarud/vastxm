from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from vastxm._log import get_logger

log = get_logger(__name__)

_URL_RE = re.compile(r"^ssh://(?P<user>[^@]+)@(?P<host>[^:]+):(?P<port>\d+)/?$")

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

    @classmethod
    def parse(cls, url: str) -> "SshTarget":
        m = _URL_RE.match(url.strip())
        if not m:
            raise ValueError(f"unexpected ssh-url: {url!r}")
        return cls(user=m["user"], host=m["host"], port=int(m["port"]))


def _ssh_argv(target: SshTarget, remote_cmd: str) -> list[str]:
    return [
        "ssh",
        "-p", str(target.port),
        *SSH_COMMON_OPTS,
        "-o", "ServerAliveInterval=30",
        f"{target.user}@{target.host}",
        "bash", "-lc", remote_cmd,
    ]


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
