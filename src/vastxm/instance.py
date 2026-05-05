from __future__ import annotations

import time
from pathlib import Path

from vastxm import vast
from vastxm._log import get_logger
from vastxm.config import LaunchConfig

log = get_logger(__name__)

TERMINAL_BAD = {"exited", "unknown", "offline"}
DEFAULT_PUBKEY_PATHS = (
    Path.home() / ".ssh" / "id_ed25519.pub",
    Path.home() / ".ssh" / "id_rsa.pub",
)


def build_offer_query(cfg: LaunchConfig) -> str:
    """Construct the vast `search offers` query string from the LaunchConfig."""
    parts = [
        f"gpu_name={cfg.gpu}",
        f"num_gpus={cfg.num_gpus}",
        f"dph_total<={cfg.max_price}",
        f"disk_space>={cfg.disk}",
        "verified=true",
        "rentable=true",
        "direct_port_count>=1",
    ]
    return " ".join(parts)


def pick_offer(cfg: LaunchConfig) -> int:
    """Search and return the cheapest matching offer's id. Raises if none match."""
    query = build_offer_query(cfg)
    log.info("searching offers: %s", query)
    offers = vast.search_offers(query, order="dph_total")
    if not offers:
        raise RuntimeError(
            f"No offers matched: {query}. Try raising --max-price or relaxing --gpu/--num-gpus."
        )
    o = offers[0]
    log.info(
        "selected offer %s: %sx %s @ $%.3f/hr (host %s)",
        o.get("id"), o.get("num_gpus"), o.get("gpu_name"),
        o.get("dph_total"), o.get("host_id"),
    )
    return int(o["id"])


def wait_for_running(instance_id: int, *, timeout_s: int = 900, poll_s: int = 10) -> dict:
    """Poll show_instance until status == 'running'. Raise on terminal failures or timeout."""
    deadline = time.time() + timeout_s
    last_status = None
    while time.time() < deadline:
        info = vast.show_instance(instance_id)
        status = info.get("actual_status") or info.get("status_msg") or info.get("intended_status")
        if status != last_status:
            log.info("instance %s: %s", instance_id, status)
            last_status = status
        if status == "running":
            return info
        if status in TERMINAL_BAD:
            raise RuntimeError(f"instance {instance_id} reached terminal status {status!r}; aborting")
        time.sleep(poll_s)
    raise TimeoutError(
        f"instance {instance_id} did not reach 'running' within {timeout_s}s (last status: {last_status})"
    )


def ensure_ssh_key() -> None:
    """If the user has no ssh keys registered with vast, upload their local ed25519 / rsa pubkey."""
    keys = vast.show_ssh_keys()
    if keys:
        log.debug("vast already has %d ssh key(s) registered", len(keys))
        return
    for p in DEFAULT_PUBKEY_PATHS:
        if p.exists():
            log.info("registering local ssh key with vast: %s", p)
            vast.create_ssh_key(p.read_text().strip())
            return
    raise RuntimeError(
        "No SSH key registered with vast.ai and no ~/.ssh/id_ed25519.pub or id_rsa.pub found locally. "
        "Generate one with `ssh-keygen -t ed25519` or pre-register via `vastai create ssh-key`."
    )
