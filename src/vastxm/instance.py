import socket
import time
from pathlib import Path

from vastxm import ssh, vast
from vastxm._log import get_logger
from vastxm.config import LaunchConfig
from vastxm.select import choose_offer
from vastxm.ssh import SshTarget

log = get_logger(__name__)

TERMINAL_BAD = {"exited", "unknown", "offline"}
DEFAULT_PUBKEY_PATHS = (
    Path.home() / ".ssh" / "id_ed25519.pub",
    Path.home() / ".ssh" / "id_rsa.pub",
)


# vast.ai's `gpu_name` field stores variant-suffixed values (e.g. "A100 SXM4",
# "H100 SXM"), not bare family names. The search query language uses underscores
# instead of spaces, so the *queryable* tokens are A100_SXM4, H100_SXM, etc.
# When a user passes a bare family name like "A100", expand it to a set query
# covering the known variants — exact-matching "A100" returns zero offers.
_GPU_FAMILY_VARIANTS: dict[str, tuple[str, ...]] = {
    "A100": ("A100_SXM4", "A100_PCIE", "A100X"),
    "H100": ("H100_SXM", "H100_PCIE", "H100_NVL"),
    "H200": ("H200_SXM", "H200"),
    "B200": ("B200_SXM", "B200"),
    "V100": ("Tesla_V100", "V100_SXM2", "V100_PCIE"),
}


def _gpu_filter(gpu: str) -> str:
    """Build the vast.ai search filter for a given gpu spec.

    Bare family names ("A100", "H100") expand to `gpu_name in [variant,...]`.
    Anything else (e.g. "A100_SXM4", "RTX_5090") is exact-matched.
    """
    variants = _GPU_FAMILY_VARIANTS.get(gpu)
    if variants:
        log.info("expanding gpu family %s to variants %s", gpu, list(variants))
        return f"gpu_name in [{','.join(variants)}]"
    return f"gpu_name={gpu}"


def build_offer_query(cfg: LaunchConfig) -> str:
    """Construct the vast `search offers` query string from the LaunchConfig."""
    parts = [
        _gpu_filter(cfg.gpu),
        f"num_gpus={cfg.num_gpus}",
        f"dph_total<={cfg.max_price}",
        f"disk_space>={cfg.disk}",
        "verified=true",
        "rentable=true",
        "direct_port_count>=1",
    ]
    return " ".join(parts)


def pick_offer(cfg: LaunchConfig) -> dict:
    """Search offers, prompt the user to choose one, and return the offer dict."""
    query = build_offer_query(cfg)
    log.info("searching offers: %s", query)
    offers = vast.search_offers(query, order="dph_total")
    if not offers:
        raise RuntimeError(
            f"No offers matched: {query}. Try raising --max-price or relaxing --gpu/--num-gpus."
        )
    o = choose_offer(offers)
    log.info(
        "selected offer %s: %sx %s @ $%.3f/hr (host %s, cuda %s)",
        o.get("id"), o.get("num_gpus"), o.get("gpu_name"),
        o.get("dph_total"), o.get("host_id"), o.get("cuda_max_good"),
    )
    return o


# vastai/base-image -auto tags (Docker Hub vastai/base-image), sorted ascending.
# Source: https://hub.docker.com/r/vastai/base-image/tags — only -auto tags are
# listed here; they correspond to host driver compatibility per CUDA version.
_VASTAI_AUTO_TAGS: tuple[tuple[tuple[int, int], str], ...] = (
    ((11, 8), "cuda-11.8.0-auto"),
    ((12, 1), "cuda-12.1.1-auto"),
    ((12, 4), "cuda-12.4.1-auto"),
    ((12, 6), "cuda-12.6.3-auto"),
    ((12, 8), "cuda-12.8.1-auto"),
    ((12, 9), "cuda-12.9.1-auto"),
    ((13, 0), "cuda-13.0.2-auto"),
    ((13, 1), "cuda-13.1.1-auto"),
    ((13, 2), "cuda-13.2.0-auto"),
)
_AUTO_SENTINEL = "vastai/base-image:auto"
_AUTO_FALLBACK = "vastai/base-image:cuda-12.4.1-auto"


def _parse_cuda(v) -> tuple[int, int] | None:
    """Parse a vast.ai cuda_max_good value (e.g. 12.8, '12.8', 13.0) to (major, minor)."""
    if v is None:
        return None
    try:
        s = str(v).strip()
        major_s, _, rest = s.partition(".")
        minor_s = rest.split(".")[0] if rest else "0"
        return (int(major_s), int(minor_s))
    except (ValueError, AttributeError):
        return None


def resolve_image(image: str, offer: dict) -> str:
    """If `image` is the auto sentinel, pick the right vastai/base-image tag for the offer's CUDA."""
    if image != _AUTO_SENTINEL:
        return image
    host_cuda = _parse_cuda(offer.get("cuda_max_good"))
    if host_cuda is None:
        log.warning(
            "offer %s has no usable cuda_max_good (%r); falling back to %s",
            offer.get("id"), offer.get("cuda_max_good"), _AUTO_FALLBACK,
        )
        return _AUTO_FALLBACK
    chosen: str | None = None
    for tag_cuda, tag in _VASTAI_AUTO_TAGS:
        if tag_cuda <= host_cuda:
            chosen = tag
        else:
            break
    if chosen is None:
        log.warning(
            "host cuda %s is below all known vastai/base-image -auto tags; using lowest (%s)",
            host_cuda, _VASTAI_AUTO_TAGS[0][1],
        )
        chosen = _VASTAI_AUTO_TAGS[0][1]
    resolved = f"vastai/base-image:{chosen}"
    log.info("auto-selected image %s for host cuda %d.%d", resolved, *host_cuda)
    return resolved


def _extract_direct_target(info: dict) -> SshTarget | None:
    """Pull the direct ssh endpoint out of a `show_instance` response, or None if not yet populated."""
    public_ip = (info.get("public_ipaddr") or "").strip()
    mapping = (info.get("ports") or {}).get("22/tcp") or []
    if not (public_ip and mapping):
        return None
    try:
        host_port = int(mapping[0].get("HostPort"))
    except (TypeError, ValueError):
        return None
    if not host_port:
        return None
    return SshTarget(user="root", host=public_ip, port=host_port)


def resolve_ssh_target(instance_id: int, *, timeout_s: int = 600, poll_s: int = 5) -> SshTarget:
    """Poll vast.ai until the direct SSH host:port mapping is populated, then return it.

    `--direct` allocates a real port on the host (`public_ipaddr` + `ports["22/tcp"][0].HostPort`)
    that maps straight to the container's sshd. Docker fills in the `ports` field shortly after
    the container reports `actual_status == "running"`, so we poll. We never fall back to the
    `ssh<N>.vast.ai` proxy — the proxy uses a reverse tunnel that's slow to program and often
    leaves the port refusing connections for minutes.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        info = vast.show_instance(instance_id)
        target = _extract_direct_target(info)
        if target is not None:
            log.info("using direct SSH target %s:%s", target.host, target.port)
            return target
        log.debug(
            "direct SSH mapping not yet populated for instance %s "
            "(public_ipaddr=%r, ports=%r); polling...",
            instance_id, info.get("public_ipaddr"), info.get("ports"),
        )
        time.sleep(poll_s)
    raise TimeoutError(
        f"direct SSH mapping for instance {instance_id} did not appear within {timeout_s}s "
        f"(expected public_ipaddr + ports['22/tcp'][0].HostPort in `vastai show instance`)"
    )


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


def _vast_status_summary(instance_id: int) -> str:
    """One-line summary of vast.ai's view of the instance, for progress logs."""
    try:
        info = vast.show_instance(instance_id)
    except Exception as e:  # noqa: BLE001 — never let a status fetch break the wait
        return f"<show_instance failed: {e}>"
    parts = []
    for field in ("actual_status", "cur_state", "next_state", "status_msg"):
        v = info.get(field)
        if v:
            parts.append(f"{field}={v}")
    return ", ".join(parts) or "<no status fields>"


def wait_for_ssh(
    target: SshTarget,
    *,
    instance_id: int | None = None,
    timeout_s: int = 1200,
    poll_s: int = 5,
    heartbeat_s: int = 30,
) -> None:
    """Poll the SSH port until it accepts a TCP connection AND completes a banner exchange.

    vast.ai's `actual_status == 'running'` only means the container has been
    scheduled. The image pull, sshd boot, and edge-proxy port mapping all happen
    AFTER that flip — so "Connection refused" for the first 1–5 minutes is
    expected, especially on hosts that don't have the image cached.

    We probe with both a TCP connect and a no-op `ssh ... true` so a half-open
    port (SYN+ACK but no banner) doesn't fool us. Every `heartbeat_s` we also
    re-fetch the vast.ai status and log it so the user can see what's blocking.
    """
    start = time.time()
    deadline = start + timeout_s
    last_err: Exception | None = None
    last_heartbeat = 0.0

    while time.time() < deadline:
        elapsed = int(time.time() - start)
        if instance_id is not None and (time.time() - last_heartbeat) >= heartbeat_s:
            log.info(
                "still waiting for ssh (%ds elapsed): vast says %s",
                elapsed, _vast_status_summary(instance_id),
            )
            last_heartbeat = time.time()

        try:
            with socket.create_connection((target.host, target.port), timeout=5):
                pass
        except OSError as e:
            last_err = e
            log.debug("tcp probe %s:%s failed: %s", target.host, target.port, e)
            time.sleep(poll_s)
            continue
        # TCP open — confirm SSH is actually serving by running a no-op.
        rc, stderr = ssh.probe(target, "true")
        if rc == 0:
            log.info("ssh on %s:%s is ready (after %ds)", target.host, target.port, elapsed)
            return
        last_err = RuntimeError(stderr or f"ssh probe rc={rc}")
        log.debug("ssh probe failed (rc=%s): %s", rc, last_err)
        time.sleep(poll_s)
    raise TimeoutError(
        f"ssh on {target.host}:{target.port} not ready within {timeout_s}s "
        f"(last error: {last_err})"
    )


def wait_for_onstart(target: SshTarget, *, timeout_s: int = 600, poll_s: int = 5) -> None:
    """Wait for the onstart_cmd to finish installing uv (probes /root/.local/bin/env)."""
    deadline = time.time() + timeout_s
    last_err: str | None = None
    while time.time() < deadline:
        rc, stderr = ssh.probe(target, "test -f /root/.local/bin/env")
        if rc == 0:
            log.info("onstart finished: uv is installed")
            return
        last_err = stderr or f"rc={rc}"
        log.debug("onstart probe not ready: %s", last_err)
        time.sleep(poll_s)
    raise TimeoutError(
        f"onstart_cmd did not finish installing uv within {timeout_s}s "
        f"(missing /root/.local/bin/env on {target.host}; last: {last_err})"
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
