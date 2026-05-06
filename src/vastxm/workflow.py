from datetime import datetime
from pathlib import Path

from vastxm import bundle, instance, ssh, vast
from vastxm._log import get_logger
from vastxm.config import LaunchConfig

log = get_logger(__name__)


ONSTART_SCRIPT = """\
set -euo pipefail
mkdir -p /workspace
curl -LsSf https://astral.sh/uv/install.sh | sh
echo 'source $HOME/.local/bin/env' >> /root/.bashrc
"""


def _remote_run_script(bundle_dirname: str, user_cmd: str) -> str:
    return f"""\
set -euo pipefail
cd /workspace
tar xzf bundle.tar.gz
source $HOME/.local/bin/env
cd {bundle_dirname}
uv sync
{user_cmd} 2>&1 | tee /workspace/train.log
"""


def _resolve_name(cfg: LaunchConfig) -> str:
    if cfg.name:
        return cfg.name
    return f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def _print_dry_run(cfg: LaunchConfig, run_name: str, bundle_dirname: str) -> None:
    print("=== DRY RUN ===")
    print(f"name              : {run_name}")
    print(f"bundle_root       : {cfg.bundle_root}")
    image_note = " (auto-resolved per offer's cuda_max_good)" if cfg.image == "vastai/base-image:auto" else ""
    print(f"image             : {cfg.image}{image_note}")
    print(f"gpu / num_gpus    : {cfg.gpu} x {cfg.num_gpus}")
    print(f"max_price (USD/hr): {cfg.max_price}")
    print(f"disk (GB)         : {cfg.disk}")
    print(f"excludes          : {list(cfg.exclude)}")
    print(f"artifact_path     : {cfg.artifact_path}")
    print(f"artifact_dest     : {cfg.artifact_dest}/{run_name}/")
    print(f"keep on exit      : {cfg.keep}")
    print()
    print("Offer query:")
    print(f"  {instance.build_offer_query(cfg)}")
    print()
    print("Onstart cmd:")
    print(ONSTART_SCRIPT)
    print("Remote run script:")
    print(_remote_run_script(bundle_dirname, cfg.cmd))


def launch(cfg: LaunchConfig) -> int:
    """Orchestrate one launch. Returns the remote training exit code (or 0 on dry-run)."""
    run_name = _resolve_name(cfg)
    bundle_root = Path(cfg.bundle_root).resolve()
    bundle_dirname = bundle_root.name

    if cfg.dry_run:
        _print_dry_run(cfg, run_name, bundle_dirname)
        return 0

    vast.ensure_installed()
    log.info("=== vastxm launch '%s' ===", run_name)

    bundle_path = bundle.create_bundle(bundle_root, excludes=cfg.exclude)

    instance.ensure_ssh_key()
    offer = instance.pick_offer(cfg)
    offer_id = int(offer["id"])
    image = instance.resolve_image(cfg.image, offer)
    created = vast.create_instance(
        offer_id,
        image=image,
        disk=cfg.disk,
        onstart_cmd=ONSTART_SCRIPT,
        ssh=True,
        direct=True,
    )
    instance_id_raw = created.get("new_contract") or created.get("id")
    if instance_id_raw is None:
        raise vast.VastError(
            f"vastai create instance returned no instance id; "
            f"the instance may still be running. Full response: {created}"
        )
    instance_id = int(instance_id_raw)
    log.info("created instance %s", instance_id)

    rc = 1
    try:
        instance.wait_for_running(instance_id)

        target = instance.resolve_ssh_target(instance_id)
        log.info("waiting for sshd on %s:%s...", target.host, target.port)
        instance.wait_for_ssh(target, instance_id=instance_id)
        log.info("waiting for onstart_cmd to finish (uv install)...")
        instance.wait_for_onstart(target)

        log.info("uploading bundle (%d KB)...", bundle_path.stat().st_size // 1024)
        vast.copy(f"local:{bundle_path}", f"{instance_id}:/workspace/bundle.tar.gz")

        log_file = Path(cfg.artifact_dest) / run_name / "train.log"
        rc = ssh.run_remote_streaming(
            target,
            _remote_run_script(bundle_dirname, cfg.cmd),
            log_file=log_file,
        )
        log.info("remote command exited with rc=%s", rc)
    finally:
        _cleanup(instance_id, cfg, run_name)
    return rc


def _cleanup(instance_id: int, cfg: LaunchConfig, run_name: str) -> None:
    """Best-effort artifact pull, then destroy unless --keep."""
    artifact_local = Path(cfg.artifact_dest) / run_name
    artifact_local.mkdir(parents=True, exist_ok=True)
    try:
        log.info("pulling artifacts %s -> %s", cfg.artifact_path, artifact_local)
        vast.copy(f"{instance_id}:{cfg.artifact_path}", f"local:{artifact_local}/")
    except Exception as e:  # noqa: BLE001 — never let cleanup mask the original failure
        log.warning("artifact pull failed: %s", e)

    if cfg.keep:
        log.info("--keep set; instance %s is still running. Stop with `vastxm stop %s`.", instance_id, instance_id)
        return
    try:
        log.info("destroying instance %s", instance_id)
        vast.destroy_instance(instance_id)
    except Exception as e:  # noqa: BLE001
        log.error("destroy failed: %s — destroy manually with `vastai destroy instance %s`", e, instance_id)
