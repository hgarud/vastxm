import tomllib
from dataclasses import dataclass, fields, replace
from pathlib import Path

from vastxm._log import get_logger

log = get_logger(__name__)

DEFAULT_IMAGE = "vastai/base-image:auto"
DEFAULT_EXCLUDES = (
    ".git/", ".venv/", "__pycache__/", ".pytest_cache/",
    "checkpoints/", "logs/", "runs/",
    "*.pt", "*.bin", "*.safetensors",
)


@dataclass(frozen=True)
class LaunchConfig:
    # Required (no sensible default)
    cmd: str                        # full command to run on instance, e.g. 'deepspeed --num_gpus=4 train.py ...'

    # Offer search
    gpu: str = "A100"               # vast gpu_name filter, e.g. "A100", "RTX_4090", "H100"
    num_gpus: int = 1
    max_price: float = 2.00         # USD/hr cap (vast field: dph_total)
    disk: int = 50                  # GB

    # Instance
    image: str = DEFAULT_IMAGE

    # Bundling
    bundle_root: str = "."          # relative path of project to ship; tar root
    exclude: tuple[str, ...] = DEFAULT_EXCLUDES

    # Artifacts
    artifact_path: str = "/workspace/output"      # where on the instance to grab from
    artifact_dest: str = "./runs"                 # where locally to put artifacts

    # Run metadata + flags
    name: str = ""                  # short label for logs and run dir; auto-filled if empty
    keep: bool = False
    dry_run: bool = False


def load_toml_defaults(path: Path) -> dict:
    """Load `[defaults]` table from a vastxm.toml; return {} if file missing."""
    if not path.exists():
        return {}
    with path.open("rb") as f:
        data = tomllib.load(f)
    return data.get("defaults", {}) or {}


def merge_config(*, cmd: str, toml_path: Path, cli_overrides: dict) -> LaunchConfig:
    """Build a LaunchConfig from (in order, last wins): hardcoded defaults, toml file, cli overrides."""
    base = LaunchConfig(cmd=cmd)
    toml = load_toml_defaults(toml_path)
    cli = {k: v for k, v in cli_overrides.items() if v is not None}

    valid = {f.name for f in fields(base)} - {"cmd"}
    unknown = sorted((toml.keys() | cli.keys()) - valid)
    if unknown:
        log.warning("ignoring unknown config keys: %s", ", ".join(unknown))

    merged: dict = {}
    merged.update({k: v for k, v in toml.items() if k in valid})
    merged.update({k: v for k, v in cli.items() if k in valid})

    if "exclude" in merged and isinstance(merged["exclude"], list):
        merged["exclude"] = tuple(merged["exclude"])

    return replace(base, **merged)
