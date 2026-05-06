import fnmatch
import tarfile
from pathlib import Path

from vastxm._log import get_logger

log = get_logger(__name__)

BUNDLE_DIR = Path(".vastxm")
BUNDLE_FILE = BUNDLE_DIR / "bundle.tar.gz"


def _matches_exclude(rel_path: str, is_dir: bool, patterns: tuple[str, ...]) -> bool:
    """Return True if `rel_path` matches any of `patterns`.

    Pattern semantics (intentionally simple):
      - 'foo/'       → directory named 'foo' anywhere in the tree (and everything inside).
      - '*.pt'       → glob match against the *basename*.
      - 'src/foo.py' → exact relative-path match.
    """
    name = Path(rel_path).name
    parts = Path(rel_path).parts
    for pat in patterns:
        if pat.endswith("/"):
            d = pat.rstrip("/")
            if d in parts or (is_dir and name == d):
                return True
        elif "/" in pat:
            if fnmatch.fnmatch(rel_path, pat):
                return True
        else:
            if fnmatch.fnmatch(name, pat):
                return True
    return False


def create_bundle(
    bundle_root: Path,
    *,
    excludes: tuple[str, ...],
    output: Path = BUNDLE_FILE,
) -> Path:
    """Create a tar.gz of `bundle_root`. Returns the output path.

    The archive's top-level directory is `bundle_root.name` so that on extract
    on the remote, members land under e.g. `world_model/...`.
    """
    bundle_root = bundle_root.resolve()
    if not bundle_root.is_dir():
        raise FileNotFoundError(f"bundle_root not a directory: {bundle_root}")

    output.parent.mkdir(parents=True, exist_ok=True)
    arc_top = bundle_root.name
    file_count = 0

    def _filter(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
        nonlocal file_count
        # tarinfo.name is the in-archive path, starting with arc_top/...
        rel = tarinfo.name
        if rel == arc_top:
            return tarinfo
        rel = rel[len(arc_top) + 1:]  # strip "<arc_top>/"
        if _matches_exclude(rel, tarinfo.isdir(), excludes):
            log.debug("exclude: %s", rel)
            return None
        file_count += 1
        return tarinfo

    log.info("bundling %s → %s", bundle_root, output)
    with tarfile.open(output, "w:gz") as tar:
        tar.add(bundle_root, arcname=arc_top, filter=_filter)
    log.info("bundle ready: %s files, %d KB", file_count, output.stat().st_size // 1024)
    return output
