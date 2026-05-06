import tarfile
from pathlib import Path

from vastxm.bundle import _matches_exclude, create_bundle


def test_matches_exclude_directory():
    assert _matches_exclude("checkpoints", True, ("checkpoints/",))
    assert _matches_exclude("checkpoints/x.pt", False, ("checkpoints/",))
    assert _matches_exclude("a/b/.git", True, (".git/",))
    assert not _matches_exclude("source.py", False, ("checkpoints/",))


def test_matches_exclude_glob():
    assert _matches_exclude("a/b/model.pt", False, ("*.pt",))
    assert _matches_exclude("foo.bin", False, ("*.bin",))
    assert not _matches_exclude("foo.txt", False, ("*.pt",))


def test_create_bundle_excludes(tmp_path: Path):
    src = tmp_path / "proj"
    (src / "checkpoints").mkdir(parents=True)
    (src / "src").mkdir()
    (src / "src" / "main.py").write_text("print('hi')\n")
    (src / "checkpoints" / "ckpt.pt").write_text("binary")
    (src / "README.md").write_text("readme")

    out = tmp_path / "bundle.tar.gz"
    create_bundle(src, excludes=("checkpoints/", "*.pt"), output=out)

    with tarfile.open(out) as tar:
        names = tar.getnames()
    assert "proj/src/main.py" in names
    assert "proj/README.md" in names
    assert not any("checkpoints" in n for n in names)
    assert not any(n.endswith(".pt") for n in names)
