from pathlib import Path

from vastxm.config import DEFAULT_IMAGE, LaunchConfig, merge_config


def test_defaults_only(tmp_path: Path):
    cfg = merge_config(cmd="echo hi", toml_path=tmp_path / "missing.toml", cli_overrides={})
    assert cfg.cmd == "echo hi"
    assert cfg.gpu == "A100"
    assert cfg.image == DEFAULT_IMAGE


def test_toml_overrides_default(tmp_path: Path):
    p = tmp_path / "vastxm.toml"
    p.write_text('[defaults]\ngpu = "H100"\nmax_price = 3.0\nexclude = ["foo/", "bar/"]\n')
    cfg = merge_config(cmd="x", toml_path=p, cli_overrides={})
    assert cfg.gpu == "H100"
    assert cfg.max_price == 3.0
    assert cfg.exclude == ("foo/", "bar/")


def test_cli_overrides_toml(tmp_path: Path):
    p = tmp_path / "vastxm.toml"
    p.write_text('[defaults]\ngpu = "H100"\n')
    cfg = merge_config(cmd="x", toml_path=p, cli_overrides={"gpu": "A40"})
    assert cfg.gpu == "A40"


def test_cli_none_does_not_override(tmp_path: Path):
    p = tmp_path / "vastxm.toml"
    p.write_text('[defaults]\ngpu = "H100"\n')
    cfg = merge_config(cmd="x", toml_path=p, cli_overrides={"gpu": None, "disk": 200})
    assert cfg.gpu == "H100"
    assert cfg.disk == 200
