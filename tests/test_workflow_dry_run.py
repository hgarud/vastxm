from vastxm.config import LaunchConfig
from vastxm.workflow import launch


def test_dry_run_prints_plan_and_returns_zero(capsys):
    cfg = LaunchConfig(cmd="echo hi", dry_run=True, bundle_root=".")
    rc = launch(cfg)
    assert rc == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "echo hi" in out
    assert "Onstart cmd" in out
    assert "Remote run script" in out
