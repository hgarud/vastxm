from vastxm.cli import _build_parser


def test_launch_required_flags():
    parser = _build_parser()
    ns = parser.parse_args(["launch", "--cmd", "x", "--dry-run"])
    assert ns.command == "launch"
    assert ns.cmd == "x"
    assert ns.dry_run is True


def test_subcommand_ls():
    parser = _build_parser()
    ns = parser.parse_args(["ls"])
    assert ns.command == "ls"


def test_pull_positional():
    parser = _build_parser()
    ns = parser.parse_args(["pull", "1234", "/remote/path", "./local"])
    assert ns.instance_id == 1234
    assert ns.remote_path == "/remote/path"
    assert ns.local_path == "./local"
