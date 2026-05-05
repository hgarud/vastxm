import pytest

from vastxm.ssh import SshTarget


def test_parse_valid():
    t = SshTarget.parse("ssh://root@ssh4.vast.ai:12345")
    assert t.user == "root"
    assert t.host == "ssh4.vast.ai"
    assert t.port == 12345


def test_parse_trailing_slash():
    t = SshTarget.parse("ssh://root@example.com:22/")
    assert t.port == 22


def test_parse_invalid_raises():
    with pytest.raises(ValueError):
        SshTarget.parse("not-an-ssh-url")
    with pytest.raises(ValueError):
        SshTarget.parse("ssh://root@host")  # no port
