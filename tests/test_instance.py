from vastxm.config import LaunchConfig
from vastxm.instance import (
    _extract_direct_target,
    _gpu_filter,
    _parse_cuda,
    build_offer_query,
    resolve_image,
)


def test_gpu_filter_family_expansion():
    f = _gpu_filter("A100")
    assert f.startswith("gpu_name in [")
    assert "A100_SXM4" in f and "A100_PCIE" in f


def test_gpu_filter_exact_match():
    assert _gpu_filter("RTX_5090") == "gpu_name=RTX_5090"


def test_build_offer_query_includes_all_fields():
    cfg = LaunchConfig(cmd="x", gpu="H100", num_gpus=2, max_price=3.0, disk=80)
    q = build_offer_query(cfg)
    assert "num_gpus=2" in q
    assert "dph_total<=3.0" in q
    assert "disk_space>=80" in q
    assert "verified=true" in q
    assert "rentable=true" in q
    assert "direct_port_count>=1" in q


def test_parse_cuda_variants():
    assert _parse_cuda("12.8") == (12, 8)
    assert _parse_cuda(13.0) == (13, 0)
    assert _parse_cuda("12.4.1") == (12, 4)
    assert _parse_cuda(None) is None
    assert _parse_cuda("garbage") is None


def test_resolve_image_passthrough_for_explicit_tag():
    assert resolve_image("custom/img:1", {"cuda_max_good": "12.8"}) == "custom/img:1"


def test_resolve_image_picks_highest_tag_below_host_cuda():
    # Host cuda 12.8 should pick cuda-12.8.1-auto, not 12.9
    img = resolve_image("vastai/base-image:auto", {"cuda_max_good": "12.8"})
    assert img == "vastai/base-image:cuda-12.8.1-auto"


def test_resolve_image_falls_back_when_cuda_missing():
    img = resolve_image("vastai/base-image:auto", {"cuda_max_good": None, "id": 1})
    assert img == "vastai/base-image:cuda-12.4.1-auto"


def test_extract_direct_target_happy_path():
    info = {
        "public_ipaddr": "203.0.113.5 ",
        "ports": {"22/tcp": [{"HostPort": "12345"}]},
    }
    t = _extract_direct_target(info)
    assert t is not None
    assert (t.user, t.host, t.port) == ("root", "203.0.113.5", 12345)


def test_extract_direct_target_missing_returns_none():
    assert _extract_direct_target({}) is None
    assert _extract_direct_target({"public_ipaddr": "1.2.3.4"}) is None
    assert _extract_direct_target({"public_ipaddr": "1.2.3.4", "ports": {}}) is None
    assert _extract_direct_target(
        {"public_ipaddr": "1.2.3.4", "ports": {"22/tcp": [{"HostPort": None}]}}
    ) is None
