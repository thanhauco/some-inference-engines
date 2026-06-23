import numpy as np

from inference_engines.concepts.fusion import (
    epilogue_traffic,
    fuse_qkv,
    fused_mlp_epilogue,
    fusion_report,
    split_qkv,
    unfused_mlp_epilogue,
)

rng = np.random.default_rng(0)


def test_fused_epilogue_matches_unfused():
    x = rng.normal(size=(8, 64))
    w = rng.normal(size=(64, 200)) / 8
    b = rng.normal(size=200)
    r = rng.normal(size=(8, 200))
    np.testing.assert_allclose(
        fused_mlp_epilogue(x, w, b, r, tile=48), unfused_mlp_epilogue(x, w, b, r)
    )


def test_qkv_fusion_is_one_gemm_with_the_same_result():
    x = rng.normal(size=(5, 32))
    wq, wk, wv = rng.normal(size=(32, 32)), rng.normal(size=(32, 8)), rng.normal(size=(32, 8))
    q, k, v = split_qkv(x @ fuse_qkv(wq, wk, wv), q_dim=32, kv_dim=8)
    np.testing.assert_allclose(q, x @ wq)
    np.testing.assert_allclose(k, x @ wk)
    np.testing.assert_allclose(v, x @ wv)


def test_fusion_cuts_traffic_and_launches():
    unfused = epilogue_traffic(256, 4096, fused=False)
    fused = epilogue_traffic(256, 4096, fused=True)
    assert unfused.kernel_launches == 4 and fused.kernel_launches == 1
    assert unfused.activation_bytes == 4 * fused.activation_bytes
    report = fusion_report(layers=32)
    assert report["launches_saved"] == 96 and report["fused_us"] < report["unfused_us"] / 3
