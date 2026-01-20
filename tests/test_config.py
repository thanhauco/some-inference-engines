import pytest

from inference_engines.config import Engine, all_profiles, get_profile


def test_every_engine_has_an_openai_compatible_profile():
    profiles = all_profiles(env={})
    assert {p.engine for p in profiles} == set(Engine)
    assert all(p.base_url.endswith("/v1") for p in profiles)
    # Distinct ports so all four can run side by side.
    assert len({p.base_url for p in profiles}) == 4


@pytest.mark.parametrize("alias", ["trt", "TRTLLM", "tensorrt_llm", "trt-llm", "tensorrt-llm"])
def test_engine_aliases(alias):
    assert Engine.parse(alias) is Engine.TENSORRT_LLM


def test_env_overrides():
    env = {
        "IE_VLLM_BASE_URL": "http://gpu:9000/v1",
        "IE_VLLM_MODEL": "mistral",
        "IE_VLLM_API_KEY": "k",
    }
    profile = get_profile("vllm", env=env)
    assert (profile.base_url, profile.default_model, profile.api_key) == (
        "http://gpu:9000/v1",
        "mistral",
        "k",
    )
    assert profile.root_url == "http://gpu:9000"


def test_tensorrt_llm_env_prefix():
    profile = get_profile(Engine.TENSORRT_LLM, env={"IE_TENSORRT_LLM_MODEL": "m"})
    assert profile.default_model == "m"
