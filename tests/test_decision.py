import pytest

from inference_engines.config import Engine
from inference_engines.decision import (
    MIGRATION_PATH,
    Hardware,
    Requirements,
    Stage,
    Workload,
    explain,
    recommend,
)


def top(**kwargs) -> Engine:
    return recommend(Requirements(**kwargs))[0].engine


def test_local_prototype_on_a_laptop_is_ollama():
    assert (
        top(hardware=Hardware.APPLE_SILICON, local_only=True, stage=Stage.PROTOTYPE)
        is Engine.OLLAMA
    )
    assert top(hardware=Hardware.CPU, concurrency=1) is Engine.OLLAMA


def test_production_chat_api_with_many_users_is_vllm():
    assert top(concurrency=200, workload=Workload.CHAT, ops_capacity=2) is Engine.VLLM
    assert top(hardware=Hardware.AMD_GPU, concurrency=100) is Engine.VLLM


def test_agentic_and_structured_workloads_are_sglang():
    assert top(concurrency=50, workload=Workload.AGENTIC) is Engine.SGLANG
    assert top(concurrency=50, workload=Workload.STRUCTURED) is Engine.SGLANG


def test_latency_critical_at_scale_with_a_strong_team_is_tensorrt_llm():
    assert top(concurrency=500, latency_critical=True, ops_capacity=3, stage=Stage.SCALE) is (
        Engine.TENSORRT_LLM
    )


def test_latency_critical_without_inference_team_is_not_tensorrt_llm():
    assert top(concurrency=100, latency_critical=True, ops_capacity=1) is not Engine.TENSORRT_LLM


@pytest.mark.parametrize("hw", [Hardware.CPU, Hardware.APPLE_SILICON, Hardware.AMD_GPU])
def test_tensorrt_llm_requires_nvidia(hw):
    recs = recommend(Requirements(hardware=hw, latency_critical=True, ops_capacity=3))
    trt = next(r for r in recs if r.engine is Engine.TENSORRT_LLM)
    assert not trt.viable and recs[-1] is trt


def test_every_score_change_is_explained():
    recs = recommend(Requirements(concurrency=64, workload=Workload.AGENTIC))
    text = explain(recs)
    for rec in recs:
        assert rec.engine.value in text
        if rec.viable:
            assert rec.reasons


def test_migration_path_order():
    assert [engine for engine, *_ in MIGRATION_PATH] == [
        Engine.OLLAMA,
        Engine.VLLM,
        Engine.SGLANG,
        Engine.TENSORRT_LLM,
    ]
