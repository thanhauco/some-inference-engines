import pytest

from inference_engines.cli import SIMULATIONS, build_parser, main


def test_engines_lists_all_four(capsys):
    assert main(["engines"]) == 0
    out = capsys.readouterr().out
    for name in ("ollama", "vllm", "sglang", "tensorrt-llm"):
        assert name in out


@pytest.mark.parametrize(
    "argv, expected",
    [
        (["launch", "ollama"], "ollama pull llama3.1:8b"),
        (["launch", "vllm", "--tp", "2"], "--tensor-parallel-size 2"),
        (["launch", "sglang", "--draft-model", "d"], "--speculative-draft-model-path d"),
        (["launch", "tensorrt-llm", "--model", "m"], "trtllm-serve m"),
    ],
)
def test_launch(capsys, argv, expected):
    assert main(argv) == 0
    assert expected in capsys.readouterr().out


def test_recommend(capsys):
    main(["recommend", "--hardware", "apple-silicon", "--local-only", "--stage", "prototype"])
    assert capsys.readouterr().out.strip().endswith("=> ollama")
    main(["recommend", "--concurrency", "64", "--workload", "agentic"])
    assert capsys.readouterr().out.strip().endswith("=> sglang")


@pytest.mark.parametrize("topic", sorted(SIMULATIONS))
def test_every_simulation_runs(capsys, topic):
    assert main(["simulate", topic]) == 0
    assert capsys.readouterr().out.strip()


def test_bench_requires_an_engine():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["bench"])
