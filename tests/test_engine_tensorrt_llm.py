import pytest

from inference_engines.engines import tensorrt_llm as trt


def test_two_step_build():
    convert = trt.convert_checkpoint_command("hf/llama", "ckpt", tp_size=2)
    assert convert[1].endswith("llama/convert_checkpoint.py")
    assert convert[convert.index("--tp_size") + 1] == "2"
    build = trt.build_command("ckpt", "engine", max_batch_size=128, max_num_tokens=8192)
    assert build[0] == "trtllm-build"
    assert build[build.index("--max_batch_size") + 1] == "128"
    assert build[build.index("--max_num_tokens") + 1] == "8192"


def test_build_rejects_inconsistent_shapes():
    with pytest.raises(ValueError):
        trt.build_command("ckpt", "engine", max_input_len=9000, max_seq_len=8192)


def test_serve_command():
    cmd = trt.launch_command("engine_dir", tp_size=4, kv_cache_free_gpu_memory_fraction=0.9)
    assert cmd[:2] == ["trtllm-serve", "engine_dir"]
    assert cmd[cmd.index("--port") + 1] == "8001"
    assert cmd[cmd.index("--kv_cache_free_gpu_memory_fraction") + 1] == "0.9"
