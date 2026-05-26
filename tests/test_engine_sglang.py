from inference_engines.engines import sglang


def test_launch_command_plain():
    cmd = sglang.launch_command("m", tp=2, mem_fraction_static=0.85)
    assert cmd[:3] == ["python", "-m", "sglang.launch_server"]
    assert "--speculative-algorithm" not in cmd
    assert cmd[cmd.index("--tp") + 1] == "2"
    assert cmd[cmd.index("--mem-fraction-static") + 1] == "0.85"


def test_launch_command_speculative():
    cmd = sglang.launch_command(
        "m", speculative_draft_model="draft", speculative_num_draft_tokens=8
    )
    assert cmd[cmd.index("--speculative-algorithm") + 1] == "EAGLE"
    assert cmd[cmd.index("--speculative-draft-model-path") + 1] == "draft"
    assert cmd[cmd.index("--speculative-num-draft-tokens") + 1] == "8"


def test_json_schema_response_format():
    fmt = sglang.json_schema_response_format("ticket", {"type": "object"})
    assert fmt == {
        "type": "json_schema",
        "json_schema": {"name": "ticket", "schema": {"type": "object"}},
    }
