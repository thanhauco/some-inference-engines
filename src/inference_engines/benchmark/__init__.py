"""Measure an engine under *your* workload: concurrency sweeps, TTFT, TPOT, throughput.

Published benchmarks fix the model, prompt lengths, output lengths, concurrency and
hardware - usually not yours. ``loadgen`` replays your prompts against any of the four
engines through the same OpenAI-compatible client, so the engine is the only variable.
"""
