"""``ie`` - command line for exploring, running and choosing inference engines.

$ ie engines                                  # the four profiles and how to start them
$ ie launch vllm --model Qwen/Qwen2.5-7B-Instruct --tp 2
$ ie chat ollama "Why is decoding memory bound?"
$ ie bench vllm --concurrency 1 8 32 --requests 64
$ ie recommend --hardware nvidia-datacenter --concurrency 200 --workload agentic
$ ie simulate batching | paged | radix | prefix | speculative | constrained | ... | all
"""

from __future__ import annotations

import argparse
import asyncio
import shlex
import sys
from collections.abc import Callable, Sequence

from .config import Engine, all_profiles, get_profile


def _print_table(rows: list[dict], file=sys.stdout) -> None:
    if not rows:
        return
    keys = list(rows[0])
    widths = {k: max(len(str(k)), *(len(str(r[k])) for r in rows)) for k in keys}
    print("  ".join(str(k).ljust(widths[k]) for k in keys), file=file)
    print("  ".join("-" * widths[k] for k in keys), file=file)
    for r in rows:
        print("  ".join(str(r[k]).ljust(widths[k]) for k in keys), file=file)


# -- engines / launch -----------------------------------------------------------------------
def cmd_engines(args: argparse.Namespace) -> int:
    for p in all_profiles():
        print(f"{p.engine.value:<13} {p.base_url:<28} {p.default_model}")
        print(f"{'':<13} {p.tagline}")
        print(f"{'':<13} $ {p.launch_hint}\n")
    return 0


def cmd_launch(args: argparse.Namespace) -> int:
    from .engines import ollama, sglang, tensorrt_llm, vllm

    engine = Engine.parse(args.engine)
    model = args.model or get_profile(engine).default_model
    if engine is Engine.OLLAMA:
        cmds = [ollama.launch_command(), ollama.pull_command(model)]
    elif engine is Engine.VLLM:
        cmds = [
            vllm.launch_command(
                model,
                tensor_parallel_size=args.tp,
                max_model_len=args.max_model_len,
                quantization=args.quantization,
            )
        ]
    elif engine is Engine.SGLANG:
        cmds = [sglang.launch_command(model, tp=args.tp, speculative_draft_model=args.draft_model)]
    else:
        cmds = [tensorrt_llm.launch_command(model, tp_size=args.tp)]
    for cmd in cmds:
        print(shlex.join(cmd))
    return 0


# -- chat / bench ---------------------------------------------------------------------------
def cmd_chat(args: argparse.Namespace) -> int:
    from .client import InferenceClient

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})
    with InferenceClient(args.engine) as client:
        gen = client.stream_chat(
            messages, model=args.model, max_tokens=args.max_tokens, temperature=args.temperature
        )
        while True:
            try:
                print(next(gen), end="", flush=True)
            except StopIteration as stop:
                completion = stop.value
                break
    tpot = f"{completion.tpot_s * 1000:.1f} ms" if completion.tpot_s else "n/a"
    ttft = f"{completion.ttft_s * 1000:.0f} ms" if completion.ttft_s else "n/a"
    print(
        f"\n\n[{client.profile.engine.value}] TTFT {ttft} | TPOT {tpot} | "
        f"total {completion.latency_s:.2f} s",
        file=sys.stderr,
    )
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from .benchmark.loadgen import DEFAULT_PROMPTS, LoadConfig, run_load
    from .benchmark.report import to_json, to_markdown
    from .client import AsyncInferenceClient

    prompts = DEFAULT_PROMPTS
    if args.prompts:
        with open(args.prompts) as f:
            prompts = [[{"role": "user", "content": line.strip()}] for line in f if line.strip()]

    async def run():
        results = []
        for engine in args.engine:
            async with AsyncInferenceClient(engine) as client:
                for c in args.concurrency:
                    cfg = LoadConfig(
                        num_requests=max(args.requests, c),
                        concurrency=c,
                        request_rate=args.rate,
                        max_tokens=args.max_tokens,
                        model=args.model,
                    )
                    results.append(await run_load(client, prompts, cfg))
        return results

    results = asyncio.run(run())
    print(to_json(results) if args.json else to_markdown(results))
    return 0


# -- recommend ------------------------------------------------------------------------------
def cmd_recommend(args: argparse.Namespace) -> int:
    from .decision import Hardware, Requirements, Stage, Workload, explain, recommend

    req = Requirements(
        hardware=Hardware(args.hardware),
        workload=Workload(args.workload),
        concurrency=args.concurrency,
        latency_critical=args.latency_critical,
        ops_capacity=args.ops,
        local_only=args.local_only,
        stage=Stage(args.stage),
    )
    recs = recommend(req)
    print(explain(recs))
    print(f"\n=> {recs[0].engine.value}")
    return 0


# -- simulate -------------------------------------------------------------------------------
def _sim_batching() -> None:
    from .concepts.batching import compare

    print("Same 200 requests, same KV memory (200k tokens), same cost model:\n")
    _print_table([r.summary() for r in compare()])


def _sim_paged() -> None:
    from .concepts.kv_cache import contiguous_utilization, paged_utilization

    lens = [180, 950, 64, 2200, 410, 75, 1300, 512]
    print(f"sequence lengths: {lens}")
    contiguous = contiguous_utilization(lens, 4096)
    print(f"contiguous (reserve 4096 each): {contiguous:6.1%} of KV memory used")
    print(f"paged (16-token blocks):        {paged_utilization(lens, 16):6.1%} of KV memory used")


def _sim_radix() -> None:
    from .concepts.radix_cache import agent_workload, prefill_savings

    prompts = agent_workload()
    print("8 agent sessions x 5 turns, 1.5k-token shared system prompt:\n")
    _print_table(
        [
            {"cache": "unbounded", **prefill_savings(prompts)},
            {"cache": "5k tokens", **prefill_savings(prompts, capacity_tokens=5000)},
        ]
    )


def _sim_prefix() -> None:
    from .concepts.prefix_caching import head_to_head

    for capacity in (3000, 1_000_000):
        print(
            f"\nvLLM vs SGLang prefix reuse, KV capacity {capacity:,} tokens "
            "(8 interleaved prompt groups x 16 requests):"
        )
        _print_table([r.__dict__ for r in head_to_head(capacity_tokens=capacity)])


def _sim_speculative() -> None:
    from .concepts.speculative import DraftLM, ToyLM, expected_speedup, generate

    target = ToyLM(seed=1)
    rows = []
    for agreement in (1.0, 0.9, 0.7, 0.4):
        _, stats = generate([1, 2], 400, target, DraftLM(target, agreement), k=4)
        rows.append({
            "draft agreement": agreement,
            "acceptance": round(stats.acceptance_rate, 2),
            "tokens/target pass": round(stats.tokens_per_target_call, 2),
            "speedup (draft 5% cost)": round(expected_speedup(stats.acceptance_rate, 4, 0.05), 2),
        })  # fmt: skip
    _print_table(rows)


def _sim_constrained() -> None:
    from .concepts.constrained import generate_json, random_char_model

    schema = {
        "type": "object",
        "properties": {
            "customer": {"type": "string", "maxLength": 12},
            "priority": {"type": "integer"},
            "category": {"enum": ["billing", "bug", "feature_request"]},
            "escalate": {"type": "boolean"},
        },
    }
    model = random_char_model(3)
    for jump in (False, True):
        text, stats = generate_json(schema, model, jump_forward=jump, seed=3)
        print(
            f"jump_forward={jump!s:<5} model calls={stats.model_calls:<3} "
            f"forced chars={stats.forced_chars:<3} {text}"
        )
    print("\n(the 'model' is random noise - the FSM alone guarantees valid JSON)")


def _sim_quantization() -> None:
    from .concepts.quantization import compare_formats

    _print_table(compare_formats())


def _sim_fusion() -> None:
    from .concepts.fusion import fusion_report

    print("MLP epilogue (bias + GELU + residual) across 32 layers, 256 tokens, H100 bandwidth:")
    for k, v in fusion_report().items():
        print(f"  {k:<20} {v}")


def _sim_memory() -> None:
    from .concepts.memory import plan

    rows = []
    for model, vram, quant, gpus in [
        ("llama-3.1-8b", 16, "q4_K_M", 1),
        ("llama-3.1-8b", 24, "bf16", 1),
        ("llama-3.1-8b", 80, "bf16", 1),
        ("llama-3.1-70b", 80, "bf16", 1),
        ("llama-3.1-70b", 80, "fp8", 1),
        ("llama-3.1-70b", 80, "bf16", 2),
    ]:
        r = plan(model, vram_gib=vram, quant=quant, num_gpus=gpus)
        rows.append(
            {
                "model": r.model,
                "gpu": f"{gpus}x{vram}GiB",
                "quant": quant,
                "weights GiB": r.weights_gib,
                "8k-token seqs": r.max_concurrent_seqs,
            }
        )
    _print_table(rows)


def _sim_interference() -> None:
    from .concepts.prefill_decode import compare

    print("32 sequences streaming while six 32k-token prompts arrive, one per second:\n")
    _print_table([r.row() for r in compare()])


def _sim_routing() -> None:
    from .concepts.routing import compare_policies

    print("16 agents x 24 requests across 4 replicas (4k-token KV cache each):\n")
    _print_table([r.row() for r in compare_policies()])


def _sim_parallelism() -> None:
    from .concepts.parallelism import ETHERNET, NVLINK, plan_layouts

    for model, link in (("llama-3.1-8b", NVLINK), ("llama-3.1-8b", ETHERNET),
                        ("llama-3.1-70b", NVLINK)):  # fmt: skip
        print(f"\n{model} on 4 x 80 GiB GPUs, {link.name}, batch 32 per replica:")
        _print_table([p.row() for p in plan_layouts(model, gpus=4, link=link)])


def _sim_offload() -> None:
    from .concepts.offload import cold_start_s, offload_curve, recommend_quant

    print("Llama 3.1 8B q4_K_M, 4k context, layers that fit on the GPU:\n")
    _print_table([p.row() for p in offload_curve()])
    for vram in (6, 8, 16):
        best = recommend_quant("llama-3.1-8b", vram)
        print(f"\n{vram:>2} GiB VRAM -> use {best.quant} ({best.tokens_per_s:.0f} tok/s)", end="")
    print(f"\n\ncold start from NVMe: {cold_start_s('llama-3.1-8b'):.1f} s")


def _sim_capacity() -> None:
    from .concepts.capacity import EXAMPLE_MEASUREMENTS, breakeven_months, compare_options

    plans = compare_options(EXAMPLE_MEASUREMENTS, peak_tok_per_s=50_000)
    print("Peak 50k output tok/s, $2.50/GPU-hour, 70% target utilization")
    print("(placeholder throughputs - plug in your own `ie bench` results):\n")
    _print_table([p.row() for p in plans])
    months = breakeven_months(60_000, plans[0], plans[-1])
    print(f"\na $60k migration from the first to the last option pays back in {months:.1f} months")


SIMULATIONS: dict[str, Callable[[], None]] = {
    "batching": _sim_batching,
    "paged": _sim_paged,
    "radix": _sim_radix,
    "prefix": _sim_prefix,
    "speculative": _sim_speculative,
    "constrained": _sim_constrained,
    "quantization": _sim_quantization,
    "fusion": _sim_fusion,
    "memory": _sim_memory,
    "interference": _sim_interference,
    "routing": _sim_routing,
    "parallelism": _sim_parallelism,
    "offload": _sim_offload,
    "capacity": _sim_capacity,
}


def cmd_simulate(args: argparse.Namespace) -> int:
    names = list(SIMULATIONS) if args.topic == "all" else [args.topic]
    for name in names:
        print(f"\n=== {name} ===")
        SIMULATIONS[name]()
    return 0


# -- parser ---------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ie", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    engines = [e.value for e in Engine]

    sub.add_parser("engines", help="list engine profiles").set_defaults(func=cmd_engines)

    p = sub.add_parser("launch", help="print the command that starts an engine")
    p.add_argument("engine", choices=engines)
    p.add_argument("--model")
    p.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    p.add_argument("--max-model-len", type=int)
    p.add_argument("--quantization")
    p.add_argument("--draft-model", help="SGLang: enable EAGLE speculative decoding")
    p.set_defaults(func=cmd_launch)

    p = sub.add_parser("chat", help="stream one chat completion and show TTFT/TPOT")
    p.add_argument("engine", choices=engines)
    p.add_argument("prompt")
    p.add_argument("--system")
    p.add_argument("--model")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("bench", help="load-test one or more running engines")
    p.add_argument("engine", nargs="+", choices=engines)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32])
    p.add_argument("--requests", type=int, default=64)
    p.add_argument("--rate", type=float, help="open-loop Poisson request rate (req/s)")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--model")
    p.add_argument("--prompts", help="text file, one prompt per line")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("recommend", help="rank engines for your requirements")
    p.add_argument(
        "--hardware",
        default="nvidia-datacenter",
        choices=["cpu", "apple-silicon", "nvidia-consumer", "nvidia-datacenter", "amd-gpu"],
    )
    p.add_argument("--workload", default="chat", choices=["chat", "agentic", "structured", "batch"])
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--latency-critical", action="store_true")
    p.add_argument(
        "--ops",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="tuning capacity: 1 none, 2 some, 3 dedicated inference team",
    )
    p.add_argument("--local-only", action="store_true")
    p.add_argument("--stage", default="production", choices=["prototype", "production", "scale"])
    p.set_defaults(func=cmd_recommend)

    p = sub.add_parser("simulate", help="run a GPU-free simulation of an engine concept")
    p.add_argument("topic", choices=[*SIMULATIONS, "all"])
    p.set_defaults(func=cmd_simulate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
