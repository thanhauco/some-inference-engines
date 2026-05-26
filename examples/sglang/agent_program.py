"""An inference *program* in SGLang's frontend language.

One support ticket becomes: classify (constrained `select`) -> branch -> draft three
candidate replies in parallel (`fork`) -> pick the best -> return structured output.
Every call shares the ticket prefix, so RadixAttention computes it once.

    ./examples/sglang/serve.sh                     # in another terminal
    pip install "sglang" && python examples/sglang/agent_program.py
"""

import sglang as sgl

SYSTEM = "You are a senior support engineer for an LLM inference platform. Be precise."


@sgl.function
def handle_ticket(s, ticket: str):
    s += sgl.system(SYSTEM)
    s += sgl.user(f"Ticket:\n{ticket}\n\nWhich team owns this?")
    # `select` scores each choice and picks one - no free text, no parsing.
    s += sgl.assistant(sgl.select("team", choices=["billing", "performance", "bug", "other"]))

    if s["team"] == "performance":
        s += sgl.user("List the three most likely root causes, one per line.")
        s += sgl.assistant(sgl.gen("causes", max_tokens=120, stop="\n\n"))

    # Draft three replies in parallel. The forks share all KV computed so far.
    drafts = s.fork(3)
    for i, d in enumerate(drafts):
        d += sgl.user(f"Draft reply #{i + 1} to the customer, max 80 words, style {i + 1}/3.")
        d += sgl.assistant(sgl.gen("reply", max_tokens=160, temperature=0.8))
    drafts.join()

    options = "\n\n".join(f"[{i}] {d['reply']}" for i, d in enumerate(drafts))
    s += sgl.user(f"Candidates:\n{options}\n\nWhich candidate is clearest? Answer with its number.")
    s += sgl.assistant(sgl.gen("best", regex=r"[012]"))
    # Record the winning draft as the final assistant turn (plain text, no generation).
    s += sgl.user("Send the chosen reply.")
    s += sgl.assistant(drafts[int(s["best"])]["reply"])


if __name__ == "__main__":
    sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))
    tickets = [
        "Since yesterday our p99 TTFT went from 300ms to 4s at the same traffic. Nothing deployed.",
        "We were charged twice for October.",
    ]
    # run_batch submits all programs at once; the runtime batches and shares prefixes.
    for state in handle_ticket.run_batch([{"ticket": t} for t in tickets], progress_bar=True):
        final = state.messages()[-1]["content"]
        print(f"team={state['team']}  best={state['best']}\n{final}\n")
