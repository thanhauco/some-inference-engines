// Overview: four engines, how to choose, and the migration path between them.
(function () {
  const { C, W, clamp, lerp, seg, eo, eio, hexA, rr, text, line, arrow, box, stat } = K;
  const A = C.amber;
  const ENGINES = [
    { name: "Ollama", color: C.teal, one: "simplest local experience", idea: ["One command", "Runs on a laptop", "Works offline"], stage: "experimenting", port: ":11434" },
    { name: "vLLM", color: C.blue, one: "scalable production serving", idea: ["Continuous batching", "PagedAttention", "Many concurrent users"], stage: "real users arrive", port: ":8000" },
    { name: "SGLang", color: C.purple, one: "programmable, agentic inference", idea: ["Inference programs", "RadixAttention", "Structured output"], stage: "workflows get complex", port: ":30000" },
    { name: "TensorRT-LLM", color: C.green, one: "maximum NVIDIA performance", idea: ["Compiled engines", "Kernel fusion", "FP8 / INT4"], stage: "latency / GPU cost critical", port: ":8001" },
  ];

  // ---- 1. four engines ------------------------------------------------------------------
  const four = {
    dur: 10, step: 1, title: "Four engines, four design centers",
    caption: "Same model, same API — very different throughput, latency and developer experience.",
    draw(ctx, p) {
      ENGINES.forEach((e, i) => {
        const a = eo(p, 0.04 + i * 0.12, 0.2 + i * 0.12), x = 48 + i * 300, y = 176 + 16 * (1 - a);
        ctx.save(); ctx.globalAlpha *= a;
        rr(ctx, x, y, 284, 430, 16, C.panel, C.line, 1.5);
        ctx.save(); ctx.beginPath(); ctx.roundRect(x, y, 284, 96, [16, 16, 0, 0]); ctx.clip();
        ctx.fillStyle = hexA(e.color, 0.22); ctx.fillRect(x, y, 284, 96);
        ctx.fillStyle = e.color; ctx.fillRect(x, y, 284, 5);
        ctx.restore();
        text(ctx, e.name, x + 24, y + 50, { size: 27, weight: 780, spacing: -0.5 });
        text(ctx, e.one, x + 24, y + 78, { size: 15, weight: 500, color: C.dim });
        e.idea.forEach((it, j) => {
          const ia = eo(p, 0.2 + i * 0.12 + j * 0.04, 0.32 + i * 0.12 + j * 0.04);
          ctx.fillStyle = e.color; ctx.globalAlpha = a * ia;
          ctx.beginPath(); ctx.arc(x + 30, y + 142 + j * 46, 5, 0, 7); ctx.fill();
          text(ctx, it, x + 48, y + 148 + j * 46, { size: 18, weight: 550 });
        });
        ctx.globalAlpha = a;
        line(ctx, x + 24, y + 290, x + 260, y + 290, C.line, 1.5);
        text(ctx, "CHOOSE WHEN", x + 24, y + 322, { size: 12, weight: 700, color: e.color, spacing: 1.5 });
        const when = [["you want the fastest path", "to running a model locally"], ["many users hit one", "production API"], ["requests are programs:", "agents, tools, schemas"], ["every millisecond and", "GPU dollar counts"]][i];
        when.forEach((w, j) => text(ctx, w, x + 24, y + 350 + j * 24, { size: 16, weight: 500 }));
        ctx.restore();
      });
    },
  };

  // ---- 2. decide by the shape of the problem ----------------------------------------------
  const QS = [
    ["Where does it run?", "a laptop, a CPU, a Mac — or just experimenting", 0],
    ["How many concurrent users?", "dozens to thousands on a production API", 1],
    ["What is a “request”?", "a program: agents, tools, branches, schemas", 2],
    ["What is a millisecond worth?", "latency or GPU cost drives the business, on NVIDIA", 3],
  ];
  const decide = {
    dur: 12, step: 2, title: "Decide by the shape of your problem",
    caption: "Benchmarks tell you who is fastest on their workload. These questions are about yours.",
    draw(ctx, p) {
      QS.forEach(([q, ans, ei], i) => {
        const a = eo(p, 0.04 + i * 0.19, 0.16 + i * 0.19), y = 176 + i * 110;
        const e = ENGINES[ei];
        ctx.save(); ctx.globalAlpha *= a;
        rr(ctx, 48, y, 760, 94, 16, C.panel, C.line, 1.5);
        ctx.fillStyle = hexA(A, 0.18); ctx.beginPath(); ctx.arc(92, y + 47, 24, 0, 7); ctx.fill();
        text(ctx, String(i + 1), 92, y + 55, { size: 22, weight: 800, color: A, align: "center" });
        text(ctx, q, 134, y + 42, { size: 22, weight: 700 });
        text(ctx, ans, 134, y + 70, { size: 16, color: C.dim });
        ctx.restore();
        const ar = eo(p, 0.14 + i * 0.19, 0.24 + i * 0.19);
        arrow(ctx, 820, y + 47, 906, y + 47, { color: e.color, prog: ar, width: 3 });
        const ba = eo(p, 0.2 + i * 0.19, 0.3 + i * 0.19);
        ctx.save(); ctx.globalAlpha *= ba;
        rr(ctx, 920 + 14 * (1 - ba), y + 12, 312, 70, 16, hexA(e.color, 0.2), e.color, 2);
        text(ctx, e.name, 946 + 14 * (1 - ba), y + 44, { size: 22, weight: 750 });
        text(ctx, e.one, 946 + 14 * (1 - ba), y + 66, { size: 14, color: C.dim });
        ctx.restore();
      });
    },
  };

  // ---- 3. migration path ------------------------------------------------------------------
  const path = {
    dur: 12, step: 3, title: "You may not stay with one engine forever",
    caption: "The model can stay the same. Changing the engine changes how it performs in production.",
    draw(ctx, p) {
      const xs = [170, 483, 797, 1110], y = 356;
      const prog = eio(p, 0.08, 0.8) * 3;
      line(ctx, xs[0], y, xs[3], y, C.line, 6);
      line(ctx, xs[0], y, lerp(xs[0], xs[3], prog / 3), y, C.text, 6);
      ENGINES.forEach((e, i) => {
        const reached = prog >= i - 0.02, a = eo(p, i * 0.04, 0.12 + i * 0.04);
        ctx.save(); ctx.globalAlpha *= a;
        ctx.fillStyle = reached ? e.color : C.panel2; ctx.strokeStyle = e.color; ctx.lineWidth = 3;
        if (reached) { ctx.shadowColor = hexA(e.color, 0.7); ctx.shadowBlur = 24; }
        ctx.beginPath(); ctx.arc(xs[i], y, 22, 0, 7); ctx.fill(); ctx.stroke();
        ctx.shadowBlur = 0;
        text(ctx, e.name, xs[i], y - 54, { size: 26, weight: 780, align: "center", color: reached ? C.text : C.dim, spacing: -0.5 });
        text(ctx, e.one, xs[i], y - 100, { size: 14, color: C.faint, align: "center" });
        ctx.restore();
        const ta = eo(prog, i - 0.25, i + 0.1) * a;
        ctx.save(); ctx.globalAlpha *= ta;
        rr(ctx, xs[i] - 130, y + 52 + 10 * (1 - ta), 260, 92, 14, hexA(e.color, 0.14), hexA(e.color, 0.8), 1.5);
        text(ctx, i === 0 ? "START" : "MOVE WHEN", xs[i], y + 82 + 10 * (1 - ta), { size: 12, weight: 700, color: e.color, align: "center", spacing: 1.5 });
        text(ctx, e.stage, xs[i], y + 116 + 10 * (1 - ta), { size: 18, weight: 650, align: "center" });
        ctx.restore();
      });
      // the model travelling along the path, unchanged
      const mx = lerp(xs[0], xs[3], prog / 3);
      rr(ctx, mx - 62, y - 178, 124, 34, 17, C.text);
      text(ctx, "same model", mx, y - 155, { size: 15, weight: 700, color: C.bg, align: "center" });
      line(ctx, mx, y - 144, mx, y - 126, C.text, 2);
      const fa = eo(p, 0.84, 0.94);
      text(ctx, "Inference engineering is becoming a discipline of its own.", 640, 596, { size: 24, weight: 650, align: "center", alpha: fa });
    },
  };

  // ---- 4. switching is a config change ------------------------------------------------------
  const swap = {
    dur: 10, step: 4, title: "Make switching a config change",
    caption: "All four serve the OpenAI API. Code against the API, not the engine.",
    draw(ctx, p, t) {
      const sel = Math.min(3, Math.floor(seg(p, 0.12, 0.96) * 4));
      const e = ENGINES[sel];
      rr(ctx, 48, 190, 420, 250, 16, "#0C1322", C.line, 1.5);
      text(ctx, "app.py", 72, 224, { size: 14, mono: true, color: C.faint });
      ["client = InferenceClient(ENGINE)", "", "reply = client.chat(messages)", "print(reply.text)"].forEach((l, i) => text(ctx, l, 72, 268 + i * 32, { size: 17, mono: true }));
      rr(ctx, 48, 460, 420, 84, 16, hexA(e.color, 0.14), e.color, 2);
      text(ctx, "ENVIRONMENT", 72, 492, { size: 12, weight: 700, color: e.color, spacing: 1.5 });
      text(ctx, `ENGINE=${e.name.toLowerCase()}`, 72, 524, { size: 22, mono: true, weight: 600 });
      const hub = box(ctx, 540, 320, 230, 96, "OpenAI API", { color: C.dim, sub: "/v1/chat/completions", size: 20 });
      arrow(ctx, 476, 368, 532, 368, { color: C.dim, width: 3 });
      ENGINES.forEach((en, i) => {
        const on = i === sel, y = 190 + i * 104;
        const b = box(ctx, 900, y, 332, 84, en.name, { color: on ? en.color : C.line, sub: "localhost" + en.port, size: 21, glow: on ? 0.7 : 0, fill: on ? hexA(en.color, 0.16) : C.panel });
        line(ctx, hub.r + 8, hub.cy, b.x - 8, b.cy, on ? en.color : "#1C2942", on ? 3 : 2);
        if (on) for (let k = 0; k < 3; k++) {
          const q = (t * 0.9 + k / 3) % 1;
          rr(ctx, lerp(hub.r + 8, b.x - 8, q) - 8, lerp(hub.cy, b.cy, q) - 5, 16, 10, 4, en.color);
        }
      });
      text(ctx, "no code change", 258, 590, { size: 18, weight: 650, align: "center", color: C.ok, alpha: eo(p, 0.3, 0.42) });
    },
  };

  // ---- 5. closing --------------------------------------------------------------------------
  const closing = {
    dur: 6, bare: true,
    draw(ctx, p) {
      text(ctx, "Choose on the shape of your problem.", 640, 300, { size: 46, weight: 780, align: "center", alpha: eo(p, 0.02, 0.2), spacing: -1 });
      text(ctx, "Then benchmark on your own workload.", 640, 364, { size: 46, weight: 780, align: "center", color: A, alpha: eo(p, 0.2, 0.4), spacing: -1 });
      ["ie recommend", "ie simulate", "ie bench"].forEach((c, i) => {
        const a = eo(p, 0.45 + i * 0.08, 0.6 + i * 0.08);
        ctx.save(); ctx.globalAlpha *= a;
        rr(ctx, 356 + i * 196, 440, 176, 46, 23, C.panel, C.line, 1.5);
        text(ctx, c, 444 + i * 196, 470, { size: 18, mono: true, align: "center" });
        ctx.restore();
      });
    },
  };

  window.VIDEOS.overview = {
    id: "overview", title: "Which inference engine?", accent: A, steps: 4,
    scenes: [
      K.titleScene("Which engine?", "vLLM, SGLang, TensorRT-LLM or Ollama — and when", ["Throughput", "Latency", "GPU utilization", "Developer experience", "Economics"], 4),
      four, decide, path, swap, closing,
    ],
  };
})();
