// TensorRT-LLM: compile -> kernel & layer fusion -> quantization -> in-flight batching.
(function () {
  const { C, W, clamp, lerp, seg, eo, eio, hexA, rr, text, line, arrow, user, box, tokenBox, bar, stat, fmt } = K;
  const A = C.green;
  const S = window.SIM;
  const PAL = [C.blue, C.purple, C.teal, C.amber, C.pink, C.ok];

  // ---- 1. compile ---------------------------------------------------------------------
  const PASSES = ["parsing graph", "fusing layers", "selecting kernels", "calibrating FP8", "planning memory", "serializing engine"];
  const compile = {
    dur: 10, step: 1, title: "Load and optimize the model",
    caption: "The model is compiled into an engine for one GPU architecture and fixed maximum shapes.",
    draw(ctx, p) {
      const stages = [
        { x: 60, label: "HF / ONNX", sub: "model weights", color: C.dim },
        { x: 360, label: "convert", sub: "layout · TP shards · quant", color: C.teal },
        { x: 660, label: "trtllm-build", sub: "TensorRT compiler", color: A },
        { x: 960, label: "Engine", sub: "the “plan”", color: A },
      ];
      const pos = seg(p, 0.08, 0.72) * 3;                 // which hop the payload is on
      stages.forEach((s, i) => {
        const reached = pos >= i - 0.02;
        box(ctx, s.x, 236, 240, 110, s.label, { color: s.color, sub: s.sub, size: 23, alpha: eo(p, i * 0.03, 0.1 + i * 0.03), glow: reached && i === Math.round(pos) ? 0.6 : 0 });
        if (i < 3) arrow(ctx, s.x + 248, 291, s.x + 292, 291, { color: pos > i ? A : C.faint, width: 3 });
      });
      // payload travelling between stages, shrinking as it is optimized
      if (p > 0.08 && pos < 3) {
        const i = Math.floor(pos), f = K.ease.inOut(pos - i);
        const x = lerp(stages[i].x + 120, stages[i + 1].x + 120, f), size = lerp(30, 18, pos / 3);
        rr(ctx, x - size, 196 - size / 2, size * 2, size, 6, hexA(A, 0.9));
      }
      // build log
      rr(ctx, 60, 392, 720, 216, 16, "#0C1322", C.line, 1.5);
      text(ctx, "$ trtllm-build --checkpoint_dir ckpt --output_dir engine \\", 84, 432, { size: 16, mono: true, color: C.text });
      text(ctx, "    --max_batch_size 128 --max_seq_len 16384", 84, 456, { size: 16, mono: true, color: C.text });
      const stage = seg(p, 0.3, 0.74) * PASSES.length;
      PASSES.forEach((name, i) => {
        if (stage < i) return;
        const done = stage >= i + 1;
        const col = i % 2, row = Math.floor(i / 2);
        text(ctx, (done ? "✓ " : "› ") + name, 84 + col * 330, 498 + row * 34, { size: 16, mono: true, color: done ? A : C.amber });
      });
      // what the engine is bound to
      const ea = eo(p, 0.72, 0.84);
      ctx.save(); ctx.globalAlpha *= ea;
      rr(ctx, 820, 392, 400, 216, 16, hexA(A, 0.1), A, 1.5);
      text(ctx, "ENGINE IS SPECIFIC TO", 846, 428, { size: 13, weight: 700, color: A, spacing: 1.5 });
      ["GPU architecture (e.g. H100)", "max batch size · 128", "max sequence · 16,384 tokens", "TensorRT-LLM version"].forEach((l, i) => {
        text(ctx, "•", 846, 466 + i * 34, { size: 18, color: A, weight: 800 });
        text(ctx, l, 868, 466 + i * 34, { size: 17, weight: 500 });
      });
      ctx.restore();
    },
  };

  // ---- 2. fusion ----------------------------------------------------------------------
  const KERN = ["MatMul", "+ bias", "GELU", "+ residual"];
  const fusion = {
    dur: 12, step: 2, title: "Kernel and layer fusion",
    caption: "Fused, the activations make one trip through GPU memory instead of four.",
    draw(ctx, p) {
      const merge = eio(p, 0.46, 0.6);
      const hbmY = 500;
      rr(ctx, 60, hbmY, 1160, 56, 14, C.panel, C.line, 1.5);
      text(ctx, "HBM  ·  GPU memory (slow compared with on-chip registers)", 640, hbmY + 35, { size: 17, weight: 600, color: C.dim, align: "center" });
      const xs0 = [110, 390, 670, 950], w0 = 220;
      let trips = 0;
      if (merge < 1) {
        const phase = seg(p, 0.06, 0.44) * 4;             // kernel currently executing
        KERN.forEach((name, i) => {
          const x = lerp(xs0[i], 400 + i * 120, merge), w = lerp(w0, 120, merge);
          const active = Math.floor(phase) === i && merge === 0;
          box(ctx, x, 236, w, 96, merge > 0.5 ? "" : name, { color: active ? A : C.dim, size: 21, glow: active ? 0.7 : 0, alpha: 1 - merge * 0.4 });
          text(ctx, `kernel ${i + 1}`, x + w / 2, 222, { size: 13, color: C.faint, align: "center", alpha: 1 - merge });
          if (phase > i) trips++;
          // read up, write down
          const f = clamp(phase - i);
          if (f > 0 && merge === 0) {
            const cx = x + w / 2;
            line(ctx, cx - 26, hbmY - 4, cx - 26, 340, hexA(A, 0.35), 2, [5, 6]);
            line(ctx, cx + 26, 340, cx + 26, hbmY - 4, hexA(C.amber, 0.35), 2, [5, 6]);
            if (f < 0.5) { ctx.fillStyle = A; ctx.beginPath(); ctx.arc(cx - 26, lerp(hbmY - 6, 342, f * 2), 8, 0, 7); ctx.fill(); }
            else if (f < 1) { ctx.fillStyle = C.amber; ctx.beginPath(); ctx.arc(cx + 26, lerp(342, hbmY - 6, (f - 0.5) * 2), 8, 0, 7); ctx.fill(); }
            text(ctx, "read", cx - 40, 430, { size: 13, color: A, align: "right" });
            text(ctx, "write", cx + 40, 430, { size: 13, color: C.amber });
          }
        });
      }
      if (merge > 0) {
        const fa = eo(p, 0.54, 0.64);
        const g = box(ctx, 400, 236, 480, 96, "MatMul + bias + GELU + residual", { color: A, size: 21, glow: 0.6 * fa, alpha: fa, sub: "one fused kernel · intermediates stay on-chip" });
        const f = seg(p, 0.64, 0.84);
        if (f > 0) {
          line(ctx, g.cx - 26, hbmY - 4, g.cx - 26, 340, hexA(A, 0.35), 2, [5, 6]);
          line(ctx, g.cx + 26, 340, g.cx + 26, hbmY - 4, hexA(C.amber, 0.35), 2, [5, 6]);
          if (f < 0.5) { ctx.fillStyle = A; ctx.beginPath(); ctx.arc(g.cx - 26, lerp(hbmY - 6, 342, f * 2), 8, 0, 7); ctx.fill(); }
          else if (f < 1) { ctx.fillStyle = C.amber; ctx.beginPath(); ctx.arc(g.cx + 26, lerp(342, hbmY - 6, (f - 0.5) * 2), 8, 0, 7); ctx.fill(); }
        }
        trips = f > 0 ? 1 : 0;
      }
      text(ctx, merge >= 1 ? "after fusion" : "eager execution", 60, 196, { size: 16, weight: 700, color: merge >= 1 ? A : C.amber });
      const ra = eo(p, 0.86, 0.94);
      ctx.save(); ctx.globalAlpha *= ra;
      [["4 → 1", "kernel launches"], [`${S.fusion.traffic_reduction_x.toFixed(0)}× less`, "activation traffic through HBM"], [String(S.fusion.launches_saved), "launches saved per pass (32 layers)"]].forEach(([v, l], i) => {
        stat(ctx, 110 + i * 390, 608, v, l, { size: 34, color: A });
      });
      ctx.restore();
      if (ra <= 0) text(ctx, `HBM round trips: ${trips}`, 1220, 196, { size: 18, weight: 700, align: "right", mono: true, color: merge >= 1 ? A : C.amber });
    },
  };

  // ---- 3. quantization ----------------------------------------------------------------
  const GW = 26, GH = 6;
  const WEIGHTS = (() => { let s = 11; const out = []; for (let i = 0; i < GW * GH; i++) { s = (s * 1664525 + 1013904223) % 4294967296; const u = s / 4294967296, v = (u - 0.5) * 2; out.push(Math.sign(v) * Math.pow(Math.abs(v), 1.6)); } return out; })();
  const FORMATS = [
    { name: "BF16", bits: 16, levels: 4096, gib: S.quantization.bf16_gib },
    { name: "FP8", bits: 8, levels: 96, gib: S.quantization.fp8_gib },
    { name: "INT4", bits: 4, levels: 7, gib: S.quantization.int4_gib },
  ];
  function wcolor(v) {
    const a = Math.min(1, Math.abs(v) * 1.15 + 0.08);
    return v >= 0 ? `rgba(118,192,67,${a})` : `rgba(76,141,255,${a})`;
  }
  const quant = {
    dur: 11, step: 3, title: "Quantization",
    caption: "Decoding is memory-bound: fewer bytes per weight means faster tokens and room for bigger batches.",
    draw(ctx, p) {
      const stage = seg(p, 0.1, 0.8) * 2.999, fi = Math.floor(stage), fmtq = FORMATS[fi];
      // weight grid
      rr(ctx, 48, 168, 760, 300, 16, C.panel, C.line, 1.5);
      text(ctx, "ONE WEIGHT MATRIX", 72, 202, { size: 13, weight: 700, color: C.dim, spacing: 1.5 });
      text(ctx, `${fmtq.levels === 4096 ? "65,536" : fmtq.bits === 8 ? "256" : "16"} representable values`, 784, 202, { size: 14, color: C.dim, align: "right" });
      for (let i = 0; i < GW * GH; i++) {
        const v = Math.round(WEIGHTS[i] * fmtq.levels) / fmtq.levels;
        rr(ctx, 72 + (i % GW) * 27.5, 222 + Math.floor(i / GW) * 38, 24, 33, 5, wcolor(v));
      }
      // bits readout
      rr(ctx, 832, 168, 400, 300, 16, C.panel, A, 1.5);
      text(ctx, "BITS PER WEIGHT", 858, 202, { size: 13, weight: 700, color: C.dim, spacing: 1.5 });
      text(ctx, String(fmtq.bits), 858, 320, { size: 120, weight: 800, color: A, spacing: -4 });
      text(ctx, fmtq.name, 1206, 320, { size: 40, weight: 700, align: "right" });
      FORMATS.forEach((f, i) => {
        const on = i === fi;
        rr(ctx, 858 + i * 118, 372, 106, 40, 20, on ? hexA(A, 0.25) : "#0D1526", on ? A : C.line, 1.5);
        text(ctx, f.name, 858 + i * 118 + 53, 398, { size: 16, weight: 650, align: "center", color: on ? C.text : C.dim });
      });
      text(ctx, "Llama 3.1 8B", 858, 446, { size: 14, color: C.faint });
      // memory bars
      rr(ctx, 48, 488, 1184, 126, 16, C.panel, C.line, 1.5);
      text(ctx, "WEIGHTS IN VRAM", 72, 522, { size: 13, weight: 700, color: C.dim, spacing: 1.5 });
      FORMATS.forEach((f, i) => {
        const a = eo(stage, i - 0.05, i + 0.25);
        if (a <= 0) return;
        const y = 536 + i * 24;
        text(ctx, f.name, 72, y + 13, { size: 14, weight: 600, color: i === fi ? C.text : C.dim });
        bar(ctx, 130, y, 800, 14, (f.gib / FORMATS[0].gib) * a, i === fi ? A : hexA(A, 0.4), "#0D1526");
        text(ctx, `${f.gib.toFixed(1)} GiB`, 946, y + 13, { size: 15, weight: 650, mono: true, alpha: a });
        if (i > 0) text(ctx, `${(16 / f.bits).toFixed(0)}× smaller`, 1208, y + 13, { size: 14, color: A, align: "right", alpha: a });
      });
    },
  };

  // ---- 4. in-flight batching + streaming ------------------------------------------------
  const serve = {
    dur: 10, step: 4, title: "In-flight batching, low-latency tokens",
    caption: "Requests join a running batch; every step moves fewer bytes and launches fewer kernels.",
    draw(ctx, p, t) {
      const eng = box(ctx, 380, 232, 300, 260, "TensorRT engine", { color: A, sub: "fused · quantized · tuned", glow: 0.5, size: 23 });
      for (let r = 0; r < 5; r++) {
        const y = 256 + r * 50;
        user(ctx, 96, y, 0.85, PAL[r], eo(p, r * 0.03, 0.12 + r * 0.03));
        // requests joining mid-flight at staggered times
        const join = 0.6 + r * 0.9;
        const q = clamp((t - join) / 0.8);
        if (q > 0 && q < 1) { ctx.fillStyle = PAL[r]; ctx.beginPath(); ctx.arc(lerp(130, eng.x - 8, K.ease.inOut(q)), lerp(y, eng.cy + (r - 2) * 30, K.ease.inOut(q)), 7, 0, 7); ctx.fill(); }
        line(ctx, 130, y, eng.x - 8, eng.cy + (r - 2) * 30, "#1C2942", 2);
        if (t > join + 0.8) {
          line(ctx, eng.r + 8, eng.cy + (r - 2) * 30, 800, y, "#1C2942", 2);
          for (let k = 0; k < 8; k++) {
            const f = ((t - join) * 0.9 + k * 0.125) % 1;
            rr(ctx, lerp(eng.r + 8, 790, f) - 8, lerp(eng.cy + (r - 2) * 30, y, f) - 5, 16, 10, 4, PAL[r]);
          }
        }
      }
      text(ctx, "requests join the running batch", 240, 546, { size: 15, color: C.dim, align: "center" });
      text(ctx, "tokens stream out", 730, 546, { size: 15, color: C.dim, align: "center" });
      const cards = [
        [`${S.quantization.fp8_x.toFixed(0)}×`, "fewer weight bytes", "per decode step · FP8 vs BF16"],
        [`${S.fusion.traffic_reduction_x.toFixed(0)}×`, "less activation traffic", "fused MLP epilogue"],
        [String(S.fusion.launches_saved), "fewer kernel launches", "per forward pass · 32 layers"],
      ];
      cards.forEach(([v, l, sub], i) => {
        const a = eo(p, 0.3 + i * 0.12, 0.45 + i * 0.12);
        ctx.save(); ctx.globalAlpha *= a;
        rr(ctx, 852, 190 + i * 136 + 10 * (1 - a), 380, 120, 16, C.panel, hexA(A, 0.6), 1.5);
        text(ctx, v, 878, 190 + i * 136 + 70, { size: 46, weight: 780, color: A, spacing: -1 });
        text(ctx, l, 1000, 190 + i * 136 + 56, { size: 17, weight: 600 });
        text(ctx, sub, 1000, 190 + i * 136 + 80, { size: 14, color: C.dim });
        ctx.restore();
      });
      text(ctx, "figures from this repo's fusion and quantization models, not from a GPU benchmark", 60, 606, { size: 14, color: C.faint, alpha: eo(p, 0.6, 0.7) });
    },
  };

  window.VIDEOS.tensorrt_llm = {
    id: "tensorrt_llm", title: "TensorRT-LLM", accent: A, steps: 4,
    scenes: [
      K.titleScene("TensorRT-LLM", "Maximum performance with NVIDIA optimizations", ["Compiled engines", "Kernel fusion", "FP8 / INT4 quantization"]),
      compile, fusion, quant, serve,
      K.summaryScene({
        when: ["Every millisecond matters, and you are willing to trade simplicity", "for deeper NVIDIA-specific optimization."],
        best: ["Peak performance on NVIDIA GPUs", "Production deployments at scale", ["Latency-critical products", "lowest time per token"], "Best GPU economics"],
        pros: ["Best performance, lowest latency", "NVIDIA-specific optimizations", ["Deep quantization support", "FP8, INT8, INT4"], "Production-grade stability"],
        cons: ["Requires NVIDIA GPUs", ["More complex build", "engines per GPU and shape"], "Newest models may arrive later", "Less flexible for agentic flows"],
      }),
    ],
  };
})();
