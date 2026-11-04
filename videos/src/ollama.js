// Ollama: pull -> load (GPU/CPU split) -> run locally -> one API -> unload when idle.
(function () {
  const { C, W, clamp, lerp, seg, eo, eio, hexA, rr, text, line, arrow, user, box, bar, stat, typed } = K;
  const A = C.teal;
  const S = window.SIM;

  function terminal(ctx, x, y, w, h, title) {
    rr(ctx, x, y, w, h, 16, "#0C1322", C.line, 1.5);
    ["#FF5F57", "#FEBC2E", "#28C840"].forEach((c, i) => { ctx.fillStyle = c; ctx.beginPath(); ctx.arc(x + 26 + i * 20, y + 24, 6, 0, 7); ctx.fill(); });
    text(ctx, title, x + w / 2, y + 29, { size: 14, color: C.faint, align: "center", mono: true });
  }
  function laptop(ctx, x, y, s, color, glow = 0) {
    ctx.save();
    if (glow) { ctx.shadowColor = hexA(color, 0.6 * glow); ctx.shadowBlur = 30 * glow; }
    rr(ctx, x - 90 * s, y - 60 * s, 180 * s, 112 * s, 10 * s, C.panel, color, 2.5);
    ctx.shadowBlur = 0;
    rr(ctx, x - 118 * s, y + 58 * s, 236 * s, 12 * s, 6 * s, color);
    ctx.restore();
  }

  // ---- 1. pull ------------------------------------------------------------------------
  const gb = S.ollama.q4_gb.toFixed(1);
  const pull = {
    dur: 9, step: 1, title: "Pull the model",
    caption: "One command downloads a quantized model from the registry — once.",
    draw(ctx, p) {
      terminal(ctx, 48, 168, 700, 440, "~ — zsh");
      const cmd = "ollama pull llama3.1:8b";
      text(ctx, "$ " + typed(cmd, seg(p, 0.04, 0.2)), 76, 244, { size: 21, mono: true });
      const dl = eio(p, 0.26, 0.74);
      if (p > 0.22) text(ctx, "pulling manifest", 76, 288, { size: 18, mono: true, color: C.dim });
      if (p > 0.26) {
        text(ctx, "pulling model layers", 76, 328, { size: 18, mono: true, color: C.dim });
        bar(ctx, 76, 346, 520, 16, dl, A, "#16223B");
        text(ctx, `${Math.round(dl * 100)}%`.padStart(4, " "), 612, 361, { size: 18, mono: true });
        text(ctx, `${(dl * gb).toFixed(1)} GB / ${gb} GB`, 76, 394, { size: 16, mono: true, color: C.faint });
      }
      if (p > 0.76) text(ctx, "verifying sha256 digest", 76, 434, { size: 18, mono: true, color: C.dim });
      if (p > 0.82) text(ctx, "success", 76, 474, { size: 18, mono: true, color: C.ok });
      if (p > 0.86) text(ctx, "$ ▍", 76, 520, { size: 21, mono: true });
      // registry -> laptop
      const reg = box(ctx, 880, 180, 260, 96, "Ollama registry", { color: C.dim, sub: "GGUF · 4-bit quantized", size: 20 });
      laptop(ctx, 1010, 500, 1, A, dl);
      text(ctx, "your machine", 1010, 600, { size: 16, color: C.dim, align: "center" });
      for (let k = 0; k < 7; k++) {
        const q = (dl * 5 + k / 7) % 1;
        if (dl <= 0 || dl >= 1) break;
        rr(ctx, 1010 - 16 + Math.sin(k * 2.1) * 26, lerp(reg.b + 10, 420, q), 32, 14, 4, hexA(A, 1 - q * 0.6));
      }
      text(ctx, `${(dl * gb).toFixed(1)} GB`, 1010, 506, { size: 26, weight: 750, align: "center", color: dl > 0 ? C.text : C.faint });
    },
  };

  // ---- 2. load: GPU / CPU split --------------------------------------------------------
  const OFF = S.ollama.offload.filter((o) => o.vram > 0);     // VRAM 4, 6, 8 GiB
  const load = {
    dur: 12, step: 2, title: "Load: as many layers on the GPU as fit",
    caption: "Every token passes through all layers — the ones left on the CPU set the pace.",
    draw(ctx, p) {
      const stage = seg(p, 0.08, 0.86) * (OFF.length - 0.001), i = Math.floor(stage), f = stage - i;
      const cur = OFF[i], prev = i > 0 ? OFF[i - 1] : { gpu_layers: 0, tok_s: S.ollama.offload[0].tok_s };
      const k = eio(f, 0.05, 0.55);
      const onGpu = Math.round(lerp(prev.gpu_layers, cur.gpu_layers, k));
      const speed = lerp(prev.tok_s, cur.tok_s, k);
      // VRAM selector
      OFF.forEach((o, j) => {
        const on = j === i;
        rr(ctx, 60 + j * 150, 176, 136, 44, 22, on ? hexA(A, 0.22) : "#0D1526", on ? A : C.line, 1.5);
        text(ctx, `${o.vram} GiB VRAM`, 60 + j * 150 + 68, 204, { size: 16, weight: 650, align: "center", color: on ? C.text : C.dim });
      });
      text(ctx, "Llama 3.1 8B · q4_K_M · 32 layers", 1220, 204, { size: 16, color: C.dim, align: "right" });
      // layers
      rr(ctx, 48, 244, 1184, 176, 16, C.panel, C.line, 1.5);
      // a token walking through all the layers; it lingers on the CPU ones
      const tokPos = (p * 30) % 1;
      const split = onGpu / 32, gpuShare = (split * 0.5) / (split * 0.5 + (1 - split) * 4);
      const layerF = tokPos < gpuShare ? (tokPos / gpuShare) * split : split + ((tokPos - gpuShare) / (1 - gpuShare)) * (1 - split);
      const walker = Math.min(31, Math.floor(layerF * 32));
      for (let l = 0; l < 32; l++) {
        const gpu = l < onGpu;
        rr(ctx, 72 + l * 35.6, 292, 29, 84, 6, gpu ? A : C.amber, l === walker ? C.text : null, 3);
        text(ctx, String(l + 1), 72 + l * 35.6 + 14.5, 398, { size: 11, color: C.faint, align: "center", mono: true });
      }
      text(ctx, `■ GPU  ${onGpu} layers`, 72, 276, { size: 16, weight: 650, color: A });
      text(ctx, `■ CPU  ${32 - onGpu} layers`, 300, 276, { size: 16, weight: 650, color: 32 - onGpu ? C.amber : C.faint });
      text(ctx, "▢ layer the current token is in", 1208, 276, { size: 14, color: C.dim, align: "right" });
      // speed
      stat(ctx, 60, 520, speed.toFixed(1), "tokens per second", { size: 72, color: onGpu === 32 ? C.ok : onGpu > 24 ? A : C.amber });
      const max = S.ollama.offload[S.ollama.offload.length - 1].tok_s;
      bar(ctx, 330, 482, 620, 22, speed / max, onGpu === 32 ? C.ok : A, "#16223B");
      text(ctx, `CPU only: ${S.ollama.offload[0].tok_s} tok/s`, 330, 540, { size: 15, color: C.dim });
      text(ctx, `fully on GPU: ${max} tok/s`, 950, 540, { size: 15, color: C.dim, align: "right" });
      text(ctx, "illustrative model: 0.5 ms per layer on GPU, 4 ms on CPU", 330, 572, { size: 14, color: C.faint });
      if (onGpu === 32) text(ctx, "fits entirely ✓", 1220, 500, { size: 22, weight: 700, color: C.ok, align: "right" });
      else text(ctx, `${32 - onGpu} layers spill`, 1220, 500, { size: 22, weight: 700, color: C.amber, align: "right" });
    },
  };

  // ---- 3. run locally ------------------------------------------------------------------
  const ANSWER = ["Sunlight", " scatters", " off", " air", " molecules,", " and", " blue", " light", " scatters", " the", " most", " —", " so", " the", " sky", " looks", " blue."];
  const run = {
    dur: 9, step: 3, title: "Run inference locally",
    caption: "The same command on a CPU, an NVIDIA or AMD GPU, or Apple Silicon. Nothing leaves the machine.",
    draw(ctx, p) {
      terminal(ctx, 48, 168, 760, 440, "ollama run");
      text(ctx, "$ " + typed("ollama run llama3.1:8b", seg(p, 0.03, 0.16)), 76, 244, { size: 21, mono: true });
      if (p > 0.2) text(ctx, ">>> " + typed("Why is the sky blue?", seg(p, 0.2, 0.34)), 76, 292, { size: 21, mono: true, color: A });
      const n = Math.floor(seg(p, 0.4, 0.9) * ANSWER.length);
      let lineStr = "", row = 0;
      ANSWER.slice(0, n).forEach((tok) => {
        if ((lineStr + tok).length > 46) { text(ctx, lineStr, 76, 348 + row * 36, { size: 21, mono: true }); lineStr = tok.trimStart(); row++; }
        else lineStr += tok;
      });
      text(ctx, lineStr + (n < ANSWER.length && p > 0.4 ? "▍" : ""), 76, 348 + row * 36, { size: 21, mono: true });
      if (p > 0.4) text(ctx, `${n} tokens`, 780, 584, { size: 15, color: C.faint, align: "right", mono: true });
      const chips = [["CPU", "any laptop or server"], ["NVIDIA / AMD GPU", "CUDA · ROCm"], ["Apple Silicon", "Metal · unified memory"]];
      const active = Math.floor(seg(p, 0.1, 0.95) * 3 - 0.001) % 3;
      chips.forEach(([name, sub], i) => {
        const on = i === active;
        box(ctx, 860, 190 + i * 136, 360, 108, name, { color: on ? A : C.line, sub, size: 21, glow: on ? 0.6 : 0, alpha: eo(p, i * 0.04, 0.14 + i * 0.04) });
      });
    },
  };

  // ---- 4. one API ----------------------------------------------------------------------
  const api = {
    dur: 9, step: 4, title: "Return the response: CLI, app or API",
    caption: "Ollama also speaks the OpenAI API, so the app can move to another engine later.",
    draw(ctx, p, t) {
      const srv = box(ctx, 500, 290, 280, 150, "ollama serve", { color: A, sub: "localhost:11434", size: 23, glow: 0.5 });
      const clients = [["CLI", "ollama run"], ["Your app", "client.chat(messages)"], ["curl", "/v1/chat/completions"]];
      clients.forEach(([name, sub], i) => {
        const y = 196 + i * 136, a = eo(p, i * 0.05, 0.15 + i * 0.05);
        const b = box(ctx, 60, y, 300, 100, name, { color: C.dim, sub, size: 20, alpha: a });
        line(ctx, b.r + 8, b.cy, srv.x - 8, srv.cy + (i - 1) * 34, "#1C2942", 2);
        for (let k = 0; k < 2; k++) {       // request out, tokens back
          const q = ((t * 0.5 + i * 0.21 + k * 0.5) % 1);
          const fwd = k === 0;
          const e = fwd ? q : 1 - q;
          rr(ctx, lerp(b.r + 8, srv.x - 8, e) - 8, lerp(b.cy, srv.cy + (i - 1) * 34, e) - 5, 16, 10, 4, fwd ? C.dim : A);
        }
      });
      const a = eo(p, 0.4, 0.55);
      ctx.save(); ctx.globalAlpha *= a;
      rr(ctx, 850, 190, 382, 396, 16, C.panel, C.line, 1.5);
      text(ctx, "SAME API, OTHER ENGINES", 876, 226, { size: 13, weight: 700, color: C.dim, spacing: 1.5 });
      const engines = [["Ollama", C.teal, ":11434"], ["vLLM", C.blue, ":8000"], ["SGLang", C.purple, ":30000"], ["TensorRT-LLM", C.green, ":8001"]];
      const sel = Math.floor(seg(p, 0.5, 0.98) * 4 - 0.001);
      engines.forEach(([name, col, port], i) => {
        const on = i === Math.max(0, sel);
        rr(ctx, 876, 250 + i * 62, 330, 48, 12, on ? hexA(col, 0.22) : "#0D1526", on ? col : C.line, 1.8);
        ctx.fillStyle = col; ctx.beginPath(); ctx.arc(900, 274 + i * 62, 6, 0, 7); ctx.fill();
        text(ctx, name, 920, 281 + i * 62, { size: 18, weight: 650, color: on ? C.text : C.dim });
        text(ctx, port + "/v1", 1190, 281 + i * 62, { size: 15, mono: true, color: on ? C.text : C.faint, align: "right" });
      });
      text(ctx, `ENGINE=${engines[Math.max(0, sel)][0].toLowerCase()} python app.py`, 876, 536, { size: 16, mono: true, color: C.text });
      text(ctx, "only the base URL changes", 876, 562, { size: 14, color: C.faint });
      ctx.restore();
    },
  };

  // ---- 5. keep-alive -------------------------------------------------------------------
  const idle = {
    dur: 8, step: 5, title: "Idle? The model unloads",
    caption: "Great for a laptop that needs its memory back. The next request pays a cold start.",
    draw(ctx, p) {
      const left = 1 - eio(p, 0.1, 0.62), secs = Math.round(left * 300);
      const cx = 300, cy = 392, r = 120;
      ctx.lineWidth = 16; ctx.lineCap = "round";
      ctx.strokeStyle = "#16223B"; ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.stroke();
      if (left > 0.002) { ctx.strokeStyle = A; ctx.beginPath(); ctx.arc(cx, cy, r, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * left); ctx.stroke(); }
      text(ctx, `${Math.floor(secs / 60)}:${String(secs % 60).padStart(2, "0")}`, cx, cy + 18, { size: 56, weight: 750, align: "center", mono: true });
      text(ctx, "OLLAMA_KEEP_ALIVE", cx, cy + 172, { size: 15, mono: true, color: C.dim, align: "center" });
      const loaded = 1 - eio(p, 0.62, 0.74);
      rr(ctx, 540, 208, 692, 150, 16, C.panel, C.line, 1.5);
      text(ctx, "MEMORY HELD BY THE MODEL", 566, 244, { size: 13, weight: 700, color: C.dim, spacing: 1.5 });
      bar(ctx, 566, 268, 640, 26, loaded, A, "#16223B");
      text(ctx, `${(S.ollama.q4_gib * loaded).toFixed(1)} GiB`, 566, 334, { size: 28, weight: 750, color: loaded > 0.05 ? C.text : C.ok });
      text(ctx, loaded > 0.05 ? "model loaded · ready" : "unloaded · memory returned to you", 1206, 334, { size: 16, color: loaded > 0.05 ? C.dim : C.ok, align: "right" });
      const ca = eo(p, 0.78, 0.88);
      ctx.save(); ctx.globalAlpha *= ca;
      rr(ctx, 540, 382, 692, 190, 16, hexA(C.amber, 0.1), C.amber, 1.5);
      text(ctx, "NEXT REQUEST", 566, 418, { size: 13, weight: 700, color: C.amber, spacing: 1.5 });
      text(ctx, `≈ ${S.ollama.cold_start_s} s`, 566, 484, { size: 56, weight: 780, color: C.amber, spacing: -1 });
      text(ctx, "cold start: reload the model from disk", 566, 518, { size: 17, weight: 500 });
      text(ctx, `${gb} GB file at 2 GB/s NVMe — before the first token`, 566, 544, { size: 14, color: C.dim });
      ctx.restore();
    },
  };

  window.VIDEOS.ollama = {
    id: "ollama", title: "Ollama", accent: A, steps: 5,
    scenes: [
      K.titleScene("Ollama", "Simple, local LLM running on your machine", ["One command", "CPU · GPU · Apple Silicon", "Works offline"]),
      pull, load, run, api, idle,
      K.summaryScene({
        when: ["You want the fastest path to local experimentation, and", "production-scale concurrency is not yet your problem."],
        best: ["Local development and testing", ["Open models on your machine", "laptop, workstation, Mac"], "Simple setup, easy to use", "Privacy-sensitive or offline work"],
        pros: ["Very easy to set up and use", ["Runs anywhere", "CPU, GPU, Apple Silicon"], "Huge library of ready models", "Lightweight, great for experiments"],
        cons: [["Not built for", "high-concurrency serving"], "Limited to one machine", ["Lower throughput than", "vLLM / SGLang / TensorRT-LLM"]],
      }),
    ],
  };
})();
