// vLLM: many users -> continuous batching -> PagedAttention -> decode -> stream.
(function () {
  const { C, W, clamp, lerp, seg, eo, eio, hexA, rr, text, line, arrow, user, box, tokenBox, bar, stat, fmt } = K;
  const A = C.blue;
  const PAL = [C.blue, C.purple, C.teal, C.amber, C.pink, C.ok, "#7DD3FC", "#FCA5A5"];
  const S = window.SIM;

  // ---- 1. many users ----------------------------------------------------------------
  const users = {
    dur: 5.5, step: 1, title: "Multiple users / requests",
    caption: "Many users hit the same model at the same time.",
    draw(ctx, p, t) {
      const srv = box(ctx, 860, 270, 320, 200, "vLLM server", { color: A, sub: "1 model · OpenAI-compatible API", glow: 0.5 });
      text(ctx, "POST /v1/chat/completions", 560, 208, { size: 17, mono: true, color: C.dim, align: "center", alpha: eo(p, 0.1, 0.3) });
      const pts = [];
      for (let i = 0; i < 8; i++) pts.push({ x: 110 + (i % 2) * 86, y: 250 + Math.floor(i / 2) * 96 });
      pts.forEach((u, i) => {
        const a = eo(p, i * 0.03, 0.15 + i * 0.03);
        user(ctx, u.x, u.y, 1.25, PAL[i], a);
        // request dots travelling to the server
        for (let k = 0; k < 6; k++) {
          const start = 0.5 + i * 0.17 + k * 0.85;
          const q = (t - start) / 1.1;
          if (q <= 0 || q >= 1) continue;
          const e = K.ease.inOut(q);
          const x = lerp(u.x + 34, srv.x - 10, e), y = lerp(u.y, srv.cy + (i - 3.5) * 14, e);
          ctx.fillStyle = PAL[i];
          ctx.globalAlpha *= 1; ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2); ctx.fill();
          ctx.fillStyle = hexA(PAL[i], 0.25); ctx.beginPath(); ctx.arc(x - 12, y - (srv.cy + (i - 3.5) * 14 - u.y) * 0.012, 4, 0, Math.PI * 2); ctx.fill();
        }
      });
      const n = Math.round(lerp(1, 64, eio(p, 0.15, 0.9)));
      stat(ctx, srv.cx, 540, String(n), "concurrent requests", { align: "center", color: A, alpha: eo(p, 0.1, 0.25) });
    },
  };

  // ---- 2. static vs continuous batching ----------------------------------------------
  const LENS = [4, 9, 3, 6, 8, 5, 3, 7, 4, 9, 6, 4, 8, 3, 5, 7, 5, 4, 6, 3, 8, 4, 6, 5, 7];
  const SLOTS = 5, T_UNITS = 26;
  function staticSchedule() {
    const bars = []; let t0 = 0;
    for (let b = 0; b * SLOTS < LENS.length; b++) {
      const batch = LENS.slice(b * SLOTS, (b + 1) * SLOTS), longest = Math.max(...batch);
      batch.forEach((len, s) => bars.push({ slot: s, start: t0, len, id: b * SLOTS + s, pad: longest - len }));
      t0 += longest;
    }
    return bars;
  }
  function continuousSchedule() {
    const bars = [], free = Array(SLOTS).fill(0);
    LENS.forEach((len, id) => {
      const s = free.indexOf(Math.min(...free));
      bars.push({ slot: s, start: free[s], len, id, pad: 0 });
      free[s] += len;
    });
    return bars;
  }
  const SCHED = { static: staticSchedule(), continuous: continuousSchedule() };

  function batchPanel(ctx, y, label, sub, bars, cursor, alpha) {
    const gx = 300, gw = 692, unit = gw / T_UNITS, rowH = 25, gap = 8;
    ctx.save(); ctx.globalAlpha *= alpha;
    rr(ctx, 48, y, 1184, 198, 16, C.panel, C.line, 1.5);
    text(ctx, label, 72, y + 44, { size: 22, weight: 700 });
    sub.forEach((s, i) => text(ctx, s, 72, y + 72 + i * 21, { size: 15, color: C.dim }));
    let used = 0, done = 0;
    for (let s = 0; s < SLOTS; s++) rr(ctx, gx, y + 20 + s * (rowH + gap), gw, rowH, 6, "#0D1526");
    bars.forEach((b) => {
      if (b.start >= T_UNITS) return;
      const by = y + 20 + b.slot * (rowH + gap);
      const vis = clamp(cursor - b.start, 0, Math.min(b.len, T_UNITS - b.start));
      if (vis > 0) rr(ctx, gx + b.start * unit + 1.5, by, vis * unit - 3, rowH, 6, PAL[b.id % PAL.length]);
      used += vis;
      if (cursor >= b.start + b.len) done++;
      const padVis = clamp(cursor - b.start - b.len, 0, Math.min(b.pad, T_UNITS - b.start - b.len));
      if (padVis > 0) {
        const px = gx + (b.start + b.len) * unit + 1.5, pw = padVis * unit - 3;
        rr(ctx, px, by, pw, rowH, 6, "#1B2740");
        ctx.save(); ctx.beginPath(); ctx.roundRect(px, by, pw, rowH, 6); ctx.clip();
        ctx.strokeStyle = "#31415F"; ctx.lineWidth = 2;
        for (let hx = px - rowH; hx < px + pw; hx += 9) { ctx.beginPath(); ctx.moveTo(hx, by + rowH); ctx.lineTo(hx + rowH, by); ctx.stroke(); }
        ctx.restore();
      }
    });
    if (cursor > 0 && cursor < T_UNITS) line(ctx, gx + cursor * unit, y + 12, gx + cursor * unit, y + 186, C.text, 2);
    const util = cursor > 0 ? used / (Math.min(cursor, T_UNITS) * SLOTS) : 0;
    stat(ctx, 1022, y + 78, String(done), "requests finished", { size: 40 });
    stat(ctx, 1022, y + 150, Math.round(util * 100) + "%", "slots doing useful work", { size: 30, color: util > 0.9 ? C.ok : C.amber });
    ctx.restore();
  }

  const batching = {
    dur: 11, step: 2, title: "Continuous batching",
    caption: "Finished sequences leave and waiting ones join at every decode step.",
    draw(ctx, p) {
      const cursor = lerp(0, T_UNITS, seg(p, 0.1, 0.8));
      batchPanel(ctx, 168, "Static batching", ["Run until the longest", "request finishes.", "Idle slots are padding."], SCHED.static, cursor, eo(p, 0, 0.08));
      batchPanel(ctx, 380, "Continuous batching", ["Re-form the batch", "every step. Free slots", "are refilled at once."], SCHED.continuous, cursor, eo(p, 0.04, 0.12));
      const a = eo(p, 0.84, 0.92);
      if (a > 0) {
        ctx.save(); ctx.globalAlpha *= a;
        rr(ctx, 300, 590, 680, 46, 23, hexA(A, 0.16), A, 1.5);
        text(ctx, `repo simulation, same GPU memory:  ${fmt(S.batching.static_tok_s)} → ${fmt(S.batching.continuous_tok_s)} tok/s`, 640, 620, { size: 19, weight: 600, align: "center" });
        ctx.restore();
      }
    },
  };

  // ---- 3. PagedAttention -------------------------------------------------------------
  const SEQ = [{ n: "A", len: 180 }, { n: "B", len: 950 }, { n: "C", len: 410 }, { n: "D", len: 2200 }];
  const COLS = 13, ROWS = 6, CELLS = COLS * ROWS;
  // deterministic scatter of physical block ids
  const ORDER = (() => {
    const ids = [...Array(CELLS).keys()]; let s = 7;
    for (let i = ids.length - 1; i > 0; i--) { s = (s * 1103515245 + 12345) % 2147483648; const j = s % (i + 1); [ids[i], ids[j]] = [ids[j], ids[i]]; }
    return ids;
  })();
  const ALLOC = (() => {  // which sequence takes the k-th allocated block (proportional to length)
    const want = [3, 16, 7, 37], got = [0, 0, 0, 0], out = [];
    while (out.length < 63) {
      let best = -1, score = -1;
      for (let i = 0; i < 4; i++) { const sc = (want[i] - got[i]) / want[i]; if (got[i] < want[i] && sc > score) { score = sc; best = i; } }
      got[best]++; out.push(best);
    }
    return out;
  })();

  const paged = {
    dur: 11, step: 3, title: "PagedAttention",
    caption: "KV cache in fixed-size pages: sequences grow block by block, anywhere in memory.",
    draw(ctx, p) {
      // left: contiguous reservations
      const la = eo(p, 0, 0.1);
      ctx.save(); ctx.globalAlpha *= la;
      rr(ctx, 48, 168, 540, 448, 16, C.panel, C.line, 1.5);
      text(ctx, "Contiguous reservation", 72, 210, { size: 22, weight: 700 });
      text(ctx, "every sequence reserves max length (4,096 tokens)", 72, 236, { size: 15, color: C.dim });
      SEQ.forEach((s, i) => {
        const y = 268 + i * 62, fill = eo(p, 0.08 + i * 0.03, 0.3 + i * 0.03) * (s.len / 4096);
        text(ctx, s.n, 72, y + 25, { size: 18, weight: 700, color: PAL[i] });
        rr(ctx, 100, y, 460, 36, 8, "#0D1526", "#31415F", 1.5);
        if (fill > 0) rr(ctx, 100, y, Math.max(8, 460 * fill), 36, 8, PAL[i]);
        text(ctx, `${fmt(s.len)} used`, 552, y + 56, { size: 13, color: C.faint, align: "right" });
      });
      const u = Math.round(S.paged.contiguous_util * 100);
      stat(ctx, 72, 572, `${Math.round(lerp(0, u, eo(p, 0.1, 0.34)))}%`, "of reserved KV memory holds tokens", { color: C.amber, size: 46 });
      ctx.restore();

      // right: paged
      const ra = eo(p, 0.3, 0.4);
      ctx.save(); ctx.globalAlpha *= ra;
      rr(ctx, 612, 168, 620, 448, 16, C.panel, A, 1.5);
      text(ctx, "Paged KV cache", 636, 210, { size: 22, weight: 700 });
      text(ctx, "16-token blocks + a block table per sequence", 636, 236, { size: 15, color: C.dim });
      const cs = 32, gx0 = 636, gy0 = 256, n = Math.floor(seg(p, 0.4, 0.86) * ALLOC.length);
      const table = [[], [], [], []];
      for (let i = 0; i < CELLS; i++) rr(ctx, gx0 + (i % COLS) * (cs + 4), gy0 + Math.floor(i / COLS) * (cs + 4), cs, cs, 6, "#0D1526", "#22304C", 1);
      for (let k = 0; k < n; k++) {
        const cell = ORDER[k], q = ALLOC[k], fresh = k === n - 1;
        table[q].push(cell);
        rr(ctx, gx0 + (cell % COLS) * (cs + 4), gy0 + Math.floor(cell / COLS) * (cs + 4), cs, cs, 6, PAL[q], fresh ? C.text : null, 2);
      }
      SEQ.forEach((s, i) => {
        const y = 490 + i * 22;
        text(ctx, `${s.n} →`, 636, y, { size: 14, weight: 700, color: PAL[i], mono: true });
        const ids = table[i].slice(-9).map((v) => String(v).padStart(2, "0")).join(" ");
        text(ctx, (table[i].length > 9 ? "… " : "") + ids, 676, y, { size: 14, mono: true, color: C.dim });
      });
      const pu = Math.round(S.paged.paged_util * 100);
      stat(ctx, 1208, 532, `${Math.round(lerp(0, pu, eo(p, 0.45, 0.86)))}%`, "of allocated blocks hold tokens", { color: C.ok, size: 46, align: "right" });
      text(ctx, "free blocks → room for more sequences", 1208, 590, { size: 14, color: C.faint, align: "right", alpha: eo(p, 0.8, 0.9) });
      ctx.restore();
    },
  };

  // ---- 4. decode --------------------------------------------------------------------
  const ROWLEN = [6, 10, 4, 8, 5, 9];
  const decode = {
    dur: 7, step: 4, title: "Decode tokens",
    caption: "One step reads the weights once and yields one token for every sequence in the batch.",
    draw(ctx, p) {
      const steps = 11, cur = seg(p, 0.12, 0.92) * steps, k = Math.floor(cur), frac = cur - k;
      const gpu = box(ctx, 60, 250, 250, 240, "GPU", { color: A, sub: "optimized kernels", glow: 0.4, size: 30, alpha: eo(p, 0, 0.1) });
      text(ctx, "weights read from HBM", gpu.cx, 524, { size: 14, color: C.dim, align: "center" });
      bar(ctx, 82, 538, 206, 12, k < steps ? frac : 1, A);
      text(ctx, `decode step ${Math.min(k + 1, steps)}`, gpu.cx, 586, { size: 17, weight: 600, align: "center", mono: true });
      arrow(ctx, 322, 370, 392, 370, { color: A, prog: eo(p, 0.05, 0.15) });
      ROWLEN.forEach((len, r) => {
        const y = 196 + r * 66;
        user(ctx, 428, y + 22, 0.85, PAL[r], eo(p, 0.03 * r, 0.12 + 0.03 * r));
        for (let c = 0; c < steps; c++) {
          const second = c > len;           // slot refilled by a new sequence
          if (c === len) {
            const a = eo(cur, c, c + 0.6);
            text(ctx, "✓", 470 + c * 66 + 28, y + 30, { size: 22, weight: 800, color: C.ok, align: "center", alpha: a });
            continue;
          }
          const a = eo(cur, c, c + 0.5);
          const col = second ? PAL[(r + 3) % PAL.length] : PAL[r];
          tokenBox(ctx, 470 + c * 66, y + 4 + (1 - a) * 8, 56, 36, "T" + (second ? c - len : c + 1), col, a);
        }
      });
      text(ctx, "✓ finished → slot refilled by a waiting request", 1220, 606, { size: 15, color: C.dim, align: "right", alpha: eo(p, 0.55, 0.7) });
    },
  };

  // ---- 5. stream --------------------------------------------------------------------
  const stream = {
    dur: 7, step: 5, title: "Stream response",
    caption: "Tokens stream back as they are generated, to every user in the batch.",
    draw(ctx, p, t) {
      const srv = box(ctx, 60, 250, 230, 170, "vLLM", { color: A, sub: "SSE stream", glow: 0.4, size: 28 });
      for (let r = 0; r < 6; r++) {
        const y = 196 + r * 44;
        line(ctx, srv.r + 12, srv.cy + (r - 2.5) * 12, 720, y + 14, "#1C2942", 2);
        user(ctx, 760, y + 14, 0.8, PAL[r]);
        for (let k = 0; k < 14; k++) {
          const q = ((t * 0.55 + k * 0.19 + r * 0.07) % 2.6) / 1.0;
          if (q > 1) continue;
          const x = lerp(srv.r + 12, 706, q), yy = lerp(srv.cy + (r - 2.5) * 12, y + 14, q);
          rr(ctx, x - 9, yy - 5, 18, 10, 4, PAL[r]);
        }
      }
      const a1 = eo(p, 0.2, 0.35), a2 = eo(p, 0.4, 0.55);
      ctx.save(); ctx.globalAlpha *= a1;
      rr(ctx, 850, 190, 382, 190, 16, C.panel, C.line, 1.5);
      text(ctx, "MEAN TIME TO FIRST TOKEN", 876, 226, { size: 13, weight: 700, color: C.dim, spacing: 1.5 });
      text(ctx, `${(S.batching.static_ttft_ms / 1000).toFixed(1)} s`, 876, 282, { size: 38, weight: 750, color: C.amber });
      text(ctx, "static", 876, 306, { size: 14, color: C.dim });
      arrow(ctx, 1010, 268, 1058, 268, { color: C.dim });
      text(ctx, `${S.batching.continuous_ttft_ms} ms`, 1078, 282, { size: 38, weight: 750, color: C.ok });
      text(ctx, "continuous", 1078, 306, { size: 14, color: C.dim });
      text(ctx, "same requests, same KV memory", 876, 352, { size: 14, color: C.faint });
      ctx.restore();
      ctx.save(); ctx.globalAlpha *= a2;
      rr(ctx, 850, 400, 382, 190, 16, C.panel, C.line, 1.5);
      text(ctx, "THROUGHPUT", 876, 436, { size: 13, weight: 700, color: C.dim, spacing: 1.5 });
      const max = S.batching.continuous_tok_s;
      text(ctx, "static", 876, 474, { size: 14, color: C.dim });
      bar(ctx, 966, 462, 240, 14, (S.batching.static_tok_s / max) * eo(p, 0.45, 0.7), C.amber);
      text(ctx, "continuous", 876, 510, { size: 14, color: C.dim });
      bar(ctx, 966, 498, 240, 14, eo(p, 0.5, 0.8), C.ok);
      text(ctx, `${fmt(S.batching.static_tok_s)} → ${fmt(S.batching.continuous_tok_s)} tok/s`, 876, 562, { size: 24, weight: 700 });
      ctx.restore();
      text(ctx, "numbers from this repo's scheduler simulation (ie simulate batching)", 60, 606, { size: 14, color: C.faint, alpha: a2 });
    },
  };

  window.VIDEOS.vllm = {
    id: "vllm", title: "vLLM", accent: A, steps: 5,
    scenes: [
      K.titleScene("vLLM", "High-throughput serving for production LLMs", ["Continuous batching", "PagedAttention", "OpenAI-compatible API"]),
      users, batching, paged, decode, stream,
      K.summaryScene({
        when: ["You are building a production API and care about concurrency, throughput", "and GPU utilization — without deep hardware-specific tuning."],
        best: ["High-throughput production serving", "Many concurrent users", ["Large open models", "Llama, Mistral, Qwen…"], "OpenAI-compatible API"],
        pros: ["Continuous batching", ["Memory efficient", "PagedAttention"], "Long-context support", "Active open-source community"],
        cons: ["More setup than Ollama", ["Focus on serving", "not every last % of tuning"], "Best on NVIDIA GPUs"],
      }),
    ],
  };
})();
