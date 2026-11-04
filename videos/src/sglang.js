// SGLang: programs -> RadixAttention -> cache-aware scheduling -> speculative decoding -> structured output.
(function () {
  const { C, W, clamp, lerp, seg, eo, eio, hexA, rr, text, line, arrow, box, tokenBox, bar, stat, fmt, MONO } = K;
  const A = C.purple;
  const PAL = [C.blue, C.teal, C.amber, C.pink];
  const S = window.SIM;

  // ---- 1. requests are programs ------------------------------------------------------
  const CODE = [
    "@sgl.function",
    "def handle_ticket(s, ticket):",
    "    s += sgl.user(ticket)",
    "    s += sgl.assistant(",
    "        sgl.select(\"team\", choices=TEAMS))",
    "    drafts = s.fork(3)",
    "    for d in drafts:",
    "        d += sgl.assistant(sgl.gen(\"reply\"))",
    "    s += sgl.assistant(",
    "        sgl.gen(\"best\", regex=r\"[012]\"))",
  ];
  const TOTAL = CODE.reduce((n, l) => n + l.length, 0);
  const TOKEN = /(@?sgl\.\w+|"[^"]*"|\bdef\b|\bfor\b|\bin\b)/g;
  function codeLine(ctx, str, x, y) {
    let cx = x;
    str.split(TOKEN).forEach((part) => {
      if (!part) return;
      const color = /^@?sgl\./.test(part) ? A : part[0] === '"' ? C.ok : /^(def|for|in)$/.test(part) ? C.amber : C.text;
      cx += text(ctx, part, cx, y, { size: 19, mono: true, color });
    });
  }
  // flow node appears once its code line (index) has been fully typed
  const program = {
    dur: 10, step: 1, title: "Requests are programs",
    caption: "gen, select, fork, constraints: the runtime sees the whole program, not one prompt.",
    draw(ctx, p) {
      rr(ctx, 48, 168, 640, 448, 16, "#0C1322", C.line, 1.5);
      ["#FF5F57", "#FEBC2E", "#28C840"].forEach((c, i) => { ctx.fillStyle = c; ctx.beginPath(); ctx.arc(74 + i * 20, 192, 6, 0, 7); ctx.fill(); });
      text(ctx, "agent_program.py", 368, 197, { size: 14, color: C.faint, align: "center", mono: true });
      let budget = Math.round(seg(p, 0.04, 0.68) * TOTAL);
      const typedLines = [];
      CODE.forEach((l, i) => {
        const n = clamp(budget, 0, l.length); budget -= n;
        typedLines.push(n === l.length);
        text(ctx, String(i + 1).padStart(2, " "), 70, 246 + i * 35, { size: 15, mono: true, color: C.faint });
        if (n > 0) codeLine(ctx, l.slice(0, n), 108, 246 + i * 35);
      });
      const node = (lineIdx, x, y, w, label, color, sub) => {
        const a = typedLines[lineIdx] ? 1 : 0;
        if (!a) return null;
        return box(ctx, x, y, w, 52, label, { color, size: 17, sub, fill: C.panel });
      };
      const cx = 968;
      const n1 = node(2, cx - 110, 176, 220, "user: ticket", C.dim);
      const n2 = node(4, cx - 110, 262, 220, "select  team", A);
      if (n1 && n2) arrow(ctx, cx, n1.b + 4, cx, n2.y - 6, { color: C.dim });
      const fx = [cx - 250, cx - 75, cx + 100];
      const forks = fx.map((x, i) => node(7, x, 372, 150, `gen reply ${i + 1}`, PAL[i]));
      if (typedLines[5] && n2) fx.forEach((x) => arrow(ctx, cx, n2.b + 4, x + 75, 366, { color: A }));
      if (typedLines[5]) {
        text(ctx, "fork(3)", 1092, 284, { size: 15, weight: 700, color: A, mono: true });
        text(ctx, "branches share", 1092, 304, { size: 13, color: C.dim });
        text(ctx, "all KV so far", 1092, 321, { size: 13, color: C.dim });
      }
      const n4 = node(9, cx - 110, 486, 220, "gen best", A, "regex [012]");
      if (n4 && forks[0]) fx.forEach((x) => arrow(ctx, x + 75, 428, cx, n4.y - 6, { color: C.dim }));
      const out = eo(p, 0.74, 0.86);
      if (out > 0 && n4) {
        arrow(ctx, cx, n4.b + 4, cx, 570, { color: C.ok, prog: out });
        text(ctx, "structured result", cx, 596, { size: 17, weight: 600, color: C.ok, align: "center", alpha: out });
      }
    },
  };

  // ---- 2. RadixAttention -------------------------------------------------------------
  const radix = {
    dur: 12, step: 2, title: "RadixAttention",
    caption: "A radix tree of KV cache: any shared prefix is computed once and reused.",
    draw(ctx, p) {
      // incoming request bar
      const segs = [
        { w: 560, label: "system prompt + tools · 1,500 tokens", key: 0 },
        { w: 330, label: "session B history", key: 1 },
        { w: 190, label: "new turn", key: 2 },
      ];
      const reqA = eo(p, 0.34, 0.42);
      text(ctx, "incoming request (session B, turn 3)", 60, 190, { size: 15, color: C.dim, alpha: reqA });
      let x = 60;
      const match = seg(p, 0.44, 0.62);          // how far the prefix match has walked
      segs.forEach((sgm, i) => {
        const matched = i < 2 && match > (i + 1) / 2 - 0.001;
        const isNew = i === 2;
        const color = matched ? C.ok : isNew && match >= 1 ? A : C.dim;
        ctx.save(); ctx.globalAlpha *= reqA;
        rr(ctx, x, 202, sgm.w, 40, 8, hexA(color, matched || (isNew && match >= 1) ? 0.25 : 0.08), color, 1.8);
        text(ctx, sgm.label, x + sgm.w / 2, 228, { size: 15, weight: 600, align: "center" });
        if (matched) text(ctx, "cached ✓", x + sgm.w - 12, 196, { size: 13, weight: 700, color: C.ok, align: "right" });
        if (isNew && match >= 1) text(ctx, "prefill only this", x + sgm.w - 4, 196, { size: 13, weight: 700, color: A, align: "right" });
        ctx.restore();
        x += sgm.w + 10;
      });

      // the tree
      const root = { x: 640, y: 292 };
      const sys = { x: 380, y: 318, w: 520, h: 50 };
      const kids = [{ n: "A", x: 150 }, { n: "B", x: 520 }, { n: "C", x: 890 }];
      const sa = eo(p, 0.03, 0.12);
      ctx.save(); ctx.globalAlpha *= sa;
      const hot0 = match > 0.02;
      rr(ctx, sys.x, sys.y, sys.w, sys.h, 12, hexA(hot0 ? C.ok : A, 0.18), hot0 ? C.ok : A, 2);
      text(ctx, "system prompt + tools  ·  1,500 tokens", 640, sys.y + 31, { size: 17, weight: 650, align: "center" });
      ctx.restore();
      kids.forEach((k, i) => {
        const a = eo(p, 0.1 + i * 0.06, 0.2 + i * 0.06);
        const hot = k.n === "B" && match > 0.5;
        const col = hot ? C.ok : PAL[i];
        arrow(ctx, 640, sys.y + sys.h + 2, k.x + 120, 412, { color: hot ? C.ok : C.faint, prog: a, width: hot ? 3.5 : 2 });
        ctx.save(); ctx.globalAlpha *= a;
        rr(ctx, k.x, 418, 240, 46, 12, hexA(col, 0.16), col, 1.8);
        text(ctx, `session ${k.n} · turns 1–2`, k.x + 120, 447, { size: 16, weight: 600, align: "center" });
        ctx.restore();
      });
      // new leaf under B
      const leaf = eo(p, 0.64, 0.76);
      if (leaf > 0) {
        arrow(ctx, 640, 468, 640, 506, { color: A, prog: leaf, width: 3 });
        ctx.save(); ctx.globalAlpha *= leaf;
        rr(ctx, 520, 512, 240, 46, 12, hexA(A, 0.22), A, 2);
        text(ctx, "turn 3 · 120 new tokens", 640, 541, { size: 16, weight: 600, align: "center" });
        ctx.restore();
      }
      const ra = eo(p, 0.8, 0.9);
      if (ra > 0) {
        ctx.save(); ctx.globalAlpha *= ra;
        rr(ctx, 250, 580, 780, 50, 25, hexA(A, 0.16), A, 1.5);
        text(ctx, `8 agent sessions × 5 turns:  ${fmt(S.radix.prompt_tokens)} → ${fmt(S.radix.prefilled_tokens)} prefill tokens  (${S.radix.prefill_reduction_x.toFixed(1)}× less)`, 640, 612, { size: 18, weight: 600, align: "center" });
        ctx.restore();
      }
    },
  };

  // ---- 3. cache-aware scheduling ------------------------------------------------------
  const ARRIVAL = [0, 0, 1, 2, 3, 1, 0, 3, 2, 2, 0, 1, 3, 2, 0, 3];
  const LPM = (() => {                       // greedy: serve whatever matches the cached prefix first
    const left = ARRIVAL.map((g, i) => ({ g, i })), out = []; let cached = -1;
    while (left.length) {
      let k = left.findIndex((r) => r.g === cached);
      if (k < 0) k = 0;
      cached = left[k].g; out.push(left.splice(k, 1)[0]);
    }
    return out;
  })();
  function lane(ctx, y, title, sub, order, pos, cursor, alpha) {
    ctx.save(); ctx.globalAlpha *= alpha;
    rr(ctx, 48, y, 1184, 196, 16, C.panel, C.line, 1.5);
    text(ctx, title, 72, y + 42, { size: 21, weight: 700 });
    text(ctx, sub, 72, y + 66, { size: 15, color: C.dim });
    let hits = 0, cached = -1, n = Math.floor(cursor);
    order.forEach((r, k) => {
      const x = pos(r, k), yy = y + 92;
      const served = k < n, hit = r.g === cached;
      if (served) { if (hit) hits++; cached = r.g; }
      rr(ctx, x, yy, 50, 50, 10, hexA(PAL[r.g], served ? 0.9 : 0.28), PAL[r.g], 1.5);
      if (served) text(ctx, hit ? "✓" : "✗", x + 25, yy + 80, { size: 20, weight: 800, color: hit ? C.ok : C.red, align: "center" });
      if (k === n && cursor < order.length) rr(ctx, x - 4, yy - 4, 58, 58, 13, null, C.text, 2.5);
    });
    const rate = n ? hits / n : 0;
    rr(ctx, 1010, y + 26, 30, 30, 8, cached < 0 ? "#0D1526" : PAL[cached], C.text, 1.5);
    text(ctx, "prefix in cache", 1052, y + 47, { size: 14, color: C.dim });
    stat(ctx, 1010, y + 126, Math.round(rate * 100) + "%", "cache hit rate", { size: 44, color: rate > 0.5 ? C.ok : C.amber });
    ctx.restore();
  }
  const sched = {
    dur: 11, step: 3, title: "Cache-aware scheduling",
    caption: "Serve requests whose prefix is already cached first, before it gets evicted.",
    draw(ctx, p) {
      const cursor = seg(p, 0.3, 0.8) * 16;
      const slot = (k) => 72 + k * 57;
      lane(ctx, 168, "First come, first served", "arrival order · 4 prompt groups · cache holds one prefix", ARRIVAL.map((g, i) => ({ g, i })), (r, k) => slot(k), cursor, eo(p, 0, 0.08));
      const move = eio(p, 0.12, 0.28);
      lane(ctx, 378, "Longest prefix match  (SGLang default)", "same requests, reordered so cached prefixes are used first", LPM, (r, k) => lerp(slot(r.i), slot(k), move), cursor, eo(p, 0.04, 0.12));
      const a = eo(p, 0.84, 0.92);
      if (a > 0) {
        ctx.save(); ctx.globalAlpha *= a;
        rr(ctx, 270, 586, 740, 48, 24, hexA(A, 0.16), A, 1.5);
        text(ctx, `repo simulation under memory pressure:  ${(S.prefix.fcfs_hit * 100).toFixed(1)}% → ${(S.prefix.lpm_hit * 100).toFixed(1)}% hit rate`, 640, 617, { size: 19, weight: 600, align: "center" });
        ctx.restore();
      }
    },
  };

  // ---- 4. speculative decoding --------------------------------------------------------
  const ROUNDS = [
    { draft: ["The", "KV", "cache", "holds"], ok: 3, fix: "stores" },
    { draft: ["keys", "and", "values", "for"], ok: 4, bonus: "every" },
    { draft: ["past", "word", "in", "a"], ok: 1, fix: "token" },
  ];
  const spec = {
    dur: 12, step: 4, title: "Speculative decoding",
    caption: "A cheap draft guesses several tokens; the target model verifies them in one pass.",
    draw(ctx, p) {
      const d = box(ctx, 60, 176, 250, 84, "Draft head", { color: C.teal, sub: "small, fast, sometimes wrong", size: 19 });
      const tg = box(ctx, 970, 176, 250, 84, "Target model", { color: A, sub: "one pass verifies k tokens", size: 19 });
      const cur = seg(p, 0.06, 0.9) * ROUNDS.length, ri = Math.min(ROUNDS.length - 1, Math.floor(cur)), q = cur - ri;
      let col = 0, passes = 0, tokens = 0;
      const sx = 66, step = 105, tw = 96;
      text(ctx, "DRAFT PROPOSES", 60, 318, { size: 13, weight: 700, color: C.teal, spacing: 1.5 });
      text(ctx, "ACCEPTED OUTPUT", 60, 478, { size: 13, weight: 700, color: C.ok, spacing: 1.5 });
      ROUNDS.forEach((r, i) => {
        const local = i < ri ? 1 : i === ri ? (cur >= ROUNDS.length ? 1 : q) : 0;
        const produced = r.ok + 1;                      // accepted + corrected-or-bonus
        if (local > 0) {
          const showDraft = i === ri && local < 0.92;
          r.draft.forEach((w, j) => {
            const da = eo(local, j * 0.07, 0.1 + j * 0.07);
            const verified = local > 0.42 + j * 0.06;
            const good = j < r.ok, dead = !good && j > r.ok;
            if (showDraft) {
              const x = sx + (col + j) * step;
              const c = verified ? (good ? C.ok : C.red) : C.teal;
              tokenBox(ctx, x, 332, tw, 40, w, c, da * (dead && verified ? 0.3 : 1), { outline: !verified });
              if (verified && !dead) text(ctx, good ? "✓" : "✗", x + tw / 2, 406, { size: 22, weight: 800, color: good ? C.ok : C.red, align: "center" });
            }
          });
          if (i === ri && local > 0.36 && local < 0.92) {
            const sweep = seg(local, 0.36, 0.7);
            rr(ctx, sx + col * step - 8, 322, 4 * step - 8 + 8, 60, 12, null, hexA(A, 0.9 * (1 - Math.abs(sweep - 0.5) * 1.2)), 2.5);
            arrow(ctx, tg.cx, tg.b + 6, sx + col * step + 2 * step, 316, { color: hexA(A, 0.7), prog: seg(local, 0.3, 0.42) });
          }
          // accepted tokens settle into the output row
          for (let j = 0; j < produced; j++) {
            const a = eo(local, 0.66 + j * 0.05, 0.8 + j * 0.05);
            const isFix = j === r.ok;
            const label = isFix ? (r.fix || r.bonus) : r.draft[j];
            const c = isFix ? (r.fix ? C.amber : A) : C.ok;
            tokenBox(ctx, sx + (col + j) * step, 492 + (1 - a) * -30, tw, 40, label, c, a);
          }
          if (local > 0.66) { passes++; tokens += produced; }
        }
        col += produced;
      });
      text(ctx, "■ accepted draft token", 60, 574, { size: 14, color: C.ok });
      text(ctx, "■ corrected by the target", 250, 574, { size: 14, color: C.amber });
      text(ctx, "■ free bonus token", 470, 574, { size: 14, color: A });
      stat(ctx, 640, 222, passes ? (tokens / passes).toFixed(1) : "–", `tokens per target pass  (${tokens} tokens, ${passes} passes)`, { align: "center", color: A, size: 46 });
      const a = eo(p, 0.92, 0.97);
      text(ctx, `repo simulation: ${S.speculative.tokens_per_pass} tokens per pass at ${Math.round(S.speculative.acceptance * 100)}% acceptance — output distribution unchanged`, 640, 618, { size: 17, weight: 600, align: "center", alpha: a });
    },
  };

  // ---- 5. structured output -----------------------------------------------------------
  // f = forced by the grammar (jump-forward), m = chosen by the model (one forward pass each)
  const PARTS = [
    ["f", '{"customer": "'], ["m", "Dana"], ["f", '", "priority": '], ["m", "2"],
    ["f", ', "category": "'], ["m", "bi"], ["f", 'lling", "escalate": '], ["m", "t"], ["f", "rue}"],
  ];
  const CHARS = PARTS.flatMap(([k, s]) => [...s].map((ch) => ({ k, ch })));
  // timeline: a model char costs 1 beat, a forced run costs 1 beat in total
  const BEATS = (() => {
    let t = 0; const out = [];
    PARTS.forEach(([k, s]) => { [...s].forEach((_, j) => out.push(k === "m" ? t + j : t)); t += k === "m" ? s.length : 1; });
    return { at: out, total: t };
  })();
  const structured = {
    dur: 10, step: 5, title: "Structured output, with jump-forward",
    caption: "The grammar masks invalid tokens; forced text is appended without running the model.",
    draw(ctx, p) {
      rr(ctx, 48, 168, 430, 250, 16, "#0C1322", C.line, 1.5);
      text(ctx, "JSON SCHEMA", 72, 202, { size: 13, weight: 700, color: C.dim, spacing: 1.5 });
      ['customer:  string', 'priority:  integer', 'category:  "billing" | "bug" |', '           "feature_request"', 'escalate:  boolean'].forEach((l, i) =>
        text(ctx, l, 72, 240 + i * 32, { size: 17, mono: true, color: i === 3 ? C.dim : C.text }));
      const beat = seg(p, 0.08, 0.84) * (BEATS.total + 0.5);
      const model = box(ctx, 560, 196, 250, 100, "Model", { color: A, sub: "forward pass", size: 22, glow: 0 });
      const gram = box(ctx, 900, 196, 300, 100, "Grammar FSM", { color: C.teal, sub: "mask + jump-forward", size: 22 });
      let calls = 0, forced = 0, shown = 0, lastKind = null;
      const x0 = 66, cw = 14.6, y = 500;
      rr(ctx, 48, 448, 1184, 96, 16, C.panel, C.line, 1.5);
      CHARS.forEach((c, i) => {
        if (BEATS.at[i] > beat) return;
        shown++; lastKind = c.k;
        if (c.k === "m") calls++; else forced++;
        const fresh = beat - BEATS.at[i] < 0.6;
        text(ctx, c.ch, x0 + i * cw, y + 8, { size: 24, mono: true, weight: 600, color: c.k === "f" ? C.teal : C.text });
        rr(ctx, x0 + i * cw + 1, y + 20, cw - 3, 4, 2, c.k === "f" ? hexA(C.teal, 0.7) : A);
        if (fresh && c.k === "m") rr(ctx, x0 + i * cw - 3, y - 20, cw + 5, 36, 6, null, A, 2);
      });
      if (shown < CHARS.length && lastKind) {
        const active = lastKind === "m" ? model : gram;
        rr(ctx, active.x - 5, active.y - 5, active.w + 10, active.h + 10, 18, null, lastKind === "m" ? A : C.teal, 3);
      }
      text(ctx, "■ chosen by the model (1 forward pass each)", 66, 584, { size: 15, color: C.text });
      text(ctx, "■ forced by the schema (no forward pass)", 470, 584, { size: 15, color: C.teal });
      stat(ctx, 560, 368, String(calls), "model calls", { size: 44, color: A });
      stat(ctx, 760, 368, String(shown), "characters emitted", { size: 44 });
      const a = eo(p, 0.88, 0.95);
      text(ctx, `repo FSM, same schema with a random model:  ${S.constrained.calls_plain} → ${S.constrained.calls_jump} model calls — always valid JSON`, 1220, 618, { size: 16, weight: 600, align: "right", alpha: a });
    },
  };

  window.VIDEOS.sglang = {
    id: "sglang", title: "SGLang", accent: A, steps: 5,
    scenes: [
      K.titleScene("SGLang", "Flexible, high-performance LLM serving with programmable control", ["Inference programs", "RadixAttention", "Structured output"]),
      program, radix, sched, spec, structured,
      K.summaryScene({
        when: ["Your application starts behaving more like an inference program", "than a simple model endpoint: agents, tools, branches, schemas."],
        best: ["Agents and tool-use workflows", ["Complex generation control", "branching, multi-step"], "Prefix-heavy, multi-call workloads", "Structured outputs in production"],
        pros: ["Flexible programmable interface", ["Fast on complex workloads", "prefix reuse + scheduling"], "Parallel tool calls / branching", "Good for agentic applications"],
        cons: ["Steeper learning curve", "Newer ecosystem than vLLM", ["Primarily optimized", "for NVIDIA GPUs"]],
      }),
    ],
  };
})();
