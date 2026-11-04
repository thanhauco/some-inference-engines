// Motion kit: a tiny deterministic timeline + canvas drawing helpers.
//
// A video is { id, title, accent, scenes: [{ dur, step, title, caption, draw(ctx, p, t) }] }.
// Every frame is a pure function of time (drawFrame(video, seconds)), so the same code
// plays live in the browser and renders frame-exact to MP4 in headless Chromium.
(function () {
  const W = 1280, H = 720;
  const C = {
    bg: "#0A0F1C", panel: "#111A2E", panel2: "#16223B", line: "#26344F",
    text: "#E8EEF9", dim: "#8EA0BF", faint: "#52637F",
    blue: "#4C8DFF", purple: "#9B7BFF", green: "#76C043", teal: "#22C7C7",
    amber: "#FFB547", red: "#FF6B6B", ok: "#4ADE80", pink: "#F472B6",
  };
  const SANS = '"Inter", system-ui, -apple-system, "Segoe UI", sans-serif';
  const MONO = '"JetBrains Mono", ui-monospace, "SF Mono", Menlo, monospace';

  const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
  const lerp = (a, b, t) => a + (b - a) * t;
  // seg(p, a, b): progress of the sub-interval [a, b] of p, clamped to 0..1.
  const seg = (p, a, b) => clamp((p - a) / (b - a));
  const ease = {
    out: (t) => 1 - Math.pow(1 - clamp(t), 3),
    inOut: (t) => { t = clamp(t); return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; },
    back: (t) => { t = clamp(t); const c = 1.70158; return 1 + (c + 1) * Math.pow(t - 1, 3) + c * Math.pow(t - 1, 2); },
  };
  const eo = (p, a, b) => ease.out(seg(p, a, b));
  const eio = (p, a, b) => ease.inOut(seg(p, a, b));

  function hexA(hex, a) {
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a})`;
  }

  function rr(ctx, x, y, w, h, r, fill, stroke, lw = 1.5) {
    r = Math.min(r, Math.abs(w) / 2, Math.abs(h) / 2);
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, r);
    if (fill) { ctx.fillStyle = fill; ctx.fill(); }
    if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lw; ctx.stroke(); }
  }

  function text(ctx, str, x, y, o = {}) {
    const { size = 20, weight = 500, color = C.text, align = "left", mono = false,
            alpha = 1, baseline = "alphabetic", spacing = 0 } = o;
    if (alpha <= 0) return 0;
    ctx.save();
    ctx.globalAlpha *= alpha;
    ctx.font = `${weight} ${size}px ${mono ? MONO : SANS}`;
    ctx.fillStyle = color;
    ctx.textAlign = align;
    ctx.textBaseline = baseline;
    if (spacing) ctx.letterSpacing = `${spacing}px`;
    ctx.fillText(str, x, y);
    const w = ctx.measureText(str).width;
    ctx.restore();
    return w;
  }

  function line(ctx, x1, y1, x2, y2, color = C.line, width = 2, dash) {
    ctx.save();
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineCap = "round";
    if (dash) ctx.setLineDash(dash);
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    ctx.restore();
  }

  // Arrow drawn up to `prog` of its length (so arrows can "grow").
  function arrow(ctx, x1, y1, x2, y2, o = {}) {
    const { color = C.dim, width = 2.5, prog = 1, head = 9, dash } = o;
    if (prog <= 0) return;
    const ex = lerp(x1, x2, prog), ey = lerp(y1, y2, prog);
    line(ctx, x1, y1, ex, ey, color, width, dash);
    const a = Math.atan2(y2 - y1, x2 - x1);
    ctx.save();
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(ex + Math.cos(a) * head * 0.6, ey + Math.sin(a) * head * 0.6);
    ctx.lineTo(ex - Math.cos(a - 0.45) * head, ey - Math.sin(a - 0.45) * head);
    ctx.lineTo(ex - Math.cos(a + 0.45) * head, ey - Math.sin(a + 0.45) * head);
    ctx.closePath(); ctx.fill();
    ctx.restore();
  }

  function user(ctx, x, y, s = 1, color = C.text, alpha = 1) {
    ctx.save();
    ctx.globalAlpha *= alpha;
    ctx.strokeStyle = color; ctx.lineWidth = 2.6 * s; ctx.lineCap = "round";
    ctx.beginPath(); ctx.arc(x, y - 11 * s, 7.5 * s, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath(); ctx.arc(x, y + 17 * s, 14 * s, Math.PI * 1.08, Math.PI * 1.92); ctx.stroke();
    ctx.restore();
  }

  // A labelled box (model, GPU, server...). Returns its geometry for wiring arrows.
  function box(ctx, x, y, w, h, label, o = {}) {
    const { color = C.blue, fill = C.panel, sub, alpha = 1, size = 20, glow = 0 } = o;
    ctx.save();
    ctx.globalAlpha *= alpha;
    if (glow > 0) { ctx.shadowColor = hexA(color, 0.7 * glow); ctx.shadowBlur = 28 * glow; }
    rr(ctx, x, y, w, h, 14, fill, color, 2);
    ctx.shadowBlur = 0;
    text(ctx, label, x + w / 2, y + h / 2 + (sub ? -6 : 7), { size, weight: 650, align: "center" });
    if (sub) text(ctx, sub, x + w / 2, y + h / 2 + 18, { size: 14, color: C.dim, align: "center" });
    ctx.restore();
    return { x, y, w, h, cx: x + w / 2, cy: y + h / 2, r: x + w, b: y + h };
  }

  function tokenBox(ctx, x, y, w, h, label, color, alpha = 1, o = {}) {
    if (alpha <= 0) return;
    ctx.save();
    ctx.globalAlpha *= alpha;
    rr(ctx, x, y, w, h, 6, o.outline ? null : hexA(color, 0.22), color, 1.6);
    if (label) text(ctx, label, x + w / 2, y + h / 2 + 5, { size: o.size || 14, weight: 600, align: "center", mono: true, color: o.textColor || C.text });
    ctx.restore();
  }

  function bar(ctx, x, y, w, h, frac, color, bg = C.panel2) {
    rr(ctx, x, y, w, h, h / 2, bg);
    if (frac > 0.001) rr(ctx, x, y, Math.max(h, w * clamp(frac)), h, h / 2, color);
  }

  // Big number with a label under it - the "result" of a scene.
  function stat(ctx, x, y, value, label, o = {}) {
    const { color = C.text, alpha = 1, size = 44, align = "left" } = o;
    text(ctx, value, x, y, { size, weight: 750, color, alpha, align, spacing: -1 });
    text(ctx, label, x, y + 26, { size: 15, color: C.dim, alpha, align });
  }

  function typed(str, p) { return str.slice(0, Math.round(str.length * clamp(p))); }
  const fmt = (n) => Math.round(n).toLocaleString("en-US");

  // ---- frame composition -------------------------------------------------------------
  function background(ctx, accent) {
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);
    const g = ctx.createRadialGradient(W * 0.82, -80, 40, W * 0.82, -80, 760);
    g.addColorStop(0, hexA(accent, 0.20));
    g.addColorStop(1, hexA(accent, 0));
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, W, H);
    ctx.save();
    ctx.globalAlpha = 0.5;
    for (let x = 40; x < W; x += 40) for (let y = 40; y < H; y += 40) {
      ctx.fillStyle = "#1A2540";
      ctx.fillRect(x, y, 1.5, 1.5);
    }
    ctx.restore();
  }

  function chrome(ctx, video, sceneIdx, scene, T, total) {
    const accent = video.accent;
    // engine pill
    ctx.save();
    ctx.font = `700 16px ${SANS}`;
    const pw = ctx.measureText(video.title).width + 44;
    rr(ctx, 48, 34, pw, 34, 17, hexA(accent, 0.16), accent, 1.5);
    ctx.fillStyle = accent; ctx.beginPath(); ctx.arc(66, 51, 5, 0, Math.PI * 2); ctx.fill();
    text(ctx, video.title, 80, 57, { size: 16, weight: 700 });
    ctx.restore();
    // step dots
    const steps = video.steps || 0;
    for (let i = 1; i <= steps; i++) {
      const x = W - 48 - (steps - i) * 26, on = scene.step === i, done = scene.step > i;
      ctx.fillStyle = on ? accent : done ? hexA(accent, 0.45) : C.line;
      ctx.beginPath(); ctx.arc(x, 51, on ? 7 : 5, 0, Math.PI * 2); ctx.fill();
    }
    // step heading
    if (scene.title) {
      const a = scene._in;
      if (scene.step) {
        ctx.save(); ctx.globalAlpha = a;
        ctx.fillStyle = accent; ctx.beginPath(); ctx.arc(70, 122, 20, 0, Math.PI * 2); ctx.fill();
        text(ctx, String(scene.step), 70, 130, { size: 22, weight: 800, align: "center", color: "#0A0F1C" });
        ctx.restore();
      }
      text(ctx, scene.title, scene.step ? 104 : 50, 132, { size: 30, weight: 700, alpha: a, spacing: -0.5 });
    }
    if (scene.caption) {
      text(ctx, scene.caption, W / 2, 668, { size: 21, weight: 500, color: C.dim, align: "center", alpha: scene._in });
    }
    // progress
    ctx.fillStyle = C.line; ctx.fillRect(0, H - 4, W, 4);
    ctx.fillStyle = accent; ctx.fillRect(0, H - 4, W * clamp(T / total), 4);
  }

  function duration(video) { return video.scenes.reduce((s, sc) => s + sc.dur, 0); }

  function drawFrame(ctx, video, T) {
    const total = duration(video);
    T = clamp(T, 0, total - 1e-6);
    let acc = 0, idx = 0;
    for (; idx < video.scenes.length; idx++) {
      if (T < acc + video.scenes[idx].dur) break;
      acc += video.scenes[idx].dur;
    }
    const scene = video.scenes[idx], t = T - acc, p = t / scene.dur;
    const fin = ease.out(clamp(t / 0.45)), fout = clamp((scene.dur - t) / 0.3);
    scene._in = Math.min(fin, fout);
    ctx._accent = video.accent;
    background(ctx, video.accent);
    ctx.save();
    ctx.globalAlpha = scene._in;
    ctx.translate(0, (1 - fin) * 14);
    scene.draw(ctx, p, t, scene);
    ctx.restore();
    if (!scene.bare) chrome(ctx, video, idx, scene, T, total);
    else { ctx.fillStyle = C.line; ctx.fillRect(0, H - 4, W, 4); ctx.fillStyle = video.accent; ctx.fillRect(0, H - 4, W * clamp(T / total), 4); }
  }

  // ---- shared scenes -----------------------------------------------------------------
  function titleScene(name, tagline, chips, dur = 3.6) {
    return {
      dur, bare: true,
      draw(ctx, p, t) {
        const accent = ctx._accent;
        const a = eo(p, 0, 0.25);
        ctx.save();
        ctx.globalAlpha *= a;
        rr(ctx, W / 2 - 46, 196 - 20 * (1 - a), 92, 6, 3, accent);
        ctx.restore();
        text(ctx, name, W / 2, 300 + 18 * (1 - a), { size: 96, weight: 800, align: "center", alpha: a, spacing: -3 });
        text(ctx, tagline, W / 2, 362, { size: 28, weight: 500, color: C.dim, align: "center", alpha: eo(p, 0.15, 0.4) });
        let total = 0; const ws = [];
        ctx.save(); ctx.font = `600 17px ${SANS}`;
        chips.forEach((c) => { const w = ctx.measureText(c).width + 36; ws.push(w); total += w + 14; });
        ctx.restore();
        let x = W / 2 - (total - 14) / 2;
        chips.forEach((c, i) => {
          const ca = eo(p, 0.3 + i * 0.07, 0.5 + i * 0.07);
          ctx.save(); ctx.globalAlpha *= ca;
          rr(ctx, x, 420 + 10 * (1 - ca), ws[i], 38, 19, hexA(accent, 0.14), hexA(accent, 0.7), 1.5);
          text(ctx, c, x + ws[i] / 2, 445 + 10 * (1 - ca), { size: 17, weight: 600, align: "center" });
          ctx.restore();
          x += ws[i] + 14;
        });
      },
    };
  }

  // Closing card: when to choose it, best-for bullets, pros and cons.
  function summaryScene(o, dur = 7) {
    return {
      dur, title: o.heading || "When to choose it", caption: o.caption,
      draw(ctx, p) {
        const accent = ctx._accent;
        const a = eo(p, 0, 0.15);
        ctx.save(); ctx.globalAlpha *= a;
        rr(ctx, 60, 170, 1160, 104, 16, hexA(accent, 0.12), accent, 2);
        text(ctx, "CHOOSE IT WHEN", 88, 204, { size: 14, weight: 700, color: accent, spacing: 2 });
        text(ctx, o.when[0], 88, 234, { size: 22, weight: 600 });
        if (o.when[1]) text(ctx, o.when[1], 88, 260, { size: 22, weight: 600 });
        ctx.restore();
        const cols = [
          { x: 60, label: "BEST FOR", color: accent, items: o.best, mark: "•" },
          { x: 456, label: "PROS", color: C.ok, items: o.pros, mark: "+" },
          { x: 852, label: "CONS", color: C.red, items: o.cons, mark: "–" },
        ];
        cols.forEach((col, ci) => {
          const ca = eo(p, 0.12 + ci * 0.1, 0.3 + ci * 0.1);
          ctx.save(); ctx.globalAlpha *= ca;
          rr(ctx, col.x, 296 + 12 * (1 - ca), 368, 314, 16, C.panel, C.line, 1.5);
          text(ctx, col.label, col.x + 26, 334, { size: 14, weight: 700, color: col.color, spacing: 2 });
          col.items.forEach((it, i) => {
            const ia = eo(p, 0.2 + ci * 0.1 + i * 0.05, 0.36 + ci * 0.1 + i * 0.05);
            text(ctx, col.mark, col.x + 26, 376 + i * 56, { size: 20, weight: 800, color: col.color, alpha: ia });
            const lines = Array.isArray(it) ? it : [it];
            lines.forEach((ln, li) => text(ctx, ln, col.x + 50, 376 + i * 56 + li * 22, { size: 17, weight: 500, alpha: ia, color: li ? C.dim : C.text }));
          });
          ctx.restore();
        });
      },
    };
  }

  window.K = { W, H, C, SANS, MONO, clamp, lerp, seg, ease, eo, eio, hexA, rr, text, line, arrow,
               user, box, tokenBox, bar, stat, typed, fmt, drawFrame, duration, titleScene, summaryScene };
  window.VIDEOS = window.VIDEOS || {};
})();
