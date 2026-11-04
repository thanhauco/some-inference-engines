#!/usr/bin/env node
// Render the motion graphics to MP4 + GIF + poster PNG with headless Chromium and ffmpeg.
//
//   node videos/render.js                 # all videos -> videos/out/
//   node videos/render.js vllm sglang     # only these
//   node videos/render.js --stills DIR    # one PNG per scene (for reviewing layout)
//
// Needs `playwright` (npm i playwright, or NODE_PATH pointing at a global install) and
// ffmpeg on PATH. Frames are drawn by index.html's renderAt(id, seconds), so the output
// is frame-exact and identical to what the player shows.
const { spawn, spawnSync } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require("playwright");

const FPS = 30;
const ROOT = __dirname;
const OUT = path.join(ROOT, "out");

function parseArgs(argv) {
  const args = { ids: [], stills: null };
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === "--stills") args.stills = argv[++i];
    else args.ids.push(argv[i]);
  }
  return args;
}

function ffmpeg(args, input) {
  const res = spawnSync("ffmpeg", ["-hide_banner", "-loglevel", "error", "-y", ...args], { input });
  if (res.status !== 0) throw new Error(`ffmpeg failed: ${res.stderr}`);
}

async function grab(page) {
  const url = await page.evaluate(() => document.getElementById("stage").toDataURL("image/png"));
  return Buffer.from(url.slice(url.indexOf(",") + 1), "base64");
}

async function renderVideo(page, info) {
  const frames = Math.round(info.duration * FPS);
  const mp4 = path.join(OUT, `${info.id}.mp4`);
  const enc = spawn("ffmpeg", [
    "-hide_banner", "-loglevel", "error", "-y",
    "-f", "image2pipe", "-framerate", String(FPS), "-c:v", "png", "-i", "-",
    "-c:v", "libx264", "-preset", "slow", "-crf", "22", "-pix_fmt", "yuv420p",
    "-movflags", "+faststart", mp4,
  ], { stdio: ["pipe", "inherit", "inherit"] });
  const done = new Promise((resolve, reject) => {
    enc.on("close", (code) => (code === 0 ? resolve() : reject(new Error(`ffmpeg exited ${code}`))));
  });
  for (let f = 0; f < frames; f++) {
    await page.evaluate(([id, t]) => window.renderAt(id, t), [info.id, f / FPS]);
    const png = await grab(page);
    if (!enc.stdin.write(png)) await new Promise((r) => enc.stdin.once("drain", r));
    if (f % 150 === 0) process.stdout.write(`\r${info.id}: ${f}/${frames} frames`);
  }
  enc.stdin.end();
  await done;
  // Small looping GIF for READMEs (no dithering: flat graphics compress far better without
  // it), and a half-size poster taken late in the second scene.
  ffmpeg(["-i", mp4, "-vf",
    "fps=8,scale=560:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64:stats_mode=diff[p];" +
    "[b][p]paletteuse=dither=none:diff_mode=rectangle",
    path.join(OUT, `${info.id}.gif`)]);
  await page.evaluate(([id, t]) => window.renderAt(id, t), [info.id, info.scenes[0] + info.scenes[1] * 0.8]);
  ffmpeg(["-f", "image2pipe", "-c:v", "png", "-i", "-", "-vf", "scale=640:-1:flags=lanczos",
    "-frames:v", "1", path.join(OUT, `${info.id}.png`)], await grab(page));
  const mb = (f) => (fs.statSync(path.join(OUT, f)).size / 1e6).toFixed(2);
  console.log(`\r${info.id}: ${info.duration.toFixed(1)} s, ${frames} frames -> ` +
    `${info.id}.mp4 (${mb(info.id + ".mp4")} MB), ${info.id}.gif (${mb(info.id + ".gif")} MB)`);
}

async function renderStills(page, info, dir) {
  let start = 0;
  for (let i = 0; i < info.scenes.length; i++) {
    for (const [tag, frac] of [["a", 0.45], ["b", 0.92]]) {
      await page.evaluate(([id, t]) => window.renderAt(id, t), [info.id, start + info.scenes[i] * frac]);
      fs.writeFileSync(path.join(dir, `${info.id}-${i}${tag}.png`), await grab(page));
    }
    start += info.scenes[i];
  }
  console.log(`${info.id}: ${info.scenes.length} scenes -> ${dir}`);
}

(async () => {
  const args = parseArgs(process.argv.slice(2));
  fs.mkdirSync(OUT, { recursive: true });
  if (args.stills) fs.mkdirSync(args.stills, { recursive: true });
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
  page.on("pageerror", (e) => { console.error("page error:", e.message); process.exitCode = 1; });
  await page.goto("file://" + path.join(ROOT, "index.html") + "?headless=1");
  await page.waitForFunction(() => window.READY === true);
  const infos = (await page.evaluate(() => window.videoInfo()))
    .filter((v) => args.ids.length === 0 || args.ids.includes(v.id));
  if (infos.length === 0) throw new Error(`no videos match ${args.ids.join(", ")}`);
  for (const info of infos) {
    if (args.stills) await renderStills(page, info, args.stills);
    else await renderVideo(page, info);
  }
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
