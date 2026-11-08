# Motion graphics: the four engines, animated

Five short explainer videos, generated entirely from code in this folder.

| Video | Length | What it shows |
|---|---|---|
| [overview.mp4](out/overview.mp4) | 0:54 | Four engines, how to choose, the migration path, switching by config |
| [vllm.mp4](out/vllm.mp4) | 0:52 | Many users → continuous batching → PagedAttention → decode → stream |
| [sglang.mp4](out/sglang.mp4) | 1:06 | Programs → RadixAttention → cache-aware scheduling → speculative decoding → structured output |
| [tensorrt_llm.mp4](out/tensorrt_llm.mp4) | 0:54 | Compile → kernel/layer fusion → quantization → in-flight batching |
| [ollama.mp4](out/ollama.mp4) | 0:58 | Pull → GPU/CPU layer split → run locally → one API → unload when idle |

Each video also has a looping `.gif` preview and a `.png` poster in [`out/`](out).

## Watch interactively

Open [`index.html`](index.html) in a browser (no build step, no server needed): pick a
video, pause, and scrub frame by frame.

## How it works

```
src/data.js          numbers exported from the Python simulations (export_data.py)
src/kit.js           ~280-line canvas motion kit: easing, timeline, drawing helpers
src/<engine>.js      the scenes of one video; every frame is a pure function of time
index.html           live player, and the page the renderer drives
render.js            headless Chromium draws each frame -> ffmpeg -> MP4, GIF, poster
```

Two properties are deliberate:

* **Deterministic.** A scene is `draw(ctx, progress, seconds)` with no hidden state, so
  the browser preview and the rendered file are identical, and any frame can be rendered
  on its own.
* **Numbers come from the code.** Every figure on screen (2,022 tok/s, 17% → 99%,
  18% → 82% hit rate, 93 → 16 model calls ...) is read from `src/data.js`, which
  `export_data.py` regenerates from the simulations in `src/inference_engines/concepts`.
  A test fails if the two drift apart. Scenes that use illustrative timings say so on
  screen.

## Re-render

```bash
python videos/export_data.py          # refresh numbers from the simulations
npm install playwright                # once; needs Chromium + ffmpeg on PATH
node videos/render.js                 # all five -> videos/out/
node videos/render.js vllm            # just one
node videos/render.js --stills /tmp/stills   # two PNGs per scene, for layout review
```

## Add or change a scene

A scene is an object in the video's `scenes` array:

```js
{
  dur: 8, step: 2, title: "Continuous batching", caption: "One line of narration.",
  draw(ctx, p, t) {            // p = 0..1 through the scene, t = seconds into it
    const a = K.eo(p, 0.1, 0.3);   // eased sub-progress between 10% and 30%
    K.box(ctx, 100, 200, 240, 120, "GPU", { color: K.C.blue, alpha: a });
  },
}
```

`K.seg` / `K.eo` / `K.eio` slice the scene's progress into staggered sub-animations;
that is all the "timeline" there is.
