# WebGPU latency budget (HANDOFF Step 1)

**Question.** How big can the Slower Roads world model be and still run at
30 fps (≤ 33 ms/frame) in a browser tab, on-device, with ONNX Runtime Web +
WebGPU, for a human playing live?

**Answer (2026-10-01, Apple M3 Max 40-core GPU, Chrome, ORT Web 1.30):**
- **Ship shape:** 16×16 tokens at 64 px, ctx 4 frames, fp16, FSQ-factorized
  head, cache commit fused into the next frame's pass, 1 (max 2) refinement
  passes per frame.
- **Size:** d384–512 × 4–6 layers (≈9–25 M transformer params).
- **Measured cost:** 11–16 ms/frame end-to-end on the M3 Max. That is 2–3×
  headroom for the weaker GPUs real players have.
- **Not affordable:** 128 px (32×32 tokens), depth ≥ 8 at P ≥ 2, 4+ MaskGIT
  passes, and the CPU/WASM fallback.

Everything below was measured with random-weight stand-ins of the Step-3
architecture. Weights don't change latency.

## What was measured

### The spike model (`export/spike_models.py`)

`FrameDynamics` is one refinement pass over the next frame:
- **Attention.** Frame-causal across frames and bidirectional within a frame:
  the next frame's 257 tokens attend to a ctx-frame KV window plus each other.
- **Positions.** Temporal RoPE uses the absolute frame index, so cached K is
  never re-roped as the window slides. Spatial position is a learned embedding.
- **M3 pieces.** It keeps the skeleton-memory cross-attention (outside the KV
  cache) and the state head.
- **In-graph sampling.** argmax + confidence run inside the graph, so a pass
  downloads 256 ids + 256 confidences rather than 256×12800 logits.

**Cache handling.**
- The KV window is a graph input.
- `present_k/v` is the window rolled by one frame.
- The frame's commit run feeds `present` back as the next frame's `past`.

**Variants.**
- **head.** `full` is a 12800-way softmax. `fsq` is 5 independent per-channel
  heads (8+8+8+5+5 = 34 logits) recombined into the id.
- **fused.** The input is [previous frame's final tokens, next frame's masked
  tokens].
  - The previous frame's exact K/V is committed in the same run that predicts
    the next frame.
  - So P=1 is one run per frame instead of two.
- **Decoder.** `FSQDecoder` mirrors `fsq_v2` (h128, ResBlocks + bottleneck
  attention + PixelShuffle). The id → code mapping is a constant-table Gather.

### The harness (`web/bench/`)

It runs ORT Web inside a module **Web Worker** on a cross-origin-isolated page
(COOP/COEP), exactly where the shipped model will run.

**Timing.**
- Wall-clock from JS, including the GPU→JS readback the frame loop needs.
- A full **frame loop** is P MaskGIT passes (cosine schedule, JS top-k
  unmasking), plus the commit, plus the FSQ decode to RGB.
- The frame loop is measured directly. It matches `P·pass + commit + decode`
  to within ~0.5 ms.

**Runs.** 70 shapes with graph capture, plus the wide, fused, non-captured,
CPU-KV, WASM and sustained runs. All raw JSON is in `web/bench/results/m3max_*.json`.

## Results

### 1. Overhead dominates, so graph capture is mandatory

The first, naive build ran d256/L4 at ~10 ms per pass. That is ~30× its FLOP
cost; ORT reached ~1–1.5 TFLOPS, about 5% of the M3 Max fp16 peak.

A kernel profile (JSEP build) showed why:
- There are ~300 dispatches per pass.
- Small matmuls run at 0.4–0.7 TFLOPS.
- RoPE and cache concat add ~100 tiny slice/mul/transpose ops.

ORT's WebGPU **graph capture** records the pass once and replays the command
buffer:

| fp16, pass ms (commit ms) | plain WebGPU EP | KV round-trip through JS | **graph capture** |
|---|---|---|---|
| d256 L4 ctx4 | 11.0 (10.7) | 13.1 (15.3) | **5.8 (5.8)** |
| d384 L8 ctx4 | 15.1 (15.1) | 17.8 (22.5) | **11.8 (11.7)** |
| d384 L8 ctx8 | 18.0 (18.1) | 22.6 (31.8) | **14.9 (15.0)** |
| d512 L12 ctx8 | 28.9 (28.9) | — | **25.5 (25.9)** |

- Capture removes a fixed ~3.5–5 ms per run.
- The KV cache must stay GPU-resident.
- The runtime build barely matters:
  - native-EP asyncify ≈ JSPI ≈ JSEP for fp32;
  - JSEP fp16 was 3× slower;
  - use `ort.webgpu.min.mjs` (native EP).

### 2. Cost scales with depth, not width

Per-pass cost with capture, fp16, unfused, ctx 4:

| | L4 | L8 | L12 |
|---|---|---|---|
| d256 | 5.8 | 9.4 | 12.8 |
| d384 | 7.1 | 11.8 | 16.4 |
| d512 | 8.2 | 14.1 | 19.6 |
| d768 | 11.0 (L2: 7.0) | | |
| d1024 | 14.3 (L2: 8.6) | | |

- A layer costs ~0.9 ms at d256 and ~1.5 ms at d512.
- Quadrupling width at fixed depth (d256 → d1024, L4) costs only 2.5×.
- So **at a fixed budget, prefer wide-and-shallow**: d512–768 × 4–6 layers,
  not d256 × 12.
- There is also a ~2–3 ms fixed floor per run: embedding, output head, argmax
  and the readback sync.

### 3. Other knobs

| knob | effect (pass ms) |
|---|---|
| **fsq head** instead of the 12800-way head | −1.3 ms (d256 L4: 5.8 → 4.5; d384 L8: 11.8 → 10.3). Free; the FSQ grid is already factorized. |
| **fp16** instead of fp32 | −10–20% (d384 L8: 13.1 → 11.8; d512 L12 ctx8: 31.1 → 25.5); half the download |
| **context frames** (d384 L8) | c1 9.3 · c2 10.2 · c4 11.8 · c8 14.9 · c16 21.6: ~0.8 ms per context frame |
| **fused commit** | the fused run is ~1.25× an unfused pass but drops the separate commit run: P=1 goes from 2 runs to 1 |
| **32×32 grid (128 px)** | 3–4× per pass (d256 L4 ctx4: 5.8 → 20.6) plus a 128 px decode of 8.4–12 ms. **Not affordable.** |

### 4. Decoder (ids → 64 px RGB, readback included, capture)

| decoder | fp16 MB | ms |
|---|---|---|
| fsq_v2-style h128, 64 px | 15.4 | 3.5–4.4 |
| h64, 64 px | 4.0 | 1.9 |
| h128, 128 px (either 16×16 or 32×32 grid) | 15–22 | 8.4–8.8 |

- The decoder is compute-bound, so capture changes little.
- At 64 px it is 10–13% of the budget.
- 128 px alone eats a quarter of the budget, before the 4× dynamics cost.

### 5. Frame budget: what fits

**Measured full frame loops**, fp16 + capture + h128 64 px decode, ctx 4, ms
per frame:

| model | core params | P=1 | P=2 | P=3 |
|---|---|---|---|---|
| d256 L4 fsq fused | 4.2 M | **9.2** | 14.6 | 20.2 |
| d384 L4 fsq fused | 9.4 M | **11.0** | 18.3 | 25.7 |
| d512 L4 fsq fused | 16.8 M | **12.7** | 21.8 | 30.8 |
| d384 L6 fsq fused | 14.2 M | **13.9** | 24.1 | 34.3 |
| d512 L6 fsq fused | 25.2 M | **16.4** | 29.1 | 41.7 |
| d768 L4 fsq fused | 37.7 M | **16.8** | 29.7 | 43.9 |
| d384 L8 fsq fused | 18.9 M | **17.3** | 31.0 | 44.6 |
| d512 L4 ctx8 fsq fused | 16.8 M | 15.5 | 27.3 | 39.4 |
| d768 L2 fsq (unfused) | 18.9 M | 14.0 | 19.2 | 24.4 |
| d384 L8 full (unfused) | 18.9 M | 27.0* | 38.5 | 50.5* |

\* is the model `P·pass + commit + decode`; every other cell is a measured
frame loop.

*Core params* = transformer blocks only (self-attn + skeleton cross-attn + MLP
≈ 16·d²·L). The 12800×d embedding and head add more to the download than to
the compute.

**Caveat on fused P ≥ 2.** The bench re-runs the fused 2-frame graph on every
pass. Pairing a fused first pass with an unfused graph for passes 2..P would
save ~1–2 ms per extra pass.

**Sustained load.** d512 L6 fsq fused at P=1 ran for 60 s: 3,638 frames,
median 16.5 ms, p90 16.7 ms. Every 5-second bucket stayed at 16.4–16.5, so
there is no thermal or allocator drift.

**Startup.**
- Session create takes 140–400 ms.
- The first pass, which compiles the shaders, takes 40–80 ms.
- Downloads are 22–75 MB fp16 per dynamics model, plus 15 MB for the decoder.
  Cache them with the Cache API, per Step 6.

### 6. WASM fallback: not real-time

ORT WASM on CPU (16 threads, SIMD) gives:
- d256 L4 fsq, P=1: 75 ms/frame (decoder 35 ms);
- d384 L4 fused: 182 ms.

A no-WebGPU device would need a much smaller model at ≤ 10 fps. Treat it as
"unsupported", not "degraded".

## Recommendation (feeds Steps 2, 3 and 6)

1. **Keep 64 px / 16×16 tokens.** 128 px fails the budget on both dynamics and
   decode, even on an M3 Max. This settles HANDOFF §6 "Is 64px enough?" for v1.
2. **Target 1 refinement pass per frame (2 max).** Step 3's "2–4 MaskGIT passes"
   is too many:
   - at P=4 only a ~19M-core d768 L2 fits 33 ms, and that's on this GPU;
   - so train for **one-step frame prediction**;
   - optionally add a cheap second pass on the lowest-confidence tokens.

   Quality at P=1 is the main open risk this hands to Step 3.
3. **Architecture:**
   - **d384–512, 4–6 layers, ctx 4.**
   - **Factorized FSQ head** (5 per-channel heads).
   - **Fused commit** (the previous frame's final tokens ride in the next
     frame's input).
   - fp16 weights and KV.
   - The recommended starting point is **d384 L6, 11.0–13.9 ms/frame on the
     M3 Max**, with **d512 L6 (16.4 ms)** as the stretch.
   - Spend extra capacity on width, not depth.
   - Each context frame costs ~0.8 ms, so ctx 8 is affordable only for small
     models.
4. **Runtime (Step 6):**
   - ORT Web native WebGPU EP (`ort.webgpu.min.mjs`) in a module worker.
   - `enableGraphCapture`, GPU-resident KV, in-graph sampling.
   - Read back only ids, confidences and RGB.
   - The pitfalls below are mandatory reading.
5. **Device spread.** Only this M3 Max was measured.
   - Base M-series or iGPU laptops have roughly ⅕–⅓ of its GPU throughput.
     Captured passes are now mostly GPU-bound, so expect **~2.5–5× slower**.
   - The d384 L4–L6 P=1 configs (11–14 ms here) are the ones likely to hold
     ≥ 20–30 fps there.
   - Plan for an adaptive tier: pick model size or P from a warmup benchmark.
   - **Next measurement:** open the bench on a weaker laptop (see
     `web/bench/README.md`, "Quick plan").

## ORT Web pitfalls found (save the next person a day)

1. **Graph capture and pre-bound outputs.**
   - `run(feeds, {out: preallocatedTensor})` recognises the bound output on
     replay only if the wasm allocator happens to reuse the same handle
     address. Otherwise it crashes intermittently: `Cannot set properties of
     undefined (setting 'Symbol(gpuBufferMetadata)')`.
   - It also ignores bound outputs whose GPU buffer is larger than the tensor.
   - **Fix:** let ORT own the outputs (`run(feeds)`), then `copyBufferToBuffer`
     from `out.x.gpuBuffer`. The captured graph reuses the same buffers.
2. **Small input buffers.** Graph-capture input buffers must be ≥ 16 bytes,
   because vec4 kernels bind 16 B. A 4-byte `t` buffer fails validation.
3. **int64 index math.** int64 `Div`/`Mod` (id → FSQ code) lands on the CPU EP,
   which blocks graph capture. Use a constant lookup table + `Gather`.
4. **fp16 conversion.** `onnxconverter_common.float16` broke `Cast` nodes in
   these graphs. Export fp16 straight from torch (`module.half()`), and keep
   scalar inputs like the frame index in fp32.
5. **Module workers.** A top-level `await import(ort)` in a module worker
   drops messages that arrive before evaluation finishes. Register `onmessage`
   first.
6. **Profiling.** Kernel timing (`env.webgpu.profiling`) only works in the
   JSEP build (`ort.min.mjs`).

## Reproduce

```bash
# models (random weights) -> web/bench/models/  (gitignored, ~10 GB for everything)
uv venv export/.venv -p 3.12 && uv pip install -p export/.venv/bin/python torch onnx onnxruntime
export/.venv/bin/python export/export_spike.py --sweep core --sweep fused --sweep wide --decoders --fp16
(cd web/bench && npm i)                      # onnxruntime-web
node web/bench/run_bench.mjs --match '_fused_fp16$' --P 1,2,3 --out results/mine.json
~/anaconda3/bin/python web/bench/analyze.py web/bench/results/*.json
```

## Making 64 px look good: learned 64 → 256 upscaler

Raw 64 px looks bad at screen size (`docs/resolution_64px.png`). A tiny
super-resolution net is the fix, and it fits the budget. The SR stage runs
after the FSQ decode: 64 px RGB in, 256 px RGB out.

### Setup (`export/sr/`)

**Data.**
- `gen_hr.mjs` renders 256 px sim frames through the same supersample + box
  capture path. There are 30 train worlds (5,430 frames) and 4 held-out worlds
  (724 frames).
- Inputs are those frames box-downsampled ×4. That is what the 64 px pipeline
  sees.
- The real Slow Roads footage is only used to look at generalization; nothing
  is trained on it.

**Model and losses.**
- The architecture is SRVGGNetCompact (conv/PReLU stack + PixelShuffle ×4 +
  nearest skip), scaled from 32 k to 1.2 M params.
- `*_l1` nets use an L1 loss.
- `*_gan` nets fine-tune with L1 + LPIPS(VGG) + a PatchGAN, which restores
  texture.
- `large_ftgan` is Real-ESRGAN `realesr-general-x4v3` (BSD-licensed)
  fine-tuned on our sim.
- Everything trained on the M3 Max via MPS in ~50 min total.

### Results

Metrics are on the 724 held-out frames, against the 256 px truth:
- **LPIPS:** lower = looks closer.
- **flicker:** mean |Δframe − Δtruth| ×100; lower = steadier.
- **WebGPU ms:** fp16, M3 Max, 64→256 including readback, non-captured.

| method | params | PSNR | LPIPS | flicker | WebGPU ms |
|---|---|---|---|---|---|
| nearest (raw 64 px) | 0 | 24.6 | 0.260 | 1.96 | 0 |
| bicubic / lanczos | 0 | 26.2 / 26.4 | 0.338 / 0.344 | 1.92 | ~0 |
| Real-ESRGAN generic, off-the-shelf | 1.2 M | 25.3 | 0.156 | 2.08 | 5.0 |
| tiny_l1 | 32 k | 30.7 | 0.097 | 1.49 | 2.1 |
| small_l1 | 89 k | 31.7 | 0.091 | 1.38 | 2.5 |
| medium_l1 | 355 k | **32.1** | 0.082 | **1.32** | 3.3 |
| small_gan | 89 k | 30.2 | 0.043 | 1.55 | 2.4 |
| medium_gan | 355 k | 30.7 | 0.037 | 1.49 | 3.3 |
| large_ftgan (Real-ESRGAN fine-tuned) | 1.2 M | 30.5 | **0.032** | 1.54 | 5.0 |

Visuals are in `docs/upscale_sheet.png`: 3 held-out sim worlds + 3 real Slow
Roads frames.

**Findings.**
- **Domain-trained SR wins.** A 32 k-param net beats a generic 1.2 M-param
  model. Perceptual error drops 2.7× (tiny) to 8× (large_ftgan) vs raw 64 px.
- **The `_gan` variants are what look right.** They restore sharp car
  edges, lane dashes, guardrails, distinct trees and grass texture.
- **No flicker cost.** Every trained net is steadier than nearest or bicubic.
  The GAN ones are ~10% less steady than the L1 ones.
- **On real Slow Roads frames** they generalize well: clean geometry and
  sim-style grass. small_gan shows a faint checker pattern in flat grass;
  medium_gan and large_ftgan don't.
- **Cost:** 2–5 ms. For example, d384 L4 fused P=1 + medium_gan comes to
  ~14 ms/frame on the M3 Max.

**Caveat.** Inputs here are clean downsamples. In the real pipeline the input
is the FSQ decoder's reconstruction, so:
- train the SR stage on tokenizer outputs once Step 2's tokenizer exists;
- or fold it into the decoder, emitting 256 px from the 16×16 tokens.

Re-measure flicker on dynamics rollouts too.
