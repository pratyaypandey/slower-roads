# Slower Roads

*An on-device, real-time, steerable driving world model. It's Slow Roads, except there's no game engine drawing the road — a neural net dreams every frame, live, in your browser, and you can turn dials that reach into its head. Slower, because there's a world model running under the hood. Cooler, for the same reason.*

---

## 0. Read this first (project orientation)

This doc onboards you into the whole idea, including *why* the design is what it is, so you can push back on it. Nothing here is sacred except the thesis.

**The one sentence everything ladders up to:**

> The first interactive driving world model that runs real-time on-device (WebGPU, no server), stays consistent enough that players don't realize it's generated, and exposes its latent dynamics as steerable directions — with ground-truth evaluation for all three claims.

If a task doesn't serve that sentence, we cut it.

**What it is as a product:** you open a webpage, you drive an endless scenic road, WASD, chill — the Slow Roads loop. The twist is a set of dials: continuously morph the weather, the biome, the time of day; blend between environments in ways a hardcoded game physically can't; and (stretch) a gravity slider. The player never sees the word "AI." They see a driving game with knobs that feel like magic. It's a static site — no login, no server, no cost, runs on their own GPU.

**What it is as research:** three underexplored things that only look like a product when stacked. (1) World models at *consumer-compute* budgets — everyone else assumes datacenter GPUs or ASICs. (2) *Ground-truth evaluation* of a world model, which almost nobody can do. (3) *Interpretability-based control* — editing the world by moving activation directions instead of retraining. The intersection of those three is an unoccupied spot in the literature.

---

## 1. Why the design looks like this (the reasoning, not just the conclusions)

You'll want the logic behind three non-obvious calls, because they're the load-bearing ones.

**The sim is an oracle, three times over.** We build our own deterministic driving sim (a dumb Three.js world) and distill *it* into the neural model, rather than training on real driving footage. That one decision buys us three things nobody training on YouTube gets:
- Perfectly-labeled, unlimited `(frame, action, next_frame)` training data.
- Ground truth to *measure drift* — we can diff a neural rollout against what the sim would actually have done, frame by frame.
- Free, perfectly-balanced contrastive pairs for steering-vector extraction — render the same seed with rain vs. clear and the label is a function argument, not a labeling project.

Clean contrastive pairs and quantitative world-model eval are both things the field fights for. We get them for the price of writing a sim. This is the single insight that makes the whole thing rigorous instead of a vibes demo.

**The neural model owns the dynamics.** At inference the world is *generated*, not painted over a live game engine. This matters because the spicy research claim — steer gravity/physics with a direction vector — only exists if the net is actually producing the physics. If a real engine computed the motion, "gravity dial" would just be a config value, not a steering vector. So the net owns dynamics.

**But drift is the enemy of the product**, and a pure generative model drifts (roads melt into grass — the exact Oasis/Genie failure). So we add a **sim-anchor slider `λ ∈ [0,1]`**:
- `λ = 1`: a cheap deterministic skeleton runs alongside and feeds the model ground-truth conditioning each frame (road centerline, car pose, coarse depth). The road *structurally cannot* melt. Maximally consistent, maximally shippable, least "pure."
- `λ = 0`: pure autoregressive generation, Oasis-style. Maximally steerable and impressive, driftiest.

Same weights, one codebase. `λ` is simultaneously a research instrument (it sweeps the drift-vs-purity tradeoff for the paper) and the product's safety valve (ship near `λ=1`, run the flashy experiments near `λ=0`). This is how the "as good as Slow Roads" goal and the "pure world model" goal stop fighting each other.

---

## 2. System architecture, model-up

```
[actions] ──┐
            ▼
      ┌─────────────┐   latent    ┌──────────────────┐  latent   ┌──────────┐  pixels
frame │  tokenizer  │───tokens───▶│  dynamics core   │──tokens──▶│ decoder  │────────▶ screen
────▶ │ (VQ-VAE/ViT)│             │ (AR transformer, │           │ (1–2 step│
      └─────────────┘             │  KV-cached)      │           │  distill)│
                                  └────────▲─────────┘           └──────────┘
                                           │ h ← h + α·v   ◀── steering directions
                          ┌────────────────┴───┐
                          │ sim skeleton (λ)   │ ◀── optional ground-truth conditioning
                          └────────────────────┘
```

**Deterministic sim (`sim/`).** Minimal Three.js driving world: procedural road, a handful of biomes, and *explicit parameters* for weather, time-of-day, gravity, friction. Emits `(frame, action, next_frame, params)` with perfect labels. Doubles as the eval oracle. This is the foundation for all three research claims and it has zero ML risk, so it's built first.

**Spatial tokenizer (`model/tokenizer/`).** VQ-VAE or VAE with a ViT-style encoder, compressing frames to a small latent grid (start target ~16×16 tokens). All dynamics training happens in this latent space — never pixels. Reconstruction must be clean before dynamics work starts.

**Action-conditioned dynamics core (`model/dynamics/`).** Autoregressive transformer over latent tokens + a control vector, trained next-token (MineWorld-style). AR over diffusion here for three concrete reasons: KV-cache reuse (real-time + it's the inference-optimization lever we actually understand), clean mid-layer activation access for steering, and predictable per-frame latency. This is the object we steer and the object that drifts.

**Decoder (`model/decoder/`).** Latent → pixels, consistency-distilled down to 1–2 steps. This is the biggest single real-time lever; multi-step diffusion decode is exactly what makes Oasis's public weights too slow to be a product.

**On-device runtime (`web/`).** ONNX Runtime Web (WebGPU execution provider) or transformers.js for the forward pass, running in a **Web Worker** — a 33ms forward pass on the main thread freezes the whole UI. Cache API persists weights after first download. **COOP/COEP headers are mandatory** or WebGPU/SharedArrayBuffer silently degrades. Warmup dummy inference to absorb shader-compile latency. WebGPU→WASM tiered fallback so weak devices still get *something*. Hosting is a static bundle + weights on a CDN; every user brings their own compute; server GPU bill is zero.

**Steering (`steering/`).** Contrastive activation addition (CAA): generate matched rollouts differing in one sim param, run both through the dynamics core, take the mean activation difference at layer `ℓ` → direction `v`. At inference, `h ← h + α·v`, where `α` is the user's knob. The interp-rich version decomposes the dynamics activations with a sparse autoencoder and steers monosemantic features instead of raw contrastive directions.

---

## 3. The engineering landmine (decide early)

Injecting `h ← h + α·v` mid-graph is **not** natural in a frozen ONNX graph. Two options, pick one before the export pipeline solidifies:

1. **Split the dynamics core** into pre-/post-submodels at the injection layer; do the add in JS/WGSL between them. Pragmatic, works with ORT Web, our default.
2. **Hand-roll the dynamics block in WGSL** for full control. More work, more flexibility.

Either way: **architect the ONNX export so *any* layer is splittable from day one.** Retrofitting mid-graph activation access after the fact is a weekend you don't get back. Also settle the **latent grid size** by working *backward* from the 33ms frame budget, not forward from how pretty frames look — grid size trades directly against fps and against how cleanly the model steers.

---

## 4. Milestones

Each milestone is independently defensible. If a later one fails, the project still has a complete story at the previous one. Checkboxes are the shared tracker — tick as we go.

### M0 — Sim + oracle harness *(no ML risk; do this first, together)*
- [ ] Three.js driving sim: procedural road, ≥3 biomes, car physics, WASD control
- [ ] Explicit params exposed: weather, time-of-day, gravity, friction
- [ ] Deterministic seeding (same seed + same actions → identical frames)
- [ ] Data export: `(frame, action, next_frame, params)` at fixed res, to disk
- [ ] Oracle replay harness: given a seed + action sequence, regenerate ground-truth frames for diffing
- [ ] **Drift metric defined and implemented** (latent + pixel divergence vs. rollout length)

*Done when:* we can generate an arbitrary labeled dataset and, given any action sequence, produce the exact ground-truth future to compare against.

### M1 — Tokenizer
- [ ] VQ-VAE/VAE trains to clean reconstruction on sim frames
- [ ] Latent grid size chosen against the frame budget (not fidelity)
- [ ] Encode/decode round-trip validated; codebook usage healthy (if VQ)

*Done when:* frames survive the latent round-trip with no visible degradation, at a grid size that leaves real-time headroom.

### M2 — Dynamics core (teacher-forced)
- [x] AR transformer over latent tokens + control vector
- [x] Next-token training loop, KV cache implemented
- [x] Coherent **short** rollout (a few seconds) under teacher forcing

*Done when:* the model produces a coherent drivable short clip that responds correctly to steering input.

*Status (2026-07-12):* **Coherence done; steering weak.** Full campaign — see `docs/M2_RESULTS.md`.
The first pass looked certified on absolute drift but was **3× worse than a copy baseline**:
root cause was the M1 tokenizer being *temporally unstable* (84% of tokens flip between
99.3%-identical frames; 57% flip from 1% pixel noise). Fixed with temporal-consistency +
noise-robustness losses (`train_tokenizer --temporal-weight/--noise-weight`): churn 84%→41%,
noise-flip 57%→8%, recon still excellent. Retrained dynamics on **10 seeds** (latent cache,
seed5=val, **seed2=pristine held-out test**), context 8, dropout+weight-decay. Result on the
held-out seed: teacher-forced token-accuracy **0.567** (was 0.06), coherent 60-frame drives
(`eval/plots/dream_tf.gif`), **beats the frozen-persistence baseline by +0.0112 free-run
(~14× the pre-campaign model)** and beats copy on the strict bar. **Honest gap:** action/steering
response is weak — forcing left vs right yields near-identical dreams (`eval/plots/steering.gif`);
needs stronger action conditioning (a concrete next step) before "responds correctly to steering"
is met. Also load-bearing: judge world models against baselines (persistence/frozen), not
absolute drift; and inference uses a *bounded* context window (unbounded prefix melts via RoPE
OOD — the eval default, and how M4's real-time KV cache runs).

### M3 — Anti-drift (the hard part; budget the most time here)
- [ ] Diffusion Forcing / per-token noise levels in training
- [ ] Training on the model's own noise-augmented rollouts (self-forcing)
- [ ] Sim-anchor `λ` conditioning path implemented
- [ ] **Drift curve:** coherence extended from seconds → minutes; plotted vs. rollout length, model size, `λ`

*Done when:* at `λ=1` the road holds for a multi-minute drive; we have the drift-vs-`λ` curve that no other world-model paper can produce.

### M4 — Real-time on-device *(product exists here)*
- [ ] Decoder consistency-distilled to 1–2 steps
- [ ] Quantization (int8, try int4 weights) with acceptable quality loss
- [ ] ONNX export with splittable injection point
- [ ] WebGPU runtime: ORT Web WebGPU EP in a Web Worker
- [ ] Cache API weight storage, COOP/COEP headers, warmup, WebGPU→WASM fallback
- [ ] **20–30fps on a discrete-GPU laptop**, measured; latency breakdown logged

*Done when:* someone opens a URL and drives, smoothly, on their own machine.

### M5 — Safe-tier steering *(the "cooler Slow Roads" product ships here)*
- [ ] CAA extraction for weather / biome / time-of-day / vibe (sim-labeled pairs)
- [ ] Direction library + in-product knobs wired to `α`
- [ ] Controllability eval (safe tier): on-target vs. off-target vs. added drift, against the oracle
- [ ] Product UX: driving loop + dials, "no AI visible" polish pass

*Done when:* a player turns a knob and the world morphs believably, and we can prove the morph is on-target against ground truth.

### M6 — Spicy steering + interp *(stretch; the ambitious paper)*
- [ ] Physics steering (gravity/friction/speed-feel) directions
- [ ] SAE decomposition of the dynamics core; hunt monosemantic factors
- [ ] Steer SAE features; controllability eval on the hard tier
- [ ] Layer sweep for best injection point (empirical, on-target/off-target per layer)

*Done when:* we can move a physics dial coherently — or we've cleanly characterized *why not*, which is itself a result. **This milestone must not hold the project hostage; M5 is a complete paper without it.**

---

## 5. Suggested division of labor

Two tracks, overlapping at the sim (M0, joint) and the ONNX export boundary. Assign by strength; this is a starting proposal, not a decree.

**Track A — Model & Research:** tokenizer, dynamics core, anti-drift training, steering extraction, SAE, all evals. This is the ML-heavy, research-credit-heavy track.

**Track B — Systems & Product:** the Three.js sim + oracle harness, ONNX export/splitting pipeline, WebGPU runtime, Web Worker plumbing, quantization/distillation-to-deployment, product UX, hosting.

**The inference-optimization crown jewel** (consistency distillation, quantization, KV-cache real-time recipe) sits on the seam and is the actual moat — whoever's stronger on systems/serving should own it, and it deserves to be *someone's* explicit responsibility, not a shared afterthought.

Realistic split if skills are lopsided toward research vs. web: research-heavy person takes Track A + the inference recipe; web/systems-heavy person takes Track B + owns getting M4 to actually hit framerate. Sim (M0) is pair-built so both understand the oracle everything is measured against.

---

## 6. Stack, compute, repo

**Stack.** Sim/product: Three.js, WebGPU. Runtime: ONNX Runtime Web (WebGPU EP) or transformers.js; Web Workers; Cache API. Training: PyTorch. Export: ONNX + a custom splitting step. Hosting: any static CDN (Vercel/Netlify/Cloudflare Pages) — remember COOP/COEP headers.

**Compute.** Deliberately cheap: train in latent space at low resolution (start 64–128px output), small model (single-digit to low-tens of M params). A single rented A100 (or a local 4090) covers M1–M3. The whole point is that inference is free (on-device); training should stay modest too.

**Repo layout.**
```
slower-roads/
  sim/          # Three.js deterministic sim + data export + oracle harness
  data/         # generated datasets (gitignored) + dataset specs
  model/
    tokenizer/  # VQ-VAE / VAE
    dynamics/   # AR latent transformer + KV cache
    decoder/    # consistency-distilled decoder
  steering/     # CAA extraction, SAE, direction library
  eval/         # drift / realtime / controllability harnesses
  export/       # ONNX export, graph splitting, quantization
  web/          # WebGPU runtime, worker, product UI (the site)
  docs/         # this file, notes, results
```

---

## 7. Evaluation (this is the paper — keep it quantitative)

Because the sim is ground truth, every claim is a number, not a vibe.

**Drift.** Latent + pixel divergence from oracle vs. rollout length, swept over model size, decoding scheme, `λ`. The figure other world-model papers can't make.

**Real-time.** FPS distribution + latency breakdown (tokenize / dynamics / decode) across discrete GPU, integrated GPU, mobile. The on-device claim lives or dies here.

**Controllability.** Per knob: on-target change magnitude (vs. the sim *rendered with the true param*), off-target leakage, added drift under steering. Success = large on-target, small off-target, minimal consistency cost.

---

## 8. Risks & kill criteria (straight, no sugarcoating)

- **Drift untamable at small scale** → raise `λ`, lean on the anchor. Product ships anyway; the drift curve is still a result.
- **Real-time unreachable at acceptable quality** → drop resolution, raise `λ`, accept 15fps, degrade gracefully.
- **Physics steering mushes** → expected failure of the M6 spike. Ship M5. Characterizing the failure is a legitimate finding.
- **Scope creep toward fidelity** → the biggest silent risk. We will not out-pretty DeepMind solo. Chasing frame beauty burns the hours ablations need. The legible research story is primary; product cleanliness serves it. A clean live demo is a great figure 1 and a terrible reason to spend three weeks on shaders.

---

## 9. Open questions to settle together

- Latent grid size — decide from the frame budget at M1.
- AR-token vs. short-diffusion dynamics — defaulting to AR; revisit only if quality forces it.
- Injection-layer strategy — graph split vs. WGSL; pick before export solidifies.
- Default shipping `λ` — probably near 1, but set it from the M3 drift curve.
- How much of M0's sim complexity is worth it — richer sim = richer world but more to distill.

---

## 10. This week (first concrete moves)

1. Stand up the repo with the layout above.
2. Pair-build the M0 sim skeleton: road + car + WASD + deterministic seeding.
3. Wire the data exporter and the oracle replay harness in the same pass — the harness is what makes everything after this measurable, so it's not optional and not "later."
4. Implement the drift metric against the oracle. Once it runs, M1 can start.

---

## Appendix — ramp-up reading

Interactive world models: Genie 3 / Project Genie, Oasis (Decart), GameNGen, DIAMOND, MineWorld (open-source, AR, closest to our dynamics core — good to read first). Rollout stability: Diffusion Forcing, Self-Forcing. Control via activation editing: contrastive activation addition, steering / persona vectors, sparse autoencoders. On-device: ONNX Runtime Web WebGPU docs, transformers.js, and any recent WebGPU 3DGS-in-browser work (e.g. Visionary) for the runtime patterns.

The gap none of them occupy: consumer-compute real-time **+** interpretability-based control **+** ground-truth evaluation. That intersection is Slower Roads.
