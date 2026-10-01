# Handoff: Slower Roads (2026-10-01)

This is for an agent picking up the repo with no prior context. Read it fully
before changing anything. It records the state at the end of the 2026-09-30 →
2026-10-01 session and the ordered plan for what comes next.

## 1. What this project is

**Slower Roads** is Slow Roads (slowroads.io, an endless relaxing driving game)
recreated as a **neural world model** that:

- runs **real-time, on-device, in the browser (WebGPU)**, so anyone can play with no server,
- takes **live keyboard input** (WASD / arrows), and
- **does not drift**: the world stays consistent indefinitely.

The world model is trained on frames from our own deterministic simulator
(`sim/`), which now looks like the real game. The simulator also provides
ground-truth "oracle" state (car pose, road layout ahead) that is used both
for evaluation and as conditioning that pins the model to the real world.

Read next, in this order:
1. `ROADMAP.md` (thesis, milestones M0–M6)
2. `docs/architecture.md` (model contract)
3. `docs/M3_RESULTS.md` (anti-drift results)
4. `docs/FIDELITY.md` (renderer rebuild)
5. `docs/TRAINING.md` (commands)

## 2. Repo map (what matters)

```
sim/                    JS simulator (no build step; served statically)
  core/                 deterministic, renderer-free world + physics (the oracle)
    sim.js              SlowSim: step(action), state, skeleton(), autopilotAction()
    road.js             road generation, heightfield, hillside cross-slopes, nearest()
    car.js              car physics + steerForYawRate() (inverse yaw model)
    scatter.js          deterministic trees/rocks/bushes/grass
    input.js            keyboard -> action smoothing (shared by demo + data gen)
    policies.js         driving policies for dataset generation
  render/               Three.js RGB renderer, rebuilt to match Slow Roads footage
    renderer.js shaders.js textures.js car.js
  demo/                 playable browser game (index.html, main.js)
  headless/             Playwright-driven capture
    generate_dataset.mjs   multi-episode dataset generator (policies)
    generate_pixels.mjs    single-run pixel dataset
    generate.mjs           state-only dataset (no browser)
    snapshot.mjs           quick-look stills
    capture_page.html dream_page.html render_dream.mjs lib.mjs
  test/core.smoke.mjs   core determinism/physics tests: `node sim/test/core.smoke.mjs`
  serve.py              no-cache static server: `python3 sim/serve.py 8777`
model/                  PyTorch
  tokenizer/            FSQ autoencoder (64px frame -> 16x16 tokens, 12,800 codes)
  dynamics/             AR transformer (ar_core.py), config.py (token layout), rollout loss
  data/dataset.py       manifest loader (frames, actions, state, skeleton anchor grid)
  train_tokenizer.py train_dynamics.py train_state_dynamics.py precompute_latents.py
eval/                   evals: eval_m3.py (long-horizon drift), fidelity.py, calibrate_view.py,
                        ingest_recording.py, eval_dream.py, eval_steering.py, profile_decode.py
deploy/                 Modal (modal_train.py, modal_gen.py, modal_serve.py) + RunPod notes
docs/                   architecture, TRAINING, M2/M3 results, FIDELITY, VAE_RECIPE
data/                   gitignored: datasets, reference footage
checkpoints/            gitignored: tokenizer.pt, tokenizer_tc.pt, dynamics_*.pt (old renderer!)
```

## 3. Current state

### Done before this session (see `docs/M2_RESULTS.md`, `docs/M3_RESULTS.md`)

- **M0 (sim + oracle).** Done.
- **M1 (FSQ tokenizer).** A temporal-consistency fix was the key unlock.
- **M2 (dynamics core).** Beats baselines; action steering is weak.
- **M3 (anti-drift).** On held-out seed2 the model passes the 2-minute
  (3,600-frame) gate, scoring below the frozen-frame baseline over the full
  rollout:
  - λ=1 normalized drift AUC **0.953** vs λ=0 (pure autoregressive) **1.045**;
    lower is better, and **1.0 = frozen frame**.
  - The recipe: skeleton-memory cross-attention (outside the causal KV cache,
    no RoPE) plus a state head regressing per-frame `{x, z, heading, speed}`
    deltas, warm-started.
  - Flags: `--mem-cross-attn --mem-tokens 16 --state-head --state-weight 0.5`,
    with `--anchor-cond`, `--self-rollout` and `--corruption-cond` as training aids.
- All existing checkpoints were trained on the **old renderer and the old
  world**. They are now stale (see below).

### Done this session

1. **Renderer rebuilt to match the real game (clean-room).** `sim/render/` is a
   new engine (custom shaders, no post-processing) matched to a 13-minute
   recording of real Slow Roads (`data/reference/`). The old renderer, its
   post-processing passes and Sky.js were deleted. Results and method are in
   `docs/FIDELITY.md`; it beats the old engine on 14 of 15 fidelity metrics.
   - **Hard rule:** Slow Roads is closed source. The GitHub "slowroads" repos
     are unofficial scrapes (CC BY-NC-ND). **Never copy, port or consult its
     code or assets.** slowroads.io also blocks automated browsers; don't try.
     New reference footage = the user screen-records, then
     `eval/ingest_recording.py`.
2. **Bugs fixed:**
   - **Capture was linear-light.** The old `capture()` frames were washed out
     because tone mapping was skipped for render targets. Now the canvas and
     `capture()` produce identical pixels.
   - **Incline judder.** `road.nearest()` was quantised to 1 m, so on slopes
     the car's height stair-stepped 0.16 m. It now refines to the exact
     arc-length.
   - **Grass covering the road.** The coarse far-terrain grid showed through
     the road on hillsides. It is now sunk beneath the near-terrain strip.
   - **Vegetation flicker/pop.** Verge grass was being regenerated every metre,
     and plants popped at hard distance cutoffs. Now there are fixed world-grid
     rows and distance fades.
   - **Autopilot drifting off-road.** Pure pursuit cut corners. It was replaced
     with a lane-keeping controller using a curvature feed-forward and the
     inverse yaw model. It now drives the **left lane** (`road.offset = -1.8`;
     + is right of travel), like the real game.
3. **World retuned to match the footage** (in `sim/core`):
   - steeper road cuts (`ROLL_SPAN` 16),
   - bigger hills (`H_FREQ`, `H_AMP`, `H_AMP2`),
   - hillside cross-slopes beside the road (`hillside()` in road.js),
   - clumpier groves, fewer conifers, trees closer to the road.

   **Car handling (WASD physics) is unchanged.**
4. **Dataset `data/train_v2/` generated.** Settings:
   - 3.0 hours total: 90 two-minute episodes × 3,600 frames (324,090 frames)
     at 64px, 30 fps. That is 17 GB of float32 `.npy`.
   - Episode `i` is world seed `10000+i`.
   - Index: `data/train_v2/index.json`. Generation took ~25 min locally on Metal.

   | profile | eps | off-road | what |
   |---|---|---|---|
   | cruise | 23 | 0% | game-style autopilot, left lane, varying cruise speed |
   | keys_lane | 22 | 0% | simulated human on WASD (reaction delay, A/D taps) |
   | keys_explore | 22 | 30.5% | random maneuvers (swerve, hard turn, off-road, brake, reverse) + recovery |
   | lane_change | 11 | 0% | keyboard lane switches + speed changes |
   | dial_mix | 12 | 0% | cruise/keys under random time of day, fog, rain, snow, biome, terrain |

   Each sample has `frame`, `action` (`{steer, throttle}` as applied), `keys`
   (e.g. `"wa"`), `state`, `skeleton` and `labels`. The format is the same one
   `model/data/dataset.py` already loads, and `--data` accepts many dirs, e.g.
   `--data data/train_v2/ep*`.

### Nothing is committed

`git status` shows ~68 changed paths. That includes uncommitted M3 work from the
previous session (`model/`, `eval/`, `deploy/`, `docs/M3_RESULTS.md`) and all of
this session's work. Before starting:

- **Commit on a branch, not `main`.**
- **Ask the user how they want it split**: e.g. "M3 anti-drift", "renderer v2 +
  fidelity tooling", "core fixes + policies + dataset gen".
- Don't commit `data/` or `checkpoints/`; they're gitignored, so keep it that way.

### Environment gotchas

- **Python.** Use `~/anaconda3/bin/python`; it has numpy, PIL and scipy. Plain
  `python` lacks numpy. Torch isn't installed locally, so train on Modal.
- **Fast local rendering on macOS:**
  `SLOWSIM_ANGLE=metal SLOWSIM_CHANNEL=chrome SLOWSIM_GL=gpu` for any
  `sim/headless/*.mjs`.
  - Metal is ~20× faster than ANGLE's OpenGL path; frames match within rounding.
  - The default with no env is SwiftShader (CPU, ~0.3 s/frame). That's what
    Modal's `deploy/modal_gen.py` uses.
- **Playwright.** Its bundled Chromium is missing locally, hence
  `SLOWSIM_CHANNEL=chrome` (system Chrome).
- **Playable demo:** `python3 sim/serve.py 8777`, then open
  `http://localhost:8777/demo/index.html`.
  - Drive with WASD or the arrows. `K` toggles autopilot, `C` gives a clean
    view, `O` changes quality.
  - Guardrails are visual only: there is no collision.
- **Modal** (workspace/profile `slower-roads-m3`; volumes `sr-m3-train`,
  `sr-m3-val`, `sr-m3-test`):
  - Long jobs need `modal deploy`, then
    `modal.Function.from_name('sr-m3', ...).spawn(argv)`. A client-attached
    `modal run` gets cancelled when the client dies.
  - All entrypoints share `App("sr-m3")`, so **run Modal jobs sequentially**.
  - Context-8 rollout training needs **A100-80GB** (use
    `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, batch ≤ 8).
  - The old 2-minute oracle evals live at `/val/data/seed5` and
    `/test/data/seed2`. They are old-renderer, so regenerate them.
- **Token layout** (`model/dynamics/config.py`):
  - 16×16 = 256 visual tokens per frame.
  - 9 action tokens (3 steer × 3 throttle buckets) offset after 12,800 visual
    codes.
  - `FRAME_STRIDE = 257`.
  - Logged action `a_i` drives frame `i → i+1`.

## 4. Two constraints the plan is built around

1. **Real-time in the browser.** The current dynamics model decodes a frame
   **one token at a time** (256 sequential forward passes per frame). Even on a
   datacenter A10G it runs at ~0.5–0.8 frames/s, and the target is 30 fps in a
   browser on consumer GPUs. The generation scheme must change; model quality
   alone can't close that gap.
2. **"Zero drift."** A free-running generative model always drifts somewhat.
   The design that gets practical zero drift:
   - **Run the deterministic sim core in the browser next to the model.**
     It's tiny pure JS.
   - **Condition every frame on its oracle skeleton and state.** This is the M3
     skeleton-memory recipe, which already beat frozen-frame for 2 minutes on
     held-out data.

   Road layout, position and heading then can't wander, and the model's job is
   rendering them consistently. The measurable target is a **drift curve that
   stays flat over ≥10 minutes** on held-out worlds and beats frozen-frame and
   persistence baselines throughout.

## 5. The plan (do in this order)

### Step 1: WebGPU latency spike (≈1 day, do first)

> **Done 2026-10-01.** See `docs/WEBGPU_BUDGET.md` (harness: `web/bench/`, `export/`).
> Summary, measured on the M3 Max:
> - **Stay at 64 px / 16×16.**
> - **Use 1 refinement pass per frame (2 max)**, not 2–4.
> - **Size:** d384–512 × 4–6 layers, ctx 4.
> - **Use** fp16, the factorized FSQ head, a fused cache commit and ORT graph capture.
> - **Cost:** 11–16 ms/frame.

Goal: measure ms/frame for the shape we'd ship, before spending GPU hours.

- Build a dummy dynamics model with random weights at candidate sizes:
  - d_model 256–384, 4–8 layers,
  - context 4–8 frames × 257 tokens,
  - the frame-parallel attention mask from Step 3.
- Export it to ONNX, then run it in **ONNX Runtime Web (WebGPU execution
  provider) inside a Web Worker**. Serve with COOP/COEP headers, which WebGPU
  and SharedArrayBuffer need.
- Measure:
  - forward-pass latency per refinement pass,
  - KV-cache handling,
  - FSQ decode (tokens → 64px RGB).

  Test on the user's Apple-silicon Mac and, if possible, a weaker laptop or iGPU.
- **Output:** a table of (size, context, passes/frame) → ms/frame, plus a
  recommended configuration fitting **≤ 33 ms/frame** (30 fps) with headroom.
- This number sets model size, grid size and context for everything below.
  Put the spike in `web/` or `export/` (see `ROADMAP.md` §6), and note the
  ROADMAP §3 landmine: keep the ONNX graph splittable for later steering.

### Step 2: retrain the tokenizer on `train_v2`

> **Done 2026-10-01.** **Chosen: `tokenizer_v2_tc025`**, on the volume at
> `sr-v2-train:/checkpoints_v2_tc025/tokenizer.pt` and locally at
> `checkpoints/v2/`. Held-out val:
> - PSNR 31.1;
> - 27% of tokens change per frame;
> - 6.4% flip under 1% noise (the old tokenizer: 41% / 8%).
>
> **Also kept, for plug-and-play:**
> - base `tokenizer_v2` (PSNR 32.8, less stable; a candidate for the
>   continuous-latent branch);
> - `tc10` (over-smoothed).
>
> **Data:**
> - The split is `data/train_v2/split.json` (80 / 5 / 5, held out by seed).
> - tc025 latents are precomputed for all 90 episodes on `sr-v2-{train,val,test}`.
> - The pre-v2 tokenizers were deleted.
>
> Workflow: `docs/TRAINING.md` "v2 tokenizer". Before Step 3, read
> `docs/research/SYNTHESIS.md`. It reshapes Step 3 into a discrete-vs-continuous
> head bakeoff, with context corruption, a projected skeleton map and adaLN
> keys built in.

- The existing `checkpoints/tokenizer*.pt` learned the old renderer's look. The
  new frames have finer detail (grass, painterly trees, bright sky).
- Before uploading to Modal, shrink the data: convert frames to uint8 (the
  loader accepts uint8 CHW/HWC `.npy` and PNG) or pack per-episode arrays.
  17 GB of float32 is wasteful.
- Hold out ~10 episodes **by seed, across all profiles**, as val/test. Never
  train on them. Mirror the existing `sr-m3-val` / `sr-m3-test` discipline.
- Train with `model/train_tokenizer.py` (see `docs/VAE_RECIPE.md` for the
  shipped recipe, incl. the temporal-consistency term). Then check:
  - reconstruction quality (`eval/eval_tokenizer.py`),
  - temporal stability, which was the M2 root cause,
  - codebook usage.
- Stay at 64px / 16×16 unless Step 1 shows room for more tokens.
- Then run `model/precompute_latents.py` over all episodes.

### Step 3: make dynamics generate a whole frame per step

- Change from token-by-token AR to **frame-level causal, within-frame
  parallel**:
  - the attention mask is causal across frames and bidirectional within a
    frame;
  - predict all 256 tokens of the next frame together, refined MaskGIT-style
    in **2–4 passes** (or an equivalent few-step scheme).

  That's ~4 forward passes per frame instead of 256.
- What carries over: the FSQ tokens, the per-frame KV cache, the action token
  slot, the M3 skeleton-memory cross-attention and the state head. Build the
  memory/state pieces in from the start.
- Code is in `model/dynamics/ar_core.py` (+ `rollout_loss.py`,
  `train_dynamics.py`, `config.py`). Keep the action/frame alignment contract,
  which is guarded by tests in `model/dynamics/test_shapes.py`, and extend the
  tests for the new mask.

### Step 4: condition on raw keys, not the 3×3 action grid

- `train_v2` records the WASD keys held every frame (`samples[i].keys`). That's
  exactly what a player supplies at inference.
- The current 3×3 steer/throttle buckets of the *smoothed* action lose
  information and don't map 1:1 to key presses. Replace them with a key-state
  token: 4 bits → up to 16 combos.
- Non-keyboard episodes (`cruise`, half of `dial_mix`) have `keys: ""`. Either
  derive keys by thresholding the applied action or treat them as a separate
  "autopilot" token. Pick one and document it.
- Update `config.py` (vocab layout) and `model/data/dataset.py`
  (`tokenize_action`).

### Step 5: anti-drift training + long-horizon gates

- Reuse the M3 recipe on the new architecture: `--self-rollout`,
  `--corruption-cond`, skeleton memory (`--mem-cross-attn`), `--state-head`,
  and λ-scaled anchor conditioning.
- Build new eval oracles with the current renderer:
  - a few **10-minute** episodes (18,000 steps) on held-out seeds, via
    `generate_dataset.mjs --steps 18000`;
  - mixed profiles.
- Run `eval/eval_m3.py`-style evaluation: normalized drift AUC vs frozen and
  persistence, survival time, λ sweep.
- **Promotion gate:**
  - the drift curve stays flat out to 10 minutes;
  - it beats frozen-frame and persistence on held-out worlds for every profile;
  - λ=1 beats λ=0.
- Also check action responsiveness: does pressing A/D in a dream turn the view
  the way the oracle does? `eval/eval_steering.py` is the starting point.

### Step 6: ship in the browser

- A Web Worker runs the **sim core** (`sim/core/*`, unchanged, deterministic)
  plus the model.
  1. Each frame, apply the player's keys to the sim.
  2. Take `sim.skeleton()` and state as conditioning.
  3. Run the dynamics passes.
  4. FSQ-decode.
  5. Post the frame to the main thread.
- Also needed:
  - weights cached via the Cache API after the first download,
  - COOP/COEP headers,
  - warmup inference to absorb shader compilation,
  - a WebGPU → WASM fallback for weak devices.
- Host as a static bundle plus weights on a CDN, so the user's device does all
  the compute.
- Reuse `sim/demo/` UX (HUD, dials) where sensible. Measure an FPS
  distribution and a latency breakdown (tokenize / dynamics / decode) per
  device class; `ROADMAP.md` §7 treats this as a headline result.

## 6. Open questions / risks to raise with the user

- **Is 64px enough?** Real Slow Roads detail (grass, trees) is lost at 64px.
  Step 1's budget decides whether 128px (32×32 tokens) is affordable.
- **Off-road fraction.** `keys_explore` is 30% off-road on average, up to ~53%
  in single episodes. If the model should stay mostly on-road, down-weight or
  filter those episodes.
- **Dataset size.** 3 h is a start, larger and more varied than the old
  training seeds. Generating more is cheap (`--hours N`, resumable). Scale it
  if the tokenizer or dynamics underfit.
- **Fidelity gaps** (`docs/FIDELITY.md` "Remaining gaps"):
  - the ground band and car shape (ours is boxier),
  - mid-band colour and edges,
  - hills not tall enough,
  - no road signs,
  - night/weather unvalidated (the reference footage is daytime only).

  These are renderer polish, independent of the model work. More user
  recordings, especially night and weather, would help.
- Each episode starts at the beginning of its road, so the first ~1 s shows an
  edge-of-world view. Trim it, or start the policy after a warm-up.
- **Stale assets.** All checkpoints and old datasets (`data/seed*`, Modal
  volumes) predate the renderer and world changes. Treat them as legacy; don't
  mix them into new training.
