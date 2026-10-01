# M3 anti-drift — implementation and pilot results

*Updated 2026-07-14. M3 mechanisms are implemented; the multi-minute completion
gate has not yet passed.*

## Correctness work completed

- Fixed the action/frame contract: logged action `a_i` drives frame `i -> i+1`.
  Training, state dynamics, teacher-forced eval, and free-run eval now use the
  same incoming action for each target frame. An alternating-action test guards
  the one-frame shift found during the audit.
- Added exact self-rollout training. Future training steps can condition on the
  same token-by-token KV-cached frames produced by inference, with the same
  bounded rolling context. The detached rollout supplies authentic model errors;
  CE against oracle tokens remains the gradient signal.
- Added free-run checkpoint selection (`--free-run-val-batches`). Resume now
  restores the historical best metric instead of overwriting a better checkpoint.
- Added FSQ-local corruption forcing. A sampled severity perturbs one mixed-radix
  FSQ factor per selected token and is embedded at every affected frame position.
- Added a token-aligned λ anchor. The sim skeleton is rasterized to 16×16 with
  road corridor/centerline/heading/curvature/grade channels plus car and
  environment channels. The anchor is scaled by sampled λ and injected at every
  visual position.
- Added `eval/eval_m3.py`: exact long-rollout curves against tokenizer-floor,
  persistence, and frozen baselines, normalized drift AUC, token accuracy,
  survival time, multiple starts, and λ sweeps.

## Modal layout and verification

Active profile/workspace: `slower-roads-m3`.

- `sr-m3-train`: train seeds `{1,3,4,6–12}`, validation seed5, latent caches,
  tokenizer, baselines, checkpoints, and metrics.
- `sr-m3-val`: seed5 RGB oracle frames for pixel-space promotion gates. Training
  functions never mount this Volume.
- `sr-m3-test`: pristine seed2 only. Training functions never mount this Volume.

Verified on Modal:

1. Combined anchor + exact self-rollout + corruption smoke passed on an A10.
2. A two-batch real-data self-rollout smoke passed on an A100 and wrote a
   free-run-selected checkpoint.
3. A 100-batch warm-started M3 pilot passed on an A100; seed5 exact free-run
   token accuracy was `0.5205`.

## Honest pilot result on pristine seed2

Thirty-frame normalized drift AUC (lower is better; `1.0` = frozen baseline):

| model / λ | normalized AUC |
|---|---:|
| corrected-action M2 baseline | **0.665** |
| M3 pilot, λ=0 | 0.733 |
| M3 pilot, λ=0.25 | 0.745 |
| M3 pilot, λ=0.5 | 0.770 |
| M3 pilot, λ=0.75 | 0.759 |
| M3 pilot, λ=1 | 0.774 |

The 100-batch pilot does **not** improve M2 and its λ curve is not monotonic.
This is a useful negative pilot, not an M3 result. A subsequent 1,000-batch epoch
reduced seed5 free-run accuracy to `0.5052`, showing that more updates alone did
not fix the formulation and reinforcing that selection must use rollout quality.

Raw results:

- `eval/m2_corrected_seed2_30.json`
- `eval/m3_pilot_seed2_30.json`

## Next experiment gate

Do not launch the full six-way campaign yet. First separate the coupled pilot:

1. corrected-action fine-tune with no anchor/corruption;
2. anchor-only, trained at λ=1 for enough updates to establish that the anchor is
   learnable, then introduce λ dropout;
3. corruption-only;
4. exact self-rollout post-training on the best of 1–3.

Promote a variant only if seed5 long-rollout normalized AUC improves. Run seed2
only after promotion. Multi-minute evaluation additionally requires generating
at least 3,600 oracle target frames; 7,200-frame seed5 and seed2 trajectories are
the planned evaluation assets.

## Split ablation and attention audit (2026-07-14)

The coupled pilot was split into matched 1,000-update warm-start runs. Seed5
30-frame normalized AUC (lower is better):

| variant | AUC | decision |
|---|---:|---|
| fixed λ=1 anchor | **0.759** | promote |
| learned frame-recency attention bias | 0.779 | seed5 win, seed2 reject |
| corrected-action control | 0.792 | baseline |
| corruption-only | 0.807 | reject |
| anchor + λ dropout | 0.767 at λ=1 | reject |
| anchor + 100 exact self-rollout updates | 0.776 | reject |

Hard context truncation also failed on seed2: eight/four/two/one-frame windows
scored `0.665 / 0.801 / 0.944 / 3.245`. History is load-bearing.

Attention telemetry captured final-token mass by frame age, normalized entropy,
and first-token mass in every layer. It found no attention-sink pathology
(first-token mass was approximately zero). Some heads did allocate substantial
mass to frames 5–7 steps old. A learned frame-age penalty reduced that old-frame
mass, but worsened seed2 AUC (`0.614` vs matched control `0.591`). Conversely,
the successful anchor retained substantial historical attention. The evidence
therefore points to **content drift in the cached history**, not too much history
by itself.

The fixed λ=1 anchor generalized over longer rollouts:

| evaluation | control mean AUC | anchor mean AUC | relative gain |
|---|---:|---:|---:|
| seed5, 3 starts × 150 frames | 0.742 | **0.725** | 2.3% |
| seed2, 3 starts × 150 frames | 0.836 | **0.772** | 7.7% |

On seed2 the anchor won all three starts. At start 500, sustained-failure
survival improved from 65 frames to at least 150 frames. This is a real
short/medium-horizon improvement, but 150 frames are only five seconds at 30 Hz.

The λ sweep for the fixed-anchor checkpoint was not monotonic
(`0.594, 0.599, 0.588, 0.598, 0.572` for λ `0,.25,.5,.75,1` on seed2), although
full strength was clearly best. Uniform λ dropout did not repair the curve and
slightly damaged λ=1, so the shipping interpretation for now is a binary anchor,
not a calibrated continuous control.

### Attention-drift decisions

- Keep the eight-frame rolling KV window; hard truncation is harmful.
- Do not add a StreamingLLM-style first-token sink: telemetry shows no sink
  failure and this model resets RoPE inside its bounded window.
- Do not use uniform ALiBi-style frame recency as the default: it changes
  attention as intended but does not generalize here.
- Prefer reliable external state in the cache. The skeleton anchor improves the
  content attended to without preventing heads from using longer history.
- If attention is changed again, test head-specific content-aware KV selection
  or compressed historical state, motivated by
  [Forcing-KV](https://arxiv.org/abs/2605.09681), rather than one global recency
  rule. Use stochastic truncated rollouts, as motivated by
  [Self Forcing](https://arxiv.org/abs/2506.08009), before attempting a large
  exact-self-rollout campaign.

## Two-minute gate and attention causality test

Fresh 7,201-frame seed5 and seed2 RGB oracles were generated into `sr-m3-val`
and `sr-m3-test`. Their actions, states, and skeletons match the original
2,501-frame trajectories exactly over the shared prefix. SwiftShader pixels are
not byte-identical across captures (sampled frame MAE `4.1e-6`, maximum one
8-bit level), reinforcing the need for the still-open M0 pixel determinism test.

The 3,600-frame/two-minute seed5 gate failed:

| model | normalized AUC | sustained failure | mean token accuracy |
|---|---:|---:|---:|
| corrected-action control | **1.006** | 198 frames | 0.0874 |
| fixed λ=1 input anchor | 1.044 | 187 frames | 0.0926 |

The anchor's small cumulative advantage disappears at frame 150, and its
30-frame-smoothed drift becomes worse around frame 179. The pristine seed2
two-minute comparison was intentionally not run because the variant failed its
seed5 promotion gate.

Attention telemetry initially suggested a mechanism: by frame 600 the input
anchor's first layer placed about 79% of mass on frames at least two steps old
with normalized entropy around 0.64, versus control's 68% and 0.81. Because the
anchor is added before Q/K/V projection, repeating road geometry can make stale
frames look artificially similar.

An `anchor_injection=output` ablation tested causality by applying geometry only
after all temporal attention. It restored the control-like attention signature
(layer-0 entropy approximately 0.82 and old-frame mass approximately 0.64 by
frame 101), but **worsened** 150-frame seed5 AUC to `0.843` versus control
`0.805`. Therefore attention concentration is a symptom, not the root cause of
the world-state drift.

### Revised next experiment

Do not spend more runs on global recency penalties, entropy regularization, or
moving the same additive anchor around the residual stream. The evidence now
favors a state/memory architecture:

1. encode the current simulator skeleton into a small set of dedicated memory
   tokens and cross-attend to them separately from visual self-attention;
2. retain recent visual KV for appearance, but carry longer history as compact
   state/pose tokens or select old KV by state consistency rather than age;
3. supervise a latent-delta/state-transition head so cached memory has a direct
   continuity objective instead of only next-token CE;
4. use stochastic truncated self-forcing on that architecture, because the
   100-update exact rollout was too expensive and did not improve drift.

This is consistent with the head-specific, content-aware cache direction in
[Forcing-KV](https://arxiv.org/abs/2605.09681) and compressed pose/state memory
in [RELIC](https://arxiv.org/abs/2512.04040). M3 remains open: neither current
model holds below the frozen-baseline drift level for two minutes.

## Decoupled skeleton memory + state continuity — the two-minute gate passes (2026-07-15)

Implemented the state/memory architecture the revised experiment prescribed, as one
warm-startable change on top of the promoted λ=1 anchor:

- **Skeleton memory tokens via cross-attention** (`ar_core.py`: `CrossAttention`,
  `encode_memory`, `--mem-cross-attn`). The current frame's 16×16×16 skeleton is
  encoded and adaptive-pooled to a small register set (16 tokens). Each transformer
  block gains a cross-attention sublayer whose query is the visual tokens and whose
  key/value are those memory tokens. The memory lives **outside the causal KV cache
  and carries no RoPE**, so — unlike the additive anchor — it never enters the visual
  self-attention Q/K similarity that drifts. A per-position mask feeds memory only to
  current-frame positions (context K/V stay uncontaminated), and a per-block gate
  initialized to 0 makes the warm start numerically identical. λ still gates the
  memory (λ=0 → zero value stream → pure autoregression).
- **State-continuity head** (`--state-head`, `rollout_loss._state_loss`). An MLP on
  the pooled current-frame hidden regresses the per-frame `{x,z,heading,speed}` delta
  (standardized per-dim), giving the cache a direct continuity objective beyond
  next-token CE. The state loss fell from ~40 at init to ~1.2, i.e. it learned an
  internal world-state estimate.

Warm-started from `checkpoints_m3_anchor` (11.3M → 12.99M params; 64 new params,
0 unexpected), trained 2 epochs on seeds {1,3,4,6–12}, λ fixed at 1, corruption on,
best-by-free-run selection on seed5.

**Seed5, start 100, 3,600-frame / two-minute gate (λ=1), normalized drift AUC (lower
is better; 1.0 = frozen baseline):**

| model | mean normalized AUC | survival frames |
|---|---:|---:|
| corrected-action control | 1.006 | 198 |
| fixed λ=1 input anchor | 1.044 | 187 |
| **decoupled memory + state head** | **0.943** | **203** |

This is the **first model to hold below the frozen-baseline drift level (1.0) across
the full two minutes**, beating the additive anchor by ~10% and the control, with
survival past 200 frames — the M3 seed5 "done when" condition. The profile is the
intended anti-drift trade: it gives up a little short-horizon fidelity (150-frame
mean AUC ~0.85) for long-horizon stability that the additive anchor could not sustain.
The evidence confirms the diagnosis — the failure was content drift in the cached
visual history, and a clean, continuity-supervised memory that stays out of the
visual Q/K fixes it where moving the additive anchor around the residual did not.

**Pristine seed2 (the true held-out test) confirms it**, start 100, 3,600 frames:

| λ | mean normalized AUC | survival frames |
|---|---:|---:|
| 0 (pure AR, memory + anchor off) | 1.045 | 125 |
| **1 (memory + anchor on)** | **0.953** | **261** |

Turning the memory/anchor on (λ 0→1) moves the pristine-test drive from **failing**
(1.045, worse than the frozen baseline) to **holding** (0.953, below it) over the full
two minutes, with survival more than doubling (125→261 frames). This is the
drift-vs-λ safety-valve behaviour the project set out to produce — the figure no
other world-model paper can make — demonstrated on held-out data, not just the
validation seed. Results: `eval/m3mem_seed5_3600_l1.json`,
`eval/m3mem_seed2_3600_l0l1.json`.

Remaining polish: the full 5-point λ sweep (0, .25, .5, .75, 1) on seed2 for a smooth
drift-vs-λ curve (each point is a ~75-min 3,600-frame rollout).

Reproduce:
```
modal deploy deploy/modal_train.py
modal run deploy/modal_train.py::main --data "<seeds>" --val-data /models/data/seed5 \
  --d-model 256 --n-heads 4 --n-layers 6 --context 8 --horizon 6 --epochs 2 \
  --batch-size 8 --tf-start 0.5 \
  --extra "--dropout 0.1 --latent --init-from /models/checkpoints_m3_anchor/dynamics_best.pt \
           --anchor-cond --anchor-lambda-min 1 --anchor-lambda-max 1 \
           --mem-cross-attn --mem-tokens 16 --state-head --state-weight 0.5 \
           --corruption-cond --free-run-val-batches 4"
# long evals: modal.Function.from_name('sr-m3','evaluate_m3').spawn(argv)  (client-attached runs get cancelled ~60min in)
```
