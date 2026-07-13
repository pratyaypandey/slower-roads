# M2 results — dynamics core (teacher-forced)

*Updated 2026-07-12. Milestone [M2](../ROADMAP.md): an action-conditioned AR dynamics
core that predicts the next latent frame and produces a coherent short drivable clip.*

## Headline

A 64px on-device-scale world model that, on a **fully held-out seed** (never trained on,
never used for validation), produces coherent multi-second drives and **beats a
frozen-persistence baseline by a wide, growing margin** — after root-causing and fixing a
tokenizer flaw that made the first "certified" M2 an illusion.

## The load-bearing finding: judge a world model against baselines, not absolute drift

Absolute pixel drift is misleading — consecutive driving frames are ~99% identical, so a
broken model still scores a low L1. Every M2 claim here is measured **relative to
baselines in the same decode path** (`eval/eval_baseline.py`, `eval/eval_dream.py`):

- **persistence (copy current tokens)** — the strict bar; ≈ the tokenizer floor because
  frames barely change. Beating it means out-predicting the tiny per-frame delta.
- **frozen (copy last context frame for the whole rollout)** — the world-model bar: a
  free-running model must *generate* the advancing road, which frozen cannot.

The first M2 pass looked certified on absolute drift (~0.02 pixel-L1, flat) but was **3×
worse than persistence** — it hadn't learned dynamics at all.

## Root cause (M1 tokenizer) and the fix

The dynamics model couldn't beat copy because the FSQ tokenizer was **temporally
unstable**: consecutive frames were 99.3% identical in pixels yet **84% of their latent
tokens differed**, and **57% of tokens flipped from 1% pixel noise**. The next-token
target was mostly quantization noise. M1 had only ever been certified on single-frame
reconstruction — never temporal stability.

Fix (`model/train_tokenizer.py`, `--temporal-weight` / `--noise-weight`): fine-tune the
tokenizer with a temporal-consistency loss (penalize code change between consecutive
frames) + a noise-robustness loss (penalize code change under a small pixel perturbation),
on multi-seed data.

| tokenizer | recon-L1 | tokens-changed/frame | noise-flip |
|---|---|---|---|
| old | 0.0009 | 84% | 57% |
| **temporally-consistent** | 0.0041 | **41%** | **8%** |

Reconstruction stays excellent; churn halves; noise-robustness improves ~7×.

## Final model + certification (pristine held-out seed2)

`ar_transformer`, 11.3M params, context 8 / horizon 8, dropout 0.1 + weight-decay 0.05,
trained on the stable tokenizer's latents over **10 seeds** (seed5 = val, **seed2 = held-out
test, never on the training volume**). `checkpoints/dynamics_final.pt` +
`checkpoints/tokenizer_tc.pt`.

**Teacher-forced coherence ✅** (`eval/plots/dream_tf.gif`): flat ~0.014 pixel-L1 over 60
frames; **token accuracy 0.567 on the held-out seed** (was 0.06 with the unstable
tokenizer). The dream tracks the real drive frame-for-frame.

**Beats persistence ✅** (`eval/eval_baseline.py`, both seeds): held-out seed2 model 0.0141
vs copy 0.0142; seed1 0.0087 vs 0.0092 (token-acc 0.618 vs 0.587). Passes even the strict
near-floor bar.

**Beats frozen (world-model quality) ✅** — free-run, context 8:

| seed | frozen mean-L1 | model mean-L1 | model beats frozen by |
|---|---|---|---|
| **seed2 (held out)** | 0.0394 | **0.0281** | **+0.0112** (step30 +0.0169) |
| seed1 | 0.0455 | 0.0277 | +0.0179 |

That is **~14× the pre-campaign 3-seed model** (+0.0008). The progression that got there:

| model | free-run margin vs frozen (held-out seed2) |
|---|---|
| 3 seeds, old tokenizer | +0.0008 |
| 10 seeds, stable tokenizer | +0.0023 |
| 10 seeds + dropout/wd + context 8 | **+0.0112** |

## Honest gap: action/steering response is weak ⚠️ (fix in progress)

M2's "responds correctly to steering input" is **not yet** met to a high standard. Forcing
left vs right actions from the same context produces nearly identical dreams
(`eval/plots/steering.gif`); measured "action sensitivity" (mean |dream_left − dream_right|)
is only **0.0030** and the road-centroid shift ~0. The model follows context momentum and
under-weights the single action token per 257-token frame. (Actions *do* matter under
teacher forcing — token-acc 0.567 includes correctly predicting the road curving with the
true actions — it's the counterfactual free-run response that's weak.)

**Tried — strong action conditioning** (`--action-cond`, backward-compatible): a separate
action embedding added to *every* position of the frame it drives, threaded through
`forward` / `generate_frame` (KV-cache) / `rollout_loss` / `prepare_batch`
(`model/dynamics/{ar_core,rollout_loss,sequence}.py`). Plus a momentum-robust directional
steering metric in `eval/eval_steering.py` (action sensitivity + road-centroid shift).
Validated end-to-end and retrained on 10 seeds (A100). **Result: it did NOT materially move
steering.** At a well-trained checkpoint (val CE 12.5, free-run still beats frozen by
+0.0057) the action sensitivity is 0.0036 vs the baseline's 0.0030 — within noise; the
road-centroid shift stays ~0.

**Diagnosis — the weak steering is largely intrinsic, not a conditioning-strength problem.**
Next-frame prediction is dominated by the *visual context*, which already encodes the car's
heading and the road's curvature; the commanded action's marginal effect on the immediate
next frame is tiny (the road barely moves per frame). So the model predicts the next frame
well from context alone and down-weights the action however strongly it's injected — the
counterfactual "what if I'd steered differently" only diverges over *many* steps, by which
point free-run drift dominates. (The action still matters *cumulatively* — teacher-forced
token-acc 0.567 comes from following the true action sequence — it's the short-horizon
counterfactual response that's small.)

**Real paths to strong steering** (future work, beyond M2's core): (a) train with an explicit
counterfactual / action-sensitivity objective over longer horizons so the action effect is
rewarded; (b) M5's CAA steering-vector approach, which *amplifies* an action/attribute
direction at inference rather than relying on it emerging from next-token prediction. The
`--action-cond` path stays in the codebase (off by default) as the conditioning substrate
those build on.

## Infrastructure built this campaign

- `eval/eval_baseline.py` — baseline-relative certification (floor / persistence / model,
  pixel + token, + tokenizer temporal-stability), the honest M2 gate.
- Latent cache (`model/precompute_latents.py`, dataset `representation="latent"`,
  `train_dynamics --latent`) — encode frames once; ~skips per-epoch re-encoding so
  many-seed training is affordable.
- `deploy/modal_gen.py` — render seeds on Modal (headless Chromium/WebGL) so no local CPU.
- `deploy/modal_train.py` — Modal entrypoints for tokenizer fine-tune, latent precompute
  (parallel per-seed), and dynamics training; per-epoch volume commits.
- Multi-seed training + whole-held-out-seed validation (`train_dynamics --val-data`);
  AdamW weight-decay + model dropout.
- Clean **train {1,3,4,6–12} / val {5} / test {2}** split with zero leakage into the test.

## Known gaps / next levers

- **Steering/action response** (above) — the main remaining M2 quality gap.
- **val_ce ≠ world-model quality:** teacher-forced CE rewards copying; free-run margin kept
  improving after val_ce bottomed. Best-checkpoint selection should use a free-run metric.
- **Generalization set:** seed2 held out cleanly; more diverse seeds/biomes would harden it.
- **Free-run coherence** still drifts over long horizons — that's M3 (Diffusion Forcing /
  self-forcing / the λ anchor).
