# World-model research: synthesis for Slower Roads (2026-10-01)

## Where this comes from

Two independent deep-research passes ran on the same brief (`BRIEF.md`):

- `codex_world_models.md`: Codex `gpt-5.6-sol`, live web search, ~25 systems,
  77 links.
- `claude_world_models.md`: a Claude research subagent, ~60 works,
  method-level reads of ~10.

Neither saw the other's report. Where they agree, the agreement is the strong
signal. Three high-stakes claims were spot-checked against primary sources:

- **Orbis** (arXiv 2507.13162): continuous beats discrete on driving data.
  Confirmed in the abstract.
- **NFD** (arXiv 2506.01380): 310M params, 30+ fps on an A100, few-step
  consistency, block-causal. Confirmed.
- **Browser Flappy Bird world model** (njkumar blog): a 5M-param, 1-step EDM
  denoiser at 30 fps on an M2 Pro through WebGPU, and 12–15 fps on an iPhone
  14 Pro. Confirmed.

## Verdict

Our design direction holds up:
- frame-parallel prediction;
- 1–2 passes;
- a small, shallow transformer;
- the sim running alongside as an oracle;
- skeleton memory outside the KV cache.

Neither report found a better-fitting published system. Both point to the same
upgrades, and to one open architectural question: **discrete FSQ tokens or
continuous latents?**

## Ranked plan (both reports agree unless marked)

| # | Change | What we do | Why (evidence) | Inference cost |
|---|---|---|---|---|
| 1 | **Corrupted + self-generated context, with a corruption-level input** | Train on clean, corrupted (±1 FSQ level and the model's real wrong tokens) and self-rolled context frames. Feed a small fixed level at inference. Curriculum from 4–8 to 16–32 frame rollouts. We already have `--corruption-cond` and `--self-rollout` from M3. | GameNGen: without context noise, quality "degrades quickly after 10–20 frames". Dreamer 4, NFD, Matrix-Game 3 and Self-Forcing all do a variant. | none |
| 2 | **Make the oracle authoritative and spatially aligned** | Project the sim's road skeleton through the sim camera into a 16×16 map (road mask, lane offset, depth), added to each token's embedding. Feed absolute state (pose, speed, road-relative position) every frame from the sim. The state head stays an auxiliary loss only. | GAIA-2 (road semantics), AlayaRenderer-Flash (31.5 fps from engine buffers), OmniDreams. Geometry gets re-pinned every frame, so drift can only appear in appearance. | ~0 (one add) |
| 3 | **Discrete vs continuous bakeoff** (the big fork) | Same backbone, two heads: (a) FSQ categorical (factorized fsq head), and (b) continuous regression of the pre-quantization FSQ latent, trained with Dreamer-4 "shortcut forcing" / an sCM-style few-step objective. 1 pass, optional 2nd. Decoder fine-tuned to accept off-grid latents. | Orbis (driving): continuous "consistently outperforms" discrete, and discrete over-copies the previous frame (~45%). All fast, high-quality systems (NFD, Dreamer 4, Oasis, Matrix-Game, the browser Flappy Bird) are continuous. Continuous output also deletes the 12,800-way head (−1.3–2 ms) and the token-flip problem. | slightly cheaper |
| 4 | **Stabilize the latent space** | Fine-tunes now running (consistency weight 0.25 / 1.0). Next: EQ-VAE-style equivariance (noised/shifted input → same latent), FSQ boundary-margin loss, decoder trained on perturbed codes, masked-patch training (Dreamer 4). Mask motion boundaries with sim geometry. | Our measured flips (33% near-static, 13% under noise); Cosmos tokenizer, Dreamer 4 tokenizer. | none (encoder is offline) |
| 5 | **Action injection by adaLN(-Zero)/FiLM in every block** | Embed the 4 WASD bits, the key transitions and a short key history. Modulate each block (and the state head). | NFD ablation: adaLN-Zero beats cross-attn and in-context tokens. Matrix-Game 2 action module. | ~0 |
| 6 | **Bounded, optional 2nd pass** (Codex) | Refine only the most uncertain fixed regions or a capped token set. Skip it on slow devices. | MaskGIT, WHAMM | +0–1 pass, adaptive |
| 7 | **Persistent anchor frame + global weather/time inputs** (Claude) | One anchor frame always in context, plus env dials as globals, against slow colour/sky drift over 10 min. | Long-horizon systems (RELIC, Matrix-Game 3) | ~+0.8 ms per context frame; measure first |
| 8 | **10-minute adversarial eval + recovery data** | Search seeds and action sequences that maximize flips, state error, action lag and discontinuity. Add off-centre-and-recover episodes. Measure responsiveness against the sim's own response to the same keys. | PROWL-1, ReWorld; both reports | none |
| 9 | **Cheap efficiency** (Claude) | Grouped-query attention, register tokens; speculative next frame while keys are unchanged (NFD, 1.26×). | NFD | saves ms |

## Both reports say no

- **Distilling from 1B+ video DiTs** (Wan, Matrix-Game, CausVid, Self-Forcing
  teachers) or using pretrained foundation models.
- **4+ step diffusion as the shipped default.** 4 × 11–16 ms already blows the
  budget.
- **Classifier-free guidance at inference.** It doubles the passes.
- **Temporal-compression VAEs or chunk-wise generation.** They add input lag.
- **Memory banks / 3D memory.** Slow Roads never revisits places, and the sim
  already is the 3D memory.
- **Others:** DINO/V-JEPA encoders in the loop, image-space diffusion, 128 px,
  12-layer models, 1D (TiTok) tokenizers, and INT8 in ORT Web (unproven there).

## How this changes Step 3

Build the frame-parallel dynamics with the output head as a switch:
- `--head fsq` (categorical, factorized);
- `--head cont` (continuous latent, shortcut/consistency objective, 1–2 steps).

Then run the bakeoff at the budget size (d384, L6, ctx 4) on `train_v2` latents.
Bake in from day one:
- #1, context corruption + self-rollout;
- #2, the projected skeleton map + absolute state;
- #5, adaLN key conditioning;
- the M3 skeleton memory and state head.

The gate is the 10-minute drift curve plus responsiveness. Both heads stay
within the measured browser budget. The continuous head is slightly cheaper.
