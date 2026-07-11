# Things to think about / research

Open questions where your judgment or a literature dive would change what we build.
Grouped by how soon they block progress. Nothing here is settled — push back.

## Decide soon (blocks the next build step)

1. **Latent grid size, from the frame budget — not fidelity.** The whole real-time
   claim hinges on this and the ROADMAP says decide it at M1 by working *backward*
   from 33 ms/frame. We provisionally set 64×64 pixels → 8×8 latent grid (64 tokens/
   frame). The AR core generates tokens one at a time, so tokens/frame is the direct
   fps lever. Question: what's the largest grid that still hits ~30fps on your
   laptop's GPU under the AR decode? This sets the tokenizer's downsample factor and
   can't be cheaply changed later. Worth a back-of-envelope: tokens/frame × per-token
   transformer latency vs 33 ms.

2. **AR next-token vs. your Schrödinger-bridge flow — which dynamics core?**
   I spec'd both in `docs/architecture.md §4`. AR is the safe path to a working M2
   and the steering work depends on its residual stream. Your flow idea is more
   novel and the FSQ-grid-snap is a genuinely nice anti-drift mechanism. Research
   angle: does anyone do discrete-target flow matching / Schrödinger bridges *onto a
   quantized codebook*? (Look at: discrete flow matching, "Bridge" models, and
   whether MeanFlow / consistency-style few-step sampling composes with the grid
   snap.) If the flow path has a real speed or drift edge, it changes the whole
   dynamics core — worth knowing before we invest in AR.

3. **Scene representation — is a 2D image even the right target?** (Your notes q.)
   The sim can emit RGB, depth, segmentation, or the sparse state vector from the
   same deterministic step. Cheaper reps = fewer codes wasted on pixel noise = the
   model spends capacity on *dynamics* not *rendering*. But too sparse and the "net
   dreams the frame" product thesis weakens. Question to resolve empirically once
   the tokenizer trains: reconstruction quality vs. token budget for RGB vs.
   depth+seg. The dataset flag is already built to A/B this.

## Think about (shapes design, not blocking yet)

4. **FSQ variants — verdict is in, but one open thread.** The study
   (`model/tokenizer/FSQ_VARIANTS.md`) showed your AEP intuition is *directionally
   right but small* (~1 dB) because `tanh` already companders. The untested claim:
   does that hold for the *actual trained encoder's* latent distribution, which may
   be heavier-tailed or multimodal than a Gaussian? Action: once the tokenizer
   trains on real sim frames, histogram the pre-quant latents per channel and re-run
   the study against *that empirical distribution*. If utilization is low, the cheap
   erf-compander seam is already in the code.

5. **Anti-drift is THE hard part — budget accordingly.** ROADMAP M3 is explicitly
   the risk. Read before we get there: Diffusion Forcing (per-token noise levels)
   and Self-Forcing (train on the model's own rollouts). Question: how do these
   compose with an AR *discrete-token* core (both were framed for continuous/
   diffusion models)? The FSQ-snap in your flow idea may be a discrete-native
   alternative to noise-augmentation — worth positioning against each other.

6. **The `λ` sim-anchor — how much conditioning is enough?** At λ=1 a cheap
   deterministic skeleton feeds ground-truth (road centerline, car pose) each frame
   so the road can't melt. Open: what's the *minimal* conditioning signal that kills
   drift? Full depth map, or just the centerline polyline + car pose? Less
   conditioning = more "pure" world model = better research story, but driftier.
   This is a dial we'll sweep for the paper's headline figure.

7. **Your dataloader/PPO idea — trust region on the tokenizer.** (Your notes.)
   Keeping the bridge/autoencoder from drifting off pretrained weights during
   rollout training. Research: is this a KL penalty toward frozen weights (simpler),
   or genuinely a PPO-style ratio clip (why would we need the RL machinery if
   there's no reward?)? Worth pinning down what the actual objective is — I suspect
   it's closer to a distillation/EWC regularizer than PPO, but if you're picturing a
   reward signal, that's a different design.

## Longer-horizon / paper-shaping

8. **Steering as the differentiator.** The interp claim (steer weather/gravity via
   activation directions) is what makes this novel vs. Oasis/GameNGen. Contrastive
   activation addition needs matched pairs — the sim gives those for free. Question:
   which sim params give *clean* steering vectors (weather, time-of-day — visual,
   likely easy) vs. *mushy* ones (gravity, friction — dynamical, likely hard, the M6
   stretch)? Reading: CAA, persona/steering vectors, SAEs for monosemantic features.

9. **Evaluation is the paper.** Because the sim is ground truth, every claim is a
   number. The figure no one else can make: drift (latent + pixel divergence from
   the oracle) vs. rollout length, swept over model size, decoding scheme, and λ.
   Think about what the x/y axes of your headline plot are *now* — it shapes what we
   log from day one.

## Quick calls I'd like your take on
- Keep the car **open-loop** (steers freely, road is scenery) or make it lane-follow?
  Currently open-loop, matching Slow Roads. Affects what "action" means to the model.
- Ship target `λ`: ROADMAP guesses near 1. Set from the M3 drift curve, or pick a
  gut default now for the first demo?
- Output resolution: start 64px (fastest to iterate) and scale up, or 128px from the
  start for a nicer demo? I lean 64 for iteration speed.
