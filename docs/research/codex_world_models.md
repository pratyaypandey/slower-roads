# What Slower Roads can borrow from recent world-model work

Cutoff: **1 October 2026**. I used papers, official project pages, official technical reports, and official repositories. “Real-time” numbers are not directly comparable: resolution, frame batching, first-frame latency, decoder cost, hardware, quantization, and proprietary kernels differ substantially.

The central conclusion is:

> Slower Roads should not imitate the large diffusion world models architecturally. It should borrow their training tricks—imperfect-context training, self-rollouts, explicit geometric state, and global action modulation—while retaining a much smaller one-pass next-frame model. Given the measured FSQ instability, a continuous-latent version is the highest-value experiment.

## 1. Architectures of interactive and real-time world models

Newest first.

| Work | Representation and dynamics | Actions and drift control | Reported speed / scale | Relevance |
|---|---|---|---|---|
| [ReWorld](https://arxiv.org/abs/2608.23565), arXiv Aug 2026 | Autoregressive diffusion backbone; mixed local/global attention heads; fixed KV cache plus a pose-indexed landmark bank. DMD-LoRA reduces inference to **4 steps**. | Physical-scale-aligned keyboard actions; palindrome trajectories teach revisits; pose retrieves nearby landmarks. | 704×1280 streaming; model size, fps and hardware not disclosed in the abstract. A 12-chunk cache recalls the start after a 64 s/384-latent out-and-back rollout. | Strong memory design, but four full passes and pose-memory machinery are unnecessary for an endless road with oracle geometry. |
| [Self Gradient Forcing](https://arxiv.org/abs/2607.20368), arXiv Jul 2026 | Two-pass training: a no-gradient autoregressive rollout followed by parallel reconstruction that lets future losses supervise earlier KV representations. | Explicitly addresses the “historical context-gradient gap.” | Reports minute-scale extrapolation from five-second training windows; deployment speed not the focus. | A practical improvement over full backpropagation through long rollouts. |
| [Wonder](https://arxiv.org/abs/2607.26037), arXiv Jul 2026 | Rectified self-forcing distillation, dense coordinate fields and sparse memory attention. | Camera control plus memory retrieval whose cost is independent of total history length. | **16 fps**; parameters and hardware not disclosed in the abstract. | Coordinate-conditioned consistency is relevant; scale/runtime are not browser evidence. |
| [ABot-World-0](https://arxiv.org/abs/2607.19191), arXiv Jul 2026 | Bidirectional teacher distilled to a causal student using teacher forcing and ODE distillation; “LongForcing” aligns extended student rollouts with a long-horizon teacher. | Raw keyboard actions; reference-character memory. | Up to **16 fps at 720p on RTX 5090**, about **19 GiB**, 1.2 s action-to-first-frame latency. | Long-rollout training is relevant; deployment footprint is not. |
| [DreamForge-World 0.1](https://arxiv.org/abs/2606.30292), arXiv Jun 2026 | Adaptation of the Wan2.1-T2V-1.3B/LongLive stack with a residual action pathway. | Live keyboard/mouse, reprompting and dual views. | **14–15 fps at 480p on RTX 4090**. | One of the more compute-conscious systems, but still far outside a laptop-browser budget. |
| [Matrix-Game 3.0](https://arxiv.org/abs/2604.08995), arXiv Apr 2026, revised Sep 2026 | Causal video diffusion; multi-segment DMD; 5B and 2×14B variants; quantization and VAE-decoder pruning. | Models prediction residuals and reinjects imperfect generated frames during training; camera-aware memory retrieval. | Up to **40 fps, 720p, 5B**; hardware not stated in the paper abstract. | Its imperfect-frame reinjection is highly transferable. The headline fps depends on a deployment stack orders of magnitude larger than Slower Roads. |
| [Interactive World Simulator](https://arxiv.org/abs/2603.08546), arXiv Mar 2026 | CNN encoder → compact continuous 2D latent; consistency-model decoder; action-conditioned consistency model with 3D convolutions, FiLM and spatiotemporal attention. Fixed-length history. | Next-frame latent supervision plus small noise on context latents. | **128×128, 15 fps on one RTX 4090**, stable interactive rollouts for **over 10 minutes**. Parameters and exact NFE count are not disclosed. | The closest conceptual match: continuous latents, next-frame prediction, noisy contexts and long fixed-window rollouts. |
| [RELIC](https://arxiv.org/abs/2512.04040), arXiv Dec 2025 | Autoregressive distilled video diffusion. Historical latent tokens are strongly compressed and tagged with relative actions and absolute camera poses inside the KV cache. | Memory-efficient self-forcing with long teacher and student rollouts. | **14B, 16 fps**; hardware not disclosed in the abstract. | Supports attaching pose/action metadata to memory, but 14B is irrelevant to browser deployment. |
| [GAIA-3](https://wayve.ai/thinking/gaia-3/), Wayve Dec 2025 | 15B latent-diffusion driving world model and a redesigned tokenizer twice GAIA-2’s size. | Structured autonomous-driving conditioning and evaluation. | Offline evaluation model; no interactive fps disclosed. | Useful evidence for structured conditioning, not for runtime architecture. |
| [Dreamer 4](https://arxiv.org/html/2509.24527v1), arXiv Sep 2025 | Causal continuous video tokenizer; block-causal dynamics with spatial attention and temporal attention only every fourth layer, plus GQA. Shortcut forcing uses **4 NFEs/frame**. | Separate tokens for keyboard/mouse components; past latents are slightly noised at inference; x-prediction was more stable than velocity prediction. | “Real-time on one GPU”; parameters, fps and GPU model are not reported. | Its block-causal layout closely validates Slower Roads’ frame-causal/bidirectional-within-frame design. Four passes do not fit your browser timings. |
| [Matrix-Game 2.0](https://arxiv.org/abs/2508.13009), arXiv Aug 2025 | Few-step causal autoregressive diffusion, distilled from a slower model. | Frame-level keyboard/mouse injection; trained on about **1,200 h** of UE/GTA5 data. | **25 fps**, minute-scale video; hardware and exact steps not reported in the abstract. | Relevant teacher→few-step recipe, not a small-device result. |
| [Genie 3](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/), DeepMind Aug 2025 | Officially described as frame-by-frame autoregressive generation; detailed tokenizer, parameters, NFE count and hardware remain undisclosed. | Real-time navigation plus promptable events; interactions can remain in memory for up to about a minute. | **720p, 24 fps**, worlds largely consistent for several minutes. | Impressive system result, but insufficient technical disclosure to reproduce or size against Slower Roads. |
| [Yume](https://arxiv.org/abs/2507.17744), arXiv Jul 2025 | Masked Video Diffusion Transformer with memory; Time-Travel SDE sampling; adversarial distillation and caching. | Keyboard actions are derived from quantized camera motions; an anti-artifact sampler corrects generation failures. | No verified fps, parameters or NFE count in the paper abstract. | Too sampling-heavy; camera-action quantization is more applicable than the generator. |
| [Hunyuan-GameCraft](https://arxiv.org/abs/2506.17201), arXiv Jun 2025 | Diffusion video model with hybrid history conditioning and distillation. | Keyboard and mouse are unified in a camera-motion representation. Trained on over one million recordings from more than 100 games, then refined with synthetic annotated data. | No verified model size, fps or step count in the abstract. | Supports expressing actions in physical/camera coordinates instead of opaque button IDs. |
| [V-JEPA 2-AC](https://arxiv.org/abs/2506.09985), arXiv Jun 2025 | Predictive latent model rather than a pixel generator; action-conditioned post-training from under 62 h of robot video after large-scale observation-only pretraining. | Actions condition predictions used for planning. | No generative-video fps reported. | Relevant if Slower Roads later separates predictive state from rendering, but it does not replace the current decoder. |
| [WHAM-RT](https://www.microsoft.com/en-us/research/project/wham/wham-rt/), Microsoft 2025 | Replaces WHAM’s token-by-token autoregression with MaskGIT-style parallel prediction. | Player controls; trained for Quake II using one week of curated gameplay. | **10+ fps at 640×360**. Hardware, model size and iteration count are not disclosed. | Direct evidence that parallel masked frames are much faster than token AR, but not that one MaskGIT iteration is sufficient. |
| [MineWorld](https://arxiv.org/abs/2504.08388), arXiv Apr 2025 | Discrete image and action tokenizers; interleaved visual/action sequence; autoregressive transformer with spatially parallel decoding. | Actions are explicit discrete tokens; action-following metrics accompany visual metrics. | **4–7 fps** across released model sizes; the official repository provides 300M/700M/1.2B checkpoints. | Even spatially parallel discrete AR is too slow; reinforces predicting the entire frame jointly. |
| [GAIA-2](https://wayve.ai/wp-content/uploads/2025/03/GAIA_2_Technical_Report.pdf), Wayve Mar 2025 | Continuous video tokenizer plus an **8.4B**, 22-block space-time transformer trained with flow matching. | Ego action and diffusion time enter every block through adaptive layer norm; other structured variables use cross-attention. Wayve reports action conditioning was more accurate with adaLN than cross-attention. | Offline generator; fps and few-step deployment not reported. | The strongest direct support for adaLN/global modulation for low-dimensional vehicle actions and cross-attention for structured scene data. |
| [WHAM/Muse](https://www.microsoft.com/en-us/research/publication/world-and-human-action-models-towards-gameplay-ideation/), Nature Feb 2025 | VQ visual tokens and an autoregressive transformer producing visuals, controller actions or both. | Controller actions are part of the learned sequence; persistent user modifications were an evaluation target. | Original WHAM was not real-time; the later WHAM-RT addresses this. | Historically important, but token-by-token generation is exactly the path Slower Roads should avoid. |
| [Genie 2](https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/), DeepMind Dec 2024 | Autoencoder latent grid plus large causally masked autoregressive latent-diffusion transformer. | Keyboard/mouse; classifier-free guidance improves action control. | Undistilled samples shown publicly; a lower-quality distilled model runs in real time. Parameters, steps, fps and hardware are undisclosed. | Architecture is relevant; scale and unavailable runtime details prevent a meaningful browser comparison. |
| [Oasis](https://oasis-model.github.io/), Decart/Etched Oct 2024 | ViT spatial autoencoder plus 500M DiT latent-diffusion backbone; Diffusion Forcing. | Live keyboard input; “dynamic noising” injects more noise early and reduces it later to preserve high-frequency detail. | **20 fps** using Decart’s inference stack; hardware is not specified. | Useful anti-drift training idea. Even the open 500M model is much larger than the likely browser envelope. |
| [GameNGen](https://arxiv.org/html/2408.14837), arXiv Aug 2024 | Stable Diffusion 1.4 latent autoencoder plus next-frame U-Net. Past frame latents are channel-concatenated; action embeddings replace text cross-attention. | Training contexts receive bucketed Gaussian noise to teach error correction. | **4 DDIM steps**, **20 fps on one TPU-v5** at 320×256. Distilled one-step reaches 50 fps but loses quality. Stable multi-minute play. | Crucial evidence for noisy-context training and for the poor quality of an undistilled one-step diffusion model. |
| [DIAMOND](https://arxiv.org/abs/2405.12399), NeurIPS 2024 | Diffusion environment model operating on visual observations rather than compact discrete state tokens. | Action-conditioned Atari/CS:GO prediction. | No generally comparable real-time fps in the paper abstract. | Supports the claim that aggressive discrete compression can discard control-relevant visual details. |
| [Genie 1](https://deepmind.google/research/publications/60474/), arXiv Feb 2024 | Spatiotemporal video tokenizer, autoregressive dynamics and a learned latent-action model; **11B** total. | Learns latent actions from unlabelled internet video. | Not real-time. | Latent actions solve a problem Slower Roads does not have because its actions and simulator state are known. |
| [GAIA-1](https://arxiv.org/abs/2309.17080), arXiv Sep 2023 | Discrete video tokens with next-token autoregression; 6.5B world model in the technical report. | Video, text and ego action tokens. | Offline generation. | A useful historical contrast: GAIA-2 moved from discrete next-token modeling to continuous latent flow matching. |

### Sparse-disclosure systems

- [Lucid v1](https://github.com/sonicCodes/lucid-v1/) reports 20 fps by default and up to 30 fps on RTX-4090-class hardware, but its repository does not provide enough model/training detail for a reliable architectural comparison.
- [Mirage](https://demo.mirage.decart.ai/) is a real-time video-to-video transformation system, not a verified keyboard-action world model. I found no primary technical report with parameters or sampler details.
- I could not verify a primary paper or detailed official technical report for a world model specifically named **“Magica.”** This may refer to Dynamics Lab’s “Magica 2,” but I am marking it **unverified**, not treating it as evidence. It should not be confused with [MAGI-1](https://arxiv.org/abs/2505.13211), a large chunk-autoregressive text-to-video model.

## 2. Few-step and one-step frame generation

### What the literature actually supports

- [GameNGen](https://arxiv.org/html/2408.14837) provides the cleanest controlled result for a constrained game. An ordinary one-step sampler scored **25.47 PSNR / 0.255 LPIPS**, versus **31.91 / 0.205 at two steps** and **32.58 / 0.198 at four**. Its distilled one-step model reached **31.10 / 0.208**. Thus one-step can work, but only after distillation; two steps already captured most of the four-step quality.

- [Shortcut Models](https://arxiv.org/abs/2410.12557), ICLR 2025, train one network conditioned on both noise level and requested step size. They support one or multiple steps without maintaining separate teacher/student networks. Their published experiments are primarily images, so their superiority at one step is not automatically transferable to long autoregressive video.

- [Dreamer 4](https://arxiv.org/html/2509.24527v1) adapts shortcuts to sequences as “shortcut forcing.” It uses continuous latents, x-prediction, independently noised timesteps and four passes per generated frame. The authors explicitly report x-prediction as more stable than velocity prediction for long frame-by-frame rollouts.

- [CausVid](https://arxiv.org/abs/2412.07772), CVPR 2025, distills a 50-step bidirectional teacher into a **four-step causal student** using video DMD, ODE-based initialization and asymmetric teacher/student attention. It reaches **9.4 fps on one GPU**. This is strong four-step evidence, not evidence for a browser-feasible one-step model.

- [Self-Forcing](https://arxiv.org/abs/2506.08009), 2025, trains against self-generated autoregressive histories using a rolling KV cache, a video-level loss and stochastic gradient truncation. It addresses exposure bias, but adds significant training complexity and is not intrinsically a one-step sampler.

- [DMD](https://arxiv.org/abs/2311.18828), 2023, converts an image diffusion model into a one-step generator and reports **20 fps** in FP16. [Matrix-Game 3](https://arxiv.org/abs/2604.08995) extends the family to multi-segment autoregressive video. DMD requires score/distribution teachers and is substantially more complicated than supervised next-frame learning on deterministic simulator data.

- [sCM](https://openai.com/index/simplifying-stabilizing-and-scaling-continuous-time-consistency-models/), 2024, produces image samples comparable to diffusion in **two steps**; a 1.5B 512² model takes 0.11 s on an A100. [Improved Consistency Training](https://openai.com/index/improved-techniques-for-training-consistency-models/) also demonstrated direct one-step training without a diffusion teacher. These are image results, but the 2026 [Interactive World Simulator](https://arxiv.org/abs/2603.08546) provides direct video-world-model evidence for consistency dynamics.

- [Diffusion Forcing](https://arxiv.org/abs/2407.01392), NeurIPS 2024, assigns independent noise levels to sequence tokens, allowing clean or lightly corrupted history and noisy future tokens in the same causal model. This is principally a training framework; it does not by itself make sampling one-step.

- Discrete flow matching, for example [Discrete Flow Matching](https://arxiv.org/abs/2407.15595), is normally iterative. I found no primary result showing that it gives a better one- or two-evaluation interactive video model than continuous consistency/shortcut approaches.

### Best choice at one or two evaluations

There is no universal paper winner, but for Slower Roads:

1. **At one evaluation:** direct continuous next-latent prediction or a directly trained consistency/shortcut model is the best fit. Your simulator makes the next frame largely deterministic after conditioning on state, geometry and action; starting from noise is likely wasted work.

2. **At two evaluations:** a first clean-latent prediction followed by a residual/error-correction pass is likely the best quality/complexity trade-off. GameNGen’s results show that two evaluations can recover much of four-step quality.

3. **MaskGIT at one iteration:** computationally attractive, but all-mask → full-frame categorical prediction removes MaskGIT’s principal iterative refinement mechanism. WHAM-RT validates parallel masked generation, but Microsoft does not disclose that it uses one iteration.

Given your measured **11–16 ms per transformer pass**, two full passes plus 6.5–7.3 ms decode/SR already consume roughly **28.5–39.3 ms on M3 Max**. Therefore two passes should be an optional quality tier, not the default path.

## 3. Anti-drift and long-horizon consistency

The most relevant techniques, in order:

### Imperfect-context training

This is the most consistently supported intervention.

- GameNGen corrupts encoded context frames with Gaussian noise and shows that removing the augmentation causes rapid deterioration after 10–20 frames. [Primary result](https://arxiv.org/html/2408.14837).
- Oasis uses training-time noise and an inference-time dynamic-noising schedule. [Project report](https://oasis-model.github.io/).
- Dreamer 4 lightly corrupts past inputs at inference and trains across per-timestep noise levels. [Paper](https://arxiv.org/html/2509.24527v1).
- Interactive World Simulator injects small noise into observation contexts specifically to make fixed-window rollouts robust. [Paper](https://arxiv.org/html/2603.08546v1).
- Matrix-Game 3 models residual errors and reinjects imperfect generated frames during training. [Paper](https://arxiv.org/abs/2604.08995).

For discrete FSQ, corruption should include more than random token replacement: use embedding noise, token dropout, decode–reencode histories, and actual model-generated histories. These better resemble the errors the deployed model will produce.

### Self-rollout training

- [Self-Forcing](https://arxiv.org/abs/2506.08009) directly closes the train/test gap by conditioning training on the model’s prior outputs.
- [Self Gradient Forcing](https://arxiv.org/abs/2607.20368) is particularly relevant for a small model: it avoids full backpropagation through the serial rollout while still letting future loss improve how prior frames are encoded into the KV cache.
- [RELIC](https://arxiv.org/abs/2512.04040) and [ABot-World-0](https://arxiv.org/abs/2607.19191) extend the approach to long teacher/student rollouts.

A Slower Roads implementation can be much simpler: generate 16–64 frames without gradients, sample one or two exit points, recompute those predictions with gradients, and supervise pixels, latent state, pose and road alignment.

### Structured state and geometry

This is where Slower Roads has an unusually large advantage.

- GAIA-2 conditions on ego motion, other agents, camera calibration, road attributes, weather and time, using adaLN for action and cross-attention for other structure. [Technical report](https://wayve.ai/wp-content/uploads/2025/03/GAIA_2_Technical_Report.pdf).
- [Context-as-Memory](https://arxiv.org/abs/2506.03141) retrieves frames by camera-pose/FOV overlap.
- [Video World Models with Long-term Spatial Memory](https://arxiv.org/abs/2506.05284) uses geometry-grounded 3D memory.
- RELIC stores absolute camera pose and relative actions with compressed memory latents.
- ReWorld uses a pose-indexed landmark bank.

Your oracle skeleton and simulator pose are stronger, cheaper signals than learned visual memory. For an endless road, a visual landmark bank has low value unless the route can loop. Absolute road-coordinate, curve profile, lane boundaries, weather/time and camera transform should remain authoritative; generated pixels should never be the source of truth for these variables.

### Error correction

The state head should not merely report what the visual model believes. Use it as a consistency constraint:

- supervise both absolute state and delta;
- derive a camera/road projection from oracle state;
- penalize disagreement between visual latent features and the projected road geometry;
- feed the next frame the simulator’s authoritative state, not the predicted state;
- train with deliberately perturbed visual contexts paired with correct oracle state.

This turns the model into a renderer of an authoritative evolving world, rather than requiring it to preserve the world entirely inside four visual frames.

## 4. Tokenizers and token flicker

### Relevant tokenizer work

- [MambaVideo Tokenizer](https://research.nvidia.com/labs/cosmos-lab/mamba-tokenizer/), 2025, introduces **channel-split FSQ**: latent channels are divided into independently quantized groups, increasing representational capacity without increasing spatial token count. NVIDIA reports reduced flicker qualitatively and, for MAGViT-v2+FSQ versus CS-FSQ, improvements from **30.69 to 31.08 PSNR** on Xiph and **30.06 to 30.75** on DAVIS. The compared configurations also differ in temporal compression, so this is promising rather than a perfectly controlled flicker result.

- [Cosmos Tokenizer](https://research.nvidia.com/labs/dir/cosmos-tokenizer/), 2025, uses causal temporal convolutions and attention, supports both continuous and FSQ latents, and adds an optical-flow reconstruction loss for temporal smoothness. Its discrete configuration uses six FSQ channels with levels `(8,8,8,5,5,5)` and a 64,000-code implicit vocabulary; its continuous tokenizer uses 16 latent channels.

- [MAGVIT-v2](https://arxiv.org/abs/2310.05737), 2023, introduced a shared image/video vocabulary using lookup-free quantization and causal video tokenization. It is much heavier than your frame tokenizer but helped establish temporally causal tokenizers for video dynamics.

- [TiTok](https://arxiv.org/abs/2406.07550), 2024, can represent a 256² image with 32 one-dimensional tokens. This is impressive global compression, but it removes the stable patch correspondence that your road-skeleton cross-attention and local next-frame prediction benefit from. It also requires a transformer tokenizer.

### How to reduce flicker without immediately abandoning FSQ

1. Train on adjacent simulator frames, not only independent images.
2. Add latent consistency under known optical flow or simulator reprojection.
3. Add noise-equivariance pairs: clean/noisy or subpixel-perturbed copies should quantize identically where ground-truth correspondence says appearance is unchanged.
4. Penalize code changes in static/reprojected regions, but exempt disocclusions and genuinely changing pixels.
5. Use hysteresis or a margin loss around FSQ bin boundaries.
6. Predict FSQ factors with separate small heads rather than a single 12,800-way head. This matches the factorized quantizer and makes channel-split FSQ practical.
7. Measure flip rate after compensating for ground-truth motion; raw same-location comparisons can confuse real motion with instability.

### Would continuous latents beat discrete FSQ here?

**Probably, yes—at least enough that it deserves the first controlled architecture experiment.**

Reasons:

- Your measured **33% token flip rate** converts tiny appearance changes into categorical target changes. A cross-entropy dynamics model must learn these discontinuities even though the underlying scene changes smoothly.
- A continuous output head is far smaller than `256 × 12,800` logits and should be cheaper in an overhead-bound browser graph.
- Interactive World Simulator demonstrates stable 10-minute action-conditioned rollouts using compact continuous latents and consistency dynamics.
- GAIA-2 replaced GAIA-1’s discrete next-token formulation with continuous latent flow matching and explicitly targets improved temporal/multi-camera coherence.
- DIAMOND shows that discrete compression can discard visually small but dynamically important details.

Risks:

- Plain MSE regression may blur multimodal futures.
- The current decoder may not tolerate unquantized latent values.
- Continuous latent magnitude can drift even when discrete codes cannot.

Those risks are manageable because your next state is largely deterministic given oracle state, skeleton and keys. Start with a bounded continuous latent—`tanh`, RMS normalization or a weak latent prior—and predict the **next clean latent directly**. Use the existing SR stage to recover sharpness. A stochastic consistency residual should only be added if deterministic regression visibly averages genuinely ambiguous effects such as foliage/weather.

## 5. Action conditioning and controllability

### Injection mechanisms

- **Adaptive layer norm/global modulation:** GAIA-2 injects vehicle action and flow time through adaLN at every transformer block and reports more accurate action conditioning than cross-attention. [Technical report](https://wayve.ai/wp-content/uploads/2025/03/GAIA_2_Technical_Report.pdf). This is the best match for four global key bits.
- **Action tokens:** Dreamer 4 independently embeds continuous, categorical and binary action components, sums their representations and interleaves action blocks with visual blocks. [Paper](https://arxiv.org/html/2509.24527v1).
- **Cross-attention tokens:** GameNGen replaces text cross-attention with a sequence of action embeddings. [Paper](https://arxiv.org/html/2408.14837).
- **Interleaved sequence tokens:** MineWorld and WHAM model actions alongside visual tokens. This is flexible but needlessly autoregressive for known four-bit controls.
- **Physical/camera representation:** Hunyuan-GameCraft converts keyboard and mouse into a shared camera-motion representation; GAIA-2 uses physical ego speed/curvature. These reduce ambiguity between key labels and the actual motion they imply.

### Recommendation for WASD

Use an action-conditioning vector containing:

- current four key states;
- key-down/key-up edges;
- duration held;
- the previous 2–4 action vectors;
- simulator-applied steering, acceleration and braking values;
- current speed and heading delta.

Map it through a small MLP and use it to modulate every block with adaLN/FiLM. Keep skeleton/state as cross-attention memory. This cleanly separates:

- **global transition request:** actions → adaLN;
- **spatial world structure:** skeleton/map → cross-attention;
- **recent visual appearance:** frame KV cache.

### Measuring responsiveness

Do not rely only on FVD or clip-level visual quality. Use the deterministic simulator to compute:

- pose-delta MAE for each action and speed bucket;
- steering-sign accuracy and braking/acceleration sign accuracy;
- action-to-visible-response latency in frames;
- paired counterfactual divergence: same history, change only one key bit;
- an inverse-dynamics classifier trained on real simulator frames and evaluated on generated frame pairs;
- per-frame action metrics rather than one clip average.

[WorldRoamBench](https://arxiv.org/abs/2606.31672) independently motivates per-frame action metrics, drift measures and action-decoupled memory evaluation, but your oracle state enables much stricter ground-truth measurement.

## 6. On-device and browser efficiency

I found **no primary-source demonstration of a neural interactive video world model running at 30 fps locally in a browser via WebGPU**.

The nearest systems remain much heavier:

- GameNGen: 20 fps on TPU-v5; one-step distilled 50 fps, but at lower quality.
- Interactive World Simulator: 15 fps, 128², RTX 4090.
- MineWorld: 4–7 fps.
- WHAM-RT: 10+ fps; undisclosed server hardware.
- Oasis: 20 fps through a proprietary inference stack.
- Lucid: 20–30 fps on RTX 4090-class hardware.
- Matrix-Game 3: 5B at 720p, with DMD, quantization and decoder pruning.
- ABot-World-0: 19 GiB on RTX 5090.

Transferable efficiency techniques are:

- full-frame parallel output rather than token AR;
- shallow/wide blocks, which your measurements already favor;
- infrequent temporal attention, as in Dreamer 4;
- grouped-query attention and bounded KV;
- small continuous or factorized-FSQ output heads;
- decoder pruning;
- training a large teacher but exporting only a one-step student;
- low-bit weights only after confirming ORT WebGPU has fast kernels for the exact operators and shapes.

Quantization is not automatically a speedup in WebGPU. If an INT8 graph introduces dequantization nodes, prevents graph capture or uses slower kernels, FP16 may remain faster. Benchmark the complete captured graph, not parameter size or theoretical FLOPs.

# Recommendations for Slower Roads

## 1. Make oracle state the invariant, not an auxiliary hint

**Change:** Feed absolute simulator state—not only deltas—at every frame: road-coordinate, lateral offset, heading, speed, camera transform, weather/time profile and a normalized future road description. Use action adaLN and skeleton/state cross-attention in every block. Retain the state head, but never feed its prediction back as authoritative state.

**Benefit:** This is the most credible route to practical zero drift over ten minutes. It also reduces how much history the visual KV cache must remember.

**Cost/risk:** Minimal inference cost; main risk is the model ignoring visual history and overfitting to state. Counter with context dropout and appearance variables not derivable from geometry.

**Respects:** browser budget, one pass, 64 px, oracle conditioning.

## 2. Train on the errors the deployed model will make

**Change:** Mix four history types during dynamics training:

1. clean ground-truth latents;
2. latents with embedding noise/dropout;
3. decoded-and-reencoded latents;
4. actual model rollout latents.

Use 16–64-frame no-gradient rollouts and recompute one or two sampled exits with gradients. Add future losses to the recomputed KV representations, following the lightweight idea behind Self Gradient Forcing.

**Benefit:** Directly attacks exposure bias and FSQ flicker amplification. This is the most consistently successful anti-drift technique across GameNGen, Oasis, Dreamer 4, Self-Forcing and Matrix-Game 3.

**Cost/risk:** Roughly 1.3–2× training compute depending on rollout reuse; no inference cost.

**Respects:** all deployment constraints.

## 3. Run a continuous-latent A/B experiment before scaling the transformer

**Change:** Bypass FSQ during dynamics training and predict a small bounded continuous latent grid with Huber/MSE plus perceptual or decoded-frame loss. Keep the same encoder/decoder capacity, token grid, transformer depth and conditioning so the quantizer is the controlled variable.

**Benefit:** Removes categorical discontinuities, reduces the output head, and likely improves one-pass temporal smoothness. Width is cheap in your measured runtime, making a 5–16-channel continuous output especially attractive.

**Cost/risk:** Requires decoder fine-tuning or tokenizer retraining; may blur if the conditioning leaves genuine uncertainty. Retain the discrete system as the baseline.

**Respects:** one pass, 64 px, browser budget; likely cheaper than 12,800-way logits.

## 4. If keeping FSQ, make stability an explicit tokenizer objective

**Change:** Add temporally aligned latent/code consistency, optical-flow or simulator-reprojection loss, input-noise invariance, and a bin-margin/hysteresis penalty. Test channel-split FSQ with factorized prediction heads.

**Benefit:** Reduces the 33%/13% flip rates at their source instead of asking dynamics to model arbitrary code boundaries.

**Cost/risk:** Over-regularization can suppress legitimate disocclusion or lighting changes. Mask the loss using simulator visibility and motion.

**Respects:** no added runtime if the exported tokenizer topology remains similar.

## 5. Use action adaLN, not just action tokens

**Change:** Modulate every dynamics block from current keys, action edges, held duration and simulator-applied controls. Continue using cross-attention for the spatial skeleton.

**Benefit:** Near-zero incremental cost and a direct conditioning path to every visual token. GAIA-2’s ablation favors this division of labor.

**Cost/risk:** The model might overreact to single-frame edges; train with realistic keyboard polling jitter and dropped/repeated action packets.

**Respects:** all constraints.

## 6. Default to one pass; make pass two adaptive

**Change:** First pass predicts all next-frame latents/tokens. Train the same model to accept that provisional prediction plus uncertainty/mask information and output a residual/refinement. Run refinement only:

- on high-end devices;
- when uncertainty or state inconsistency crosses a threshold;
- after scene/weather transitions;
- or at a lower frequency than the display rate.

**Benefit:** Captures much of the two-step quality without permanently doubling latency.

**Cost/risk:** Dynamic control flow and two captured graphs complicate ORT integration. A two-pass-every-frame configuration does not fit the measured budget reliably, particularly on consumer laptops.

**Respects:** one-pass default, optional two-pass ceiling.

## 7. Train and evaluate at deployment horizons

**Change:** Maintain fixed regression gates at 1 s, 10 s, 2 min and 10 min. Record:

- simulator-state error;
- projected road-boundary error;
- action latency;
- code/latent change in reprojected-static regions;
- LPIPS/SSIM against deterministic oracle frames;
- failure time under adversarial alternating actions;
- recovery after injecting corrupted context.

**Benefit:** Prevents short-clip visual quality from concealing slow geometric drift or weak controls.

**Cost/risk:** Long evaluations are time-consuming but can be batched and are deterministic.

**Respects:** no inference architecture cost.

## 8. Optimize the exported graph only after settling the latent design

**Change:** Benchmark FP16, weight-only INT8 where supported, smaller feed-forward ratios, GQA, decoder pruning and fused/factorized output heads in captured ORT Web graphs.

**Benefit:** May recover the margin needed for slower consumer laptops.

**Cost/risk:** Quantization can regress both visual detail and speed if WebGPU kernels are poorly matched. Measure end-to-end frame latency and compilation/capture behavior.

## Tempting ideas that do not fit

- **Four-step diffusion/shortcut forcing:** Dreamer 4, GameNGen and CausVid support its quality, but four × 11–16 ms is already 44–64 ms before decoding.
- **Large bidirectional-teacher/DMD pipelines as the first experiment:** effective at scale, but disproportionate for three hours of deterministic-domain data.
- **5B–15B interactive foundation models:** their headline fps depends on datacenter GPUs, aggressive distillation, low-bit kernels or proprietary inference systems.
- **Long visual landmark banks:** useful for revisitable open worlds, but lower-value than the oracle road skeleton/state for an endless route. Add one only if routes loop or scenery persistence becomes a product requirement.
- **TiTok-style 1D/global tokens:** token count falls, but spatial correspondence and cheap convolutional decoding are lost.
- **Raw token-by-token AR:** MineWorld and WHAM show why this is too slow, even with substantial parallelization.
- **Generating at 128 px inside the world model:** your actual WebGPU measurements outweigh papers’ datacenter throughput. Keep 64 px dynamics plus the existing learned SR stage.
- **Frame interpolation to advertise 30 fps:** it weakens immediate WASD response unless interpolation is explicitly action-aware.
- **Assuming one unrefined MaskGIT pass will retain quality:** WHAM-RT validates MaskGIT broadly, not one-iteration decoding.

# Bottom line

The recommended near-term experiment sequence is:

1. add action adaLN and richer absolute oracle state;
2. train with noisy, reencoded and self-generated histories;
3. compare current FSQ against an otherwise identical continuous-latent predictor;
4. add explicit temporal stability losses to whichever tokenizer wins;
5. only then test an adaptive second refinement pass.

If the continuous model wins, the likely final design is a **one-pass, shallow frame-causal transformer predicting a clean continuous 16×16 latent**, globally modulated by actions, cross-attending to authoritative skeleton/state, trained on its own imperfect histories, and decoded by the existing small decoder/SR stack. That is much closer to the successful principles of Interactive World Simulator and Dreamer 4 than to the computational form of the large 2025–2026 video-diffusion systems.

## Sources

Primary sources used above:

- [ReWorld](https://arxiv.org/abs/2608.23565), [Wonder](https://arxiv.org/abs/2607.26037), [ABot-World-0](https://arxiv.org/abs/2607.19191), [Matrix-Game 3](https://arxiv.org/abs/2604.08995), [Interactive World Simulator](https://arxiv.org/abs/2603.08546), [Self Gradient Forcing](https://arxiv.org/abs/2607.20368).
- [RELIC](https://arxiv.org/abs/2512.04040), [Dreamer 4](https://arxiv.org/html/2509.24527v1), [Matrix-Game 2](https://arxiv.org/abs/2508.13009), [Yume](https://arxiv.org/abs/2507.17744), [Hunyuan-GameCraft](https://arxiv.org/abs/2506.17201), [MineWorld](https://arxiv.org/abs/2504.08388), [V-JEPA 2](https://arxiv.org/abs/2506.09985).
- [Genie 3](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/), [Genie 2](https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/), [Genie 1](https://deepmind.google/research/publications/60474/).
- [GAIA-3](https://wayve.ai/thinking/gaia-3/), [GAIA-2 technical report](https://wayve.ai/wp-content/uploads/2025/03/GAIA_2_Technical_Report.pdf), [GAIA-1](https://arxiv.org/abs/2309.17080).
- [GameNGen](https://arxiv.org/html/2408.14837), [Oasis](https://oasis-model.github.io/), [DIAMOND](https://arxiv.org/abs/2405.12399), [WHAM/Muse](https://www.microsoft.com/en-us/research/publication/world-and-human-action-models-towards-gameplay-ideation/), [WHAM-RT](https://www.microsoft.com/en-us/research/project/wham/wham-rt/).
- [Diffusion Forcing](https://arxiv.org/abs/2407.01392), [Self-Forcing](https://arxiv.org/abs/2506.08009), [CausVid](https://arxiv.org/abs/2412.07772), [Shortcut Models](https://arxiv.org/abs/2410.12557), [DMD](https://arxiv.org/abs/2311.18828), [sCM](https://openai.com/index/simplifying-stabilizing-and-scaling-continuous-time-consistency-models/).
- [MambaVideo Tokenizer](https://research.nvidia.com/labs/cosmos-lab/mamba-tokenizer/), [Cosmos Tokenizer](https://research.nvidia.com/labs/dir/cosmos-tokenizer/), [TiTok](https://arxiv.org/abs/2406.07550), [MAGVIT-v2](https://arxiv.org/abs/2310.05737).
- [Context-as-Memory](https://arxiv.org/abs/2506.03141), [Video World Models with Long-term Spatial Memory](https://arxiv.org/abs/2506.05284), [WorldRoamBench](https://arxiv.org/abs/2606.31672).

Unverified: technical claims for “Magica 2”; detailed architecture for Mirage; model/training details beyond the Lucid v1 repository’s runtime claims.