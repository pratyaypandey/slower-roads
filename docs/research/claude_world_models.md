# What Slower Roads can borrow from recent world-model work (2023 → Oct 2026)

*Independent literature review (Claude). Researched 2026-10-01 against primary sources (arXiv abstracts/HTML/PDF, official
project pages, model cards). Every work has a link. Numbers are quoted from the primary source unless marked
**[secondary]** (from press/blog coverage only) or **[unverified]** (I could not confirm it).*

---

## TL;DR

1. **The literature mostly agrees with the bet we already made.** Systems that stay stable over minutes do one or more
   of three things: (a) train on corrupted or self-generated context, (b) condition on an external state or anchor
   rather than only on pixels, and (c) use few-step continuous latents with x-prediction. We already have the oracle
   (b). The cheapest wins are (a) and making (b) **spatially aligned**.
2. **Our discrete-FSQ token flip problem is a known failure mode, not bad luck.** Orbis ran a controlled
   continuous-vs-discrete comparison on *driving*: continuous flow matching beat the discrete (MaskGIT) variant
   across settings, and the discrete model copied the last context frame's token about 45% of the time. Every
   2024–2026 system that reached real time at a quality worth copying runs on continuous latents (DIAMOND, GameNGen,
   Oasis, Lucid, Dreamer 4, NFD, Matrix-Game). The two exceptions are WHAMM (MaskGIT) and MineWorld (AR tokens, 2–6 fps).
3. **1–2 network evaluations per frame is enough when the next frame is nearly deterministic, and ours is.** The sim
   gives us pose, skeleton and weather. DIAMOND says single-step EDM works for deterministic transitions. GameNGen's
   1-step distilled model loses little (PSNR 31.10). NFD reaches 1–4 steps with sCM plus adversarial heads. Dreamer 4's
   shortcut forcing gets within reach of 64-step quality at K=4. Two browser projects (Flappy Bird at 1 step, Neural
   Drive at 2 steps) already ship this recipe in WebGPU.
4. **What doesn't fit our budget:** large video DiTs and their DMD/Self-Forcing distillation pipelines, 3D causal VAEs
   with temporal compression (they add latency), chunk-wise generation (adds input lag), retrieval memory banks,
   classifier-free guidance at inference (doubles the passes), and many-step MaskGIT (Genie used 25 steps).

---

## Q1. Architectures of real-time / interactive world models

### Comparison table (single-GPU speed unless noted)

| Model (date) | Latent / tokenizer | Dynamics | Steps / frame | Speed & HW | Params | Actions injected via | Anti-drift |
|---|---|---|---|---|---|---|---|
| **Dreamer 4** (Sep 2025) | causal transformer tokenizer (400M), MAE patch-dropout p~U(0,0.9), 512×16 bottleneck → 256 spatial tokens ×32 ch, tanh | block-causal transformer, shortcut forcing, x-prediction | **K=4** | **21 FPS, 1×H100**, 640×360 | 2B total (1.6B dyn) | each action component → tokens, **summed** + learned embedding | context corrupted to τ_ctx=0.1; x-prediction; ramp loss weight |
| **GameNGen** (Aug 2024, ICLR'25) | SD-1.4 VAE (decoder fine-tuned with MSE) | SD-1.4 U-Net, 64-frame context | 4 DDIM (1-step distilled) | 20 FPS (50 FPS distilled), 1×TPU | ~SD 1.4 | action embedding replaces the text cross-attention | **context noise aug α≤0.7, 10 buckets**; agent-play data |
| **DIAMOND** (May 2024, NeurIPS'24) | pixels (Atari 64×64); CS:GO 56×30 + upsampler | EDM U-Net, 4-frame stack | 3 (1 works for deterministic games) | CS:GO **10 Hz on RTX 3090** | Atari 4M; CS:GO 381M (51M upsampler) | adaptive GroupNorm | EDM's x-like preconditioning stays stable even at 1 step |
| **Oasis** (Oct 2024) | ViT autoencoder | DiT + Diffusion Forcing | n/a | 20 FPS (500M open model, 1×H100 per Dreamer 4) | 500M (open) | per-frame | "dynamic noising" of context at inference |
| **Lucid v1** (Nov 2024) | 15 tokens/frame (aggressive AE + GAN loss) | causal diffusion transformer, diffusion forcing | n/a | 25 FPS RTX 4090 (own note); **44 FPS 1×H100** (Dreamer 4 Table 1) | 1–1.1B | n/a | diffusion forcing; 1 s context |
| **MineWorld** (Apr 2025) | VQ (aMUSEd init), 336 tok/frame, 8k codes | token-by-token AR transformer + parallel (diagonal) decoding | sequential (≈3× speedup) | 3–6 FPS (paper); **2 FPS on H100** per Dreamer 4 | 300M / 700M / 1.2B | 11 action tokens interleaved | none special |
| **NFD / NFD+** (Jun 2025) | 2D VAE, 16× spatial, 24×14 tokens @384×224 | block-causal DiT (bidirectional within a frame) | **1–4** (sCM + adversarial) | **31.14 FPS, 310M, 1×A100** | 130M / 310M / 774M | **adaLN-Zero (beat cross-attn and in-context)** | Gaussian noise on context; speculative frames when action unchanged (1.26×) |
| **WHAMM** (Microsoft, Apr 2025) | ViT-VQGAN, 576 tok @640×360 | MaskGIT: 500M backbone + 250M refiner | few refinement passes | 10+ FPS | ~750M | interleaved tokens | 0.9 s context only |
| **WHAM / Muse** (Nature, Feb 2025) | ViT-VQGAN, 540 tok | decoder-only AR | sequential | ~1 frame/s | 1.6B | action tokens | — |
| **Matrix-Game 2.0** (Aug 2025) | Wan 3D causal VAE (8×8 spatial, 4× temporal) | Wan-1.3B-derived causal DiT, Self-Forcing + DMD | 3 | **25 FPS, 1×H100**, 352×640 | 1.8B | mouse **concatenated** to latents; keyboard **cross-attn** | self-forcing; 6-frame KV cache |
| **Matrix-Game 3.0** (Apr 2026) | Wan VAE + pruned "MG-LightVAE" (2.6–5.2× decode speedup) | 5B / 2×14B DiT, multi-segment self-rollout distillation | few | 40 FPS 720p using **8 GPUs** (+1 for VAE) | 5B | keyboard cross-attn; mouse via self-attn | **error buffer: inject residuals δ=x̂−x into history**; camera-aware memory retrieval; sink latent |
| **Matrix-Game 3.5** (Aug 2026) | same family | patch memory + tiled PRoPE, self-rollout DMD | few | real-time (multi-GPU) **[details unverified]** | — | — | geometry-aware memory |
| **Yan-Sim** (Tencent, Aug 2025) | VAE with **32× spatial**, 2× temporal, 16 ch; pruned decoder | causal DiT, shift-window denoising | 4 DDIM | 1080p 50–60 FPS on **2 GPUs**, 0.11 s latency | — | MLP → 768-d token per frame, frame-local **action cross-attn** | KV cache; FP8 (1.5–2×), CUDA graphs |
| **Hunyuan-GameCraft** (Jun 2025) | HunyuanVideo VAE | MM-DiT + hybrid history conditioning | distilled **[fps unverified]** | — | — | keyboard/mouse → **shared camera-pose space** | hybrid history condition |
| **Yume** (Jul 2025) | Wan VAE | masked video DiT + memory | adversarial distillation + caching | — | — | quantized camera motion | memory module |
| **Genie** (Feb 2024) | ST-ViViT VQ, 1024 codes | ST-transformer **MaskGIT, 25 steps/frame** | 25 | ~1 FPS | 11B | **additive latent-action embeddings beat concatenation** | 16-frame memory |
| **Genie 2** (Dec 2024, blog) | video autoencoder | AR latent diffusion transformer | — | not real-time | — | keyboard/mouse | remembers off-screen parts **[blog only]** |
| **Genie 3** (Aug 2025, blog) | — | AR frame generation | — | **720p 24 FPS real-time** | — | navigation + text "world events" | consistent for minutes, ~1 min visual memory **[no paper]** |
| **GAIA-1** (Sep 2023) | discrete tokens | AR transformer + video diffusion decoder | — | offline | 9B **[from paper body, not re-checked]** | action & text tokens | — |
| **GAIA-2** (Mar 2025) | continuous video tokenizer | latent diffusion | — | offline | — | **structured conditioning: ego dynamics, agents, road semantics** | — |
| **GAIA-3** (Dec 2025, press) | 2× larger tokenizer | — | — | offline | 15B | — | **[press release only]** |
| **Cosmos** (Jan 2025) | Cosmos Tokenizer (causal; continuous 16-d or **FSQ (8,8,8,5,5,5)=64k**) | diffusion & AR WFMs | — | offline | 77–105M tokenizers | — | — |
| **V-JEPA 2 / V-JEPA 2-AC** (Jun 2025) | no pixel decoder (JEPA embeddings) | action-conditioned predictor | — | planning, not rendering | — | — | — |
| **Mirage / Magica** (Dynamics Lab, Jul–Aug 2025) | — | "transformer + diffusion" | — | 16 FPS browser *stream* (server-side), ~200 ms latency | — | — | **[secondary only; no tech report found]** |
| **WanToFight** (Jul 2026) | Wan VAE + pruned decoder | Wan-1.3B block-causal AR, distilled | few | **30 FPS 512×384 on RTX 5090** | 1.3B | keyboard injection learned by curriculum | — |
| **DreamForge-World 0.1** (Jun 2026) | Wan2.1 VAE | LongLive AR stack (Wan2.1-T2V-1.3B) + Matrix-Game-style actions | few | **14–15 FPS 480p on 1×RTX 4090** | ~1.3B | Matrix-Game style | minute-scale |
| **AlayaRenderer-Flash** (Jul 2026) | Wan VAE → **tiny TAEHV-style decoder** + tiny shared G-buffer encoder | Wan-2.1 AR streaming, **conditioned on engine G-buffers** | **4** | **31.54 FPS, 832×448, 1×H200** | — | (world state comes from the physics engine) | first-frame **appearance anchor** + hierarchical history compression |
| **Magpie** (Aug 2026) | — | game engine owns state; a generative render server draws frames | — | real-time **[numbers unverified]** | — | engine | engine-side state |
| **StatePlay** (Jul 2026) | — | mixture-of-transformers predicting **frames + explicit game state** | — | — | — | — | state L1 < 0.06; +18.6% mechanics fidelity |
| **W² / Streaming multi-agent w/ World State Registers** (Jul 2026) | — | AR diffusion + learnable **world-state register tokens** updated per chunk | — | — | — | — | registers supervised by agent status / BEV / text |
| **PlayGen** (Dec 2024) | — | AR DiT diffusion | — | real-time over 1000+ frames on **RTX 2060** | — | — | — |

### Driving-specific world models

- **DriveGAN** (CVPR'21), [arXiv 2104.15060](https://arxiv.org/abs/2104.15060). A neural driving simulator learned from
  160 h of video, with disentangled, controllable content and theme (weather). It is the closest older analog to our
  goal.
- **Vista** (NeurIPS'24), [arXiv 2405.17398](https://arxiv.org/abs/2405.17398). Generalizable driving world model with
  multi-level controls (from commands down to steering) and a latent replacement strategy for long rollouts. It is
  offline and SVD-scale.
- **DrivingWorld** (ICPR'26), [arXiv 2412.19505](https://arxiv.org/abs/2412.19505). GPT-style, next-state plus
  next-token prediction. It uses masking and reweighting to cut long-term drift and generates for 40 s or more.
- **Epona** (ICCV'25), [arXiv 2506.24113](https://arxiv.org/abs/2506.24113). Autoregressive diffusion driving world
  model with decoupled spatio-temporal factorization and a **"chain-of-forward" training** strategy against error
  accumulation.
- **Orbis** (Dec 2025 v2), [arXiv 2507.13162](https://arxiv.org/abs/2507.13162). 469M parameters, 280 h of driving data,
  rollouts up to 20 s. See Q4 for its continuous-vs-discrete result.

### Browser / on-device precedents (most relevant to us)

- **Flappy Bird world model in the browser** (njkumar, Jul 2025),
  [blog](https://www.njkumar.com/optimizing-flappy-bird-world-model-to-run-in-a-web-browser/). Built on DIAMOND, cut
  from a 330M denoiser + 51M upsampler down to a **5M denoiser + 112K upsampler**. Works at 64×36 resolution.
  - **EDM made 1 step possible**, where DDPM had needed 3.
  - Uses ONNX + WebGPU with fp16 (some ops kept fp32).
  - Speed: **30 FPS on an M2 Pro, 12–15 FPS on an iPhone 14 Pro**. The WASM fallback is 2–3× slower.
- **Neural Drive** (Asankhaya Sharma, 2026), [model card](https://huggingface.co/codelion/neural-drive-model) and
  [demo](https://huggingface.co/spaces/codelion/neural-drive). A kart game trained on SuperTuxKart.
  - ConvVAE (/8, 8 channels) feeding a LatentDiT (d768 × 12 layers, patch 3, 130.8M params), at 384×192.
  - Context is 7 frames plus the controls. The k=8 teacher is distilled to a **k=2** student.
  - Ships as a 262 MB fp16 ONNX graph in WebGPU that prefills the context once and runs both steps inside one graph.
  - Speed is only **[secondary]** ([Korben](https://korben.info/en/neural-drive-kart-game-no-engine-browser-webgpu.html)):
    about 10 FPS on M-series MacBooks and 2.5 FPS in Firefox on a Mac Studio. It confirms our measurement that
    12-layer, 130M models do not fit 33 ms.

---

## Q2. Few-step / one-step generation (what is best at 1–2 NFE per frame?)

| Method | Link | Key fact | Fit for us |
|---|---|---|---|
| **Diffusion Forcing** (NeurIPS'24) | [2407.01392](https://arxiv.org/abs/2407.01392) | Gives every token/frame its own noise level, so causal rollout and lightly-noised history come for free | Base objective for a continuous branch |
| **History-Guided Video Diffusion / DFoT** (ICML'25) | [2502.06764](https://arxiv.org/abs/2502.06764) | Variable-length history plus history guidance | Guidance doubles NFEs, so skip at inference |
| **Shortcut models** | [2410.12557](https://arxiv.org/abs/2410.12557) | Condition on step size d. One network and one training phase cover 1, 2 or 4 steps; beats consistency models and reflow | **High**: no teacher, and the step count is chosen at inference |
| **Shortcut forcing (Dreamer 4)** | [2509.24527](https://arxiv.org/abs/2509.24527) | Diffusion forcing combined with shortcut models; **x-prediction**; ramp weight w(τ)=0.9τ+0.1. In their ablation, K=64 gives 0.8 FPS and K=4 shortcut gives 9.1 FPS at near-64-step FVD. x-prediction plus x-loss took FVD from 326 to 151, and the ramp weight to 102. | **High**: the closest recipe to what we need |
| **sCM** (ICLR'25 oral) | [2410.11081](https://arxiv.org/abs/2410.11081) | Continuous-time consistency, 2-step FID 1.48 on ImageNet-64 | Used inside NFD |
| **MeanFlow** | [2505.13447](https://arxiv.org/abs/2505.13447) | Average-velocity flow, **1-NFE FID 3.43** on ImageNet-256, no distillation | An alternative to shortcut; AlayaRenderer-Flash uses "Mean Flow Distillation" |
| **DMD2** | [2405.14867](https://arxiv.org/abs/2405.14867) | One-step distribution matching plus GAN loss | Needs a strong teacher, so it's heavy for us |
| **CausVid** | [2412.07772](https://arxiv.org/abs/2412.07772) | Bidirectional 50-step teacher distilled to a 4-step causal student; 9.4 FPS | Teacher-student pipeline at 1B+ scale |
| **Self Forcing** (NeurIPS'25 spotlight) | [2506.08009](https://arxiv.org/abs/2506.08009) | Trains on its own KV-cached rollouts with a video-level DMD/SiD/GAN loss and gradient truncation (only the last step backpropagates). 4 steps on Wan-1.3B: 17 FPS chunk-wise, **8.9 FPS frame-wise** on an H100 | Idea: yes. Scale: no |
| **Self-Forcing++** | [2510.02283](https://arxiv.org/abs/2510.02283) | Minute-scale rollouts (up to 4m15s) by supervising segments of its own long rollouts | Idea only |
| **Rolling Forcing** | [2509.25161](https://arxiv.org/abs/2509.25161) | Jointly denoises a window of frames at rising noise levels, plus an attention sink | **Adds frames of latency**, so no |
| **Resampling Forcing** | [2512.15702](https://arxiv.org/abs/2512.15702) | Teacher-free: self-resampling simulates inference errors in the history; top-k history routing | **High**: teacher-free and trainable from scratch |
| **APT2 (AAPT)** (NeurIPS'25) | [2506.09350](https://arxiv.org/abs/2506.09350) | **1 NFE per latent frame** via adversarial post-training plus student forcing; 8B model, 24 FPS 736×416 on one H100 | The adversarial-1-step idea transfers; the model size doesn't |
| **Next-Frame Diffusion (NFD)** | [2506.01380](https://arxiv.org/abs/2506.01380) | Video sCM with per-frame timesteps and adversarial heads; **310M model at 31 FPS on an A100**; adaLN-Zero actions; speculative frames | **High**: small model and the same block-causal layout as ours |
| **ForgeWM** (Aug 2026) | [2608.14022](https://arxiv.org/abs/2608.14022) | Progressive conversion of an action-conditioned generator to **1/2/4-step** causal models (teacher-forced causal training, then DMD); a 1-step preview can be refined on replay | Moderate |
| **MWM** (Mar 2026) | [2603.07799](https://arxiv.org/abs/2603.07799) | Few-step distillation that preserves action-conditioned **rollout consistency** (ICSD) | Moderate |
| **MaskGIT** | [2202.04200](https://arxiv.org/abs/2202.04200) | Parallel masked decoding, ~8–12 steps typical; up to 64× faster than AR | Our current approach. At 1–2 passes it is effectively a classifier per token |
| **DiMO** | [2503.15457](https://arxiv.org/abs/2503.15457) | **First one-step distillation of masked diffusion models**, via token-level distribution matching | The discrete-branch option if we stay discrete |
| **Duo** (ICML'25) | [2506.10892](https://arxiv.org/abs/2506.10892) | Discrete consistency distillation through a Gaussian↔uniform-discrete duality | Text-only evidence, so speculative for us |
| **FSQ + continuous diffusion** | [2606.09962](https://arxiv.org/abs/2606.09962) | Argues FSQ codes are an especially good space for *continuous* diffusion over categorical data (TTS) | Supports running continuous dynamics over our FSQ grid |

**Verdict for 1–2 NFE.** The strongest small-model evidence comes from five systems:
- NFD: 310M params, 1–4 steps, sCM plus adversarial, 31 FPS.
- Dreamer 4: shortcut forcing with x-prediction.
- GameNGen: 1-step distilled, PSNR 31.10 vs about 29.4 at 4 steps, at 50 FPS.
- DIAMOND: EDM, one step is enough for deterministic dynamics.
- The browser Flappy Bird model: 1-step EDM.

On the discrete side, nothing shows 1–2-pass MaskGIT matching continuous few-step quality for video. WHAMM and Genie
use multiple passes, and DiMO is image-only.

---

## Q3. Anti-drift and long-horizon consistency

| Technique | Source | Exact recipe | Matches our setup? |
|---|---|---|---|
| **Noise augmentation of context + noise-level embedding** | GameNGen [2408.14837](https://arxiv.org/abs/2408.14837) | α ~ U(0, 0.7) added to context latents; the level is fed in through 10 learned bucket embeddings. Without it "quality degrades quickly after 10–20 frames" | Yes. For discrete tokens, the analog is random FSQ-level perturbation with a corruption-level embedding |
| **Light context corruption at inference** | Dreamer 4 (τ_ctx=0.1); NFD (Gaussian on context); Oasis "dynamic noising" | Train with diffusion forcing, then at inference feed slightly noised history | Yes, costs nothing |
| **Self-generated context during training** | Self Forcing, Self-Forcing++, APT2 "student forcing", Epona "chain-of-forward", Resampling Forcing | Roll out the model and train on its own outputs | We already do a multi-step rollout loss, so keep it and make the horizon longer |
| **Error banks / residual injection** | Matrix-Game 3.0 [2604.08995](https://arxiv.org/abs/2604.08995); Stable Video Infinity [2510.09212](https://arxiv.org/abs/2510.09212); Vorch-Director [2608.05776](https://arxiv.org/abs/2608.05776) | Store real model residuals δ=x̂−x and add γδ to clean history (MG3); "error-recycling fine-tuning" (SVI); match each residual to its noise level (Vorch) | **Yes, and cheap.** It is realistic corruption without a full rollout each step. For tokens, bank the actual *wrong-token patterns* |
| **Attention sink / anchor frame** | Rolling Forcing; LongLive [2509.22622](https://arxiv.org/abs/2509.22622) (frame sink); AlayaRenderer-Flash (first frame as appearance anchor); MG3 (sink latent) | Keep the first frame's KV permanently in context | Yes, for sky, weather and time-of-day palette. Costs one more frame of keys |
| **Structured / state conditioning** | GAIA-2 [2503.20523](https://arxiv.org/abs/2503.20523) (ego dynamics, road semantics); AlayaRenderer-Flash [2607.18703](https://arxiv.org/abs/2607.18703) (G-buffers concatenated to the noisy latent before patch embedding); StatePlay [2607.26754](https://arxiv.org/abs/2607.26754); W² world-state registers [2607.21594](https://arxiv.org/abs/2607.21594); DiffusionRenderer [2501.18590](https://arxiv.org/abs/2501.18590); EPE [2105.04619](https://arxiv.org/abs/2105.04619) | Engine supplies state, the network renders | **This is our oracle design**, and the 2026 "generative renderer" line validates it. The difference: they inject *spatially aligned* buffers, while we cross-attend to skeleton tokens |
| **Memory retrieval** | WorldMem [2504.12369](https://arxiv.org/abs/2504.12369); MG3 camera-aware retrieval; Matrix-Game 3.5 patch memory; SSM world model [2505.20171](https://arxiv.org/abs/2505.20171); MagicWorld [2511.18886](https://arxiv.org/abs/2511.18886) | Bank of past frames with poses, retrieved when the view overlaps | **Not needed.** Slow Roads doesn't revisit places, and the oracle replaces memory |
| **x-prediction instead of v/ε-prediction** | Dreamer 4; DIAMOND (EDM) | Predict the clean target, which avoids high-frequency errors that compound | Yes for a continuous branch. Discrete heads already predict x |
| **Copy-or-generate per token** | ITC [2605.16457](https://arxiv.org/abs/2605.16457); Δ-IRIS [2406.19320](https://arxiv.org/abs/2406.19320) (delta tokens) | Each token either copies from the previous frame or is generated fresh | Mixed. Orbis shows discrete models *over*-copy (≈45%), and on a forward-scrolling road copying is mostly wrong |
| **Data matters** | GameNGen | Agent-play data beat random play: 19.02 vs 16.84 PSNR after 3 s of autoregressive rollout | We already have driving profiles. Add recovery/perturbation driving (off-centre, then correcting) |

---

## Q4. Tokenizers

- **Orbis** ([2507.13162](https://arxiv.org/abs/2507.13162), Freiburg) is the most decisive evidence for our question.
  - **Setup:** a hybrid tokenizer that skips the VQ bottleneck with 50% probability during training, so the same
    latents serve both a flow-matching model and a MaskGIT model.
  - **Result:** on driving, the continuous model "consistently outperforms Orbis-MG across all architectures and
    inference settings".
  - **Failure mode:** the discrete classifier "chooses the exact same token as the last context frame approximately
    45% of the times".
  - **Caveat:** the continuous side uses 30 ODE steps, not 1–2.
- **Cosmos Tokenizer** ([page](https://research.nvidia.com/labs/cosmos-lab/cosmos-tokenizer/),
  [2501.03575](https://arxiv.org/abs/2501.03575)). Causal temporal conv + attention with a wavelet front end. The
  discrete variant uses **FSQ (8,8,8,5,5,5) = 64k**, which is essentially our level design. The continuous variant has
  16 channels. Tokenizers are 77–105M params.
- **Dreamer 4 tokenizer.** **MAE patch dropout p~U(0,0.9)** during tokenizer training "improve[s] the spatial
  consistency of videos generated by the dynamics model". The bottleneck is a low-dimensional tanh, i.e. continuous,
  bounded latents much like pre-rounding FSQ.
- **Lucid v1** ([note](https://www.lucid.ai/notes/lucid-v1)): 15 tokens per frame with a GAN perceptual loss.
  **TiTok** ([2406.07550](https://arxiv.org/abs/2406.07550)) encodes an image as 32 1D tokens. **DeltaTok / DeltaWorld**
  ([2604.04913](https://arxiv.org/abs/2604.04913), CVPR'26) encodes one continuous delta token per frame, with 35× fewer
  params. All three show that tiny latents work. However, our budget is depth-bound rather than token-bound, and 1D
  tokens lose the spatial alignment we want for skeleton conditioning.
- **EQ-VAE** ([2502.09509](https://arxiv.org/abs/2502.09509), ICML'25). Equivariance regularization that makes latents
  smoother. It works for discrete and continuous AEs (MaskGIT included), and 5 epochs of SD-VAE fine-tuning gave 7×
  faster DiT convergence. **It applies directly to our flip problem**: a consistency loss under small shifts and noise.
- **iFSQ** ([2601.17124](https://arxiv.org/abs/2601.17124)). Swaps FSQ's tanh for a distribution-matching map to get a
  uniform prior and better bin use. It is a one-line change. Whether it raises or lowers flip rate is
  **[unverified for our case]**.
- **Tiny decoders.** TAEHV ([GitHub](https://github.com/madebyollin/taehv)) is a tiny distilled video decoder with a
  streaming wrapper, used by AlayaRenderer-Flash. MG-LightVAE in MG3 (50% pruned) decodes 2.6× faster at PSNR 31.84. We
  are already fine here: about 4 ms for decode plus about 3 ms for the SR stage.
- **3D causal VAEs with temporal compression** (Wan, Yan's 32× spatial / 2× temporal): **don't fit**. Temporal
  compression means you can't show frame t until frames t..t+k exist, which adds input latency.

**Would continuous latents + few-step diffusion beat FSQ for us?** Probably yes, for three reasons:
1. Orbis shows it on driving, and every fast, high-quality system we found uses continuous latents.
2. Our 33% flip rate is a *quantization-boundary* artifact. Nearly identical frames produce pre-quantization latents
   that are close, so a continuous target removes the flicker by construction.
3. A continuous head (5 floats per token) is cheaper than even the factorized 34-logit FSQ head.

**Main risk:** a 1-step regression blurs where the future is uncertain (foliage, clouds). Our oracle conditioning
shrinks that uncertainty, and the GAN-fine-tuned SR stage already restores texture. DIAMOND, GameNGen-1-step and the
browser Flappy model show that 1-step quality is acceptable in near-deterministic games.

---

## Q5. Action conditioning and how responsiveness is measured

**How the strong models inject actions**
- **NFD** compared adaLN-Zero, cross-attention and in-context tokens; **adaLN-Zero won**.
- **Genie:** additive action embeddings beat concatenation.
- **Dreamer 4:** each action component gets its own embedding (linear for continuous, lookup for binary), and the
  embeddings are **summed** into action tokens interleaved with frames.
- **DIAMOND:** adaptive GroupNorm.
- **GameNGen:** actions replace the text in cross-attention.
- **Matrix-Game 2 / 3:** keyboard goes through cross-attention; mouse is concatenated to the latents (MG2) or goes
  through self-attention (MG3).
- **Yan:** per-frame action cross-attention, where each frame's tokens attend only to that frame's action token.
- **Hunyuan-GameCraft:** maps keys into a shared camera-motion space. Yume does something similar with quantized camera
  motion.

**How responsiveness is measured**
- **Genie ΔtPSNR** (t=4): PSNR with the inferred actions minus PSNR with random actions.
- **MineWorld:** an inverse-dynamics model labels the generated video, then precision/recall/F1 against the input
  actions. This correlates with human controllability ratings (r=0.56).
- **Matrix-Game GameWorld Score** ([2506.18701](https://arxiv.org/abs/2506.18701)): action controllability as one axis.
- **Dreamer 4:** humans try to complete tasks inside the model (it completed 14 of 16; Oasis-large 5 of 16).
- **Our advantage:** we can run a state regressor or IDM on generated frames and compare against the *sim's own*
  response to the same keys. That gives response latency in frames and steering/speed error, which is better than any
  of the published metrics.

**Fit for us.** With 4-bit key state, an adaLN-Zero or additive embedding costs essentially nothing. The bigger
question is whether keys should drive the picture directly or **through the oracle**. The sim integrates keys into
pose, and pose is already conditioned on. Raw key bits are mostly useful for short-latency cues such as brake lights,
the steering animation and camera roll.

---

## Q6. Efficiency for on-device / browser

**What fits**
- **Browser proof points.** The Flappy Bird model ran a 5M EDM denoiser at 1 step in fp16 ONNX on WebGPU at 30 FPS on
  an M2 Pro. Neural Drive (130.8M, d768×12, k=2) is about 10 FPS on M-series **[secondary]**. Together they bracket our
  measured budget: shallow and small fits, 12 layers does not.
- **Wide and shallow, factorized attention.** Dreamer 4's speed cascade went from 9.1 to 30.1 FPS through:
  - long-range temporal attention only every 4 layers;
  - GQA;
  - time-factorized long context;
  - register tokens.
  
  This matches our "depth costs, width is cheap" measurement.
- **Graph capture / CUDA graphs.** Yan reports a further 1.15× from torch.compile on top of CUDA graphs. We already
  depend on ORT graph capture.
- **Quantization.** Reported gains:
  - INT8 on attention projections: Matrix-Game 3.0.
  - INT8 with minimal loss: LongLive.
  - FP8 at 1.5–2× over FP16: Yan.
  
  On ORT Web/WebGPU, int8 matmul support and speed are **[unverified]**, and our profile says we're overhead-bound, so
  expect little.
- **Speculative frames.** NFD predicts the next N frames assuming the action doesn't change and throws them away if it
  does (1.26× at N=2). We already know the sim's next state, so we could speculatively compute frame t+1 while the user
  is still holding the same keys. Worth it only on slow GPUs.
- **Single-graph multi-step.** Neural Drive puts both denoise steps inside one ONNX graph, so there is one dispatch per
  frame. We already do this with the fused commit.
- **Pruned or tiny decoders** (MG-LightVAE, TAEHV, Yan's pruned decoder). Not our bottleneck.

**What doesn't fit (scale)**
- Every 1B+ Wan/SkyReels-derived system: Matrix-Game 2/3/3.5, Self Forcing, CausVid, LongLive, DreamForge,
  WanToFight, AlayaRenderer-Flash, Yume, GameCraft and Yan. They need H100/4090-class GPUs or several GPUs.

---

## Recommendations for Slower Roads (ranked: impact × feasibility)

### 1. Train on corrupted and self-generated context, with a corruption-level embedding
- **Change:**
  - During dynamics training, corrupt context tokens two ways:
    - (a) **±1-level perturbation on random FSQ channels** at rate p ~ U(0, p_max). This is the discrete analog of
      GameNGen's α≤0.7 noise.
    - (b) **Replace them with tokens from a bank of real model errors.** Save the model's own 1-pass predictions,
      in the spirit of MG3's error buffer and SVI's error recycling.
  - Give the dynamics model the corruption level as a bucketed embedding (GameNGen used 10 buckets).
  - At inference, feed a small fixed level (Dreamer 4 used τ_ctx=0.1).
  - Keep and lengthen the existing multi-step rollout loss (Self Forcing, Epona's chain-of-forward).
- **Benefit:** GameNGen degrades after 10–20 frames without this. It also makes the model tolerant of our 13–33% token
  flips, because it learns to treat context tokens as noisy evidence.
- **Cost / risk:** training only, zero inference cost. Risk: too much corruption makes the model ignore context and
  lean on the oracle, which may be fine for us.
- **Respects:** browser budget, 1–2 passes, 64 px, oracle.

### 2. Make the oracle conditioning spatially aligned (the "neural renderer" framing)
- **Change:**
  - In the sim (JS), project the road skeleton ahead into the camera as a cheap 16×16 (or 64×64 → 16×16) condition map,
    one value per token. Channels could be road mask, lane centre/edge distance, a depth proxy, and a sky/terrain
    split.
  - **Add it to each token embedding** (`h = embed(z) + W·g`, the way AlayaRenderer-Flash concatenates G-buffers
    before patch embedding).
  - Keep the skeleton cross-attention for look-ahead beyond the view if it still helps.
- **Benefit:** GAIA-2, AlayaRenderer-Flash, DiffusionRenderer, EPE and Magpie all show that structured state rendered
  into the image plane is the strongest anti-drift signal there is. Road geometry becomes locked to the sim every
  frame, so drift can only show up in texture and appearance.
- **Cost / risk:** one small projection per frame in JS/WebGPU and a single matmul. Risk: train/inference parity of the
  projection code, so share one implementation.
- **Respects:** all constraints. This is the main lever behind "practical zero drift".

### 3. Fix tokenizer temporal stability before re-training dynamics
- **Change:** fine-tune the FSQ autoencoder with:
  - **a consistency loss** on pre-quantization latents between a frame and its slightly augmented copy (≤1% pixel
    noise, sub-pixel shift), EQ-VAE style;
  - **MAE-style patch dropout** (Dreamer 4, p~U(0,0.9));
  - **decoder robustness to code perturbation**: decode from ±1-level-perturbed codes and still match the target, so
    dynamics errors decode gracefully.

  Then measure flip rate and reconstruction PSNR. Optionally try iFSQ's distribution-matching activation.
- **Benefit:** attacks the root cause we already identified, and helps both the discrete and continuous branches.
- **Cost / risk:** a few GPU-hours. Possible small loss in reconstruction PSNR. No inference cost.
- **Respects:** 64 px, 4 ms decode.

### 4. A/B a continuous-latent dynamics branch with shortcut forcing (x-prediction), K ∈ {1, 2}
- **Change:**
  - Keep the same backbone, KV cache, oracle and action conditioning. Swap the target from FSQ ids to the continuous
    bounded FSQ latent `zb` (256×5). The decoder takes `zb` directly, or `zb` rounded.
  - Train with **shortcut forcing**: Dreamer 4's diffusion forcing plus shortcut with **x-prediction**, x-loss and ramp
    weight 0.9τ+0.1, step sizes {1, ½, ¼}. Infer with K=1 or 2.
  - The head becomes a linear layer to 5 floats, which is cheaper than the 34-logit FSQ head.
  - If 1-step looks soft, add a small adversarial head (NFD, APT2), or rely on the existing GAN-fine-tuned SR stage.
- **Benefit:**
  - It removes flicker from quantization-boundary flips.
  - The Orbis driving result, NFD (310M, 1–4 steps, 31 FPS) and the browser Flappy/Neural-Drive precedents all point
    this way.
  - K is selectable at runtime, so fast GPUs get K=2 and slow ones K=1.
- **Cost / risk:**
  - Moderate: a new objective and loss code, roughly the same compute as current training.
  - Risk: blur in uncertain regions (mitigated by the oracle and SR), and losing the categorical sampling diversity we
    don't really need.
  - Run it head-to-head against the improved discrete model from recommendations 1–3. Judge on 10-minute rollouts
    against the oracle-state metrics, not on single-frame scores.
- **Respects:** 1–2 passes (K=passes), the browser budget (identical backbone), 64 px.

### 5. Inject actions with adaLN-Zero or additive embeddings; measure response latency against the sim
- **Change:** embed the 4-bit key state (16-way lookup) plus any continuous steer/speed from the oracle, and apply it as
  adaLN-Zero modulation (NFD's winner) or an additive embedding on all tokens (Genie). Evaluate responsiveness three
  ways:
  - (a) the state-head / IDM read on generated frames vs the sim's state after the same keys;
  - (b) frames until a visible response;
  - (c) a ΔPSNR-style score: real keys vs shuffled keys over 4–8 frames.
- **Benefit:** near-zero-cost controllability, plus a metric that catches "ignores input".
- **Cost / risk:** trivial. adaLN adds per-layer scale/shift, which is negligible at our sizes.
- **Respects:** all.

### 6. Add an anchor frame (attention sink) and global appearance tokens
- **Change:** keep the episode's first (or most recent "clean") frame's KV permanently in the window, as in LongLive,
  Rolling Forcing, AlayaRenderer-Flash and MG3. Add oracle weather, time-of-day and profile as global tokens or adaLN
  inputs.
- **Benefit:** stops slow palette and sky drift over 10+ minutes, which the oracle skeleton cannot pin down.
- **Cost / risk:** +256 keys (one frame) in attention, roughly +5–10% pass time at ctx 4, so measure it. Alternatively
  use a few learned "style register" tokens (Dreamer 4 registers; W² world-state registers).
- **Respects:** the budget if verified. Oracle.

### 7. Cheap efficiency borrowings
- **Change:**
  - Use register tokens.
  - Use GQA in attention, which shrinks the KV that has to be committed.
  - If ctx ever grows past 4: run temporal attention only every 2–4 layers (Dreamer 4).
  - On slow GPUs: speculatively compute frame t+1 while keys are unchanged (NFD).
- **Benefit:** headroom for the 2.5–5× slower laptops.
- **Cost / risk:** low.

### 8. Data: add recovery driving and perturbation episodes
- **Change:** add episodes that deliberately drift off-centre and correct, plus brake/throttle transitions. GameNGen
  showed agent-quality data matters for long rollouts.
- **Benefit:** better coverage of off-distribution states. Cheap, because the sim is unlimited.

### Tempting but doesn't fit
- **Distilling from a large video DiT** (Wan / SkyReels / Matrix-Game / Self Forcing with DMD or CausVid). Teachers are
  1.3B+ and students still run at 9–25 FPS on an H100. Use the *ideas* (self-rollout, DMD), not the models.
- **3D causal VAEs with temporal compression** (Wan 4×, Yan 2×) and **chunk-wise generation** (Self Forcing chunks of 3,
  AlayaRenderer-Flash chunks of 4). Both add frames of input latency.
- **Rolling Forcing–style joint multi-frame denoising windows**, for the same latency reason.
- **Classifier-free / history guidance at inference** (DFoT). It doubles NFEs. Only use it if guidance is distilled
  into the weights, as AlayaRenderer-Flash does.
- **Many-step MaskGIT** (Genie: 25 steps), **token-by-token AR** (MineWorld: 2–6 FPS even with parallel decoding), and
  **two-model MaskGIT** (WHAMM: 500M backbone + 250M refiner).
- **Memory retrieval banks / SSM long-context** (WorldMem, MG3/3.5, the SSM world model). There's no revisiting in
  Slow Roads, and the oracle supplies what memory would.
- **128 px / 32×32 tokens and 12-layer models.** Our budget rules them out, and Neural Drive (12 layers, 130M) only gets
  about 10 FPS on M-series **[secondary]**.
- **1D / ultra-compact tokenizers** (TiTok, Lucid's 15 tokens, DeltaTok). Attractive, but they break the per-token
  spatial alignment that recommendation 2 relies on. Our cost is depth-dominated, so fewer tokens buys little.
- **INT8/FP8 quantization.** Wins in CUDA systems, but unproven in ORT Web, and we're overhead-bound.

---

## Sources

### Real-time / interactive world models
- Dreamer 4, "Training Agents Inside of Scalable World Models" (Hafner, Yan, Lillicrap, Sep 2025): https://arxiv.org/abs/2509.24527
- GameNGen, "Diffusion Models Are Real-Time Game Engines" (ICLR 2025): https://arxiv.org/abs/2408.14837
- DIAMOND, "Diffusion for World Modeling: Visual Details Matter in Atari" (NeurIPS 2024): https://arxiv.org/abs/2405.12399
- Oasis (Decart / Etched, 2024): https://oasis-model.github.io/
- Lucid v1 (Nov 2024): https://www.lucid.ai/notes/lucid-v1
- MineWorld (Apr 2025): https://arxiv.org/abs/2504.08388
- Next-Frame Diffusion / NFD (Jun 2025): https://arxiv.org/abs/2506.01380
- WHAMM (Microsoft Research blog): https://www.microsoft.com/en-us/research/articles/whamm-real-time-world-modelling-of-interactive-environments/
- WHAM / Muse (Nature 2025): https://www.nature.com/articles/s41586-025-08600-3 · https://huggingface.co/microsoft/wham
- Matrix-Game (Jun 2025): https://arxiv.org/abs/2506.18701
- Matrix-Game 2.0: https://arxiv.org/abs/2508.13009
- Matrix-Game 3.0: https://arxiv.org/abs/2604.08995
- Matrix-Game 3.5: https://arxiv.org/abs/2608.29910
- Yan (Tencent, Aug 2025): https://arxiv.org/abs/2508.08601
- Hunyuan-GameCraft: https://arxiv.org/abs/2506.17201 (fps not verified)
- Yume: https://arxiv.org/abs/2507.17744
- Genie (Feb 2024): https://arxiv.org/abs/2402.15391
- Genie 2 (blog, Dec 2024): https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/
- Genie 3 (blog, Aug 2025): https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/
- Mirage / Magica (Dynamics Lab): **[secondary only]** https://the-decoder.com/mirage-2-allows-users-to-turn-sketches-and-photos-into-interactive-game-worlds/
- WanToFight (Jul 2026): https://arxiv.org/abs/2607.12592
- DreamForge-World 0.1 (Jun 2026): https://arxiv.org/abs/2606.30292
- AlayaRenderer-Flash, "Generative World Renderer at the Speed of Play" (Jul 2026): https://arxiv.org/abs/2607.18703
- Magpie (Aug 2026): https://arxiv.org/abs/2608.27168
- StatePlay (Jul 2026): https://arxiv.org/abs/2607.26754
- W², streaming multi-agent world-state registers (Jul 2026): https://arxiv.org/abs/2607.21594
- PlayGen, "Playable Game Generation" (Dec 2024): https://arxiv.org/abs/2412.00887
- GameFactory (ICCV 2025): https://arxiv.org/abs/2501.08325
- V-JEPA 2: https://arxiv.org/abs/2506.09985
- Cosmos: https://arxiv.org/abs/2501.03575

### Driving world models
- GAIA-1: https://arxiv.org/abs/2309.17080
- GAIA-2: https://arxiv.org/abs/2503.20523
- GAIA-3 (press): https://wayve.ai/thinking/gaia-3/
- DriveGAN: https://arxiv.org/abs/2104.15060
- Vista: https://arxiv.org/abs/2405.17398
- DrivingWorld: https://arxiv.org/abs/2412.19505
- Epona: https://arxiv.org/abs/2506.24113
- Orbis: https://arxiv.org/abs/2507.13162

### Few-step generation and anti-drift
- Diffusion Forcing: https://arxiv.org/abs/2407.01392
- DFoT / History Guidance: https://arxiv.org/abs/2502.06764
- Shortcut models: https://arxiv.org/abs/2410.12557
- sCM: https://arxiv.org/abs/2410.11081
- MeanFlow: https://arxiv.org/abs/2505.13447
- DMD2: https://arxiv.org/abs/2405.14867
- CausVid: https://arxiv.org/abs/2412.07772
- Self Forcing: https://arxiv.org/abs/2506.08009
- Self-Forcing++: https://arxiv.org/abs/2510.02283
- Rolling Forcing: https://arxiv.org/abs/2509.25161
- Resampling Forcing: https://arxiv.org/abs/2512.15702
- LongLive: https://arxiv.org/abs/2509.22622
- APT2 / AAPT: https://arxiv.org/abs/2506.09350
- ForgeWM: https://arxiv.org/abs/2608.14022
- MWM: https://arxiv.org/abs/2603.07799
- MaskGIT: https://arxiv.org/abs/2202.04200
- DiMO: https://arxiv.org/abs/2503.15457
- Duo: https://arxiv.org/abs/2506.10892
- Stable Video Infinity: https://arxiv.org/abs/2510.09212
- Vorch-Director: https://arxiv.org/abs/2608.05776
- WorldMem: https://arxiv.org/abs/2504.12369
- Long-context SSM video world models: https://arxiv.org/abs/2505.20171
- MagicWorld: https://arxiv.org/abs/2511.18886
- ITC: https://arxiv.org/abs/2605.16457
- GIT-STORM: https://arxiv.org/abs/2410.07836

### Tokenizers and rendering
- Δ-IRIS: https://arxiv.org/abs/2406.19320
- DeltaTok: https://arxiv.org/abs/2604.04913
- TiTok: https://arxiv.org/abs/2406.07550
- EQ-VAE: https://arxiv.org/abs/2502.09509
- iFSQ: https://arxiv.org/abs/2601.17124
- FSQ + continuous diffusion: https://arxiv.org/abs/2606.09962
- Cosmos Tokenizer: https://research.nvidia.com/labs/cosmos-lab/cosmos-tokenizer/
- TAEHV: https://github.com/madebyollin/taehv
- DiffusionRenderer: https://arxiv.org/abs/2501.18590
- EPE (Enhancing Photorealism Enhancement): https://arxiv.org/abs/2105.04619

### Browser precedents
- Flappy Bird world model in the browser: https://www.njkumar.com/optimizing-flappy-bird-world-model-to-run-in-a-web-browser/
- Neural Drive model card: https://huggingface.co/codelion/neural-drive-model
- Neural Drive speeds **[secondary]**: https://korben.info/en/neural-drive-kart-game-no-engine-browser-webgpu.html

### Caveats
- **MineWorld FPS:** the paper reports 3–6 FPS (hardware wording ambiguous). Dreamer 4 measured 2 FPS on one H100.
- **Lucid v1 FPS:** 25 FPS on an RTX 4090 per its own note; Dreamer 4 measured 44 FPS on an H100.
- **GAIA-1 at 9B params:** from memory of the paper body, not re-checked.
- **Hunyuan-GameCraft speed** and **Matrix-Game 3.5 speed:** not verified.
- **Genie 2/3 and Mirage/Magica:** no technical papers. Blog or press only.
