# Research brief: what can Slower Roads borrow from recent world-model work?

You are doing deep literature research (2023 -> late 2026, newest first) for a small,
real-time, action-conditioned video world model. Search the web thoroughly (arXiv, project
pages, GitHub, blogs from DeepMind/Google, OpenAI, Decart, Wayve, NVIDIA, Microsoft, Tencent,
Skywork, Etched, Dynamics Lab, etc.). VERIFY every claim against a primary source and give a
link for each. Do not invent papers or numbers; if unsure, say so.

## The project (read carefully: recommendations must fit these constraints)
- Slower Roads: "Slow Roads" (endless relaxing driving game) recreated as a neural world model
  that runs REAL-TIME (30 fps, <=33 ms/frame) ON-DEVICE IN THE BROWSER (ONNX Runtime Web,
  WebGPU), takes live WASD keyboard input, and must not drift over 10+ minutes.
- Training data: our own deterministic Three.js simulator (unlimited data, ground-truth
  oracle state: car pose, road skeleton ahead). 3 h dataset so far, 90 episodes, 5 driving
  profiles incl. weather/time-of-day.
- Tokenizer: FSQ conv autoencoder, 64x64 RGB -> 16x16 tokens, 12,800-code implicit vocab
  (levels 8,8,8,5,5), ~12M params. Decode ~4 ms in WebGPU. A learned 64->256 super-resolution
  stage (SRVGG-style, 89k-355k params, GAN-finetuned) adds ~2.5-3.3 ms and looks good.
  Known issue: tokens are temporally unstable (~33% of tokens flip between near-identical
  frames, 13% flip under 1% pixel noise); this hurt our earlier autoregressive dynamics.
- Dynamics (being redesigned now): transformer over FSQ tokens, frame-causal across frames,
  bidirectional within a frame, predicting all 256 tokens of the next frame in 1 (max 2)
  parallel passes (MaskGIT-style). KV cache of 4 context frames. Skeleton-memory
  cross-attention (oracle road-skeleton tokens, outside the KV cache, no RoPE) + a state head
  regressing per-frame {x, z, heading, speed} deltas -- this combination beat a frozen-frame
  baseline over 2-minute rollouts (the earlier token-by-token AR model drifted).
- MEASURED BROWSER BUDGET (M3 Max, ORT Web, graph capture, fp16): latency is dominated by
  depth (~1-1.5 ms per layer), width is cheap; d384-512 x 4-6 layers fits at ~11-16 ms/frame
  for one pass; 128 px / 32x32 tokens does NOT fit; ORT Web is overhead-bound (~5% of peak).
  Consumer laptops are an estimated 2.5-5x slower.
- Inference loop in the browser: the deterministic sim core runs alongside the model each
  frame, so oracle skeleton/state conditioning is available at inference ("practical zero
  drift" design). Conditioning on raw key-state (4 bits) is planned.

## Questions to answer
1. ARCHITECTURES of real-time / interactive world models (e.g. GameNGen, DIAMOND, Oasis,
   Genie 2/3, MineWorld, Matrix-Game, Hunyuan-GameCraft, Yume, Mirage, Magica, Lucid,
   WHAM/Muse, GAIA-1/2, Cosmos, V-JEPA 2, Dreamer 4, any 2025-2026 successors). For each:
   tokenizer/latent space, dynamics type (AR tokens, diffusion, flow, masked), steps per frame,
   fps achieved and on what hardware, params, action conditioning, how they fight drift.
2. FEW-STEP / ONE-STEP generation for frame prediction: diffusion forcing, Self-Forcing,
   CausVid, shortcut models, consistency / distillation (DMD, sCM), MaskGIT one-step,
   discrete flow matching, "next-frame diffusion", etc. What gives the best quality at 1-2
   network evaluations per frame?
3. ANTI-DRIFT / long-horizon consistency: noise augmentation of context, self-forcing,
   rollout training, memory (WorldMem, state-space memory, frame retrieval), conditioning on
   structured state/maps, error-correction tricks. Which match our oracle-conditioning setup?
4. TOKENIZERS for world models: temporally stable / causal video tokenizers, FSQ vs VQ vs
   continuous latents, small-latent tokenizers (e.g. 1D/TiTok-style), tricks to cut token
   flicker. Would continuous latents + few-step diffusion beat discrete FSQ for us?
5. ACTION CONDITIONING / controllability: how the best models inject keyboard actions
   (adaLN, cross-attn, per-token add, action tokens) and how they measure responsiveness.
6. EFFICIENCY for on-device/browser: small-model results, distillation, quantization,
   anything shown running in a browser or on a laptop GPU in real time.

## Output
Write a markdown report:
- Section per question with the relevant works (title, venue/date, link, 1-3 line summary,
  and concrete numbers when available).
- A final "Recommendations for Slower Roads" section: a ranked list (most impactful and
  feasible first) of specific techniques to adopt, each with: what to change in our design,
  expected benefit, cost/risk, and which constraint it respects (browser budget, 1-2 passes,
  64 px, oracle conditioning). Also list things that look tempting but DON'T fit our budget.
- A "Sources" list. Mark anything you could not verify.
