# Making the FSQ tokenizer highly performant — research synthesis

Synthesis of a 5-way research fan-out (quantization, architecture, losses, world-model
tokenizers, training/eval) into a prioritized, citable stack of techniques for our
tokenizer: an **FSQ conv-AE, 64×64 RGB → 16×16×5 latent, 256 tokens/frame, vocab 12,800,
single-pass conv decoder**, feeding an AR dynamics core, decode eventually real-time on
WebGPU. Content is stylized low-poly (flat regions + a small high-contrast car).

## The one-paragraph verdict

Our design is **already the sweet spot** the 2023–2025 world-model literature converges
on (MineWorld 336 tok / vocab 8k, WHAM 256, Genie 256, IRIS 256, Cosmos chose FSQ for its
discrete/AR path). **Keep FSQ, keep ~256 tokens, keep the single-pass conv decoder** — do
*not* move to diffusion decode or LFQ/large-vocab. The remaining wins are on **decoder
architecture, the loss recipe, and the training recipe**, plus **AR-rollout parallelism**
(a dynamics-core change, not the tokenizer). Because the tokenizer is graded by a
downstream *pixel rollout* loss (faithfulness), avoid anything that *hallucinates* detail
(GANs, heavy LPIPS) and prefer things that *sharpen the true signal*.

## Prioritized stack (impact × ease, do top-down)

| # | Change | Where | Impact | Effort |
|---|--------|-------|-------:|-------:|
| 1 | **PixelShuffle (depth-to-space) + ICNR init** replaces ConvTranspose | decoder | ★★★★★ | trivial |
| 2 | **One self-attention block at the 16×16 bottleneck** (enc + dec) | both | ★★★★★ | trivial |
| 3 | **Saliency/edge-weighted L1** (up-weight the car) | loss | ★★★★★ | trivial |
| 4 | **Focal Frequency Loss** (sharpen edges, no hallucination) | loss | ★★★★☆ | easy |
| 5 | **Residual blocks (GN+SiLU)**, 1–2 per resolution stage | both | ★★★★☆ | low |
| 6 | **EMA of weights (0.999)** + AdamW β(0.9,0.95) + warmup/cosine + bf16 | training | ★★★★☆ | low |
| 7 | **Measure codebook utilization** (perplexity/entropy); right-size vocab | eval | ★★★★☆ | trivial |
| 8 | **Light VGG-LPIPS (w≈0.1)** + small SSIM; sigmoid→tanh output | loss | ★★★☆☆ | low |
| 9 | **Heavy-encoder / lean-decoder** asymmetry (decoder runs per-frame) | design | ★★★☆☆ | design |
| 10 | **Distill heavy decoder → tiny** (TAESD-style) once quality is set | later | ★★★☆☆ | medium |
| — | GAN/PatchGAN | loss | high ceiling / **high risk** | **defer** — hallucinates, hurts rollout faithfulness |
| — | LFQ / BSQ / rotation-trick / entropy-penalty | quantizer | **N/A at our scale** | skip |

### 1. Upsampling — biggest visual win
ConvTranspose(4,stride2) causes **checkerboard artifacts** (Odena 2016), which are glaring
on flat low-poly regions *and* inject periodic noise into the token stream the AR core
must model. Replace with `Conv2d(→4C,3) → PixelShuffle(2)` + **ICNR init** (Shi 2016 ESPCN;
Aitken 2017) — artifact-free *and* the cheapest WebGPU upsample (conv runs at low res, then
a pure index shuffle).

### 2. Bottleneck self-attention
At 16×16 = 256 tokens, full self-attention is O(256²) ≈ free. One `AttnBlock` (LDM/VQGAN
style) gives the decoder a global receptive field in one layer → flat regions stay
internally consistent and the car's 1–2 tokens can be placed/colored coherently. Linear
attention is unnecessary at this length.

### 3–4, 8. Loss recipe (faithfulness-first)
Base in gamma/sRGB space, output **tanh** (data in [-1,1]; sigmoid saturates on the
flat-white/black extremes and highest-contrast car pixels):
```
L = 1.0·L1_saliency_weighted   # w = 1 + 4·car_mask  (or 1 + 2·norm(|∇I|) if no mask)
  + 0.1·FocalFrequencyLoss     # ramp in after ~2k steps
  + 0.1·LPIPS(vgg)             # modest; can hallucinate, keep low
  + 0.1·SSIM (2–3 scale)       # contrast; helps the car
  + 0.5·edge/gradient          # (existing term — keep)
# GAN held in reserve: only if edges still soft after the above, heavily regularized
# (R1 lazy + LeCAM + DiffAug, 10–20k recon-only warmup) — and watch the rollout loss.
```
Saliency/mask weighting is the **single highest-leverage change for the car** — and we
have a synthetic sim, so a ground-truth car mask/bbox is available for free (prefer it over
image-derived saliency). Focal Frequency Loss (Jiang 2021) is the best sharpener with no
adversarial downside.

### 6–7. Training + eval
AdamW β(0.9,0.95), wd 1e-4, LR ~2e-4, linear warmup 5% → cosine, batch 128–256, **bf16
autocast**, `channels_last`, `torch.compile(max-autotune)` (shapes are static), grad-clip
1.0, **EMA(0.999)** of *weights* (FSQ has no codebook to EMA) — evaluate with EMA weights.
Unlimited deterministic sim frames ⇒ **stream fresh data, ~1 epoch, no aug except hflip**;
coverage of the state distribution matters more than augmentation. Eval panel: **PSNR,
SSIM, LPIPS** on a fixed val set each N steps (primary A/B signal); **rFID + codebook
utilization/perplexity** at end. FSQ should hit ~100% utilization — but on low-entropy
frames only a *fraction* of 12,800 codes may be emitted; measure it, and if usage is low,
trim levels (e.g. `[8,5,5,5]`=1024 or `[7,5,5,5,5]`=4096, or MineWorld's 8,192) to lighten
the AR head — **only if reconstruction holds**. The overriding tie-breaker is **token
predictability**: prefer the tokenizer whose tokens a tiny dynamics model predicts best,
even at a small pixel-metric cost.

### Quantizer — keep FSQ (do not churn it)
FSQ gives ~100% utilization, no collapse, no aux losses (Mentzer 2024); Cosmos picked it
for the discrete/AR path. LFQ/BSQ target *huge* vocab (2¹⁸) we don't want; the rotation
trick and entropy-penalty are **VQ/learned-codebook only — N/A to FSQ**. If the car's edges
lose detail, a **2-stage Residual-FSQ** raises effective bitrate without growing per-token
vocab, before touching net width.

## The $8 Runpod A/B plan
One mid GPU (4090/A5000, ~$0.4–0.7/hr) → ~12–18 GPU-hr, ~8 short runs. Hold fixed: data
stream/seed, a fixed 1000-frame val set, optimizer/schedule/precision, eval protocol.
Per run ~8–15 min (3–6k steps at hidden 32–64) — **rank variants on val-LPIPS at a fixed
step count** (curves separate early and rarely cross). Sweep, one axis at a time: LR
{1e-4,2e-4,4e-4} first, then the arch stack (baseline vs +pixelshuffle+attn+resblocks),
then loss (edge-only vs +FFL+saliency+LPIPS), then EMA on/off. Confirm the winner with one
longer run. Always **terminate the pod** when done.

## Key citations
FSQ — Mentzer 2024 (2309.15505) · LFQ/MAGVIT-v2 — Yu 2023 (2310.05737) · BSQ — Zhao 2024
(2406.07548) · Rotation trick — Fifty 2024 (2410.06424, *VQ-only*) · Residual-VQ —
Zeghidour 2021 (2107.03312) · Checkerboard/ICNR — Odena 2016, Shi 2016, Aitken 2017
(1707.02937) · VQGAN/adaptive-λ — Esser 2021 · LDM/SD-VAE — Rombach 2022 · Focal Frequency
Loss — Jiang 2021 (2012.12821) · LPIPS — Zhang 2018 · MS-SSIM+L1 — Zhao 2017 · EMA/EDM2 —
Karras 2023 (2312.02696) · ConvNeXt — Liu 2022 · DC-AE — Chen 2025 (2410.10733) · TAESD —
madebyollin · MineWorld — 2504.08388 (parallel/diagonal decode, 3× real-time) · WHAM —
Nature 2025 · Genie — 2402.15391 · Oasis/GameNGen — 2408.14837 · DIAMOND — 2405.12399 ·
IRIS/Δ-IRIS — 2209.00588 / 2406.19320 · Cosmos Tokenizer — 2501.03575 · LlamaGen/TiTok —
2406.07550.
