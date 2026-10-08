# Model weights

The trained weights are attached to GitHub Releases, not stored in git. To
fetch them into `checkpoints/` (where every script expects them), with
sha256-verified downloads:

```bash
scripts/fetch_models.sh                 # latest: models-v2-2026-10
```

## models-v2-2026-10

All are trained only on our own simulator (renderer v2, `train_v2`) and held
out by seed. Metrics are on the 5 held-out val worlds (`data/train_v2/split.json`).

### Tokenizers (`checkpoints/v2/`)

All are fsq_v2, h128, 64 px → 16×16 tokens over 12,800 codes. EMA weights only.
Load with `model.registry.load_tokenizer(path)`.

| file | role | PSNR | tokens changed / frame | flips @1% noise | codes used |
|---|---|---|---|---|---|
| `tokenizer_v2_tc025.pt` | **default**: consistency fine-tune, weight 0.25 | 31.1 | 27% | 6.4% | 4.7k |
| `tokenizer_v2.pt` | base, 20 epochs; best recon, less stable (continuous-latent candidate) | 32.8 | 43% | 13% | 11.4k |
| `tokenizer_v2_tc10.pt` | consistency weight 1.0; very stable but over-smoothed | 28.8 | 13% | 2.4% | 986 |

### 64 → 256 upscalers (`checkpoints/sr/`)

These are SRVGGNetCompact state_dicts. Load with
`export/sr/sr_study.py` `build(<arch>)`.

| file | params | LPIPS ↓ | WebGPU ms | note |
|---|---|---|---|---|
| `medium_gan.pt` | 355k | 0.037 | 3.3 | **recommended** |
| `large_ftgan.pt` | 1.2M | 0.032 | 5.0 | quality ceiling. Fine-tuned from Real-ESRGAN `realesr-general-x4v3` (BSD-3-Clause, © Xintao Wang) |
| `small_gan.pt`, `small_l1.pt`, `medium_l1.pt`, `tiny_l1.pt` | 32k–355k | 0.043–0.097 | 2.1–3.3 | cheaper or steadier variants |

Details: `docs/WEBGPU_BUDGET.md` (upscaler section) and `docs/TRAINING.md` (v2 tokenizer).

## Publishing a new set

```bash
export/.venv/bin/python scripts/export_weights.py --out dist/models   # slim + sha256 manifest
gh release create <tag> dist/models/* --target <branch> --title "<title>" --notes-file <notes>
```

Then bump the default tag in `scripts/fetch_models.sh`.
