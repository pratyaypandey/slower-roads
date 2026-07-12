# Slower Roads

An on-device, real-time, steerable driving world model. See `ROADMAP.md` for the
full thesis and `docs/architecture.md` for the model interface contract.

This branch (`sim/m0-driving-sim`) delivers **M0 (the sim + oracle)** and the
**first-draft model side** (Tasks 1–3 from `jais_notes/notes.md`), built to be
verified without a GPU and trained later on one.

## What's here

```
sim/            deterministic Three.js driving sim + oracle harness  (M0, JS)
  src/            renderer-agnostic core: prng, road, car, params, world, sim
  headless/       data-gen (pixels via gl; state-only with no deps) + determinism proof
  browser/        live WASD driving (vite)
model/          the world-model side  (PyTorch, CPU/numpy-verifiable)
  tokenizer/      FSQ autoencoder (fsq_v2) + FSQ-variants study
  dynamics/       AR transformer core + multi-step rollout loss
  data/           Dataset over sim manifests (rgb | state | both)
eval/           tokenizer/dream evals + A/B experiment scripts + serve.py
deploy/         serving the model — RunPod (RUNPOD_SERVE.md) + Modal (modal_serve.py)
scripts/        orchestration (run_overnight.sh — the shared-tokenizer sweep)
docs/           architecture, TRAINING, tokenizer research, VAE_RECIPE (the tokenizer recipe)
plans/          M0 design specs (SIM.md, SIM_UPGRADE.md)
```

Key docs: `docs/VAE_RECIPE.md` (how the shipped tokenizer was built + the A/B
experiments), `docs/architecture.md` (the shapes/tokens/loss contract), and
`deploy/` for running the model off RunPod or Modal.

## Run it

### The driving sim (JS — works now)
```bash
cd sim
npm install                 # gl/pngjs are optional; three + vite always install
npm run prove               # determinism proof (no deps needed) — expect PASS
npm run dev                 # live driving, WASD, open the printed URL
npm run gen -- --seed 1 --steps 300 --res 128x128   # pixel dataset (needs gl)
node headless/generate_state.js --seed 1 --steps 300 # state-only dataset (no gl)
```

### The model (PyTorch — trains on your GPU box)
The Python side is verified here without torch (shape/logic tests via numpy). On a
machine with torch + numpy:
```bash
# from repo root
python -m model.tokenizer.test_shapes    # FSQ round-trip + shapes
python -m model.dynamics.test_shapes      # AR core, KV-cache, steering hook
python -m model.data.test_dataset         # (frame, action, next_frame) alignment
python -m eval.test_drift                 # drift metric behavior
python model/tokenizer/fsq_variants_study.py   # numpy only — the Task 2 study
```
Then generate data with the sim (`npm run gen`) and wire the training loop per
`model/README.md` and `docs/architecture.md`.

## Status vs the notes' tasks

| Task | State |
|---|---|
| 0. Driving sim | **Done + verified.** Determinism proven; real car + scenery; pixel & state data-gen. |
| 1. FSQ world-model training script (MineWorld-style) | **Drafted + shape-verified.** Tokenizer (~0.86M), AR dynamics (~9.85M), rollout loss. Trains on GPU. |
| 2. FSQ variants (AEP intuition) | **Done + measured.** See `model/tokenizer/FSQ_VARIANTS.md`: uniform FSQ wins in practice (tanh already companders). |
| 3. Architecture pseudocode | **Done.** `docs/architecture.md`, incl. the Schrödinger-bridge flow branch. |

## Verified vs. still needs your machine
- **Verified here (no GPU):** sim determinism; sim geometry against real three.js;
  FSQ quantizer math (exact over all 12800 codes); action-tokenizer agreement across
  components; dataset tuple alignment; drift metric; the FSQ study's numbers.
- **Needs your box:** actual pixel rendering (browser + `gl` data-gen); the
  torch-tensor assertions and any real training.
