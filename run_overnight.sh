#!/usr/bin/env bash
# Overnight high-fidelity run: big dataset -> bigger tokenizer -> bigger dynamics
# -> tokenizer_recon.png + dream.gif. Everything at 64px (frame size isn't
# plumbed for 128 yet). Safe to run unattended: fails fast, logs each stage,
# checkpoints every epoch so an interrupted stage can --resume.
#
#   GPU=4 bash run_overnight.sh              # pick a free GPU (nvidia-smi)
#   GPU=4 STEPS=6000 bash run_overnight.sh   # override dataset size
#
# Tunables (env vars, with defaults):
set -euo pipefail

GPU="${GPU:-4}"                 # which CUDA device (check nvidia-smi for a free one)
STEPS="${STEPS:-6000}"          # dataset frames (bigger = sharper, slower to capture)
SIZE="${SIZE:-64}"             # capture resolution (64 only — 128 not wired)
TOK_HIDDEN="${TOK_HIDDEN:-128}" # tokenizer width (was 64; bigger = sharper recon)
TOK_EPOCHS="${TOK_EPOCHS:-120}" # tokenizer epochs (was 20; train to convergence)
GRAD_WEIGHT="${GRAD_WEIGHT:-0.5}"   # edge loss (keeps the car)
DYN_DMODEL="${DYN_DMODEL:-384}"     # dynamics width (was 256)
DYN_LAYERS="${DYN_LAYERS:-6}"       # dynamics depth (was 4)
DYN_EPOCHS="${DYN_EPOCHS:-80}"      # dynamics epochs (was 20)
DATA="${DATA:-data/seed1}"
DREAM_STEPS="${DREAM_STEPS:-60}"    # frames to dream in the eval GIF

export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p logs
ts() { date "+%H:%M:%S"; }
say() { echo "[$(ts)] === $* ==="; }

# Preflight: fail NOW (not 20 min in) if the env is wrong. Needs the slowroads
# conda env active (torch on GPU + numpy) and playwright/gifenc for capture.
say "preflight: checking env"
python -c "import torch, numpy; assert torch.cuda.is_available(), 'no CUDA — activate the slowroads env / check the driver'; print('  torch', torch.__version__, 'CUDA ok; numpy', numpy.__version__)" \
  || { echo "ENV NOT READY: run 'conda activate slowroads' first (needs torch+CUDA+numpy)."; exit 1; }
( cd sim && node -e "require('playwright'); require('gifenc')" ) 2>/dev/null \
  || { echo "ENV NOT READY: cd sim && npm install && npx playwright install chromium."; exit 1; }

say "config: GPU=$GPU STEPS=$STEPS SIZE=${SIZE}px tok(hidden=$TOK_HIDDEN,epochs=$TOK_EPOCHS) dyn(d=$DYN_DMODEL,layers=$DYN_LAYERS,epochs=$DYN_EPOCHS)"

# 1. Generate the pixel dataset (CPU renderer — no GPU contention, reliable).
say "1/5 generating $STEPS frames at ${SIZE}px -> $DATA"
node sim/headless/generate_pixels.mjs --seed 1 --steps "$STEPS" --size "$SIZE" 2>&1 | tee logs/1_gen.log

# 2. Verify the frames are real + moving BEFORE spending hours training on them.
say "2/5 verifying frames are real and moving"
python3 - "$DATA" <<'PY'
import numpy as np, json, sys
d = sys.argv[1]
m = json.load(open(f"{d}/manifest.json"))
n = len(m["samples"])
a = np.load(f"{d}/" + m["samples"][n // 10]["frame"])
b = np.load(f"{d}/" + m["samples"][n // 2]["frame"])
std = float(a.std()); diff = float(np.abs(a - b).mean())
print(f"  frame std={std:.4f}  frame-to-frame diff={diff:.4f}")
assert std > 0.05, "FROZEN/GRAY frames — aborting before training"
assert diff > 0.02, "frames not changing (camera not following?) — aborting"
print("  OK: real, moving frames")
PY

# 3. Train the tokenizer (bigger + longer + edge loss to keep the car).
say "3/5 training tokenizer (hidden=$TOK_HIDDEN, epochs=$TOK_EPOCHS, grad=$GRAD_WEIGHT)"
python -m model.train_tokenizer --data "$DATA" \
    --hidden "$TOK_HIDDEN" --grad-weight "$GRAD_WEIGHT" --epochs "$TOK_EPOCHS" \
    2>&1 | tee logs/3_tokenizer.log
python -m eval.eval_tokenizer --data "$DATA" --ckpt checkpoints/tokenizer.pt \
    2>&1 | tee logs/3_tokenizer_eval.log
say "-> eval/plots/tokenizer_recon.png written"

# 4. Train the dynamics core on the frozen tokenizer (bigger + longer).
say "4/5 training dynamics (d_model=$DYN_DMODEL, n_layers=$DYN_LAYERS, epochs=$DYN_EPOCHS)"
python -m model.train_dynamics --data "$DATA" --tokenizer checkpoints/tokenizer.pt \
    --d-model "$DYN_DMODEL" --n-layers "$DYN_LAYERS" --epochs "$DYN_EPOCHS" \
    2>&1 | tee logs/4_dynamics.log

# 5. Dream — the payoff GIF.
say "5/5 dreaming $DREAM_STEPS frames"
python -m eval.eval_dream --data "$DATA" \
    --tokenizer checkpoints/tokenizer.pt --dynamics checkpoints/dynamics.pt \
    --steps "$DREAM_STEPS" 2>&1 | tee logs/5_dream.log

say "DONE. Artifacts:"
echo "    eval/plots/tokenizer_recon.png   (original vs FSQ reconstruction)"
echo "    eval/plots/dream.gif             (true drive vs model's dream)"
echo "    logs/                            (per-stage output)"
echo "  scp them back:  scp gpublaze:~/slower-roads/eval/plots/{tokenizer_recon.png,dream.gif} ."
