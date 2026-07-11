#!/usr/bin/env bash
# Overnight experiment sweep. Trains the EXPENSIVE shared parts once (pixel
# dataset + tokenizer), then sweeps dynamics variants so you wake up to a
# side-by-side comparison: one dream GIF + drift log per experiment.
#
#   GPU=4 bash run_overnight.sh
#   GPU=4 STEPS=6000 EXPERIMENTS="baseline tf_sched flow_bridge" bash run_overnight.sh
#
# Each experiment writes:
#   checkpoints/<name>/dynamics.pt
#   eval/plots/dream_<name>.gif
#   logs/<name>.{train,dream}.log     (drift numbers are in the dream log)
set -euo pipefail

GPU="${GPU:-4}"
STEPS="${STEPS:-6000}"           # dataset frames
SIZE="${SIZE:-64}"
DATA="${DATA:-data/seed1}"
# Shared tokenizer (trained once, frozen, reused by every experiment):
TOK_HIDDEN="${TOK_HIDDEN:-128}"
TOK_EPOCHS="${TOK_EPOCHS:-120}"
GRAD_WEIGHT="${GRAD_WEIGHT:-0.5}"
# Dynamics (shared size across experiments so the comparison is apples-to-apples):
DYN_DMODEL="${DYN_DMODEL:-384}"
DYN_LAYERS="${DYN_LAYERS:-6}"
DYN_EPOCHS="${DYN_EPOCHS:-25}"    # baseline showed CE converges by ~20
DREAM_STEPS="${DREAM_STEPS:-60}"
# Which experiments to run (space-separated). See the case block below.
EXPERIMENTS="${EXPERIMENTS:-baseline tf_sched flow_bridge}"
# SKIP_SHARED=1 reuses the existing data/ + checkpoints/tokenizer.pt (skips the
# ~50-min data-gen + tokenizer stages) and jumps straight to the dynamics sweep.
SKIP_SHARED="${SKIP_SHARED:-0}"

export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p logs
ts() { date "+%H:%M:%S"; }
say() { echo "[$(ts)] === $* ==="; }

# --- preflight: fail in second 1 if the env is wrong -----------------------
say "preflight"
python -c "import torch,numpy; assert torch.cuda.is_available(); print('  torch',torch.__version__,'CUDA ok')" \
  || { echo "ENV: run 'conda activate slowroads' (needs torch+CUDA+numpy)."; exit 1; }
( cd sim && node -e "require('playwright'); require('gifenc')" ) 2>/dev/null \
  || { echo "ENV: cd sim && npm install && npx playwright install chromium."; exit 1; }

# --- shared stages: dataset + tokenizer (trained once, reused by every experiment).
# SKIP_SHARED=1 reuses what's already on disk and jumps to the dynamics sweep.
if [ "$SKIP_SHARED" = "1" ]; then
  say "SKIP_SHARED=1 — reusing existing $DATA + checkpoints/tokenizer.pt"
  [ -f "$DATA/manifest.json" ] || { echo "  no $DATA/manifest.json — can't skip data gen"; exit 1; }
  [ -f checkpoints/tokenizer.pt ] || { echo "  no checkpoints/tokenizer.pt — can't skip tokenizer"; exit 1; }
else
  say "data: $STEPS frames @ ${SIZE}px -> $DATA"
  node sim/headless/generate_pixels.mjs --seed 1 --steps "$STEPS" --size "$SIZE" 2>&1 | tee logs/gen.log
  python3 - "$DATA" <<'PY'
import numpy as np, json, sys
d=sys.argv[1]; m=json.load(open(f"{d}/manifest.json")); n=len(m["samples"])
a=np.load(f"{d}/"+m["samples"][n//10]["frame"]); b=np.load(f"{d}/"+m["samples"][n//2]["frame"])
assert a.std()>0.05 and np.abs(a-b).mean()>0.02, "gray/frozen frames — aborting"
print(f"  frames OK (std {a.std():.3f}, diff {np.abs(a-b).mean():.3f})")
PY

  say "tokenizer: hidden=$TOK_HIDDEN epochs=$TOK_EPOCHS grad=$GRAD_WEIGHT"
  python -m model.train_tokenizer --data "$DATA" \
      --hidden "$TOK_HIDDEN" --grad-weight "$GRAD_WEIGHT" --epochs "$TOK_EPOCHS" \
      2>&1 | tee logs/tokenizer.log
fi
python -m eval.eval_tokenizer --data "$DATA" --ckpt checkpoints/tokenizer.pt \
    2>&1 | tee logs/tokenizer_eval.log
say "-> eval/plots/tokenizer_recon.png"

# --- dynamics experiment sweep ---------------------------------------------
# Each case sets EXTRA (extra train_dynamics flags). Add your own here.
run_experiment() {
  local name="$1"; shift
  local extra="$*"
  local out="checkpoints/$name"
  mkdir -p "$out"
  say "experiment '$name'  (dynamics flags: ${extra:-none})"
  python -m model.train_dynamics --data "$DATA" --tokenizer checkpoints/tokenizer.pt \
      --out "$out" --epochs "$DYN_EPOCHS" $extra \
      2>&1 | tee "logs/$name.train.log"
  python -m eval.eval_dream --data "$DATA" \
      --tokenizer checkpoints/tokenizer.pt --dynamics "$out/dynamics.pt" \
      --steps "$DREAM_STEPS" 2>&1 | tee "logs/$name.dream.log"
  # eval_dream writes eval/plots/dream.gif; tag it by experiment so they persist.
  mv eval/plots/dream.gif "eval/plots/dream_$name.gif" 2>/dev/null || true
  say "-> eval/plots/dream_$name.gif"
}

# One experiment failing must not kill the sweep — log it and move on, so an
# unattended run still delivers the experiments that DO work.
for exp in $EXPERIMENTS; do
  ( set -e
    case "$exp" in
      baseline)    run_experiment baseline    --arch ar_transformer --d-model "$DYN_DMODEL" --n-layers "$DYN_LAYERS" ;;
      tf_sched)    run_experiment tf_sched     --arch ar_transformer --d-model "$DYN_DMODEL" --n-layers "$DYN_LAYERS" --tf-start 0.5 ;;
      big)         run_experiment big          --arch ar_transformer --d-model 512 --n-layers 8 ;;
      flow_bridge) run_experiment flow_bridge  --arch flow_bridge ;;
      *)           echo "  (skipping unknown experiment '$exp')" ;;
    esac
  ) || say "experiment '$exp' FAILED (see logs/$exp.*.log) — continuing sweep"
done

say "DONE. Compare the runs:"
echo "    eval/plots/tokenizer_recon.png     (shared tokenizer quality)"
for exp in $EXPERIMENTS; do echo "    eval/plots/dream_$exp.gif        + logs/$exp.dream.log (drift)"; done
echo "  drift per experiment:"
echo "    grep 'pixel drift' logs/*.dream.log"
echo "  scp back:  scp gpublaze:~/slower-roads/eval/plots/'*.gif' gpublaze:~/slower-roads/eval/plots/tokenizer_recon.png ."
