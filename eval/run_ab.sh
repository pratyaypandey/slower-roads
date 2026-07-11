#!/bin/bash
# Tokenizer A/B v2 — run unattended on the pod (nohup), writes ab.log, ends with ALL_DONE.
cd /workspace/sr || exit 1
export PYTHONUNBUFFERED=1
pip install -q --break-system-packages lpips pillow matplotlib 2>&1 | tail -1

TRAIN="python -m model.train_tokenizer --data data/seed1 --epochs 40 --batch-size 64"

echo "########## TRAIN fsq (baseline, edge) ##########"
$TRAIN --arch fsq    --grad-weight 0.5 --out ck_fsq
echo "########## TRAIN fsq_v2 (arch, edge) ##########"
$TRAIN --arch fsq_v2 --grad-weight 0.5 --out ck_v2edge
echo "########## TRAIN fsq_v2 (arch + loss stack) ##########"
$TRAIN --arch fsq_v2 --grad-weight 0.5 --loss-stack --out ck_v2stack
echo "########## TRAIN fsq_v2 (stack + trimmed vocab 4375) ##########"
$TRAIN --arch fsq_v2 --grad-weight 0.5 --loss-stack --levels "7,5,5,5,5" --out ck_v2trim

for d in ck_fsq ck_v2edge ck_v2stack ck_v2trim; do
  echo "########## EVAL $d ##########"
  python -m eval.eval_tokenizer --data data/seed1 --ckpt $d/tokenizer.pt --out plots_$d
done

echo "########## UPSCALE COMPARE ##########"
python -m eval.upscale_compare --data data/seed1 \
  --ckpts ck_fsq/tokenizer.pt:fsq,ck_v2edge/tokenizer.pt:v2_edge,ck_v2stack/tokenizer.pt:v2_stack,ck_v2trim/tokenizer.pt:v2_4k \
  --frames 0,400,900,1400,2000,2500 --scale 6 --out compare_all.png

echo "########## DECODE LATENCY ##########"
python -m eval.profile_decode \
  --ckpts ck_fsq/tokenizer.pt:fsq,ck_v2edge/tokenizer.pt:v2_edge,ck_v2stack/tokenizer.pt:v2_stack,ck_v2trim/tokenizer.pt:v2_4k

echo "########## TOKEN PREDICTABILITY (full-vocab: edge vs stack) ##########"
python -m eval.token_predictability --data data/seed1 --ckpt ck_v2edge/tokenizer.pt  --frames 1500 --steps 400
python -m eval.token_predictability --data data/seed1 --ckpt ck_v2stack/tokenizer.pt --frames 1500 --steps 400

echo "ALL_DONE"
