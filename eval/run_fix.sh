#!/bin/bash
# Re-run the two loss-stack variants with LPIPS disabled (its forward crashed on
# the pod), then redo eval/upscale/profile/predictability across all four. nohup.
cd /workspace/sr || exit 1
export PYTHONUNBUFFERED=1
TRAIN="python -m model.train_tokenizer --data data/seed1 --epochs 40 --batch-size 64 --arch fsq_v2 --grad-weight 0.5 --loss-stack --lpips-weight 0"

echo "########## TRAIN fsq_v2 (stack, no lpips) ##########"
$TRAIN --out ck_v2stack
echo "########## TRAIN fsq_v2 (stack + trim 4375, no lpips) ##########"
$TRAIN --levels "7,5,5,5,5" --out ck_v2trim

for d in ck_v2stack ck_v2trim; do
  echo "########## EVAL $d ##########"
  python -m eval.eval_tokenizer --data data/seed1 --ckpt $d/tokenizer.pt --out plots_$d
done

echo "########## UPSCALE COMPARE (all 4) ##########"
python -m eval.upscale_compare --data data/seed1 \
  --ckpts ck_fsq/tokenizer.pt:fsq,ck_v2edge/tokenizer.pt:v2_edge,ck_v2stack/tokenizer.pt:v2_stack,ck_v2trim/tokenizer.pt:v2_4k \
  --frames 0,400,900,1400,2000,2500 --scale 6 --out compare_all.png

echo "########## DECODE LATENCY ##########"
python -m eval.profile_decode \
  --ckpts ck_fsq/tokenizer.pt:fsq,ck_v2edge/tokenizer.pt:v2_edge,ck_v2stack/tokenizer.pt:v2_stack,ck_v2trim/tokenizer.pt:v2_4k

echo "########## TOKEN PREDICTABILITY (v2_stack) ##########"
python -m eval.token_predictability --data data/seed1 --ckpt ck_v2stack/tokenizer.pt --frames 1500 --steps 400
echo "FIX_DONE"
