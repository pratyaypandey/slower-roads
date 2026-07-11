#!/bin/bash
# Push v2_stack reconstruction loss down toward ~0.001-0.002. Isolates the levers:
#   L1  training only  : EMA + cosine + long schedule, hidden 64  (same capacity as the 0.008 run)
#   L2  + capacity      : hidden 128
#   L3  + more bits     : hidden 128, 6-channel FSQ (vocab 102400, ~16.6 bits/tok, same 256 tokens)
# All use the loss stack (saliency-L1 + edge + FFL), LPIPS off (crashes on the pod).
# eval_tokenizer reports the plain-pixel L1 we're driving down. nohup -> lowloss.log, ends LOWLOSS_DONE.
cd /workspace/sr || exit 1
export PYTHONUNBUFFERED=1
pip install -q --break-system-packages pillow matplotlib 2>&1 | tail -1

EP=250
COMMON="--data data/seed1 --arch fsq_v2 --batch-size 64 --grad-weight 0.5 --loss-stack --lpips-weight 0 --cosine --ema 0.999 --epochs $EP"

echo "########## TRAIN L1  (EMA+cosine, hidden 64) ##########"
python -m model.train_tokenizer $COMMON --hidden 64  --out ck_L1
echo "########## TRAIN L2  (+capacity, hidden 128) ##########"
python -m model.train_tokenizer $COMMON --hidden 128 --out ck_L2
echo "########## TRAIN L3  (+bits, hidden 128, 6-ch FSQ 102400) ##########"
python -m model.train_tokenizer $COMMON --hidden 128 --levels "8,8,8,8,5,5" --out ck_L3

for d in ck_L1 ck_L2 ck_L3; do
  echo "########## EVAL $d ##########"
  python -m eval.eval_tokenizer --data data/seed1 --ckpt $d/tokenizer.pt --out plots_$d
done

echo "########## UPSCALE COMPARE (baseline vs L1/L2/L3) ##########"
python -m eval.upscale_compare --data data/seed1 \
  --ckpts ck_L1/tokenizer.pt:L1_train,ck_L2/tokenizer.pt:L2_cap,ck_L3/tokenizer.pt:L3_bits \
  --frames 0,400,900,1400,2000,2500 --scale 6 --out compare_lowloss.png

echo "LOWLOSS_DONE"
