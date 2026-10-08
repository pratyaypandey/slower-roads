#!/bin/bash
# Upscaler ladder, sequential on the local GPU (MPS). ~60 min total.
set -e
cd "$(dirname "$0")/../.."
PY=export/.venv/bin/python; S=export/sr/sr_study.py
$PY $S train --arch tiny   --loss l1 --minutes 5
$PY $S train --arch small  --loss l1 --minutes 7
$PY $S train --arch medium --loss l1 --minutes 9
$PY $S train --arch small  --loss gan --init small_l1  --minutes 7  --lr 1e-4
$PY $S train --arch medium --loss gan --init medium_l1 --minutes 9  --lr 1e-4
$PY $S train --arch large  --loss gan --init realesrgan --name large_ftgan --minutes 12 --lr 5e-5 --batch 12
