#!/usr/bin/env bash
# Download the released model weights into checkpoints/ and verify sha256.
#   scripts/fetch_models.sh            # default release tag below
#   scripts/fetch_models.sh <tag>
# Tokenizers -> checkpoints/v2/, upscalers -> checkpoints/sr/<name>.pt (the paths the
# training/eval scripts already use). See MODELS.md for what each file is.
set -euo pipefail
TAG="${1:-models-v2-2026-10}"
REPO="pratyaypandey/slower-roads"
BASE="https://github.com/$REPO/releases/download/$TAG"
cd "$(dirname "$0")/.."
mkdir -p checkpoints/v2 checkpoints/sr
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
curl -fsSL "$BASE/manifest.json" -o "$tmp/manifest.json"
python3 - "$tmp/manifest.json" "$BASE" <<'PY'
import hashlib, json, os, sys, urllib.request
man, base = json.load(open(sys.argv[1])), sys.argv[2]
for name, meta in man.items():
    dst = f"checkpoints/sr/{name[3:]}" if name.startswith("sr_") else f"checkpoints/v2/{name}"
    def ok():
        if not os.path.exists(dst): return False
        h = hashlib.sha256(open(dst, "rb").read()).hexdigest()
        return h == meta["sha256"]
    if ok():
        print(f"ok (cached) {dst}"); continue
    urllib.request.urlretrieve(f"{base}/{name}", dst)
    if not ok(): sys.exit(f"sha256 mismatch for {dst}")
    print(f"ok {dst}  {meta['bytes'] / 1e6:.1f} MB")
PY
# Pretrained base of sr_large_ftgan (BSD-3, Real-ESRGAN); only needed to re-run the SR study.
[ -f checkpoints/sr/realesr-general-x4v3.pth ] || curl -fsSL -o checkpoints/sr/realesr-general-x4v3.pth \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth
echo "done: checkpoints/v2/ and checkpoints/sr/"
