"""Slim checkpoints into shareable inference weights + a sha256 manifest.

Training checkpoints carry optimizer state and raw (non-EMA) weights; the
released files keep only what `model.registry.load_tokenizer` needs
(builder, cfg, EMA weights), so they load exactly like the originals.

    export/.venv/bin/python scripts/export_weights.py --out dist/models
Then publish with `gh release create` (see MODELS.md); collaborators fetch with
`scripts/fetch_models.sh`.
"""
import argparse
import hashlib
import json
import os
import shutil

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOKENIZERS = {   # release name -> local checkpoint
    "tokenizer_v2_tc025.pt": "checkpoints/v2/tokenizer_v2_tc025.pt",   # chosen default
    "tokenizer_v2.pt": "checkpoints/v2/tokenizer_v2.pt",
    "tokenizer_v2_tc10.pt": "checkpoints/v2/tokenizer_v2_tc10.pt",
}
UPSCALERS = ["tiny_l1", "small_l1", "small_gan", "medium_l1", "medium_gan", "large_ftgan"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "dist", "models"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = {}
    for name, src in TOKENIZERS.items():
        ck = torch.load(os.path.join(ROOT, src), map_location="cpu")
        slim = {k: ck[k] for k in ("builder", "cfg", "hidden", "epoch") if k in ck}
        slim["model"] = ck["model"]            # EMA weights (what eval/inference use)
        torch.save(slim, os.path.join(a.out, name))
    for n in UPSCALERS:                         # plain state_dicts, already small
        shutil.copy(os.path.join(ROOT, "checkpoints", "sr", n + ".pt"), os.path.join(a.out, f"sr_{n}.pt"))
    for f in sorted(os.listdir(a.out)):
        if f.endswith(".pt"):
            p = os.path.join(a.out, f)
            files[f] = {"bytes": os.path.getsize(p), "sha256": sha256(p)}
            print(f"{f:28s} {files[f]['bytes'] / 1e6:6.1f} MB  {files[f]['sha256'][:12]}")
    json.dump(files, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
