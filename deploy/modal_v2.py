"""v2 pipeline on Modal (renderer v2 / train_v2 data): tokenizer train, eval, latents.

Separate app + volumes from the legacy `sr-m3` stack so old-renderer data and
checkpoints can never mix in. Workspace/profile: slower-roads-m3.

  Volumes (create_if_missing):
    sr-v2-train  /models/data/train_v2/{split.json, <80 train episodes>}, /models/checkpoints_v2/
    sr-v2-val    /val/data/train_v2/<5 val episodes>
    sr-v2-test   /test/data/train_v2/<5 test episodes>   (mounted only by eval_tok on request)
  Each episode dir holds manifest.json + frames_u8.npy (model/data/frames.py).

  # upload (staging dir of hardlinks, see docs/TRAINING.md "v2 tokenizer")
  MODAL_PROFILE=slower-roads-m3 modal volume put sr-v2-train data/v2_upload/train /data/train_v2
  # train (detached: deploy once, then spawn -- a client-attached run dies with the client)
  MODAL_PROFILE=slower-roads-m3 modal deploy deploy/modal_v2.py
  MODAL_PROFILE=slower-roads-m3 python deploy/modal_v2.py spawn-tokenizer --epochs 20
"""

import os
import sys

import modal

app = modal.App("sr-v2")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "pillow")
    .add_local_python_source("model", "eval")
)
train_vol = modal.Volume.from_name("sr-v2-train", create_if_missing=True)
val_vol = modal.Volume.from_name("sr-v2-val", create_if_missing=True)
test_vol = modal.Volume.from_name("sr-v2-test", create_if_missing=True)

SPLIT = "/models/data/train_v2/split.json"
CKPT_DIR = "/models/checkpoints_v2"


@app.function(image=image, gpu="A100", volumes={"/models": train_vol}, timeout=12 * 60 * 60)
def train_tok(argv: list[str]):
    """model.train_tokenizer on the train split; checkpoints every epoch, committing
    the volume after each so a stopped run keeps its progress."""
    import threading
    import time
    import torch
    from model.train_tokenizer import main

    print("cuda:", torch.cuda.get_device_name(0), flush=True)
    stop = threading.Event()

    def committer():                      # trainer has no epoch hook: commit every 10 min
        while not stop.wait(600):
            train_vol.commit()
    threading.Thread(target=committer, daemon=True).start()
    t0 = time.time()
    main(argv)
    stop.set()
    train_vol.commit()
    print(f"done in {(time.time() - t0) / 3600:.2f} h; committed {CKPT_DIR}")


@app.function(image=image, gpu="A10G", timeout=2 * 60 * 60,
              volumes={"/models": train_vol, "/val": val_vol, "/test": test_vol})
def eval_tok(argv: list[str]):
    from eval.eval_tokenizer import main
    main(argv)
    train_vol.commit()


@app.function(image=image, gpu="A10G", timeout=60 * 60,
              volumes={"/models": train_vol, "/val": val_vol, "/test": test_vol})
def precompute_one(episode_dir: str, tokenizer: str):
    """latents.npy (N, 256) int32 for ONE episode, written next to its frames."""
    import torch
    from model.registry import load_tokenizer
    from model.precompute_latents import encode_seed
    tok, _ = load_tokenizer(tokenizer, default_cfg={"hidden": 128}, map_location="cuda")
    tok = tok.cuda().eval()
    out, shape = encode_seed(tok, episode_dir, 256, "cuda")
    for v in (train_vol, val_vol, test_vol):
        v.commit()
    return {"episode": episode_dir, "shape": list(shape)}


def tokenizer_argv(epochs=20, batch_size=64, lr=3e-4, hidden=128, temporal_weight=0.05,
                   noise_weight=0.05, noise_std=0.02, grad_weight=0.5, extra=""):
    # docs/VAE_RECIPE.md recipe (fsq_v2, h128, loss stack w/o LPIPS, cosine, EMA) with
    # the temporal + noise consistency terms on from the start (the M2 root-cause fix).
    return ["--split", SPLIT + ":train", "--arch", "fsq_v2", "--hidden", str(hidden),
            "--out", CKPT_DIR, "--frame-cache", "--cosine", "--ema", "0.999",
            "--loss-stack", "--lpips-weight", "0", "--grad-weight", str(grad_weight),
            "--epochs", str(epochs), "--batch-size", str(batch_size), "--lr", str(lr),
            "--temporal-weight", str(temporal_weight), "--noise-weight", str(noise_weight),
            "--noise-std", str(noise_std), "--device", "cuda", *extra.split()]


if __name__ == "__main__":
    # Detached launches against the deployed app (survive this client exiting).
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["spawn-tokenizer", "spawn-eval", "latents"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--extra", default="")
    ap.add_argument("--ckpt", default=CKPT_DIR + "/tokenizer.pt")
    ap.add_argument("--eval-split", default="val")
    ap.add_argument("--split-file", default="data/train_v2/split.json")
    a = ap.parse_args()
    if a.cmd == "spawn-tokenizer":
        fn = modal.Function.from_name("sr-v2", "train_tok")
        argv = tokenizer_argv(a.epochs, a.batch_size, a.lr, extra=a.extra)
        print("spawn train_tok:", " ".join(argv))
        print("call id:", fn.spawn(argv).object_id)
    elif a.cmd == "spawn-eval":
        fn = modal.Function.from_name("sr-v2", "eval_tok")
        mount = {"val": "/val", "test": "/test", "train": "/models"}[a.eval_split]
        split = json.load(open(a.split_file))[a.eval_split]
        dirs = [f"{mount}/data/train_v2/{d}" for d in split]
        out = f"/models/eval_v2/tokenizer_{a.eval_split}.json"
        argv = ["--data", *dirs, "--ckpt", a.ckpt, "--stride", "3", "--out", out, "--device", "cuda", *a.extra.split()]
        print("call id:", fn.spawn(argv).object_id, "->", out)
    else:
        fn = modal.Function.from_name("sr-v2", "precompute_one")
        s = json.load(open(a.split_file))
        dirs = ([f"/models/data/train_v2/{d}" for d in s["train"]] + [f"/val/data/train_v2/{d}" for d in s["val"]]
                + [f"/test/data/train_v2/{d}" for d in s["test"]])
        for r in fn.starmap([(d, a.ckpt) for d in dirs]):
            print(r)
