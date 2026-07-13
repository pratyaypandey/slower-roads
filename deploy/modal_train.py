"""Train the M2 dynamics core on Modal — the GPU counterpart to modal_serve.py.

Same shape as the tokenizer server: the image + GPU + storage are code, and the
job runs on demand with per-second billing (no pod to leave running). The
frozen tokenizer + the training data live on the Modal Volume `sr-models`; the
run writes dynamics.pt / dynamics_best.pt / dynamics_metrics.jsonl back to it.

  # one-time: push the dataset (tokenizer.pt is already on the volume)
  modal volume put sr-models data/seed1 /data/seed1

  modal run deploy/modal_train.py                      # train with defaults
  modal run deploy/modal_train.py --epochs 80 --batch-size 32
  modal run deploy/modal_train.py --extra "--tf-start 0.5 --n-layers 6"

  # pull results back
  modal volume get sr-models /checkpoints/dynamics_best.pt      checkpoints/
  modal volume get sr-models /checkpoints/dynamics_metrics.jsonl checkpoints/

Workspace: slower-roads. Volume: sr-models (same store as the tokenizer).
"""

import modal

app = modal.App("sr-dynamics")

# Image is code, not a Dockerfile: base + deps + the local packages the trainer
# and evals import. Mirrors modal_serve.py; adds pillow for the eval GIFs.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "pillow")
    .add_local_python_source("model", "eval")
)
vol = modal.Volume.from_name("sr-models")


@app.function(image=image, gpu="A100", volumes={"/models": vol}, timeout=6 * 60 * 60)
def train(argv: list[str]):
    """Run model.train_dynamics.main with data + checkpoint paths on the volume.

    The core is ~10M params over ~2.5k-token sequences, so an A10G is ample and
    cheaper than an A100 — bump the gpu= above only if throughput demands it.
    """
    import torch
    from model.train_dynamics import main

    print("cuda:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")

    # Commit after every epoch so checkpoints + metrics persist to the volume as
    # they're written — a crash or early `modal app stop` keeps whatever epochs
    # already ran, and `modal volume get` can pull the best checkpoint mid-run.
    def commit(epoch):
        vol.commit()

    main(argv, on_epoch_end=commit)
    vol.commit()
    print("committed checkpoints to volume sr-models")


@app.function(image=image, gpu="A10G", volumes={"/models": vol}, timeout=6 * 60 * 60)
def train_tok(argv: list[str]):
    """Run model.train_tokenizer with data + checkpoint paths on the volume.

    The tokenizer trainer checkpoints every epoch (no callback hook), so we commit
    the volume once at the end. Used for the temporal-consistency retrain (K)."""
    import torch
    from model.train_tokenizer import main

    print("cuda:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    main(argv)
    vol.commit()
    print("committed tokenizer to volume sr-models")


@app.function(image=image, gpu="A10G", volumes={"/models": vol}, timeout=60 * 60)
def precompute_one(seed_dir: str, tokenizer: str):
    """Precompute frozen-tokenizer latents.npy for ONE seed on the volume, then
    commit. Parallelized per-seed (loading 2501 small .npy frames off the volume is
    the bottleneck, so one container per seed + per-seed commit is fast + robust)."""
    import torch
    from model.registry import load_tokenizer
    from model.precompute_latents import encode_seed
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok, _ = load_tokenizer(tokenizer, default_cfg={"hidden": 64}, map_location=dev)
    tok = tok.to(dev).eval()
    out, shape = encode_seed(tok, seed_dir, 64, dev)
    vol.commit()
    return {"seed_dir": seed_dir, "shape": list(shape)}


@app.local_entrypoint()
def latents(tokenizer: str = "/models/checkpoints_tc/tokenizer.pt",
            seeds: str = "1,3,4,5,6,7,8,9,10,11,12"):
    dirs = [f"/models/data/seed{s}" for s in seeds.split(",")]
    print(f"precomputing latents for {len(dirs)} seeds in parallel...")
    for res in precompute_one.starmap([(d, tokenizer) for d in dirs]):
        print("done:", res)


@app.local_entrypoint()
def tokenizer(epochs: int = 15, batch_size: int = 32, lr: float = 1e-4,
              hidden: int = 128, temporal_weight: float = 0.05, noise_weight: float = 0.05,
              noise_std: float = 0.02, out_name: str = "tokenizer_tc.pt",
              data: str = "/models/data/seed1 /models/data/seed3 /models/data/seed4 /models/data/seed5",
              resume: str = "/models/tokenizer.pt", extra: str = ""):
    # Fine-tune the existing (temporally-unstable) tokenizer with the temporal +
    # noise consistency losses on multi-seed data, holding seed2 out entirely.
    # Writes to a NEW name so the old tokenizer.pt stays intact until the retrain
    # is validated. --frame-cache is required for the temporal pairs.
    argv = ["--data", *data.split(), "--arch", "fsq_v2", "--hidden", str(hidden),
            "--out", "/models/checkpoints_tc", "--frame-cache", "--cosine", "--ema", "0.999",
            "--loss-stack", "--epochs", str(epochs), "--batch-size", str(batch_size),
            "--lr", str(lr), "--temporal-weight", str(temporal_weight),
            "--noise-weight", str(noise_weight), "--noise-std", str(noise_std)]
    if resume:
        argv += ["--resume", resume, "--reset-epoch"]  # fine-tune: fresh epochs, new objective
    argv += extra.split()
    print("launching tokenizer:", " ".join(argv))
    train_tok.remote(argv)


@app.local_entrypoint()
def main(epochs: int = 40, batch_size: int = 16, lr: float = 3e-4,
         context: int = 4, horizon: int = 6, d_model: int = 256,
         n_layers: int = 4, n_heads: int = 4, tf_start: float = 0.5,
         eval_every: int = 1, patience: int = 6,
         data: str = "/models/data/seed1", val_data: str = "",
         tokenizer: str = "/models/tokenizer.pt",
         out: str = "/models/checkpoints", extra: str = ""):
    # Assemble the same CLI train_dynamics parses locally, pointed at the volume.
    # --amp is a no-op off CUDA but on for the A10G run. `data` is space-separated
    # seed dirs (train on all); `val_data` is a held-out seed for the generalization
    # val (whole trajectory). Validate every epoch + early-stop on val. `extra`
    # passes flags through verbatim.
    argv = ["--data", *data.split(),
            "--tokenizer", tokenizer, "--out", out,
            "--epochs", str(epochs), "--batch-size", str(batch_size), "--lr", str(lr),
            "--context", str(context), "--horizon", str(horizon),
            "--d-model", str(d_model), "--n-layers", str(n_layers), "--n-heads", str(n_heads),
            "--tf-start", str(tf_start), "--eval-every", str(eval_every),
            "--patience", str(patience), "--amp"]
    if val_data:
        argv += ["--val-data", val_data]
    argv += extra.split()
    print("launching:", " ".join(argv))
    train.remote(argv)
