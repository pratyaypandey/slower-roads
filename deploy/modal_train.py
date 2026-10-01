"""Train and evaluate M3 on Modal.

Same shape as the tokenizer server: the image + GPU + storage are code, and the
job runs on demand with per-second billing (no pod to leave running). The
frozen tokenizer + the training data live on the Modal Volume `sr-m3-train`; the
run writes dynamics.pt / dynamics_best.pt / dynamics_metrics.jsonl back to it.

  # one-time: push the dataset (tokenizer.pt is already on the volume)
  modal volume put sr-m3-train data/seed1 /data/seed1

  modal run deploy/modal_train.py                      # train with defaults
  modal run deploy/modal_train.py --epochs 80 --batch-size 32
  modal run deploy/modal_train.py --extra "--tf-start 0.5 --n-layers 6"

  # pull results back
  modal volume get sr-m3-train /checkpoints/dynamics_best.pt      checkpoints/
  modal volume get sr-m3-train /checkpoints/dynamics_metrics.jsonl checkpoints/

Workspace/profile: slower-roads-m3. Train Volume: sr-m3-train.
"""

import os
import modal

app = modal.App("sr-m3")

# Image is code, not a Dockerfile: base + deps + the local packages the trainer
# and evals import. Mirrors modal_serve.py; adds pillow for the eval GIFs.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "pillow")
    .add_local_python_source("model", "eval")
)
TRAIN_VOLUME = os.environ.get("SR_MODAL_TRAIN_VOLUME", "sr-m3-train")
TEST_VOLUME = os.environ.get("SR_MODAL_TEST_VOLUME", "sr-m3-test")
VAL_VOLUME = os.environ.get("SR_MODAL_VAL_VOLUME", "sr-m3-val")
vol = modal.Volume.from_name(TRAIN_VOLUME)
test_vol = modal.Volume.from_name(TEST_VOLUME)
val_vol = modal.Volume.from_name(VAL_VOLUME)


@app.function(image=image, gpu="A100-80GB", volumes={"/models": vol}, timeout=6 * 60 * 60)
def train(argv: list[str]):
    """Run model.train_dynamics.main with data + checkpoint paths on the volume.

    Context-8 multi-step rollout training re-forwards a growing ~3.6k-token
    sequence (context+horizon) with autograd retained across the horizon, so peak
    memory is large and the variable per-step length fragments the CUDA caching
    allocator over a full epoch. Run on an 80GB A100 with expandable_segments to
    avoid the fragmentation OOM that killed the 40GB run mid-epoch. Eval/smoke
    stay on A10G.
    """
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
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
    print(f"committed checkpoints to volume {TRAIN_VOLUME}")


@app.function(image=image, gpu="A10G", timeout=30 * 60)
def smoke_m3():
    """GPU smoke of corrected actions, exact self-rollout, and λ conditioning."""
    import torch
    from model.train_dynamics import main

    print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
    main([
        "--smoke", "--device", "cuda", "--context", "2", "--horizon", "2",
        "--d-model", "64", "--n-layers", "2", "--n-heads", "4",
        "--anchor-cond", "--self-rollout", "--corruption-cond",
        "--corruption-min", "0.1", "--corruption-max", "0.2",
        "--attention-recency-bias", "0.05",
        "--mem-cross-attn", "--mem-tokens", "16", "--state-head", "--state-weight", "0.5",
    ])
    return {"ok": True, "device": torch.cuda.get_device_name(0)}


@app.function(
    image=image, gpu="A100",
    volumes={"/models": vol, "/test": test_vol, "/val": val_vol},
    timeout=6 * 60 * 60,
)
def evaluate_m3(argv: list[str]):
    # Long-horizon eval is a single-sequence sequential decode (3,600 frames x 256
    # tokens), so it is latency-bound; an A100 roughly halves the ~1 h A10G run.
    from eval.eval_m3 import main

    main(argv)
    vol.commit()


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


@app.function(image=image, gpu="A10G",
              volumes={"/models": vol, "/val": val_vol, "/test": test_vol},
              timeout=60 * 60)
def steering_eval(argv: list[str]):
    """Re-measure action responsiveness (left-vs-right dream divergence) with the
    corrected action/frame contract. Reads RGB frames from the val/test oracle
    volumes (the train volume keeps only latents), writes the GIF back."""
    from eval.eval_steering import main

    main(argv)
    vol.commit()


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


@app.local_entrypoint()
def smoke():
    print(smoke_m3.remote())


@app.local_entrypoint()
def steering(data: str = "/val/seed5",
             tokenizer: str = "/models/checkpoints/tokenizer_tc.pt",
             dynamics: str = "/models/checkpoints_m3_corrected/dynamics_best.pt",
             context: int = 8, steps: int = 30, window: int = 8,
             out: str = "/models/eval/steering"):
    argv = ["--data", data, "--tokenizer", tokenizer, "--dynamics", dynamics,
            "--context", str(context), "--steps", str(steps), "--window", str(window),
            "--out", out]
    steering_eval.remote(argv)


@app.local_entrypoint()
def m3_eval(data: str = "/test/seed2",
            tokenizer: str = "/models/checkpoints/tokenizer_tc.pt",
            dynamics: str = "/models/checkpoints_m3/dynamics_best.pt",
            starts: str = "100", horizons: str = "30,150,300",
            lambdas: str = "0,0.25,0.5,0.75,1",
            context: int = 8, window: int = 8,
            attention_every: int = 0,
            progress_every: int = 300,
            out: str = "/models/eval/m3_metrics.json"):
    argv = [
        "--data", data, "--tokenizer", tokenizer, "--dynamics", dynamics,
        "--starts", starts, "--horizons", horizons, "--lambdas", lambdas,
        "--context", str(context), "--window", str(window),
        "--attention-every", str(attention_every),
        "--progress-every", str(progress_every),
        "--out", out,
    ]
    evaluate_m3.remote(argv)
