"""Tokenizer evaluation: reconstruction, temporal stability, noise robustness,
codebook use -- per driving profile, on held-out episodes.

    python -m eval.eval_tokenizer --ckpt checkpoints/tokenizer_v2.pt \
        --split data/train_v2/split.json:val --out eval/tokenizer_v2_val.json
    python -m eval.eval_tokenizer --data data/seed1 --ckpt checkpoints/tokenizer.pt   # single dir

Metrics (all over every `--stride`-th frame of each episode):
  l1 / psnr       pixel reconstruction vs the input frame
  flip            fraction of the 256 tokens that change between consecutive frames
  flip_static     same, only on near-static pairs (pixel L1 between the two input
                  frames < --static-thresh). The M2 root cause was 84% of tokens
                  flipping between 99.3%-identical frames; a stable tokenizer keeps
                  this low, so dynamics sees change only where content changed.
  flip_noise      fraction of tokens that flip under N(0, --noise-std) pixel noise
  usage / perplexity   distinct codes used / exp(entropy) of the code histogram
Also writes a recon grid PNG (one original/recon pair per episode).
"""

import argparse
import json
import math
import os
from collections import defaultdict

import numpy as np
import torch

from model.data.frames import episode_frames
from model.registry import load_tokenizer


def resolve(args):
    if args.split:
        path, _, name = args.split.partition(":")
        root = os.path.dirname(os.path.abspath(path))
        return [os.path.join(root, d) for d in json.load(open(path))[name or "val"]]
    return args.data


def profile_of(d):
    try:
        return json.load(open(os.path.join(d, "manifest.json"))).get("policy") or os.path.basename(d)
    except (OSError, ValueError):
        return os.path.basename(d)


@torch.no_grad()
def eval_episode(model, d, device, stride, batch, noise_std, static_thresh, gen):
    fr = episode_frames(d)
    idx = np.arange(0, len(fr), stride)
    # consecutive pairs (i, i+1) at the true frame rate, sampled every `stride`
    out = defaultdict(float)
    n_frames = n_pairs = n_static = 0
    codes = []
    for s in range(0, len(idx), batch):
        i = idx[s:s + batch]
        i = i[i + 1 < len(fr)]
        x = torch.from_numpy(np.ascontiguousarray(fr[i])).to(device).float() / 255
        xn = torch.from_numpy(np.ascontiguousarray(fr[i + 1])).to(device).float() / 255
        recon, tok, _ = model(x)
        _, tok_n, _ = model(xn)
        noisy = (x + noise_std * torch.randn(x.shape, generator=gen, device="cpu").to(device)).clamp(0, 1)
        _, tok_noise, _ = model(noisy)
        mse = ((recon - x) ** 2).mean(dim=(1, 2, 3))
        out["l1"] += (recon - x).abs().mean(dim=(1, 2, 3)).sum().item()
        out["psnr"] += (10 * torch.log10(1 / mse.clamp_min(1e-10))).sum().item()
        flip = (tok != tok_n).float().mean(1)
        static = (x - xn).abs().mean(dim=(1, 2, 3)) < static_thresh
        out["flip"] += flip.sum().item()
        out["flip_static"] += flip[static].sum().item()
        out["flip_noise"] += (tok != tok_noise).float().mean(1).sum().item()
        n_frames += len(i)
        n_pairs += len(i)
        n_static += int(static.sum())
        codes.append(tok.flatten().cpu())
    res = {"l1": out["l1"] / n_frames, "psnr": out["psnr"] / n_frames, "flip": out["flip"] / n_pairs,
           "flip_static": out["flip_static"] / max(1, n_static), "static_pairs": n_static,
           "flip_noise": out["flip_noise"] / n_frames, "frames": n_frames}
    return res, torch.cat(codes)


def usage_stats(codes, k):
    h = torch.bincount(codes, minlength=k).float()
    p = h[h > 0] / h.sum()
    return int((h > 0).sum()), float(torch.exp(-(p * p.log()).sum()))


@torch.no_grad()
def grid(model, dirs, device, path):
    from PIL import Image
    tiles = []
    for d in dirs:
        fr = episode_frames(d)
        x = torch.from_numpy(np.ascontiguousarray(fr[len(fr) // 2:len(fr) // 2 + 1])).to(device).float() / 255
        r, _, _ = model(x)
        pair = torch.cat([x[0], r[0].clamp(0, 1)], dim=1)                 # original above recon
        tiles.append((pair.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    im = Image.fromarray(np.concatenate(tiles, axis=1)).resize((len(tiles) * 192, 384), Image.NEAREST)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    im.save(path)
    return path


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", default=["data/seed1"])
    p.add_argument("--split", default=None, help="<split.json>:<name>, e.g. data/train_v2/split.json:val")
    p.add_argument("--ckpt", default="checkpoints/tokenizer.pt")
    p.add_argument("--stride", type=int, default=6)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--noise-std", type=float, default=0.01)
    p.add_argument("--static-thresh", type=float, default=0.01)
    p.add_argument("--out", default=None, help="metrics JSON path (grid PNG written next to it)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                   ("mps" if torch.backends.mps.is_available() else "cpu"))
    args = p.parse_args(argv)

    device = torch.device(args.device)
    model, _ = load_tokenizer(args.ckpt, default_cfg={"hidden": 64}, map_location="cpu")
    model = model.to(device).eval()
    dirs = resolve(args)
    gen = torch.Generator().manual_seed(0)
    per_ep, by_prof, all_codes = {}, defaultdict(list), []
    for d in dirs:
        r, codes = eval_episode(model, d, device, args.stride, args.batch, args.noise_std, args.static_thresh, gen)
        r["usage"], r["perplexity"] = usage_stats(codes, model.codebook_size)
        per_ep[os.path.basename(d)] = r
        by_prof[profile_of(d)].append(r)
        all_codes.append(codes)
        print(f"{os.path.basename(d):32s} L1 {r['l1']:.4f}  PSNR {r['psnr']:.2f}  flip {r['flip']:.3f}  "
              f"flip_static {r['flip_static']:.3f} (n={r['static_pairs']})  flip_noise {r['flip_noise']:.3f}  "
              f"usage {r['usage']}", flush=True)
    keys = ["l1", "psnr", "flip", "flip_static", "flip_noise"]
    w = lambda rs, k: sum(r[k] * r["frames"] for r in rs) / sum(r["frames"] for r in rs)
    summary = {k: w(list(per_ep.values()), k) for k in keys}
    summary["usage"], summary["perplexity"] = usage_stats(torch.cat(all_codes), model.codebook_size)
    profiles = {pr: {k: w(rs, k) for k in keys} for pr, rs in by_prof.items()}
    print("overall:", "  ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in summary.items()))
    for pr, m in sorted(profiles.items()):
        print(f"  {pr:14s}", "  ".join(f"{k} {v:.4f}" for k, v in m.items()))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump({"ckpt": args.ckpt, "dirs": [os.path.basename(d) for d in dirs], "summary": summary,
                   "profiles": profiles, "episodes": per_ep, "codebook": model.codebook_size}, open(args.out, "w"), indent=1)
        print("wrote", args.out, "and", grid(model, dirs, device, args.out.replace(".json", ".png")))


if __name__ == "__main__":
    main()
