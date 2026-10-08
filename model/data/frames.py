"""Episode frame storage: packed uint8 arrays, with the per-frame files as fallback.

A train_v2 episode is 3,600 frames; as per-frame float32 .npy that is 177 MB and
3,600 file opens. `pack` writes one `frames_u8.npy` (N, 3, H, W) uint8 per
episode instead: 44 MB, one mmap, bit-exact (the capture path quantizes to
uint8 before the float32 conversion). Every reader goes through
`episode_frames`, which prefers the packed file and falls back to the
manifest's per-frame entries.

    python -m model.data.frames pack data/train_v2/ep*      # writes frames_u8.npy per episode
    python -m model.data.frames split data/train_v2          # writes split.json (val/test by seed)
"""

import argparse
import json
import os

import numpy as np

PACKED = "frames_u8.npy"


def _per_frame_u8(path):
    arr = np.load(path)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):          # HWC(A) -> CHW
        arr = np.transpose(arr[..., :3], (2, 0, 1))
    if arr.dtype != np.uint8:
        arr = np.round(arr * 255.0).clip(0, 255).astype(np.uint8)
    return arr


def episode_frames(episode_dir, mmap=True):
    """(N, 3, H, W) uint8 frames of one episode, aligned 1:1 with manifest samples."""
    packed = os.path.join(episode_dir, PACKED)
    if os.path.exists(packed):
        return np.load(packed, mmap_mode="r" if mmap else None)
    manifest = json.load(open(os.path.join(episode_dir, "manifest.json")))
    return np.stack([_per_frame_u8(os.path.join(episode_dir, s["frame"])) for s in manifest["samples"]])


def pack(episode_dir):
    out = os.path.join(episode_dir, PACKED)
    if os.path.exists(out):
        return out, np.load(out, mmap_mode="r").shape
    frames = episode_frames(episode_dir)
    tmp = out + ".tmp.npy"
    np.save(tmp, frames)
    os.replace(tmp, out)
    return out, frames.shape


def make_split(root, per_profile=(("val", 1), ("test", 1))):
    """Hold out whole episodes (= world seeds) across every driving profile:
    per profile, the last episode -> test, the one before -> val. Never trained on."""
    index = json.load(open(os.path.join(root, "index.json")))["episodes"]
    by = {}
    for e in index:
        by.setdefault(e["policy"], []).append(e)
    split = {"val": [], "test": [], "train": []}
    for pol, eps in by.items():
        eps = sorted(eps, key=lambda e: e["seed"])
        split["test"].append(eps[-1]["dir"])
        split["val"].append(eps[-2]["dir"])
        split["train"] += [e["dir"] for e in eps[:-2]]
    for k in split:
        split[k].sort()
    json.dump(split, open(os.path.join(root, "split.json"), "w"), indent=1)
    return split


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["pack", "split"])
    p.add_argument("dirs", nargs="+")
    a = p.parse_args(argv)
    if a.cmd == "pack":
        for d in a.dirs:
            out, shape = pack(d)
            print(f"{out} {shape}")
    else:
        s = make_split(a.dirs[0])
        print({k: len(v) for k, v in s.items()}, "val:", s["val"], "test:", s["test"])


if __name__ == "__main__":
    main()
