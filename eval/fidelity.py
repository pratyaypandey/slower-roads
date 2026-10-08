"""Visual-fidelity gap between our renderer and the real Slow Roads reference.

Clean-room rebuild target: we can't (and don't) reproduce Slow Roads frame for
frame, so fidelity is measured as a DISTRIBUTION match between reference frames
(eval/ingest_recording.py) and frames from our sim (sim/headless/generate_pixels.mjs,
ideally at --size 512). All metrics are numpy + PIL only.

Metrics (each a distance; lower = closer to the game):
  lab_{L,a,b}_{sky,mid,ground}  1-D Wasserstein distance of CIELAB channels per
                                horizontal band (top 35% / middle / bottom 35%)
  edges_{sky,mid,ground}        W1 of per-frame mean gradient magnitude per band
                                (flat low-poly shading vs noisy texture)
  spectrum_slope                W1 of the per-frame radial power-spectrum slope
  horizon                       W1 of the horizon row (fraction of height)
  temporal                      W1 of mean |frame_t+1 - frame_t| (30 fps motion/flicker)

Noise floor (`ratio = ours / floor`; the gate in docs/FIDELITY.md is every ratio
<= --gate):
  clips   (default) how far a random stretch of real footage the same size as
          ours sits from the full reference: as many contiguous clips as ours has
          datasets, each as long as ours, median over draws. This is the right
          yardstick for a short render. "Is it as close as a real drive of this
          length would be?"
  halves  first vs second half of the reference (a much stricter, long-drive floor)

Usage:
    python -m eval.fidelity --ref data/reference/drive1 [--ref ...] \
        --ours data/seed1[,data/seed2,...] [--ours ...] [--name new] [--size 256] [--gate 3] \
        [--floor clips|halves] \
        [--out eval/fidelity_report.json] [--sheet eval/fidelity_sheet.png]
"""

import argparse
import json
import os

try:
    import numpy as np
except ImportError:
    raise SystemExit("numpy not found: run with ~/anaconda3/bin/python (or pip install numpy pillow)")

BANDS = {"sky": (0.0, 0.35), "mid": (0.35, 0.65), "ground": (0.65, 1.0)}


# --- loading -----------------------------------------------------------------
def frame_paths(manifest_dir):
    """Frame paths of one dataset, preferring the full-res reference PNGs."""
    mpath = os.path.join(manifest_dir, "manifest.json")
    if not os.path.isfile(mpath):
        raise SystemExit(f"no dataset at {manifest_dir} (missing manifest.json): "
                         "ingest a recording there first (python -m eval.ingest_recording)")
    with open(mpath) as f:
        m = json.load(f)
    full = (m.get("reference") or {}).get("full_dir")
    paths = []
    for s in m["samples"]:
        p = s["frame"]
        if full:
            p = os.path.join(full, os.path.splitext(os.path.basename(p))[0] + ".png")
        paths.append(os.path.join(manifest_dir, p))
    return paths


def load_rgb(path, size):
    """-> (size, size, 3) float32 in [0, 1]."""
    from PIL import Image

    if path.endswith(".npy"):
        arr = np.load(path)
        if arr.shape[0] == 3:
            arr = arr.transpose(1, 2, 0)
        img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    else:
        img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BILINEAR if img.size[0] < size else Image.BOX)
    return np.asarray(img, np.float32) / 255.0


def frame_pairs(dirs):
    pairs = []
    for d in dirs:
        ps = frame_paths(d)
        pairs += [(ps[i], ps[i + 1]) for i in range(len(ps) - 1)]
    if not pairs:
        raise ValueError(f"no frame pairs in {dirs}")
    return pairs


def load_set(dirs, size, max_frames):
    """Sample up to max_frames (frame, next_frame) pairs evenly across datasets."""
    return load_pairs(frame_pairs(dirs), size, max_frames)


def load_pairs(pairs, size, max_frames):
    idx = np.linspace(0, len(pairs) - 1, min(max_frames, len(pairs))).astype(int)
    frames = np.stack([load_rgb(pairs[i][0], size) for i in idx])
    nxt = np.stack([load_rgb(pairs[i][1], size) for i in idx])
    return frames, nxt


# --- per-frame features ------------------------------------------------------
def srgb_to_lab(rgb):
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = lin @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], axis=-1)


def band(x, name):
    h = x.shape[1]
    lo, hi = BANDS[name]
    return x[:, int(lo * h):int(hi * h)]


def spectrum_slope(gray):
    """Log-log slope of the radially averaged power spectrum, per frame."""
    n = gray.shape[-1]
    p = np.abs(np.fft.fftshift(np.fft.fft2(gray - gray.mean(axis=(1, 2), keepdims=True)),
                               axes=(1, 2))) ** 2
    yy, xx = np.indices((n, n))
    r = np.hypot(yy - n // 2, xx - n // 2).astype(int).ravel()
    keep = (r >= 2) & (r < n // 2)
    counts = np.bincount(r[keep])
    radial = np.stack([np.bincount(r[keep], fr.ravel()[keep]) for fr in p])
    ks = np.nonzero(counts)[0]
    lx = np.log(ks)
    ly = np.log(radial[:, ks] / counts[ks] + 1e-12)
    lx_c = lx - lx.mean()
    return (ly - ly.mean(axis=1, keepdims=True)) @ lx_c / (lx_c @ lx_c)


def horizon_row(gray):
    """Row of the strongest mean-luminance step in the top 80% (fraction of height)."""
    prof = gray.mean(axis=2)
    d = np.abs(np.diff(prof, axis=1))[:, : int(0.8 * gray.shape[1])]
    return d.argmax(axis=1) / gray.shape[1]


def features(frames, nxt):
    """Per-frame 1-D feature samples, keyed by metric name."""
    lab = srgb_to_lab(frames)
    gray = frames @ np.array([0.2126, 0.7152, 0.0722])
    gy, gx = np.gradient(gray, axis=(1, 2))
    grad = np.hypot(gx, gy)
    feats = {}
    for b in BANDS:
        lb = band(lab, b)
        for c, ch in enumerate("Lab"):
            # pixel-level colour distribution, subsampled to keep W1 cheap
            feats[f"lab_{ch}_{b}"] = lb[..., c][:, ::2, ::2].ravel()
        feats[f"edges_{b}"] = band(grad, b).mean(axis=(1, 2))
    feats["spectrum_slope"] = spectrum_slope(gray)
    feats["horizon"] = horizon_row(gray)
    feats["temporal"] = np.abs(nxt - frames).mean(axis=(1, 2, 3))
    return feats


def w1(a, b, q=512):
    """1-D Wasserstein-1 distance via matched quantiles."""
    qs = np.linspace(0, 1, q)
    return float(np.abs(np.quantile(a, qs) - np.quantile(b, qs)).mean())


def distances(fa, fb):
    return {k: w1(fa[k], fb[k]) for k in fa}


# --- report ------------------------------------------------------------------
def contact_sheet(ref, ours_sets, path, n=6):
    """Rows: reference, then each ours set; n evenly spaced frames per row."""
    from PIL import Image

    rows = [ref] + [o for _, o in ours_sets]
    tiles = [np.concatenate([r[i] for i in np.linspace(0, len(r) - 1, n).astype(int)], axis=1)
             for r in rows]
    Image.fromarray((np.concatenate(tiles, axis=0) * 255).astype(np.uint8)).save(path)


def clip_floor(ref_pairs, f_ref, n_clips, clip_len, size, max_frames, draws=12, seed=0):
    """Median distance from the full reference to random real clips shaped like ours."""
    rng = np.random.default_rng(seed)
    clip_len = min(clip_len, len(ref_pairs) // max(1, n_clips))
    runs = []
    for _ in range(draws):
        starts = rng.integers(0, len(ref_pairs) - clip_len + 1, size=n_clips)
        pairs = [p for s0 in starts for p in ref_pairs[s0:s0 + clip_len]]
        runs.append(distances(f_ref, features(*load_pairs(pairs, size, max_frames))))
    return {k: float(np.median([r[k] for r in runs])) for k in runs[0]}


def evaluate(ref_dirs, ours_groups, size=256, max_frames=400, gate=3.0, floor_mode="clips"):
    ref_pairs = frame_pairs(ref_dirs)
    ref, ref_next = load_pairs(ref_pairs, size, max_frames)
    half = len(ref) // 2
    if half < 8:
        raise ValueError("need at least 16 reference frames for a noise floor")
    f_ref = features(ref, ref_next)
    report = {"size": size, "reference": ref_dirs, "n_ref": len(ref), "floor_mode": floor_mode,
              "gate": gate, "ours": {}}
    if floor_mode == "halves":
        halves = distances(features(ref[:half], ref_next[:half]), features(ref[half:], ref_next[half:]))
    loaded, floors = [], {}
    for name, dirs in ours_groups:
        ours, ours_next = load_set(dirs, size, max_frames)
        loaded.append((name, ours))
        if floor_mode == "halves":
            floor = halves
        else:
            shape = (len(dirs), len(frame_pairs(dirs)) // len(dirs))
            if shape not in floors:
                floors[shape] = clip_floor(ref_pairs, f_ref, shape[0], shape[1], size, max_frames)
            floor = floors[shape]
        dist = distances(f_ref, features(ours, ours_next))
        ratio = {k: dist[k] / max(floor[k], 1e-6) for k in dist}
        report["ours"][name] = {
            "dirs": dirs, "n": len(ours), "floor": floor, "distance": dist, "ratio": ratio,
            "worst": sorted(ratio, key=ratio.get, reverse=True)[:5],
            "pass": all(r <= gate for r in ratio.values()),
        }
    return report, ref, loaded


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ref", action="append", required=True, help="reference dataset dir")
    ap.add_argument("--ours", action="append", required=True,
                    help="sim dataset dir, or comma-separated dirs evaluated as one group")
    ap.add_argument("--name", action="append",
                    help="label per --ours (repeat --name/--ours pairs to compare renderers)")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--gate", type=float, default=3.0)
    ap.add_argument("--floor", choices=["clips", "halves"], default="clips")
    ap.add_argument("--out", default="eval/fidelity_report.json")
    ap.add_argument("--sheet", default="eval/fidelity_sheet.png")
    a = ap.parse_args()

    groups = [g.split(",") for g in a.ours]
    names = a.name or [os.path.basename(os.path.normpath(g[0])) for g in groups]
    if len(names) != len(groups):
        ap.error("give one --name per --ours")
    report, ref, loaded = evaluate(a.ref, list(zip(names, groups)), a.size, a.max_frames, a.gate, a.floor)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)
    contact_sheet(ref, loaded, a.sheet)

    first = report["ours"][names[0]]
    print(f"floor: {a.floor}")
    print(f"{'metric':<16}{'floor':>9}" + "".join(f"{n[:12]:>14}" for n in names))
    for k in first["floor"]:
        print(f"{k:<16}{first['floor'][k]:>9.4f}"
              + "".join(f"{report['ours'][n]['ratio'][k]:>13.1f}x" for n in names))
    for n in names:
        o = report["ours"][n]
        print(f"{n}: {'PASS' if o['pass'] else 'FAIL'} (gate {a.gate}x); worst {o['worst'][:3]}")
    print(f"report -> {a.out}, sheet -> {a.sheet}")


if __name__ == "__main__":
    main()
