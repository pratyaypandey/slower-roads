"""Camera + palette calibration stats for a dataset (reference or sim render).

Prints the median car bounding box (the white car in the lower middle), the
horizon row, and median colours of the sky / road / grass / foliage / rock
classes. Run it on the reference and on a render, then tune the v2 renderer's
CAM and PAL constants until the numbers agree (docs/FIDELITY.md, step 4).

    python -m eval.calibrate_view data/reference/drive1 data/fidelity_baseline/v2_seed1
"""

import sys

import numpy as np

from eval.fidelity import frame_paths, load_rgb, horizon_row


def car_bbox(f):
    """Bounding box (fractions) of the bright, unsaturated blob in the lower middle."""
    mx, mn = f.max(-1), f.min(-1)
    white = (mx > 0.9) & (mx - mn < 0.06)
    h, w = white.shape
    white[: int(0.42 * h)] = False   # below the horizon haze
    white[:, : int(0.2 * w)] = white[:, int(0.8 * w):] = False
    ys, xs = np.nonzero(white)
    if len(ys) < 0.01 * h * w:
        return None
    # Robust extents: ignore stray white pixels (lane lines, clouds).
    y0, y1 = np.percentile(ys, [1, 99]) / h
    x0, x1 = np.percentile(xs, [2, 98]) / w
    return y0, y1, x0, x1


def palette(F):
    r, g, b = F[..., 0], F[..., 1], F[..., 2]
    v = F.max(-1)
    s = (v - F.min(-1)) / np.maximum(v, 1e-3)
    H = F.shape[1]
    rows = np.arange(H)[None, :, None]
    cols = np.arange(H)[None, None, :]
    car = (cols > 0.27 * H) & (cols < 0.73 * H) & (rows > 0.47 * H)
    classes = {
        "sky": (rows < 0.08 * H) & (b > r + 0.03) & (v > 0.6),
        "road": (rows > 0.6 * H) & (s < 0.12) & (v > 0.3) & (v < 0.62) & ~car,
        "grass": (rows > 0.4 * H) & (g > r) & (g > b + 0.1) & (v > 0.5) & ~car,
        "foliage": (g > r) & (g > b + 0.04) & (v < 0.4) & (v > 0.08),
        "rock": (r > g) & (r > b + 0.05) & (s > 0.15) & (s < 0.45) & (v > 0.3) & (v < 0.7),
    }
    out = {}
    for k, m in classes.items():
        px = F[m]
        out[k] = (len(px) / m.size, (np.median(px, 0) * 255).round().astype(int) if len(px) else None)
    return out


def stats(d, n=80, size=256):
    ps = frame_paths(d)
    idx = np.linspace(0, len(ps) - 1, min(n, len(ps))).astype(int)
    F = np.stack([load_rgb(ps[i], size) for i in idx])
    boxes = [b for b in (car_bbox(f) for f in F) if b]
    gray = F @ np.array([0.2126, 0.7152, 0.0722])
    return {
        "car": np.median(np.array(boxes), 0) if boxes else None,
        "car_found": len(boxes) / len(F),
        "horizon": float(np.median(horizon_row(gray))),
        "palette": palette(F),
    }


def main():
    for d in sys.argv[1:]:
        s = stats(d)
        print(f"== {d}")
        if s["car"] is not None:
            y0, y1, x0, x1 = s["car"]
            print(f"  car  top {y0:.3f} bottom {y1:.3f} left {x0:.3f} right {x1:.3f} "
                  f"width {x1 - x0:.3f} height {y1 - y0:.3f} (found {s['car_found']:.0%})")
        print(f"  horizon {s['horizon']:.3f}")
        for k, (frac, med) in s["palette"].items():
            print(f"  {k:8s} {frac:6.1%}  {med}")


if __name__ == "__main__":
    main()
