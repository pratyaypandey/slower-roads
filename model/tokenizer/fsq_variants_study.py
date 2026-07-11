"""Empirical comparison of FSQ quantizer variants on synthetic sources.

Tests Jai's intuition (jais_notes/notes.md): a uniform FSQ grid spends equal
code density everywhere, but if the encoder latent is Gaussian-ish then a
uniform grid wastes codes in low-probability corners. Companding matched to the
source density (AEP / rate-distortion argument) should do better at the same
bitrate. Lloyd-Max is the MSE-optimal scalar quantizer and gives the lower bound.

Numpy only. Each variant is a scalar quantizer defined by decision thresholds on
an iid source channel. Reconstruction uses the per-cell centroid (what a trained
decoder converges to), so MSE is comparable across variants at fixed rate
(rate = log2(L) bits/channel). FSQ operates channel-independently, so per-channel
scalar analysis is exact; multi-channel numbers are just the per-channel numbers
scaled by the channel count.
"""

import numpy as np


def sample_source(name, n, rng):
    if name == "gaussian":
        return rng.standard_normal(n)
    if name == "laplacian":
        return rng.laplace(0.0, 1.0 / np.sqrt(2.0), n)  # unit variance
    if name == "uniform":
        return rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), n)  # unit variance
    raise ValueError(name)


def centroid_mse(x, edges):
    """MSE of quantizing x with the given interior decision thresholds,
    reconstructing each cell to its conditional mean (centroid)."""
    idx = np.searchsorted(edges, x)
    recon = np.empty_like(x)
    total = 0.0
    for c in range(len(edges) + 1):
        mask = idx == c
        if not np.any(mask):
            continue
        recon[mask] = x[mask].mean()
    err = x - recon
    return float(np.mean(err * err)), idx


def code_entropy(idx, levels):
    counts = np.bincount(idx, minlength=levels).astype(np.float64)
    p = counts / counts.sum()
    nz = p > 0
    return float(-np.sum(p[nz] * np.log2(p[nz])))


def uniform_fsq_edges(x, levels):
    """Standard FSQ: tanh-bound to [-1, 1], then uniform levels. Uniform decision
    thresholds in tanh space map back to source space via atanh, giving the
    x-space partition this variant induces. (tanh is itself mildly density-
    matched, so this is already better than a raw uniform grid on x.)"""
    grid = np.linspace(-1.0, 1.0, levels + 1)[1:-1]
    return np.arctanh(grid)


def companded_edges(x, levels):
    """Equiprobable cells: decision thresholds at the source quantiles. This is
    the companding solution (warp through the source CDF, then uniform quantize),
    which makes every code equally likely -> maximal code entropy."""
    qs = np.linspace(0.0, 1.0, levels + 1)[1:-1]
    return np.quantile(x, qs)


def lloyd_max_edges(x, levels, iters=100):
    """MSE-optimal scalar quantizer via Lloyd's algorithm on the empirical source.
    Returns interior decision thresholds. This is the achievable lower bound."""
    reps = np.quantile(x, np.linspace(0.0, 1.0, levels + 2)[1:-1]).copy()
    reps = np.sort(reps)
    for _ in range(iters):
        edges = 0.5 * (reps[:-1] + reps[1:])
        idx = np.searchsorted(edges, x)
        new = reps.copy()
        for c in range(levels):
            mask = idx == c
            if np.any(mask):
                new[c] = x[mask].mean()
        if np.allclose(new, reps, atol=1e-9):
            reps = new
            break
        reps = new
    return 0.5 * (reps[:-1] + reps[1:])


def run_variant(variant, x, levels):
    if variant == "uniform_fsq":
        edges = uniform_fsq_edges(x, levels)
        mse, idx = centroid_mse(x, edges)
    elif variant == "companded":
        edges = companded_edges(x, levels)
        mse, idx = centroid_mse(x, edges)
    elif variant == "lloyd_max":
        edges = lloyd_max_edges(x, levels)
        mse, idx = centroid_mse(x, edges)
    else:
        raise ValueError(variant)
    ent = code_entropy(idx, levels)
    return mse, ent


def main():
    rng = np.random.default_rng(0)
    n = 400_000
    sources = ["gaussian", "laplacian", "uniform"]
    variants = ["uniform_fsq", "companded", "lloyd_max"]
    level_sweep = [5, 8, 16]

    rows = []
    for src in sources:
        x = sample_source(src, n, rng)
        for L in level_sweep:
            rate = np.log2(L)
            per = {}
            for v in variants:
                mse, ent = run_variant(v, x, L)
                per[v] = (mse, ent, ent / rate)
            rows.append((src, L, rate, per))

    variants_hdr = {
        "uniform_fsq": "UniformFSQ",
        "companded": "Companded",
        "lloyd_max": "LloydMax",
    }
    lines = []
    header = (
        f"{'source':<10} {'L':>3} {'bits':>5} | "
        + " | ".join(
            f"{variants_hdr[v]+' MSE':>16} {'util%':>6}" for v in variants
        )
    )
    lines.append(header)
    lines.append("-" * len(header))
    for src, L, rate, per in rows:
        parts = [f"{src:<10} {L:>3} {rate:>5.2f}"]
        for v in variants:
            mse, ent, util = per[v]
            parts.append(f"{mse:>16.6e} {100*util:>6.1f}")
        lines.append(" | ".join(parts))

    lines.append("")
    lines.append("PSNR-style MSE-reduction vs UniformFSQ (higher = better):")
    lines.append(
        f"{'source':<10} {'L':>3} | {'Companded':>12} {'LloydMax':>12}   (dB lower distortion)"
    )
    for src, L, rate, per in rows:
        base = per["uniform_fsq"][0]
        c_db = 10.0 * np.log10(base / per["companded"][0])
        l_db = 10.0 * np.log10(base / per["lloyd_max"][0])
        lines.append(f"{src:<10} {L:>3} | {c_db:>12.2f} {l_db:>12.2f}")

    table = "\n".join(lines)
    print(table)

    summary_path = __file__.rsplit("/", 1)[0] + "/fsq_variants_results.txt"
    with open(summary_path, "w") as f:
        f.write(table + "\n")
    print(f"\nsaved: {summary_path}")


if __name__ == "__main__":
    main()
