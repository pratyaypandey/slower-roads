"""Verify FSQ tokenizer shapes and quantization math.

Degrades gracefully so it runs anywhere:
    - torch present   -> exercise the real FSQAutoencoder end-to-end (shapes).
    - torch absent     -> numpy reimplementation of the FSQ math, if numpy is present.
    - numpy absent too -> pure-Python reimplementation.

All three paths assert the same numeric properties of the quantizer:
    1. bounded codes land in each channel's level range,
    2. indices lie in [0, prod(L)),
    3. codes_to_indices / indices_to_codes are exact inverses.
"""

import math
import os
import sys

# Run the same way everywhere: `python3 model/tokenizer/test_shapes.py` from the
# repo root. Inject the repo root so absolute `model.` imports resolve.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

LEVELS = [8, 8, 8, 5, 5]
G = 8
EPS = 1e-3


# --- reference FSQ math (numeric, framework-free) --------------------------
# Mirrors FSQ._bound / quantize / codes_to_indices / indices_to_codes.

def _bound_scalar(z, L):
    half_l = (L - 1) * (1 + EPS) / 2
    offset = 0.5 if L % 2 == 0 else 0.0
    shift = math.atanh(offset / half_l) if offset else 0.0
    return math.tanh(z + shift) * half_l - offset


def quantize_vec(z_vec, levels):
    return [round(_bound_scalar(z, L)) for z, L in zip(z_vec, levels)]


def basis_of(levels):
    basis, acc = [], 1
    for L in levels:
        basis.append(acc)
        acc *= L
    return basis


def codes_to_index(codes, levels):
    basis = basis_of(levels)
    return sum((c + L // 2) * b for c, L, b in zip(codes, levels, basis))


def index_to_codes(idx, levels):
    basis = basis_of(levels)
    return [((idx // b) % L) - L // 2 for L, b in zip(levels, basis)]


def prod(xs):
    p = 1
    for x in xs:
        p *= x
    return p


# --- numeric property checks (used by numpy + pure-python paths) -----------

def check_fsq_math(rand_uniform):
    """rand_uniform() -> a float in [-3, 3]; run assertions over a grid of samples."""
    codebook = prod(LEVELS)

    seen = set()
    for _ in range(4000):
        z = [rand_uniform() for _ in LEVELS]
        codes = quantize_vec(z, LEVELS)

        for c, L in zip(codes, LEVELS):
            lo, hi = -(L // 2), L - 1 - (L // 2)
            assert lo <= c <= hi, f"code {c} out of range for L={L}"

        idx = codes_to_index(codes, LEVELS)
        assert 0 <= idx < codebook, f"index {idx} out of [0,{codebook})"

        assert index_to_codes(idx, LEVELS) == codes, "indices_to_codes not inverse"
        seen.add(idx)

    # every index round-trips through codes and back
    for idx in range(codebook):
        assert codes_to_index(index_to_codes(idx, LEVELS), LEVELS) == idx

    return codebook, len(seen)


# --- entry points ----------------------------------------------------------

def run_torch():
    import torch
    from model.tokenizer.fsq_autoencoder import (
        FSQAutoencoder,
        reconstruction_loss,
        count_parameters,
    )

    torch.manual_seed(0)
    model = FSQAutoencoder(levels=LEVELS)
    B = 2
    frame = torch.rand(B, 3, 64, 64)
    recon, indices, z_cont = model(frame)

    C = len(LEVELS)
    assert z_cont.shape == (B, G * G, C), z_cont.shape
    assert indices.shape == (B, G * G), indices.shape
    assert recon.shape == (B, 3, 64, 64), recon.shape
    assert indices.min() >= 0 and indices.max() < model.fsq.codebook_size
    assert 0.0 <= float(recon.min()) and float(recon.max()) <= 1.0

    # round-trip: indices -> codes -> indices is exact
    codes = model.fsq.indices_to_codes(indices)
    assert torch.equal(model.fsq.codes_to_indices(codes), indices)

    # STE: quantized output carries a gradient back to the continuous latent.
    # z must be a leaf tensor for .grad to populate, so detach before requiring
    # grad (z_cont already has a grad_fn from the encoder).
    z = z_cont.detach().clone().requires_grad_(True)
    model.fsq.quantize(z).sum().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()

    _ = reconstruction_loss(recon, frame)
    print(f"[torch] shapes OK  params={count_parameters(model)/1e6:.2f}M  "
          f"codebook={model.fsq.codebook_size}")


def run_numpy():
    import numpy as np
    rng = np.random.default_rng(0)
    codebook, seen = check_fsq_math(lambda: float(rng.uniform(-3, 3)))
    print(f"[numpy] FSQ math OK  codebook={codebook}  distinct_indices_sampled={seen}")


def run_python():
    import random
    random.seed(0)
    codebook, seen = check_fsq_math(lambda: random.uniform(-3, 3))
    print(f"[python] FSQ math OK  codebook={codebook}  distinct_indices_sampled={seen}")


if __name__ == "__main__":
    try:
        run_torch()
    except ImportError:
        try:
            run_numpy()
        except ImportError:
            run_python()
    print("all FSQ assertions passed")
