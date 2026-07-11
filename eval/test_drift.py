"""Numpy-only verification of the drift metric (eval/drift.py).

Asserts the two behaviors ROADMAP §7 asks the metric to have: a perfect rollout
gives ~0 drift at every step, and a perturbation that compounds with rollout
length produces a monotonically growing curve.

Run: python3 eval/test_drift.py
"""

import os
import sys

import numpy as np

# Run the same way everywhere: `python3 eval/test_drift.py` from the repo root.
# Inject the repo root so the absolute `eval.` import resolves.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from eval import drift  # noqa: E402


def test_perfect_prediction_is_zero():
    rng = np.random.default_rng(0)
    H = 8
    z = rng.standard_normal((H, 64, 5))
    frames = rng.uniform(0, 1, (H, 3, 64, 64))
    curves = drift.drift_curves(pred_latents=z, gt_latents=z,
                                pred_frames=frames, gt_frames=frames)
    assert np.allclose(curves["latent_l2"], 0.0)
    assert np.allclose(curves["pixel_l1"], 0.0)
    print("ok  perfect prediction -> ~0 drift at every step")


def test_drift_grows_with_rollout_length():
    # Model that's right at step 0 and accumulates error each step (compounding
    # autoregressive drift): step k off by a perturbation of magnitude ~k.
    rng = np.random.default_rng(1)
    H = 10
    gt_z = rng.standard_normal((H, 64, 5))
    gt_frames = rng.uniform(0, 1, (H, 3, 32, 32))

    step = np.arange(H)
    pred_z = gt_z + step.reshape(H, 1, 1) * rng.standard_normal((H, 64, 5)) * 0.1
    pred_frames = gt_frames + step.reshape(H, 1, 1, 1) * 0.01  # steady per-step pixel drift

    curves = drift.drift_curves(pred_latents=pred_z, gt_latents=gt_z,
                                pred_frames=pred_frames, gt_frames=gt_frames,
                                pixel_metric="mse")
    lat, pix = curves["latent_l2"], curves["pixel_mse"]
    assert lat[0] == 0.0 and pix[0] == 0.0
    assert np.all(np.diff(lat) > 0), lat       # strictly increasing latent drift
    assert np.all(np.diff(pix) > 0), pix       # strictly increasing pixel drift
    print(f"ok  drift grows with step: latent {lat[0]:.3f}->{lat[-1]:.3f}, "
          f"pixel {pix[0]:.4f}->{pix[-1]:.4f}")


def test_metric_variants_and_shapes():
    rng = np.random.default_rng(2)
    a = rng.uniform(0, 1, (5, 3, 16, 16))
    b = rng.uniform(0, 1, (5, 3, 16, 16))
    assert drift.pixel_drift(a, b, "l1").shape == (5,)
    assert drift.pixel_drift(a, b, "mse").shape == (5,)
    assert np.all(drift.pixel_drift(a, b, "l1") >= 0)
    # L1 (mean abs) and MSE (mean sq) of the same [0,1] error differ.
    assert not np.allclose(drift.pixel_drift(a, b, "l1"), drift.pixel_drift(a, b, "mse"))
    for bad in [np.zeros((3, 4)), np.zeros((4, 4))]:
        try:
            drift.latent_drift(np.zeros((3, 4)), bad) if bad.shape[0] == 4 else None
        except ValueError:
            pass
    try:
        drift.latent_drift(np.zeros((3, 4)), np.zeros((4, 4)))
    except ValueError:
        print("ok  metric variants, non-negativity, shape-mismatch guard")
    else:
        raise AssertionError("expected ValueError on shape mismatch")


if __name__ == "__main__":
    test_perfect_prediction_is_zero()
    test_drift_grows_with_rollout_length()
    test_metric_variants_and_shapes()
    print("\nall drift tests passed")
