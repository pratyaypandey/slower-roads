"""Companding transforms for the FSQ bottleneck (ablation slot).

FSQ's default is a uniform grid (companding=None). The AEP/rate-distortion study
in fsq_variants_study.py showed a density-matched (companded) quantizer beats the
uniform grid by ~1 dB on Gaussian latents — small, because tanh already companders
most of it. This module is the plug-and-play slot to try a learned/erf compander
on the *real* trained latent distribution.

A compander is a callable applied to z_cont BEFORE the tanh bound in FSQ.quantize;
it must be monotonic and shape-preserving. `get_compander(name)` returns one by
name so it can be selected via config and saved in the checkpoint.
"""


def get_compander(name):
    if name in (None, "identity"):
        return None  # FSQ treats None as identity — the uniform-grid default
    if name == "erf":
        raise NotImplementedError(
            "erf compander is a registered ablation stub. Intended: map z through "
            "an erf/Gaussian-CDF warp so quantization cells are denser near the "
            "origin (equiprobable under a Gaussian latent). Fill in after checking "
            "the trained latent histogram — see model/tokenizer/FSQ_VARIANTS.md."
        )
    raise KeyError(f"unknown compander {name!r}; available: identity, erf(stub)")
