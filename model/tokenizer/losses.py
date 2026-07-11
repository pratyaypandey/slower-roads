"""Tokenizer loss stack (docs/tokenizer_research.md) — sharper, faithful recon
without the GAN's hallucination risk (which the downstream rollout loss punishes).

Terms:
  - saliency-weighted L1: up-weight high-gradient regions (the car, road/lane edges).
    Uses the image's own gradient magnitude as the saliency map — no mask/data-regen
    needed (a ground-truth car mask, if exported later, drops in the same way).
  - Focal Frequency Loss (Jiang 2021): penalize the missing high-frequency edge
    energy directly in the DFT, focally weighting the hard frequencies. Best
    sharpener with no adversarial downside.
  - LPIPS (optional; Zhang 2018): light perceptual term, VGG backbone. Lazily
    imported so training runs without the dep (weight forced to 0 then).
  - edge/gradient: reuse the baseline term (fsq_autoencoder._gradient_loss).

All terms operate in [0,1] (the sigmoid decoder's range); LPIPS is fed [-1,1].
"""

import torch
import torch.nn.functional as F

from model.tokenizer.fsq_autoencoder import _gradient_loss


def saliency_weight(target, alpha=2.0):
    """Per-image weight map 1 + alpha * normalized gradient magnitude. (B,1,H,W)."""
    dx = (target[..., :, 1:] - target[..., :, :-1]).abs()
    dy = (target[..., 1:, :] - target[..., :-1, :]).abs()
    gx = F.pad(dx, (0, 1))                    # back to (B,C,H,W)
    gy = F.pad(dy, (0, 0, 0, 1))
    g = (gx + gy).mean(dim=1, keepdim=True)   # (B,1,H,W)
    gmax = g.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    return 1.0 + alpha * (g / gmax)


def weighted_l1(recon, target, alpha=2.0):
    w = saliency_weight(target, alpha)
    return (w * (recon - target).abs()).mean()


def focal_frequency_loss(recon, target, focal_alpha=1.0):
    """Focal frequency loss: weighted L2 between the 2D DFT spectra, focally
    emphasizing the frequencies the recon gets most wrong."""
    fr = torch.fft.rfft2(recon, norm="ortho")
    ft = torch.fft.rfft2(target, norm="ortho")
    d2 = (fr - ft).abs() ** 2                                  # (B,C,H,W/2+1) freq distance
    w = d2.detach() ** focal_alpha
    w = w / w.amax(dim=(2, 3), keepdim=True).clamp_min(1e-8)   # normalize per (b,c) to [0,1]
    return (w * d2).mean()


class _LPIPS:
    """Lazy LPIPS wrapper; disables itself if the `lpips` package is absent."""

    def __init__(self, net="vgg", device="cpu"):
        self.fn = None
        try:
            import lpips
            self.fn = lpips.LPIPS(net=net, verbose=False).to(device).eval()
            for p in self.fn.parameters():
                p.requires_grad_(False)
        except Exception as e:  # noqa: BLE001
            print(f"[losses] LPIPS unavailable ({e}); w_lpips forced to 0")

    def __call__(self, recon, target):
        return self.fn(recon * 2 - 1, target * 2 - 1).mean()   # [0,1] -> [-1,1]


def make_lpips(device="cpu"):
    return _LPIPS(device=device)


def tokenizer_loss(recon, target, *, w_l1=1.0, saliency_alpha=2.0, w_edge=0.5,
                   w_ffl=0.1, w_lpips=0.1, lpips_fn=None):
    """Combined stack. Returns (total, parts_dict) so the trainer can log terms."""
    parts = {}
    total = w_l1 * weighted_l1(recon, target, saliency_alpha)
    parts["wl1"] = total.detach()
    if w_edge > 0:
        e = w_edge * _gradient_loss(recon, target); total = total + e; parts["edge"] = e.detach()
    if w_ffl > 0:
        f = w_ffl * focal_frequency_loss(recon, target); total = total + f; parts["ffl"] = f.detach()
    if w_lpips > 0 and lpips_fn is not None and lpips_fn.fn is not None:
        p = w_lpips * lpips_fn(recon, target); total = total + p; parts["lpips"] = p.detach()
    return total, parts


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(2, 3, 64, 64)
    y = (x + 0.1 * torch.rand_like(x)).clamp(0, 1)
    total, parts = tokenizer_loss(y, x, lpips_fn=make_lpips())
    print("total", float(total), {k: float(v) for k, v in parts.items()})
