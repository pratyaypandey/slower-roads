"""FSQ convolutional autoencoder (tokenizer).

Implements the §2 contract in docs/architecture.md:
    encode:  (B,3,64,64) -> z_cont (B, G*G, C)
    quantize: FSQ bound-then-round with a straight-through estimator
    codes_to_indices / indices_to_codes: mixed-radix over the C channels
    decode:  indices/codes -> (B,3,64,64), sigmoid output

The "codebook" is the implicit product grid of per-channel levels, so there is
no codebook, commitment loss, or dead codes.
"""

import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.dynamics.config import G, LEVELS  # single source for grid + levels
from model.registry import register_tokenizer


class FSQ(nn.Module):
    """Finite Scalar Quantization over `levels` per-channel bins.

    The bound is tanh scaled into each channel's level range (the §2
    `(L/2)*tanh` idea); an eps + half-offset makes even-level channels round to
    exactly L bins instead of L+1. `companding` is an optional pre-bound
    transform (identity by default) left as a clean seam for the non-uniform
    FSQ variants studied in fsq_variants_study.py.
    """

    def __init__(self, levels, companding: Optional[Callable] = None, eps: float = 1e-3):
        super().__init__()
        self.companding = companding if companding is not None else (lambda x: x)
        self.eps = eps

        levels_t = torch.tensor(levels, dtype=torch.int64)
        basis = torch.cumprod(torch.cat([torch.ones(1, dtype=torch.int64), levels_t[:-1]]), dim=0)
        self.register_buffer("levels", levels_t, persistent=False)
        self.register_buffer("basis", basis, persistent=False)          # mixed-radix strides
        self.register_buffer("half_width", levels_t // 2, persistent=False)  # shift to non-negative

    @property
    def num_channels(self) -> int:
        return int(self.levels.numel())

    @property
    def codebook_size(self) -> int:
        return int(torch.prod(self.levels).item())

    def _bound(self, z):
        L = self.levels.to(z.dtype)
        half_l = (L - 1) * (1 + self.eps) / 2
        offset = torch.where(L % 2 == 0, torch.tensor(0.5, dtype=z.dtype, device=z.device),
                             torch.tensor(0.0, dtype=z.dtype, device=z.device))
        shift = torch.atanh(offset / half_l)
        return torch.tanh(z + shift) * half_l - offset

    def quantize(self, z_cont):
        """z_cont (..., C) -> integer-grid codes (..., C), symmetric around 0, STE."""
        zb = self._bound(self.companding(z_cont))
        zq = zb + (torch.round(zb) - zb).detach()  # straight-through: grad flows through zb
        return zq

    def normalize(self, zq):
        """Grid codes -> roughly [-1, 1] continuous input for the decoder."""
        return zq / self.half_width.to(zq.dtype)

    def codes_to_indices(self, zq):
        """(..., C) integer-grid codes -> (...) int64 index in [0, prod(L))."""
        shifted = (zq + self.half_width).round().to(torch.int64)  # -> [0, L-1] per channel
        return (shifted * self.basis).sum(dim=-1)

    def indices_to_codes(self, indices):
        """(...) int64 index -> (..., C) integer-grid codes (inverse of codes_to_indices)."""
        per_channel = (indices.unsqueeze(-1) // self.basis) % self.levels
        return per_channel.to(torch.float32) - self.half_width


def conv_block(cin, cout, stride):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 4, stride=stride, padding=1),
        nn.GroupNorm(8, cout),
        nn.SiLU(),
    )


def deconv_block(cin, cout):
    return nn.Sequential(
        nn.ConvTranspose2d(cin, cout, 4, stride=2, padding=1),
        nn.GroupNorm(8, cout),
        nn.SiLU(),
    )


class Encoder(nn.Module):
    """(B,3,64,64) -> z_cont (B, G*G, C). Three stride-2 convs: 64->32->16->8."""

    def __init__(self, channels, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            conv_block(3, hidden, stride=2),
            conv_block(hidden, hidden * 2, stride=2),
            conv_block(hidden * 2, hidden * 2, stride=2),
            nn.Conv2d(hidden * 2, channels, 1),
        )

    def forward(self, frame):
        h = self.net(frame)                      # (B, C, 8, 8)
        return h.flatten(2).transpose(1, 2)      # (B, G*G, C)


class Decoder(nn.Module):
    """codes (B, G*G, C) -> (B,3,64,64). Mirror of the encoder, sigmoid output."""

    def __init__(self, channels, hidden=64):
        super().__init__()
        self.proj = nn.Conv2d(channels, hidden * 2, 1)
        self.net = nn.Sequential(
            deconv_block(hidden * 2, hidden * 2),  # 8->16
            deconv_block(hidden * 2, hidden),      # 16->32
            deconv_block(hidden, hidden),          # 32->64
            nn.Conv2d(hidden, 3, 3, padding=1),
        )

    def forward(self, codes):
        b, n, c = codes.shape
        h = codes.transpose(1, 2).reshape(b, c, G, G)
        h = self.proj(h)
        return torch.sigmoid(self.net(h))


class FSQAutoencoder(nn.Module):
    # levels default comes from config.LEVELS so the tokenizer and dynamics vocab
    # can't silently disagree; the value is unchanged from the old literal.
    def __init__(self, levels=tuple(LEVELS), hidden=64, companding: Optional[Callable] = None):
        super().__init__()
        self.fsq = FSQ(list(levels), companding=companding)
        self.encoder = Encoder(self.fsq.num_channels, hidden)
        self.decoder = Decoder(self.fsq.num_channels, hidden)

    def encode(self, frame):
        return self.encoder(frame)

    def decode(self, zq):
        return self.decoder(self.fsq.normalize(zq))

    def decode_indices(self, indices):
        return self.decode(self.fsq.indices_to_codes(indices))

    def forward(self, frame):
        z_cont = self.encode(frame)
        zq = self.fsq.quantize(z_cont)
        indices = self.fsq.codes_to_indices(zq)
        recon = self.decode(zq)
        return recon, indices, z_cont

    # --- Tokenizer protocol (model/interfaces.py) ---
    @property
    def codebook_size(self):
        return self.fsq.codebook_size

    @property
    def tokens_per_frame(self):
        return G * G


@register_tokenizer("fsq")
def _build_fsq(levels=tuple(LEVELS), hidden=64, companding=None):
    return FSQAutoencoder(levels=levels, hidden=hidden, companding=companding)


def reconstruction_loss(recon, target, kind="l1"):
    return F.l1_loss(recon, target) if kind == "l1" else F.mse_loss(recon, target)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    torch.manual_seed(0)
    model = FSQAutoencoder()
    print(f"levels={list(model.fsq.levels)}  C={model.fsq.num_channels}  "
          f"codebook={model.fsq.codebook_size}")
    print(f"parameters: {count_parameters(model) / 1e6:.2f}M")

    frame = torch.rand(2, 3, 64, 64)
    recon, indices, z_cont = model(frame)
    print(f"z_cont {tuple(z_cont.shape)}  indices {tuple(indices.shape)}  recon {tuple(recon.shape)}")
    print(f"indices in [{int(indices.min())}, {int(indices.max())}] < {model.fsq.codebook_size}")
    print(f"recon range [{recon.min():.3f}, {recon.max():.3f}]")
    print(f"L1 recon loss: {reconstruction_loss(recon, frame):.4f}")
