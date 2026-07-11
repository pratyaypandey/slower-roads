"""fsq_v2 — an improved FSQ autoencoder tokenizer (research stack, see
docs/tokenizer_research.md). Registered as "fsq_v2" so it A/Bs against the
baseline "fsq" behind the same interface (train_tokenizer --arch fsq_v2). The FSQ
quantizer and losses are reused from fsq_autoencoder; only the encoder/decoder
architecture changes:

  1. PixelShuffle (depth-to-space) + ICNR init instead of ConvTranspose — removes
     the checkerboard artifacts that are glaring on flat low-poly regions and that
     inject periodic noise into the token stream (Odena 2016; Shi 2016; Aitken 2017).
  2. One self-attention block at the 16×16 bottleneck (enc + dec) — 256 tokens, so
     O(256²) is ~free, and it gives a global receptive field so flat regions stay
     consistent and the small car's few tokens are placed coherently (VQGAN/LDM).
  3. Residual blocks (GroupNorm+SiLU, last conv zero-init) at each resolution — the
     standard LDM/VQGAN quality lever.

Same [levels]/grid contract as the baseline, so the AR dynamics vocab is identical.
"""

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.dynamics.config import G, LEVELS
from model.registry import register_tokenizer
from model.tokenizer.fsq_autoencoder import FSQ, _num_downsamples


def _groups(c):
    return min(8, c) if c % 8 == 0 else 1


class ResBlock(nn.Module):
    """GroupNorm→SiLU→Conv3×3 twice, residual; last conv zero-init for stability."""

    def __init__(self, c):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(c), c)
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(c), c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class AttnBlock(nn.Module):
    """Spatial self-attention over H*W positions (free at the 16×16 bottleneck)."""

    def __init__(self, c):
        super().__init__()
        self.norm = nn.GroupNorm(_groups(c), c)
        self.qkv = nn.Conv2d(c, 3 * c, 1)
        self.proj = nn.Conv2d(c, c, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, c, h * w).unbind(1)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # (b, hw, c)
        out = F.scaled_dot_product_attention(q, k, v)                      # (b, hw, c)
        out = out.transpose(1, 2).reshape(b, c, h, w)
        return x + self.proj(out)


def icnr_(weight, scale=2, init=nn.init.kaiming_normal_):
    """ICNR init (Aitken 2017): initialize a pre-PixelShuffle conv as if it were a
    nearest-neighbour upsample, so the shuffle starts checkerboard-free."""
    out_c, in_c, kh, kw = weight.shape
    sub = torch.zeros(out_c // (scale ** 2), in_c, kh, kw)
    init(sub)
    sub = sub.transpose(0, 1).contiguous().view(in_c, out_c // (scale ** 2), -1)
    kernel = sub.repeat(1, 1, scale ** 2).view(in_c, out_c, kh, kw).transpose(0, 1)
    with torch.no_grad():
        weight.copy_(kernel)


class Upsample(nn.Module):
    """2× upsample via sub-pixel conv (conv→4C then PixelShuffle), ICNR-initialized."""

    def __init__(self, c_in, c_out, scale=2):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out * scale * scale, 3, padding=1)
        icnr_(self.conv.weight, scale)
        self.shuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        return self.shuffle(self.conv(x))


class EncoderV2(nn.Module):
    def __init__(self, channels, hidden=64, grid=G, frame_size=64):
        super().__init__()
        n = _num_downsamples(frame_size, grid)
        layers = [nn.Conv2d(3, hidden, 3, padding=1)]
        cin = hidden
        for i in range(n):
            cout = hidden if i == 0 else hidden * 2
            layers += [nn.Conv2d(cin, cout, 4, stride=2, padding=1), ResBlock(cout)]
            cin = cout
        # Bottleneck: res → attn → res, then project to the FSQ channels.
        layers += [ResBlock(cin), AttnBlock(cin), ResBlock(cin),
                   nn.GroupNorm(_groups(cin), cin), nn.SiLU(), nn.Conv2d(cin, channels, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, frame):
        h = self.net(frame)                      # (B, C, grid, grid)
        return h.flatten(2).transpose(1, 2)      # (B, grid*grid, C)


class DecoderV2(nn.Module):
    def __init__(self, channels, hidden=64, grid=G, frame_size=64):
        super().__init__()
        self.grid = grid
        n = _num_downsamples(frame_size, grid)
        width = hidden * 2 if n > 1 else hidden
        self.proj = nn.Conv2d(channels, width, 1)
        blocks = [ResBlock(width), AttnBlock(width), ResBlock(width)]
        cin = width
        for i in range(n):
            cout = hidden * 2 if i < n - 1 else hidden
            blocks += [Upsample(cin, cout), ResBlock(cout)]
            cin = cout
        blocks += [nn.GroupNorm(_groups(cin), cin), nn.SiLU(), nn.Conv2d(cin, 3, 3, padding=1)]
        self.net = nn.Sequential(*blocks)

    def forward(self, codes):
        b, n, c = codes.shape
        h = codes.transpose(1, 2).reshape(b, c, self.grid, self.grid)
        return torch.sigmoid(self.net(self.proj(h)))


class FSQAutoencoderV2(nn.Module):
    def __init__(self, levels=tuple(LEVELS), hidden=64, companding: Optional[Callable] = None,
                 grid=G, frame_size=64):
        super().__init__()
        self.grid = grid
        self.fsq = FSQ(list(levels), companding=companding)
        self.encoder = EncoderV2(self.fsq.num_channels, hidden, grid=grid, frame_size=frame_size)
        self.decoder = DecoderV2(self.fsq.num_channels, hidden, grid=grid, frame_size=frame_size)

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

    @property
    def codebook_size(self):
        return self.fsq.codebook_size

    @property
    def tokens_per_frame(self):
        return self.grid * self.grid


@register_tokenizer("fsq_v2")
def _build_fsq_v2(levels=tuple(LEVELS), hidden=64, companding=None):
    return FSQAutoencoderV2(levels=levels, hidden=hidden, companding=companding)


if __name__ == "__main__":
    torch.manual_seed(0)
    from model.tokenizer.fsq_autoencoder import count_parameters, reconstruction_loss
    m = FSQAutoencoderV2()
    print(f"fsq_v2 params: {count_parameters(m) / 1e6:.2f}M  codebook={m.codebook_size}  "
          f"tokens/frame={m.tokens_per_frame}")
    frame = torch.rand(2, 3, 64, 64)
    recon, idx, z = m(frame)
    print(f"z {tuple(z.shape)}  indices {tuple(idx.shape)} in [{int(idx.min())},{int(idx.max())}]  "
          f"recon {tuple(recon.shape)} range [{recon.min():.3f},{recon.max():.3f}]")
    print(f"L1 {reconstruction_loss(recon, frame):.4f}")
