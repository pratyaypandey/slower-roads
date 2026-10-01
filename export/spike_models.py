"""Random-weight stand-ins for the shape we'd ship in the browser (HANDOFF Step 1).

FrameDynamics is the Step-3 architecture, written as ONE refinement pass over
the next frame:

  * frame-level causal / within-frame bidirectional: the S = 1 + N tokens of
    the frame being generated attend to the cached context frames plus each
    other, with no mask at all (the cache only ever holds past frames);
  * temporal RoPE on the frame index (absolute index, so K is roped once at
    commit time and never re-roped as the window slides) + a learned spatial
    embedding for position within the frame;
  * M3 skeleton-memory cross-attention (outside the KV cache, no RoPE) fed by a
    1x1 conv + pool over the skeleton raster, and the M3 state head;
  * sampling inside the graph (argmax + confidence), so a pass downloads N ids
    + N confidences rather than N x 12800 logits.

The KV cache is a fixed window of `ctx` frames passed in as past_k/past_v
(L, H, ctx*S, hd). Each pass also emits present_k/present_v = the window
rolled by one frame with this pass's K/V appended; the runtime keeps the
present from the frame's last ("commit") pass and feeds it back as the next
frame's past. Head "full" predicts all 12800 codes; head "fsq" predicts the
5 FSQ channels independently (8+8+8+5+5 = 34 logits) and recombines the id.

Every layer boundary is a plain residual add, so the graph can be split at any
block for the ROADMAP §3 steering injection.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LEVELS = [8, 8, 8, 5, 5]
NUM_VISUAL = math.prod(LEVELS)       # 12800
NUM_KEYS = 16                        # 4 WASD bits (HANDOFF Step 4)
MASK_ID = NUM_VISUAL + NUM_KEYS      # MaskGIT [MASK]
VOCAB = MASK_ID + 1
SKEL_CHANNELS = 16
STATE_DIM = 4


def rope_tables(t, n_rot, base=10000.0):
    """cos/sin for scalar frame index t (shape (1,)) over n_rot rotary dims."""
    inv = 1.0 / (base ** (torch.arange(0, n_rot, 2, dtype=torch.float32) / n_rot))
    ang = t.float().reshape(1, 1) * inv.reshape(1, -1)   # (1, n_rot/2), fp32 even in fp16 models
    return torch.cos(ang), torch.sin(ang)


def apply_rope(x, cos, sin, n_rot):
    """x (B,H,T,hd): rotate the first n_rot dims by the frame angle (shared by all T)."""
    xr, xp = x[..., :n_rot], x[..., n_rot:]
    x1, x2 = xr[..., 0::2], xr[..., 1::2]
    r1 = x1 * cos - x2 * sin
    r2 = x1 * sin + x2 * cos
    xr = torch.stack([r1, r2], dim=-1).flatten(-2)
    return torch.cat([xr, xp], dim=-1)


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.hd = h, d // h
        self.n_rot = self.hd // 2                        # temporal RoPE on half the head
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.norm_x = nn.LayerNorm(d)
        self.xq = nn.Linear(d, d)
        self.xkv = nn.Linear(d, 2 * d)
        self.xproj = nn.Linear(d, d)
        self.gate = nn.Parameter(torch.tensor(0.5))
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def heads(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.h, self.hd).transpose(1, 2)

    def forward(self, x, past_k, past_v, cos, sin, memory, mask=None):
        B, T, D = x.shape
        q, k, v = self.qkv(self.norm1(x)).split(D, dim=-1)
        q, k, v = self.heads(q), self.heads(k), self.heads(v)
        q = apply_rope(q, cos, sin, self.n_rot)
        k = apply_rope(k, cos, sin, self.n_rot)
        kk = torch.cat([past_k, k], dim=2)               # (B,H,ctx*S+S,hd)
        vv = torch.cat([past_v, v], dim=2)
        att = (q @ kk.transpose(-2, -1)) * (1.0 / math.sqrt(self.hd))
        if mask is not None:
            att = att + mask
        out = att.softmax(-1) @ vv
        x = x + self.proj(out.transpose(1, 2).reshape(B, T, D))
        # skeleton memory cross-attention
        mq = self.heads(self.xq(self.norm_x(x)))
        mk, mv = self.xkv(memory).split(D, dim=-1)
        mk, mv = self.heads(mk), self.heads(mv)
        m = ((mq @ mk.transpose(-2, -1)) * (1.0 / math.sqrt(self.hd))).softmax(-1) @ mv
        x = x + self.gate * self.xproj(m.transpose(1, 2).reshape(B, T, D))
        x = x + self.mlp(self.norm2(x))
        return x, k, v


class FrameDynamics(nn.Module):
    def __init__(self, d=384, layers=8, heads=None, grid=16, mem_tokens=16, head="full", fused=False):
        super().__init__()
        # fused: the input is [previous frame's final tokens, next frame's masked
        # tokens] (2S). The previous frame's K/V -- computed from its final
        # tokens, so exact -- is rolled into the cache in the same run that
        # predicts the next frame, so P=1 costs one run per frame, not two.
        self.fused = fused
        self.d, self.L = d, layers
        self.H = heads or d // 64
        self.hd = d // self.H
        self.N = grid * grid
        self.S = self.N + 1
        self.grid = grid
        self.head_kind = head
        self.mem_side = int(round(mem_tokens ** 0.5))
        self.embed = nn.Embedding(VOCAB, d)
        self.pos = nn.Parameter(torch.randn(1, self.S, d) * 0.02)
        if fused:
            S = self.S       # prev-frame queries must not see the next frame's keys
            m = torch.zeros(2 * S, 2 * S)
            m[:S, S:] = float("-inf")
            self.register_buffer("fuse_mask", m, persistent=False)
        self.mem_encoder = nn.Conv2d(SKEL_CHANNELS, d, 1)
        self.blocks = nn.ModuleList([Block(d, self.H) for _ in range(layers)])
        self.norm_f = nn.LayerNorm(d)
        self.state_head = nn.Sequential(nn.Linear(d, d // 2), nn.SiLU(), nn.Linear(d // 2, STATE_DIM))
        if head == "full":
            self.out = nn.Linear(d, NUM_VISUAL, bias=False)
        else:
            self.out = nn.Linear(d, sum(LEVELS), bias=False)
            basis = [1]
            for l in LEVELS[:-1]:
                basis.append(basis[-1] * l)
            self.register_buffer("basis", torch.tensor(basis, dtype=torch.int64), persistent=False)

    def forward(self, tokens, t, skeleton, past_k, past_v):
        """tokens (1,S) int64; t (1,) fp32 frame index (fp32 in fp16 models too); skeleton (1,16,g,g);
        past_k/v (L,H,ctx*S,hd). Returns ids (1,N) int64, conf (1,N), state (1,4),
        present_k/v (L,H,ctx*S,hd)."""
        S = self.S
        x = self.embed(tokens) + (torch.cat([self.pos, self.pos], 1) if self.fused else self.pos)
        mem = F.adaptive_avg_pool2d(self.mem_encoder(skeleton), (self.mem_side, self.mem_side))
        mem = mem.flatten(2).transpose(1, 2)
        n_rot = self.blocks[0].n_rot
        mask = None
        if self.fused:
            cp, sp = rope_tables(t - 1, n_rot)
            cn, sn = rope_tables(t, n_rot)
            cos = torch.cat([cp.expand(S, -1), cn.expand(S, -1)], 0)
            sin = torch.cat([sp.expand(S, -1), sn.expand(S, -1)], 0)
            ctxlen = past_k.shape[2]
            mask = torch.cat([torch.zeros(2 * S, ctxlen), self.fuse_mask], 1).to(x.dtype)
        else:
            cos, sin = rope_tables(t, n_rot)
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        ks, vs = [], []
        for i, blk in enumerate(self.blocks):
            x, k, v = blk(x, past_k[i:i + 1], past_v[i:i + 1], cos, sin, mem, mask)
            ks.append(k)
            vs.append(v)
        x = self.norm_f(x)
        if self.fused:
            x = x[:, S:]                                 # predict only the next frame
            tokens = tokens[:, S:]
        state = self.state_head(x[:, 1:].mean(1))
        h = x[:, 1:]                                     # visual positions only
        if self.head_kind == "full":
            logits = self.out(h)                         # (1,N,12800)
            mx, ids = logits.max(-1)
            conf = torch.exp(mx - torch.logsumexp(logits, -1))
        else:
            logits = self.out(h)                         # (1,N,34)
            parts = torch.split(logits, LEVELS, dim=-1)
            ids = torch.zeros_like(tokens[:, 1:])
            conf = torch.ones_like(h[..., 0])
            for c, p in enumerate(parts):
                mx, ic = p.max(-1)
                ids = ids + ic * self.basis[c]
                conf = conf * torch.exp(mx - torch.logsumexp(p, -1))
        new_k = torch.cat(ks, 0)[:, :, :S]               # (L,H,S,hd); fused: the prev frame's
        new_v = torch.cat(vs, 0)[:, :, :S]
        present_k = torch.cat([past_k[:, :, self.S:], new_k], dim=2)
        present_v = torch.cat([past_v[:, :, self.S:], new_v], dim=2)
        return ids, conf, state, present_k, present_v


# ---- FSQ decoder (mirrors model/tokenizer/fsq_v2.DecoderV2, hidden 128) ----

def _groups(c):
    return min(8, c) if c % 8 == 0 else 1


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(_groups(c), c), nn.Conv2d(c, c, 3, padding=1)
        self.n2, self.c2 = nn.GroupNorm(_groups(c), c), nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.c2(F.silu(self.n2(self.c1(F.silu(self.n1(x))))))


class AttnBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm = nn.GroupNorm(_groups(c), c)
        self.qkv = nn.Conv2d(c, 3 * c, 1)
        self.proj = nn.Conv2d(c, c, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, c, h * w).unbind(1)
        att = (q.transpose(1, 2) @ k) * (1.0 / math.sqrt(c))
        out = v @ att.softmax(-1).transpose(1, 2)
        return x + self.proj(out.reshape(b, c, h, w))


class FSQDecoder(nn.Module):
    """ids (1,N) int64 -> RGB (1,3,F,F) in [0,1]."""

    def __init__(self, hidden=128, grid=16, frame=64):
        super().__init__()
        self.grid = grid
        n = int(round(math.log2(frame // grid)))
        # id -> normalized FSQ code via a constant (12800, 5) table: a Gather runs
        # on the GPU, whereas int64 div/mod lands on the CPU EP and blocks graph capture.
        idx = torch.arange(NUM_VISUAL)
        cols, base = [], 1
        for l in LEVELS:
            cols.append(((idx // base) % l).float() / (l // 2) - 1.0)
            base *= l
        self.register_buffer("code_table", torch.stack(cols, 1), persistent=False)
        width = hidden * 2 if n > 1 else hidden
        self.proj = nn.Conv2d(len(LEVELS), width, 1)
        blocks = [ResBlock(width), AttnBlock(width), ResBlock(width)]
        cin = width
        for i in range(n):
            cout = hidden * 2 if i < n - 1 else hidden
            blocks += [nn.Conv2d(cin, cout * 4, 3, padding=1), nn.PixelShuffle(2), ResBlock(cout)]
            cin = cout
        blocks += [nn.GroupNorm(_groups(cin), cin), nn.SiLU(), nn.Conv2d(cin, 3, 3, padding=1)]
        self.net = nn.Sequential(*blocks)

    def forward(self, ids):
        z = F.embedding(ids, self.code_table.to(self.proj.weight.dtype))   # (1,N,5) in ~[-1,1]
        z = z.transpose(1, 2).reshape(1, len(LEVELS), self.grid, self.grid)
        return torch.sigmoid(self.net(self.proj(z)))


def n_params(m):
    return sum(p.numel() for p in m.parameters())
