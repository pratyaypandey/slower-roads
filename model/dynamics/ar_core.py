"""Branch A: autoregressive transformer dynamics core (§4).

Interleaves action + visual tokens into one sequence and predicts the next
token. One shared embedding table covers visual codes [0, NUM_VISUAL_TOKENS)
and the 9 action tokens offset above them (§3). Exposes a residual-stream hook
at every block so a steering direction can be injected (h <- h + alpha*v), and
a KV-cached decode that generates the TOKENS_PER_FRAME (256) visual tokens of
the next frame.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.dynamics.config import (
    C,
    FRAME_STRIDE,
    G,
    LEVELS,
    NUM_ACTION_TOKENS,
    NUM_VISUAL_TOKENS,
    STATE_DIM,
    TOKENS_PER_FRAME,
    VOCAB_SIZE,
    action_to_token_id,
    tokenize_action,
)
from model.registry import register_dynamics


def build_rope_cache(seq_len, head_dim, base=10000.0, device=None, dtype=torch.float32):
    """Precompute RoPE cos/sin tables of shape (seq_len, head_dim)."""
    assert head_dim % 2 == 0, "RoPE needs an even head dim"
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=dtype) / half))
    pos = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.outer(pos, inv_freq)              # (seq_len, half)
    emb = torch.cat([freqs, freqs], dim=-1)         # (seq_len, head_dim)
    return emb.cos(), emb.sin()


def apply_rope(x, cos, sin):
    """Rotary embedding. x: (B, n_heads, T, head_dim); cos/sin: (T, head_dim)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + rotated * sin


class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, recency_bias=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        # Optional ALiBi-style penalty measured in FRAME age rather than token
        # distance.  Tokens within the current 16x16 raster are not penalized;
        # older frames receive a small learned per-head penalty.  This directly
        # targets temporal attention dilution without imposing a spatial scan
        # direction inside a frame.
        if recency_bias > 0:
            init = max(float(recency_bias), 1e-6)
            raw = math.log(math.expm1(init))
            self.recency_raw = nn.Parameter(torch.full((n_heads,), raw))
        else:
            self.register_parameter("recency_raw", None)
        self.capture_attention = False
        self.last_attention = None

    def recency_slopes(self):
        return None if self.recency_raw is None else F.softplus(self.recency_raw)

    def _mask(self, q_len, k_len, offset, device, dtype, causal):
        """Additive frame-recency and causal mask broadcastable over B,H,Q,K."""
        mask = None
        slopes = self.recency_slopes()
        if slopes is not None:
            q_pos = torch.arange(offset, offset + q_len, device=device)
            k_pos = torch.arange(k_len, device=device)
            frame_age = (
                torch.div(q_pos[:, None], FRAME_STRIDE, rounding_mode="floor")
                - torch.div(k_pos[None, :], FRAME_STRIDE, rounding_mode="floor")
            ).clamp_min(0)
            mask = (
                -slopes.to(dtype)[None, :, None, None]
                * frame_age.to(dtype)[None, None, :, :]
            )
        if causal:
            q_pos = torch.arange(offset, offset + q_len, device=device)
            k_pos = torch.arange(k_len, device=device)
            future = k_pos[None, :] > q_pos[:, None]
            causal_mask = torch.zeros(q_len, k_len, device=device, dtype=dtype)
            causal_mask.masked_fill_(future, float("-inf"))
            causal_mask = causal_mask[None, None]
            mask = causal_mask if mask is None else mask + causal_mask
        return mask

    def forward(self, x, cos, sin, kv_cache=None):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).split(x.shape[-1], dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # RoPE uses absolute positions; during cached decode the new tokens sit
        # at offset = number of positions already cached.
        offset = 0 if kv_cache is None else kv_cache["len"]
        if offset + T > cos.shape[0]:
            raise ValueError(
                f"sequence position {offset + T} exceeds the RoPE cache "
                f"({cos.shape[0]}). Raise ARDynamics(max_seq_len=...) — a dream of "
                f"H frames from T context needs (T+H)*{FRAME_STRIDE} positions."
            )
        cos_t = cos[offset:offset + T]
        sin_t = sin[offset:offset + T]
        q = apply_rope(q, cos_t, sin_t)
        k = apply_rope(k, cos_t, sin_t)

        if kv_cache is not None:
            if kv_cache["k"] is not None:
                k = torch.cat([kv_cache["k"], k], dim=2)
                v = torch.cat([kv_cache["v"], v], dim=2)
            kv_cache["k"], kv_cache["v"] = k, v
            kv_cache["len"] = k.shape[2]
            # New queries attend to the whole prefix already in the cache, so no
            # extra mask is needed for the single-step (or block) decode.
            attn_mask = None
            is_causal = k.shape[2] == T  # only the prefill chunk is self-causal
        else:
            attn_mask = None
            is_causal = True

        bias = self._mask(
            T, k.shape[2], offset, q.device, q.dtype, causal=is_causal
        )
        if bias is not None:
            attn_mask = bias
            is_causal = False
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal
        )
        if self.capture_attention:
            # Capture only the final query.  This is diagnostic-only and avoids
            # materializing a T x T attention matrix during ordinary operation.
            scores = torch.matmul(q[:, :, -1:, :], k.transpose(-2, -1)) \
                / math.sqrt(self.head_dim)
            if bias is not None:
                scores = scores + bias[:, :, -1:, :]
            self.last_attention = scores.softmax(dim=-1)[:, :, 0].detach()
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.proj(out)


class CrossAttention(nn.Module):
    """Visual tokens (query) attend to a small set of skeleton memory tokens
    (key/value). No RoPE, no causal mask: the memory is a per-frame set of
    ground-truth structural tokens that lives OUTSIDE the causal KV cache, so it
    never enters the visual self-attention Q/K similarity that drifts."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.kv = nn.Linear(d_model, 2 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x, memory):
        B, T, _ = x.shape
        M = memory.shape[1]
        q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k, v = self.kv(memory).split(memory.shape[-1], dim=-1)
        k = k.view(B, M, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, M, self.n_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)  # every query sees all M
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0, recency_bias=0.0,
                 mem_cross_attn=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads, recency_bias=recency_bias)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
        self.drop = nn.Dropout(dropout)  # residual dropout (no params; eval() disables)
        # Optional cross-attention into skeleton memory tokens. A learned scalar
        # gate initialized to 0 makes a warm-started checkpoint start numerically
        # identical, then learn to pull structure from the clean memory.
        if mem_cross_attn:
            self.norm_cross = nn.LayerNorm(d_model)
            self.cross_attn = CrossAttention(d_model, n_heads)
            self.cross_gate = nn.Parameter(torch.zeros(()))
        else:
            self.norm_cross = None
            self.cross_attn = None
            self.register_parameter("cross_gate", None)

    def forward(self, x, cos, sin, kv_cache=None, memory=None, mem_mask=None):
        x = x + self.drop(self.attn(self.norm1(x), cos, sin, kv_cache=kv_cache))
        if self.cross_attn is not None and memory is not None:
            cross = self.cross_attn(self.norm_cross(x), memory)
            if mem_mask is not None:
                # Only current-frame positions receive memory; context positions
                # stay untouched so their cached K/V are not contaminated.
                cross = cross * mem_mask.unsqueeze(-1)
            x = x + self.cross_gate * cross
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


class ARDynamics(nn.Module):
    """Small causal transformer over the interleaved action/visual vocab."""

    def __init__(self, d_model=256, n_heads=4, n_layers=4, d_ff=None,
                 max_seq_len=32768, vocab_size=VOCAB_SIZE, dropout=0.0,
                 action_cond=False, anchor_cond=False, anchor_channels=16,
                 corruption_cond=False, attention_recency_bias=0.0,
                 anchor_injection="input", mem_cross_attn=False, mem_tokens=16,
                 state_head=False):
        super().__init__()
        from model.dynamics.config import NUM_ACTION_TOKENS
        d_ff = d_ff or 4 * d_model
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.embed = nn.Embedding(vocab_size, d_model)
        self.embed_drop = nn.Dropout(dropout)
        # Strong action conditioning: a separate embedding added to EVERY position
        # of the frame an action drives (not just the lone action token, which the
        # core under-weights → weak steering). Backward-compatible: off by default,
        # so old checkpoints (no this table) still load.
        self.action_cond = nn.Embedding(NUM_ACTION_TOKENS, d_model) if action_cond else None
        self.anchor_cond = anchor_cond
        if anchor_injection not in {"input", "output"}:
            raise ValueError("anchor_injection must be 'input' or 'output'")
        self.anchor_injection = anchor_injection
        self.anchor_encoder = (
            nn.Conv2d(anchor_channels, d_model, kernel_size=1) if anchor_cond else None
        )
        self.anchor_output_gate = (
            nn.Parameter(torch.tensor(0.0))
            if anchor_cond and anchor_injection == "output" else None
        )
        self.corruption_cond = corruption_cond
        self.corruption_embed = (
            nn.Sequential(nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
            if corruption_cond else None
        )
        # Skeleton memory tokens: a dedicated encoder (independent of the additive
        # anchor path) rasterizes the current frame's skeleton and pools it to a
        # small register set that visual tokens cross-attend to. mem_side^2 tokens.
        self.mem_cross_attn = mem_cross_attn
        self.mem_side = max(1, round(mem_tokens ** 0.5))
        self.mem_tokens = self.mem_side ** 2
        self.mem_encoder = (
            nn.Conv2d(anchor_channels, d_model, kernel_size=1) if mem_cross_attn else None
        )
        # State-continuity head: regress the per-frame sim-state delta from the
        # pooled current-frame hidden, so the cache carries an explicit evolving
        # world state rather than only next-token statistics.
        self.state_head = (
            nn.Sequential(nn.Linear(d_model, d_model // 2), nn.SiLU(),
                          nn.Linear(d_model // 2, STATE_DIM))
            if state_head else None
        )
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout=dropout,
                   recency_bias=attention_recency_bias,
                   mem_cross_attn=mem_cross_attn) for _ in range(n_layers)]
        )
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        cos, sin = build_rope_cache(max_seq_len, d_model // n_heads)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, tokens, return_hidden=False, steer=None, kv_caches=None,
                cond_ids=None, anchor_emb=None, noise_levels=None,
                memory=None, mem_mask=None, return_state=False):
        """tokens: (B, T) int64 in [0, VOCAB_SIZE).

        return_hidden=True also returns the per-block residual streams (a list
        of (B, T, d_model), one entry after each block plus the input) — the
        seam for steering / probing.
        steer: optional dict {layer_index: (d_model,) or (B,T,d_model)} added to
               the residual stream right after that block (h <- h + v).
        kv_caches: optional list (len n_layers) of per-layer cache dicts for
                   incremental decode.
        """
        x = self.embed_drop(self.embed(tokens))
        if self.action_cond is not None and cond_ids is not None:
            x = x + self.action_cond(cond_ids)          # per-position action conditioning
        if self.anchor_encoder is not None:
            if anchor_emb is None:
                raise ValueError("anchor-conditioned dynamics requires anchor_emb")
            if anchor_emb.shape[:2] != tokens.shape:
                raise ValueError(
                    f"anchor/token shape mismatch: {tuple(anchor_emb.shape[:2])} vs "
                    f"{tuple(tokens.shape)}"
                )
            if self.anchor_injection == "input":
                x = x + anchor_emb
        if self.corruption_embed is not None:
            if noise_levels is None:
                noise_levels = torch.zeros(tokens.shape, device=tokens.device, dtype=x.dtype)
            x = x + self.corruption_embed(noise_levels.to(x.dtype).unsqueeze(-1))
        hiddens = [x] if return_hidden else None
        for i, block in enumerate(self.blocks):
            cache = None if kv_caches is None else kv_caches[i]
            x = block(x, self.rope_cos, self.rope_sin, kv_cache=cache,
                      memory=memory, mem_mask=mem_mask)
            if steer is not None and i in steer:
                x = x + steer[i]
            if return_hidden:
                hiddens.append(x)
        if self.anchor_encoder is not None and self.anchor_injection == "output":
            # Keep geometry out of Q/K similarity.  A bounded learned gate starts
            # at scale 1 and lets fine-tuning attenuate or strengthen the local
            # structural condition without changing temporal attention directly.
            x = x + (2.0 * self.anchor_output_gate.sigmoid()) * anchor_emb
        x = self.norm_f(x)
        logits = self.head(x)
        if return_state:
            # Pool the current-frame positions (mem_mask marks them) and regress
            # the state delta. Falls back to the last position if no mask given.
            if self.state_head is None:
                state_pred = None
            elif mem_mask is not None:
                w = mem_mask.to(x.dtype)
                pooled = (x * w.unsqueeze(-1)).sum(1) / w.sum(1, keepdim=True).clamp_min(1.0)
                state_pred = self.state_head(pooled)
            else:
                state_pred = self.state_head(x[:, -1])
            if return_hidden:
                return logits, hiddens, state_pred
            return logits, state_pred
        if return_hidden:
            return logits, hiddens
        return logits

    def empty_kv_caches(self):
        return [
            {"k": None, "v": None, "len": 0} for _ in range(len(self.blocks))
        ]

    def attention_profile(self):
        """Summarize the most recently captured final-query attention by frame age."""
        profiles = []
        for layer, block in enumerate(self.blocks):
            weights = block.attn.last_attention
            if weights is None:
                continue
            weights = weights.float()
            k_len = weights.shape[-1]
            q_frame = (k_len - 1) // FRAME_STRIDE
            ages = q_frame - torch.div(
                torch.arange(k_len, device=weights.device),
                FRAME_STRIDE, rounding_mode="floor",
            )
            by_age = {}
            for age in range(int(ages.max().item()) + 1):
                mass = weights[..., ages == age].sum(dim=-1).mean().item()
                by_age[str(age)] = mass
            entropy = -(weights.clamp_min(1e-12) * weights.clamp_min(1e-12).log()) \
                .sum(dim=-1) / max(math.log(k_len), 1e-12)
            slopes = block.attn.recency_slopes()
            profiles.append({
                "layer": layer,
                "normalized_entropy": entropy.mean().item(),
                "first_token_mass": weights[..., 0].mean().item(),
                "frame_mass": by_age,
                "recency_slopes": None if slopes is None else slopes.detach().cpu().tolist(),
            })
        return profiles

    def training_step(self, tokens):
        """Next-token cross-entropy over the flattened sequence (§4).

        tokens: (B, T). Predicts token[t+1] from tokens[:t+1]; loss is the mean
        CE over all shifted positions.
        """
        logits = self.forward(tokens)
        logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
        targets = tokens[:, 1:].reshape(-1)
        return F.cross_entropy(logits, targets)

    def encode_anchor(self, anchor, anchor_lambda=1.0):
        """(B,T,C,G,G) skeleton rasters -> (B,T,TOKENS_PER_FRAME,d)."""
        if self.anchor_encoder is None:
            return None
        b, t, c, g1, g2 = anchor.shape
        h = self.anchor_encoder(anchor.reshape(b * t, c, g1, g2))
        h = h.flatten(2).transpose(1, 2).reshape(b, t, g1 * g2, self.d_model)
        lam = torch.as_tensor(anchor_lambda, device=h.device, dtype=h.dtype)
        if lam.ndim == 0:
            lam = lam.expand(b)
        return h * lam.reshape(b, 1, 1, 1)

    def anchor_frames(self, anchor, anchor_lambda=1.0):
        """Add one pooled action slot before each token-aligned anchor grid."""
        vis = self.encode_anchor(anchor, anchor_lambda)
        action = vis.mean(dim=2, keepdim=True)
        return torch.cat([action, vis], dim=2)  # (B,T,FRAME_STRIDE,d)

    def anchor_sequence(self, anchor, anchor_lambda=1.0):
        frames = self.anchor_frames(anchor, anchor_lambda)
        return frames.reshape(frames.shape[0], -1, frames.shape[-1])

    def encode_memory(self, anchor, anchor_lambda=1.0):
        """(B,C,G,G) skeleton raster for ONE frame -> (B, mem_tokens, d).

        λ scales the whole memory: at λ=0 the value stream is zero so the
        cross-attention contributes nothing and the model is pure autoregressive.
        """
        if self.mem_encoder is None:
            return None
        b, c, g1, g2 = anchor.shape
        h = self.mem_encoder(anchor)                       # (B, d, G, G)
        h = F.adaptive_avg_pool2d(h, (self.mem_side, self.mem_side))
        h = h.flatten(2).transpose(1, 2)                   # (B, mem_tokens, d)
        lam = torch.as_tensor(anchor_lambda, device=h.device, dtype=h.dtype)
        if lam.ndim == 0:
            lam = lam.expand(b)
        return h * lam.reshape(b, 1, 1)

    @staticmethod
    def corrupt_fsq_tokens(tokens, severity):
        """Perturb one mixed-radix FSQ factor, preserving local code geometry."""
        levels = list(LEVELS)
        rem = tokens.clone()
        codes = []
        for level in levels:
            codes.append(rem % level)
            rem = torch.div(rem, level, rounding_mode="floor")
        codes = torch.stack(codes, dim=-1)
        mask = torch.rand(tokens.shape, device=tokens.device) < severity
        channel = torch.randint(0, len(levels), tokens.shape, device=tokens.device)
        delta = torch.where(
            torch.rand(tokens.shape, device=tokens.device) < 0.5,
            -torch.ones_like(tokens), torch.ones_like(tokens),
        )
        for c, level in enumerate(levels):
            take = mask & (channel == c)
            changed = (codes[..., c] + delta).clamp(0, level - 1)
            codes[..., c] = torch.where(take, changed, codes[..., c])
        out = torch.zeros_like(tokens)
        radix = 1
        for c, level in enumerate(levels):
            out += codes[..., c] * radix
            radix *= level
        return out

    @torch.no_grad()
    def generate_frame(self, context_tokens, action_id, sample=False,
                       temperature=1.0, steer=None, n_tokens=TOKENS_PER_FRAME,
                       context_anchor_emb=None, anchor=None, anchor_lambda=1.0,
                       context_noise=None, capture_final_attention=False):
        """KV-cached decode of the next frame's `n_tokens` visual tokens.

        context_tokens: (B, T_ctx) prior interleaved tokens (may be empty T=0).
        action_id: (B,) int64 action id in [0, NUM_ACTION_TOKENS) for frame t.
        n_tokens: visual tokens per frame (defaults to the config grid; overridable
                  so behavioural tests can run at a smaller, CPU-fast grid).
        Returns (B, n_tokens) predicted visual token indices in [0, NUM_VISUAL_TOKENS).
        """
        device = self.embed.weight.device
        B = action_id.shape[0]
        caches = self.empty_kv_caches()

        u_t = (action_id + NUM_VISUAL_TOKENS).view(B, 1)  # offset into shared vocab
        act_col = action_id.view(B, 1)                    # this frame's action-cond id
        if context_tokens is not None and context_tokens.shape[1] > 0:
            prefill = torch.cat([context_tokens, u_t], dim=1)
        else:
            prefill = u_t

        # Action-conditioning ids for the prefill. The context is frame-aligned
        # ([u,z,...] blocks of FRAME_STRIDE), so each frame's action is its block's
        # first token; u_t (new frame) conditions on action_id.
        cond = None
        if self.action_cond is not None:
            if context_tokens is not None and context_tokens.shape[1] > 0:
                T = context_tokens.shape[1] // FRAME_STRIDE
                ctx_actions = context_tokens.view(B, T, FRAME_STRIDE)[:, :, 0] - NUM_VISUAL_TOKENS
                cond = torch.cat([ctx_actions.repeat_interleave(FRAME_STRIDE, dim=1), act_col], dim=1)
            else:
                cond = act_col

        prefill_anchor = None
        visual_anchor = None
        if self.anchor_encoder is not None:
            if anchor is None:
                raise ValueError("anchor-conditioned generation requires current anchor")
            current = self.anchor_frames(anchor[:, None], anchor_lambda)[:, 0]
            visual_anchor = current[:, 1:]
            if context_tokens is not None and context_tokens.shape[1] > 0:
                if context_anchor_emb is None:
                    raise ValueError("anchor-conditioned generation requires context_anchor_emb")
                prefill_anchor = torch.cat([context_anchor_emb, current[:, :1]], dim=1)
            else:
                prefill_anchor = current[:, :1]

        prefill_noise = None
        if self.corruption_embed is not None:
            if context_tokens is not None and context_tokens.shape[1] > 0:
                if context_noise is None:
                    context_noise = torch.zeros(
                        context_tokens.shape, device=device, dtype=torch.float32
                    )
                prefill_noise = torch.cat(
                    [context_noise, torch.zeros(B, 1, device=device)], dim=1
                )
            else:
                prefill_noise = torch.zeros(B, 1, device=device)

        # Skeleton memory tokens for the current frame. Only current-frame
        # positions attend to them; in the prefill that is the final position (u_t).
        memory = None
        prefill_mem_mask = None
        if self.mem_encoder is not None:
            if anchor is None:
                raise ValueError("memory-conditioned generation requires current anchor")
            memory = self.encode_memory(anchor, anchor_lambda)
            prefill_mem_mask = torch.zeros(prefill.shape, device=device)
            prefill_mem_mask[:, -1] = 1.0

        # Prefill: run the whole prefix once, populating the cache. The final
        # position's logits predict the first visual token of the new frame.
        logits = self.forward(
            prefill, steer=steer, kv_caches=caches, cond_ids=cond,
            anchor_emb=prefill_anchor, noise_levels=prefill_noise,
            memory=memory, mem_mask=prefill_mem_mask,
        )
        next_logits = logits[:, -1, :]

        out = []
        for _ in range(n_tokens):
            next_logits = next_logits[:, :NUM_VISUAL_TOKENS]  # visual tokens only
            if sample:
                probs = F.softmax(next_logits / temperature, dim=-1)
                tok = torch.multinomial(probs, num_samples=1)  # (B,1)
            else:
                tok = next_logits.argmax(dim=-1, keepdim=True)
            out.append(tok)
            # Feed the just-generated token back; it belongs to this frame, so it
            # conditions on the same action.
            step_cond = act_col if self.action_cond is not None else None
            step_anchor = None if visual_anchor is None else visual_anchor[:, len(out) - 1:len(out)]
            final_capture = capture_final_attention and len(out) == n_tokens
            if final_capture:
                for block in self.blocks:
                    block.attn.capture_attention = True
            logits = self.forward(
                tok, steer=steer, kv_caches=caches, cond_ids=step_cond,
                anchor_emb=step_anchor,
                noise_levels=(None if self.corruption_embed is None
                              else torch.zeros(B, 1, device=device)),
                memory=memory,
                mem_mask=(None if memory is None else torch.ones(B, 1, device=device)),
            )
            if final_capture:
                for block in self.blocks:
                    block.attn.capture_attention = False
            next_logits = logits[:, -1, :]

        return torch.cat(out, dim=1)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    # --- Dynamics protocol (model/interfaces.py) ---
    @torch.no_grad()
    def prepare_batch(self, tokenizer, item, horizon, device,
                      ce_weight=1.0, pixel_weight=1.0, teacher_forcing=0.0,
                      self_rollout=False, context_window=None,
                      anchor_lambda_min=0.0, anchor_lambda_max=1.0,
                      corruption_min=0.0, corruption_max=0.0, state_weight=0.0):
        """Encode a dataset item into this core's token-sequence rollout inputs."""
        from model.dynamics.sequence import build_context, frame_cond_ids

        ctx_actions = item["context_actions"].to(device)
        tgt_actions = item["target_actions"].to(device)

        if "context_tokens" in item:
            # Latent-cache path: the dataset already carries frozen-tokenizer token
            # indices, so we skip the tokenizer forward entirely (the big speedup).
            # No frames -> no decoded-pixel monitor (pixel_weight is forced to 0).
            ctx_tokens = item["context_tokens"].to(device).long()   # (B, T, tok)
            tgt_tokens = item["target_tokens"].to(device).long()    # (B, H, tok)
            gt_frames = None
            pixel_weight = 0.0
        else:
            def encode(frames):
                b, n = frames.shape[:2]
                _, idx, _ = tokenizer(frames.reshape(b * n, *frames.shape[2:]))
                return idx.reshape(b, n, TOKENS_PER_FRAME)

            ctx_frames = item["context_frames"].float().to(device)
            gt_frames = item["target_frames"].float().to(device)
            ctx_tokens = encode(ctx_frames)
            tgt_tokens = encode(gt_frames)

        context_noise = None
        if self.corruption_embed is not None:
            b, t = ctx_tokens.shape[:2]
            frame_noise = torch.empty(b, t, 1, device=device).uniform_(
                float(corruption_min), float(corruption_max)
            )
            ctx_tokens = self.corrupt_fsq_tokens(ctx_tokens, frame_noise)
            context_noise = frame_noise.repeat_interleave(FRAME_STRIDE, dim=1).squeeze(-1)
        z_ctx = build_context(ctx_actions, ctx_tokens)

        batch = {
            "z_ctx": z_ctx,
            "cond_ctx": frame_cond_ids(ctx_actions),   # per-position action ids for the context
            "action_ids": tgt_actions,
            "target_tokens": tgt_tokens,
            "gt_frames": gt_frames,
            "horizon": horizon,
            "ce_weight": ce_weight,
            "pixel_weight": pixel_weight,
            "teacher_forcing": teacher_forcing,
            "self_rollout": self_rollout,
            "context_window": context_window or item["context_actions"].shape[1],
            "context_noise": context_noise,
        }
        # Both the additive anchor and the memory-token cross-attention consume the
        # per-frame skeleton raster (and its λ gate), so load anchors when either is on.
        if self.anchor_encoder is not None or self.mem_encoder is not None:
            if "context_anchor" not in item or "target_anchor" not in item:
                raise ValueError(
                    "--anchor-cond / --mem-cross-attn require dataset skeleton anchors"
                )
            batch["context_anchor"] = item["context_anchor"].float().to(device)
            batch["target_anchor"] = item["target_anchor"].float().to(device)
            b = batch["context_anchor"].shape[0]
            lo, hi = float(anchor_lambda_min), float(anchor_lambda_max)
            batch["anchor_lambda"] = torch.empty(b, device=device).uniform_(lo, hi)
        if self.state_head is not None:
            if "context_state" not in item or "target_state" not in item:
                raise ValueError("--state-head requires dataset state vectors")
            batch["context_state"] = item["context_state"].float().to(device)
            batch["target_state"] = item["target_state"].float().to(device)
            batch["state_weight"] = state_weight
        return batch

    def loss(self, batch, decoder):
        """Multi-step rollout loss for this core. `batch` carries the encoded
        rollout inputs; `decoder` maps predicted visual tokens -> frames for the
        pixel term. Delegates to rollout_loss so the math lives in one place."""
        from model.dynamics.rollout_loss import rollout_loss
        return rollout_loss(
            self, decoder,
            batch["z_ctx"], batch["action_ids"], batch["target_tokens"],
            batch["gt_frames"], batch["horizon"],
            ce_weight=batch.get("ce_weight", 1.0),
            pixel_weight=batch.get("pixel_weight", 1.0),
            teacher_forcing=batch.get("teacher_forcing", 0.0),
            cond_ctx=batch.get("cond_ctx"),
            self_rollout=batch.get("self_rollout", False),
            context_window=batch.get("context_window"),
            context_anchor=batch.get("context_anchor"),
            target_anchor=batch.get("target_anchor"),
            anchor_lambda=batch.get("anchor_lambda", 0.0),
            context_noise=batch.get("context_noise"),
            context_state=batch.get("context_state"),
            target_state=batch.get("target_state"),
            state_weight=batch.get("state_weight", 0.0),
        )


@register_dynamics("ar_transformer")
def _build_ar(d_model=256, n_heads=4, n_layers=4, dropout=0.0, action_cond=False,
              anchor_cond=False, anchor_channels=16, corruption_cond=False,
              attention_recency_bias=0.0, anchor_injection="input",
              mem_cross_attn=False, mem_tokens=16, state_head=False):
    return ARDynamics(d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                      dropout=dropout, action_cond=action_cond,
                      anchor_cond=anchor_cond, anchor_channels=anchor_channels,
                      corruption_cond=corruption_cond,
                      attention_recency_bias=attention_recency_bias,
                      anchor_injection=anchor_injection,
                      mem_cross_attn=mem_cross_attn, mem_tokens=mem_tokens,
                      state_head=state_head)


if __name__ == "__main__":
    model = ARDynamics()
    n = model.param_count()
    print(f"ARDynamics params: {n:,} ({n / 1e6:.2f}M)")
    print(f"vocab: {VOCAB_SIZE} = {NUM_VISUAL_TOKENS} visual + {NUM_ACTION_TOKENS} action")
    print(f"tokens/frame: {TOKENS_PER_FRAME}, levels={LEVELS}, C={C}, G={G}")
