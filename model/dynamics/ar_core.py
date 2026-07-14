"""Branch A: autoregressive transformer dynamics core (§4).

Interleaves action + visual tokens into one sequence and predicts the next
token. One shared embedding table covers visual codes [0, NUM_VISUAL_TOKENS)
and the 9 action tokens offset above them (§3). Exposes a residual-stream hook
at every block so a steering direction can be injected (h <- h + alpha*v), and
a KV-cached decode that generates the TOKENS_PER_FRAME (256) visual tokens of
the next frame.
"""

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
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

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

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal
        )
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
        self.drop = nn.Dropout(dropout)  # residual dropout (no params; eval() disables)

    def forward(self, x, cos, sin, kv_cache=None):
        x = x + self.drop(self.attn(self.norm1(x), cos, sin, kv_cache=kv_cache))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


class ARDynamics(nn.Module):
    """Small causal transformer over the interleaved action/visual vocab."""

    def __init__(self, d_model=256, n_heads=4, n_layers=4, d_ff=None,
                 max_seq_len=32768, vocab_size=VOCAB_SIZE, dropout=0.0,
                 action_cond=False):
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
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout=dropout) for _ in range(n_layers)]
        )
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        cos, sin = build_rope_cache(max_seq_len, d_model // n_heads)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, tokens, return_hidden=False, steer=None, kv_caches=None,
                cond_ids=None):
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
        hiddens = [x] if return_hidden else None
        for i, block in enumerate(self.blocks):
            cache = None if kv_caches is None else kv_caches[i]
            x = block(x, self.rope_cos, self.rope_sin, kv_cache=cache)
            if steer is not None and i in steer:
                x = x + steer[i]
            if return_hidden:
                hiddens.append(x)
        x = self.norm_f(x)
        logits = self.head(x)
        if return_hidden:
            return logits, hiddens
        return logits

    def empty_kv_caches(self):
        return [
            {"k": None, "v": None, "len": 0} for _ in range(len(self.blocks))
        ]

    def training_step(self, tokens):
        """Next-token cross-entropy over the flattened sequence (§4).

        tokens: (B, T). Predicts token[t+1] from tokens[:t+1]; loss is the mean
        CE over all shifted positions.
        """
        logits = self.forward(tokens)
        logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
        targets = tokens[:, 1:].reshape(-1)
        return F.cross_entropy(logits, targets)

    @torch.no_grad()
    def generate_frame(self, context_tokens, action_id, sample=False,
                       temperature=1.0, steer=None, n_tokens=TOKENS_PER_FRAME):
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

        # Prefill: run the whole prefix once, populating the cache. The final
        # position's logits predict the first visual token of the new frame.
        logits = self.forward(prefill, steer=steer, kv_caches=caches, cond_ids=cond)
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
            logits = self.forward(tok, steer=steer, kv_caches=caches, cond_ids=step_cond)
            next_logits = logits[:, -1, :]

        return torch.cat(out, dim=1)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    # --- Dynamics protocol (model/interfaces.py) ---
    @torch.no_grad()
    def prepare_batch(self, tokenizer, item, horizon, device,
                      ce_weight=1.0, pixel_weight=1.0, teacher_forcing=0.0):
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
            z_ctx = build_context(ctx_actions, ctx_tokens)
            gt_frames = None
            pixel_weight = 0.0
        else:
            def encode(frames):
                b, n = frames.shape[:2]
                _, idx, _ = tokenizer(frames.reshape(b * n, *frames.shape[2:]))
                return idx.reshape(b, n, TOKENS_PER_FRAME)

            ctx_frames = item["context_frames"].float().to(device)
            gt_frames = item["target_frames"].float().to(device)
            z_ctx = build_context(ctx_actions, encode(ctx_frames))
            tgt_tokens = encode(gt_frames)

        return {
            "z_ctx": z_ctx,
            "cond_ctx": frame_cond_ids(ctx_actions),   # per-position action ids for the context
            "action_ids": tgt_actions,
            "target_tokens": tgt_tokens,
            "gt_frames": gt_frames,
            "horizon": horizon,
            "ce_weight": ce_weight,
            "pixel_weight": pixel_weight,
            "teacher_forcing": teacher_forcing,
        }

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
        )


@register_dynamics("ar_transformer")
def _build_ar(d_model=256, n_heads=4, n_layers=4, dropout=0.0, action_cond=False):
    return ARDynamics(d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                      dropout=dropout, action_cond=action_cond)


if __name__ == "__main__":
    model = ARDynamics()
    n = model.param_count()
    print(f"ARDynamics params: {n:,} ({n / 1e6:.2f}M)")
    print(f"vocab: {VOCAB_SIZE} = {NUM_VISUAL_TOKENS} visual + {NUM_ACTION_TOKENS} action")
    print(f"tokens/frame: {TOKENS_PER_FRAME}, levels={LEVELS}, C={C}, G={G}")
