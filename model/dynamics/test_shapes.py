"""Correctness checks for the AR dynamics core.

If torch is present, exercise real tensor shapes end-to-end. If it is not, fall
back to the pure-logic pieces (sequence interleaving, action bucketing, mixed
vocab offsets, causal-mask shape, KV-cache index bookkeeping, mixed-radix code
indices) using only the stdlib so this file runs and reports real results on a
machine with neither torch nor numpy.
"""

import os
import sys
from math import prod

# Run the same way everywhere: `python3 model/dynamics/test_shapes.py` from the
# repo root. Inject the repo root so absolute `model.` imports resolve.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from model.dynamics.config import (
    FRAME_STRIDE,
    G,
    LEVELS,
    NUM_ACTION_TOKENS,
    NUM_VISUAL_TOKENS,
    TOKENS_PER_FRAME,
    VOCAB_SIZE,
    action_to_token_id,
    causal_mask_bool,
    interleave_frame_layout,
    is_action_token,
    kv_cache_positions,
    tokenize_action,
)


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    return cond


def codes_to_index(code, levels):
    """Mixed-radix flatten of a per-channel FSQ code (§2 codes_to_indices)."""
    idx, radix = 0, 1
    for c, lvl in enumerate(levels):
        idx += code[c] * radix
        radix *= lvl
    return idx


def logic_tests():
    ok = True
    print("Pure-logic tests (no torch/numpy required):")

    # Vocab layout / mixed vocab offsets (§3).
    ok &= check("NUM_VISUAL_TOKENS == prod(LEVELS)", NUM_VISUAL_TOKENS == prod(LEVELS))
    ok &= check("VOCAB_SIZE == visual + action",
                VOCAB_SIZE == NUM_VISUAL_TOKENS + NUM_ACTION_TOKENS)
    ok &= check("action_to_token_id offsets above visual",
                action_to_token_id(0) == NUM_VISUAL_TOKENS
                and action_to_token_id(8) == NUM_VISUAL_TOKENS + 8)
    ok &= check("is_action_token boundary",
                (not is_action_token(NUM_VISUAL_TOKENS - 1))
                and is_action_token(NUM_VISUAL_TOKENS))
    ok &= check("TOKENS_PER_FRAME == G*G",
                TOKENS_PER_FRAME == G * G)

    # Action bucketing: {steer, throttle} in [-1,1] -> si*THROTTLE_BUCKETS + ti.
    seen = {tokenize_action(s, t)
            for s in (-1.0, 0.0, 1.0) for t in (-1.0, 0.0, 1.0)}
    ok &= check("action tokens within [0,9)", all(0 <= a < 9 for a in seen))
    ok &= check("all 9 (steer,throttle) buckets reachable", seen == set(range(9)))
    ok &= check("straight + coast -> center bucket 4",
                tokenize_action(0.0, 0.0) == 4)
    ok &= check("hard-right + full throttle -> si=2,ti=2 -> 8",
                tokenize_action(1.0, 1.0) == 8)
    ok &= check("hard-left + full brake -> si=0,ti=0 -> 0",
                tokenize_action(-1.0, -1.0) == 0)

    # Sequence interleaving layout (§4): [u_t, z_t[0..63]] per step, flattened.
    T = 3
    actions = [action_to_token_id(k % NUM_ACTION_TOKENS) for k in range(T)]
    visuals = [[t * 100 + i for i in range(TOKENS_PER_FRAME)] for t in range(T)]
    seq = interleave_frame_layout(actions, visuals)
    ok &= check("flattened length == T*FRAME_STRIDE",
                len(seq) == T * FRAME_STRIDE)
    ok &= check("every frame starts with its action token",
                all(seq[t * FRAME_STRIDE] == actions[t] for t in range(T)))
    ok &= check("visual tokens follow each action in order",
                seq[1:1 + TOKENS_PER_FRAME] == visuals[0])

    # Causal mask shape + lower-triangular property.
    n = 5
    mask = causal_mask_bool(n)
    ok &= check("causal mask is n x n", len(mask) == n and all(len(r) == n for r in mask))
    ok &= check("row i allows exactly i+1 keys",
                all(sum(mask[i]) == i + 1 for i in range(n)))
    ok &= check("no attention to the future",
                all(not mask[i][j] for i in range(n) for j in range(n) if j > i))

    # KV-cache index bookkeeping: prefill then one-token-at-a-time decode.
    ctx_len = FRAME_STRIDE          # a one-frame context + action prefill
    gen = TOKENS_PER_FRAME
    pos = kv_cache_positions(ctx_len, gen)
    ok &= check("cache positions contiguous from 0",
                pos == list(range(ctx_len + gen)))
    ok &= check("cache length after decode step k == ctx_len + k + 1",
                all(pos[ctx_len + k] == ctx_len + k for k in range(gen)))

    # Mixed-radix code -> index (§2), the same layout the vocab assumes.
    ok &= check("code [0,0,0,0,0] -> 0", codes_to_index([0, 0, 0, 0, 0], LEVELS) == 0)
    top = [lvl - 1 for lvl in LEVELS]
    ok &= check("max code -> prod(L)-1",
                codes_to_index(top, LEVELS) == prod(LEVELS) - 1)
    ok &= check("radix carry: [0,1,0,0,0] -> LEVELS[0]",
                codes_to_index([0, 1, 0, 0, 0], LEVELS) == LEVELS[0])

    print(f"\nLogic tests: {'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return ok


def torch_tests():
    import torch
    from model.dynamics.ar_core import ARDynamics, build_rope_cache, apply_rope
    from model.dynamics.rollout_loss import rollout_loss
    from model.dynamics.sequence import build_context

    print("\nTorch shape tests:")
    ok = True
    B, T_ctx = 2, FRAME_STRIDE
    model = ARDynamics(d_model=64, n_heads=4, n_layers=2, max_seq_len=2048)
    n = model.param_count()
    print(f"  param count: {n:,} ({n / 1e6:.2f}M)")

    tokens = torch.randint(0, VOCAB_SIZE, (B, T_ctx))
    logits = model(tokens)
    ok &= check("forward logits shape (B,T,V)",
                tuple(logits.shape) == (B, T_ctx, VOCAB_SIZE))

    logits, hiddens = model(tokens, return_hidden=True)
    ok &= check("return_hidden yields n_layers+1 residual streams",
                len(hiddens) == len(model.blocks) + 1)
    ok &= check("each hidden is (B,T,d_model)",
                all(tuple(h.shape) == (B, T_ctx, model.d_model) for h in hiddens))

    # Steering hook: injecting a direction changes the output deterministically.
    v = torch.ones(model.d_model)
    steered = model(tokens, steer={0: 5.0 * v})
    ok &= check("steering hook perturbs logits",
                not torch.allclose(steered, model(tokens)))

    loss = model.training_step(tokens)
    ok &= check("training_step returns scalar", loss.dim() == 0)

    action_id = torch.randint(0, NUM_ACTION_TOKENS, (B,))
    frame = model.generate_frame(tokens, action_id)
    ok &= check("generate_frame shape (B,TOKENS_PER_FRAME)",
                tuple(frame.shape) == (B, TOKENS_PER_FRAME))
    ok &= check("generated tokens are visual ids",
                bool((frame < NUM_VISUAL_TOKENS).all()))

    # KV-cache decode must match a full non-cached forward at the join point.
    prefill = torch.cat([tokens, (action_id + NUM_VISUAL_TOKENS).view(B, 1)], dim=1)
    ref = model(prefill)[:, -1, :NUM_VISUAL_TOKENS].argmax(-1)
    cached = model.generate_frame(tokens, action_id)[:, 0]
    ok &= check("cached first-token == non-cached argmax", bool((ref == cached).all()))

    anchored = ARDynamics(
        d_model=64, n_heads=4, n_layers=2, max_seq_len=2048,
        anchor_cond=True,
    )
    context_anchor = torch.rand(B, 1, 16, G, G)
    current_anchor = torch.rand(B, 16, G, G)
    context_anchor_emb = anchored.anchor_sequence(context_anchor, anchor_lambda=0.5)
    anchored_frame = anchored.generate_frame(
        tokens, action_id, context_anchor_emb=context_anchor_emb,
        anchor=current_anchor, anchor_lambda=0.5,
    )
    ok &= check("lambda anchor generates token-aligned frame",
                tuple(anchored_frame.shape) == (B, TOKENS_PER_FRAME))
    zero_anchor = anchored.anchor_sequence(context_anchor, anchor_lambda=0.0)
    ok &= check("lambda=0 zeros the anchor path", bool((zero_anchor == 0).all()))

    # Frame-recency attention remains causal in full and cached paths and its
    # final-token diagnostic accounts for all attention mass.
    recency = ARDynamics(
        d_model=64, n_heads=4, n_layers=2, max_seq_len=2048,
        attention_recency_bias=0.05,
    )
    rec_logits = recency(tokens)
    rec_frame = recency.generate_frame(
        tokens, action_id, n_tokens=4, capture_final_attention=True,
    )
    profile = recency.attention_profile()
    mass = sum(profile[0]["frame_mass"].values())
    ok &= check("frame-recency attention full/cached shapes",
                rec_logits.shape == logits.shape and rec_frame.shape == (B, 4))
    ok &= check("attention profile mass normalized with positive slopes",
                abs(mass - 1.0) < 1e-5
                and min(profile[0]["recency_slopes"]) > 0)

    output_anchor = ARDynamics(
        d_model=64, n_heads=4, n_layers=2, max_seq_len=2048,
        anchor_cond=True, anchor_injection="output",
    )
    out_frame = output_anchor.generate_frame(
        tokens, action_id,
        context_anchor_emb=output_anchor.anchor_sequence(context_anchor, 1.0),
        anchor=current_anchor, anchor_lambda=1.0, n_tokens=4,
    )
    ok &= check("output-only anchor generates without entering temporal Q/K",
                out_frame.shape == (B, 4)
                and output_anchor.anchor_output_gate is not None)

    # Skeleton memory tokens + state-continuity head: gate starts at zero
    # (warm-start neutral), memory pools to a square token grid, generation runs.
    mem_state = ARDynamics(
        d_model=64, n_heads=4, n_layers=2, max_seq_len=2048,
        anchor_cond=True, mem_cross_attn=True, mem_tokens=16, state_head=True,
    )
    ok &= check("mem tokens rounded to a square grid", mem_state.mem_tokens == 16)
    ok &= check("cross-attn gate initialized to zero (warm-start neutral)",
                float(mem_state.blocks[0].cross_gate) == 0.0)
    mem_frame = mem_state.generate_frame(
        tokens, action_id,
        context_anchor_emb=mem_state.anchor_sequence(context_anchor, 1.0),
        anchor=current_anchor, anchor_lambda=1.0, n_tokens=4,
    )
    ok &= check("mem+state generate_frame shape", tuple(mem_frame.shape) == (B, 4))

    clean_codes = torch.randint(0, NUM_VISUAL_TOKENS, (B, 2, TOKENS_PER_FRAME))
    unchanged = ARDynamics.corrupt_fsq_tokens(
        clean_codes, torch.zeros(B, 2, 1)
    )
    corrupted = ARDynamics.corrupt_fsq_tokens(
        clean_codes, torch.ones(B, 2, 1)
    )
    ok &= check("zero corruption preserves FSQ tokens", bool((unchanged == clean_codes).all()))
    ok &= check("FSQ-local corruption stays in visual vocabulary",
                bool((corrupted >= 0).all() and (corrupted < NUM_VISUAL_TOKENS).all()))

    # Rollout loss end-to-end with a stub decoder (callable, not imported).
    # Grid-agnostic: reshape the tok visual tokens onto a sqrt(tok) grid and
    # upsample to the 64x64 frame, so this survives grid changes (8x8 -> 16x16).
    H = 3
    def stub_decoder(vis_tokens):
        b, tok = vis_tokens.shape
        gg = int(round(tok ** 0.5))
        grid = (vis_tokens.float() / NUM_VISUAL_TOKENS).view(b, 1, gg, gg)
        return torch.nn.functional.interpolate(
            grid, size=(64, 64), mode="nearest").expand(b, 3, 64, 64).contiguous()

    z_ctx = tokens
    actions = torch.randint(0, NUM_ACTION_TOKENS, (B, H))
    targets = torch.randint(0, NUM_VISUAL_TOKENS, (B, H, TOKENS_PER_FRAME))
    gt = torch.rand(B, H, 3, 64, 64)
    total, parts = rollout_loss(model, stub_decoder, z_ctx, actions, targets, gt, H)
    ok &= check("rollout_loss returns scalar + parts",
                total.dim() == 0 and "ce" in parts and "pixel" in parts)

    # Memory-token + state-continuity rollout: reports a 'state' loss whose
    # gradient reaches both the state head and the cross-attention gate.
    Hs = 2
    ms_actions = torch.randint(0, NUM_ACTION_TOKENS, (B, Hs))
    ms_targets = torch.randint(0, NUM_VISUAL_TOKENS, (B, Hs, TOKENS_PER_FRAME))
    total_ms, parts_ms = rollout_loss(
        mem_state, stub_decoder, tokens, ms_actions, ms_targets, None, Hs,
        pixel_weight=0.0, context_anchor=torch.rand(B, 1, 16, G, G),
        target_anchor=torch.rand(B, Hs, 16, G, G), anchor_lambda=1.0,
        context_state=torch.rand(B, 1, 4), target_state=torch.rand(B, Hs, 4),
        state_weight=0.5,
    )
    ok &= check("mem+state rollout returns a 'state' loss part",
                "state" in parts_ms and total_ms.dim() == 0)
    total_ms.backward()
    ok &= check("state-continuity gradient reaches the state head + cross gate",
                mem_state.state_head[-1].weight.grad is not None
                and mem_state.blocks[0].cross_gate.grad is not None)

    # Exact self-rollout must feed the same generated frame into the next step
    # and retain only the configured rolling context window.
    class SpyRollout(torch.nn.Module):
        action_cond = None

        def __init__(self):
            super().__init__()
            self.prefixes = []

        def forward(self, seq, cond_ids=None, anchor_emb=None, noise_levels=None,
                    memory=None, mem_mask=None, return_state=False):
            return torch.zeros(
                seq.shape[0], seq.shape[1], VOCAB_SIZE,
                device=seq.device, requires_grad=True,
            )

        @torch.no_grad()
        def generate_frame(self, prefix, action_id, **kwargs):
            self.prefixes.append(prefix.clone())
            value = len(self.prefixes)
            return torch.full(
                (prefix.shape[0], TOKENS_PER_FRAME), value,
                dtype=torch.long, device=prefix.device,
            )

    spy = SpyRollout()
    one_ctx_actions = torch.randint(0, NUM_ACTION_TOKENS, (B, 1))
    one_ctx_visual = torch.randint(0, NUM_VISUAL_TOKENS, (B, 1, TOKENS_PER_FRAME))
    one_ctx = build_context(one_ctx_actions, one_ctx_visual)
    sr_actions = torch.randint(0, NUM_ACTION_TOKENS, (B, 2))
    sr_targets = torch.randint(0, NUM_VISUAL_TOKENS, (B, 2, TOKENS_PER_FRAME))
    rollout_loss(
        spy, stub_decoder, one_ctx, sr_actions, sr_targets, None, 2,
        pixel_weight=0.0, self_rollout=True, context_window=1,
    )
    exact_feedback = (
        len(spy.prefixes) == 2
        and spy.prefixes[1].shape[1] == FRAME_STRIDE
        and bool((spy.prefixes[1][:, 1:] == 1).all())
        and spy.training
    )
    ok &= check("self-rollout feeds exact generated frame with bounded context", exact_feedback)

    print(f"\nTorch tests: {'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return ok


if __name__ == "__main__":
    logic_ok = logic_tests()
    try:
        import torch  # noqa: F401
    except ImportError:
        print("\ntorch not installed -> skipping tensor shape tests "
              "(logic tests above are the verifiable surface here).")
        raise SystemExit(0 if logic_ok else 1)
    torch_ok = torch_tests()
    raise SystemExit(0 if (logic_ok and torch_ok) else 1)
