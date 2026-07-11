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

    print("\nTorch shape tests:")
    ok = True
    B, T_ctx = 2, FRAME_STRIDE
    model = ARDynamics(d_model=64, n_heads=4, n_layers=2, max_seq_len=1024)
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

    # Rollout loss end-to-end with a stub decoder (callable, not imported).
    H = 3
    def stub_decoder(vis_tokens):
        b, tok = vis_tokens.shape
        return (vis_tokens.float() / NUM_VISUAL_TOKENS).view(b, 1, 1, tok).expand(
            b, 3, 64, 64).contiguous()

    z_ctx = tokens
    actions = torch.randint(0, NUM_ACTION_TOKENS, (B, H))
    targets = torch.randint(0, NUM_VISUAL_TOKENS, (B, H, TOKENS_PER_FRAME))
    gt = torch.rand(B, H, 3, 64, 64)
    total, parts = rollout_loss(model, stub_decoder, z_ctx, actions, targets, gt, H)
    ok &= check("rollout_loss returns scalar + parts",
                total.dim() == 0 and "ce" in parts and "pixel" in parts)

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
