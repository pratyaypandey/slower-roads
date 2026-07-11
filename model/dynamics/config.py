"""Shared, torch-free config for the dynamics core.

Holds the vocab layout, action tokenization, and the sequence-layout / index
bookkeeping that both ar_core.py (torch) and test_shapes.py (pure python) rely
on. Nothing here imports torch or numpy, so the layout logic stays verifiable
on a machine with neither installed.
"""

from math import prod

# Latent grid (mirrors the tokenizer contract, §1/§2).
LEVELS = [8, 8, 8, 5, 5]      # FSQ levels per channel
C = len(LEVELS)               # FSQ channels per token
G = 8                         # latent grid side -> G*G tokens per frame

TOKENS_PER_FRAME = G * G                 # 64 visual tokens per frame
NUM_VISUAL_TOKENS = prod(LEVELS)         # 12800 visual code indices
NUM_ACTION_TOKENS = 9                    # §3: 3 longitudinal * 3 lateral buckets
VOCAB_SIZE = NUM_VISUAL_TOKENS + NUM_ACTION_TOKENS

# One frame step in the flattened sequence is [u_t, z_t[0..TOKENS_PER_FRAME-1]].
FRAME_STRIDE = 1 + TOKENS_PER_FRAME      # 65

# Action bucket edges (§3).
LONGITUDINAL_EDGES = [-1.0, -0.33, 0.33, 1.0]   # over (throttle - brake)
LATERAL_EDGES = [-1.0, -0.5, 0.5, 1.0]          # over steer


def bucket(value, edges):
    """Index of the interval in `edges` that `value` falls into, clamped to
    [0, len(edges)-2]. `edges` are ascending boundaries defining len-1 buckets."""
    idx = 0
    for i in range(len(edges) - 1):
        if value >= edges[i]:
            idx = i
    return min(idx, len(edges) - 2)


def tokenize_action(throttle, brake, steer):
    """Discrete action token in [0, NUM_ACTION_TOKENS) per §3."""
    ti = bucket(throttle - brake, LONGITUDINAL_EDGES)   # 3 longitudinal buckets
    si = bucket(steer, LATERAL_EDGES)                   # 3 lateral buckets
    return ti * 3 + si


def action_to_token_id(action_id):
    """Map an action id in [0, NUM_ACTION_TOKENS) to its shared-vocab token id."""
    return NUM_VISUAL_TOKENS + action_id


def is_action_token(token_id):
    return token_id >= NUM_VISUAL_TOKENS


def interleave_frame_layout(action_tokens, visual_tokens):
    """Build one flattened token sequence for a single sample (§4).

    action_tokens: list of length T (already vocab-offset action ids)
    visual_tokens: list of T lists, each length TOKENS_PER_FRAME (visual ids)
    Returns a flat list of length T * FRAME_STRIDE laid out as
        [u_0, z_0[0..63], u_1, z_1[0..63], ...].
    """
    seq = []
    for u_t, z_t in zip(action_tokens, visual_tokens):
        seq.append(u_t)
        seq.extend(z_t)
    return seq


def causal_mask_bool(seq_len):
    """Lower-triangular boolean mask, True where attention is allowed
    (query i may attend to key j iff j <= i). Pure-python for verification."""
    return [[j <= i for j in range(seq_len)] for i in range(seq_len)]


def kv_cache_positions(context_len, num_generated):
    """Absolute sequence positions written to the KV cache during an
    incremental decode: `context_len` prefill positions followed by
    `num_generated` one-at-a-time decode positions. The cache length after
    step k (0-indexed) is context_len + k + 1."""
    return list(range(context_len + num_generated))
