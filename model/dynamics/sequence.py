"""Token-sequence layout shared by training, inference, and eval.

The AR dynamics core consumes one flat sequence per window, interleaved as
[u_0, z_0(tok visual), u_1, z_1, ...] where action ids are offset above the
visual codebook into the shared vocab. This offset was duplicated across the
trainer, the rollout loss, generation, and eval; centralizing it here keeps the
layout definition in one place (and out of a training script that eval imported).
"""

import torch

from model.dynamics.config import NUM_VISUAL_TOKENS, FRAME_STRIDE


def action_to_vocab(action_ids):
    """Map action ids [0, NUM_ACTION_TOKENS) into the shared vocab, above the
    visual codes. Works on any tensor shape."""
    return action_ids + NUM_VISUAL_TOKENS


def frame_cond_ids(action_ids):
    """Per-position action-conditioning ids for a frame-aligned sequence.

    action_ids: (B, T) the action driving each frame. Returns (B, T*FRAME_STRIDE)
    with each frame's action repeated across its whole [u_t, z_t...] block — so the
    strong action-conditioning path can add the action to *every* position of the
    frame it drives (not just the one action token, which the core under-weights).
    """
    return action_ids.repeat_interleave(FRAME_STRIDE, dim=1)


def build_context(action_ids_ctx, visual_ctx):
    """Interleave a window into the flat AR sequence.

    action_ids_ctx: (B, T) int64 action ids (pre-offset).
    visual_ctx:     (B, T, tok) int64 visual code ids.
    returns:        (B, T * (1 + tok)) flat interleaved tokens.
    """
    b, t = action_ids_ctx.shape
    u = action_to_vocab(action_ids_ctx).unsqueeze(-1)   # (B, T, 1)
    seq = torch.cat([u, visual_ctx], dim=-1)            # (B, T, 1+tok)
    return seq.reshape(b, t * seq.shape[-1])
