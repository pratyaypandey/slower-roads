"""Multi-step rollout loss (§5).

Two-part loss: token-space cross-entropy (dense gradient to the dynamics core)
plus a pixel loss on the DECODED multi-step rollout (the drift signal). The
decoder is injected as a callable so this file never imports the tokenizer's
decoder concretely.
"""

import torch
import torch.nn.functional as F

from model.dynamics.config import (
    NUM_VISUAL_TOKENS,
    TOKENS_PER_FRAME,
)
from model.dynamics.sequence import action_to_vocab


def _frame_logits(model, prefix, target_visual, cond_seq=None):
    """Teacher-forced logits for one frame's TOKENS_PER_FRAME visual tokens.

    prefix: (B, P) tokens ending in the frame's action token u_t.
    target_visual: (B, TOKENS_PER_FRAME) ground-truth visual ids for this frame.
    cond_seq: optional (B, P+TOKENS_PER_FRAME-1) per-position action-cond ids.
    Returns (logits (B, TOKENS_PER_FRAME, V), full_seq (B, P+TOKENS_PER_FRAME)).
    The i-th logit predicts target_visual[:, i]: position (P-1) predicts token 0,
    and target_visual[:, :-1] is fed in to predict the rest within the frame.
    """
    seq = torch.cat([prefix, target_visual[:, :-1]], dim=1)
    logits = model.forward(seq, cond_ids=cond_seq)
    frame_logits = logits[:, prefix.shape[1] - 1:, :]   # (B, TOKENS_PER_FRAME, V)
    return frame_logits, seq


def default_pixel_loss(frame_hat, frame_gt):
    """L1 in [0,1] pixel space. Swap for L1+LPIPS-lite when available."""
    return F.l1_loss(frame_hat, frame_gt)


def rollout_loss(model, decoder, z_ctx, action_ids, target_tokens, gt_frames, H,
                 ce_weight=1.0, pixel_weight=1.0, pixel_loss=default_pixel_loss,
                 teacher_forcing=0.0, cond_ctx=None):
    """Roll H steps, accumulating token CE + decoded-pixel loss (§5).

    model:         ARDynamics.
    decoder:       callable (B, TOKENS_PER_FRAME) int -> (B,3,64,64) float in [0,1].
    z_ctx:         (B, T_ctx) interleaved context tokens (action-offset applied).
    action_ids:    (B, H) int64 action ids in [0, NUM_ACTION_TOKENS) per step.
    target_tokens: (B, H, TOKENS_PER_FRAME) ground-truth visual ids per step.
    gt_frames:     (B, H, 3, 64, 64) ground-truth frames per step.
    H:             rollout horizon.
    teacher_forcing: prob in [0,1] that a step feeds GROUND-TRUTH tokens back into
        the context instead of the model's own prediction (scheduled sampling).
        0 = pure free-running rollout (the anti-drift default); 1 = full teacher
        forcing. Decided per-step so the trainer can anneal it across training.

    Returns (total_loss, {"ce": ce_total, "pixel": pixel_total}).
    """
    from model.dynamics.config import FRAME_STRIDE, TOKENS_PER_FRAME as TPF
    use_cond = getattr(model, "action_cond", None) is not None
    prefix = z_ctx
    cond = cond_ctx if use_cond else None                       # (B, len(prefix))
    ce_total = z_ctx.new_zeros((), dtype=torch.float32)
    pixel_total = z_ctx.new_zeros((), dtype=torch.float32)

    for k in range(H):
        u_t = action_to_vocab(action_ids[:, k]).unsqueeze(1)   # (B,1)
        prefix_k = torch.cat([prefix, u_t], dim=1)
        target_k = target_tokens[:, k, :]                          # (B, tokens)

        cond_seq = None
        if use_cond:
            a_k = action_ids[:, k:k + 1]                           # (B,1) this frame's action
            # cond for [prefix_k, target_visual[:-1]]: prefix's cond + a_k for u_t
            # and for the TPF-1 fed-in target visual positions.
            cond_seq = torch.cat([cond, a_k.expand(-1, 1 + TPF - 1)], dim=1)

        frame_logits, _ = _frame_logits(model, prefix_k, target_k, cond_seq)
        ce_total = ce_total + F.cross_entropy(
            frame_logits.reshape(-1, frame_logits.shape[-1]),
            target_k.reshape(-1),
        )

        pred_tokens = frame_logits[..., :NUM_VISUAL_TOKENS].argmax(dim=-1)  # (B,tok)
        # The decoded-pixel term is a non-differentiable drift MONITOR (argmax
        # blocks gradient into the core; the CE is the learning signal). Skip the
        # expensive decode when it's off — this is what makes latent-cache training
        # (frames never loaded, pixel_weight=0) fast.
        if pixel_weight > 0 and gt_frames is not None:
            frame_hat = decoder(pred_tokens)
            pixel_total = pixel_total + pixel_loss(frame_hat, gt_frames[:, k])

        # Autoregress: feed back ground-truth tokens with prob teacher_forcing,
        # else the model's own prediction (scheduled sampling; anneal in trainer).
        if teacher_forcing > 0.0 and torch.rand(()) < teacher_forcing:
            step_tokens = target_k
        else:
            step_tokens = pred_tokens
        prefix = torch.cat([prefix, u_t, step_tokens], dim=1)
        if use_cond:
            # the appended [u_t, step_tokens] is one FRAME_STRIDE block, all a_k.
            cond = torch.cat([cond, action_ids[:, k:k + 1].expand(-1, FRAME_STRIDE)], dim=1)

    total = ce_weight * ce_total + pixel_weight * pixel_total
    return total, {"ce": ce_total.detach(), "pixel": pixel_total.detach()}
