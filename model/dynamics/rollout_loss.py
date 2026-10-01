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


def _frame_logits(model, prefix, target_visual, cond_seq=None, anchor_seq=None,
                  noise_seq=None, memory=None, return_state=False):
    """Teacher-forced logits for one frame's TOKENS_PER_FRAME visual tokens.

    prefix: (B, P) tokens ending in the frame's action token u_t.
    target_visual: (B, TOKENS_PER_FRAME) ground-truth visual ids for this frame.
    cond_seq: optional (B, P+TOKENS_PER_FRAME-1) per-position action-cond ids.
    memory: optional (B, M, d) current-frame skeleton memory tokens; only the
        current frame's positions (the appended [u_t, target_visual[:-1]] block)
        attend to them, so the cached context K/V stay uncontaminated.
    Returns (frame_logits (B, TOKENS_PER_FRAME, V), full_seq, state_pred).
    The i-th logit predicts target_visual[:, i]: position (P-1) predicts token 0,
    and target_visual[:, :-1] is fed in to predict the rest within the frame.
    """
    seq = torch.cat([prefix, target_visual[:, :-1]], dim=1)
    mem_mask = None
    if memory is not None or return_state:
        cur_start = prefix.shape[1] - 1        # index of u_t, the first current-frame pos
        mem_mask = torch.zeros(seq.shape, device=seq.device)
        mem_mask[:, cur_start:] = 1.0
    out = model.forward(
        seq, cond_ids=cond_seq, anchor_emb=anchor_seq, noise_levels=noise_seq,
        memory=memory, mem_mask=mem_mask, return_state=return_state,
    )
    logits, state_pred = out if return_state else (out, None)
    frame_logits = logits[:, prefix.shape[1] - 1:, :]   # (B, TOKENS_PER_FRAME, V)
    return frame_logits, seq, state_pred


def default_pixel_loss(frame_hat, frame_gt):
    """L1 in [0,1] pixel space. Swap for L1+LPIPS-lite when available."""
    return F.l1_loss(frame_hat, frame_gt)


def _state_loss(state_pred, target_state, context_state, k):
    """Smooth-L1 between the predicted and true per-frame state delta.

    The delta (frame k's state minus the previous frame's) is naturally small and
    centered; standardizing per-dim by the detached batch std keeps the four
    channels (x, z, heading, speed) at comparable loss scales."""
    prev = context_state[:, -1] if k == 0 else target_state[:, k - 1]
    delta = target_state[:, k] - prev                      # (B, STATE_DIM)
    scale = delta.detach().std(dim=0, keepdim=True).clamp_min(1e-3)
    return F.smooth_l1_loss(state_pred / scale, delta / scale)


def rollout_loss(model, decoder, z_ctx, action_ids, target_tokens, gt_frames, H,
                 ce_weight=1.0, pixel_weight=1.0, pixel_loss=default_pixel_loss,
                 teacher_forcing=0.0, cond_ctx=None, self_rollout=False,
                 context_window=None, context_anchor=None, target_anchor=None,
                 anchor_lambda=0.0, context_noise=None, context_state=None,
                 target_state=None, state_weight=0.0):
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
    self_rollout: feed frames produced by the exact KV-cached inference path.
        False retains the cheaper approximation derived from teacher-forced
        within-frame logits; True removes that train/inference mismatch.
    context_window: maximum number of complete frames retained in the rolling
        prefix. Defaults to the number of frames in z_ctx, matching inference.

    Returns (total_loss, {"ce": ce_total, "pixel": pixel_total}).
    """
    from model.dynamics.config import FRAME_STRIDE, TOKENS_PER_FRAME as TPF
    use_cond = getattr(model, "action_cond", None) is not None
    prefix = z_ctx
    cond = cond_ctx if use_cond else None                       # (B, len(prefix))
    use_anchor = getattr(model, "anchor_encoder", None) is not None
    anchor_prefix = (
        model.anchor_sequence(context_anchor, anchor_lambda) if use_anchor else None
    )
    use_noise = getattr(model, "corruption_embed", None) is not None
    noise_prefix = context_noise
    if use_noise and noise_prefix is None:
        noise_prefix = torch.zeros(prefix.shape, device=prefix.device)
    use_mem = getattr(model, "mem_encoder", None) is not None
    use_state = getattr(model, "state_head", None) is not None and state_weight > 0
    if use_state and (context_state is None or target_state is None):
        raise ValueError("state-head rollout requires context/target state vectors")
    if context_window is None:
        context_window = max(1, z_ctx.shape[1] // FRAME_STRIDE)
    ce_total = z_ctx.new_zeros((), dtype=torch.float32)
    pixel_total = z_ctx.new_zeros((), dtype=torch.float32)
    state_total = z_ctx.new_zeros((), dtype=torch.float32)

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

        anchor_seq = None
        current_anchor_frame = None
        if use_anchor:
            if context_anchor is None or target_anchor is None:
                raise ValueError("anchor-conditioned rollout requires context/target anchors")
            current_anchor_frame = model.anchor_frames(
                target_anchor[:, k:k + 1], anchor_lambda
            )[:, 0]
            # seq contains prefix + action + target_visual[:-1], i.e. the first
            # TPF positions of this frame's FRAME_STRIDE anchor block.
            anchor_seq = torch.cat(
                [anchor_prefix, current_anchor_frame[:, :TPF]], dim=1
            )

        noise_seq = None
        if use_noise:
            noise_seq = torch.cat([
                noise_prefix,
                torch.zeros(prefix.shape[0], TPF, device=prefix.device),
            ], dim=1)
        memory = None
        if use_mem:
            if target_anchor is None:
                raise ValueError("memory-conditioned rollout requires target anchors")
            memory = model.encode_memory(target_anchor[:, k], anchor_lambda)
        frame_logits, _, state_pred = _frame_logits(
            model, prefix_k, target_k, cond_seq, anchor_seq, noise_seq,
            memory=memory, return_state=use_state,
        )
        ce_total = ce_total + F.cross_entropy(
            frame_logits.reshape(-1, frame_logits.shape[-1]),
            target_k.reshape(-1),
        )
        if use_state:
            state_total = state_total + _state_loss(
                state_pred, target_state, context_state, k
            )

        approx_tokens = frame_logits[..., :NUM_VISUAL_TOKENS].argmax(dim=-1)  # (B,tok)
        if self_rollout:
            # Generate exactly as inference does: token-by-token with KV cache,
            # conditioned on earlier generated tokens inside this frame. The
            # rollout is deliberately detached; CE above remains the gradient
            # signal, while future steps learn to recover from authentic model
            # errors (a discrete self-forcing/DAgger-style update).
            # ``no_grad`` does not disable dropout, so temporarily mirror the
            # eval-mode behavior used by deployment and restore the caller's
            # module state immediately afterward.
            was_training = model.training
            if was_training:
                model.eval()
            try:
                pred_tokens = model.generate_frame(
                    prefix, action_ids[:, k],
                    context_anchor_emb=(None if anchor_prefix is None else anchor_prefix.detach()),
                    anchor=(None if target_anchor is None else target_anchor[:, k]),
                    anchor_lambda=anchor_lambda,
                    context_noise=noise_prefix,
                )
            finally:
                if was_training:
                    model.train()
        else:
            pred_tokens = approx_tokens
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
        if use_anchor:
            anchor_prefix = torch.cat([anchor_prefix, current_anchor_frame], dim=1)
        if use_noise:
            noise_prefix = torch.cat([
                noise_prefix,
                torch.zeros(prefix.shape[0], FRAME_STRIDE, device=prefix.device),
            ], dim=1)
        max_tokens = context_window * FRAME_STRIDE
        if prefix.shape[1] > max_tokens:
            prefix = prefix[:, -max_tokens:]
            if use_cond:
                cond = cond[:, -max_tokens:]
            if use_anchor:
                anchor_prefix = anchor_prefix[:, -max_tokens:]
            if use_noise:
                noise_prefix = noise_prefix[:, -max_tokens:]

    total = ce_weight * ce_total + pixel_weight * pixel_total + state_weight * state_total
    parts = {"ce": ce_total.detach(), "pixel": pixel_total.detach()}
    if use_state:
        parts["state"] = state_total.detach()
    return total, parts
