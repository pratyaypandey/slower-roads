"""Schrödinger-bridge dynamics (Jai's notes idea) — a swappable alternative to
the AR transformer, selectable as `--dynamics-arch flow_bridge`.

Instead of classifying the next frame's tokens, learn a flow that transports the
current latent z_t to the next z_{t+1} through continuous FSQ-code space. The
decoder domain (FSQ codes) is discrete + bounded, which the notes exploit twice:
  - discreteness: each flow step reads the t=1 endpoint estimate and SNAPS it onto
    the FSQ grid — a built-in per-step anti-drift anchor.
  - boundedness: the transport stays in a compact normalized-code box [-1, 1].

Representation: we flow in the tokenizer's NORMALIZED code space (FSQ.normalize),
so z is continuous in ~[-1, 1] with C channels per token, shape (B, tok, C). The
grid snap is FSQ.quantize -> FSQ.normalize; the final token ids come from
FSQ.codes_to_indices.

Small, separated pieces mirror the notes' pseudocode: `VelocityNet` (the learned
drift), `flow_step` (one Euler step + endpoint snap), `predict_next` (run the
flow K steps, keep the newest snapped estimate). Speculative/parallel-step
decoding (also in the notes) is left as a clearly-marked stub — the flow works
without it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.tokenizer.fsq_autoencoder import FSQ
from model.dynamics.config import LEVELS, NUM_ACTION_TOKENS, TOKENS_PER_FRAME
from model.registry import register_dynamics


def _sinusoidal(t, dim):
    """Standard sinusoidal embedding of a scalar flow-time t in [0,1]."""
    half = dim // 2
    freqs = torch.exp(-torch.arange(half, device=t.device) * (torch.log(torch.tensor(10000.0)) / half))
    ang = t[..., None] * freqs
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class VelocityNet(nn.Module):
    """Predicts the flow velocity at (z_s, s, action): a per-token MLP conditioned
    on the flow time and the action. Operates on each token's C-vector, sharing
    weights across the tok positions (like a 1x1 conv over the token axis)."""

    def __init__(self, channels, action_dim=16, time_dim=32, hidden=256):
        super().__init__()
        self.time_dim = time_dim
        self.action_embed = nn.Embedding(NUM_ACTION_TOKENS, action_dim)
        self.net = nn.Sequential(
            nn.Linear(channels + action_dim + time_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, channels),
        )

    def forward(self, z_s, s, action_id):
        # z_s (B, tok, C); s (B,) flow time; action_id (B,).
        b, tok, c = z_s.shape
        t = _sinusoidal(s, self.time_dim)[:, None, :].expand(b, tok, self.time_dim)
        a = self.action_embed(action_id)[:, None, :].expand(b, tok, -1)
        return self.net(torch.cat([z_s, a, t], dim=-1))


class FlowBridge(nn.Module):
    """Dynamics core that transports z_t -> z_{t+1} via a learned flow with an
    FSQ-grid snap at each step."""

    def __init__(self, levels=tuple(LEVELS), steps=8, hidden=256):
        super().__init__()
        self.fsq = FSQ(list(levels))
        self.velocity = VelocityNet(self.fsq.num_channels, hidden=hidden)
        self.steps = steps

    def snap_to_grid(self, z):
        """Nearest FSQ grid point of a normalized-code tensor, returned in the
        same normalized space. Uses the (non-differentiable) quantizer, so callers
        detach as needed."""
        codes = self.fsq.quantize(z * self.fsq.half_width.to(z.dtype))
        return self.fsq.normalize(codes)

    def flow_step(self, z_cur, s, ds, action_id):
        """One Euler step of the flow, plus the endpoint snap (notes' core loop).
        Returns (z_next, snapped_endpoint_estimate)."""
        v = self.velocity(z_cur, s, action_id)
        z_next = z_cur + v * ds
        # Endpoint estimate at t=1: z_cur + v*(1-s); snap it onto the grid as the
        # running next-frame guess.
        z_end = z_cur + v * (1.0 - s[:, None, None])
        running = self.snap_to_grid(z_end)
        return z_next, running

    @torch.no_grad()
    def predict_next(self, z_t, action_id):
        """Run the flow K steps from z_t, keeping the newest snapped estimate."""
        b = z_t.shape[0]
        z_cur = z_t
        running = self.snap_to_grid(z_t)
        ds = 1.0 / self.steps
        for i in range(self.steps):
            s = torch.full((b,), i * ds, device=z_t.device)
            z_cur, running = self.flow_step(z_cur, s, ds, action_id)
        return running  # normalized-code estimate of z_{t+1}

    # --- Dynamics protocol (model/interfaces.py) ---
    def loss(self, batch, decoder):
        """Flow-matching loss. batch provides continuous normalized codes for the
        current and next frame (z_cur, z_next, each (B, tok, C)) and the action
        (B,). Optionally adds a pixel term via decoder on the snapped endpoint.

        Flow matching: at a random flow time s, the target velocity of the
        straight path from z_cur to z_next is (z_next - z_cur); train the net to
        predict it at the interpolated point.
        """
        z_cur, z_next, action_id = batch["z_cur"], batch["z_next"], batch["action_ids"]
        b = z_cur.shape[0]
        s = torch.rand(b, device=z_cur.device)
        z_s = (1 - s[:, None, None]) * z_cur + s[:, None, None] * z_next
        v_pred = self.velocity(z_s, s, action_id)
        v_target = z_next - z_cur
        flow_loss = F.mse_loss(v_pred, v_target)

        parts = {"flow": flow_loss.detach()}
        total = flow_loss
        if batch.get("pixel_weight", 0.0) > 0 and decoder is not None:
            # Decode the snapped endpoint and compare to the true next frame.
            idx = self.codes_to_indices(self.snap_to_grid(z_s + v_pred * (1 - s[:, None, None])))
            frame_hat = decoder(idx)
            pix = F.l1_loss(frame_hat, batch["gt_next_frame"])
            parts["pixel"] = pix.detach()
            total = total + batch["pixel_weight"] * pix
        return total, parts

    def codes_to_indices(self, z_norm):
        """Normalized codes -> visual token ids (B, tok)."""
        codes = self.fsq.quantize(z_norm * self.fsq.half_width.to(z_norm.dtype))
        return self.fsq.codes_to_indices(codes)

    @torch.no_grad()
    def generate_frame(self, z_t, action_id):
        """Inference: transport z_t -> z_{t+1} and return its token ids (B, tok).
        Note the input is a continuous latent (B, tok, C), unlike the AR core's
        token context — the flow works in code space."""
        return self.codes_to_indices(self.predict_next(z_t, action_id))

    def speculative_generate(self, *args, **kwargs):
        """STUB (notes follow-on): confirm several near-collinear flow steps in
        parallel / binary-search along the near-straight trajectory to cut the K
        sequential steps. Not needed for correctness; predict_next is the path."""
        raise NotImplementedError(
            "speculative flow decoding is a planned optimization — use "
            "predict_next / generate_frame for now."
        )

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


@register_dynamics("flow_bridge")
def _build_flow_bridge(levels=tuple(LEVELS), steps=8, hidden=256):
    return FlowBridge(levels=levels, steps=steps, hidden=hidden)
