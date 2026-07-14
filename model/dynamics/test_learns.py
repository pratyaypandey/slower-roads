"""M2 "done when": the AR core actually learns action-conditioned dynamics and
rolls out coherently under teacher forcing.

The shape tests prove the plumbing; this proves the *behaviour*. We build a tiny
deterministic toy world whose next frame is a pure function of the previous frame
and the action — the frame scrolls spatially by shift(action) (an
attention-copy task, and a fair analogue of a driving world sliding past) — and
train the real ARDynamics until it fits it. A model that has learned the dynamics:

  1. predicts next frames with high teacher-forced accuracy, and
  2. reproduces a multi-step *autoregressive* rollout (its own KV-cached
     generation, feeding predictions back) against ground truth.

(2) is the load-bearing check: it exercises the exact KV-cached generate_frame
path M2 ships, and shows the frame stays coherent as errors could compound. This
is an overfit demonstration — a fixed batch, trained to convergence — which is the
standard, fast sanity gate for "the training loop + model + generation learn
dynamics." Generalization is a later (GPU, real-data) concern. Runs on CPU in a
few minutes; tune the budget with flags.

    python -m model.dynamics.test_learns [--steps N] [--quick]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch

from model.dynamics.ar_core import ARDynamics
from model.dynamics.config import NUM_ACTION_TOKENS
from model.dynamics.sequence import action_to_vocab, build_context

V_SMALL = 24          # toy world draws token values in [0, V_SMALL)
# Behavioural test runs at a small per-frame grid (not the production 256) so it
# converges in seconds on CPU while exercising the real AR + KV-cache paths.
TOK = 16


def shift_for(action_id):
    """Per-action spatial scroll amount. Actions 0..8 -> shifts -4..+4 (the world
    slides left/right with steering)."""
    return action_id - (NUM_ACTION_TOKENS // 2)


def roll(frame, action_id):
    """One step of the toy dynamics: the frame scrolls spatially by shift(action),
    values preserved (an attention-copy task, not modular arithmetic). This is both
    the faster thing to learn and a better analogue of a driving world sliding past.
    frame: (B, TOK) int; action_id: (B,) int."""
    s = shift_for(action_id)                                # (B,)
    idx = (torch.arange(TOK)[None, :] - s[:, None]) % TOK   # (B, TOK) gather indices
    return torch.gather(frame, 1, idx)


def make_sequences(B, n_frames, gen):
    """Random start frames + actions rolled forward deterministically.
    Convention (matches the AR layout): action a_k produces frame z_k, i.e.
    z_k = roll(z_{k-1}, a_k) — so u_k sits right before the frame it explains.
    Returns visuals (B, n_frames, TOK) and actions (B, n_frames)."""
    frame = torch.randint(0, V_SMALL, (B, TOK), generator=gen)
    actions = torch.randint(0, NUM_ACTION_TOKENS, (B, n_frames), generator=gen)
    visuals = [frame]
    for k in range(1, n_frames):
        frame = roll(frame, actions[:, k])
        visuals.append(frame)
    return torch.stack(visuals, dim=1), actions


def flat_sequence(visuals, actions):
    """Interleave [u_0, z_0, u_1, z_1, ...] over all frames -> (B, n*(1+TOK))."""
    return build_context(actions, visuals)


def teacher_forced_accuracy(model, visuals, actions):
    """Fraction of visual next-token predictions that are exactly right."""
    seq = flat_sequence(visuals, actions)          # (B, n*(1+TOK))
    logits = model(seq)
    pred = logits[:, :-1].argmax(-1)
    target = seq[:, 1:]
    stride = 1 + TOK
    # Only score positions whose target is a visual token (i.e. not the action slot
    # at the start of each frame). Target index j is an action slot iff (j+1) % stride == 0.
    idx = torch.arange(target.shape[1])
    visual_pos = ((idx + 1) % stride) != 0
    correct = (pred[:, visual_pos] == target[:, visual_pos]).float().mean()
    return correct.item()


@torch.no_grad()
def rollout_accuracy(model, visuals, actions, context, horizon):
    """Autoregressive KV-cached rollout vs ground truth, per predicted frame."""
    B = visuals.shape[0]
    ctx = flat_sequence(visuals[:, :context], actions[:, :context])
    accs = []
    for k in range(horizon):
        a = actions[:, context + k]
        pred = model.generate_frame(ctx, a, n_tokens=TOK)       # (B, TOK) greedy
        gt = visuals[:, context + k]
        accs.append((pred == gt).float().mean().item())
        u = action_to_vocab(a).unsqueeze(1)
        ctx = torch.cat([ctx, u, pred], dim=1)                  # feed prediction back
    return accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--quick", action="store_true", help="fewer steps; may not fully converge")
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(os.cpu_count() or 4)
    gen = torch.Generator().manual_seed(1)

    context, horizon = 3, 3
    steps = 200 if args.quick else args.steps
    n_frames = context + horizon

    model = ARDynamics(d_model=128, n_heads=4, n_layers=3,
                       max_seq_len=n_frames * (1 + TOK) + 8)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    print(f"toy dynamics: frame scrolls by shift(action) in [-4,4]; "
          f"model {model.param_count() / 1e6:.2f}M params; {steps} steps (generalization)")

    # Train on a FRESH batch each step (random start frame + actions) so the model
    # must learn the transition RULE, not memorize sequences — a stronger gate.
    B = 32
    model.train()
    for step in range(steps):
        visuals, actions = make_sequences(B, n_frames, gen)
        loss = model.training_step(flat_sequence(visuals, actions))
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % max(1, steps // 6) == 0:
            print(f"  step {step + 1}/{steps}  ce {loss.item():.4f}")

    model.eval()
    ev = torch.Generator().manual_seed(999)               # fresh, never-trained sequences
    visuals, actions = make_sequences(192, n_frames, ev)
    tf = teacher_forced_accuracy(model, visuals, actions)
    ro = rollout_accuracy(model, visuals, actions, context, horizon)
    chance = 1.0 / V_SMALL

    print(f"\nchance next-token accuracy: {chance:.3f}")
    print(f"teacher-forced next-token accuracy: {tf:.3f}")
    print("autoregressive rollout accuracy per step: " +
          "  ".join(f"{a:.3f}" for a in ro))

    ok = True
    ok &= _check("teacher-forced accuracy > 0.7 (learned the rule, >8x chance)", tf > 0.7)
    ok &= _check("autoregressive rollout coherent (first step > 0.7)", ro[0] > 0.7)
    ok &= _check("rollout never collapses to chance (all > 5x chance)",
                 all(a > 5 * chance for a in ro))
    print(f"\nM2 learns-dynamics: {'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if ok else 1)


def _check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    return cond


if __name__ == "__main__":
    main()
