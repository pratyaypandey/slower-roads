"""State-space dynamics: learn (state, action) -> next_state on the sparse
{x, z, heading, speed} vectors from the sim. No renderer, no tokenizer — this is
the scene-representation branch (§6) that needs no gl, so it trains today while
the pixel path is blocked on the native build.

It de-risks the core question — can the net learn the driving dynamics? — with a
multi-step rollout loss (feed predictions back, §0/§5 anti-drift structure), the
same shape the pixel dynamics core uses.

    node sim/headless/generate_state.js --seed 1 --steps 5000
    python -m model.train_state_dynamics --data data/seed1_state --epochs 40

--smoke runs a CPU random-tensor pass, no data needed.
"""

import argparse
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.data.dataset import SimSequenceDataset
from model.dynamics.config import NUM_ACTION_TOKENS

STATE_DIM = 4  # x, z, heading, speed


class StateDynamics(nn.Module):
    """Predicts the next-state delta from (state, action). Predicting the delta
    (not the absolute state) keeps targets small and centered, which matters
    because x/z grow unboundedly along a drive."""

    def __init__(self, hidden=128):
        super().__init__()
        self.action_embed = nn.Embedding(NUM_ACTION_TOKENS, 16)
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM + 16, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, STATE_DIM),
        )

    def forward(self, state, action_id):
        a = self.action_embed(action_id)
        return state + self.net(torch.cat([state, a], dim=-1))


def step_weights(H, mode, device):
    # Per-step loss weights (Jai's feedback, fix 1). 'decay' downweights the
    # noisy far-future terms for stability; 'increase' emphasizes long-horizon
    # coherence (the product goal); 'flat' is uniform. Normalized to mean 1 so
    # the overall loss scale is comparable across modes.
    k = torch.arange(H, dtype=torch.float32, device=device)
    if mode == "decay":
        w = 0.9 ** k
    elif mode == "increase":
        w = 1.0 + k
    else:
        w = torch.ones(H, device=device)
    return w / w.mean()


def rollout_loss(model, init_state, actions, target_states,
                 teacher_forcing=0.0, weight_mode="flat"):
    # init_state (B, STATE_DIM); actions (B, H); target_states (B, H, STATE_DIM).
    # Roll H steps feeding predictions back, so the model trains on its own
    # trajectory (anti-drift / anti-exposure-bias), not one-step teacher forcing.
    #
    # Jai's feedback on compounding error, both applied here:
    #  - teacher_forcing in [0,1]: blend the fed-back state toward the ground
    #    truth (scheduled sampling) to ground early training. Annealed to 0 by
    #    the caller so inference-time free-running is still learned.
    #  - weight_mode: per-step loss weighting (see step_weights).
    H = actions.shape[1]
    w = step_weights(H, weight_mode, init_state.device)
    state = init_state
    loss = init_state.new_zeros(())
    for k in range(H):
        pred = model(state, actions[:, k])
        loss = loss + w[k] * F.mse_loss(pred, target_states[:, k])
        # Next input: the prediction, optionally pulled toward the target. At
        # tf=0 this is pure free-running; at tf=1 it's full teacher forcing.
        if teacher_forcing > 0.0:
            state = (1 - teacher_forcing) * pred + teacher_forcing * target_states[:, k]
        else:
            state = pred
    return loss / H


def train(args):
    device = torch.device(args.device)
    model = StateDynamics(hidden=args.hidden).to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"state-dynamics parameters: {n / 1e3:.1f}K")

    if args.smoke:
        B, H = 4, args.horizon
        init = torch.randn(B, STATE_DIM, device=device)
        actions = torch.randint(0, NUM_ACTION_TOKENS, (B, H), device=device)
        targets = torch.randn(B, H, STATE_DIM, device=device)
        loss = rollout_loss(model, init, actions, targets,
                            teacher_forcing=0.5, weight_mode="decay")
        loss.backward()
        print(f"[smoke] rollout loss {loss.item():.4f} (tf=0.5, decay) — OK")
        return

    dataset = SimSequenceDataset(
        os.path.join(args.data, "manifest.json"),
        context=1, horizon=args.horizon, representation="state",
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Standardize per-dim: x/z span thousands while heading/speed are O(1), so a
    # raw MSE is ~100000x more sensitive to position than to the steering/accel
    # dynamics we actually want learned. Run the whole rollout in z-scored space
    # so all four dims contribute equally; stats are saved with the checkpoint.
    mean, std = dataset_state_stats(dataset, device)
    print(f"state mean {mean.tolist()}\nstate std  {std.tolist()}")

    os.makedirs(args.out, exist_ok=True)
    for epoch in range(args.epochs):
        # Anneal teacher forcing from tf_start -> 0 linearly across training, so
        # early epochs are grounded (stable) and late epochs are free-running
        # (learns to self-correct, matching inference). Constant grounding would
        # just trade instability for exposure-bias drift.
        frac = epoch / max(1, args.epochs - 1)
        tf = args.tf_start * (1 - frac)
        running = 0.0
        for item in loader:
            # context=1, so the last context state is the rollout's start.
            init = ((item["context_state"][:, -1].float().to(device)) - mean) / std
            actions = item["target_actions"].to(device)
            targets = (item["target_state"].float().to(device) - mean) / std
            loss = rollout_loss(model, init, actions, targets,
                                teacher_forcing=tf, weight_mode=args.weight_mode)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            running += loss.item()
        avg = running / max(1, len(loader))
        print(f"epoch {epoch + 1}/{args.epochs}  rollout_mse {avg:.5f}  "
              f"rmse {math.sqrt(avg):.4f}  tf {tf:.2f}")

    ckpt = os.path.join(args.out, "state_dynamics.pt")
    torch.save({"model": model.state_dict(), "hidden": args.hidden,
                "state_mean": mean.cpu(), "state_std": std.cpu()}, ckpt)
    print(f"saved {ckpt}")


def dataset_state_stats(dataset, device):
    # Mean/std per state dim over every state in the dataset. Chunks may be torch
    # tensors (torch present) or numpy arrays; as_tensor handles both, and cat
    # stacks the (T,4)/(H,4) chunks into one (N,4) table.
    chunks = []
    for i in range(len(dataset)):
        item = dataset[i]
        chunks.append(torch.as_tensor(item["context_state"], dtype=torch.float32))
        chunks.append(torch.as_tensor(item["target_state"], dtype=torch.float32))
    flat = torch.cat(chunks, dim=0).to(device)
    mean = flat.mean(dim=0)
    std = flat.std(dim=0).clamp_min(1e-6)  # guard dims with no variance
    return mean, std


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/seed1_state")
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--grad-clip", type=float, default=1.0, dest="grad_clip")
    p.add_argument("--tf-start", type=float, default=0.5, dest="tf_start",
                   help="initial teacher-forcing ratio, annealed to 0 (0 = pure free-run)")
    p.add_argument("--weight-mode", choices=["flat", "decay", "increase"],
                   default="flat", dest="weight_mode",
                   help="per-step rollout loss weighting")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--smoke", action="store_true")
    train(p.parse_args())


if __name__ == "__main__":
    main()
