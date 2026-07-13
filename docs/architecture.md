# Architecture — tokenization → dynamics → decode (Task 3)

This is the **interface contract** for the model side. It pins the shapes and
formats every component agrees on, so the tokenizer, dynamics core, and decoder
can be built and trained independently. It also writes down the two research
ideas from `jais_notes/notes.md` (Schrödinger-bridge flow dynamics;
scene-representation choice) as explicit, comparable design branches rather than
buried assumptions.

Everything below is pseudocode + shapes. Real implementations live in
`model/tokenizer/`, `model/dynamics/`, `model/data/`.

---

## 0. The one loop everything serves

```
frame_t ──encode──▶ latent tokens z_t ──┐
actions a_t ─────────tokenize──▶ u_t ───┤
                                        ▼
                             dynamics: predict z_{t+1}
                                        │
              (roll this forward H steps autoregressively)
                                        ▼
        decode each ẑ_{t+k} ──▶ pixels ──▶ compare to ground-truth frames
                                        ▼
                          multi-step rollout loss
```

The load-bearing choice (from the notes): **loss is computed over a multi-step
rollout, decoded, and compared** — not just one-step teacher forcing. Training
on the model's own predictions is what fights drift (Self-Forcing / Diffusion
Forcing, ROADMAP M3).

---

## 1. Shapes and symbols (the contract — do not diverge)

| symbol | meaning | shape / dtype |
|---|---|---|
| `B` | batch | — |
| `T` | context frames | — |
| `H` | rollout horizon (frames predicted before loss) | e.g. 4–8 |
| `frame` | RGB observation | `(B,3,64,64)` uint8→float in [0,1] |
| `G` | latent grid side | `16` (so 16×16 = 256 tokens/frame; was 8 → bumped at M1 to keep small detail like the car) |
| `C` | FSQ channels per token | `len(levels)`, e.g. 4–6 |
| `L` | FSQ levels per channel | e.g. `[8,8,8,5,5]` |
| `z` | latent code indices | `(B,T,G*G)` int64 in `[0, prod(L))` |
| `z_cont` | pre-quant latent | `(B,T,G*G,C)` float |
| `a` | raw action | `{throttle,brake,steer}` per frame |
| `u` | action token | `(B,T)` int64, small vocab |
| `V` | total token vocab | `prod(L)` visual + action tokens + specials |

**Resolution 64×64 and grid 16×16 are provisional** and, per ROADMAP §3, must be
set from the 33 ms frame budget — not from fidelity. The grid was raised 8×8→16×16
at M1 (256 tokens/frame) because 8×8 dropped the car; revisit against the budget.

---

## 2. Visual tokenization (FSQ)

FSQ instead of VQ-VAE: no codebook, no commitment loss, no dead codes — the
"codebook" is the implicit product grid of per-channel levels.

```
encode(frame):                      # (B,3,64,64) -> (B, G*G, C)
    h = conv_stem(frame)            # downsample 64->8 spatially, to C channels
    z_cont = h.reshape(B, G*G, C)
    return z_cont

quantize(z_cont):                   # FSQ: bound then round (straight-through)
    zb = (L/2) * tanh(z_cont)       # bound each channel into its level range
    zq = round(zb)
    zq = zb + stop_grad(zq - zb)    # STE: gradient flows through zb
    return zq                       # (B, G*G, C) on the integer grid

codes_to_indices(zq):               # grid point -> single int per token
    # mixed-radix over the C channels -> index in [0, prod(L))
    return sum_c (zq_shifted[...,c] * radix[c])   # (B, G*G) int64
```

Decoder is the mirror: `indices -> zq -> deconv -> (B,3,64,64)`.

**Which FSQ variant?** Open research question (Task 2). The uniform product grid
is the baseline. Jai's AEP intuition — that a Gaussian-ish latent wastes codes in
the grid's low-probability corners, so a companded/non-uniform grid should win —
is being measured empirically in `model/tokenizer/fsq_variants_study.py`. The
tokenizer takes `levels` and an optional `companding` transform as config so the
winning variant drops in without reshaping anything downstream.

---

## 3. Action tokenization

Actions are low-dim and continuous (`throttle,brake∈[0,1]`, `steer∈[-1,1]`). Two
options; we take **(a)** for the AR core.

**(a) Discrete action token (default, MineWorld-style).** Bin the action into a
small vocab so it's just another token in the sequence:

```
tokenize_action(a):                 # -> int in [0, A)
    ti = bucket(a.throttle - a.brake, edges=[-1,-.33,.33,1])  # 3 long. buckets
    si = bucket(a.steer,             edges=[-1,-.5,.5,1])      # 3 lateral buckets
    return ti*3 + si                 # A = 9 action tokens
```

**(b) Continuous control vector** added to token embeddings (kept for the
steering work — CAA adds directions in the same space). Noted, not the default.

Action vocab is offset above the visual vocab so one embedding table covers both:
`token_id = code_index` for visual, `prod(L) + action_id` for actions.

---

## 4. Dynamics core — two branches

Both consume `(context tokens)` and produce `z_{t+1}`. Branch A is the ROADMAP
default and the safe path to a working M2. Branch B is Jai's flow idea, spec'd so
it can be swapped behind the same interface.

### Branch A — Autoregressive transformer (default)

Interleave visual + action tokens into one sequence, predict next token.

```
sequence per frame step:  [u_t, z_t[0], z_t[1], ..., z_t[G*G-1]]
train:   standard next-token cross-entropy over the flattened sequence
infer:   KV-cached greedy/sampled decode of the G*G visual tokens for t+1,
         conditioned on u_t; then feed back in for t+2 (autoregressive)
```

- **Why AR over diffusion** (ROADMAP §2): KV-cache reuse for real-time,
  clean mid-layer activations for steering (`h ← h + α·v`), predictable latency.
- **Steering hook:** expose every block's residual stream so a direction can be
  added at layer ℓ. Export must keep the injection layer splittable (ROADMAP §3).

### Branch B — Schrödinger-bridge flow (Jai's idea, from notes)

The decoder domain is discrete + bounded (the FSQ grid). So instead of
classifying the next token, learn a **flow / Schrödinger bridge** that transports
`z_t → z_{t+1}` continuously, reading off a running endpoint estimate and
**snapping it to the FSQ grid** each step:

```
flow_step(z_t, s, action):          # s in [0,1] along the bridge
    v = velocity_net(z_current, s, action)   # learned drift
    z_current = z_current + v * ds
    z_end_hat = endpoint_estimate(z_current, s)   # bridge's t=1 guess
    running = quantize(z_end_hat)     # discretize onto FSQ grid  <-- the anchor
    return z_current, running

predict_next(z_t, action):
    z_current = z_t;  running = None
    for s in linspace(0,1,K):
        z_current, new_running = flow_step(z_current, s, action)
        running = new_running          # keep newest, discard prior estimate
    return running
```

Why this is attractive here: the FSQ grid's **discreteness makes each step
denoise onto a valid code** (a built-in anti-drift snap), and its
**boundedness** keeps the bridge in a compact domain. Speculative decoding
(also from the notes) claws back real-time: confirm several near-collinear flow
steps in parallel, or binary-search along the near-straight trajectory, instead
of K sequential steps.

**Decision:** build Branch A first (it unblocks M2 and the steering work depends
on its residual stream). Prototype Branch B in `model/dynamics/flow_bridge.py`
as a parallel experiment; compare on the drift curve. Same in/out contract
(`predict_next(z_t, u_t) -> z_{t+1}`) so eval code doesn't care which runs.

---

## 5. Multi-step rollout loss (the important part)

```
rollout_loss(model, decoder, z_ctx, actions, gt_frames, H):
    z = z_ctx
    total = 0
    for k in range(H):
        z_next = model.predict_next(z, actions[k])     # one predicted frame
        # (A) token-space loss: cheap, dense, trains the dynamics core
        total += ce(model.logits, z_next_target[k])    # AR branch
        # (B) pixel-space loss on the DECODED rollout: what the notes ask for
        frame_hat = decoder(z_next)
        total += pixel_loss(frame_hat, gt_frames[k])   # e.g. L1 + LPIPS-lite
        z = feed_back(z, z_next)      # autoregress: next input uses prediction
    return total
```

Two-part loss on purpose: token CE gives dense gradient to the dynamics core;
decoded-pixel loss over the multi-step rollout is what actually measures whether
frames *stay coherent as they compound* — the drift signal. Gradient through the
decoded rollout is the mechanism the notes describe ("multiple steps are
simulated, decoded, and then compared for finding the loss").

Trust-region note (Jai's dataloader idea): keep the bridge/autoencoder from
wandering off their pretrained weights during rollout training — a KL/PPO-style
penalty toward the frozen tokenizer. Spec'd here, implemented if drift training
destabilizes the tokenizer.

---

## 6. Scene representation (Jai's open question)

"Is a 2D image even the right, sparsest scene representation?" The sim already
exports both RGB **and** a sparse state vector per frame (`model/data` contract),
so the tokenizer's *input* is swappable:

- **RGB (default):** general, matches MineWorld/Oasis, net owns the pixels.
- **Geometric buffers** (depth/segmentation/normals): sparser, image-shaped so
  the same conv encoder works, and it's exactly the sim-anchor `λ` conditioning.
- **State vector:** tiny, but then the model learns dynamics not rendering —
  undercuts the "net dreams the frame" product thesis.

The contract keeps `frame` as the encoder input tensor; which buffer fills it is
a dataset flag, not an architecture change. This is the knob that lets the
question be answered empirically instead of assumed.

---

## 7. Build order (for the parallel tracks)

1. **Tokenizer** (`model/tokenizer/fsq_autoencoder.py`) — encode/quantize/decode,
   reconstruction loss. Verifiable by shape + round-trip on random tensors.
2. **Dynamics A** (`model/dynamics/ar_core.py`) — AR transformer, next-token
   train step, KV-cache infer stub. Verifiable by shape on tiny tensors.
3. **Data + loss** (`model/data/`, `eval/drift.py`) — Dataset over the sim
   manifest, the rollout loss above, drift metric vs the oracle.
4. **FSQ variants** (Task 2) and **Flow bridge B** (Branch B) — parallel
   experiments that plug into the same contract.

Each is CPU-shape-testable without a GPU; real training runs on Jai's machine.
