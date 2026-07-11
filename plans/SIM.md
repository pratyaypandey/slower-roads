# SIM.md — The Deterministic Driving Sim (M0)

*The oracle. A minimal, deterministic Three.js driving world that we distill into the neural
model. It is simultaneously: (1) the training-data generator, (2) the ground-truth eval oracle,
(3) the source of clean contrastive pairs for steering, and (4) the λ-anchor skeleton that runs
live alongside the net at inference. Every design choice below serves at least one of those roles.*

See `../ROADMAP.md` for how this feeds the rest of the system. This doc is the M0 spec.

---

## 0. The core bet

The instinct is to treat "surreal and cool" as a creative goal we pay for in performance and
model-friendliness. That's backwards. Every downstream stage wants the *same* thing the surreal
look already gives us:

- **Autoencoder (M1)** wants **low-entropy** frames — big flat color regions, little high-freq
  detail — so a ~16×16 latent reconstructs cleanly.
- **Real-time decode (M4)** wants the same: less to represent → fewer steps → higher fps.
- **CAA + SAE steering (M5/M6)** want the factors of variation to be **smooth, independent, and
  roughly linear in pixel space**, so `h ← h + α·v` is a clean linear edit.

A **stylized dream** look — flat/toon shading, posterized/banded color, bold silhouettes, heavy
fog, low-poly, gradient skies — *is* that low-entropy, smooth, compressible signal. Photorealism
(per-blade grass, PBR, high-freq texture) fights all four stages at once.

> **We don't trade beauty against feasibility. We pick a visual language where the coolest look
> and the most tractable signal are the same choice.** Leaning *harder* into stylization makes
> the research easier, not just prettier.

This also structurally defeats the roadmap's #1 silent risk (scope creep toward fidelity):
"ugly-realistic" is off the table by construction.

---

## 1. Four load-bearing design principles

**1. The world is a *product of orthogonal dials*, not a set of scenes.**
A small number (~6–10) of **independent, continuous** generative factors multiply into a
combinatorially huge surreal space. This resolves light-weight-vs-expressive: tiny asset count
(great for client-side + compression), enormous world (great for expressive). It also hands
M5/M6 clean contrastive pairs for free — render the same seed at two fog values and the label
*is* the function argument.

**2. At data-gen time, sample the *product* — including impossible combos.**
Sunny + heavy fog + three moons + low gravity. Slow Roads can't do "midnight at noon"; our net
can — *because* we train on the independent product rather than natural co-occurrence.
Independent sampling is precisely what makes the steering vectors disentangle.

**3. Every dial's param→pixel map must be smooth and monotonic.**
No discrete pops (a light snapping on). Fog density blends smoothly toward fog color; time-of-day
smoothly rotates the sun and lerps the sky gradient. Smoothness of param→pixel ⇒ linearity of the
latent direction ⇒ CAA/steering actually works. This is a real authoring constraint on each dial.

**4. One deterministic core, two output heads.**
The same seeded state produces:
- **RGB + aux head** — pretty posterized frames + auxiliary channels, for training.
- **Skeleton head** — headless geometry only (road centerline, car pose, coarse depth, seg mask),
  for the **λ anchor** at inference. Must be cheap enough to run *alongside* the net at 30fps.

The aux channels are free from a sim and impossible from YouTube — that's the moat, restated at
the data level.

---

## 2. Visual style (committed: "Slow Roads polish")

**Render resolution is decoupled from model resolution.** This was the key correction: the
autoencoder trains on 128² frames, but *humans* never look at 128² frames. So the renderer draws
at **full display resolution**, and only the data-export path (`renderer.capture(size)`)
downsamples to the small square frame the model sees. Rendering small was a mistake — it made the
sim look broken and had nothing to do with the model's needs.

Look, targeting [slowroads.io](https://slowroads.io):

- **Full-res, antialiased (MSAA), filmic tone mapping** (ACES). Crisp, not blocky.
- **Soft aerial haze.** Exponential fog whose colour tracks the horizon → distant terrain
  desaturates into the sky. This is *the* signature depth cue and the biggest single "it looks
  real/atmospheric" lever. Also a clean steerable scalar (the `fog` dial).
- **Physically-based sky** (Three's Preetham `Sky`) driven by the sun position from `timeOfDay`
  → gradient sky, sun glow, horizon reddening at dusk, all for free.
- **Smooth-shaded rolling terrain** over a *single shared heightfield*: the road drapes over the
  field (its centerline y = H at the centerline) and off-road terrain eases to the same field, so
  road and ground **meet smoothly instead of forming a trench**. Muted, natural, slightly
  desaturated palette (greens/tans/mauves per biome).
- **Clean marked road** as the λ-anchor — the most legible element, asphalt with a centre line.
- **Implemented machinery** (to escape the "programmer-primitive" look): PCF soft **shadows**
  (sun-driven, frustum follows the car); an **EffectComposer** post stack (bloom → vignette →
  ACES OutputPass, MSAA); **dense InstancedMesh vegetation** (pines / broadleaf / bushes / rocks /
  grass, scattered road-relative so it never lands on the road, density + species from the biome
  dials); terrain shaded by **slope + altitude** (rock on steep, snow up high); soft **billboard
  clouds**; a translucent **water plane at a global sea level** with the road lifted into a
  **causeway** across low ground; and a **car with wheels** that spin/steer and a chassis that
  tilts to the terrain normal (sampled fore/aft + left/right).
- **Surfaces have friction.** The physics reads what's under the car (`core/car.js` `SURFACE`):
  tarmac is fast and planted, grass/sand/snow drag you down and get loose. Combined with tree
  collision, leaving the road is a real (self-correcting) cost — a gentle incentive to stay on it.
  Top speed is tuned to a relaxed cruise (~18 m/s), not a racer.
- **Vegetation is core-owned, not decoration.** `core/scatter.js` is the single deterministic
  source of trees/rocks/etc.; the renderer *draws* it and the physics *collides* with it (car is
  pushed out of trunks off-road; the road corridor stays clear). This matters beyond looks: the
  world now has solid obstacles that are part of the sim state, which the world model must
  eventually learn. Scatter + terrain are rebuilt per-frame — the perf lever to watch at M4.
- **Posterize is demoted to an optional grade,** not the identity. `render/posterize.js` still
  exists and can be applied *in the data pipeline* (downsample → optional gentle band) if we ever
  want lower-entropy training frames — but it must be *temporally stable* (soft-posterize with
  smoothed band edges) or it injects fake high-frequency flicker the dynamics model wastes
  capacity on. The autoencoder still gets its low-entropy target simply because 128² of a
  stylized scene *is* low-entropy; we don't need blocky banding to get there.

---

## 3. The dials (steering targets)

Orthogonal, continuous, smooth param→pixel. Expressiveness comes from their **continuous
product**, not from adding more factors — a small count is what gives the SAE a shot at recovering
monosemantic features and keeps contrastive extraction clean.

| Dial | Scalar(s) | Drives | Milestone |
|---|---|---|---|
| **Time-of-day** | θ ∈ [0, 2π) | sun position, sky gradient, ambient + shadow color | M5 |
| **Fog** | density ∈ [0,1] | whole-frame fog blend, draw distance | M5 (first dial) |
| **Precipitation** | rain, snow ∈ [0,1] | particle overlay, ground wetness/whiteness | M5 |
| **Biome** | 2D vector | ground palette, terrain amplitude, flora type/density | M5 |
| **Sky / atmosphere** | palette, star density, #suns/moons, aurora | backdrop | M5 |
| **Road geometry** | curve freq, hilliness amp | track shape | M5 |
| **Physics** | gravity, friction, speed-feel | car dynamics (the spicy dials) | **M6** |

The physics dials are reserved for M6 (the stretch interp paper) but the sim exposes them from
day one so the data exists.

---

## 4. Actions — continuous control vector (committed)

The action is a **conditioning input** to the AR dynamics core, not a prediction target (the
*player* is the action source — we never model `P(action)`).

- **Representation:** continuous `(steer, throttle) ∈ [-1,1]²`, fed via a small linear/MLP
  projection into a control vector. **No quantization.**
- **Why continuous, not binned:** nearby values → nearby embeddings *for free*. That
  smoothness/ordering prior matches the physics (steering response is smooth), is more
  sample-efficient, and generalizes to values never seen in training. Fine-grained bins (e.g.
  256-way) are the worst of both worlds: continuous-in-spirit but stripped of the ordering prior,
  with hundreds of sparse, jaggedly-learned embeddings.
- **The one contingency:** if the M2 dynamics core commits to a strictly **uniform token stream**
  (actions interleaved as tokens, MineWorld-style), actions would want to be discrete tokens to
  live in that stream. So this default is *explicitly contingent on the M2 architecture call* —
  continuous wins as long as actions enter as a conditioning vector (the roadmap's "control
  vector"), which is the current plan.

The sim records the raw float regardless — data-gen never coarsens the input.

---

## 5. Data specification

**Per frame:**

```
RGB      : 128×128, posterized
action   : (steer, throttle) ∈ [-1,1]²   (raw float)
params   : the full dial vector at this frame
aux      : { depth (coarse), seg_mask (road/ground/sky/obstacle),
             centerline_offset, car_pose }
```

**Episode** = `seed + param_schedule + action_sequence`. Three flavors:

1. **Static-param** — dials held constant for the whole episode. Source of **clean CAA
   contrastive pairs** (same seed + same actions, one param differs → the label is the function
   arg).
2. **Natural** — dials set to believable, co-occurring values. Teaches sensible defaults.
3. **Sweeping** — dials move continuously mid-drive. Teaches morph *dynamics* (how the world
   transitions when a knob turns), not just static settings.

**Endless road:** spline generated from `seed + distance` via seeded noise, chunked ahead as the
car advances. Infinite variety, infinite labeled data.

---

## 6. Determinism (non-negotiable — the oracle depends on it)

- **Fixed timestep physics.** Never delta-time. Integer step counts, no wall-clock.
- **Single injected seeded PRNG** everywhere. No `Math.random()`, no `Date.now()`. Same seed +
  same action sequence → identical state trajectory.
- **CPU state is the true oracle.** GPU float differences make *rendered pixels* non-portable
  across hardware even when *state* is portable. So: the deterministic guarantee is on CPU state
  (car pose, params, spline); rendered-frame eval must be **pinned to one fixed renderer**.
- **Oracle replay harness:** given `seed + action_sequence`, regenerate the exact ground-truth
  future for frame-by-frame diffing against a neural rollout.

---

## 7. Two output heads, concretely

| | RGB + aux head | Skeleton head |
|---|---|---|
| **Purpose** | training data | live λ-anchor at inference |
| **Output** | posterized 128px RGB + aux channels | centerline, car pose, coarse depth, seg mask |
| **When** | offline data-gen (batch, as fast as GPU allows) | real-time, in-browser, beside the net |
| **Budget** | throughput | must fit in the 30fps frame budget alongside the model |

Same deterministic core; the heads differ only in what they emit.

---

## 8. M0 build order (from the roadmap)

1. Deterministic seeded core: road spline + car physics + fixed-timestep loop.
2. Continuous `(steer, throttle)` control loop (WASD → continuous, on-screen for data-gen).
3. Low-res renderer + posterize post-pass (verify temporal stability early).
4. Expose all dials as explicit params (incl. physics dials, reserved for M6).
5. Data exporter: `(frame, action, next_frame, params, aux)` to disk.
6. Oracle replay harness: `seed + actions → ground-truth frames`.
7. **Drift metric** (latent + pixel divergence vs. rollout length) against the oracle.

**Done when:** we can generate an arbitrary labeled dataset and, given any action sequence,
produce the exact ground-truth future to compare against.

---

## 9. The through-line to keep honest

Every choice pushes toward **low entropy + orthogonal factors + smooth maps.** When something
tempts us toward fidelity, that temptation is the tell that it's wrong for this project. A clean
live demo is a great figure 1 and a terrible reason to spend three weeks on shaders.
