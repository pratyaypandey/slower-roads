# sim/

Deterministic Three.js driving sim + oracle harness (ROADMAP milestone M0).

Same core runs three ways: a **state-only** loop (no renderer, for the
determinism proof and any dynamics that never need pixels), a **headless**
renderer (data-gen), and the **browser** (live WASD driving). All three share
`src/` unchanged, so the frames the model trains on match what a player sees.

## Layout

```
src/         renderer-agnostic core (no three import except world.js, which takes it injected)
  prng.js      Alea seeded PRNG — the determinism backbone
  road.js      procedural centerline from seed
  car.js       fixed-timestep car physics (DT = 1/30)
  params.js    gravity / friction / weather / timeOfDay dials
  world.js     Three.js scene assembly (THREE passed in, never imported)
  sim.js       top-level step()/render() loop; THREE optional
headless/    Node data-gen + oracle harness
  actions.js         reproducible scripted action sequence
  prove_determinism.js  same seed+actions => identical trajectory
  renderer.js        headless-gl WebGLRenderer + pixel readback
  generate.js        exports (frame, action, next_frame, params) tuples
browser/     index.html + main.js — live WASD driving via vite
```

## Run

First, install deps (needs a normal network — the npm registry must be
reachable; `gl` is a native module and compiles on install):

```
cd sim && npm install
```

**Determinism proof (no deps needed — pure Node):**
```
npm run prove
```
Expected: `PASS ... trajectories bit-identical.`

**Live driving in the browser:**
```
npm run dev        # vite serves browser/; open the printed URL, drive with WASD
```

**Generate a dataset:**
```
npm run gen -- --seed 1 --steps 300 --res 128x128
# writes ../data/seed1/frames/*.png + manifest.json
```

## Known unknown

`gl` (headless-gl) is native and Node 26 is new — if `npm install` fails to
build it, that only blocks `npm run gen` (headless pixel export). `npm run prove`
and `npm run dev` are unaffected. Report the build error and we'll pin a Node
version or swap the headless backend.

## Data contract

Each `manifest.json` sample carries **both** the RGB frame path and the sparse
`state` vector `{x, z, heading, speed}` from the same deterministic step. A
training tuple is `(samples[i].frame, samples[i].action, samples[i+1].frame)`.
Emitting both views lets the tokenizer be trained on pixels, on the sparse
state, or on both — the "is a 2D image even the right scene representation?"
question stays open rather than hardcoded to pixels.

## Design note: the car is open-loop

The car obeys steering input and treats the road as scenery, not a rail (like
Slow Roads). So **the seed changes the road and the rendered frames, not the
car's state trajectory** — seed-sensitivity is a pixel-level property, proven in
the data-gen path, not in the state-only proof.
