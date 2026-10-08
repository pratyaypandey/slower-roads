# sim/ — public API contract

The sim is split into a **deterministic core** (pure JS, no rendering) and an **RGB
renderer** (Three.js). The demo (and later the data exporter and λ-anchor) consume these.
Design rationale lives in `../plans/SIM.md`.

## Module layout

```
sim/
  core/index.js      # SlowSim + dial schema (no Three.js dependency)
  render/renderer.js # SimRenderer (imports the bare specifier "three")
  render/shaders.js  # GLSL: shared light/haze model, terrain, cards, sky, water, car
  render/textures.js # seeded procedural textures (noise, leaf/conifer/grass cards)
  render/car.js      # the white hatchback (geometry solved from reference footage)
  vendor/three.module.js  # vendored Three.js r160 (+ jsm/utils/BufferGeometryUtils.js)
```

Because `render/renderer.js` imports `"three"`, any HTML page loading it needs an
**import map** resolving that bare specifier to the vendored file, e.g.:

```html
<script type="importmap">
{ "imports": { "three": "./vendor/three.module.js" } }
</script>
```

(Path is relative to the HTML file. From `sim/demo/index.html` that is `../vendor/three.module.js`.)

## Core — `SlowSim`

```js
import { SlowSim, DIAL_SCHEMA, DIAL_KEYS } from '../core/index.js';

const sim = new SlowSim({ seed: 42, dt: 1/30 });   // dt is the FIXED timestep
```

| Member | Description |
|---|---|
| `new SlowSim({ seed, dt, dials })` | `dials` optionally overrides defaults. |
| `sim.step(action)` | Advance one fixed tick. `action = { steer:[-1,1], throttle:[-1,1] }`. Returns `sim.state`. Call once per rendered frame. |
| `sim.state` | Snapshot: `{ t, step, seed, car:{x,y,z,heading,speed,slip,vy,grounded}, road:{d,offset,center,tangent}, dials:{...}, action }`. `road.offset` is the signed lateral distance (m) of the car from the centerline (+ = right of travel). `road.center` is the nearest centerline sample `{ d, x, y, z, heading, curvature, width }` — `width` is the full road width, so `abs(offset) > width/2` means off-road. `road.tangent` is the unit travel direction `{x, z}`. |
| `sim.setDials(partial)` | Set dial **targets**; values ease smoothly toward them (no pops). |
| `sim.snapDials(partial)` | Set dials instantly (skip easing). |
| `sim.reset(seed?, dials?)` | Reseed and restart deterministically. |
| `sim.roadAhead(back, ahead, spacing)` | Centerline samples (used by renderer; demo usually won't need it). |
| `sim.props(back, ahead)` | Deterministic roadside props near the car. |
| `sim.snapshot()` / `SlowSim.fromSnapshot(o)` | Exact serialize / restore. |

**Dials** (`DIAL_SCHEMA` gives `{min,max,default,smooth,wrap}` per key):
`timeOfDay` (0–2π, wraps), `fog`, `rain`, `snow`, `biomeX`, `biomeY`, `starDensity`,
`moons` (0–3), `aurora`, `curveAmp`, `hilliness`, `gravity`, `friction`, `speedFeel`.

Iterate `DIAL_KEYS` + `DIAL_SCHEMA` to build UI sliders generically — do **not** hardcode
the list, so new dials appear automatically.

## Renderer — `SimRenderer`

The v2 engine, rebuilt clean-room to match recorded Slow Roads footage
(`../docs/FIDELITY.md`). Every material is a custom shader authored in display
colours with no tone mapping or post-processing, so what the canvas shows and what
`capture()` returns are the same image.

```js
import { SimRenderer } from '../render/renderer.js';

const r = new SimRenderer(canvasEl, { dataSize: 64 });
r.setDisplaySize(canvas.clientWidth, canvas.clientHeight); // CSS px of the display
r.stepCamera(sim, sim.dt);             // once per fixed sim step (deterministic chase cam)
r.render(sim, { alpha, prevCar });     // draw; interp optional (smooth display between steps)
```

| Member | Description |
|---|---|
| `new SimRenderer(canvas, { dataSize })` | Full-resolution renderer; `dataSize` is the default `capture()` size. |
| `r.stepCamera(sim, dt)` | Advance the chase camera one fixed step (call once per `sim.step`). Near-rigid mount, slight yaw lag. |
| `r.render(sim, interp?)` | Update + draw one frame. `interp = { alpha, prevCar }` interpolates between fixed steps. |
| `r.capture(size?)` | Square RGBA `Uint8Array` (bottom-left origin): the display image at aspect 1, supersampled, box-filtered, then softened slightly to match the reference. |
| `r.captureChannels(size?)` | `{ rgb, depth, size }`: `capture()` plus a packed depth map. |
| `r.setDisplaySize(w, h)` / `r.setQuality('low'\|'medium'\|'high')` / `r.setExposure(v)` | Display size, pixel-ratio tier, global light scale. |
| `r.resetPresentation()` | Drop camera/terrain/vegetation caches (after `sim.reset`). |
| `r.stats()` | `{ calls, triangles, instances }` for a diagnostics overlay. |

What it draws is the sim's own world, so the oracle state in a dataset describes
exactly what is on screen. The road, heightfield, hillsides and tree/rock/bush/grass
scatter come from `sim/core`. Verge grass and guardrails are render-only decoration,
deterministic in seed + arc-length; guardrails have no collision. Dials drive the
look: time of day, fog, rain, snow, biome tint, stars.

## Minimal loop (what the demo wires up)

```js
const sim = new SlowSim({ seed: 42 });
const r = new SimRenderer(canvas);
const keys = {};
addEventListener('keydown', e => keys[e.key.toLowerCase()] = true);
addEventListener('keyup',   e => keys[e.key.toLowerCase()] = false);

function frame() {
  const steer = (keys['d'] || keys['arrowright'] ? 1 : 0) - (keys['a'] || keys['arrowleft'] ? 1 : 0);
  const throttle = (keys['w'] || keys['arrowup'] ? 1 : 0) - (keys['s'] || keys['arrowdown'] ? 1 : 0);
  sim.step({ steer, throttle });
  r.render(sim);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
```

Note: `sim.step` uses a fixed `dt`; for a first version, stepping once per animation
frame is fine. (A fixed-timestep accumulator can come later for frame-rate independence.)
