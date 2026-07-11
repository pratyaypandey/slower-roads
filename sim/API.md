# sim/ — public API contract

The sim is split into a **deterministic core** (pure JS, no rendering) and an **RGB
renderer** (Three.js). The demo (and later the data exporter and λ-anchor) consume these.
Design rationale lives in `../plans/SIM.md`.

## Module layout

```
sim/
  core/index.js      # SlowSim + dial schema (no Three.js dependency)
  render/renderer.js # SimRenderer (imports the bare specifier "three")
  vendor/three.module.js  # vendored Three.js r160
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

```js
import { SimRenderer } from '../render/renderer.js';

const r = new SimRenderer(canvasEl);   // canvas is a normal <canvas>
r.setDisplaySize(canvas.clientWidth, canvas.clientHeight); // CSS px of the display
r.render(sim);                         // draws sim.state; call every frame
```

| Member | Description |
|---|---|
| `new SimRenderer(canvas, opts?)` | Renders internally at low res (default 160×128) then posterizes to the canvas. |
| `r.render(sim)` | Update + draw one frame from `sim.state`. |
| `r.setDisplaySize(w, h)` | Set the display backing-store size (call on resize). Internal res is fixed. |
| `r.setPosterize({ levels, softness, saturation })` | Tune the band count / edge softness / saturation live. |
| `r.capture()` | Returns the low-res RGBA `Uint8Array` (the training-data RGB head; demo may ignore). |

The canvas is upscaled from the low internal resolution with nearest-neighbour, so the
display should use `image-rendering: pixelated` for a crisp dream look.

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
