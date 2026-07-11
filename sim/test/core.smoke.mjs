import { SlowSim } from '../core/index.js';
import { terrainSurfaceHeight } from '../core/road.js';
import assert from 'node:assert/strict';

// Deterministic action script (no randomness in the test itself).
function drive(sim, n) {
  const frames = [];
  for (let i = 0; i < n; i++) {
    const steer = Math.sin(i * 0.05) * 0.6;
    const throttle = i < 5 ? 1 : 0.8;
    sim.step({ steer, throttle });
    const s = sim.state;
    frames.push([s.car.x, s.car.y, s.car.z, s.car.heading, s.car.speed]);
  }
  return frames;
}

// 1) Determinism: same seed + same actions -> identical trajectory.
const a = drive(new SlowSim({ seed: 7 }), 600);
const b = drive(new SlowSim({ seed: 7 }), 600);
const identical = JSON.stringify(a) === JSON.stringify(b);
console.log('determinism (seed 7 == seed 7):', identical);

// 2) Different seed -> different road.
const c = drive(new SlowSim({ seed: 8 }), 600);
console.log('seed 7 != seed 8:', JSON.stringify(a) !== JSON.stringify(c));

// 3) Sanity: car actually moves, stays finite, gains speed.
const last = a[a.length - 1];
console.log('final car [x,y,z,head,speed]:', last.map((v) => +v.toFixed(2)));
console.log('all finite:', a.every((f) => f.every(Number.isFinite)));

// 4) Snapshot round-trip continues identically.
const s1 = new SlowSim({ seed: 7 });
drive(s1, 300);
const snap = s1.snapshot();
const s2 = SlowSim.fromSnapshot(snap);
const tailA = drive(s1, 100);
const tailB = drive(s2, 100);
console.log('snapshot replay identical:', JSON.stringify(tailA) === JSON.stringify(tailB));

// 5) Dials ease smoothly (no NaN, monotone-ish toward target).
const sd = new SlowSim({ seed: 3 });
sd.setDials({ fog: 1.0, timeOfDay: 4.5 });
let prevFog = sd.state.dials.fog;
let monotone = true;
for (let i = 0; i < 120; i++) {
  sd.step({ steer: 0, throttle: 1 });
  const f = sd.state.dials.fog;
  if (f < prevFog - 1e-9) monotone = false;
  prevFog = f;
}
console.log('fog eased up monotonically to', +sd.state.dials.fog.toFixed(3), '(monotone:', monotone + ')');
console.log('props near car:', sd.props().length, 'dials keys:', Object.keys(sd.state.dials).length);

// 6) Gravity/contact regression: normal gravity stays planted on the engineered
// road profile, while low gravity loses contact at deterministic crests.
function countRoadLaunches(gravity, steps = 1200) {
  const sim = new SlowSim({ seed: 42, dt: 1 / 60, dials: { gravity } });
  let launches = 0;
  let wasGrounded = true;
  for (let i = 0; i < steps; i++) {
    const s = sim.state;
    const roadHeading = Math.atan2(s.road.tangent.x, s.road.tangent.z);
    const headingError = Math.atan2(
      Math.sin(roadHeading - s.car.heading),
      Math.cos(roadHeading - s.car.heading),
    );
    const steer = Math.max(-1, Math.min(1, headingError * 3.5 + s.road.offset * 0.18));
    const next = sim.step({ steer, throttle: 1 });
    if (wasGrounded && !next.car.grounded) launches++;
    wasGrounded = next.car.grounded;
  }
  return launches;
}

const normalGravityLaunches = countRoadLaunches(1);
const lowGravityLaunches = countRoadLaunches(0.2);
assert.equal(normalGravityLaunches, 0);
assert.ok(lowGravityLaunches > 0);
console.log('gravity contact launches (normal / low):', normalGravityLaunches, '/', lowGravityLaunches);

// 7) The deterministic core must cover the renderer's 680 m far band. Otherwise
// distant road/scenery samples collapse onto the final generated node.
const horizonSim = new SlowSim({ seed: 42 });
const farRoad = horizonSim.roadAhead(0, 680, 8);
assert.ok(horizonSim.road.length >= 680);
assert.ok(farRoad.at(-1).d >= 672);
assert.equal(new Set(farRoad.map((p) => p.d)).size, farRoad.length);
console.log('far horizon generated / unique samples:', horizonSim.road.length, '/', farRoad.length);

// 8) Vegetation bases use the exact same road-blended surface as physics/rendering.
const anchored = horizonSim.scatter(0, 220);
let maxAnchorError = 0;
for (const o of anchored) {
  const center = horizonSim.road.sampleAt(o.d);
  const expected = terrainSurfaceHeight(
    center, o.lateral, o.x, o.z, horizonSim.seed, horizonSim.state.dials.hilliness,
  );
  maxAnchorError = Math.max(maxAnchorError, Math.abs(o.y - expected));
}
assert.ok(maxAnchorError < 1e-9);
console.log('vegetation surface anchor max error:', maxAnchorError);
