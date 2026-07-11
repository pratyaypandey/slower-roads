// Oracle property proof: same seed + same actions => identical state trajectory.
// Runs two fully independent sims and asserts every state field matches exactly.
// This is state-only (no renderer), so it validates the deterministic core that
// everything downstream — the drift metric, the data exporter — relies on.
//
// Run: npm run prove

import { createSim } from "../src/sim.js";
import { makeActionSequence } from "./actions.js";

const SEED = 42;
const STEPS = 600; // 20 s at 30 Hz
const actions = makeActionSequence(STEPS);

function rollout() {
  const sim = createSim(null, { seed: SEED });
  const trajectory = [];
  for (let i = 0; i < STEPS; i++) {
    sim.step(actions[i]);
    const s = sim.state;
    trajectory.push([s.x, s.z, s.heading, s.speed]);
  }
  return trajectory;
}

const a = rollout();
const b = rollout();

let mismatches = 0;
let firstMismatch = null;
for (let i = 0; i < STEPS; i++) {
  for (let k = 0; k < 4; k++) {
    if (a[i][k] !== b[i][k]) {
      mismatches++;
      if (!firstMismatch) firstMismatch = { step: i, field: k, a: a[i][k], b: b[i][k] };
    }
  }
}

// Control: the sim must not be inert. The car is open-loop (it follows steering
// input, not the road — seed drives visuals, not physics), so we check the
// trajectory is non-trivial: the car covered real distance and steering actually
// bent its heading. Seed-sensitivity lives in the rendered frames and is proven
// in the pixel path once a renderer is wired.
const last = a[STEPS - 1];
const dist = Math.hypot(last[0] - a[0][0], last[1] - a[0][1]);
const headingSpread = a.reduce(
  (m, s) => ({ lo: Math.min(m.lo, s[2]), hi: Math.max(m.hi, s[2]) }),
  { lo: Infinity, hi: -Infinity }
);
const turned = headingSpread.hi - headingSpread.lo;
const nonTrivial = dist > 10 && turned > 0.1;

const finalPose = `x=${last[0].toFixed(3)} z=${last[1].toFixed(3)} heading=${last[2].toFixed(3)} speed=${last[3].toFixed(3)}`;

if (mismatches === 0 && nonTrivial) {
  console.log(`PASS  ${STEPS} steps, seed ${SEED}: trajectories bit-identical.`);
  console.log(`      final pose: ${finalPose}`);
  console.log(`      control: drove ${dist.toFixed(1)} m, heading swept ${turned.toFixed(2)} rad.`);
  process.exit(0);
} else {
  if (mismatches > 0) {
    console.error(`FAIL  ${mismatches} field mismatches. First:`, firstMismatch);
  }
  if (!nonTrivial) {
    console.error(`FAIL  trajectory looks inert: dist=${dist.toFixed(1)}m turned=${turned.toFixed(2)}rad.`);
  }
  process.exit(1);
}
