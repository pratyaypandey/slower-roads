// Road-following data-gen: a pursuit controller steers the car toward a point
// ahead on the centerline, so the car actually drives ON the road (unlike the
// open-loop scripted actions, which wander off it over time). This produces the
// on-road frames we want to render and train the pixel world model on. Still
// injects small steering noise so the action distribution isn't degenerate.
//
// Emits the same manifest schema as generate_state.js (state + road + actions),
// so the software renderer and dataloader consume it unchanged.
//
// Run: node headless/generate_drive.js [--seed N] [--steps N] [--out DIR]

import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { createSim, DT } from "../src/sim.js";
import { createPrng } from "../src/prng.js";

const REPO_ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const args = parseArgs(process.argv.slice(2));
const SEED = args.seed ?? 1;
const STEPS = args.steps ?? 2000;
const OUT = args.out ?? join(REPO_ROOT, "data", `seed${SEED}_drive`);

const sim = createSim(null, { seed: SEED });
const noise = createPrng("drive-noise");

// Pursuit controller: look ahead along the car's own arc length, steer toward
// the centerline point there. Progress `s` advances with distance travelled.
let s = 0;
const LOOKAHEAD = 12; // metres ahead to aim for

function followAction(state) {
  s += state.speed * DT;
  const target = sim.road.sample(s + LOOKAHEAD);
  // Heading error between where the car points and where the target is.
  const dx = target.x - state.x;
  const dz = target.z - state.z;
  const desired = Math.atan2(dx, dz);
  let err = desired - state.heading;
  while (err > Math.PI) err -= 2 * Math.PI;
  while (err < -Math.PI) err += 2 * Math.PI;
  const steer = Math.max(-1, Math.min(1, err * 1.5 + noise.signed(0.05)));
  return { throttle: 0.7 + noise.signed(0.15), brake: 0, steer };
}

// Seed the car onto the road start.
const start = sim.road.sample(0);
sim.world; // no-op; sim already positioned car at road start in createSim

const manifest = {
  seed: SEED,
  steps: STEPS,
  representation: "state",
  params: sim.params,
  dt: DT,
  road: {
    width: sim.road.width,
    points: sim.road.points.map((p) => [p.x, p.z]),
    headings: Array.from(sim.road.headings),
  },
  samples: [{ action: null, state: snap(sim.state) }],
};

for (let i = 0; i < STEPS; i++) {
  const action = followAction(sim.state);
  sim.step(action);
  manifest.samples.push({ action, state: snap(sim.state) });
}

mkdirSync(OUT, { recursive: true });
writeFileSync(join(OUT, "manifest.json"), JSON.stringify(manifest, null, 2));
console.log(`Wrote ${STEPS + 1} road-following samples to ${OUT} (seed ${SEED}).`);

function snap(s) {
  return { x: s.x, z: s.z, heading: s.heading, speed: s.speed };
}

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    if (argv[i].startsWith("--")) {
      out[argv[i].slice(2)] = /^\d+$/.test(argv[i + 1]) ? Number(argv[i + 1]) : argv[i + 1];
      i++;
    }
  }
  return out;
}
