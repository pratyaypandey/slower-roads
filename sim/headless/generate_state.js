// State-only data-gen: exports (state, action, next_state, params) with NO
// renderer, so it runs anywhere Node runs — no gl, no native build. This is the
// sparse scene representation from the notes; useful for prototyping the dynamics
// core and dataloader before the pixel path (generate.js, needs gl) is available.
//
// Run: node headless/generate_state.js [--seed N] [--steps N] [--out DIR]

import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { createSim, DT } from "../src/sim.js";
import { makeActionSequence } from "./actions.js";

// Anchor the default output to <repo>/data regardless of the caller's cwd
// (this file is sim/headless/, so the repo root is two levels up).
const REPO_ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");

const args = parseArgs(process.argv.slice(2));
const SEED = args.seed ?? 1;
const STEPS = args.steps ?? 300;
const OUT = args.out ?? join(REPO_ROOT, "data", `seed${SEED}_state`);

const sim = createSim(null, { seed: SEED });
const actions = makeActionSequence(STEPS);

mkdirSync(OUT, { recursive: true });

const manifest = {
  seed: SEED,
  steps: STEPS,
  representation: "state",
  params: sim.params,
  dt: DT,
  // Road centerline + width so a renderer (e.g. the numpy software renderer)
  // can draw the scene from state alone, no JS/gl needed.
  road: {
    width: sim.road.width,
    points: sim.road.points.map((p) => [p.x, p.z]),
    headings: Array.from(sim.road.headings),
  },
  // Same tuple structure as the pixel manifest, minus frame paths: a training
  // tuple is (samples[i].state, samples[i].action, samples[i+1].state).
  samples: [{ action: null, state: snap(sim.state) }],
};

for (let i = 0; i < STEPS; i++) {
  const action = actions[i];
  sim.step(action);
  manifest.samples.push({ action, state: snap(sim.state) });
}

writeFileSync(join(OUT, "manifest.json"), JSON.stringify(manifest, null, 2));
console.log(`Wrote ${STEPS + 1} state samples to ${OUT} (seed ${SEED}, no renderer).`);

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
