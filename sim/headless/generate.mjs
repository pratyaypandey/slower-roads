// Headless data-gen for the ML pipeline. Drives the deterministic sim core (no
// renderer, no GPU) with the built-in autopilot so the car stays on the road,
// and writes a manifest with, per step: the action, the full state, the
// lambda-anchor skeleton, and the frame labels. This is the state+skeleton
// dataset — pixels come from the separate WebGL exporter (generate_pixels.mjs).
//
// Run: node sim/headless/generate.mjs [--seed N] [--steps N] [--out DIR]

import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { SlowSim } from "../core/index.js";

const REPO_ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const args = parseArgs(process.argv.slice(2));
const SEED = args.seed ?? 1;
const STEPS = args.steps ?? 2000;
const OUT = args.out ?? join(REPO_ROOT, "data", `seed${SEED}`);

const sim = new SlowSim({ seed: SEED });

// One sample per step: the state BEFORE the action, the action taken, and the
// oracle channels. A training tuple is (samples[i], samples[i+1]) — the action
// at i drives state i -> state i+1, matching the dt=1/30 fixed timestep.
const samples = [];
for (let i = 0; i < STEPS; i++) {
  const action = sim.autopilotAction();
  samples.push({
    action,
    state: sim.state,
    skeleton: sim.skeleton(),
    labels: sim.frameLabels(),
  });
  sim.step(action);
}
// Final state after the last action (so the last tuple has its next-state).
samples.push({ action: null, state: sim.state, skeleton: sim.skeleton(), labels: sim.frameLabels() });

const manifest = {
  seed: SEED,
  steps: STEPS,
  dt: sim.dt,
  representation: "state",   // no frames here; generate_pixels.mjs adds "rgb"
  samples,
};

mkdirSync(OUT, { recursive: true });
writeFileSync(join(OUT, "manifest.json"), JSON.stringify(manifest));
console.log(`Wrote ${samples.length} samples to ${OUT} (seed ${SEED}, headless, no renderer).`);

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
