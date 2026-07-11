// Data-gen: roll the sim forward and export (frame, action, next_frame, params)
// tuples. Following the scene-representation idea, each sample carries both the
// RGB frame and the sparse state vector from the same deterministic step, so the
// tokenizer can later be trained on whichever representation wins — pixels are
// one view, not the only one.
//
// Run: npm run gen -- [--seed N] [--steps N] [--res WxH] [--out DIR]

import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { PNG } from "pngjs";
import { createHeadlessRenderer } from "./renderer.js";
import { createSim } from "../src/sim.js";
import { makeActionSequence } from "./actions.js";

// Anchor default output to <repo>/data regardless of the caller's cwd.
const REPO_ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");

const args = parseArgs(process.argv.slice(2));
const SEED = args.seed ?? 1;
const STEPS = args.steps ?? 300;
const [W, H] = (args.res ?? "128x128").split("x").map(Number);
const OUT = args.out ?? join(REPO_ROOT, "data", `seed${SEED}`);

const { renderer, readPixels } = createHeadlessRenderer(W, H);
const THREE = (await import("three")).default ?? (await import("three"));
const sim = createSim(THREE, { seed: SEED });
const actions = makeActionSequence(STEPS);

mkdirSync(join(OUT, "frames"), { recursive: true });

const manifest = {
  seed: SEED,
  steps: STEPS,
  resolution: [W, H],
  params: sim.params,
  dt: 1 / 30,
  // Each entry pairs frame i with the action taken and the resulting state, so a
  // (frame, action, next_frame) tuple is (samples[i].frame, samples[i].action,
  // samples[i+1].frame). state is the sparse scene representation.
  samples: [],
};

// Frame 0: the initial observation before any action.
sim.render(renderer);
writeFrame(readPixels(), join(OUT, "frames", "000000.png"), W, H);
manifest.samples.push({ frame: "frames/000000.png", action: null, state: snap(sim.state) });

for (let i = 0; i < STEPS; i++) {
  const action = actions[i];
  sim.step(action);
  sim.render(renderer);
  const name = String(i + 1).padStart(6, "0") + ".png";
  writeFrame(readPixels(), join(OUT, "frames", name), W, H);
  manifest.samples.push({ frame: `frames/${name}`, action, state: snap(sim.state) });
}

writeFileSync(join(OUT, "manifest.json"), JSON.stringify(manifest, null, 2));
console.log(`Wrote ${STEPS + 1} frames + manifest to ${OUT} (${W}x${H}, seed ${SEED}).`);

function snap(s) {
  return { x: s.x, z: s.z, heading: s.heading, speed: s.speed };
}

function writeFrame(rgba, path, w, h) {
  const png = new PNG({ width: w, height: h });
  png.data = Buffer.from(rgba.buffer, rgba.byteOffset, rgba.byteLength);
  writeFileSync(path, PNG.sync.write(png));
}

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    if (argv[i].startsWith("--")) {
      const key = argv[i].slice(2);
      const val = argv[i + 1];
      out[key] = /^\d+$/.test(val) ? Number(val) : val;
      i++;
    }
  }
  return out;
}
