// Batch dataset generation: many episodes of rendered driving under a mix of driving
// policies (core/policies.js), one dataset dir per episode in the generate_pixels.mjs
// manifest format (plus `keys` per sample and a `policy` field), and an index.
//
// Resumable: episodes whose manifest exists are skipped, so re-running continues.
// Each episode is a fresh world (seed). A lost WebGL context restarts the browser and
// retries the episode.
//
// Run (local GPU):
//   SLOWSIM_CHANNEL=chrome SLOWSIM_GL=gpu node sim/headless/generate_dataset.mjs \
//       --hours 3 --out data/train_v2 [--steps 3600] [--size 64] [--start-seed 10000]
//
// Default mix (fraction of episodes): cruise .25, keys_lane .25, keys_explore .25,
// lane_change .12, dial_mix .13. See core/policies.js for what each profile does.

import { mkdirSync, writeFileSync, existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { launchOptions, serveDir, writeNpyRGB } from "./lib.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const SIM_DIR = join(HERE, "..");
const REPO_ROOT = join(HERE, "..", "..");

const args = parseArgs(process.argv.slice(2));
const STEPS = Number(args.steps ?? 3600);           // 2 min at 30 fps
const SIZE = Number(args.size ?? 64);
const OUT = args.out ?? join(REPO_ROOT, "data", "train_v2");
const START_SEED = Number(args["start-seed"] ?? 10000);
const DT = 1 / 30;
const EPISODES = Number(args.episodes ?? Math.round((Number(args.hours ?? 3) * 3600) / (STEPS * DT)));
const MIX = [["cruise", 0.25], ["keys_lane", 0.25], ["keys_explore", 0.25], ["lane_change", 0.12], ["dial_mix", 0.13]];

// Deterministic plan: interleave profiles so any prefix of the run is balanced.
const plan = [];
const counts = Object.fromEntries(MIX.map(([p]) => [p, 0]));
for (let i = 0; i < EPISODES; i++) {
  // pick the profile furthest below its target share
  let best = null, gap = -Infinity;
  for (const [p, f] of MIX) { const g = f * (i + 1) - counts[p]; if (g > gap) { gap = g; best = p; } }
  counts[best]++;
  const seed = START_SEED + i;
  plan.push({ i, policy: best, seed, policySeed: (seed * 7919 + 17) >>> 0, dir: `ep${String(i).padStart(4, "0")}_${best}_s${seed}` });
}

const { chromium } = await import("playwright");
const server = await serveDir(SIM_DIR);
const url = `http://localhost:${server.address().port}/headless/capture_page.html`;
let browser = null, page = null;
async function openPage() {
  if (browser) await browser.close().catch(() => {});
  browser = await chromium.launch(launchOptions());
  page = await browser.newPage();
  page.on("pageerror", (e) => console.error("[page error]", e.message));
  await page.goto(url);
  await page.waitForFunction("window.__ready === true");
}

mkdirSync(OUT, { recursive: true });
console.log(`${EPISODES} episodes x ${STEPS} steps (${(EPISODES * STEPS * DT / 3600).toFixed(2)} h) at ${SIZE}px -> ${OUT}`);
console.log("mix:", JSON.stringify(counts));
const t0 = Date.now();
let done = 0;
for (const ep of plan) {
  const dir = join(OUT, ep.dir);
  if (existsSync(join(dir, "manifest.json"))) { done++; continue; }
  let result = null;
  for (let attempt = 0; attempt < 3 && !result; attempt++) {
    try {
      if (!page) await openPage();
      result = await page.evaluate(([s, n, sz, o]) => window.captureDrive(s, n, sz, o),
        [ep.seed, STEPS, SIZE, { policy: ep.policy, policySeed: ep.policySeed }]);
    } catch (e) {
      console.error(`episode ${ep.i} attempt ${attempt + 1} failed: ${e.message.split("\n")[0]}`);
      page = null;
    }
  }
  if (!result) { console.error(`episode ${ep.i} skipped after 3 attempts`); continue; }
  mkdirSync(join(dir, "frames"), { recursive: true });
  let off = 0, speed = 0;
  const samples = result.samples.map((s, k) => {
    const rel = join("frames", `${String(k).padStart(6, "0")}.npy`);
    writeNpyRGB(join(dir, rel), s.rgba, SIZE);
    if (Math.abs(s.state.road.offset) > s.state.road.center.width / 2) off++;
    speed += s.state.car.speed;
    return { frame: rel, action: s.action, keys: s.keys, state: s.state, skeleton: s.skeleton, labels: s.labels };
  });
  const stats = { offroad: off / samples.length, meanSpeed: speed / samples.length };
  writeFileSync(join(dir, "manifest.json"), JSON.stringify({
    seed: ep.seed, steps: STEPS, dt: result.dt, resolution: [SIZE, SIZE], representation: "rgb",
    renderer: "v2", policy: ep.policy, policySeed: ep.policySeed, episode: ep.i, stats, samples,
  }));
  done++;
  const el = (Date.now() - t0) / 1000;
  console.log(`[${done}/${EPISODES}] ${ep.dir} offroad ${(100 * stats.offroad).toFixed(0)}% speed ${stats.meanSpeed.toFixed(1)} | ${el.toFixed(0)}s elapsed`);
}
if (browser) await browser.close();
server.close();

// Index of every finished episode (resumed ones included).
const index = plan.filter((ep) => existsSync(join(OUT, ep.dir, "manifest.json"))).map((ep) => {
  const m = JSON.parse(readFileSync(join(OUT, ep.dir, "manifest.json"), "utf8"));
  return { dir: ep.dir, policy: ep.policy, seed: ep.seed, frames: m.samples.length, ...m.stats };
});
writeFileSync(join(OUT, "index.json"), JSON.stringify({ steps: STEPS, size: SIZE, dt: DT, episodes: index }, null, 1));
console.log(`index: ${index.length} episodes, ${(index.reduce((s, e) => s + e.frames, 0) * DT / 3600).toFixed(2)} h -> ${join(OUT, "index.json")}`);

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i += 2) out[argv[i].replace(/^--/, "")] = argv[i + 1];
  return out;
}
