// Pixel data-gen via the sim's real WebGL renderer, driven headlessly with
// Playwright (headless Chrome has a GPU-backed WebGL). Produces the same
// manifest as generate.mjs PLUS a `.npy` RGB frame per sample, so it's a drop-in
// "rgb" dataset for the tokenizer / AR + flow dynamics.
//
// Why a browser: the renderer needs WebGL (readRenderTargetPixels); there is no
// headless-GL path in the sim. Kept SEPARATE from generate.mjs so the headless
// state path never depends on a browser.
//
// Setup (on the GPU box):  cd sim && npm install
//   (package.json's optional dep on playwright; then `npx playwright install chromium`)
// Run:  node sim/headless/generate_pixels.mjs [--seed N] [--steps N] [--size N] [--out DIR]

import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { launchOptions, serveDir, writeNpyRGB } from "./lib.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const SIM_DIR = join(HERE, "..");           // served root (so ../vendor, ../core resolve)
const REPO_ROOT = join(HERE, "..", "..");

const args = parseArgs(process.argv.slice(2));
const SEED = args.seed ?? 1;
const STEPS = args.steps ?? 2000;
const SIZE = args.size ?? 64;
const OUT = args.out ?? join(REPO_ROOT, "data", `seed${SEED}`);

const { chromium } = await importPlaywright();
const server = await serveDir(SIM_DIR);
const port = server.address().port;

const browser = await chromium.launch(launchOptions());
const page = await browser.newPage();
page.on("console", (m) => console.log("[page]", m.text()));
await page.goto(`http://localhost:${port}/headless/capture_page.html`);
await page.waitForFunction("window.__ready === true");

console.log(`capturing ${STEPS} steps at ${SIZE}px (seed ${SEED})...`);
const result = await page.evaluate(
  ([s, n, sz]) => window.captureDrive(s, n, sz), [SEED, STEPS, SIZE]
);

await browser.close();
server.close();

// Write frames as (3, SIZE, SIZE) float32 .npy and build the manifest.
mkdirSync(join(OUT, "frames"), { recursive: true });
const samples = result.samples.map((s, i) => {
  const rel = join("frames", `${String(i).padStart(6, "0")}.npy`);
  writeNpyRGB(join(OUT, rel), s.rgba, SIZE);
  return { frame: rel, action: s.action, state: s.state, skeleton: s.skeleton, labels: s.labels };
});
const manifest = {
  seed: SEED, steps: STEPS, dt: result.dt, resolution: [SIZE, SIZE],
  representation: "rgb", renderer: "v2", samples,
};
writeFileSync(join(OUT, "manifest.json"), JSON.stringify(manifest));
console.log(`Wrote ${samples.length} frames + manifest to ${OUT} (${SIZE}px, seed ${SEED}).`);

// --- helpers ---------------------------------------------------------------

async function importPlaywright() {
  try {
    return await import("playwright");
  } catch {
    console.error(
      "playwright not installed. On the GPU box:\n" +
      "  cd sim && npm install playwright && npx playwright install chromium"
    );
    process.exit(1);
  }
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
