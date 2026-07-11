// Option A: render the state model's dreamed trajectory through the REAL sim
// renderer, side by side with the true trajectory, into one GIF. Reuses the
// working state model (no pixel training) + the Playwright/WebGL path.
//
// Pipeline:
//   python -m eval.export_dream_poses --data data/seed1 --ckpt checkpoints/state_dynamics.pt
//   node sim/headless/render_dream.mjs [--poses data/dream_poses.json] [--size 256] [--out eval/plots/state_dream.gif]
//
// Needs playwright + chromium (see generate_pixels.mjs setup notes).

import { readFileSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import http from "node:http";
import { readFile } from "node:fs/promises";

const HERE = dirname(fileURLToPath(import.meta.url));
const SIM_DIR = join(HERE, "..");
const REPO_ROOT = join(HERE, "..", "..");

const args = parseArgs(process.argv.slice(2));
const POSES = args.poses ?? join(REPO_ROOT, "data", "dream_poses.json");
const SIZE = args.size ?? 256;
const OUT = args.out ?? join(REPO_ROOT, "eval", "plots", "state_dream.gif");

const data = JSON.parse(readFileSync(POSES, "utf8"));

// Headless rendering backend. Default = SwiftShader (CPU software GL): on a
// shared/busy GPU, sustained ReadPixels readbacks stall and the driver KILLS the
// WebGL context mid-capture (frames after that come back blank/gray). CPU
// rendering has no GPU contention, never loses the context, and — per the sim's
// own docs — GPU pixels aren't reproducible across hardware anyway, so CPU is
// the more correct choice for a training dataset. Set SLOWSIM_GL=gpu to try the
// hardware path on an idle box.
const USE_GPU = process.env.SLOWSIM_GL === "gpu";
const CHROME_GL_ARGS = USE_GPU
  ? ["--no-sandbox", "--use-gl=angle", "--use-angle=gl", "--ignore-gpu-blocklist", "--enable-gpu"]
  : ["--no-sandbox", "--use-gl=angle", "--use-angle=swiftshader"];

const { chromium } = await importPlaywright();
const server = await serveDir(SIM_DIR);
const port = server.address().port;
const browser = await chromium.launch({ args: CHROME_GL_ARGS });
const page = await browser.newPage();
page.on("console", (m) => console.log("[page]", m.text()));
await page.goto(`http://localhost:${port}/headless/dream_page.html`);
await page.waitForFunction("window.__ready === true");

console.log(`rendering ${data.steps} frames x2 (true + dream) at ${SIZE}px...`);
const trueFrames = await page.evaluate(
  ([s, p, sz]) => window.renderPoses(s, p, sz), [data.seed, data.true, SIZE]
);
const dreamFrames = await page.evaluate(
  ([s, p, sz]) => window.renderPoses(s, p, sz), [data.seed, data.dream, SIZE]
);
await browser.close();
server.close();

mkdirSync(dirname(OUT), { recursive: true });
await writeSideBySideGif(trueFrames, dreamFrames, SIZE, OUT);
console.log(`saved ${OUT}  (left = true drive, right = model's dreamed drive)`);

// --- helpers ---------------------------------------------------------------

async function importPlaywright() {
  try { return await import("playwright"); }
  catch {
    console.error("playwright not installed. On the GPU box:\n" +
      "  cd sim && npm install playwright && npx playwright install chromium");
    process.exit(1);
  }
}

async function writeSideBySideGif(left, right, size, out) {
  // GIF via gifenc (tiny, pure-JS, no native build). Optional dep — clear error
  // if absent rather than a fragile hand-rolled encoder.
  let gifenc;
  try { gifenc = await import("gifenc"); }
  catch {
    console.error("gifenc not installed. On the GPU box:  cd sim && npm install gifenc");
    process.exit(1);
  }
  // gifenc is CJS; under ESM import() its exports may sit on .default. Accept both.
  const g = gifenc.GIFEncoder ? gifenc : gifenc.default;
  const { GIFEncoder, quantize, applyPalette } = g;
  const gap = 4;
  const w = size * 2 + gap, h = size;
  const enc = GIFEncoder();
  for (let i = 0; i < left.length; i++) {
    const rgba = composite(left[i], right[i], size, gap);
    const pal = quantize(rgba, 256);
    const idx = applyPalette(rgba, pal);
    enc.writeFrame(idx, w, h, { palette: pal, delay: 100 });
  }
  enc.finish();
  writeFileSync(out, Buffer.from(enc.bytes()));
}

// Decode two base64 RGBA frames and lay them side by side (flip WebGL rows).
function composite(leftB64, rightB64, size, gap) {
  const L = Buffer.from(leftB64, "base64");
  const R = Buffer.from(rightB64, "base64");
  const w = size * 2 + gap, h = size;
  const out = new Uint8Array(w * h * 4).fill(255);
  const blit = (src, xoff) => {
    for (let y = 0; y < size; y++) {
      const sy = size - 1 - y; // WebGL bottom-left -> top-left
      for (let x = 0; x < size; x++) {
        const s = (sy * size + x) * 4;
        const d = (y * w + xoff + x) * 4;
        out[d] = src[s]; out[d + 1] = src[s + 1]; out[d + 2] = src[s + 2]; out[d + 3] = 255;
      }
    }
  };
  blit(L, 0);
  blit(R, size + gap);
  return out;
}

function serveDir(root) {
  const types = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript" };
  const srv = http.createServer(async (req, res) => {
    try {
      const p = join(root, decodeURIComponent(req.url.split("?")[0]));
      const body = await readFile(p);
      res.writeHead(200, { "content-type": types[p.slice(p.lastIndexOf("."))] || "application/octet-stream" });
      res.end(body);
    } catch { res.writeHead(404); res.end("not found"); }
  });
  return new Promise((resolve) => srv.listen(0, () => resolve(srv)));
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
