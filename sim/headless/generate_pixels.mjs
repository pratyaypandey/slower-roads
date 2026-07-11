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
import http from "node:http";
import { readFile } from "node:fs/promises";

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

// CPU (SwiftShader) by default — a shared GPU stalls on sustained ReadPixels and
// kills the context mid-capture (see render_dream.mjs). SLOWSIM_GL=gpu opts into
// hardware on an idle box.
const GL_ARGS = process.env.SLOWSIM_GL === "gpu"
  ? ["--no-sandbox", "--use-gl=angle", "--use-angle=gl", "--ignore-gpu-blocklist", "--enable-gpu"]
  : ["--no-sandbox", "--use-gl=angle", "--use-angle=swiftshader"];
const browser = await chromium.launch({ args: GL_ARGS });
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
  representation: "rgb", samples,
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

// Minimal static server for the sim dir (serves core/, render/, vendor/, headless/).
function serveDir(root) {
  const types = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript" };
  const srv = http.createServer(async (req, res) => {
    try {
      const p = join(root, decodeURIComponent(req.url.split("?")[0]));
      const body = await readFile(p);
      const ext = p.slice(p.lastIndexOf("."));
      res.writeHead(200, { "content-type": types[ext] || "application/octet-stream" });
      res.end(body);
    } catch {
      res.writeHead(404); res.end("not found");
    }
  });
  return new Promise((resolve) => srv.listen(0, () => resolve(srv)));
}

// Decode base64 RGBA (row-major, WebGL bottom-left origin) -> (3,size,size)
// float32 in [0,1], rows flipped to conventional top-left, and write a v1 .npy.
function writeNpyRGB(path, b64, size) {
  const rgba = Buffer.from(b64, "base64");
  const chw = new Float32Array(3 * size * size);
  for (let y = 0; y < size; y++) {
    const srcY = size - 1 - y;                 // flip: WebGL origin is bottom-left
    for (let x = 0; x < size; x++) {
      const src = (srcY * size + x) * 4;
      const dst = y * size + x;
      chw[dst] = rgba[src] / 255;                       // R plane
      chw[size * size + dst] = rgba[src + 1] / 255;     // G plane
      chw[2 * size * size + dst] = rgba[src + 2] / 255; // B plane
    }
  }
  writeFileSync(path, npyBuffer(chw, [3, size, size]));
}

// Numpy .npy v1.0 writer for a float32 C-contiguous array.
function npyBuffer(float32, shape) {
  const header = `{'descr': '<f4', 'fortran_order': False, 'shape': (${shape.join(", ")}), }`;
  const prelude = 10; // magic(6)+ver(2)+hlen(2)
  const pad = 64 - ((prelude + header.length + 1) % 64);
  const hdr = header + " ".repeat(pad) + "\n";
  const buf = Buffer.alloc(prelude + hdr.length + float32.byteLength);
  buf.write("\x93NUMPY", 0, "binary");
  buf[6] = 1; buf[7] = 0;
  buf.writeUInt16LE(hdr.length, 8);
  buf.write(hdr, 10, "binary");
  Buffer.from(float32.buffer, float32.byteOffset, float32.byteLength).copy(buf, prelude + hdr.length);
  return buf;
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
