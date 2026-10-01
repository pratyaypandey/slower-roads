// Quick-look stills for renderer iteration: drive the autopilot and tile captures
// at the given steps into one PNG (same capture() path the datasets use).
//
// Run:  node sim/headless/snapshot.mjs [--seed N] [--at 0,150,300]
//                                      [--size 256] [--cols 4] [--out snap.png]
// GL flags as generate_pixels.mjs (SLOWSIM_GL=gpu, SLOWSIM_CHANNEL=chrome).

import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import http from "node:http";
import { readFile } from "node:fs/promises";

const HERE = dirname(fileURLToPath(import.meta.url));
const SIM_DIR = join(HERE, "..");

const args = parseArgs(process.argv.slice(2));
const SEED = Number(args.seed ?? 1);
const STOPS = String(args.at ?? "0,150,300,450,600,750,900,1050").split(",").map(Number).sort((a, b) => a - b);
const SIZE = Number(args.size ?? 256);
const COLS = Number(args.cols ?? 4);
const OUT = args.out ?? "snapshot.png";

const { chromium } = await import("playwright");
const server = await serveDir(SIM_DIR);
const GL_ARGS = process.env.SLOWSIM_GL === "gpu"
  ? ["--no-sandbox", "--use-gl=angle", "--use-angle=gl", "--ignore-gpu-blocklist", "--enable-gpu"]
  : ["--no-sandbox", "--use-gl=angle", "--use-angle=swiftshader"];
const browser = await chromium.launch({ args: GL_ARGS, channel: process.env.SLOWSIM_CHANNEL });
const page = await browser.newPage();
page.on("console", (m) => console.log("[page]", m.text()));
page.on("pageerror", (e) => console.error("[page error]", e.message));
await page.goto(`http://localhost:${server.address().port}/headless/capture_page.html`);
await page.waitForFunction("window.__ready === true");
const t0 = Date.now();
const url = await page.evaluate(([s, st, sz, c]) => window.captureStills(s, st, sz, c), [SEED, STOPS, SIZE, COLS]);
await browser.close();
server.close();
writeFileSync(OUT, Buffer.from(url.split(",")[1], "base64"));
console.log(`wrote ${OUT} (${STOPS.length} stills, seed ${SEED}, ${((Date.now() - t0) / 1000).toFixed(1)}s)`);

function serveDir(root) {
  const types = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript" };
  const srv = http.createServer(async (req, res) => {
    try {
      const p = join(root, decodeURIComponent(req.url.split("?")[0]));
      const body = await readFile(p);
      res.writeHead(200, { "content-type": types[p.slice(p.lastIndexOf("."))] || "application/octet-stream" });
      res.end(body);
    } catch {
      res.writeHead(404); res.end("not found");
    }
  });
  return new Promise((resolve) => srv.listen(0, () => resolve(srv)));
}

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i += 2) out[argv[i].replace(/^--/, "")] = argv[i + 1];
  return out;
}
