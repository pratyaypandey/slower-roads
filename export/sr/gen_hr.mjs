// Render high-res (default 256 px) sim frames for the 64->256 upscaler study.
// Same capture path as the training data (supersampled + box-filtered), so a box
// downsample of these frames matches what the 64 px pipeline sees.
//   SLOWSIM_ANGLE=metal SLOWSIM_CHANNEL=chrome SLOWSIM_GL=gpu \
//     node export/sr/gen_hr.mjs --out data/sr_hr --episodes 30 --test 4
import { mkdirSync, writeFileSync, existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
const HERE = dirname(fileURLToPath(import.meta.url));
const SIM = join(HERE, "..", "..", "sim");
const { launchOptions, serveDir } = await import(join(SIM, "headless", "lib.mjs"));
const { chromium } = createRequire(join(SIM, "package.json"))("playwright");

const a = Object.fromEntries(process.argv.slice(2).reduce((r, v, i, x) => (i % 2 ? r : [...r, [x[i].slice(2), x[i + 1]]]), []));
const OUT = a.out ?? "data/sr_hr", SIZE = Number(a.size ?? 256), EPS = Number(a.episodes ?? 30), TEST = Number(a.test ?? 4);
const STEPS = Number(a.steps ?? 600), STRIDE = Number(a.stride ?? 3), SKIP = 60;   // skip the edge-of-world start
const POLICIES = ["cruise", "keys_lane", "keys_explore", "dial_mix", "lane_change"];

const server = await serveDir(SIM);
const browser = await chromium.launch(launchOptions());
const page = await browser.newPage();
await page.goto(`http://localhost:${server.address().port}/headless/capture_page.html`);
await page.waitForFunction("window.__ready === true");
const t0 = Date.now();
for (let i = 0; i < EPS + TEST; i++) {
  const split = i < EPS ? "train" : "test", seed = (split === "train" ? 20000 : 30000) + i, policy = POLICIES[i % POLICIES.length];
  const dir = join(OUT, split, `s${seed}_${policy}`);
  if (existsSync(join(dir, "done"))) continue;
  mkdirSync(dir, { recursive: true });
  // returns PNG data URLs for every STRIDE-th frame, encoded in the page (small transfer)
  const pngs = await page.evaluate(async ([seed, steps, size, policy, stride, skip]) => {
    const res = await window.captureDrive(seed, steps, size, { policy, policySeed: seed * 7919 + 17 });
    const c = new OffscreenCanvas(size, size), g = c.getContext("2d"), out = [];
    for (let k = skip; k < res.samples.length; k += stride) {
      const bin = atob(res.samples[k].rgba), img = new ImageData(size, size);
      for (let y = 0; y < size; y++) for (let x = 0; x < size * 4; x++)       // flip WebGL bottom-left origin
        img.data[y * size * 4 + x] = bin.charCodeAt((size - 1 - y) * size * 4 + x);
      g.putImageData(img, 0, 0);
      const b = await c.convertToBlob({ type: "image/png" });
      out.push(btoa(String.fromCharCode(...new Uint8Array(await b.arrayBuffer()))));
    }
    return out;
  }, [seed, STEPS, SIZE, policy, STRIDE, SKIP]);
  pngs.forEach((p, k) => writeFileSync(join(dir, `${String(k).padStart(4, "0")}.png`), Buffer.from(p, "base64")));
  writeFileSync(join(dir, "done"), "");
  console.log(`[${i + 1}/${EPS + TEST}] ${split} ${dir} ${pngs.length} frames ${((Date.now() - t0) / 1000).toFixed(0)}s`);
}
await browser.close(); server.close();
