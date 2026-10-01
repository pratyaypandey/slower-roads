// Drive the bench page in system Chrome (WebGPU on the real GPU) and save results.
//
//   node web/bench/run_bench.mjs --match 'dyn_.*_fp16' --kv gpu --out results/m3max_core.json
//   node web/bench/run_bench.mjs --match '^dec_' --out results/m3max_dec.json
//   node web/bench/run_bench.mjs --match 'dyn_d256_L4_c4_g16_full$' --ep wasm
//
// Flags: --match <regex over model names>  --ep webgpu|wasm  --kv capture|gpu|cpu (default capture)
//        --sustain <sec> --sustainP <P> (long frame loop to check thermals)
//        --rt native|jspi|jsep (ORT Web build)  --P 2,4 (refinement passes for the frame loop)  --iters N  --headed
import { chromium } from "../../sim/node_modules/playwright/index.mjs";
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join, dirname } from "node:path";
import { serve, ROOT } from "./serve.mjs";

const argv = process.argv.slice(2);
const arg = (k, d) => { const i = argv.indexOf(`--${k}`); return i < 0 ? d : argv[i + 1]; };
const match = new RegExp(arg("match", "."));
const ep = arg("ep", "webgpu"), kv = arg("kv", "capture"), runtime = arg("rt", "native");
const sustainSec = Number(arg("sustain", 0)), sustainP = Number(arg("sustainP", 4));
const P = arg("P", "2,4").split(",").filter(Boolean).map(Number);
const iters = Number(arg("iters", 30));
const decName = arg("dec", "auto");
const out = join(ROOT, arg("out", `results/run_${Date.now()}.json`));

const manifest = JSON.parse(readFileSync(join(ROOT, "models/manifest.json"), "utf8"));
const plan = [];
for (const m of Object.values(manifest)) {
  if (!match.test(m.name)) continue;
  if (m.kind === "dec") { plan.push({ dec: m, ep, kv, runtime, passIters: iters }); continue; }
  const dn = decName === "auto" ? `dec_h128_g${m.grid}_f${m.grid * 4}${m.dtype === "fp16" ? "_fp16" : ""}` : decName;
  plan.push({ dyn: m, dec: decName === "none" ? undefined : manifest[dn], ep, kv, runtime, P,
              passIters: iters, frameIters: Math.max(5, Math.round(iters * 0.66)), sustainSec, sustainP });
}
console.log(`${plan.length} jobs`);

const srv = await serve(0);
const browser = await chromium.launch({
  channel: "chrome", headless: !argv.includes("--headed"),
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPU", "--ignore-gpu-blocklist",
         "--disable-background-timer-throttling", "--disable-renderer-backgrounding"],
});
const page = await browser.newPage();
page.on("console", (m) => { if (m.type() === "error") console.log("[page]", m.text()); });
await page.exposeFunction("progress", (r) => {
  const m = r.job.dyn || r.job.dec;
  if (r.error) return console.log(`${m.name}: ERROR ${r.error.split("\n")[0]}`);
  const f = (x) => (x == null ? "-" : x.toFixed(2));
  console.log(`${m.name.padEnd(34)} pass ${f(r.pass?.median)}  commit ${f(r.commitPass?.median)}  ` +
              `dec ${f(r.decode?.median)}  frame ${Object.entries(r.frame || {}).map(([p, s]) => `P${p}=${f(s.median)}`).join(" ")}  first ${f(r.firstPassMs ?? r.decFirstMs)}` +
              (r.sustain ? `  sustain p50 ${f(r.sustain.all.median)} p90 ${f(r.sustain.all.p90)} buckets ${r.sustain.buckets.map((b) => b.median.toFixed(1)).join(",")}` : ""));
});
await page.goto(`http://localhost:${srv.address().port}/`);
await page.waitForFunction(() => window.benchReady, null, { timeout: 60000 });
const env = await page.evaluate(() => window.benchEnv);
console.log("env:", JSON.stringify(env.adapter), "f16:", env.features.includes("shader-f16"), "isolated:", env.crossOriginIsolated);
const results = await page.evaluate((plan) => window.runPlan(plan, window.progress), plan);
mkdirSync(dirname(out), { recursive: true });
writeFileSync(out, JSON.stringify({ env, date: new Date().toISOString(), results }, null, 1));
console.log(`wrote ${out}`);
await browser.close();
srv.close();
