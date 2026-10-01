// Run ad-hoc jobs (JSON list) through the bench page; prints + returns raw results.
//   node web/bench/probe.mjs jobs.json [out.json]
import { chromium } from "../../sim/node_modules/playwright/index.mjs";
import { readFileSync, writeFileSync } from "node:fs";
import { serve } from "./serve.mjs";
const jobs = JSON.parse(readFileSync(process.argv[2], "utf8"));
const srv = await serve(0);
const browser = await chromium.launch({ channel: "chrome", headless: !process.argv.includes("--headed"),
  args: ["--enable-unsafe-webgpu", "--ignore-gpu-blocklist", "--disable-background-timer-throttling", "--disable-renderer-backgrounding"] });
const page = await browser.newPage();
page.on("console", (m) => { if ((m.type() === "error" && !m.text().includes("W:onnx")) || m.text().startsWith("CHK")) console.log("[page]", m.text()); });
page.on("pageerror", (e) => console.log("[pageerror]", e.message));
await page.goto(`http://localhost:${srv.address().port}/`);
await page.waitForFunction(() => window.benchReady, null, { timeout: 60000 });
const res = await page.evaluate((jobs) => window.runPlan(jobs), jobs);
if (process.argv[3] && !process.argv[3].startsWith("--")) writeFileSync(process.argv[3], JSON.stringify(res, null, 1));
for (const r of res) console.log(JSON.stringify({ name: (r.job.probe?.file || r.job.dyn?.name || r.job.dec?.name), err: r.error?.slice(0, 200), probe: r.probe?.median, pass: r.pass?.median, dec: r.decode?.median, kernels: r.kernels?.length }));
await browser.close(); srv.close();
