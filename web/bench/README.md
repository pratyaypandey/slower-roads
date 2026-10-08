# web/bench: WebGPU world-model latency bench

The Step-1 spike from `HANDOFF.md`: random-weight stand-ins of the dynamics
model and FSQ decoder, run in ONNX Runtime Web (WebGPU) inside a Web Worker.
Results and recommendation: `docs/WEBGPU_BUDGET.md`.

| file | role |
|---|---|
| `serve.mjs` | static server with COOP/COEP (`node web/bench/serve.mjs 8790`) |
| `index.html` | page: env probe, Quick/Full plan buttons, results table, JSON download |
| `worker.js` | ORT session + KV cache + MaskGIT frame loop; `?rt=native\|jspi\|jsep` |
| `run_bench.mjs` | drives the page in system Chrome via Playwright, writes `results/*.json` |
| `probe.mjs` | ad-hoc job lists (probes, profiling) |
| `analyze.py` | results → markdown tables |
| `models/` | exported ONNX + `manifest.json` (gitignored; `export/export_spike.py`) |

## Running on another device (the weaker-GPU measurement)

WebGPU needs a secure context, so it works on `localhost` or over HTTPS, not
on a plain LAN IP. Use one of these setups:

- **Clone the repo on the other device.** Export the models (or copy
  `web/bench/models/`), run `npm i` here, then `node web/bench/serve.mjs`.
  Open `http://localhost:8790/`, press **Run quick plan**, then **Download
  results JSON**.
- **Serve from this machine through a tunnel that preserves headers**, e.g.
  `cloudflared tunnel --url http://localhost:8790`. Then open the https URL
  on the device.

The quick plan only needs the `*_fsq_fused_fp16` models, the
`dyn_d384_L8_c4_g16_full_fp16` baseline and `dec_h128_g16_f64_fp16`
(~350 MB).
