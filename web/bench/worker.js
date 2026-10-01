// Latency-bench worker: runs ONNX Runtime Web (WebGPU EP, or WASM fallback)
// off the main thread, exactly where the shipped model will run (HANDOFF §5 Step 6).
//
// One job = one dynamics model (+ optional decoder) and a list of measurements:
//   pass  - single refinement pass latency (no cache commit)
//   frame - full frame loop: P MaskGIT passes + 1 commit pass (K/V of the final
//           tokens rolled into the cache) + FSQ decode, wall-clock per frame.
// The KV cache is either kept resident on the GPU (kv:"gpu", ORT gpu-buffer
// outputs fed straight back in) or round-tripped through JS (kv:"cpu").
// ?rt= picks the ORT Web build: native WebGPU EP via asyncify (default), the
// native EP via JS Promise Integration, or the older JS execution provider.
const RUNTIMES = { native: "ort.webgpu.min.mjs", jspi: "ort.jspi.min.mjs", jsep: "ort.min.mjs" };
const RT = new URL(self.location.href).searchParams.get("rt") || "native";
// Loaded lazily inside the handler: a top-level await would drop messages that
// arrive before the module finishes evaluating.
let ort;
const ortReady = import(`./node_modules/onnxruntime-web/dist/${RUNTIMES[RT]}`).then((m) => {
  ort = m;
  ort.env.wasm.wasmPaths = new URL("./node_modules/onnxruntime-web/dist/", self.location.href).href;
});

const NUM_VISUAL = 12800, MASK_ID = 12800 + 16;

const f16 = (x) => {           // float -> IEEE half bits (enough for small ints / zeros)
  const f = new Float32Array([x]), u = new Uint32Array(f.buffer)[0];
  const s = (u >>> 16) & 0x8000, e = ((u >>> 23) & 0xff) - 112, m = (u >>> 13) & 0x3ff;
  return e <= 0 ? s : e >= 31 ? s | 0x7c00 : s | (e << 10) | m;
};
const floatTensor = (dtype, n, dims, fill = 0) => dtype === "fp16"
  ? new ort.Tensor("float16", new Uint16Array(n).fill(f16(fill)), dims)
  : new ort.Tensor("float32", new Float32Array(n).fill(fill), dims);

const stats = (xs) => {
  const s = [...xs].sort((a, b) => a - b), q = (p) => s[Math.min(s.length - 1, Math.floor(p * s.length))];
  return { n: s.length, median: q(0.5), p10: q(0.1), p90: q(0.9), min: s[0], mean: s.reduce((a, b) => a + b, 0) / s.length };
};

async function makeSession(file, ep, gpuOutputs = [], capture = false) {
  const opts = { graphOptimizationLevel: "all" };
  if (ep === "webgpu") {
    opts.executionProviders = [{ name: "webgpu" }];
    if (capture) opts.enableGraphCapture = true;
    if (gpuOutputs.length) opts.preferredOutputLocation = Object.fromEntries(gpuOutputs.map((o) => [o, "gpu-buffer"]));
  } else {
    opts.executionProviders = ["wasm"];
  }
  const t0 = performance.now();
  const buf = await (await fetch(`models/${file}`)).arrayBuffer();
  const t1 = performance.now();
  const sess = await ort.InferenceSession.create(buf, opts);
  return { sess, fetchMs: t1 - t0, createMs: performance.now() - t1 };
}

class Dyn {
  constructor(meta, sess, kv) {
    Object.assign(this, { meta, sess, kv });
    const { layers, heads, hd, ctx, S, grid, dtype } = meta;
    this.N = grid * grid;
    this.kvDims = [layers, heads, ctx * S, hd];
    this.kvLen = layers * heads * ctx * S * hd;
    this.pastK = floatTensor(dtype, this.kvLen, this.kvDims);
    this.pastV = floatTensor(dtype, this.kvLen, this.kvDims);
    this.fused = !!meta.fused;          // input = [prev frame final, next frame masked]
    this.off = this.fused ? S : 0;
    this.tokens = new BigInt64Array(this.fused ? 2 * S : S);
    this.skel = floatTensor(dtype, 16 * grid * grid, [1, 16, grid, grid], 0.5);
    this.frameIdx = 0;
  }
  async pass(commit) {
    const { dtype } = this.meta;
    const feeds = {
      tokens: new ort.Tensor("int64", this.tokens, [1, this.tokens.length]),
      t: new ort.Tensor("float32", new Float32Array([this.frameIdx]), [1]),
      skeleton: this.skel, past_k: this.pastK, past_v: this.pastV,
    };
    const fetches = commit ? ["ids", "conf", "state", "present_k", "present_v"] : ["ids", "conf", "state"];
    const out = await this.sess.run(feeds, fetches);
    if (commit) {
      if (this.kv === "gpu" && this.pastK.location === "gpu-buffer") { this.pastK.dispose(); this.pastV.dispose(); }
      this.pastK = out.present_k; this.pastV = out.present_v;
      if (this.kv === "cpu") {   // force the JS round trip: download, then re-upload next pass
        this.pastK = new ort.Tensor(this.pastK.type, this.pastK.data.slice(), this.kvDims);
        this.pastV = new ort.Tensor(this.pastV.type, this.pastV.data.slice(), this.kvDims);
      }
    }
    return out;
  }
  // One generated frame: P MaskGIT passes with a cosine unmask schedule, then a
  // commit pass on the final tokens.  Returns the final ids (BigInt64Array).
  // Fused models commit the previous frame inside the first pass instead.
  async frame(P, keys = 0) {
    const N = this.N, tok = this.tokens, off = this.off;
    if (this.fused) tok.copyWithin(0, off, off + this.meta.S);
    tok[off] = BigInt(NUM_VISUAL + keys);
    tok.fill(BigInt(MASK_ID), off + 1);
    const masked = new Uint8Array(N).fill(1);
    for (let p = 0; p < P; p++) {
      const out = await this.pass(this.fused && p === 0);
      const ids = out.ids.data, conf = out.conf.data;
      const target = p === P - 1 ? N : Math.floor(N * (1 - Math.cos((Math.PI / 2) * (p + 1) / P)));
      const have = N - masked.reduce((a, b) => a + b, 0);
      const cand = [];
      for (let i = 0; i < N; i++) if (masked[i]) cand.push(i);
      cand.sort((a, b) => Number(conf[b]) - Number(conf[a]));
      for (let j = 0; j < target - have && j < cand.length; j++) {
        const i = cand[j]; masked[i] = 0; tok[off + 1 + i] = ids[i];
      }
    }
    if (!this.fused) await this.pass(true);
    this.frameIdx++;
    return tok.subarray(off + 1);
  }
}

// Graph-capture variant (kv:"capture"): ORT records the whole pass once and
// replays the command buffer, removing per-kernel JS/WASM dispatch cost. That
// needs every input in a fixed GPU buffer, so: tokens/t are written with
// queue.writeBuffer, the cache is committed by a GPU copy of ORT's present output
// into the past buffer, and ids/conf come back through one mapped staging buffer.
class DynCaptured extends Dyn {
  constructor(meta, sess) {
    super(meta, sess, "capture");
    const { layers, heads, hd, ctx, S, grid, dtype } = meta;
    const dev = this.dev = ort.env.webgpu.device, eb = dtype === "fp16" ? 2 : 4;
    const ftype = dtype === "fp16" ? "float16" : "float32";
    const U = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    const mk = (bytes) => dev.createBuffer({ size: Math.ceil(bytes / 16) * 16, usage: U });  // vec4 kernels bind >= 16 B
    const T = (buf, dataType, dims) => ort.Tensor.fromGpuBuffer(buf, { dataType, dims });
    const kvBytes = layers * heads * ctx * S * hd * eb, N = this.N;
    this.eb = eb;
    this.buf = { tokens: mk(this.tokens.length * 8), t: mk(4), skel: mk(16 * grid * grid * eb), pk: mk(kvBytes), pv: mk(kvBytes) };
    const b = this.buf;
    this.feeds = { tokens: T(b.tokens, "int64", [1, this.tokens.length]), t: T(b.t, "float32", [1]), skeleton: T(b.skel, ftype, [1, 16, grid, grid]),
                   past_k: T(b.pk, ftype, this.kvDims), past_v: T(b.pv, ftype, this.kvDims) };
    this.staging = dev.createBuffer({ size: Math.ceil((N * 8 + N * eb) / 16) * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    this.kvBytes = kvBytes;
  }
  async pass(commit) {
    const { dev, buf: b, N, eb } = this;
    dev.queue.writeBuffer(b.tokens, 0, this.tokens);
    dev.queue.writeBuffer(b.t, 0, new Float32Array([this.frameIdx]));
    // Outputs are left to ORT (a pre-bound output is only recognised on replay when
    // the wasm allocator reuses the same handle address, so it fails intermittently).
    const out = await this.sess.run(this.feeds);
    const enc = dev.createCommandEncoder();
    enc.copyBufferToBuffer(out.ids.gpuBuffer, 0, this.staging, 0, N * 8);
    enc.copyBufferToBuffer(out.conf.gpuBuffer, 0, this.staging, N * 8, Math.ceil((N * eb) / 4) * 4);
    if (commit) {
      enc.copyBufferToBuffer(out.present_k.gpuBuffer, 0, b.pk, 0, this.kvBytes);
      enc.copyBufferToBuffer(out.present_v.gpuBuffer, 0, b.pv, 0, this.kvBytes);
    }
    dev.queue.submit([enc.finish()]);
    await this.staging.mapAsync(GPUMapMode.READ);
    const raw = this.staging.getMappedRange();
    const ids = new BigInt64Array(raw.slice(0, N * 8));
    const conf = eb === 2 ? new Uint16Array(raw.slice(N * 8, N * 8 + N * 2)) : new Float32Array(raw.slice(N * 8, N * 12));
    this.staging.unmap();
    return { ids: { data: ids }, conf: { data: conf } };
  }
  destroy() { for (const x of Object.values(this.buf)) x.destroy(); this.staging.destroy(); }
}

// FSQ decoder: ids -> RGB, read back to JS (what a 2D-canvas blit would need).
// Captured mode binds fixed GPU buffers like DynCaptured.
class Dec {
  constructor(meta, sess, capture) {
    Object.assign(this, { meta, sess, capture });
    const n = this.n = meta.grid * meta.grid;
    if (!capture) return;
    const dev = this.dev = ort.env.webgpu.device, eb = meta.dtype === "fp16" ? 2 : 4;
    const U = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.rgbBytes = 3 * meta.frame * meta.frame * eb;
    this.bIds = dev.createBuffer({ size: n * 8, usage: U });
    this.staging = dev.createBuffer({ size: this.rgbBytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    this.feeds = { ids: ort.Tensor.fromGpuBuffer(this.bIds, { dataType: "int64", dims: [1, n] }) };
  }
  async run(ids) {
    if (!this.capture) return (await this.sess.run({ ids: new ort.Tensor("int64", ids.slice(), [1, this.n]) })).rgb.data;
    this.dev.queue.writeBuffer(this.bIds, 0, ids);
    const out = await this.sess.run(this.feeds);       // ORT-owned output (see DynCaptured.pass)
    const enc = this.dev.createCommandEncoder();
    enc.copyBufferToBuffer(out.rgb.gpuBuffer, 0, this.staging, 0, this.rgbBytes);
    this.dev.queue.submit([enc.finish()]);
    await this.staging.mapAsync(GPUMapMode.READ);
    const rgb = this.staging.getMappedRange().slice(0);
    this.staging.unmap();
    return rgb;
  }
  async release() {
    await this.sess.release();
    if (this.capture) { this.bIds.destroy(); this.staging.destroy(); }
  }
}

// Generic probe: run an arbitrary model on zero inputs (floor/overhead studies).
async function probeJob(job) {
  const { file, feeds: spec, ep = "webgpu", iters = 50, sessionOptions = {} } = job.probe;
  const { sess } = await makeSession(file, ep);
  const feeds = {};
  for (const [k, { type, dims }] of Object.entries(spec)) {
    const n = dims.reduce((a, b) => a * b, 1);
    feeds[k] = new ort.Tensor(type, type === "int64" ? new BigInt64Array(n) : type === "float16" ? new Uint16Array(n) : new Float32Array(n), dims);
  }
  for (let i = 0; i < 5; i++) await sess.run(feeds);
  const ts = [];
  for (let i = 0; i < iters; i++) { const t0 = performance.now(); await sess.run(feeds); ts.push(performance.now() - t0); }
  await sess.release();
  return { job, runtime: RT, probe: stats(ts) };
}

let STAGE = "";
async function benchJob(job) {
  if (job.probe) return probeJob(job);
  const { dyn, dec, ep = "webgpu", kv = "gpu", warmup = 5, passIters = 30, frameIters = 20, P = [] } = job;
  const res = { job, ep, kv, runtime: RT };
  let d = null, decoder = null;
  if (dyn) {
    const capture = kv === "capture" && ep === "webgpu";
    const s = await makeSession(dyn.file, ep, kv === "gpu" && ep === "webgpu" ? ["present_k", "present_v"] : [], capture);
    res.dynLoad = { fetchMs: s.fetchMs, createMs: s.createMs };
    d = capture ? new DynCaptured(dyn, s.sess) : new Dyn(dyn, s.sess, ep === "webgpu" ? kv : "cpu");
    const tw = performance.now();
    STAGE = "dyn-first";
    await d.pass(true);
    STAGE = "dyn-loop";
    res.firstPassMs = performance.now() - tw;          // shader compile + upload
    for (let i = 0; i < warmup; i++) await d.pass(i % 2 === 0);
    const tp = [], tc = [];
    for (let i = 0; i < passIters; i++) {
      let t0 = performance.now(); await d.pass(false); tp.push(performance.now() - t0);
      t0 = performance.now(); await d.pass(true); tc.push(performance.now() - t0);
    }
    res.pass = stats(tp);
    res.commitPass = stats(tc);
  }
  if (dec) {
    const capture = kv === "capture" && ep === "webgpu";
    const s = await makeSession(dec.file, ep, [], capture);
    decoder = new Dec(dec, s.sess, capture);
    res.decLoad = { fetchMs: s.fetchMs, createMs: s.createMs };
    const n = dec.grid * dec.grid;
    const ids = new BigInt64Array(n).map((_, i) => BigInt((i * 7919) % NUM_VISUAL));
    STAGE = "dec";
    let t0 = performance.now(); await decoder.run(ids); res.decFirstMs = performance.now() - t0;
    for (let i = 0; i < warmup; i++) await decoder.run(ids);
    const td = [];
    for (let i = 0; i < passIters; i++) { t0 = performance.now(); await decoder.run(ids); td.push(performance.now() - t0); }
    res.decode = stats(td);
  }
  if (d) {
    res.frame = {};
    const decOk = dec && dec.grid === dyn.grid;
    STAGE = "frame";
    for (const p of P) {
      await d.frame(p);                                // warm the schedule path
      const tf = [];
      for (let i = 0; i < frameIters; i++) {
        const t0 = performance.now();
        const ids = await d.frame(p, i % 16);
        if (decOk) await decoder.run(ids);
        tf.push(performance.now() - t0);
      }
      res.frame[p] = stats(tf);
    }
    res.frameIncludesDecode = !!decOk;
    if (job.sustainSec) {   // long run at one P: does throughput hold (thermals, GC, allocator)?
      const tAll = [], buckets = [], start = performance.now();
      let bucket = [], bStart = start;
      while (performance.now() - start < job.sustainSec * 1000) {
        const t0 = performance.now();
        const ids = await d.frame(job.sustainP, tAll.length % 16);
        if (decOk) await decoder.run(ids);
        const dt = performance.now() - t0;
        tAll.push(dt); bucket.push(dt);
        if (performance.now() - bStart > 5000) { buckets.push(stats(bucket)); bucket = []; bStart = performance.now(); }
      }
      res.sustain = { P: job.sustainP, sec: job.sustainSec, all: stats(tAll), buckets };
    }
    await d.sess.release();
    if (d.destroy) d.destroy();
    else if (d.pastK.location === "gpu-buffer") { d.pastK.dispose(); d.pastV.dispose(); }
  }
  if (decoder) await decoder.release();
  return res;
}

self.onmessage = async (e) => {
  const { id, cmd, job, threads } = e.data;
  try {
    await ortReady;
    if (cmd === "init") {
      if (e.data.ownDevice) {   // hand ORT a device we created (with shader-f16) instead of letting it make one
        const ad = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
        ort.env.webgpu.device = await ad.requestDevice({
          requiredFeatures: [...ad.features].filter((f) => ["shader-f16", "timestamp-query", "subgroups"].includes(f)),
          requiredLimits: { maxBufferSize: ad.limits.maxBufferSize, maxStorageBufferBindingSize: ad.limits.maxStorageBufferBindingSize,
                            maxComputeWorkgroupStorageSize: ad.limits.maxComputeWorkgroupStorageSize } });
      }
      if (e.data.profile) {
        self.kernelLog = [];
        ort.env.webgpu.profiling = { mode: "default", ondata: (d) => self.kernelLog.push(d) };
      }
      ort.env.wasm.numThreads = threads || navigator.hardwareConcurrency;
      const adapter = navigator.gpu && await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
      const info = adapter ? adapter.info : null;
      self.postMessage({ id, ok: true, result: {
        ortVersion: ort.env.versions?.web, runtime: RT, crossOriginIsolated: self.crossOriginIsolated,
        hardwareConcurrency: navigator.hardwareConcurrency, userAgent: navigator.userAgent,
        adapter: info && { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description },
        features: adapter ? [...adapter.features] : [],
        limits: adapter ? { maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize } : null,
      } });
    } else if (cmd === "bench") {
      const result = await benchJob(job);
      if (self.kernelLog) result.kernels = self.kernelLog.slice(-4000);
      self.postMessage({ id, ok: true, result });
    }
  } catch (err) {
    self.postMessage({ id, ok: false, error: `[stage ${STAGE}] ` + String(err && err.stack || err) });
  }
};
