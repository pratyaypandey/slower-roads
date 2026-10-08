// Static server for the latency bench with the cross-origin-isolation headers
// WebGPU-in-worker + SharedArrayBuffer (threaded WASM fallback) need.
//   node web/bench/serve.mjs [port]      then open http://localhost:<port>/
import http from "node:http";
import { readFile, stat } from "node:fs/promises";
import { join, extname, dirname } from "node:path";
import { fileURLToPath } from "node:url";

export const ROOT = dirname(fileURLToPath(import.meta.url));
const TYPES = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript",
  ".json": "application/json", ".wasm": "application/wasm", ".onnx": "application/octet-stream" };

export function serve(port = 0) {
  const srv = http.createServer(async (req, res) => {
    let p = join(ROOT, decodeURIComponent(req.url.split("?")[0]));
    try {
      if ((await stat(p)).isDirectory()) p = join(p, "index.html");
      const body = await readFile(p);
      res.writeHead(200, {
        "content-type": TYPES[extname(p)] || "application/octet-stream",
        "cross-origin-opener-policy": "same-origin",
        "cross-origin-embedder-policy": "require-corp",
        "cache-control": "no-store",
      });
      res.end(body);
    } catch {
      res.writeHead(404); res.end("not found");
    }
  });
  return new Promise((resolve) => srv.listen(port, () => resolve(srv)));
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const srv = await serve(Number(process.argv[2] || 8790));
  console.log(`bench: http://localhost:${srv.address().port}/`);
}
