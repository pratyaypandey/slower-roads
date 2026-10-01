// Shared helpers for the browser-driven capture scripts.

import { writeFileSync } from "node:fs";
import { join } from "node:path";
import http from "node:http";
import { readFile } from "node:fs/promises";

// CPU (SwiftShader) by default: a shared GPU can stall on sustained ReadPixels and
// lose the context. SLOWSIM_GL=gpu opts into hardware on an idle box;
// SLOWSIM_CHANNEL=chrome uses the system Chrome instead of Playwright's bundled build.
export function launchOptions() {
  // SLOWSIM_ANGLE picks ANGLE's GPU backend (gl | metal | vulkan | d3d11); metal is
  // much faster on macOS than the deprecated OpenGL path.
  const args = process.env.SLOWSIM_GL === "gpu"
    ? ["--no-sandbox", "--use-gl=angle", `--use-angle=${process.env.SLOWSIM_ANGLE || "gl"}`, "--ignore-gpu-blocklist", "--enable-gpu"]
    : ["--no-sandbox", "--use-gl=angle", "--use-angle=swiftshader"];
  return { args, channel: process.env.SLOWSIM_CHANNEL };
}

/** Minimal static server for the sim dir (serves core/, render/, vendor/, headless/). */
export function serveDir(root) {
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

/** Base64 RGBA (row-major, WebGL bottom-left origin) -> (3,size,size) float32 in
 *  [0,1], rows flipped to top-left, written as a v1 .npy. */
export function writeNpyRGB(path, b64, size) {
  const rgba = Buffer.from(b64, "base64");
  const chw = new Float32Array(3 * size * size);
  for (let y = 0; y < size; y++) {
    const srcY = size - 1 - y;
    for (let x = 0; x < size; x++) {
      const src = (srcY * size + x) * 4, dst = y * size + x;
      chw[dst] = rgba[src] / 255;
      chw[size * size + dst] = rgba[src + 1] / 255;
      chw[2 * size * size + dst] = rgba[src + 2] / 255;
    }
  }
  writeFileSync(path, npyBuffer(chw, [3, size, size]));
}

/** Numpy .npy v1.0 bytes for a float32 C-contiguous array. */
export function npyBuffer(float32, shape) {
  const header = `{'descr': '<f4', 'fortran_order': False, 'shape': (${shape.join(", ")}), }`;
  const prelude = 10;
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
