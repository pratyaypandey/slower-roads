// Headless WebGL renderer backed by `gl` (headless-gl). Produces a Three.js
// WebGLRenderer that draws into an offscreen framebuffer we can read back as
// pixels — no window, no display. Kept separate from the sim so the same core
// runs here and in the browser unchanged.

import gl from "gl";
import * as THREE from "three";

export function createHeadlessRenderer(width, height) {
  const context = gl(width, height, { preserveDrawingBuffer: true });

  const renderer = new THREE.WebGLRenderer({
    context,
    antialias: false,
    powerPreference: "high-performance",
  });
  renderer.setSize(width, height, false);

  // Read the current framebuffer back as a top-to-bottom RGBA buffer. WebGL's
  // origin is bottom-left, so we flip rows to get conventional image order.
  function readPixels() {
    const rgba = new Uint8Array(width * height * 4);
    context.readPixels(0, 0, width, height, context.RGBA, context.UNSIGNED_BYTE, rgba);
    const flipped = new Uint8Array(rgba.length);
    const rowBytes = width * 4;
    for (let y = 0; y < height; y++) {
      const src = y * rowBytes;
      const dst = (height - 1 - y) * rowBytes;
      flipped.set(rgba.subarray(src, src + rowBytes), dst);
    }
    return flipped;
  }

  return { THREE, renderer, readPixels, width, height };
}
