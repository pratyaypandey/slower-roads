// Procedural textures for the v2 engine: tileable value noise plus hand-"painted"
// alpha cards (leaf clumps, conifers, grass tufts) drawn stroke by stroke on a 2D
// canvas. Everything is seeded (no Math.random), so a renderer instance always
// builds the same textures. Cards are near-greyscale; colour comes from per-instance
// tints so one texture serves every biome.

import * as THREE from 'three';

/** Small seeded PRNG (mulberry32). */
export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Tileable fbm value noise, one independent field per RGBA channel. */
export function noiseTexture(size = 256) {
  const data = new Uint8Array(size * size * 4);
  const periods = [4, 16, 32, 8];   // lattice cells per tile, per channel
  for (let c = 0; c < 4; c++) {
    const field = new Float32Array(size * size);
    let amp = 1, total = 0;
    for (let o = 0; o < 4; o++) {
      const p = periods[c] << o;
      const r = rng(1013 * (c + 1) + o * 7919);
      const lat = new Float32Array(p * p).map(() => r());
      for (let y = 0; y < size; y++) {
        const fy = (y / size) * p, y0 = Math.floor(fy), ty = smooth(fy - y0);
        for (let x = 0; x < size; x++) {
          const fx = (x / size) * p, x0 = Math.floor(fx), tx = smooth(fx - x0);
          const a = lat[(y0 % p) * p + (x0 % p)], b = lat[(y0 % p) * p + ((x0 + 1) % p)];
          const cc = lat[((y0 + 1) % p) * p + (x0 % p)], d = lat[((y0 + 1) % p) * p + ((x0 + 1) % p)];
          field[y * size + x] += amp * (a + (b - a) * tx + (cc - a) * ty + (a - b - cc + d) * tx * ty);
        }
      }
      total += amp; amp *= 0.5;
    }
    // Normalise to the full 0..1 range so thresholds in the shaders are meaningful.
    let lo = Infinity, hi = -Infinity;
    for (const v of field) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    for (let i = 0; i < size * size; i++) data[i * 4 + c] = Math.round(((field[i] - lo) / (hi - lo)) * 255);
  }
  const tex = new THREE.DataTexture(data, size, size, THREE.RGBAFormat);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.generateMipmaps = true;
  tex.needsUpdate = true;
  return tex;
}

function smooth(t) { return t * t * (3 - 2 * t); }

function canvasTexture(w, h, draw, { repeat = false } = {}) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const g = c.getContext('2d');
  draw(g, w, h);
  const tex = new THREE.CanvasTexture(c);
  if (repeat) tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.anisotropy = 4;
  tex.colorSpace = THREE.NoColorSpace;   // authored values are used as-is
  return tex;
}

const grey = (v, a = 1) => `rgba(${v | 0},${v | 0},${v | 0},${a})`;
const tinted = (v, dg, a = 1) => `rgba(${(v * 0.92) | 0},${Math.min(255, v + dg) | 0},${(v * 0.8) | 0},${a})`;

/** Broadleaf canopy clump: hundreds of small leaf dabs inside a lumpy disc, lit from
 *  the top-left and darker underneath (baked self-shadow). */
export function leafClumpTexture(seed = 11) {
  return canvasTexture(256, 256, (g, w, h) => {
    const r = rng(seed);
    const lobes = Array.from({ length: 7 }, () => ({
      x: 128 + (r() - 0.5) * 110, y: 128 + (r() - 0.5) * 90, rad: 48 + r() * 34,
    }));
    const inside = (x, y) => lobes.some((l) => (x - l.x) ** 2 + (y - l.y) ** 2 < l.rad ** 2);
    for (let i = 0; i < 3200; i++) {
      const x = 8 + r() * 240, y = 8 + r() * 240;
      if (!inside(x, y)) continue;
      const light = 1.05 - (y / h) * 0.8 - (x / w) * 0.12;          // top-left light, dark underside
      const v = 55 + 175 * Math.max(0, Math.min(1, light + (r() - 0.5) * 0.4));
      g.fillStyle = tinted(v, 14 * r());
      g.beginPath();
      g.ellipse(x, y, 3 + r() * 6, 2.2 + r() * 4, r() * Math.PI, 0, Math.PI * 2);
      g.fill();
    }
  });
}

/** Conifer silhouette: tiers of drooping needle strokes on a central stem. */
export function coniferTexture(seed = 23) {
  return canvasTexture(128, 256, (g, w, h) => {
    const r = rng(seed);
    const top = 6, base = h - 20;
    g.fillStyle = grey(60); g.fillRect(w / 2 - 2, base - 10, 4, h - base + 10);   // stem
    for (let i = 0; i < 900; i++) {
      const t = Math.pow(r(), 0.8);                 // more branches low down
      const y = top + t * (base - top);
      const half = (4 + t * (w / 2 - 8)) * (0.75 + 0.35 * r());
      const side = r() < 0.5 ? -1 : 1;
      const len = half * (0.4 + 0.6 * r());
      const v = 55 + 120 * Math.max(0, Math.min(1, 1 - t * 0.5 + (r() - 0.5) * 0.5 - (side > 0 ? 0.1 : 0)));
      g.strokeStyle = tinted(v, 10 * r());
      g.lineWidth = 1.2 + r() * 1.8;
      g.beginPath();
      g.moveTo(w / 2, y);
      g.quadraticCurveTo(w / 2 + side * len * 0.6, y + 2, w / 2 + side * len, y + 5 + len * 0.25);
      g.stroke();
    }
  });
}

/** Grass tuft: a fan of tapered blades, darker at the root. */
export function grassTuftTexture(seed = 37) {
  return canvasTexture(256, 256, (g, w, h) => {
    const r = rng(seed);
    for (let i = 0; i < 90; i++) {
      const x0 = w / 2 + (r() - 0.5) * 120;
      const lean = (x0 - w / 2) * (0.6 + r()) + (r() - 0.5) * 60;
      const hgt = 110 + r() * 135;
      const v = 105 + r() * 130;
      const grd = g.createLinearGradient(0, h, 0, h - hgt);
      grd.addColorStop(0, tinted(v * 0.82, 4));
      grd.addColorStop(1, tinted(v, 12));
      g.fillStyle = grd;
      const bw = 4 + r() * 4;
      g.beginPath();
      g.moveTo(x0 - bw, h);
      g.quadraticCurveTo(x0 + lean * 0.3, h - hgt * 0.6, x0 + lean, h - hgt);
      g.quadraticCurveTo(x0 + lean * 0.3 + bw, h - hgt * 0.6, x0 + bw, h);
      g.fill();
    }
  });
}

/** Tiling ground detail (luminance): short grass strokes seen from above. */
export function grassDetailTexture(seed = 41) {
  return canvasTexture(256, 256, (g, w, h) => {
    const r = rng(seed);
    g.fillStyle = grey(128); g.fillRect(0, 0, w, h);
    for (let i = 0; i < 5000; i++) {
      const x = r() * w, y = r() * h, len = 3 + r() * 7, a = -Math.PI / 2 + (r() - 0.5) * 1.4;
      g.strokeStyle = grey(60 + r() * 150, 0.8);
      g.lineWidth = 1 + r() * 1.3;
      for (const ox of [-w, 0, w]) for (const oy of [-h, 0, h]) {   // wrap for tiling
        g.beginPath();
        g.moveTo(x + ox, y + oy);
        g.lineTo(x + ox + Math.cos(a) * len, y + oy + Math.sin(a) * len);
        g.stroke();
      }
    }
  }, { repeat: true });
}
