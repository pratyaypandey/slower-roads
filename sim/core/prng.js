// Deterministic pseudo-randomness for the sim.
//
// Everything stochastic in the sim routes through here so that (seed + action
// sequence) fully determines the state trajectory. No Math.random(), no
// Date.now(), no wall-clock anywhere in core/. See ../../plans/SIM.md §6.

/**
 * mulberry32 — a tiny, fast, well-distributed 32-bit seeded PRNG.
 * Returns a function yielding floats in [0, 1).
 */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Deterministic integer hash -> float in [0, 1). Order-independent, stateless. */
export function hash1(n, seed = 0) {
  let h = Math.imul(n ^ seed, 0x27d4eb2d);
  h ^= h >>> 15;
  h = Math.imul(h, 0x85ebca6b);
  h ^= h >>> 13;
  return (h >>> 0) / 4294967296;
}

/** Deterministic 2D integer hash -> float in [0, 1). */
export function hash2(x, y, seed = 0) {
  return hash1(Math.imul(x | 0, 73856093) ^ Math.imul(y | 0, 19349663), seed);
}

const smooth = (t) => t * t * (3 - 2 * t); // smoothstep, for C1-continuous noise
const lerp = (a, b, t) => a + (b - a) * t;

/**
 * Value noise in 1D, continuous, in [-1, 1]. Smooth param->pixel maps (SIM.md §1.3)
 * start here: the road curvature and terrain are built from smooth noise, never steps.
 */
export function noise1(x, seed = 0) {
  const i = Math.floor(x);
  const f = x - i;
  const a = hash1(i, seed);
  const b = hash1(i + 1, seed);
  return (lerp(a, b, smooth(f)) * 2 - 1);
}

/** Value noise in 2D, continuous, in [-1, 1]. Used for terrain height. */
export function noise2(x, y, seed = 0) {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const fx = x - ix;
  const fy = y - iy;
  const s = smooth(fx);
  const t = smooth(fy);
  const a = lerp(hash2(ix, iy, seed), hash2(ix + 1, iy, seed), s);
  const b = lerp(hash2(ix, iy + 1, seed), hash2(ix + 1, iy + 1, seed), s);
  return (lerp(a, b, t) * 2 - 1);
}

/** Fractal Brownian motion over noise1: layered octaves for richer-but-smooth signal. */
export function fbm1(x, seed = 0, octaves = 4, lacunarity = 2, gain = 0.5) {
  let sum = 0;
  let amp = 0.5;
  let freq = 1;
  let norm = 0;
  for (let o = 0; o < octaves; o++) {
    sum += amp * noise1(x * freq, seed + o * 1013);
    norm += amp;
    amp *= gain;
    freq *= lacunarity;
  }
  return sum / norm;
}

/** Fractal Brownian motion over noise2. */
export function fbm2(x, y, seed = 0, octaves = 4, lacunarity = 2, gain = 0.5) {
  let sum = 0;
  let amp = 0.5;
  let freq = 1;
  let norm = 0;
  for (let o = 0; o < octaves; o++) {
    sum += amp * noise2(x * freq, y * freq, seed + o * 1013);
    norm += amp;
    amp *= gain;
    freq *= lacunarity;
  }
  return sum / norm;
}
