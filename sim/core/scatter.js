// Deterministic world scatter: trees, rocks, bushes, grass.
//
// This lives in core (not the renderer) because vegetation is part of the *world*, not
// just decoration: the renderer draws it and the physics collides with it, and both
// must agree exactly. It is a pure function of (seed, road geometry, dials), keyed on
// arc-length so the same seed + drive always produces the same forest (SIM.md §5).
//
// Objects are placed road-*relative* (beyond the shoulder, both sides) so nothing ever
// lands on the road itself — you only hit a tree if you leave the road.

import { hash1, fbm2 } from './prng.js';
import { terrainSurfaceHeight, SEA_LEVEL } from './road.js';

const SPACING = 6;        // metres between scatter rows along the road
const LMIN = 13;          // first lateral offset off the centerline (past the shoulder)
const LMAX = 210;         // furthest scattered object
const LSTEP = 7;          // lateral spacing between candidate slots
const GRASS_LMAX = 46;    // grass only fills a near band
const GRASS_LSTEP = 2.6;

// Collision radius per type (metres). Grass/bush are not collidable (you brush through).
export const COLLIDABLE = { pine: true, tree: true, rock: true, bush: false, grass: false };

/**
 * Scatter objects for road arc-lengths in [dFrom, dTo].
 * The scenic director (optional) modulates density: enclosed forest beats thicken, open
 * vista beats thin out, and the open (vista) side of the road is deliberately cleared.
 * @returns array of { d, lateral, x, z, y, type, scale, rot, radius, collidable }
 */
export function scatter(road, dFrom, dTo, seed, dials, director = null, { grass = true } = {}) {
  const out = [];
  const lush = dials.biomeX;
  const hilliness = dials.hilliness;
  const i0 = Math.ceil(dFrom / SPACING);
  const i1 = Math.floor(dTo / SPACING);

  for (let i = i0; i <= i1; i++) {
    const d = i * SPACING;
    // Director-driven density: 0.55x in the open .. 1.5x in dense forest, and the vista
    // side is opened up so reveals read.
    const beat = director ? director.beatAt(d) : null;
    const densMul = beat ? 0.55 + beat.enclosure * 0.95 : 1;
    const vistaSide = beat ? beat.vistaSide : 0;

    for (let side = -1; side <= 1; side += 2) {
      const sideMul = side === vistaSide ? 0.32 : 1;
      // Trees / rocks / bushes.
      for (let L = LMIN, k = 0; L <= LMAX; L += LSTEP, k++) {
        const s0 = (i * 2749 + k * 131 + (side > 0 ? 977 : 13)) >>> 0;
        const h1 = hash1(s0 + 1, seed), h2 = hash1(s0 + 2, seed), h3 = hash1(s0 + 3, seed);
        const Lj = L + (h1 - 0.5) * LSTEP, aj = (h2 - 0.5) * SPACING;
        const pd = Math.max(0, Math.min(road.length, d + aj));
        const anchor = road.sampleAt(pd);
        const rx = Math.cos(anchor.heading), rz = -Math.sin(anchor.heading);
        const lateral = side * Lj;
        const x = anchor.x + rx * lateral;
        const z = anchor.z + rz * lateral;
        // Ecological clustering (SIM_UPGRADE §5.4): a low-frequency grove field gates
        // density so trees gather in copses with clearings between, instead of an even
        // lattice. Rock/bush persist in the clearings for ground interest.
        const grove = fbm2(x * 0.012, z * 0.012, seed + 880, 3);   // [-1,1]
        const groveMul = 0.35 + smoothstep(-0.15, 0.4, grove) * 1.35;
        const h0 = hash1(s0, seed);
        const density = (0.34 + lush * 0.4 - (L / LMAX) * 0.15) * densMul * sideMul;
        const treeDensity = density * groveMul;
        if (h0 > treeDensity && h0 > density * 0.4) continue;      // clearings still get rocks/bushes
        const y = terrainSurfaceHeight(anchor, lateral, x, z, seed, hilliness);
        if (y < SEA_LEVEL + 0.5) continue;                        // no forest underwater
        const rot = h3 * 6.2832;
        let type, scale, radius;
        const inGrove = h0 < treeDensity;
        if (inGrove && h0 < treeDensity * (0.25 + lush * 0.4)) { type = 'pine'; scale = 0.8 + h1 * 1.1; radius = 0.6 * scale; }
        else if (inGrove && h0 < treeDensity * (0.6 + lush * 0.3)) { type = 'tree'; scale = 0.9 + h1 * 1.3; radius = 0.7 * scale; }
        else if (h2 < 0.5) { type = 'rock'; scale = 0.7 + h1 * 1.6; radius = 0.9 * scale; }
        else { type = 'bush'; scale = 0.7 + h1 * 1.2; radius = 0.8 * scale; }
        out.push({ d: pd, lateral, x, z, y, type, scale, rot, radius, collidable: COLLIDABLE[type] });
      }
      // Grass (near band, non-collidable).
      if (!grass) continue;
      for (let L = 6, k = 0; L <= GRASS_LMAX; L += GRASS_LSTEP, k++) {
        const s0 = (i * 5311 + k * 17 + (side > 0 ? 71 : 3)) >>> 0;
        const h0 = hash1(s0, seed + 9);
        if (h0 > (0.5 + lush * 0.4) * (beat ? 0.7 + beat.enclosure * 0.5 : 1)) continue;
        const h1 = hash1(s0 + 1, seed + 9), h2 = hash1(s0 + 2, seed + 9);
        const Lj = L + (h1 - 0.5) * GRASS_LSTEP, aj = (h2 - 0.5) * SPACING;
        const pd = Math.max(0, Math.min(road.length, d + aj));
        const anchor = road.sampleAt(pd);
        const rx = Math.cos(anchor.heading), rz = -Math.sin(anchor.heading);
        const lateral = side * Lj;
        const x = anchor.x + rx * lateral;
        const z = anchor.z + rz * lateral;
        const y = terrainSurfaceHeight(anchor, lateral, x, z, seed, hilliness);
        if (y < SEA_LEVEL + 0.3) continue;
        out.push({ d: pd, lateral, x, z, y, type: 'grass', scale: 0.6 + h1 * 0.9, rot: h2 * 6.2832, radius: 0, collidable: false });
      }
    }
  }
  return out;
}

function smoothstep(e0, e1, x) {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
}
