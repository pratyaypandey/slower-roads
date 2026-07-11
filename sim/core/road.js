// Procedural endless road.
//
// The road centerline is a spline parameterized by arc-length `d` (metres), built
// forward from the seed as the car advances (SIM.md §5). It is the λ-anchor — the
// one thing that must never melt — so it is the most deterministic object in the sim:
// samples are baked into a cache as they are generated and never retroactively change.
//
// curveAmp / hilliness dials scale the noise *at generation time* and are then frozen
// into the cached geometry, which keeps (seed + actions + dial schedule) fully replayable.

import { fbm1, fbm2 } from './prng.js';

const STEP = 2.0;              // metres between cached centerline samples
const MAX_CURVATURE = 0.035;   // rad/m at full curveAmp
const CURVE_FREQ = 0.010;      // spatial frequency of the winding
const ROAD_WIDTH = 9.0;
const MAX_GRADE = 0.16;        // 16% maximum engineered road grade
const MAX_GRADE_RATE = 0.012;  // grade change per metre (vertical curvature)
const GRADE_RESPONSE = 20;     // metres over which the road approaches terrain height
const SHOULDER = 3;
const ROLL_SPAN = 46;

// Shared terrain heightfield. The road drapes over THIS field (its centerline y is
// H at the centerline), and the off-road terrain is the same field — so road and
// ground meet smoothly instead of forming a trench.
//
// Two scales: broad mountains + medium rolling hills, so there is real relief within
// the view distance rather than an almost-flat plain.
const H_FREQ = 0.0026;   // broad ridges / mountains
const H_AMP = 62;
const H_FREQ2 = 0.011;   // medium rolling hills
const H_AMP2 = 16;
const H_SEED = 6373;
export function heightField(x, z, seed, hilliness) {
  const s = (seed + H_SEED) >>> 0;
  const broad = fbm2(x * H_FREQ, z * H_FREQ, s, 4) * H_AMP;
  const rolling = fbm2(x * H_FREQ2, z * H_FREQ2, s + 4099, 3) * H_AMP2;
  return (broad + rolling) * hilliness;
}

/**
 * The single rendered/physical ground surface around the road. Callers provide the
 * nearest centerline sample and signed lateral distance so distant scatter can use its
 * own road-distance hint instead of searching around the car.
 */
export function terrainSurfaceHeight(center, lateral, x, z, seed, hilliness) {
  const half = center.width / 2;
  const off = smoothstep(half + SHOULDER, half + SHOULDER + ROLL_SPAN, Math.abs(lateral));
  const terrain = heightField(x, z, seed, hilliness);
  const micro = fbm2(x * 0.12, z * 0.12, seed + 51, 2) * 0.35 * off;
  return center.y + (terrain - center.y) * off + micro;
}

// Global sea level. Terrain below this is underwater (the renderer draws a water
// plane); the road is lifted to stay just above it (a causeway across low ground).
export const SEA_LEVEL = -16;
const CAUSEWAY = 2.5;

export class Road {
  constructor(seed) {
    this.seed = seed >>> 0;
    // Cache of samples: parallel arrays for cheap interpolation.
    this._s = [{ d: 0, x: 0, z: 0, y: 0, heading: 0, curvature: 0, grade: 0, width: ROAD_WIDTH }];
  }

  get length() {
    return this._s[this._s.length - 1].d;
  }

  /** Extend the baked centerline until it reaches at least distance `d`, using current dials. */
  ensureUpTo(d, dials) {
    const curveAmp = dials ? dials.curveAmp : 0.5;
    const hilliness = dials ? dials.hilliness : 0.4;
    // The seed node used to start at y=0 regardless of the terrain. On worlds whose
    // first generated sample was much lower, the car dropped several metres during
    // its first tick and the camera read that discontinuity as a violent bounce.
    if (this._s.length === 1) {
      this._s[0].y = Math.max(heightField(0, 0, this.seed, hilliness), SEA_LEVEL + CAUSEWAY);
    }
    let last = this._s[this._s.length - 1];
    while (last.d < d) {
      const nd = last.d + STEP;
      const curvature = fbm1(nd * CURVE_FREQ, this.seed, 4) * MAX_CURVATURE * curveAmp;
      const heading = last.heading + curvature * STEP;
      const x = last.x + Math.sin(heading) * STEP;
      const z = last.z + Math.cos(heading) * STEP;
      // Treat the terrain as a target, then fit an engineered vertical profile toward
      // it. Directly sampling terrain at every node made a piecewise-linear road whose
      // slope changed abruptly every two metres; those fake crests repeatedly defeated
      // the gravity/contact test. Bounded grade and grade-rate give gravity a meaningful
      // curvature to act against while still following broad landforms.
      const targetY = Math.max(heightField(x, z, this.seed, hilliness), SEA_LEVEL + CAUSEWAY);
      const desiredGrade = clamp((targetY - last.y) / GRADE_RESPONSE, -MAX_GRADE, MAX_GRADE);
      const gradeDelta = clamp(desiredGrade - last.grade, -MAX_GRADE_RATE * STEP, MAX_GRADE_RATE * STEP);
      const grade = clamp(last.grade + gradeDelta, -MAX_GRADE, MAX_GRADE);
      const y = last.y + grade * STEP;
      last = { d: nd, x, z, y, heading, curvature, grade, width: ROAD_WIDTH };
      this._s.push(last);
    }
  }

  /** Sample the centerline at arc-length `d` (linearly interpolated between cached nodes). */
  sampleAt(d) {
    const s = this._s;
    if (d <= 0) return { ...s[0] };
    if (d >= this.length) return { ...s[s.length - 1] };
    const idx = Math.floor(d / STEP);
    const a = s[idx];
    const b = s[idx + 1] || a;
    const t = (d - a.d) / STEP;
    return {
      d,
      x: a.x + (b.x - a.x) * t,
      z: a.z + (b.z - a.z) * t,
      y: a.y + (b.y - a.y) * t,
      heading: a.heading + (b.heading - a.heading) * t,
      curvature: a.curvature + (b.curvature - a.curvature) * t,
      grade: a.grade + (b.grade - a.grade) * t,
      width: ROAD_WIDTH,
    };
  }

  /** Unit tangent (world XZ) at arc-length `d`. */
  tangentAt(d) {
    const h = this.sampleAt(d).heading;
    return { x: Math.sin(h), z: Math.cos(h) };
  }

  /**
   * Nearest point on the centerline to world (x, z), searched in a window around
   * `dHint` (the car's last known arc-length). Returns arc-length, the point, the
   * tangent, and the signed lateral offset (+right of travel direction).
   */
  nearest(x, z, dHint = 0, window = 40) {
    const from = Math.max(0, dHint - window);
    const to = dHint + window;
    let bestD = dHint;
    let bestDist = Infinity;
    for (let d = from; d <= to; d += STEP * 0.5) {
      const p = this.sampleAt(d);
      const dist = (p.x - x) ** 2 + (p.z - z) ** 2;
      if (dist < bestDist) {
        bestDist = dist;
        bestD = d;
      }
    }
    const p = this.sampleAt(bestD);
    const tan = this.tangentAt(bestD);
    // Signed lateral offset: cross product of tangent and (car - center) in XZ.
    const dx = x - p.x;
    const dz = z - p.z;
    const lateral = tan.x * dz - tan.z * dx; // +ve => car is to the right of travel
    return { d: bestD, point: p, tangent: tan, lateral };
  }

  /** Deterministic roadside object list within an arc-length window. Sparse (SIM.md §3). */
  props(fromD, toD, dials) {
    const out = [];
    const spacing = 22; // metres between candidate slots
    const i0 = Math.ceil(fromD / spacing);
    const i1 = Math.floor(toD / spacing);
    for (let i = i0; i <= i1; i++) {
      const r = fbm1(i * 3.1 + 0.5, this.seed + 4211, 2) * 0.5 + 0.5; // [0,1]
      if (r < 0.55) continue; // keep it sparse
      const d = i * spacing;
      const c = this.sampleAt(d);
      const tan = this.tangentAt(d);
      const side = (i % 2 === 0) ? 1 : -1;
      const off = c.width * 0.6 + 2 + r * 8;
      // normal = tangent rotated 90deg in XZ
      const nx = tan.z * side;
      const nz = -tan.x * side;
      const kind = Math.floor(r * 997) % 4; // 0 monolith,1 arch,2 tree,3 crystal
      out.push({
        d,
        x: c.x + nx * off,
        z: c.z + nz * off,
        y: this.sampleAt(d).y,   // centerline elevation; renderer adds off-road hill
        lateral: off,            // metres from the centerline (for terrain seating)
        scale: 1 + r * 2.5,
        kind,
        seedR: r,
      });
    }
    return out;
  }
}

function clamp(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v;
}

function smoothstep(edge0, edge1, x) {
  const t = clamp((x - edge0) / (edge1 - edge0), 0, 1);
  return t * t * (3 - 2 * t);
}
