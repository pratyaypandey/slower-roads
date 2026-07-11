// Procedural road centerline. The path is a polyline advanced by a heading that
// drifts via seeded noise, so a given seed always produces the same road. We can
// sample position/tangent at any arc-length s, which the car and (later) the
// sim-anchor conditioning both read from.

const SEGMENT_LEN = 2; // metres between centerline samples
const MAX_TURN = 0.035; // max heading change per segment (radians)

export function createRoad(prng, { segments = 4096, width = 8 } = {}) {
  const points = new Array(segments);
  const headings = new Float32Array(segments);

  let x = 0;
  let z = 0;
  let heading = 0; // radians, 0 = +z
  // Low-frequency curvature target the heading eases toward, so turns feel like
  // sweeping bends rather than per-segment jitter.
  let curvature = 0;

  for (let i = 0; i < segments; i++) {
    points[i] = { x, z };
    headings[i] = heading;

    curvature += prng.signed(0.006);
    curvature *= 0.96; // decay keeps the road from spiralling
    heading += Math.max(-MAX_TURN, Math.min(MAX_TURN, curvature));

    x += Math.sin(heading) * SEGMENT_LEN;
    z += Math.cos(heading) * SEGMENT_LEN;
  }

  function sample(s) {
    const f = s / SEGMENT_LEN;
    const i = Math.max(0, Math.min(segments - 2, Math.floor(f)));
    const t = f - i;
    const a = points[i];
    const b = points[i + 1];
    return {
      x: a.x + (b.x - a.x) * t,
      z: a.z + (b.z - a.z) * t,
      heading: headings[i] + (headings[i + 1] - headings[i]) * t,
    };
  }

  return { points, headings, width, segmentLen: SEGMENT_LEN, length: segments * SEGMENT_LEN, sample };
}
