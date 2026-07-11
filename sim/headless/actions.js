// Deterministic scripted action sequence. Uses its own seeded PRNG (independent
// of the world seed) so the "same actions" half of the determinism proof is
// itself reproducible — replaying seed S with this script always applies the
// identical inputs.

import { createPrng } from "../src/prng.js";

export function makeActionSequence(length, seed = "actions") {
  const prng = createPrng(seed);
  const actions = new Array(length);
  let steer = 0;
  for (let i = 0; i < length; i++) {
    // Smooth, correlated steering (random walk) rather than per-frame noise, so
    // the car drives in believable arcs.
    steer += prng.signed(0.15);
    steer = Math.max(-1, Math.min(1, steer * 0.9));
    actions[i] = { throttle: 0.8, brake: 0, steer };
  }
  return actions;
}
