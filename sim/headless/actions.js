// Deterministic scripted action sequence. Uses its own seeded PRNG (independent
// of the world seed) so the "same actions" half of the determinism proof is
// itself reproducible — replaying seed S with this script always applies the
// identical inputs.

import { createPrng } from "../src/prng.js";

export function makeActionSequence(length, seed = "actions") {
  const prng = createPrng(seed);
  const actions = new Array(length);
  let steer = 0;
  let throttle = 0.8;
  for (let i = 0; i < length; i++) {
    // Smooth, correlated steering that actually explores the full lock range —
    // weaker mean-reversion (0.97) so the car holds real turns, not a near-zero
    // wobble. Occasional throttle/brake changes so the model sees the whole
    // action space rather than one constant token.
    steer += prng.signed(0.2);
    steer = Math.max(-1, Math.min(1, steer * 0.97));
    if (prng.next() < 0.03) throttle = prng.range(0.2, 1); // ~every 33 steps
    const brake = prng.next() < 0.02 ? prng.range(0.3, 1) : 0; // occasional brake
    actions[i] = { throttle, brake, steer };
  }
  return actions;
}
