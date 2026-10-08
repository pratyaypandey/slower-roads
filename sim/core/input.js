// Keyboard -> continuous action, shared by the demo (a human at the keys) and the
// dataset policies (simulated keystrokes), so recorded actions match what a player
// actually produces: steering eases toward the held direction and recentres faster
// on release; throttle eases toward W/S.

export const STEER_RISE = 7.5;      // 1/s toward a held A/D
export const STEER_RETURN = 11;     // 1/s back to centre on release
export const THROTTLE_RATE = 8;     // 1/s toward W/S
export const KEYBOARD_STEER = 0.82; // steer magnitude of a held A/D

export function approachExp(cur, target, rate, dt) {
  return cur + (target - cur) * (1 - Math.exp(-rate * dt));
}

/**
 * One fixed step of keyboard smoothing. `keys` = { w, a, s, d } booleans (A = left,
 * which is +steer). Returns the next { steer, throttle } given the previous action.
 */
export function keysToAction(prev, keys, dt) {
  const steerTarget = ((keys.a ? 1 : 0) - (keys.d ? 1 : 0)) * KEYBOARD_STEER;
  const throttleTarget = (keys.w ? 1 : 0) - (keys.s ? 1 : 0);
  return {
    steer: approachExp(prev.steer, steerTarget, steerTarget === 0 ? STEER_RETURN : STEER_RISE, dt),
    throttle: approachExp(prev.throttle, throttleTarget, THROTTLE_RATE, dt),
  };
}
