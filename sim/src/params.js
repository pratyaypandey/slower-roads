// Explicit world parameters. These are literal values the sim reads, which is
// what buys us free contrastive pairs later: render the same seed with two
// different param sets and the label is just the diff. Only gravity/friction
// are wired into dynamics for the skeleton; weather/timeOfDay are declared now
// so the data schema is stable before the visuals that consume them exist.

export const DEFAULT_PARAMS = {
  gravity: 9.81, // m/s^2
  friction: 0.9, // ground grip coefficient, 0..1
  weather: 0, // 0 = clear .. 1 = storm (visual-only for now)
  timeOfDay: 0.5, // 0 = midnight, 0.5 = noon, 1 = midnight (visual-only for now)
};

export function makeParams(overrides = {}) {
  return { ...DEFAULT_PARAMS, ...overrides };
}
