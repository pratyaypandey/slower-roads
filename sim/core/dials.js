// The dials: the orthogonal, continuous generative factors of the world.
//
// These are the steering targets (SIM.md §3). Each is a plain scalar (or a small
// vector) with a smooth param->pixel effect. At inference these become the knobs
// the player turns; at data-gen time we sample their *product*, including
// physically-impossible combinations, so the learned steering vectors disentangle.
//
// Every dial is smoothed toward its target each step (no discrete pops) so that the
// param->pixel map stays C1-continuous -> latent directions stay linear -> CAA works.

/**
 * Dial schema. `min`/`max` bound the UI + sampling range. `smooth` is the
 * exponential-smoothing half-life-ish rate per second (0 = instant, higher = snappier).
 * `wrap` marks angular dials that wrap around their range.
 */
export const DIAL_SCHEMA = {
  // --- Atmosphere / lighting -------------------------------------------------
  timeOfDay: { min: 0, max: Math.PI * 2, default: 1.6, smooth: 1.5, wrap: true },
  fog:       { min: 0, max: 1,           default: 0.25, smooth: 2.0 },
  rain:      { min: 0, max: 1,           default: 0.0,  smooth: 2.0 },
  snow:      { min: 0, max: 1,           default: 0.0,  smooth: 2.0 },

  // --- Biome (continuous 2D plane) ------------------------------------------
  biomeX:    { min: 0, max: 1, default: 0.5, smooth: 1.0 }, // arid <-> lush
  biomeY:    { min: 0, max: 1, default: 0.5, smooth: 1.0 }, // earthly <-> alien

  // --- Sky / backdrop --------------------------------------------------------
  starDensity: { min: 0, max: 1, default: 0.3, smooth: 1.5 },
  moons:       { min: 0, max: 3, default: 1,   smooth: 1.0 },
  aurora:      { min: 0, max: 1, default: 0.0, smooth: 1.5 },

  // --- Road geometry ---------------------------------------------------------
  curveAmp:  { min: 0, max: 1, default: 0.5, smooth: 0.5 }, // how winding
  hilliness: { min: 0, max: 1, default: 0.55, smooth: 0.5 }, // elevation amplitude

  // --- Physics (recorded from day one; steered only at M6) -------------------
  gravity:   { min: 0.2, max: 2, default: 1.0, smooth: 1.0 },
  friction:  { min: 0.2, max: 1, default: 1.0, smooth: 1.0 },
  speedFeel: { min: 0.5, max: 2, default: 1.0, smooth: 1.0 },
};

export const DIAL_KEYS = Object.keys(DIAL_SCHEMA);

/** Build a full dial vector from defaults, overlaying any provided values (clamped). */
export function makeDials(overrides = {}) {
  const d = {};
  for (const k of DIAL_KEYS) {
    const spec = DIAL_SCHEMA[k];
    d[k] = clampDial(k, overrides[k] !== undefined ? overrides[k] : spec.default);
  }
  return d;
}

export function clampDial(key, value) {
  const spec = DIAL_SCHEMA[key];
  if (!spec) return value;
  if (spec.wrap) {
    const span = spec.max - spec.min;
    return spec.min + ((((value - spec.min) % span) + span) % span);
  }
  return Math.min(spec.max, Math.max(spec.min, value));
}

/**
 * Smoothly advance `current` dials toward `target` over dt seconds.
 * Exponential smoothing; wrap-aware for angular dials so time-of-day takes the
 * short way around. Mutates and returns `current`.
 */
export function smoothDials(current, target, dt) {
  for (const k of DIAL_KEYS) {
    const spec = DIAL_SCHEMA[k];
    const rate = spec.smooth;
    const alpha = rate <= 0 ? 1 : 1 - Math.exp(-rate * dt);
    let cur = current[k];
    let tgt = target[k];
    if (spec.wrap) {
      const span = spec.max - spec.min;
      let delta = ((tgt - cur + span / 2) % span + span) % span - span / 2;
      current[k] = clampDial(k, cur + delta * alpha);
    } else {
      current[k] = cur + (tgt - cur) * alpha;
    }
  }
  return current;
}
