// Driving policies for dataset generation. Deterministic in (name, seed): each
// returns, per fixed step, the action sim.step() consumes plus the keys "pressed".
// Keyboard policies go through core/input.js, the same smoothing the demo applies to a
// human player, so recorded actions match what someone at the keyboard produces.
//
//   const pol = makePolicy('keys_explore', 1234, sim);
//   const { action, keys } = pol.act(sim);   // keys: e.g. "wa" ("" = nothing held)
//
// Profiles:
//   cruise        the game-style autopilot (continuous control) in the left lane, with
//                 a slowly varying cruise speed. Matches the reference footage.
//   keys_lane     a simulated human holding the lane on WASD: reaction delay, A/D taps,
//                 W/S to manage speed.
//   keys_explore  random maneuvers every few seconds (swerves, hard turns, off-road
//                 excursions, braking to a stop, reversing, coasting), recovering to the
//                 road when far off. Counterfactual actions for steerability.
//   lane_change   keyboard lane-keeping that switches lanes and cruise speed.
//   dial_mix      cruise or keys_lane under randomized, slowly changing world dials
//                 (time of day, fog, rain, snow, biome, terrain).

import { mulberry32 } from './prng.js';
import { keysToAction } from './input.js';

export const POLICIES = ['cruise', 'keys_lane', 'keys_explore', 'lane_change', 'dial_mix'];

const LEFT_LANE = -1.8, RIGHT_LANE = 1.8;

export function makePolicy(name, seed, sim) {
  if (!POLICIES.includes(name)) throw new Error(`unknown policy ${name}`);
  const r = mulberry32(seed >>> 0);
  const uni = (a, b) => a + (b - a) * r();
  const dt = sim.dt;
  let prev = { steer: 0, throttle: 0 };
  let t = 0;

  // --- building blocks -----------------------------------------------------------
  const pressed = (k) => {
    prev = keysToAction(prev, k, dt);
    return { action: { ...prev }, keys: (k.w ? 'w' : '') + (k.a ? 'a' : '') + (k.s ? 's' : '') + (k.d ? 'd' : '') };
  };

  // A human holding a lane on the keyboard: every reaction interval, compare the
  // steering a lane-keeper wants with the deadband and press A/D (or nothing); manage
  // speed with W (pulsed to hold a cruise) and S before tight bends.
  const human = { next: 0, keys: { w: false, a: false, s: false, d: false }, db: 0.12, duty: 0.6 };
  const keyLane = (sim, offset, cruise) => {
    if (t >= human.next) {
      human.next = t + uni(0.1, 0.28);
      human.db = uni(0.06, 0.2);
      const want = sim.autopilotAction({ offset, cruise });
      const k = { w: false, a: false, s: false, d: false };
      if (want.steer > human.db) k.a = true;
      else if (want.steer < -human.db) k.d = true;
      if (want.throttle >= 0.6) k.w = true;
      else if (want.throttle < 0) k.s = r() < 0.7;
      else k.w = r() < human.duty;   // feathering W to hold speed
      human.keys = k;
    }
    return pressed(human.keys);
  };

  const offsetOf = (sim) => sim.state.road.offset;
  const headingErr = (sim) => {
    const st = sim.state;
    const road = Math.atan2(st.road.tangent.x, st.road.tangent.z);
    let d = (st.car.heading - road) % (Math.PI * 2);
    if (d > Math.PI) d -= Math.PI * 2;
    if (d < -Math.PI) d += Math.PI * 2;
    return d;
  };

  // --- profiles ------------------------------------------------------------------
  if (name === 'cruise') {
    let cruise = uni(13, 20), nextChange = uni(12, 35);
    return {
      act(sim) {
        t += dt;
        if (t > nextChange) { cruise = uni(11, 22); nextChange = t + uni(12, 35); }
        const a = sim.autopilotAction({ offset: LEFT_LANE, cruise });
        prev = a;
        return { action: a, keys: '' };
      },
    };
  }

  if (name === 'keys_lane') {
    let cruise = uni(12, 20), nextChange = uni(15, 40);
    human.duty = uni(0.45, 0.75);
    return {
      act(sim) {
        t += dt;
        if (t > nextChange) { cruise = uni(10, 22); nextChange = t + uni(15, 40); }
        return keyLane(sim, LEFT_LANE, cruise);
      },
    };
  }

  if (name === 'lane_change') {
    let lane = LEFT_LANE, cruise = uni(12, 20), nextLane = uni(4, 10), nextCruise = uni(8, 20);
    return {
      act(sim) {
        t += dt;
        if (t > nextLane) { lane = lane === LEFT_LANE ? RIGHT_LANE : LEFT_LANE; nextLane = t + uni(4, 12); }
        if (t > nextCruise) { cruise = uni(8, 24); nextCruise = t + uni(8, 20); }
        return keyLane(sim, lane, cruise);
      },
    };
  }

  if (name === 'dial_mix') {
    const keyboard = r() < 0.5;
    let cruise = uni(12, 20);
    const randomDials = () => ({
      timeOfDay: uni(0, Math.PI * 2),
      fog: r() < 0.35 ? uni(0.3, 0.9) : uni(0, 0.3),
      rain: r() < 0.25 ? uni(0.3, 1) : 0,
      snow: r() < 0.15 ? uni(0.3, 1) : 0,
      biomeX: uni(0.1, 0.95), biomeY: uni(0.2, 0.8),
    });
    sim.snapDials({ ...randomDials(), hilliness: uni(0.35, 0.8), curveAmp: uni(0.3, 0.8) });
    let nextDials = uni(20, 50);
    return {
      act(sim) {
        t += dt;
        if (t > nextDials) { sim.setDials(randomDials()); nextDials = t + uni(20, 50); cruise = uni(11, 21); }
        if (keyboard) return keyLane(sim, LEFT_LANE, cruise);
        const a = sim.autopilotAction({ offset: LEFT_LANE, cruise });
        prev = a;
        return { action: a, keys: '' };
      },
    };
  }

  // keys_explore: a maneuver scheduler with recovery.
  const MANEUVERS = [
    ['lane', 4], ['straight', 1.2], ['coast', 1], ['brake_stop', 0.7], ['reverse', 0.4],
    ['swerve', 1.4], ['hard_turn', 0.6], ['offroad', 0.4], ['lane_change', 1.2],
  ];
  const total = MANEUVERS.reduce((s, m) => s + m[1], 0);
  const pick = () => { let x = r() * total; for (const [m, w] of MANEUVERS) { if ((x -= w) <= 0) return m; } return 'lane'; };
  let m = 'lane', until = uni(2, 5), lane = LEFT_LANE, cruise = uni(10, 20), side = 1, phase = 0, swerveNext = 0;
  const start = (name) => {
    m = name; phase = 0; side = r() < 0.5 ? 1 : -1; swerveNext = 0;
    until = t + ({ lane: uni(3, 8), straight: uni(1, 3), coast: uni(1, 3), brake_stop: uni(2.5, 4.5), reverse: uni(1.5, 3),
      swerve: uni(2, 4), hard_turn: uni(0.6, 2), offroad: uni(2.5, 4.5), lane_change: uni(3, 6), recover: 30 })[name];
    if (name === 'lane') { lane = r() < 0.8 ? LEFT_LANE : RIGHT_LANE; cruise = uni(8, 22); }
    if (name === 'lane_change') lane = lane === LEFT_LANE ? RIGHT_LANE : LEFT_LANE;
  };
  return {
    act(sim) {
      t += dt;
      const off = offsetOf(sim), herr = Math.abs(headingErr(sim)), speed = sim.state.car.speed;
      // Recover when far off the road or facing the wrong way (but let 'offroad' and
      // 'reverse' run their course within limits).
      if (m !== 'recover' && (Math.abs(off) > (m === 'offroad' ? 12 : 7) || (herr > 1.4 && m !== 'reverse'))) start('recover');
      if (m === 'recover' && Math.abs(off) < 3 && herr < 0.3) start('lane');
      if (t > until) start(m === 'recover' ? 'recover' : pick());
      const k = { w: false, a: false, s: false, d: false };
      switch (m) {
        case 'lane': case 'lane_change': return keyLane(sim, lane, cruise);
        case 'recover': return keyLane(sim, LEFT_LANE, 8);
        case 'straight': k.w = true; break;
        case 'coast': break;
        case 'brake_stop': if (speed > 0.3) k.s = true; break;
        case 'reverse': if (phase === 0 && speed > 0.3) k.s = true; else { phase = 1; k.s = true; if (r() < 0.02) side = -side; k[side > 0 ? 'a' : 'd'] = r() < 0.4; } break;
        case 'swerve': k.w = r() < 0.8; if (t > swerveNext) { side = -side; swerveNext = t + uni(0.25, 0.7); } k[side > 0 ? 'a' : 'd'] = true; break;
        case 'hard_turn': k.w = true; k[side > 0 ? 'a' : 'd'] = true; break;
        case 'offroad':
          // Leave the road to one side, then drive across the grass.
          k.w = true;
          if (phase === 0) { k[side > 0 ? 'a' : 'd'] = true; if (Math.abs(off) > 6) phase = 1; }
          else if (r() < 0.15) k[r() < 0.5 ? 'a' : 'd'] = true;
          break;
      }
      return pressed(k);
    },
  };
}
