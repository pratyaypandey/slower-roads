// Arcade car physics. Fixed-timestep, deterministic (SIM.md §6): given the same
// state + action + dt + dials it always produces the same next state.
//
// A light bicycle-ish model: throttle drives longitudinal speed, steering yaws the
// heading (scaled by speed so you can't pirouette in place), and low `friction`
// bleeds grip into lateral slip (skid). `friction` and `speedFeel` have real
// dynamical effect here from day one; `gravity` is applied in the vertical model in
// sim.js. So the physics dials are live state, not just recorded labels.

const ACCEL = 10;       // m/s^2 at full throttle
const BRAKE = 20;       // m/s^2 at full brake
const MAX_SPEED = 28;   // m/s at speedFeel = 1 (~100 km/h) — brisk but not a racer
const MAX_REVERSE = 8;
const MAX_YAW = 0.72;   // rad/s before speed attenuation
const SLIP_DECAY = 3.5; // how fast lateral slip washes out

// Per-surface driving feel. `rollDrag` is the rolling resistance (higher = slower),
// `gripMul` scales cornering grip, `maxScale` caps top speed. Tarmac is fast and
// planted; grass/dirt drags you down and gets loose; snow is fast but slippery.
export const SURFACE = {
  road:  { rollDrag: 0.55, gripMul: 1.00, maxScale: 1.00 },
  grass: { rollDrag: 3.20, gripMul: 0.68, maxScale: 0.55 },
  sand:  { rollDrag: 4.40, gripMul: 0.55, maxScale: 0.45 },
  snow:  { rollDrag: 1.40, gripMul: 0.42, maxScale: 0.80 },
};

export function makeCar() {
  return {
    x: 0, z: 0, y: 0,
    heading: 0,   // radians; 0 = +z
    speed: 0,     // m/s along heading
    slip: 0,      // lateral velocity (m/s), +right
    airborne: 0,  // vertical velocity when launched (reserved for gravity dial)
    vy: 0,
  };
}

/**
 * Advance the car one fixed tick. Mutates `car`.
 * @param car   car state (see makeCar)
 * @param action {steer:[-1,1], throttle:[-1,1]}
 * @param dt    fixed timestep (s)
 * @param dials smoothed dial values (uses gravity, friction, speedFeel)
 * @param surface one of SURFACE (defaults to road) — the ground under the car
 */
export function stepCar(car, action, dt, dials, surface = SURFACE.road) {
  const steer = clamp(action.steer ?? 0, -1, 1);
  const throttle = clamp(action.throttle ?? 0, -1, 1);
  const grip = dials.friction * surface.gripMul;   // combined world + surface grip
  const maxSpeed = MAX_SPEED * dials.speedFeel * surface.maxScale;

  // --- Longitudinal ---------------------------------------------------------
  if (throttle >= 0) {
    car.speed += throttle * ACCEL * dt;
  } else {
    // Brake toward zero, then reverse.
    if (car.speed > 0) car.speed += throttle * BRAKE * dt;
    else car.speed += throttle * ACCEL * 0.5 * dt;
  }
  car.speed -= surface.rollDrag * car.speed * dt;          // surface rolling drag
  car.speed = clamp(car.speed, -MAX_REVERSE, maxSpeed);

  // --- Yaw ------------------------------------------------------------------
  // Steering authority ramps in with speed (no in-place spins) and with grip.
  const speed = Math.abs(car.speed);
  const speedFactor = speed / (speed + 7);
  // Keyboard input eventually reaches full lock, so reduce steering authority as
  // speed rises. The shaped input leaves room for small corrections while retaining
  // enough lock for low-speed recovery and the tightest generated roads.
  const steerShape = Math.sign(steer) * Math.pow(Math.abs(steer), 1.25);
  const highSpeedT = smoothstep(7, MAX_SPEED, speed);
  const speedAuthority = 1 - highSpeedT * 0.48;
  const dir = car.speed >= 0 ? 1 : -1;
  const yawRate = steerShape * MAX_YAW * speedFactor * speedAuthority * grip * dir;
  car.heading += yawRate * dt;

  // --- Lateral slip (skid) --------------------------------------------------
  // Cornering demands centripetal force; grip below 1 lets some leak into slip.
  const corneringLoad = yawRate * car.speed;
  car.slip += corneringLoad * (1 - grip) * dt;
  car.slip -= SLIP_DECAY * grip * car.slip * dt;
  car.slip = clamp(car.slip, -14, 14);

  // --- Integrate position ---------------------------------------------------
  const fx = Math.sin(car.heading);
  const fz = Math.cos(car.heading);
  const rx = fz;   // right vector = forward rotated -90deg in XZ
  const rz = -fx;
  car.x += (fx * car.speed + rx * car.slip) * dt;
  car.z += (fz * car.speed + rz * car.slip) * dt;

  return car;
}

function clamp(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v;
}

function smoothstep(edge0, edge1, x) {
  const t = clamp((x - edge0) / (edge1 - edge0), 0, 1);
  return t * t * (3 - 2 * t);
}
