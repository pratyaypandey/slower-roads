// Car physics on a fixed timestep. Simple bicycle-ish model: throttle/brake move
// speed, steer turns heading scaled by speed, friction bleeds speed off. Fixed
// dt (not wall-clock) is what makes rollouts reproducible — the determinism proof
// relies on this never reading real time.

export const DT = 1 / 30; // seconds per step; also the training frame interval

const ENGINE = 14; // forward accel at full throttle (m/s^2)
const BRAKE = 22; // decel at full brake
const MAX_SPEED = 28;
const STEER_RATE = 1.6; // radians/s at reference speed

export function createCar() {
  return { x: 0, z: 0, heading: 0, speed: 0 };
}

// action: { throttle: 0..1, brake: 0..1, steer: -1..1 }
export function stepCar(car, action, params) {
  const throttle = clamp(action.throttle ?? 0, 0, 1);
  const brake = clamp(action.brake ?? 0, 0, 1);
  const steer = clamp(action.steer ?? 0, -1, 1);

  car.speed += throttle * ENGINE * DT;
  car.speed -= brake * BRAKE * DT;

  // Rolling resistance scales with (1 - friction): more grip, less coast loss.
  const drag = (1 - params.friction) * 6 + 0.4;
  car.speed -= drag * DT;
  car.speed = clamp(car.speed, 0, MAX_SPEED);

  // Steering authority fades toward zero at a standstill.
  const speedFactor = Math.min(1, car.speed / 8);
  car.heading += steer * STEER_RATE * speedFactor * DT;

  car.x += Math.sin(car.heading) * car.speed * DT;
  car.z += Math.cos(car.heading) * car.speed * DT;

  return car;
}

function clamp(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v;
}
