// Top-level sim: owns state (prng, road, car, params) and exposes the loop the
// data exporter and browser both drive — step(action) advances one fixed dt,
// render() draws the current state through whatever renderer the caller wired.
// Everything here is deterministic given (seed, params, action sequence).

import { createPrng } from "./prng.js";
import { createRoad } from "./road.js";
import { createCar, stepCar, DT } from "./car.js";
import { createWorld, updateWorld } from "./world.js";
import { makeParams } from "./params.js";

export { DT };

// THREE is optional: pass it to get a renderable scene graph (browser + pixel
// data-gen); omit it for a pure state-only sim (determinism proof, dynamics
// that never need pixels). The state trajectory is itself the sparsest scene
// representation, so much of the oracle harness needs no renderer at all.
export function createSim(THREE, { seed = 1, params = {} } = {}) {
  const resolvedParams = makeParams(params);
  const prng = createPrng(seed);
  const road = createRoad(prng);
  const car = createCar();
  const world = THREE ? createWorld(THREE, road, resolvedParams) : null;

  // Start the car on the road, aligned with the opening heading.
  const start = road.sample(0);
  car.x = start.x;
  car.z = start.z;
  car.heading = start.heading;

  let frame = 0;

  function step(action = {}) {
    stepCar(car, action, resolvedParams);
    frame++;
    return car;
  }

  function render(renderer) {
    if (!world) throw new Error("createSim was called without THREE; no scene to render");
    updateWorld(world, car);
    renderer.render(world.scene, world.camera);
  }

  return {
    step,
    render,
    get state() {
      return { x: car.x, z: car.z, heading: car.heading, speed: car.speed, frame };
    },
    params: resolvedParams,
    road,
    world,
  };
}
