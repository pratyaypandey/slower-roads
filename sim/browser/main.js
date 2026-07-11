// Browser entry: mounts the shared sim core to a canvas and drives it with WASD
// on a fixed-timestep accumulator, so on-screen physics matches the headless
// data-gen exactly. Same createSim, same DT — the only difference from the
// exporter is a real WebGLRenderer and live keyboard input.

import * as THREE from "three";
import { createSim, DT } from "../src/sim.js";

const canvas = document.getElementById("view");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });

const sim = createSim(THREE, { seed: 1 });

const keys = new Set();
addEventListener("keydown", (e) => keys.add(e.code));
addEventListener("keyup", (e) => keys.delete(e.code));

function currentAction() {
  const throttle = keys.has("KeyW") ? 1 : 0;
  const brake = keys.has("KeyS") ? 1 : 0;
  let steer = 0;
  if (keys.has("KeyA")) steer -= 1;
  if (keys.has("KeyD")) steer += 1;
  return { throttle, brake, steer };
}

function resize() {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  renderer.setSize(w, h, false);
  sim.world.camera.aspect = w / h;
  sim.world.camera.updateProjectionMatrix();
}
addEventListener("resize", resize);
resize();

// Fixed-timestep loop: accumulate wall-clock time and step the sim in exact DT
// increments so physics stays deterministic regardless of display refresh rate.
let last = performance.now();
let acc = 0;
function frame(now) {
  acc += (now - last) / 1000;
  last = now;
  const action = currentAction();
  while (acc >= DT) {
    sim.step(action);
    acc -= DT;
  }
  sim.render(renderer);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
