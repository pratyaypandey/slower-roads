// SlowSim demo — wires the deterministic core to the RGB renderer so a human can
// drive the stylized-dream world and turn the generative dials live.
//
// This is an internal iteration tool (SIM.md §8/§9): clarity over polish. It does not
// touch the core or renderer — it only consumes their public API (see sim/API.md).

import { SlowSim, DIAL_SCHEMA, DIAL_KEYS } from '../core/index.js';
import { SimRenderer } from '../render/renderer.js';

const canvas = document.getElementById('view');
const FIXED = 1 / 60;              // fixed simulation step — decoupled from display rate
const sim = new SlowSim({ seed: 42, dt: FIXED });
const renderer = new SimRenderer(canvas);

// --- Display sizing (full-res; the model's small frame comes from renderer.capture) ---
const stage = document.getElementById('stage');
function resize() {
  const w = stage.clientWidth;
  const h = stage.clientHeight;
  if (w > 0 && h > 0) renderer.setDisplaySize(w, h);
}
window.addEventListener('resize', resize);
resize();

// --- Keyboard driving ------------------------------------------------------------
// Keys give discrete -1/0/1 targets; we ramp the actual action toward the target so
// the control feels analog rather than binary (steer/throttle are continuous [-1,1]).
const keys = Object.create(null);
const DRIVE_KEYS = new Set([
  'w', 'a', 's', 'd',
  'arrowup', 'arrowdown', 'arrowleft', 'arrowright',
]);

window.addEventListener('keydown', (e) => {
  const k = e.key.toLowerCase();
  keys[k] = true;
  if (DRIVE_KEYS.has(k)) e.preventDefault(); // keep arrows from scrolling the panel
});
window.addEventListener('keyup', (e) => { keys[e.key.toLowerCase()] = false; });
// Drop held keys if focus leaves the window, so the car doesn't drive off unattended.
window.addEventListener('blur', () => { for (const k in keys) keys[k] = false; });

let steer = 0;
let throttle = 0;
const STEER_RISE = 7.5;
const STEER_RETURN = 11;
const THROTTLE_RATE = 8;
const KEYBOARD_STEER = 0.82;

function approachExp(cur, target, rate, dt) {
  return cur + (target - cur) * (1 - Math.exp(-rate * dt));
}

// --- Dial sliders (generic — iterate the schema so new dials appear automatically) -
const dialEls = {}; // key -> { slider, valEl }
const dialsRoot = document.getElementById('dials');

function fmtDial(key, v) {
  const spec = DIAL_SCHEMA[key];
  if (spec.wrap) return `${v.toFixed(2)} rad`; // timeOfDay etc.
  return v.toFixed(2);
}

for (const key of DIAL_KEYS) {
  const spec = DIAL_SCHEMA[key];
  const row = document.createElement('div');
  row.className = 'dial';

  const label = document.createElement('div');
  label.className = 'dial-label';
  const name = document.createElement('span');
  name.className = 'dial-name';
  name.textContent = key;
  const val = document.createElement('span');
  val.className = 'dial-val';
  val.textContent = fmtDial(key, spec.default);
  label.append(name, val);

  const slider = document.createElement('input');
  slider.type = 'range';
  slider.min = String(spec.min);
  slider.max = String(spec.max);
  slider.step = String((spec.max - spec.min) / 200);
  slider.value = String(spec.default);

  slider.addEventListener('input', () => {
    const v = parseFloat(slider.value);
    val.textContent = fmtDial(key, v);
    sim.setDials({ [key]: v }); // eases smoothly toward target (no pops)
  });

  row.append(label, slider);
  dialsRoot.appendChild(row);
  dialEls[key] = { slider, valEl: val };
}

document.getElementById('reset-dials').addEventListener('click', () => {
  const defaults = {};
  for (const key of DIAL_KEYS) {
    const spec = DIAL_SCHEMA[key];
    defaults[key] = spec.default;
    dialEls[key].slider.value = String(spec.default);
    dialEls[key].valEl.textContent = fmtDial(key, spec.default);
  }
  sim.setDials(defaults);
});

// --- Look controls ---------------------------------------------------------------
const EXPOSURE_DEFAULT = 1.0;
const postRoot = document.getElementById('posterize');
{
  const row = document.createElement('div');
  row.className = 'dial';
  const label = document.createElement('div');
  label.className = 'dial-label';
  const name = document.createElement('span');
  name.className = 'dial-name';
  name.textContent = 'exposure';
  const val = document.createElement('span');
  val.className = 'dial-val';
  val.textContent = EXPOSURE_DEFAULT.toFixed(2);
  label.append(name, val);

  const slider = document.createElement('input');
  slider.type = 'range';
  slider.min = '0.3'; slider.max = '2.0'; slider.step = '0.05';
  slider.value = String(EXPOSURE_DEFAULT);
  slider.addEventListener('input', () => {
    const v = parseFloat(slider.value);
    val.textContent = v.toFixed(2);
    renderer.setExposure(v);
  });
  row.append(label, slider);
  postRoot.appendChild(row);

  document.getElementById('reset-post').addEventListener('click', () => {
    slider.value = String(EXPOSURE_DEFAULT);
    val.textContent = EXPOSURE_DEFAULT.toFixed(2);
    renderer.setExposure(EXPOSURE_DEFAULT);
  });
}

// --- Seed control ----------------------------------------------------------------
const seedInput = document.getElementById('seed-input');

// Re-seed the world but keep whatever dial targets the user has dialed in.
function regenerate(seed) {
  const dials = {};
  for (const key of DIAL_KEYS) dials[key] = parseFloat(dialEls[key].slider.value);
  sim.reset(seed >>> 0, dials);
  renderer.resetPresentation();
  renderer.stepCamera(sim, FIXED);
  prevCar = snapCar(sim.state);
  steer = 0;
  throttle = 0;
  acc = 0;
  last = performance.now();
  seedInput.value = String(seed >>> 0);
}

document.getElementById('regen-btn').addEventListener('click', () => {
  regenerate(parseInt(seedInput.value, 10) || 0);
});
document.getElementById('seed-btn').addEventListener('click', () => {
  regenerate((Math.random() * 0xffffffff) >>> 0);
});

// --- HUD -------------------------------------------------------------------------
const hudSpeed = document.getElementById('hud-speed');
const hudOffset = document.getElementById('hud-offset');
const hudStatus = document.getElementById('hud-status');

function updateHud(state) {
  hudSpeed.textContent = state.car.speed.toFixed(1);
  const off = state.road.offset;
  hudOffset.textContent = off.toFixed(1);
  const halfWidth = (state.road.center && state.road.center.width ? state.road.center.width : 8) / 2;
  const offRoad = Math.abs(off) > halfWidth;
  hudStatus.textContent = !state.car.grounded ? 'airborne' : offRoad ? 'OFF ROAD' : 'on road';
  hudStatus.classList.toggle('warn', offRoad || !state.car.grounded);
}

// --- Diagnostics overlay (press P; absent from any captured frame) ---------------
const diag = document.createElement('div');
diag.id = 'diag';
diag.style.cssText = 'position:absolute;left:14px;bottom:44px;padding:8px 10px;font:11px/1.5 monospace;' +
  'color:#cfe;background:rgba(10,11,16,0.6);border-radius:6px;white-space:pre;display:none;pointer-events:none';
stage.appendChild(diag);
let diagOn = false;
const QUALITY = ['balanced', 'high', 'low'];
let qualityIdx = 0;

// --- Product modes (Phase 6): autopilot, clean cinematic view, procedural audio ------
let autoOn = false;   // K — deterministic road-follower (relaxed play / soak tests)
let cleanOn = false;  // C — hide UI chrome for a cinematic view
const panel = document.getElementById('panel');
const hud = document.getElementById('hud');

let audio = null;     // M — lazy Web Audio engine/wind mix (needs a user gesture)
function initAudio() {
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const engine = ctx.createOscillator(); engine.type = 'sawtooth';
  const engineGain = ctx.createGain(); engineGain.gain.value = 0;
  const engineFilter = ctx.createBiquadFilter(); engineFilter.type = 'lowpass'; engineFilter.frequency.value = 900;
  engine.connect(engineFilter).connect(engineGain).connect(ctx.destination); engine.start();
  const buf = ctx.createBuffer(1, ctx.sampleRate * 2, ctx.sampleRate);
  const data = buf.getChannelData(0);
  for (let i = 0; i < data.length; i++) data[i] = Math.random() * 2 - 1;
  const noise = ctx.createBufferSource(); noise.buffer = buf; noise.loop = true;
  const bp = ctx.createBiquadFilter(); bp.type = 'bandpass'; bp.frequency.value = 700;
  const windGain = ctx.createGain(); windGain.gain.value = 0;
  noise.connect(bp).connect(windGain).connect(ctx.destination); noise.start();
  audio = { ctx, engine, engineGain, windGain, muted: false };
}
function updateAudio(state) {
  if (!audio || audio.muted) return;
  const spd = Math.abs(state.car.speed);
  audio.engine.frequency.value = 55 + spd * 7 + (state.action.throttle > 0 ? spd * 4 : 0);
  audio.engineGain.gain.value = 0.015 + Math.min(0.07, spd * 0.004);
  audio.windGain.gain.value = Math.min(0.05, spd * 0.0028);
}

window.addEventListener('keydown', (e) => {
  const k = e.key.toLowerCase();
  if (k === 'p') { diagOn = !diagOn; diag.style.display = diagOn ? 'block' : 'none'; }
  if (k === 'o') { qualityIdx = (qualityIdx + 1) % QUALITY.length; renderer.setQuality(QUALITY[qualityIdx]); }
  if (k === 'k') { autoOn = !autoOn; }
  if (k === 'c') {
    cleanOn = !cleanOn;
    panel.style.display = cleanOn ? 'none' : '';
    hud.style.display = cleanOn ? 'none' : '';
    beatBar.style.opacity = cleanOn ? '0' : '1';
    // Collapse the panel's grid column so the canvas fills the whole width.
    document.body.style.gridTemplateColumns = cleanOn ? '1fr' : '';
    resize();
  }
  if (k === 'm') {
    if (!audio) initAudio();
    else { audio.muted = !audio.muted; if (audio.muted) { audio.engineGain.gain.value = 0; audio.windGain.gain.value = 0; } }
  }
});

// --- Scenic beat label + timeline (Phase 2) --------------------------------------
// A small readout of the current scenic beat, and (with diagnostics on) a colored
// timeline of upcoming beats so the authored rhythm is visible.
const beatBar = document.createElement('div');
beatBar.id = 'beatbar';
beatBar.style.cssText = 'position:absolute;top:14px;left:50%;transform:translateX(-50%);' +
  'font:11px/1.4 monospace;color:#dfe;background:rgba(10,11,16,0.5);padding:6px 12px;border-radius:6px;' +
  'text-align:center;pointer-events:none;letter-spacing:0.04em';
stage.appendChild(beatBar);

function updateBeat(state, sim) {
  const b = state.beat;
  const lm = b.landmark ? ` · ★ ${b.landmark}` : '';
  let line = `${autoOn ? '▶AUTO · ' : ''}${b.role.toUpperCase()} · ${b.landform} · ${b.routeClass}${lm}`;
  if (diagOn) {
    const d0 = state.road.d;
    const upcoming = sim.beatsInRange(d0, d0 + 1600).map((x) => {
      const near = x.d0 <= d0 && x.d1 >= d0;
      const tag = x.landmark ? x.role[0].toUpperCase() + '★' : (near ? x.role[0].toUpperCase() : x.role[0]);
      return tag;
    }).join('›');
    line += `   [${upcoming}]`;
  }
  beatBar.textContent = line;
}

// --- Main loop: fixed-timestep accumulator + render interpolation -----------------
// The simulation advances in fixed FIXED-second steps regardless of display refresh
// rate (SIM_UPGRADE.md §7.1), so physics speed is identical at 60 Hz and 120 Hz. The
// rendered frame interpolates between the last two steps for smooth motion.
const snapCar = (s) => ({ x: s.car.x, y: s.car.y, z: s.car.z, heading: s.car.heading, slip: s.car.slip, roadD: s.road.d });
const MAX_STEPS = 5;
let acc = 0;
let last = performance.now();
let prevCar = snapCar(sim.state);
let fps = 60, fpsFrames = 0, fpsSince = last;

function frame(now) {
  let dt = (now - last) / 1000;
  last = now;
  if (dt > 0.25) dt = 0.25;         // clamp long pauses (tab switch) — no death spiral
  acc += dt;

  const steerTarget = ((keys['a'] || keys['arrowleft'] ? 1 : 0) -
    (keys['d'] || keys['arrowright'] ? 1 : 0)) * KEYBOARD_STEER;
  const throttleTarget = (keys['w'] || keys['arrowup'] ? 1 : 0) - (keys['s'] || keys['arrowdown'] ? 1 : 0);

  let steps = 0;
  while (acc >= FIXED && steps < MAX_STEPS) {
    prevCar = snapCar(sim.state);
    let action;
    if (autoOn) {
      action = sim.autopilotAction();     // deterministic road-follower
      steer = action.steer; throttle = action.throttle; // stay in sync for a smooth handoff
    } else {
      const steerRate = steerTarget === 0 ? STEER_RETURN : STEER_RISE;
      steer = approachExp(steer, steerTarget, steerRate, FIXED);
      throttle = approachExp(throttle, throttleTarget, THROTTLE_RATE, FIXED);
      action = { steer, throttle };
    }
    sim.step(action);
    renderer.stepCamera(sim, FIXED);
    acc -= FIXED;
    steps++;
  }

  renderer.render(sim, { alpha: acc / FIXED, prevCar });
  updateHud(sim.state);
  updateBeat(sim.state, sim);
  updateAudio(sim.state);

  fpsFrames++;
  if (now - fpsSince >= 500) {
    fps = (fpsFrames * 1000) / (now - fpsSince);
    fpsFrames = 0; fpsSince = now;
  }
  if (diagOn) {
    try {
      const s = renderer.stats();
      diag.textContent =
        `fps       ${fps.toFixed(0)}\nsteps/fr  ${steps}\ndraw call ${s.calls}\ntris      ${(s.triangles / 1000).toFixed(0)}k\ninstances ${s.instances}\nquality   ${QUALITY[qualityIdx]}  (O)`;
    } catch (err) { console.error('diag error:', err); }
  }

  requestAnimationFrame(frame);
}
renderer.stepCamera(sim, FIXED); // seed the spring so the first frame is framed correctly
requestAnimationFrame(frame);
