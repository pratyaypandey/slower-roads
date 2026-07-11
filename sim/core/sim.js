// SlowSim — the deterministic simulation orchestrator.
//
// Pure JS, zero rendering dependency (SIM.md §1.4 "one core, two heads"): this is the
// oracle. It owns the road, the car, and the dials, and advances them one fixed tick
// at a time. A renderer (RGB head) or a headless skeleton extractor (λ-anchor head)
// both consume `sim.state` — neither is required for the sim to run and be replayable.
//
//   const sim = new SlowSim({ seed: 42 });
//   sim.setDials({ fog: 0.8 });
//   sim.step({ steer: 0.3, throttle: 1 });
//   const s = sim.state;   // { t, step, car, road, dials, action }

import { Road, terrainSurfaceHeight, SEA_LEVEL } from './road.js';
import { scatter } from './scatter.js';
import { Director } from './director.js';
import { makeCar, stepCar, SURFACE } from './car.js';
import { makeDials, smoothDials, clampDial, DIAL_KEYS } from './dials.js';

const GRAVITY_G = 11.0;       // base gravitational accel (scaled by the gravity dial)
// Must cover the renderer's far terrain band. If this is shorter than the visual
// horizon, Road.sampleAt() clamps every distant request to the final node, bunching
// scenery at one point and making it appear only when the car gets closer.
const HORIZON = 720;

export class SlowSim {
  constructor({ seed = 1, dt = 1 / 30, dials = {} } = {}) {
    this.dt = dt;
    this._init(seed, dials);
  }

  _init(seed, dialOverrides) {
    this.seed = seed >>> 0;
    this.step_ = 0;
    this.t = 0;
    this.road = new Road(this.seed);
    this.director = new Director(this.seed);   // deterministic scenic plan (Phase 2)
    this.car = makeCar();
    this.car.grounded = true;
    this.carD = 0;                             // car's arc-length along the road
    this.dials = makeDials(dialOverrides);     // smoothed, live values
    this.target = makeDials(dialOverrides);    // where the dials are heading
    this.lastAction = { steer: 0, throttle: 0 };
    this.road.ensureUpTo(HORIZON, this.dials);
    this.car.y = this.road.sampleAt(0).y;
  }

  /** Reseed and restart deterministically. */
  reset(seed = this.seed, dials = {}) {
    this._init(seed, dials);
  }

  /** Set dial targets (partial). Dials smoothly ease toward these each step. */
  setDials(partial) {
    for (const k of DIAL_KEYS) {
      if (partial[k] !== undefined) this.target[k] = clampDial(k, partial[k]);
    }
  }

  /** Snap dials instantly to values (no easing) — for data-gen static-param episodes. */
  snapDials(partial) {
    this.setDials(partial);
    for (const k of DIAL_KEYS) {
      if (partial[k] !== undefined) this.dials[k] = this.target[k];
    }
  }

  /** Terrain height at world (x, z): flat at road level on the ribbon, easing to the
   *  shared heightfield off the shoulder — so road and ground meet without a trench. */
  terrainHeight(x, z) {
    const near = this.road.nearest(x, z, this.carD);
    return terrainSurfaceHeight(near.point, near.lateral, x, z, this.seed, this.dials.hilliness);
  }

  /** Advance one fixed tick under a continuous action {steer, throttle} in [-1,1]. */
  step(action = { steer: 0, throttle: 0 }) {
    const dt = this.dt;
    this.lastAction = { steer: action.steer ?? 0, throttle: action.throttle ?? 0 };

    // 1) Ease dials toward their targets (smooth param->pixel; SIM.md §1.3).
    smoothDials(this.dials, this.target, dt);

    // 2) Horizontal car physics, on whatever surface is under the car.
    this._surface = this._surfaceUnderCar();
    const previousSpeed = this.car.speed;
    stepCar(this.car, this.lastAction, dt, this.dials, this._surface);

    // 3) Vertical model: glued to terrain, but crests can launch the car; low gravity
    //    => long, floaty, surreal hang time. Fully determined by state.
    const groundY = this.terrainHeight(this.car.x, this.car.z);
    if (this.car.grounded) {
      const impliedVy = (groundY - this.car.y) / dt;
      this.car.y = groundY;
      const contact = this._roadContactMotion(previousSpeed, dt);
      // On a crest, staying on the road requires downward acceleration. Contact is
      // lost only when that requirement exceeds gravity; this is the normal-force
      // condition N <= 0, expressed along the road's vertical profile.
      const gravityAccel = GRAVITY_G * this.dials.gravity;
      const leavesGround = this._surface.name === 'road' && contact.accel < -gravityAccel;
      if (leavesGround) {
        this.car.vy = contact.vy;
        this.car.grounded = false;
      } else {
        this.car.vy = this._surface.name === 'road' ? contact.vy : impliedVy;
      }
    } else {
      this.car.vy -= GRAVITY_G * this.dials.gravity * dt;
      this.car.y += this.car.vy * dt;
      if (this.car.y <= groundY) {
        this.car.y = groundY;
        this.car.vy = 0;
        this.car.grounded = true;
      }
    }

    // 4) Advance arc-length bookkeeping and extend the road ahead.
    let near = this.road.nearest(this.car.x, this.car.z, this.carD);
    this.carD = near.d;
    this.road.ensureUpTo(this.carD + HORIZON, this.dials);

    // 5) Collide with trees/rocks (off-road only — the road corridor is clear).
    this._collide();
    near = this.road.nearest(this.car.x, this.car.z, this.carD);
    this.carD = near.d;

    this.step_++;
    this.t += dt;
    return this.state;
  }

  /** Vertical velocity and acceleration required to remain on the road at this pose. */
  _roadContactMotion(previousSpeed, dt) {
    const near = this.road.nearest(this.car.x, this.car.z, this.carD);
    const h = this.car.heading;
    const fx = Math.sin(h), fz = Math.cos(h);
    const rx = fz, rz = -fx;
    const vx = fx * this.car.speed + rx * this.car.slip;
    const vz = fz * this.car.speed + rz * this.car.slip;
    const alongSpeed = near.tangent.x * vx + near.tangent.z * vz;

    const w = 2;
    const before = this.road.sampleAt(Math.max(0, near.d - w));
    const after = this.road.sampleAt(Math.min(this.road.length, near.d + w));
    const span = Math.max(1, after.d - before.d);
    const verticalCurvature = (after.grade - before.grade) / span;
    const longitudinalAccel = (this.car.speed - previousSpeed) / dt;
    return {
      vy: near.point.grade * alongSpeed,
      accel: verticalCurvature * alongSpeed * alongSpeed + near.point.grade * longitudinalAccel,
    };
  }

  /** Immutable-ish snapshot of the current state, for renderers and data export. */
  get state() {
    const near = this.road.nearest(this.car.x, this.car.z, this.carD);
    return {
      t: this.t,
      step: this.step_,
      seed: this.seed,
      car: {
        x: this.car.x, y: this.car.y, z: this.car.z,
        heading: this.car.heading,
        speed: this.car.speed,
        slip: this.car.slip,
        vy: this.car.vy,
        grounded: this.car.grounded,
      },
      road: { d: this.carD, offset: near.lateral, center: near.point, tangent: near.tangent },
      surface: this._surface ? this._surface.name : 'road',
      beat: this._beatLabel(this.director.beatAt(this.carD)),
      dials: { ...this.dials },
      action: { ...this.lastAction },
    };
  }

  /** Centerline samples from `back` metres behind the car to `ahead` metres in front. */
  roadAhead(back = 40, ahead = 160, spacing = 2) {
    const out = [];
    for (let d = Math.max(0, this.carD - back); d <= this.carD + ahead; d += spacing) {
      out.push(this.road.sampleAt(d));
    }
    return out;
  }

  /** Deterministic roadside props near the car. */
  props(back = 40, ahead = 160) {
    return this.road.props(Math.max(0, this.carD - back), this.carD + ahead, this.dials);
  }

  /** Deterministic world scatter (trees/rocks/bushes/grass) — drawn by the renderer
   *  and collided with by the physics. Both call this so they agree exactly. The scenic
   *  director modulates density (open vistas thin out; forest beats thicken). */
  scatter(back = 40, ahead = 300, opts) {
    return scatter(this.road, Math.max(0, this.carD - back), this.carD + ahead, this.seed, this.dials, this.director, opts);
  }

  /** Compact scenic-beat label for sim.state (renderer HUD + dataset metadata). */
  _beatLabel(b) {
    return { role: b.role, landform: b.landform, routeClass: b.routeClass, enclosure: b.enclosure, landmark: b.landmark };
  }

  /** The scenic beat / chapter the car is currently in (labels for renderer + dataset). */
  beatAt(d = this.carD) { return this.director.beatAt(d); }
  chapterAt(d = this.carD) { return this.director.chapterAt(d); }
  beatsInRange(d0, d1) { return this.director.beatsInRange(d0, d1); }

  /** Landmark placements overlapping [carD-back, carD+ahead], seated on the terrain and
   *  offset to the open (vista) side of the road. */
  landmarks(back = 60, ahead = 620) {
    const d0 = Math.max(0, this.carD - back), d1 = this.carD + ahead;
    const out = [];
    for (const b of this.director.beatsInRange(d0, d1)) {
      if (!b.landmark) continue;
      const d = Math.min(this.road.length, (b.d0 + b.d1) / 2);
      const c = this.road.sampleAt(d);
      const rx = Math.cos(c.heading), rz = -Math.sin(c.heading);
      const off = 26 + (b.landmark === 'viaduct' ? 0 : 22);   // viaducts straddle the road
      const lateral = b.vistaSide * off;
      const x = c.x + rx * lateral, z = c.z + rz * lateral;
      const y = terrainSurfaceHeight(c, lateral, x, z, this.seed, this.dials.hilliness);
      out.push({ id: b.index, d, x, y, z, type: b.landmark, side: b.vistaSide, biome: b.biome });
    }
    return out;
  }

  /** Which surface is under the car right now (drives grip/drag/top-speed). */
  _surfaceUnderCar() {
    const near = this.road.nearest(this.car.x, this.car.z, this.carD);
    const half = near.point.width / 2;
    if (Math.abs(near.lateral) < half + 0.5) return { ...SURFACE.road, name: 'road' };
    if (this.dials.snow > 0.5) return { ...SURFACE.snow, name: 'snow' };
    if (this.dials.biomeX < 0.3) return { ...SURFACE.sand, name: 'sand' };
    return { ...SURFACE.grass, name: 'grass' };
  }

  /** Push the car out of any collidable object it overlaps, and bleed its speed. */
  _collide() {
    const obs = this.scatter(8, 10, { grass: false });
    const R = 1.3; // car half-extent for collision
    for (const o of obs) {
      if (!o.collidable) continue;
      const dx = this.car.x - o.x, dz = this.car.z - o.z;
      const minD = R + o.radius;
      const dist2 = dx * dx + dz * dz;
      if (dist2 >= minD * minD || dist2 < 1e-6) continue;
      const dist = Math.sqrt(dist2);
      const nx = dx / dist, nz = dz / dist;
      const push = minD - dist;
      this.car.x += nx * push;
      this.car.z += nz * push;
      this.car.speed *= 0.25;   // hitting a trunk kills most of your momentum
      this.car.slip *= 0.25;
    }
  }

  /** Deterministic road-follower (SIM_UPGRADE §11.4) — pure-pursuit steering toward a
   *  look-ahead point plus a lateral-error correction, with curvature-aware speed. Used
   *  for relaxed play, soak tests, regression captures, and dataset action distributions.
   *  Returns an action in the same convention sim.step() consumes. */
  autopilotAction() {
    const speed = Math.abs(this.car.speed);
    // Clean pure-pursuit: aim at a look-ahead point on the centerline. The look-ahead
    // itself corrects lateral drift, so no separate error term is needed (which is what
    // caused the earlier weave). Look-ahead grows with speed for stability.
    const la = 11 + speed * 0.9;
    const tgt = this.road.sampleAt(this.carD + la);
    const dx = tgt.x - this.car.x, dz = tgt.z - this.car.z;
    const headErr = angleDiff(Math.atan2(dx, dz), this.car.heading);
    // + steer turns the car toward increasing heading (its right).
    const steer = clamp(headErr * 1.25, -1, 1);
    // A relaxed cruise: lower speed means less lateral slip and tighter tracking on the
    // curvy generated roads. Slow further for upcoming curvature.
    const curv = Math.abs(this.road.sampleAt(this.carD + 22).curvature);
    const targetSpeed = 16 * (1 - Math.min(0.55, curv * 55));
    const throttle = this.car.speed < targetSpeed - 1 ? 1 : this.car.speed > targetSpeed + 1 ? -0.4 : 0.2;
    return { steer, throttle };
  }

  /** Per-frame dataset metadata (SIM_UPGRADE §10.2) — labels for stratified datasets,
   *  hard-case eval, and later interpretability. Cheap because the sim already knows them. */
  frameLabels() {
    const b = this.director.beatAt(this.carD);
    const ch = this.director.chapterAt(this.carD);
    const near = this.road.nearest(this.car.x, this.car.z, this.carD);
    return {
      seed: this.seed, step: this.step_, d: this.carD,
      chapter: ch.index, routeClass: b.routeClass,
      beat: b.index, role: b.role, landform: b.landform, enclosure: b.enclosure,
      landmark: b.landmark, vistaSide: b.vistaSide,
      biomeWeights: { x: this.dials.biomeX, y: this.dials.biomeY },
      weather: { fog: this.dials.fog, rain: this.dials.rain, snow: this.dials.snow },
      timeOfDay: this.dials.timeOfDay,
      curvature: near.point.curvature, grade: near.point.grade,
      surface: this._surface ? this._surface.name : 'road',
      offRoad: Math.abs(near.lateral) > near.point.width / 2,
      grounded: this.car.grounded,
    };
  }

  /**
   * The λ-anchor skeleton (SIM_UPGRADE §5, §10.3): a small deterministic structure the
   * on-device anchor conditions on — road look-ahead in the car's local frame plus
   * coarse semantic context. Small enough to run beside neural inference, and it never
   * inherits render complexity.
   */
  skeleton({ ahead = 140, n = 16 } = {}) {
    const fx = Math.sin(this.car.heading), fz = Math.cos(this.car.heading);
    const rx = fz, rz = -fx; // right = forward rotated -90deg
    const road = [];
    for (let k = 0; k < n; k++) {
      const dd = this.carD + (k / (n - 1)) * ahead;
      const p = this.road.sampleAt(dd);
      const dx = p.x - this.car.x, dz = p.z - this.car.z;
      road.push({
        forward: fx * dx + fz * dz,        // metres ahead in the car frame
        lateral: rx * dx + rz * dz,        // metres right of the car
        headingDelta: angleDiff(p.heading, this.car.heading),
        curvature: p.curvature,
        grade: p.grade,
      });
    }
    const b = this.director.beatAt(this.carD);
    const marks = this.landmarks(0, ahead + 60);
    const next = marks.length ? { type: marks[0].type, forward: marks[0].d - this.carD, side: marks[0].side } : null;
    return {
      car: { speed: this.car.speed, slip: this.car.slip, grounded: this.car.grounded, surface: this._surface ? this._surface.name : 'road' },
      road, width: this.road.sampleAt(this.carD).width,
      beat: { role: b.role, landform: b.landform, routeClass: b.routeClass, enclosure: b.enclosure },
      nextLandmark: next,
      env: { fog: this.dials.fog, rain: this.dials.rain, snow: this.dials.snow, timeOfDay: this.dials.timeOfDay, biomeX: this.dials.biomeX, biomeY: this.dials.biomeY },
    };
  }

  /** Full serialization for exact mid-stream replay (includes the baked road cache). */
  snapshot() {
    return {
      seed: this.seed, dt: this.dt, step: this.step_, t: this.t,
      car: { ...this.car }, carD: this.carD,
      dials: { ...this.dials }, target: { ...this.target },
      lastAction: { ...this.lastAction },
      road: this.road._s.map((n) => ({ ...n })),
    };
  }

  static fromSnapshot(snap) {
    const sim = new SlowSim({ seed: snap.seed, dt: snap.dt });
    sim.step_ = snap.step;
    sim.t = snap.t;
    Object.assign(sim.car, snap.car);
    sim.carD = snap.carD;
    Object.assign(sim.dials, snap.dials);
    Object.assign(sim.target, snap.target);
    sim.lastAction = { ...snap.lastAction };
    sim.road._s = snap.road.map((n, i, nodes) => {
      if (n.grade !== undefined) return { ...n };
      const a = nodes[Math.max(0, i - 1)];
      const b = nodes[Math.min(nodes.length - 1, i + 1)];
      const span = Math.max(1, b.d - a.d);
      return { ...n, grade: (b.y - a.y) / span };
    });
    return sim;
  }
}

function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }

/** Shortest signed difference a-b wrapped to [-pi, pi]. */
function angleDiff(a, b) {
  let d = (a - b) % (Math.PI * 2);
  if (d > Math.PI) d -= Math.PI * 2;
  if (d < -Math.PI) d += Math.PI * 2;
  return d;
}
