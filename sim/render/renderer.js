// SimRenderer (v2): the sim's RGB head, rebuilt to match recorded Slow Roads footage.
//
// Clean-room: built from observation of the game (docs/FIDELITY.md) with no Slow
// Roads code or assets. It keeps the v1 SimRenderer API (stepCamera / render /
// capture / captureChannels / setDisplaySize / setQuality / setExposure / stats),
// so the demo, capture/dream pages and data-gen scripts use it unchanged.
//
// Pipeline: every material is a custom shader authored in display colours with one
// shared lighting/haze model, and nothing is tone-mapped or post-processed. The
// canvas and capture() therefore produce identical pixels. capture() supersamples and
// box-filters, like the reference frames (area-downsampled from 1432px).
//
// The world stays the sim's (road, heightfield, scatter), so the oracle state in
// the datasets describes exactly what is drawn.

import * as THREE from 'three';
import { fbm2, hash1 } from '../core/prng.js';
import { terrainSurfaceHeight, heightField, SEA_LEVEL } from '../core/road.js';
import * as SH from './shaders.js';
import * as TX from './textures.js';
import { buildCarGeometry, buildWheelGeometry } from './car.js';

// Colours here are display values used verbatim by the shaders; stop three.js from
// linearising hex colours (it assumes sRGB input and a linear working space).
THREE.ColorManagement.enabled = false;

// --- camera (calibrated against the reference chase view) -------------------
const CAM = {
  // Solved jointly with the car's profile from the reference's median car sprite
  // (rows of roof / rear screen / deck lip / rear panel / bumper, deck width, and
  // the ~0.39 water-horizon row); see eval/calibrate_view.py.
  fov: 60,          // vertical, degrees; capture() keeps it at aspect 1
  back: 6.08,       // metres behind the car centre
  height: 2.73,     // metres above the car
  pitch: 7.25,      // degrees down
  yawRate: 7,       // 1/s: slight lag behind the car's heading in corners
  pitchRate: 14,    // near-rigid: the reference car holds its place in frame
};

// --- road look ----------------------------------------------------------------
// Matched in image space to the reference (median frames, per-column grass/asphalt):
// a ~7.2 m road, the car's centre ~1.8 m from the left asphalt edge, dashes ~1.8 m to
// its right, edge lines just inside the asphalt edge.
const ROAD = { half: 3.6, edgeIn: 3.22, edgeOut: 3.36, dashHalf: 0.075, dashPeriod: 9, dashOn: 0.34, rail: 4.35 };

// Near strip: road-relative rows (snapped to world arc-length so geometry doesn't
// swim) and lateral columns (dense at the road, sparse out to 150 m).
const ROW_BANDS = [[60, 1], [160, 2], [360, 4], [760, 8]];   // [until (m ahead), spacing]
const ROW_BACK = 30;
const COLS_HALF = [0, 0.9, 1.8, 2.6, 3.22, 3.36, 3.6, 4.0, 4.5, 5.0, 5.6, 6.2, 6.9, 7.2, 8.3, 9.6, 11.2,
  13, 15.2, 17.8, 21, 25, 30, 36, 43, 51, 60, 71, 84, 100, 120, 150];
const LAT = [...COLS_HALF.slice(1).reverse().map((u) => -u), ...COLS_HALF];
const MAX_ROWS = 400;
const RAIL_MAX = 8000;   // guardrail vertices
const CAPTURE_SOFTNESS = 0.4;   // capture blur sigma in px per 256 px of output (reference softness)

// Far grid: world-aligned heightfield around the camera, under the near strip.
const FAR_N = 150, FAR_CELL = 24;

// Vegetation draw window (arc-length metres behind / ahead of the car).
const VEG_BACK = 30, VEG_AHEAD = 450;
const CAP = { canopy: 6000, conifer: 1500, trunk: 4200, bush: 2400, rock: 900, tuft: 12000 };

// Default-dial palette, sampled from the reference footage (eval/fidelity.py).
const PAL = {
  zenith: 0xd2edf8, horizon: 0xe9f7f9, haze: 0xdcecee, cloud: 0xffffff, cloudShade: 0xd6e2e8,
  grassA: 0x838c4e, grassB: 0x97935c, grassDry: 0xa28a69, forest: 0x3b5626,
  rock: 0x958a70, gravel: 0xa39c86, asphalt: 0x5d6366, line: 0xf7fafa, sand: 0xb7ab86,
  water: 0x6f9ec0, waterDeep: 0x3f6f96,
  canopy: [0x415828, 0x506930, 0x5d7635, 0x354e23], conifer: [0x304926, 0x39542f, 0x445d31],
  bush: [0x557236, 0x637f3c], tuft: [0xd9dd82, 0xedde92, 0xc8d074, 0xf8dba1], trunk: 0x4e3d2e, stone: 0x8d877a,
  rail: 0xb9bec2, post: 0x8e9396,
};

export class SimRenderer {
  constructor(canvas, { dataSize = 128 } = {}) {
    this.dataSize = dataSize;
    this.rw = 0; this.rh = 0;
    this._exposure = 1;

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.outputColorSpace = THREE.LinearSRGBColorSpace;   // shaders emit display values
    this.renderer.toneMapping = THREE.NoToneMapping;
    this.renderer.info.autoReset = false;

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(CAM.fov, 16 / 9, 0.3, 9000);

    this.u = {
      uSunDir: { value: new THREE.Vector3(0.3, 0.9, 0.2).normalize() },
      uSunCol: { value: new THREE.Color() },
      uAmbSky: { value: new THREE.Color() },
      uAmbGround: { value: new THREE.Color() },
      uCamPos: { value: new THREE.Vector3() },
      uFogCol: { value: new THREE.Color() },
      uFogSun: { value: new THREE.Color() },
      uFogDen: { value: 0.0012 },
      uTime: { value: 0 },
      uSnow: { value: 0 },
      uSnowCol: { value: new THREE.Color(0xf1f4f8) },
    };
    this.noise = TX.noiseTexture(256);

    this._cam = { x: 0, y: 0, z: 0, yaw: 0, pitch: CAM.pitch * Math.PI / 180, init: false };
    this._camPrev = { ...this._cam };
    this._susp = { pitch: 0, roll: 0, gradA: undefined, gradS: undefined };
    this._dummy = new THREE.Object3D();
    this._color = new THREE.Color();

    this._buildSky();
    this._buildTerrain();
    this._buildWater();
    this._buildVegetation();
    this._buildRails();
    this._buildCar();
    this._buildPrecip();
  }

  // --- materials --------------------------------------------------------------
  _mat(vertexShader, fragmentShader, uniforms = {}, opts = {}) {
    return new THREE.ShaderMaterial({ vertexShader, fragmentShader, uniforms: { ...this.u, ...uniforms }, ...opts });
  }

  _solid(color, { spec = 0, gloss = 20, wrap = 0.2, vertexColors = false, amb = 1, sun = 1 } = {}) {
    return this._mat(SH.solidVert, SH.solidFrag, {
      uColor: { value: new THREE.Color(color) }, uSpec: { value: spec }, uGloss: { value: gloss }, uWrap: { value: wrap },
      uAmbScale: { value: amb }, uSunScale: { value: sun },
    }, { vertexColors, side: THREE.DoubleSide });
  }

  _card(map, { wrap = 0.5, sway = 0.05, edge = 1, edgeGrow = 0 } = {}) {
    return this._mat(SH.cardVert, SH.cardFrag, {
      uMap: { value: map }, uWrap: { value: wrap }, uSway: { value: sway }, uEdge: { value: edge },
      uEdgeGrow: { value: edgeGrow },
    }, { side: THREE.DoubleSide, alphaToCoverage: true, transparent: false });
  }

  // --- construction ------------------------------------------------------------
  _buildSky() {
    this.skyU = {
      uNoise: { value: this.noise }, uZenith: { value: new THREE.Color() }, uHorizon: { value: new THREE.Color() },
      uCloudCol: { value: new THREE.Color() }, uCloudShade: { value: new THREE.Color() },
      uCloudCover: { value: 0.55 }, uCloudOff: { value: new THREE.Vector2() },
      uStars: { value: new THREE.Color(0xffffff) }, uNight: { value: 0 },
    };
    const mat = this._mat(SH.skyVert, SH.skyFrag, this.skyU, { side: THREE.BackSide, depthWrite: false });
    this.sky = new THREE.Mesh(new THREE.SphereGeometry(4000, 48, 24), mat);
    this.sky.frustumCulled = false;
    this.sky.renderOrder = -1;
    this.scene.add(this.sky);
  }

  _buildTerrain() {
    this.terrainU = {
      uNoise: { value: this.noise }, uGrassTex: { value: TX.grassDetailTexture() },
      uGrassA: { value: new THREE.Color(PAL.grassA) }, uGrassB: { value: new THREE.Color(PAL.grassB) },
      uGrassDry: { value: new THREE.Color(PAL.grassDry) }, uRockCol: { value: new THREE.Color(PAL.rock) },
      uGravelCol: { value: new THREE.Color(PAL.gravel) }, uAsphaltCol: { value: new THREE.Color(PAL.asphalt) },
      uLineCol: { value: new THREE.Color(PAL.line) }, uForestCol: { value: new THREE.Color(PAL.forest) },
      uSandCol: { value: new THREE.Color(PAL.sand) }, uWet: { value: 0 }, uSeaLevel: { value: SEA_LEVEL },
      uHalfRoad: { value: ROAD.half }, uEdgeIn: { value: ROAD.edgeIn }, uEdgeOut: { value: ROAD.edgeOut },
      uDashHalf: { value: ROAD.dashHalf }, uDashPeriod: { value: ROAD.dashPeriod }, uDashOn: { value: ROAD.dashOn },
    };
    // Near strip.
    const cols = LAT.length;
    this.near = makeGridMesh(MAX_ROWS, cols, this._mat(SH.terrainVert, SH.terrainFrag, this.terrainU));
    this.scene.add(this.near.mesh);
    // Far grid, pushed back in depth so the near strip always wins where they overlap.
    const farMat = this._mat(SH.terrainVert, SH.terrainFrag, this.terrainU, {
      polygonOffset: true, polygonOffsetFactor: 3, polygonOffsetUnits: 6,
    });
    this.far = makeGridMesh(FAR_N + 1, FAR_N + 1, farMat);
    this.scene.add(this.far.mesh);
    this._nearKey = null; this._farKey = null;
  }

  _buildWater() {
    const mat = this._mat(SH.terrainVert, SH.waterFrag, {
      uNoise: { value: this.noise }, uWater: { value: new THREE.Color(PAL.water) }, uWaterDeep: { value: new THREE.Color(PAL.waterDeep) },
    });
    const geo = new THREE.PlaneGeometry(9000, 9000, 1, 1).rotateX(-Math.PI / 2);
    for (const k of ['aLat', 'aD', 'aForest']) geo.setAttribute(k, new THREE.BufferAttribute(new Float32Array(4), 1));
    this.water = new THREE.Mesh(geo, mat);
    this.water.frustumCulled = false;
    this.scene.add(this.water);
  }

  _buildVegetation() {
    const leaf = TX.leafClumpTexture(), conifer = TX.coniferTexture(), tuft = TX.grassTuftTexture();
    const inst = (geo, mat, cap) => {
      const m = new THREE.InstancedMesh(geo, mat, cap);
      m.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      m.setColorAt(0, new THREE.Color(1, 1, 1));
      m.count = 0;
      m.frustumCulled = false;
      this.scene.add(m);
      return m;
    };
    this.veg = {
      canopy: inst(canopyGeometry(22, 101), this._card(leaf, { wrap: 0.55, sway: 0.08 }), CAP.canopy),
      conifer: inst(coniferGeometry(), this._card(conifer, { wrap: 0.6, sway: 0.05 }), CAP.conifer),
      bush: inst(canopyGeometry(8, 202), this._card(leaf, { wrap: 0.7, sway: 0.05 }), CAP.bush),
      tuft: inst(tuftGeometry(), this._card(tuft, { wrap: 0.9, sway: 0.12, edge: 2, edgeGrow: 0.12 }), CAP.tuft),
      trunk: inst(solidGeo(new THREE.CylinderGeometry(0.16, 0.26, 1, 6).translate(0, 0.5, 0)), this._solid(0xffffff), CAP.trunk),
      rock: inst(solidGeo(new THREE.IcosahedronGeometry(1, 0)), this._solid(0xffffff, { wrap: 0.3 }), CAP.rock),
    };
    this._vegKey = null;
  }

  _buildRails() {
    this.railMat = this._solid(PAL.rail, { spec: 0.35, gloss: 30, wrap: 0.3, vertexColors: true });
    this.railGeo = new THREE.BufferGeometry();
    for (const [k, n] of [['position', 3], ['normal', 3], ['color', 3]])
      this.railGeo.setAttribute(k, new THREE.BufferAttribute(new Float32Array(RAIL_MAX * n), n).setUsage(THREE.DynamicDrawUsage));
    this.rail = new THREE.Mesh(this.railGeo, this.railMat);
    this.rail.frustumCulled = false;
    this.scene.add(this.rail);
    this.posts = new THREE.InstancedMesh(solidGeo(new THREE.BoxGeometry(0.12, 0.8, 0.12).translate(0, 0.4, 0)),
      this._solid(PAL.post, { wrap: 0.3 }), 600);
    this.posts.count = 0; this.posts.frustumCulled = false;
    this.scene.add(this.posts);
  }

  _buildCar() {
    this.car = new THREE.Group();
    this.chassis = new THREE.Mesh(buildCarGeometry(), this._solid(0xffffff, { spec: 0.15, gloss: 40, wrap: 0, vertexColors: true, amb: 0.36, sun: 1.7 }));
    this.car.add(this.chassis);
    this.wheels = [];
    const wg = buildWheelGeometry(), wm = this._solid(0xffffff, { vertexColors: true });
    for (const [x, z, front] of [[0.98, 1.36, true], [-0.98, 1.36, true], [0.98, -1.42, false], [-0.98, -1.42, false]]) {
      const steer = new THREE.Group();
      steer.position.set(x, 0.36, z);
      const w = new THREE.Mesh(wg, wm);
      steer.add(w);
      this.car.add(steer);
      this.wheels.push({ steer, mesh: w, front });
    }
    const sh = new THREE.Mesh(new THREE.PlaneGeometry(2.7, 6.6).rotateX(-Math.PI / 2),
      new THREE.ShaderMaterial({ vertexShader: SH.shadowVert, fragmentShader: SH.shadowFrag,
        uniforms: { uStrength: { value: 0.95 } }, transparent: true, depthWrite: false,
        polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2 }));
    sh.position.y = 0.03;
    this.car.add(sh);
    this.scene.add(this.car);
  }

  _buildPrecip() {
    this.precipN = 1200;
    this.precipBase = new Float32Array(this.precipN * 3);
    const r = TX.rng(77);
    for (let i = 0; i < this.precipN * 3; i += 3) {
      this.precipBase[i] = (r() - 0.5) * 80; this.precipBase[i + 1] = r() * 50; this.precipBase[i + 2] = (r() - 0.5) * 80;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(this.precipN * 3), 3).setUsage(THREE.DynamicDrawUsage));
    this.precipMat = new THREE.PointsMaterial({ color: 0xdde7f0, size: 0.12, transparent: true, opacity: 0 });
    this.precip = new THREE.Points(geo, this.precipMat);
    this.precip.frustumCulled = false;
    this.scene.add(this.precip);
  }

  // --- camera ----------------------------------------------------------------
  /** Advance the deterministic chase camera one fixed sim step (call once per sim.step). */
  stepCamera(sim, dt) {
    const st = sim.state, c = this._cam;
    const grade = sim.road.sampleAt(st.road.d).grade;
    const tPitch = CAM.pitch * Math.PI / 180 - Math.atan(grade);
    const first = !c.init;
    if (first) { c.yaw = st.car.heading; c.pitch = tPitch; c.init = true; }
    else Object.assign(this._camPrev, c);
    c.yaw += angleDiff(st.car.heading, c.yaw) * (1 - Math.exp(-CAM.yawRate * dt));
    c.pitch += (tPitch - c.pitch) * (1 - Math.exp(-CAM.pitchRate * dt));
    c.x = st.car.x - Math.sin(c.yaw) * CAM.back;
    c.z = st.car.z - Math.cos(c.yaw) * CAM.back;
    const ty = Math.max(st.car.y + CAM.height - grade * CAM.back, sim.terrainHeight(c.x, c.z) + 1.0);
    c.y = first ? ty : c.y + (ty - c.y) * (1 - Math.exp(-20 * dt));
    if (first) Object.assign(this._camPrev, c);
  }

  resetPresentation() {
    this._cam.init = false;
    this._nearKey = this._farKey = this._vegKey = null;
    this._susp = { pitch: 0, roll: 0, gradA: undefined, gradS: undefined };
  }

  _applyCamera(a) {
    const p = this._camPrev, c = this._cam;
    const x = p.x + (c.x - p.x) * a, y = p.y + (c.y - p.y) * a, z = p.z + (c.z - p.z) * a;
    const yaw = p.yaw + angleDiff(c.yaw, p.yaw) * a, pitch = p.pitch + (c.pitch - p.pitch) * a;
    this.camera.position.set(x, y, z);
    this.camera.lookAt(x + Math.sin(yaw) * Math.cos(pitch), y - Math.sin(pitch), z + Math.cos(yaw) * Math.cos(pitch));
    this.camera.fov = CAM.fov;
    this.camera.updateProjectionMatrix();
    this.u.uCamPos.value.copy(this.camera.position);
    this.sky.position.copy(this.camera.position);
    this.water.position.set(x, SEA_LEVEL, z);
  }

  // --- per frame ------------------------------------------------------------------
  render(sim, interp) {
    this.renderer.info.reset();
    const st = sim.state;
    const a = interp ? interp.alpha : 1;
    const prev = interp ? interp.prevCar : null;
    const car = prev ? {
      x: prev.x + (st.car.x - prev.x) * a, y: prev.y + (st.car.y - prev.y) * a, z: prev.z + (st.car.z - prev.z) * a,
      heading: lerpAngle(prev.heading, st.car.heading, a), roadD: prev.roadD + (st.road.d - prev.roadD) * a,
    } : { x: st.car.x, y: st.car.y, z: st.car.z, heading: st.car.heading, roadD: st.road.d };
    if (!this._cam.init) this.stepCamera(sim, sim.dt);

    this.u.uTime.value = st.t;
    this._updateAtmosphere(st);
    this._applyCamera(a);
    this._updateNear(sim, st);
    this._updateFar(sim, st);
    this._updateVegetation(sim, st);
    this._updateCar(sim, st, car);
    this._updatePrecip(st);
    this.renderer.setRenderTarget(null);
    this.renderer.render(this.scene, this.camera);
  }

  _updateAtmosphere(st) {
    const d = st.dials, u = this.u;
    const elev = Math.sin(d.timeOfDay) * 78, azi = 180 + Math.cos(d.timeOfDay) * 110;
    u.uSunDir.value.setFromSphericalCoords(1, THREE.MathUtils.degToRad(90 - elev), THREE.MathUtils.degToRad(azi));
    const day = smoothstep(-5, 14, elev), dusk = Math.exp(-(((elev - 4) / 11) ** 2)) * smoothstep(-8, 0, elev);
    const night = 1 - day;
    const ex = this._exposure;
    // Fog dial: the default (0.25) reproduces the reference haze; 1 = dense fog.
    const fog = d.fog, grey = smoothstep(0.35, 1, fog) + d.rain * 0.6;

    const set = (c, dayHex, duskHex, nightHex, g = 0) => {
      c.set(dayHex).lerp(this._color.set(duskHex), dusk * 0.8).lerp(this._color.set(nightHex), night);
      if (g) c.lerp(this._color.setRGB(0.78, 0.8, 0.82), Math.min(1, g));
      return c.multiplyScalar(ex);
    };
    set(u.uSunCol.value, 0x6c6c68, 0x80542c, 0x10141c, grey * 0.8);
    set(u.uAmbSky.value, 0xd2dada, 0x8a7e86, 0x34405e, grey * 0.4);   // night: moonlit, not black
    set(u.uAmbGround.value, 0x9a9e8c, 0x5a4e48, 0x1e2536, grey * 0.3);
    set(u.uFogCol.value, PAL.haze, 0xe2b99a, 0x26304a, grey);
    set(u.uFogSun.value, 0xf4f6ea, 0xf2a468, 0x1a2234, grey);
    u.uFogDen.value = 0.0013 + fog * fog * 0.009 + d.rain * 0.002;
    const s = this.skyU;
    set(s.uZenith.value, PAL.zenith, 0x7f98b8, 0x0c1426, grey);
    set(s.uHorizon.value, PAL.horizon, 0xf0c9a2, 0x26304a, grey);
    set(s.uCloudCol.value, PAL.cloud, 0xf7d2b4, 0x2a3040, grey * 0.5);
    set(s.uCloudShade.value, PAL.cloudShade, 0xb89a9a, 0x161c28, grey * 0.5);
    s.uCloudCover.value = Math.min(0.95, 0.52 + fog * 0.3 + d.rain * 0.5);
    s.uCloudOff.value.set(st.t * 0.004, st.t * 0.0015);
    s.uNight.value = night * d.starDensity * 1.5;
    u.uSnow.value = d.snow;
    this.terrainU.uWet.value = d.rain;

    // Biome dials tint vegetation and ground: arid <-> lush (x), earthly <-> alien (y).
    const tintGround = (c, hex) => biomeTint(c.set(hex), d.biomeX, d.biomeY);
    tintGround(this.terrainU.uGrassA.value, PAL.grassA);
    tintGround(this.terrainU.uGrassB.value, PAL.grassB);
    tintGround(this.terrainU.uGrassDry.value, PAL.grassDry);
    tintGround(this.terrainU.uForestCol.value, PAL.forest);
    this._biome = [d.biomeX, d.biomeY];
  }

  // Near strip: rows along the road, rebuilt whenever the car advances 0.5 m.
  _updateNear(sim, st) {
    const dl = st.dials;
    const key = `${sim.seed}:${Math.round(st.road.d * 2)}:${Math.round(dl.hilliness * 1000)}:${sim.road.length | 0}`;
    if (key === this._nearKey) return;
    this._nearKey = key;
    const road = sim.road, seed = sim.seed, hill = dl.hilliness;
    const rows = [];
    let d = Math.max(0, Math.floor(st.road.d - ROW_BACK));
    const end = Math.min(road.length, st.road.d + ROW_BANDS[ROW_BANDS.length - 1][0]);
    while (d <= end && rows.length < MAX_ROWS) {
      rows.push(d);
      const ahead = d - st.road.d;
      const sp = (ROW_BANDS.find(([until]) => ahead < until) || ROW_BANDS[ROW_BANDS.length - 1])[1];
      d = Math.floor(d / sp + 1e-6) * sp + sp;
    }
    // Rows are snapped to world arc-length and the road never changes once baked, so
    // a row's vertices are computed once and reused while it stays in view.
    const ck = `${seed}:${Math.round(hill * 1000)}`;
    if (this._rowCacheKey !== ck) { this._rowCache = new Map(); this._rowCacheKey = ck; }
    const cache = this._rowCache, g = this.near, C = LAT.length;
    for (let i = 0; i < rows.length; i++) {
      let row = cache.get(rows[i]);
      if (!row) {
        row = nearRow(road, rows[i], seed, hill);
        if (rows[i] + 40 < road.length) cache.set(rows[i], row);   // window fully baked
      }
      for (let j = 0, o = 0; j < C; j++, o += 7)
        g.setVertex(i * C + j, row[o], row[o + 1], row[o + 2], row[o + 3], row[o + 4], row[o + 5], rows[i], row[o + 6]);
    }
    if (cache.size > MAX_ROWS * 2) {
      for (const k of cache.keys()) if (k < st.road.d - ROW_BACK - 8) cache.delete(k);
    }
    g.commit(rows.length, C);
    this._updateRails(sim, st);
  }

  // Far grid: world-aligned heights; near the road, sampled from the true (flattened)
  // surface and dropped a metre so it never pokes through the near strip.
  _updateFar(sim, st) {
    const cx = Math.round(this.camera.position.x / FAR_CELL), cz = Math.round(this.camera.position.z / FAR_CELL);
    const key = `${sim.seed}:${cx}:${cz}:${Math.round(st.dials.hilliness * 1000)}:${(sim.road.length / 48) | 0}`;
    if (key === this._farKey) return;
    this._farKey = key;
    const road = sim.road, seed = sim.seed, hill = st.dials.hilliness;
    // Bucket road samples for the near-road test.
    const B = 48, buckets = new Map(), bkey = (bx, bz) => bx * 65536 + bz;
    for (let d = Math.max(0, st.road.d - 400); d <= road.length; d += 6) {
      const p = road.sampleAt(d);
      const k = bkey(Math.floor(p.x / B), Math.floor(p.z / B));
      if (!buckets.has(k)) buckets.set(k, []);
      buckets.get(k).push(p);
    }
    const g = this.far, N = FAR_N + 1, half = (FAR_N * FAR_CELL) / 2;
    const nearFrom = st.road.d - ROW_BACK + 5;
    const nearTo = Math.min(road.length, st.road.d + ROW_BANDS[ROW_BANDS.length - 1][0]) - 10;
    const x0 = cx * FAR_CELL - half, z0 = cz * FAR_CELL - half, e = FAR_CELL * 0.5;
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        const x = x0 + j * FAR_CELL, z = z0 + i * FAR_CELL;
        let h = heightField(x, z, seed, hill);
        const hx = heightField(x + e, z, seed, hill), hz = heightField(x, z + e, seed, hill);
        // Nearest bucketed road sample within 150 m.
        let best = null, bd = 150 * 150;
        const bx = Math.floor(x / B), bz = Math.floor(z / B);
        for (let a = -3; a <= 3; a++) for (let b = -3; b <= 3; b++) {
          const list = buckets.get(bkey(bx + a, bz + b));
          if (!list) continue;
          for (const p of list) { const dd = (p.x - x) ** 2 + (p.z - z) ** 2; if (dd < bd) { bd = dd; best = p; } }
        }
        if (best) {
          // Where the near strip covers this point, sink the far grid out of sight: its
          // 24 m cells interpolate across road cuts and crests and would otherwise poke
          // up through the road. Elsewhere use the true surface, a metre down.
          const n = road.nearest(x, z, best.d, 8);
          h = terrainSurfaceHeight(n.point, n.lateral, x, z, seed, hill) - 1.0;
          const u = -n.lateral;   // renderer lateral (see nearRow)
          const lim = rowLimits(road, n.d);
          const cover = Math.min(COLS_HALF[COLS_HALF.length - 1], u > 0 ? lim.pos : lim.neg);
          if (n.d >= nearFrom && n.d <= nearTo && Math.abs(u) < cover - 1.2 * FAR_CELL) h -= 60;
        }
        g.setVertex(i * N + j, x, h, z, (h - hx) / e, (h - hz) / e, 999, 0, forestMask(x, z, seed));
      }
    }
    g.commit(N, N);
  }

  _updateRails(sim, st) {
    // Guardrail runs on 48 m segments: always where the ground falls away beside the
    // road, occasionally elsewhere. Decided per segment from world data only.
    const road = sim.road, seed = sim.seed, hill = st.dials.hilliness;
    const SEG = 48, pos = [], nrm = [], col = [], posts = [];
    const d0 = Math.max(0, st.road.d - 20), d1 = Math.min(road.length, st.road.d + 320);
    const railOn = (s, side) => {
      const c = road.sampleAt((s + 0.5) * SEG);
      const rx = Math.cos(c.heading), rz = -Math.sin(c.heading);
      const u = side * 14;
      const drop = c.y - terrainSurfaceHeight(c, u, c.x + rx * u, c.z + rz * u, seed, hill);
      return drop > 1.2 || hash1(s * 2 + (side > 0 ? 1 : 0), seed + 313) < 0.18;
    };
    const cache = new Map();
    const on = (s, side) => { const k = s * 2 + (side > 0 ? 1 : 0); if (!cache.has(k)) cache.set(k, railOn(s, side)); return cache.get(k); };
    const light = new THREE.Color(PAL.rail), dark = new THREE.Color(PAL.rail).multiplyScalar(0.72);
    for (const side of [-1, 1]) {
      let prev = null;
      for (let d = Math.floor(d0 / 2) * 2; d <= d1; d += 2) {
        if (!on(Math.floor(d / SEG), side)) { prev = null; continue; }
        const c = road.sampleAt(d);
        const rx = Math.cos(c.heading), rz = -Math.sin(c.heading);
        const u = side * ROAD.rail;
        const p = { x: c.x + rx * u, y: c.y, z: c.z + rz * u, nx: -side * rx, nz: -side * rz };
        if (prev) {
          // W-beam: two bands (upper lit, lower shaded) facing the road.
          for (const [ya, yb, cc] of [[0.72, 0.9, light], [0.56, 0.72, dark]]) {
            const quadPts = [[prev, ya], [p, ya], [p, yb], [prev, ya], [p, yb], [prev, yb]];
            for (const [q, y] of quadPts) {
              pos.push(q.x, q.y + y, q.z); nrm.push(q.nx, cc === light ? 0.35 : -0.25, q.nz); col.push(cc.r, cc.g, cc.b);
            }
          }
        }
        if (d % 4 === 0) posts.push(p);
        prev = p;
      }
    }
    const nv = Math.min(RAIL_MAX, pos.length / 3);
    for (const [k, arr] of [['position', pos], ['normal', nrm], ['color', col]]) {
      const at = this.railGeo.attributes[k];
      at.array.set(arr.length > nv * 3 ? arr.slice(0, nv * 3) : arr);
      at.needsUpdate = true;
    }
    this.railGeo.setDrawRange(0, nv);
    const dm = this._dummy;
    this.posts.count = Math.min(posts.length, 600);
    for (let i = 0; i < this.posts.count; i++) {
      dm.position.set(posts[i].x - posts[i].nx * 0.12, posts[i].y, posts[i].z - posts[i].nz * 0.12);
      dm.rotation.set(0, 0, 0); dm.scale.set(1, 1, 1); dm.updateMatrix();
      this.posts.setMatrixAt(i, dm.matrix);
    }
    this.posts.instanceMatrix.needsUpdate = true;
  }

  _updateVegetation(sim, st) {
    const d = st.dials;
    const key = `${sim.seed}:${Math.floor(st.car.x)}:${Math.floor(st.car.z)}:${Math.round(d.biomeX * 500)}:${Math.round(d.biomeY * 500)}:${Math.round(d.hilliness * 1000)}`;
    if (key === this._vegKey) return;
    this._vegKey = key;
    const seed = sim.seed, dm = this._dummy, col = this._color;
    const n = { canopy: 0, conifer: 0, trunk: 0, bush: 0, rock: 0, tuft: 0 };
    const put = (k, x, y, z, sx, sy, rot, tint) => {
      if (n[k] >= CAP[k]) return;
      dm.position.set(x, y, z); dm.rotation.set(0, rot, 0); dm.scale.set(sx, sy, sx); dm.updateMatrix();
      this.veg[k].setMatrixAt(n[k], dm.matrix);
      if (tint !== undefined) this.veg[k].setColorAt(n[k], biomeTint(col.set(tint), d.biomeX, d.biomeY, 0.6));
      n[k]++;
    };
    const pick = (arr, h) => arr[Math.floor(h * arr.length) % arr.length];
    const cx = this.camera.position.x, cz = this.camera.position.z;
    for (const o of sim.scatter(VEG_BACK, VEG_AHEAD)) {
      const dist = Math.hypot(o.x - cx, o.z - cz);
      // Grow in/out with distance so nothing pops at the edge of the draw window.
      const f = vegFade(o.type, dist);
      if (f <= 0.02) continue;
      const h = hash1(Math.floor(o.x * 7.1) * 131 + Math.floor(o.z * 3.3), seed + 5);
      switch (o.type) {
        case 'tree': {
          // Bushy broadleaf: a main crown plus 1-2 offset lobes for an irregular,
          // painterly silhouette; the crown sits low so little trunk shows.
          const s = o.scale * 1.25 * f, tint = pick(PAL.canopy, h);
          put('canopy', o.x, o.y + 3.1 * s, o.z, 2.9 * s, 2.5 * s, o.rot, tint);
          const lobes = 1 + (h > 0.5 ? 1 : 0);
          for (let k = 0; k < lobes; k++) {
            const a = o.rot + k * 2.4 + h * 3, r = 1.3 * s;
            put('canopy', o.x + Math.cos(a) * r, o.y + (2.2 + k * 0.8) * s, o.z + Math.sin(a) * r,
              2.1 * s, 1.9 * s, a, pick(PAL.canopy, (h * 7 + k * 0.37) % 1));
          }
          put('trunk', o.x, o.y, o.z, s, 2.6 * s, 0, PAL.trunk);
          break;
        }
        case 'pine': {
          const s = o.scale * 1.2 * f;
          put('conifer', o.x, o.y, o.z, 4 * s, 9 * s, o.rot, pick(PAL.conifer, h));
          put('trunk', o.x, o.y, o.z, 0.8 * s, 2.0 * s, 0, PAL.trunk);
          break;
        }
        case 'bush': { const s = o.scale * f; put('bush', o.x, o.y + 0.6 * s, o.z, 1.5 * s, 1.1 * s, o.rot, pick(PAL.bush, h)); break; }
        case 'rock': { const s = o.scale * f; put('rock', o.x, o.y + 0.2 * s, o.z, 0.9 * s, 0.55 * s, o.rot, PAL.stone); break; }
        default: if (Math.abs(o.lateral) < 16) put('tuft', o.x, o.y, o.z, 0.8 * o.scale * f, 0.6 * o.scale * f, o.rot, pick(PAL.tuft, h));
      }
    }
    // Verge grass: tall tufts along both shoulders (render-only; non-collidable).
    // Rows sit on a fixed 0.8 m arc-length grid and are keyed by their index, so each
    // tuft stays put as the window slides (a car-relative start reshuffled them every metre).
    const road = sim.road, hill = d.hilliness, VS = 0.8;
    const i0 = Math.ceil(Math.max(0, st.road.d - 15) / VS), i1 = Math.floor(Math.min(road.length, st.road.d + 130) / VS);
    for (let i = i0; i <= i1; i++) {
      const c = road.sampleAt(i * VS);
      const rx = Math.cos(c.heading), rz = -Math.sin(c.heading);
      for (let k = 0; k < 10; k++) {
        const hh = hash1(i * 17 + k, seed + 71);
        if (hh > 0.55) continue;
        const side = k % 2 ? 1 : -1;
        const u = side * (ROAD.half + 0.9 + Math.pow(hash1(i * 29 + k, seed + 72), 1.6) * 7);
        const x = c.x + rx * u, z = c.z + rz * u;
        const f = vegFade('verge', Math.hypot(x - cx, z - cz));
        if (f <= 0.02) continue;
        const y = terrainSurfaceHeight(c, u, x, z, seed, hill);
        if (y < SEA_LEVEL + 0.3) continue;
        const s = (0.5 + hh * 0.9) * f;
        put('tuft', x, y, z, 1.5 * s, 0.75 * s, hh * 40, pick(PAL.tuft, hash1(i * 7 + k, seed + 73)));
      }
    }
    for (const k of Object.keys(this.veg)) {
      this.veg[k].count = n[k];
      this.veg[k].instanceMatrix.needsUpdate = true;
      if (this.veg[k].instanceColor) this.veg[k].instanceColor.needsUpdate = true;
    }
  }

  _updateCar(sim, st, car) {
    const { x, y, z, heading } = car;
    this.car.position.set(x, y, z);
    const fx = Math.sin(heading), fz = Math.cos(heading), rx = Math.cos(heading), rz = -Math.sin(heading);
    const L = 1.5, W = 0.85, s = this._susp;
    const gradA = (sim.terrainHeight(x + fx * L, z + fz * L) - sim.terrainHeight(x - fx * L, z - fz * L)) / (2 * L);
    const gradS = (sim.terrainHeight(x + rx * W, z + rz * W) - sim.terrainHeight(x - rx * W, z - rz * W)) / (2 * W);
    s.gradA = s.gradA === undefined ? gradA : s.gradA + (gradA - s.gradA) * 0.08;
    s.gradS = s.gradS === undefined ? gradS : s.gradS + (gradS - s.gradS) * 0.08;
    const f = new THREE.Vector3(fx, s.gradA, fz).normalize();
    const r = new THREE.Vector3(rx, s.gradS, rz).normalize();
    const up = new THREE.Vector3().crossVectors(f, r).normalize();
    r.crossVectors(up, f).normalize();
    this.car.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(r, up, f));
    const thr = st.action.throttle;
    const rollT = clamp(-st.action.steer * Math.min(Math.abs(st.car.speed) / 9, 1) * 0.05, -0.08, 0.08);
    s.pitch += (clamp(-thr * 0.025, -0.03, 0.03) - s.pitch) * 0.07;
    s.roll += (rollT - s.roll) * 0.07;
    this.chassis.rotation.set(s.pitch, 0, s.roll);
    const spin = car.roadD / 0.36, steer = clamp(st.action.steer, -1, 1) * 0.45;
    for (const w of this.wheels) { w.mesh.rotation.x = spin; w.steer.rotation.y = w.front ? steer : 0; }
  }

  _updatePrecip(st) {
    const k = Math.max(st.dials.rain, st.dials.snow);
    this.precip.visible = k > 0.001;
    if (!this.precip.visible) return;
    const snow = st.dials.snow > st.dials.rain;
    this.precipMat.opacity = Math.min(0.8, k);
    this.precipMat.size = snow ? 0.3 : 0.1;
    const cam = this.camera.position, fall = snow ? 5 : 30, span = 50;
    const pos = this.precip.geometry.attributes.position.array, b = this.precipBase;
    for (let i = 0; i < pos.length; i += 3) {
      pos[i] = cam.x + b[i] + (snow ? Math.sin(st.t + i) * 2 : 0);
      pos[i + 1] = cam.y + span - ((b[i + 1] + st.t * fall) % span) - 10;
      pos[i + 2] = cam.z + b[i + 2];
    }
    this.precip.geometry.attributes.position.needsUpdate = true;
  }

  // --- controls / io --------------------------------------------------------
  setExposure(v) { this._exposure = v; }

  setQuality(tier) {
    const cap = Math.min(window.devicePixelRatio || 1, 2);
    this.renderer.setPixelRatio(tier === 'low' ? 1 : tier === 'high' ? cap : Math.min(cap, 1.5));
    if (this.rw) this.setDisplaySize(this.rw, this.rh);
  }

  stats() {
    const info = this.renderer.info.render;
    let instances = 0;
    for (const k in this.veg) instances += this.veg[k].count;
    return { calls: info.calls, triangles: info.triangles, instances };
  }

  setDisplaySize(w, h) {
    this.rw = w; this.rh = h;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  /** Release GPU resources and the WebGL context (before creating another renderer). */
  dispose() {
    this.scene.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) for (const m of [].concat(o.material)) m.dispose();
    });
    for (const rt of [this._capRT, this._depthRT]) if (rt) rt.dispose();
    this.renderer.dispose();
    this.renderer.forceContextLoss();
  }

  /** Square RGBA frame (the model's data head): the display image at aspect 1,
   *  supersampled then box-filtered to `size`. Bottom-left origin, like readPixels. */
  capture(size = this.dataSize) {
    const ss = Math.max(1, Math.min(8, Math.ceil(640 / size)));
    const big = size * ss;
    if (!this._capRT || this._capRT.width !== big) {
      if (this._capRT) this._capRT.dispose();
      this._capRT = new THREE.WebGLRenderTarget(big, big, { samples: 4 });
      this._capBuf = new Uint8Array(big * big * 4);
    }
    const prevAspect = this.camera.aspect;
    this.camera.aspect = 1;
    this.camera.updateProjectionMatrix();
    this.renderer.setRenderTarget(this._capRT);
    this.renderer.render(this.scene, this.camera);
    this.renderer.readRenderTargetPixels(this._capRT, 0, 0, big, big, this._capBuf);
    this.renderer.setRenderTarget(null);
    this.camera.aspect = prevAspect;
    this.camera.updateProjectionMatrix();
    // Box-filter down to `size`, then a slight Gaussian: the reference is softer than
    // a supersampled render (spectrum slope matches at sigma ~0.4 px per 256 px).
    const src = this._capBuf, inv = 1 / (ss * ss), f = new Float32Array(size * size * 3);
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
      let r = 0, g = 0, b = 0;
      for (let yy = 0; yy < ss; yy++) {
        let o = ((y * ss + yy) * big + x * ss) * 4;
        for (let xx = 0; xx < ss; xx++, o += 4) { r += src[o]; g += src[o + 1]; b += src[o + 2]; }
      }
      const q = (y * size + x) * 3;
      f[q] = r * inv; f[q + 1] = g * inv; f[q + 2] = b * inv;
    }
    const sigma = CAPTURE_SOFTNESS * size / 256;
    const img = sigma > 0.15 ? blur3(f, size, sigma) : f;
    const out = new Uint8Array(size * size * 4);
    for (let i = 0, q = 0; i < size * size; i++, q += 3) {
      out[i * 4] = img[q] + 0.5; out[i * 4 + 1] = img[q + 1] + 0.5; out[i * 4 + 2] = img[q + 2] + 0.5; out[i * 4 + 3] = 255;
    }
    return out;
  }

  /** RGB + depth (v1 API). Depth via a MeshDepthMaterial override at aspect 1. */
  captureChannels(size = this.dataSize) {
    const rgb = this.capture(size);
    if (!this._depthRT || this._depthRT.width !== size) {
      if (this._depthRT) this._depthRT.dispose();
      this._depthRT = new THREE.WebGLRenderTarget(size, size);
      this._depthMat = new THREE.MeshDepthMaterial();
    }
    const prevAspect = this.camera.aspect;
    this.camera.aspect = 1; this.camera.updateProjectionMatrix();
    this.scene.overrideMaterial = this._depthMat;
    this.renderer.setRenderTarget(this._depthRT);
    this.renderer.render(this.scene, this.camera);
    const depth = new Uint8Array(size * size * 4);
    this.renderer.readRenderTargetPixels(this._depthRT, 0, 0, size, size, depth);
    this.renderer.setRenderTarget(null);
    this.scene.overrideMaterial = null;
    this.camera.aspect = prevAspect; this.camera.updateProjectionMatrix();
    return { rgb, depth, size };
  }
}

/** Separable Gaussian (radius 2) over a packed RGB float image, edge-clamped. */
function blur3(src, n, sigma) {
  const w = [0, 1, 2].map((k) => Math.exp(-(k * k) / (2 * sigma * sigma)));
  const norm = w[0] + 2 * (w[1] + w[2]);
  const k = w.map((v) => v / norm);
  const tmp = new Float32Array(src.length), out = new Float32Array(src.length);
  const pass = (a, b, horiz) => {
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) for (let c = 0; c < 3; c++) {
      let s = 0;
      for (let d = -2; d <= 2; d++) {
        const xx = horiz ? Math.min(n - 1, Math.max(0, x + d)) : x;
        const yy = horiz ? y : Math.min(n - 1, Math.max(0, y + d));
        s += a[(yy * n + xx) * 3 + c] * k[Math.abs(d)];
      }
      b[(y * n + x) * 3 + c] = s;
    }
  };
  pass(src, tmp, true);
  pass(tmp, out, false);
  return out;
}

// --- geometry helpers -------------------------------------------------------------
/** How far each side of a near-strip row may extend: the inner side of a bend stays
 *  inside its radius so neighbouring rows never fold over each other. */
function rowLimits(road, d) {
  let kPos = 0, kNeg = 0;
  for (let s = -36; s <= 36; s += 12) {
    const k = road.sampleAt(Math.max(0, d + s)).curvature;
    if (k > kPos) kPos = k;
    if (-k > kNeg) kNeg = -k;
  }
  return { pos: kPos > 1e-4 ? 0.7 / kPos : 1e9, neg: kNeg > 1e-4 ? 0.7 / kNeg : 1e9 };
}

/** Vertices of one near-strip row at arc-length d: [x, y, z, -dh/dx, -dh/dz, u, forest] per column. */
function nearRow(road, d, seed, hill) {
  const c = road.sampleAt(d);
  const rx = Math.cos(c.heading), rz = -Math.sin(c.heading);
  const { pos: limPos, neg: limNeg } = rowLimits(road, d);
  const out = new Float32Array(LAT.length * 7);
  for (let j = 0, o = 0; j < LAT.length; j++, o += 7) {
    let u = LAT[j];
    if (u > limPos) u = limPos; else if (-u > limNeg) u = -limNeg;
    const x = c.x + rx * u, z = c.z + rz * u;
    const h = terrainSurfaceHeight(c, u, x, z, seed, hill);
    const e = 0.6 + Math.abs(u) * 0.03;
    const hx = terrainSurfaceHeight(c, u + e * rx, x + e, z, seed, hill);
    const hz = terrainSurfaceHeight(c, u + e * rz, x, z + e, seed, hill);
    out[o] = x; out[o + 1] = h; out[o + 2] = z;
    out[o + 3] = (h - hx) / e; out[o + 4] = (h - hz) / e; out[o + 5] = u; out[o + 6] = forestMask(x, z, seed);
  }
  return out;
}

/** A dynamic rows x cols grid with the terrain attributes. */
function makeGridMesh(rows, cols, material) {
  const n = rows * cols;
  const geo = new THREE.BufferGeometry();
  const pos = new Float32Array(n * 3), nrm = new Float32Array(n * 3);
  const lat = new Float32Array(n), dd = new Float32Array(n), forest = new Float32Array(n);
  const attr = (a, k) => new THREE.BufferAttribute(a, k).setUsage(THREE.DynamicDrawUsage);
  geo.setAttribute('position', attr(pos, 3));
  geo.setAttribute('normal', attr(nrm, 3));
  geo.setAttribute('aLat', attr(lat, 1));
  geo.setAttribute('aD', attr(dd, 1));
  geo.setAttribute('aForest', attr(forest, 1));
  const index = new Uint32Array((rows - 1) * (cols - 1) * 6);
  let k = 0;
  for (let i = 0; i < rows - 1; i++) for (let j = 0; j < cols - 1; j++) {
    const a = i * cols + j, b = a + 1, c = a + cols, d = c + 1;
    index[k++] = a; index[k++] = c; index[k++] = b; index[k++] = b; index[k++] = c; index[k++] = d;
  }
  geo.setIndex(new THREE.BufferAttribute(index, 1));
  const mesh = new THREE.Mesh(geo, material);
  mesh.frustumCulled = false;
  return {
    mesh,
    setVertex(i, x, y, z, gx, gz, u, d, f) {
      pos[i * 3] = x; pos[i * 3 + 1] = y; pos[i * 3 + 2] = z;
      const l = Math.hypot(gx, 1, gz);
      nrm[i * 3] = gx / l; nrm[i * 3 + 1] = 1 / l; nrm[i * 3 + 2] = gz / l;
      lat[i] = u; dd[i] = d; forest[i] = f;
    },
    commit(usedRows, usedCols) {
      for (const k of ['position', 'normal', 'aLat', 'aD', 'aForest']) geo.attributes[k].needsUpdate = true;
      geo.setDrawRange(0, Math.max(0, usedRows - 1) * (usedCols - 1) * 6);
    },
  };
}

function solidGeo(g) {
  g.deleteAttribute('uv');
  return g;
}

/** Leaf-clump canopy: quads scattered over a unit sphere, normals pointing outward
 *  from the centre so the clump shades like a soft volume. */
function canopyGeometry(count, seed) {
  const r = TX.rng(seed), pos = [], nrm = [], uv = [];
  for (let i = 0; i < count; i++) {
    const th = r() * Math.PI * 2, ph = Math.acos(1 - 2 * Math.pow(r(), 0.8));
    const c = new THREE.Vector3(Math.sin(ph) * Math.cos(th), Math.cos(ph) * 0.8, Math.sin(ph) * Math.sin(th)).multiplyScalar(0.42);
    const n = c.clone().normalize();
    const t = new THREE.Vector3().crossVectors(n, new THREE.Vector3(r() - 0.5, 1, r() - 0.5)).normalize();
    const b = new THREE.Vector3().crossVectors(n, t).normalize();
    const s = 0.5 + r() * 0.25;
    const corners = [[-1, -1], [1, -1], [1, 1], [-1, -1], [1, 1], [-1, 1]];
    for (const [a, bb] of corners) {
      const p = c.clone().addScaledVector(t, a * s).addScaledVector(b, bb * s);
      pos.push(p.x, p.y, p.z);
      const pn = p.clone().normalize().lerp(n, 0.3).normalize();
      nrm.push(pn.x, pn.y, pn.z);
      uv.push((a + 1) / 2, (bb + 1) / 2);
    }
  }
  return cardGeo(pos, nrm, uv);
}

/** Conifer: three crossed vertical cards, normals leaning outward and up. */
function coniferGeometry() {
  const pos = [], nrm = [], uv = [];
  for (const ang of [0, Math.PI / 3, (2 * Math.PI) / 3]) {
    const ax = Math.cos(ang), az = Math.sin(ang);
    const corners = [[-1, 0], [1, 0], [1, 1], [-1, 0], [1, 1], [-1, 1]];
    for (const [a, h] of corners) {
      pos.push(ax * a * 0.5, h, az * a * 0.5);
      const n = new THREE.Vector3(ax * a, 0.55, az * a).normalize();
      nrm.push(n.x, n.y, n.z);
      uv.push((a + 1) / 2, h);
    }
  }
  return cardGeo(pos, nrm, uv);
}

/** Grass tuft: two crossed cards, normals mostly up. */
function tuftGeometry() {
  const pos = [], nrm = [], uv = [];
  for (const ang of [0, Math.PI / 2]) {
    const ax = Math.cos(ang), az = Math.sin(ang);
    for (const [a, h] of [[-1, 0], [1, 0], [1, 1], [-1, 0], [1, 1], [-1, 1]]) {
      pos.push(ax * a * 0.5, h, az * a * 0.5);
      nrm.push(ax * a * 0.3, 1, az * a * 0.3);
      uv.push((a + 1) / 2, h);
    }
  }
  return cardGeo(pos, nrm, uv);
}

function cardGeo(pos, nrm, uv) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('normal', new THREE.Float32BufferAttribute(nrm, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  return g;
}

// --- colour / math helpers -------------------------------------------------------
/** Distance fade (0..1) per vegetation type: plants shrink into the ground before
 *  their draw window ends, so nothing pops in or out. Each `gone` sits inside its
 *  window (scatter: VEG_AHEAD arc metres; verge: 130 m). */
const VEG_FADE = {
  tree: [320, 420], pine: [320, 420], bush: [190, 250], rock: [210, 270],
  grass: [32, 52], verge: [70, 110],
};
function vegFade(type, dist) {
  const [full, gone] = VEG_FADE[type] || VEG_FADE.grass;
  return 1 - smoothstep(full, gone, dist);
}

/** Forest cover in [0,1]: the same grove field the core scatter clusters trees on. */
function forestMask(x, z, seed) {
  return smoothstep(-0.05, 0.45, fbm2(x * 0.012, z * 0.012, seed + 880, 3));
}

/** Biome dials: arid (x=0) dries and yellows, lush (x=1) deepens green; alien (y=1)
 *  rotates hue. Default (0.5, 0.5) returns the reference colour unchanged. */
function biomeTint(c, bx, by, k = 1) {
  const hsl = {};
  c.getHSL(hsl);
  const dx = (bx - 0.5) * 2 * k, dy = Math.max(0, by - 0.5) * 2 * k;
  hsl.h = (hsl.h - dx * 0.03 + dy * 0.35 + 1) % 1;
  hsl.s = clamp(hsl.s * (1 + dx * 0.25), 0, 1);
  hsl.l = clamp(hsl.l * (1 - dx * 0.08), 0, 1);
  return c.setHSL(hsl.h, hsl.s, hsl.l);
}

function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }
function smoothstep(e0, e1, x) { const t = clamp((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t); }
function angleDiff(a, b) {
  let d = (a - b) % (Math.PI * 2);
  if (d > Math.PI) d -= Math.PI * 2;
  if (d < -Math.PI) d += Math.PI * 2;
  return d;
}
function lerpAngle(a, b, t) { return a + angleDiff(b, a) * t; }
