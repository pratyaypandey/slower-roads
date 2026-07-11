// SimRenderer — the RGB head (SIM.md §1.4, §7), "Slow Roads polish" look.
//
// Full-resolution, shadowed, post-processed render of the sim state: physically-based
// sky + clouds, soft aerial haze, smooth rolling terrain shaded by slope/altitude, a
// clean marked road, and dense instanced roadside vegetation. Display resolution is
// decoupled from the model's data resolution — humans view full-res here; capture(size)
// downsamples to the small square frame the autoencoder trains on. The camera is a pure
// function of sim state for reproducibility.

import * as THREE from 'three';
import { Sky } from '../vendor/Sky.js';
import { fbm2, hash1 } from '../core/prng.js';
import { terrainSurfaceHeight, SEA_LEVEL } from '../core/road.js';
import { EffectComposer } from '../vendor/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from '../vendor/jsm/postprocessing/RenderPass.js';
import { ShaderPass } from '../vendor/jsm/postprocessing/ShaderPass.js';
import { UnrealBloomPass } from '../vendor/jsm/postprocessing/UnrealBloomPass.js';
import { OutputPass } from '../vendor/jsm/postprocessing/OutputPass.js';
import { VignetteShader } from '../vendor/jsm/shaders/VignetteShader.js';

// Terrain strip.
const ROWS = 96, COLS = 65, LATERAL = 320, BACK = 90, AHEAD = 680;

// How far ahead to draw scattered vegetation (the core owns the scatter itself).
// Extend beyond the far-tree fade so new scatter rows enter at zero scale instead of
// becoming visible on the same frame they enter the active window.
const VEG_AHEAD = 420;
const CAP = { canopy: 1500, pine: 1100, trunk: 2600, rock: 800, bush: 1300, grass: 3200 };

export class SimRenderer {
  constructor(canvas, { dataSize = 128 } = {}) {
    this.dataSize = dataSize;
    this.rw = 0; this.rh = 0;

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.0;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.info.autoReset = false; // accumulate across composer passes; reset per frame
    this._exposure = 1.0;

    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.FogExp2(0xcfd6dd, 0.0016);
    this.camera = new THREE.PerspectiveCamera(58, 16 / 9, 0.5, 8000);

    this._sunDir = new THREE.Vector3();
    this._tmp = new THREE.Color();
    this._dummy = new THREE.Object3D();

    // Deterministic spring-arm camera state (advanced once per fixed sim step via
    // stepCamera, interpolated for display) + visual-suspension state. Scratch vectors
    // are reused to avoid per-frame allocation on the hot path.
    this._camPos = new THREE.Vector3();
    this._camAim = new THREE.Vector3();
    this._camPosPrev = new THREE.Vector3();
    this._camAimPrev = new THREE.Vector3();
    this._camInit = false;
    this._vegKey = null;
    this._suspPitch = 0; this._suspRoll = 0;
    this._v1 = new THREE.Vector3(); this._v2 = new THREE.Vector3();
    this._v3 = new THREE.Vector3(); this._v4 = new THREE.Vector3();

    this.lateralU = new Float32Array(COLS);
    for (let j = 0; j < COLS; j++) {
      const t = (j / (COLS - 1)) * 2 - 1;
      this.lateralU[j] = Math.sign(t) * LATERAL * Math.pow(Math.abs(t), 2.2);
    }

    this._buildSky();
    this._buildEnv();
    this._buildBackdrop();
    this._buildClouds();
    this._buildLights();
    this._buildTerrain();
    this._buildWater();
    this._buildVegetation();
    this._buildLandmarks();
    this._buildCar();
    this._buildStars();
    this._buildPrecip();
    this._buildComposer();
  }

  // --- construction ---------------------------------------------------------
  _buildSky() {
    this.sky = new Sky();
    this.sky.scale.setScalar(6000);
    const u = this.sky.material.uniforms;
    u.turbidity.value = 6; u.rayleigh.value = 2.4;
    u.mieCoefficient.value = 0.005; u.mieDirectionalG.value = 0.8;
    this.scene.add(this.sky);
  }

  _buildEnv() {
    // Image-based lighting from the sky: gives materials (car paint, water, foliage)
    // correct ambient reflections instead of reflecting a black void. Regenerated only
    // when the sun moves notably, so it costs nothing while the time-of-day is steady.
    this.pmrem = new THREE.PMREMGenerator(this.renderer);
    this.envScene = new THREE.Scene();
    this.envSky = new Sky();
    this.envSky.scale.setScalar(1000);
    this.envScene.add(this.envSky);
    this._lastEnvElev = 999;
  }

  _updateEnv(elevDeg) {
    if (Math.abs(elevDeg - this._lastEnvElev) < 5) return;   // only on notable sun change
    this._lastEnvElev = elevDeg;
    const su = this.sky.material.uniforms, eu = this.envSky.material.uniforms;
    eu.turbidity.value = su.turbidity.value;
    eu.rayleigh.value = su.rayleigh.value;
    eu.mieCoefficient.value = su.mieCoefficient.value;
    eu.mieDirectionalG.value = su.mieDirectionalG.value;
    eu.sunPosition.value.copy(su.sunPosition.value);
    if (this._envRT) this._envRT.dispose();
    this._envRT = this.pmrem.fromScene(this.envScene, 0, 1, 20000);
    this.scene.environment = this._envRT.texture;
  }

  _buildBackdrop() {
    // Distant mountain silhouettes give the horizon structure (SIM_UPGRADE §5.3 background
    // layer). Two jagged rings at different depths; fog blends them to a faint ridge line.
    this.backdrop = new THREE.Group();
    this.backdrop.add(this._mountainRing(1350, 200, 46, 0x3c4a5e, 17));
    this.backdrop.add(this._mountainRing(1950, 330, 34, 0x323d50, 91));
    this.scene.add(this.backdrop);
  }

  _mountainRing(radius, maxH, segs, color, salt) {
    const pos = [];
    for (let i = 0; i < segs; i++) {
      const a0 = (i / segs) * Math.PI * 2, a1 = ((i + 1) / segs) * Math.PI * 2, am = ((i + 0.5) / segs) * Math.PI * 2;
      const h = maxH * (0.35 + 0.65 * hash1(i, salt));
      pos.push(Math.cos(a0) * radius, -20, Math.sin(a0) * radius);
      pos.push(Math.cos(am) * radius, h, Math.sin(am) * radius);
      pos.push(Math.cos(a1) * radius, -20, Math.sin(a1) * radius);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pos), 3));
    geo.computeVertexNormals();
    const mat = new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide, fog: true });
    const m = new THREE.Mesh(geo, mat);
    m.frustumCulled = false;
    return m;
  }

  _buildLandmarks() {
    this.landmarkPool = [];   // reconciled each frame against sim.landmarks()
    this.landmarkGroup = new THREE.Group();
    this.scene.add(this.landmarkGroup);
  }

  _buildClouds() {
    const c = document.createElement('canvas');
    c.width = c.height = 128;
    const g = c.getContext('2d');
    const grd = g.createRadialGradient(64, 64, 2, 64, 64, 62);
    grd.addColorStop(0, 'rgba(255,255,255,0.95)');
    grd.addColorStop(0.5, 'rgba(255,255,255,0.5)');
    grd.addColorStop(1, 'rgba(255,255,255,0)');
    g.fillStyle = grd; g.fillRect(0, 0, 128, 128);
    const tex = new THREE.CanvasTexture(c);
    this.clouds = [];
    for (let i = 0; i < 16; i++) {
      const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, opacity: 0.5, depthWrite: false, fog: true }));
      const scale = 400 + hash1(i * 31 + 5, 99) * 700;
      s.scale.set(scale, scale * 0.55, 1);
      s.userData.base = {
        a: hash1(i * 7 + 1, 7) * Math.PI * 2,
        r: 900 + hash1(i * 13, 3) * 1800,
        y: 550 + hash1(i * 17, 11) * 450,
        drift: 4 + hash1(i * 5, 21) * 8,
      };
      this.clouds.push(s);
      this.scene.add(s);
    }
  }

  _buildLights() {
    this.sun = new THREE.DirectionalLight(0xfff2e0, 3.0);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(2048, 2048);
    const sc = this.sun.shadow.camera;
    sc.near = 1; sc.far = 1100; sc.left = -160; sc.right = 160; sc.top = 160; sc.bottom = -160;
    sc.updateProjectionMatrix();
    this.sun.shadow.bias = -0.0004;
    this.sun.shadow.normalBias = 0.04;
    this.scene.add(this.sun, this.sun.target);
    this.hemi = new THREE.HemisphereLight(0xbcd0ea, 0x6a6350, 0.6);
    this.scene.add(this.hemi);
  }

  _buildTerrain() {
    const n = ROWS * COLS;
    const geo = new THREE.BufferGeometry();
    this.tPos = new Float32Array(n * 3);
    this.tCol = new Float32Array(n * 3);
    const index = [];
    for (let i = 0; i < ROWS - 1; i++)
      for (let j = 0; j < COLS - 1; j++) {
        const a = i * COLS + j, b = a + 1, c = a + COLS, d = c + 1;
        index.push(a, c, b, b, c, d);
      }
    geo.setAttribute('position', new THREE.BufferAttribute(this.tPos, 3).setUsage(THREE.DynamicDrawUsage));
    geo.setAttribute('color', new THREE.BufferAttribute(this.tCol, 3).setUsage(THREE.DynamicDrawUsage));
    geo.setIndex(index);
    const mat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.96, metalness: 0 });
    mat.envMapIntensity = 0.35;   // matte ground: a little IBL fill, keep shadow contrast
    this.terrain = new THREE.Mesh(geo, mat);
    this.terrain.receiveShadow = true;
    this.terrain.frustumCulled = false;
    this.scene.add(this.terrain);
  }

  _buildVegetation() {
    const S = (geo, mat, cap, cast = true) => {
      const m = new THREE.InstancedMesh(geo, mat, cap);
      m.castShadow = cast; m.receiveShadow = true;
      // Bounds are refreshed when the cached instance window advances, allowing
      // Three.js to skip the complete vegetation batch when it leaves the view.
      m.frustumCulled = true;
      m.count = 0;
      this.scene.add(m);
      return m;
    };
    this.foliageMat = new THREE.MeshStandardMaterial({ color: 0x4f7a3c, roughness: 0.9, flatShading: true });
    this.pineMat = new THREE.MeshStandardMaterial({ color: 0x3d5f33, roughness: 0.9, flatShading: true });
    this.trunkMat = new THREE.MeshStandardMaterial({ color: 0x5a4433, roughness: 1 });
    this.rockMat = new THREE.MeshStandardMaterial({ color: 0x74716a, roughness: 1, flatShading: true });
    this.bushMat = new THREE.MeshStandardMaterial({ color: 0x627a44, roughness: 1, flatShading: true });
    this.grassMat = new THREE.MeshStandardMaterial({
      color: 0x6f9a4a, roughness: 1, flatShading: true, side: THREE.DoubleSide,
    });
    for (const m of [this.foliageMat, this.pineMat, this.trunkMat, this.rockMat, this.bushMat, this.grassMat]) m.envMapIntensity = 0.35;

    this.veg = {
      canopy: S(new THREE.IcosahedronGeometry(2.1, 1), this.foliageMat, CAP.canopy),
      pine: S(new THREE.ConeGeometry(1.7, 6, 7), this.pineMat, CAP.pine),
      trunk: S(new THREE.CylinderGeometry(0.22, 0.32, 2.4, 5), this.trunkMat, CAP.trunk),
      rock: S(new THREE.IcosahedronGeometry(1.3, 0), this.rockMat, CAP.rock, false),
      bush: S(new THREE.IcosahedronGeometry(1.1, 0), this.bushMat, CAP.bush, false),
      grass: S(makeGrassGeometry(), this.grassMat, CAP.grass, false),
    };
  }

  _buildCar() {
    this.car = new THREE.Group();
    // Body/cabin/lights ride on a chassis subgroup that leans (squat/dive/roll) on the
    // suspension; the wheels stay on the car group so they keep contact with the ground.
    this.chassis = new THREE.Group();
    this.car.add(this.chassis);
    const W = 1.85; // body width

    // Body: a low-poly car silhouette extruded from a side profile (hood, raked
    // windshield, roofline, trunk) across the width — reads as a real car, not a box.
    const p = new THREE.Shape();
    p.moveTo(-2.02, 0.16);
    p.lineTo(-2.06, 0.60);
    p.lineTo(-1.78, 0.72);        // trunk lip
    p.lineTo(-1.10, 0.80);
    p.lineTo(-0.72, 1.30);        // C-pillar up to the roof
    p.lineTo(0.34, 1.34);         // roof
    p.lineTo(0.98, 0.82);         // raked windshield
    p.lineTo(1.74, 0.76);         // hood
    p.lineTo(2.06, 0.66);
    p.lineTo(2.03, 0.20);         // front bumper
    p.closePath();
    const bodyGeo = new THREE.ExtrudeGeometry(p, {
      depth: W, bevelEnabled: true, bevelSize: 0.06, bevelThickness: 0.06, bevelSegments: 1, steps: 1,
    });
    bodyGeo.rotateY(-Math.PI / 2);   // profile length -> +Z (forward), extrude -> X (width)
    bodyGeo.translate(W / 2, 0, 0);  // centre the width
    bodyGeo.computeVertexNormals();
    const paint = new THREE.MeshStandardMaterial({ color: 0xbf3326, roughness: 0.5, metalness: 0.0, flatShading: true });
    paint.envMapIntensity = 0.55;
    const body = new THREE.Mesh(bodyGeo, paint);
    body.castShadow = true;
    this.chassis.add(body);

    // Greenhouse glass (windscreen + side windows).
    const glass = new THREE.MeshStandardMaterial({ color: 0x33404a, roughness: 0.15, metalness: 0.3 });
    const cabin = new THREE.Mesh(new THREE.BoxGeometry(W * 0.82, 0.46, 1.55), glass);
    cabin.position.set(0, 1.06, -0.18);
    cabin.castShadow = true;
    this.chassis.add(cabin);

    // Head- and tail-lights (emissive; modulated at night / under braking).
    this.headMat = new THREE.MeshStandardMaterial({ color: 0xfff6d8, emissive: 0xffefbe, emissiveIntensity: 0.2 });
    this.tailMat = new THREE.MeshStandardMaterial({ color: 0x4a0c0c, emissive: 0xff1e1e, emissiveIntensity: 0.35 });
    const lightGeo = new THREE.BoxGeometry(0.34, 0.16, 0.08);
    for (const sx of [-0.64, 0.64]) {
      const hl = new THREE.Mesh(lightGeo, this.headMat); hl.position.set(sx, 0.58, 2.04); this.chassis.add(hl);
      const tl = new THREE.Mesh(lightGeo, this.tailMat); tl.position.set(sx, 0.60, -2.06); this.chassis.add(tl);
    }

    // Wheels: axle along local X, so mesh.rotation.x spins them. Front wheels live in
    // a steer group (rotation.y) so they turn with the steering input.
    this.WHEEL_R = 0.44;
    const wg = new THREE.CylinderGeometry(this.WHEEL_R, this.WHEEL_R, 0.42, 16);
    wg.rotateZ(Math.PI / 2);
    const wm = new THREE.MeshStandardMaterial({ color: 0x161619, roughness: 0.7 });
    const hub = new THREE.MeshStandardMaterial({ color: 0x9a9aa2, roughness: 0.4, metalness: 0.6 });
    this.wheels = [];
    for (const [px, pz, front] of [[0.95, 1.32, true], [-0.95, 1.32, true], [0.95, -1.4, false], [-0.95, -1.4, false]]) {
      const steer = new THREE.Group();
      steer.position.set(px, this.WHEEL_R, pz);
      const mesh = new THREE.Mesh(wg, wm);
      mesh.castShadow = true;
      // A little hub disc so the spin reads.
      const cap = new THREE.Mesh(new THREE.CylinderGeometry(this.WHEEL_R * 0.4, this.WHEEL_R * 0.4, 0.44, 8), hub);
      cap.rotation.z = Math.PI / 2;
      mesh.add(cap);
      steer.add(mesh);
      this.car.add(steer);
      this.wheels.push({ steer, mesh, front });
    }
    this.scene.add(this.car);
  }

  _buildWater() {
    const geo = new THREE.PlaneGeometry(6000, 6000, 1, 1);
    geo.rotateX(-Math.PI / 2);
    const mat = new THREE.MeshStandardMaterial({
      color: 0x2f6d92, roughness: 0.12, metalness: 0.25, transparent: true, opacity: 0.86,
    });
    this.water = new THREE.Mesh(geo, mat);
    this.water.position.y = SEA_LEVEL;
    this.water.renderOrder = 1;
    this.scene.add(this.water);
  }

  _buildStars() {
    const N = 1200, pos = new Float32Array(N * 3);
    for (let i = 0; i < N; i++) {
      const a = fract(i * 0.123) * Math.PI * 2, b = fract(i * 0.789) * 0.5 + 0.02, r = 4000;
      pos[i * 3] = Math.cos(a) * Math.cos(b) * r;
      pos[i * 3 + 1] = Math.sin(b) * r;
      pos[i * 3 + 2] = Math.sin(a) * Math.cos(b) * r;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    this.starMat = new THREE.PointsMaterial({ color: 0xffffff, size: 7, sizeAttenuation: true, transparent: true, opacity: 0, fog: false });
    this.stars = new THREE.Points(geo, this.starMat);
    this.stars.frustumCulled = false;
    this.scene.add(this.stars);

    this.moons = [];
    for (let i = 0; i < 3; i++) {
      const m = new THREE.Mesh(new THREE.SphereGeometry(70, 16, 12), new THREE.MeshBasicMaterial({ color: 0xdfe3ee, fog: false }));
      m.visible = false; this.scene.add(m); this.moons.push(m);
    }
  }

  _buildPrecip() {
    this.precipN = 1000;
    const geo = new THREE.BufferGeometry();
    this.precipBase = new Float32Array(this.precipN * 3);
    this.precipPos = new Float32Array(this.precipN * 3);
    for (let i = 0; i < this.precipN; i++) {
      this.precipBase[i * 3] = (fract(i * 0.1234) - 0.5) * 100;
      this.precipBase[i * 3 + 1] = fract(i * 0.789) * 70;
      this.precipBase[i * 3 + 2] = (fract(i * 0.531) - 0.5) * 100;
    }
    geo.setAttribute('position', new THREE.BufferAttribute(this.precipPos, 3).setUsage(THREE.DynamicDrawUsage));
    this.precipMat = new THREE.PointsMaterial({ color: 0xcfe0f0, size: 0.22, transparent: true, opacity: 0, fog: true });
    this.precip = new THREE.Points(geo, this.precipMat);
    this.precip.frustumCulled = false;
    this.scene.add(this.precip);
  }

  _buildComposer() {
    const rt = new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, samples: 4 });
    this.composer = new EffectComposer(this.renderer, rt);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    // strength, radius, threshold — high threshold so only true highlights bloom
    // (keeps the bright sky from washing the horizon to white).
    this.bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.22, 0.7, 1.1);
    this.composer.addPass(this.bloom);
    this.vignette = new ShaderPass(VignetteShader);
    this.vignette.uniforms.offset.value = 1.0;
    this.vignette.uniforms.darkness.value = 1.05;
    this.composer.addPass(this.vignette);
    this.composer.addPass(new OutputPass());
  }

  // --- per-frame ------------------------------------------------------------
  /**
   * Advance the deterministic spring camera one fixed sim step. Called once per
   * sim.step() from the demo's fixed-timestep loop so the camera is replayable and
   * refresh-rate independent (see SIM_UPGRADE.md §7.1–7.2).
   */
  stepCamera(sim, dt) {
    const st = sim.state;
    const h = st.car.heading;
    const spd = Math.abs(st.car.speed);
    const fx = Math.sin(h), fz = Math.cos(h);

    // Desired position: behind + above the car, close enough that the world rushes past
    // (sense of speed) but pulled back enough that the road reads. Above terrain to avoid clipping.
    const back = 12.5 + spd * 0.12;
    const height = 5.0 + spd * 0.04;
    const dp = this._v1.set(st.car.x - fx * back, 0, st.car.z - fz * back);
    dp.y = Math.max(st.car.y + height, sim.terrainHeight(dp.x, dp.z) + 2.1);

    // Curve-aware aim: look along the road centerline ahead, not just the car heading,
    // so upcoming curves and crests are visible before the car reaches them.
    const laDist = 15 + spd * 0.5;
    const ahead = sim.road.sampleAt(st.road.d + laDist);
    const da = this._v2.set(ahead.x, ahead.y + 1.8, ahead.z);

    if (!this._camInit) { this._camPos.copy(dp); this._camAim.copy(da); this._camInit = true; }
    this._camPosPrev.copy(this._camPos);
    this._camAimPrev.copy(this._camAim);
    // Horizontal tracking stays responsive, but elevation follows a slower, asymmetric
    // envelope. Road samples are only metre-scale and the terrain-clearance max can
    // change abruptly while steering; feeding those changes directly into lookAt made
    // the whole horizon bob. Rise quickly enough to avoid clipping, settle down slowly.
    const posXZ = 1 - Math.exp(-10 * dt);
    const posYRate = dp.y > this._camPos.y ? 8 : 3.5;
    const posY = 1 - Math.exp(-posYRate * dt);
    this._camPos.x += (dp.x - this._camPos.x) * posXZ;
    this._camPos.z += (dp.z - this._camPos.z) * posXZ;
    this._camPos.y += (dp.y - this._camPos.y) * posY;

    const aimXZ = 1 - Math.exp(-13 * dt);
    const aimY = 1 - Math.exp(-4.5 * dt);
    this._camAim.x += (da.x - this._camAim.x) * aimXZ;
    this._camAim.z += (da.z - this._camAim.z) * aimXZ;
    this._camAim.y += (da.y - this._camAim.y) * aimY;
  }

  resetPresentation() {
    this._camInit = false;
    this._vegKey = null;
    this._terrKey = null;
    this._gradA = undefined;
    this._gradS = undefined;
    this._suspPitch = 0;
    this._suspRoll = 0;
  }

  render(sim, interp) {
    this.renderer.info.reset(); // accumulate scene + post passes for one frame (see stats())
    const st = sim.state;
    const a = interp ? interp.alpha : 1;
    const prev = interp ? interp.prevCar : null;
    // Interpolated car pose between the last two fixed steps -> smooth at any refresh.
    const car = prev ? {
      x: prev.x + (st.car.x - prev.x) * a,
      y: prev.y + (st.car.y - prev.y) * a,
      z: prev.z + (st.car.z - prev.z) * a,
      heading: lerpAngle(prev.heading, st.car.heading, a),
      slip: prev.slip + (st.car.slip - prev.slip) * a,
      roadD: prev.roadD + (st.road.d - prev.roadD) * a,
    } : { x: st.car.x, y: st.car.y, z: st.car.z, heading: st.car.heading, slip: st.car.slip, roadD: st.road.d };

    this._updateAtmosphere(st);
    this._applyCamera(st, a);
    this._updateBackdrop();
    this._updateTerrain(sim, st);
    this._updateVegetation(sim, st);
    this._updateLandmarks(sim, st);
    this._updateCar(sim, st, car);
    this._updateStarsMoons(st);
    this._updateClouds(st);
    this._updatePrecip(st);
    this.composer.render();
  }

  _applyCamera(st, alpha) {
    if (!this._camInit) return;
    const cp = this._v3.lerpVectors(this._camPosPrev, this._camPos, alpha);
    const ca = this._v4.lerpVectors(this._camAimPrev, this._camAim, alpha);
    this.camera.position.copy(cp);
    this.camera.lookAt(ca);
    this.camera.fov = 55 + clamp(Math.abs(st.car.speed) / 28, 0, 1) * 7 * st.dials.speedFeel;
    this.camera.updateProjectionMatrix();

    // Sun/shadow frustum, water plane, and star dome ride with the view.
    this.sun.target.position.set(st.car.x, st.car.y, st.car.z);
    this.sun.position.copy(this._sunDir).multiplyScalar(400).add(this.sun.target.position);
    this.stars.position.copy(cp);
    this.water.position.set(cp.x, SEA_LEVEL, cp.z);
  }

  _updateAtmosphere(st) {
    const d = st.dials;
    const theta = d.timeOfDay;
    const elevDeg = Math.sin(theta) * 78;
    const aziDeg = 180 + Math.cos(theta) * 110;
    this._sunDir.setFromSphericalCoords(1, THREE.MathUtils.degToRad(90 - elevDeg), THREE.MathUtils.degToRad(aziDeg));

    const day = smoothstep(-4, 12, elevDeg);
    const night = 1 - day;
    const sunset = Math.exp(-Math.pow(elevDeg / 12, 2));

    const su = this.sky.material.uniforms;
    su.sunPosition.value.copy(this._sunDir);
    su.rayleigh.value = 2.6 + sunset * 2.4;
    su.turbidity.value = 4 + d.fog * 6;
    su.mieCoefficient.value = 0.004 + d.fog * 0.008;
    this._updateEnv(elevDeg);   // refresh image-based lighting when the sun moves

    this.sun.color.copy(mixC(0xffb066, 0xfff4e2, day));
    this.sun.intensity = 0.12 + day * 3.0;

    // The env map now supplies most of the ambient fill, so keep the hemi light light.
    this.hemi.color.copy(mixC(0x24304a, 0xbcd4f0, day));
    this.hemi.groundColor.copy(tintFromBiome(d.biomeX, d.biomeY).multiplyScalar(0.5));
    this.hemi.intensity = 0.08 + day * 0.18;

    const haze = mixC(0x1b2436, 0xcdd6de, day);
    const duskHaze = mixC(haze.getHex(), 0xe7b489, sunset * day * 0.7);
    const fogc = mixC(duskHaze.getHex(), tintFromBiome(d.biomeX, d.biomeY).getHex(), 0.12);
    this.scene.fog.color.copy(fogc);
    this.scene.fog.density = 0.0011 + d.fog * d.fog * 0.015;

    this.renderer.toneMappingExposure = this._exposure * (0.36 + day * 0.32);
    this._day = day; this._night = night;
    this._fogColor = fogc;

    // Water reads bluer by day, dark at night, faintly warmed at dusk.
    this.water.material.color.copy(mixC(0x14273a, 0x2f6d92, day)).lerp(new THREE.Color(0x6a5a6e), sunset * day * 0.25);
  }

  _updateTerrain(sim, st) {
    // Rebuild the terrain strip only when the car has actually moved (or a terrain dial
    // changed), not every frame (SIM_UPGRADE §9.2 — eliminate hot-path regeneration).
    const d = st.dials;
    const key = `${sim.seed}:${Math.round(st.car.x)}:${Math.round(st.car.z)}:` +
      `${Math.round(d.hilliness * 1000)}:${Math.round(d.biomeX * 500)}:${Math.round(d.biomeY * 500)}:${Math.round(d.snow * 200)}:${Math.round(d.rain * 200)}`;
    if (key === this._terrKey) return;
    this._terrKey = key;

    const spacing = (BACK + AHEAD) / (ROWS - 1);
    const samples = sim.roadAhead(BACK, AHEAD, spacing);
    const hilliness = st.dials.hilliness;
    const snow = st.dials.snow, wet = st.dials.rain;
    const ground = tintFromBiome(st.dials.biomeX, st.dials.biomeY);
    const asphalt = this._tmp.setHex(0x45474e);
    const rock = new THREE.Color(0x6b6560);
    // Route class varies road treatment (SIM_UPGRADE §4.5): country lanes have no centre
    // line, causeways/coastal roads get a solid line, others dashed.
    const rc = st.beat.routeClass;
    const hasCenter = rc !== 'country' && rc !== 'forest';
    const solidCenter = rc === 'causeway' || rc === 'coastal';
    const hasEdge = rc === 'mountain' || rc === 'scenic' || rc === 'causeway';

    // Pass 1: positions.
    for (let i = 0; i < ROWS; i++) {
      const c = samples[Math.min(i, samples.length - 1)];
      const rx = Math.cos(c.heading), rz = -Math.sin(c.heading);
      for (let j = 0; j < COLS; j++) {
        const u = this.lateralU[j];
        const wx = c.x + rx * u, wz = c.z + rz * u;
        const vi = (i * COLS + j) * 3;
        this.tPos[vi] = wx;
        this.tPos[vi + 1] = terrainSurfaceHeight(c, u, wx, wz, sim.seed, hilliness);
        this.tPos[vi + 2] = wz;
      }
    }
    // Pass 2: colours from slope + altitude + road markings.
    for (let i = 0; i < ROWS; i++) {
      const c = samples[Math.min(i, samples.length - 1)];
      const half = c.width / 2;
      for (let j = 0; j < COLS; j++) {
        const u = this.lateralU[j], au = Math.abs(u);
        const vi = (i * COLS + j) * 3;
        let r, g, b;
        if (au <= half) {
          if (au < 0.11 && hasCenter && (solidCenter || i % 6 < 3)) { r = 0.6; g = 0.54; b = 0.32; }
          else if (hasEdge && au > half - 0.5 && au < half - 0.08) { r = 0.66; g = 0.66; b = 0.62; }
          else { r = asphalt.r; g = asphalt.g; b = asphalt.b; if (wet > 0) { r *= 1 - wet * 0.4; g *= 1 - wet * 0.4; b *= 1 - wet * 0.35; } }
        } else {
          const y = this.tPos[vi + 1];
          const slope = this._slopeAt(i, j);
          const wx = this.tPos[vi], wz = this.tPos[vi + 2];
          const v = 0.82 + 0.18 * (fbm2(wx * 0.06, wz * 0.06, sim.seed + 12, 3) * 0.5 + 0.5);
          const base = ground;
          r = base.r * v; g = base.g * v; b = base.b * v;
          // Rock on steep slopes.
          const rk = smoothstep(0.35, 0.75, slope);
          r += (rock.r - r) * rk; g += (rock.g - g) * rk; b += (rock.b - b) * rk;
          // Snow on high, flat-ish ground.
          if (snow > 0) { const t = snow * smoothstep(6, 22, y) * (1 - rk * 0.5); r += (0.93 - r) * t; g += (0.95 - g) * t; b += (0.99 - b) * t; }
        }
        this.tCol[vi] = r; this.tCol[vi + 1] = g; this.tCol[vi + 2] = b;
      }
    }
    const geo = this.terrain.geometry;
    geo.attributes.position.needsUpdate = true;
    geo.attributes.color.needsUpdate = true;
    geo.computeVertexNormals();
    geo.computeBoundingSphere();
  }

  _slopeAt(i, j) {
    const idx = i * COLS + j;
    const y = this.tPos[idx * 3 + 1];
    const yi = this.tPos[(Math.min(ROWS - 1, i + 1) * COLS + j) * 3 + 1];
    const yj = this.tPos[(i * COLS + Math.min(COLS - 1, j + 1)) * 3 + 1];
    return Math.min(1, (Math.abs(yi - y) + Math.abs(yj - y)) / 6);
  }

  _updateVegetation(sim, st) {
    const d = st.dials;

    // Material blends remain continuous even while the much more expensive instance
    // layout is cached.
    this.foliageMat.color.copy(mixC(0x4f7a3c, tintFromBiome(d.biomeX, d.biomeY).getHex(), 0.35));
    this.pineMat.color.copy(mixC(0x37552e, tintFromBiome(d.biomeX, d.biomeY).getHex(), 0.25));
    this.grassMat.color.copy(mixC(0x6f9a4a, tintFromBiome(d.biomeX, d.biomeY).getHex(), 0.4));
    // Snow accumulates on the canopies (SIM_UPGRADE §6.3 weather response).
    if (d.snow > 0) {
      const w = new THREE.Color(0xeef2f6);
      this.foliageMat.color.lerp(w, d.snow * 0.55);
      this.pineMat.color.lerp(w, d.snow * 0.6);
      this.grassMat.color.lerp(w, d.snow * 0.5);
    }

    // Refresh after each metre of world-space movement. This remains far cheaper than
    // uploading at 60-120 Hz, but keeps distance fades below a visible step and works
    // when the car travels off-road without advancing road distance. Fine dial buckets
    // make terrain-height/species transitions move a little at a time instead of in 2%
    // batches.
    const key = `${sim.seed}:${Math.floor(st.car.x)}:${Math.floor(st.car.z)}:` +
      `${Math.round(d.biomeX * 1000)}:${Math.round(d.hilliness * 1000)}`;
    if (key === this._vegKey) return;
    this._vegKey = key;

    const counts = { canopy: 0, pine: 0, trunk: 0, rock: 0, bush: 0, grass: 0 };
    const dummy = this._dummy;
    const put = (mesh, key, x, y, z, scaleY, rot, yOff = 0, scaleXZ = scaleY) => {
      if (counts[key] >= CAP[key]) return false;
      dummy.position.set(x, y + yOff, z);
      dummy.rotation.set(0, rot, 0);
      dummy.scale.set(scaleXZ, scaleY, scaleXZ);
      dummy.updateMatrix();
      mesh.setMatrixAt(counts[key]++, dummy.matrix);
      return true;
    };

    // Draw the SAME scatter the physics collides with (core is the single source).
    const objs = sim.scatter(BACK, VEG_AHEAD);
    for (let n = 0; n < objs.length; n++) {
      const o = objs[n];
      const dx = o.x - st.car.x, dz = o.z - st.car.z;
      const dist2 = dx * dx + dz * dz;
      const fade = vegetationFade(o.type, Math.sqrt(dist2));
      // Cull the complete multipart tree before its trunk becomes subpixel. Keeping a
      // tiny canopy after the trunk vanished was the source of trunkless silhouettes.
      const isTree = o.type === 'pine' || o.type === 'tree';
      if (fade <= (isTree ? 0.06 : 0.001)) continue;
      const s = o.scale * fade;
      switch (o.type) {
        case 'pine': {
          const trunkY = s * 0.8;
          const trunkXZ = o.scale * 0.8 * Math.sqrt(fade);
          // Lift the cone above its base so the lower trunk remains visible; with the
          // cone resting on the ground the trunk existed but was completely enclosed.
          put(this.veg.pine, 'pine', o.x, o.y, o.z, s, o.rot, 3.9 * s);
          put(this.veg.trunk, 'trunk', o.x, o.y, o.z, trunkY, o.rot, 1.2 * trunkY, trunkXZ);
          break;
        }
        case 'tree': {
          const trunkY = s * 0.9;
          const trunkXZ = o.scale * 0.9 * Math.sqrt(fade);
          put(this.veg.canopy, 'canopy', o.x, o.y, o.z, s, o.rot, 3.4 * s);
          put(this.veg.trunk, 'trunk', o.x, o.y, o.z, trunkY, o.rot, 1.2 * trunkY, trunkXZ);
          break;
        }
        case 'rock': put(this.veg.rock, 'rock', o.x, o.y, o.z, s, o.rot, 1.3 * s); break;
        case 'bush': put(this.veg.bush, 'bush', o.x, o.y, o.z, s, o.rot, 1.1 * s); break;
        default: put(this.veg.grass, 'grass', o.x, o.y, o.z, s, o.rot);
      }
    }
    for (const key of Object.keys(this.veg)) {
      const m = this.veg[key];
      m.count = counts[key];
      m.instanceMatrix.needsUpdate = true;
      if (m.computeBoundingSphere) m.computeBoundingSphere();
    }
  }

  _updateBackdrop() {
    // Ride with the camera so the ridge line is always on the horizon (infinite backdrop).
    if (this._camInit) this.backdrop.position.set(this.camera.position.x, 0, this.camera.position.z);
  }

  _updateLandmarks(sim, st) {
    const marks = sim.landmarks();
    const ground = tintFromBiome(st.dials.biomeX, st.dials.biomeY);
    for (let i = 0; i < marks.length; i++) {
      const mk = marks[i];
      let node = this.landmarkPool[i];
      if (!node || node.userData.type !== mk.type) {
        if (node) this.landmarkGroup.remove(node);
        node = makeLandmark(mk.type, ground);
        node.userData.type = mk.type;
        this.landmarkPool[i] = node;
        this.landmarkGroup.add(node);
      }
      node.visible = true;
      node.position.set(mk.x, mk.y, mk.z);
      node.rotation.y = (mk.id % 6) * 1.05;
    }
    for (let i = marks.length; i < this.landmarkPool.length; i++) {
      if (this.landmarkPool[i]) this.landmarkPool[i].visible = false;
    }
  }

  _updateCar(sim, st, car) {
    const { x, y, z, heading } = car;
    this.car.position.set(x, y, z);

    // Orient the car to the terrain: sample heights fore/aft and left/right to get
    // pitch (climb hills) and roll (cambered ground), so it hugs the surface.
    const fx = Math.sin(heading), fz = Math.cos(heading);
    const rx = Math.cos(heading), rz = -Math.sin(heading);
    const L = 1.7, W = 0.95;
    const hF = sim.terrainHeight(x + fx * L, z + fz * L);
    const hB = sim.terrainHeight(x - fx * L, z - fz * L);
    const hR = sim.terrainHeight(x + rx * W, z + rz * W);
    const hLt = sim.terrainHeight(x - rx * W, z - rz * W);
    // Terrain heights come from ~1 m-quantized road samples, so the raw pitch/roll
    // jitters frame to frame. Low-pass ONLY the tilt (not the heading) so the chassis
    // sits steady on the ground while steering stays crisp.
    const gradA = (hF - hB) / (2 * L), gradS = (hR - hLt) / (2 * W);
    this._gradA = this._gradA === undefined ? gradA : this._gradA + (gradA - this._gradA) * 0.055;
    this._gradS = this._gradS === undefined ? gradS : this._gradS + (gradS - this._gradS) * 0.055;
    const forwardV = this._v1.set(fx, this._gradA, fz).normalize();
    const rightV = this._v2.set(rx, this._gradS, rz).normalize();
    const upV = this._v3.crossVectors(forwardV, rightV).normalize();
    rightV.crossVectors(upV, forwardV).normalize();
    this.car.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(rightV, upV, forwardV));

    // Visual suspension on the chassis subgroup: nose lifts under power, dives under
    // braking, and the body leans out of corners. Wheels stay on the ground.
    const throttle = st.action.throttle;
    const rollTgt = clamp(-st.action.steer * Math.min(Math.abs(st.car.speed) / 9, 1) * 0.09 + car.slip * 0.015, -0.13, 0.13);
    const pitchTgt = clamp(-throttle * 0.05, -0.06, 0.06);
    this._suspPitch += (pitchTgt - this._suspPitch) * 0.07;
    this._suspRoll += (rollTgt - this._suspRoll) * 0.07;
    this.chassis.rotation.set(this._suspPitch, 0, this._suspRoll);

    // Wheels: spin by distance travelled, front wheels steer with the input.
    const spin = car.roadD / this.WHEEL_R;
    const steer = clamp(st.action.steer, -1, 1) * 0.5;
    for (const w of this.wheels) {
      w.mesh.rotation.x = spin;
      w.steer.rotation.y = w.front ? steer : 0;
    }

    // Lights: brake glow when decelerating, headlights up at night.
    this.tailMat.emissiveIntensity = throttle < -0.05 ? 1.4 : 0.3;
    this.headMat.emissiveIntensity = 0.15 + this._night * 1.4;
  }

  _updateStarsMoons(st) {
    this.starMat.opacity = Math.min(0.9, this._night * st.dials.starDensity * 1.4);
    this.stars.visible = this.starMat.opacity > 0.01;
    const count = Math.round(st.dials.moons);
    for (let i = 0; i < 3; i++) {
      const m = this.moons[i];
      if (i >= count) { m.visible = false; continue; }
      m.visible = this._night > 0.05;
      const a = i * 2.3 + 1.0;
      m.position.set(this.camera.position.x + Math.cos(a) * 3000, 1400 + i * 200, this.camera.position.z + Math.sin(a) * 3000);
      m.material.opacity = this._night;
    }
  }

  _updateClouds(st) {
    const cam = this.camera.position;
    const op = 0.5 * this._day * (0.4 + st.dials.fog * 0.6);
    const tint = mixC(0xffffff, this._fogColor.getHex(), 0.4);
    for (let i = 0; i < this.clouds.length; i++) {
      const s = this.clouds[i], b = s.userData.base;
      const ang = b.a + st.t * b.drift * 0.0016;
      s.position.set(cam.x + Math.cos(ang) * b.r, b.y, cam.z + Math.sin(ang) * b.r);
      s.material.opacity = op;
      s.material.color.copy(tint);
    }
  }

  _updatePrecip(st) {
    const intensity = Math.max(st.dials.rain, st.dials.snow);
    this.precipMat.opacity = Math.min(0.75, intensity * 0.85);
    if (intensity <= 0.001) { this.precip.visible = false; return; }
    this.precip.visible = true;
    const isSnow = st.dials.snow > st.dials.rain;
    this.precipMat.size = isSnow ? 0.4 : 0.16;
    this.precipMat.color.setHex(isSnow ? 0xffffff : 0xaec4e0);
    const cam = this.camera.position, fall = isSnow ? 6 : 32, span = 70;
    for (let i = 0; i < this.precipN; i++) {
      const bx = this.precipBase[i * 3], by = this.precipBase[i * 3 + 1], bz = this.precipBase[i * 3 + 2];
      const drift = isSnow ? Math.sin(st.t * 0.8 + i) * 3 : 0;
      const y = span - ((by + st.t * fall) % span);
      this.precipPos[i * 3] = cam.x + bx + drift;
      this.precipPos[i * 3 + 1] = cam.y + y - 12;
      this.precipPos[i * 3 + 2] = cam.z + bz;
    }
    this.precip.geometry.attributes.position.needsUpdate = true;
  }

  // --- controls / io --------------------------------------------------------
  setExposure(v) { this._exposure = v; }

  /** Quality tier (SIM_UPGRADE §9.6): scales DPR, shadow resolution, and bloom without
   *  touching the deterministic sim or oracle state. */
  setQuality(tier) {
    this._quality = tier;
    const cap = Math.min(window.devicePixelRatio || 1, 2);
    this.renderer.setPixelRatio(tier === 'low' ? 1 : tier === 'high' ? cap : Math.min(cap, 1.5));
    const sm = tier === 'low' ? 1024 : 2048;
    if (this.sun.shadow.mapSize.x !== sm) {
      this.sun.shadow.mapSize.set(sm, sm);
      if (this.sun.shadow.map) { this.sun.shadow.map.dispose(); this.sun.shadow.map = null; }
    }
    if (this.bloom) this.bloom.enabled = tier !== 'low';
    if (this.rw) this.setDisplaySize(this.rw, this.rh);
  }

  /** Render stats for the diagnostics overlay. */
  stats() {
    const info = this.renderer.info.render;
    let instances = 0;
    for (const k in this.veg) instances += this.veg[k].count;
    return { calls: info.calls, triangles: info.triangles, instances };
  }

  setDisplaySize(w, h) {
    this.rw = w; this.rh = h;
    this.renderer.setSize(w, h, false);
    this.composer.setSize(w, h);
    this.bloom.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  /** Downsampled square RGBA frame (the model's data head). Bypasses post-processing. */
  capture(size = this.dataSize) {
    if (!this._capRT || this._capRT.width !== size) {
      if (this._capRT) this._capRT.dispose();
      this._capRT = new THREE.WebGLRenderTarget(size, size, { minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter });
    }
    const prevAspect = this.camera.aspect;
    this.camera.aspect = 1;
    this.camera.updateProjectionMatrix();
    this.renderer.setRenderTarget(this._capRT);
    this.renderer.render(this.scene, this.camera);
    const buf = new Uint8Array(size * size * 4);
    this.renderer.readRenderTargetPixels(this._capRT, 0, 0, size, size, buf);
    this.renderer.setRenderTarget(null);
    this.camera.aspect = prevAspect;
    this.camera.updateProjectionMatrix();
    return buf;
  }

  /** Data head with an auxiliary depth channel (SIM_UPGRADE §10.3). RGB is the graded
   *  frame; depth is a linear-ish depth map (RGBA-packed) from a MeshDepth override.
   *  Cheap oracle channels the sim already knows (road look-ahead, drivable mask, surface)
   *  come from sim.skeleton()/frameLabels() rather than pixels. */
  captureChannels(size = this.dataSize) {
    const rgb = this.capture(size);
    if (!this._depthRT || this._depthRT.width !== size) {
      if (this._depthRT) this._depthRT.dispose();
      this._depthRT = new THREE.WebGLRenderTarget(size, size, { minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter });
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

// --- helpers ----------------------------------------------------------------
// Landmark families (SIM_UPGRADE §5.6) — big, iconic, readable-in-silhouette shapes
// placed by the scenic director at landmark beats. One group per type.
function makeLandmark(type, ground) {
  const g = new THREE.Group();
  const stone = () => new THREE.MeshStandardMaterial({ color: 0x6b6b73, roughness: 0.9, flatShading: true });
  const dark = () => new THREE.MeshStandardMaterial({ color: 0x2f2f38, roughness: 0.9, flatShading: true });
  if (type === 'arch') {
    const a = new THREE.Mesh(new THREE.TorusGeometry(9, 1.8, 6, 14, Math.PI), stone());
    a.position.y = 0; a.castShadow = true; g.add(a);
  } else if (type === 'lone_tree') {
    const trunk = new THREE.Mesh(new THREE.CylinderGeometry(1.1, 1.6, 12, 7), new THREE.MeshStandardMaterial({ color: 0x5a4433, roughness: 1 }));
    trunk.position.y = 6;
    const canopy = new THREE.Mesh(new THREE.IcosahedronGeometry(9, 1), new THREE.MeshStandardMaterial({ color: mixC(0x4a6b3a, ground.getHex(), 0.35).getHex(), roughness: 0.9, flatShading: true }));
    canopy.position.y = 15;
    trunk.castShadow = canopy.castShadow = true;
    g.add(trunk, canopy);
  } else if (type === 'monolith_ring') {
    for (let i = 0; i < 7; i++) {
      const a = (i / 7) * Math.PI * 2;
      const m = new THREE.Mesh(new THREE.BoxGeometry(2.2, 11 + (i % 3) * 3, 2.2), dark());
      m.position.set(Math.cos(a) * 10, (11 + (i % 3) * 3) / 2, Math.sin(a) * 10);
      m.castShadow = true; g.add(m);
    }
  } else if (type === 'crystal_spire') {
    const mat = new THREE.MeshStandardMaterial({ color: 0x9a6cff, emissive: 0x3a1f7a, roughness: 0.3, metalness: 0.2, flatShading: true });
    for (let i = 0; i < 4; i++) {
      const s = 1 - i * 0.18;
      const c = new THREE.Mesh(new THREE.OctahedronGeometry(4 * s, 0), mat);
      c.position.set((i - 1.5) * 3, 6 * s, (i % 2) * 2);
      c.scale.y = 2.4; c.castShadow = true; g.add(c);
    }
  } else if (type === 'tower') {
    const b = new THREE.Mesh(new THREE.CylinderGeometry(3, 4.5, 22, 8), stone());
    b.position.y = 11;
    const cap = new THREE.Mesh(new THREE.ConeGeometry(4, 6, 8), new THREE.MeshStandardMaterial({ color: 0x7a3b3b, roughness: 0.8, flatShading: true }));
    cap.position.y = 25;
    b.castShadow = cap.castShadow = true; g.add(b, cap);
  } else { // viaduct — a row of arches straddling low ground
    for (let i = -2; i <= 2; i++) {
      const pier = new THREE.Mesh(new THREE.BoxGeometry(2, 16, 3), stone());
      pier.position.set(i * 7, 8, 0); pier.castShadow = true; g.add(pier);
    }
    const deck = new THREE.Mesh(new THREE.BoxGeometry(38, 1.6, 5), stone());
    deck.position.y = 16.5; deck.castShadow = true; g.add(deck);
  }
  return g;
}

const BIOME = { aridEarth: 0xb8a878, lushEarth: 0x6f8f5a, aridAlien: 0x9a7f8f, lushAlien: 0x5f9a8f };
function tintFromBiome(bx, by) {
  const arid = mixC(BIOME.aridEarth, BIOME.aridAlien, by);
  const lush = mixC(BIOME.lushEarth, BIOME.lushAlien, by);
  return mixC(arid.getHex(), lush.getHex(), bx);
}
function mixC(hexA, hexB, t) { const a = new THREE.Color(hexA), b = new THREE.Color(hexB); return a.lerp(b, clamp(t, 0, 1)); }
function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }
function smoothstep(e0, e1, x) { const t = clamp((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t); }
function fract(x) { return x - Math.floor(x); }

function vegetationFade(type, distance) {
  let full, gone;
  switch (type) {
    case 'grass': full = 80; gone = 145; break;
    case 'bush': full = 165; gone = 255; break;
    case 'rock': full = 205; gone = 295; break;
    default: full = 300; gone = 390; break; // trees form the far silhouette
  }
  return 1 - smoothstep(full, gone, distance);
}

function makeGrassGeometry() {
  const positions = [];
  const width = 0.16, height = 1.1, lean = 0.12;
  for (const angle of [0, Math.PI / 3, Math.PI * 2 / 3]) {
    const rx = Math.cos(angle), rz = Math.sin(angle);
    const fx = -rz, fz = rx;
    positions.push(
      -rx * width, 0, -rz * width,
       rx * width, 0,  rz * width,
       fx * lean, height, fz * lean,
    );
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.computeVertexNormals();
  return geometry;
}
/** Shortest-arc interpolation between two angles (radians). */
function lerpAngle(a, b, t) {
  let d = ((b - a + Math.PI) % (Math.PI * 2)) - Math.PI;
  if (d < -Math.PI) d += Math.PI * 2;
  return a + d * t;
}
