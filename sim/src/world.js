// Scene assembly. Builds the Three.js scene graph from the road geometry and a
// car mesh, and exposes a chase camera that tracks the car. Renderer-agnostic:
// the caller passes in the THREE module and owns the actual renderer (WebGL in
// the browser, headless-gl in Node), so this file never touches a canvas.
//
// Everything is deterministic: roadside props are derived from road geometry
// plus an index hash, never a live RNG, so a seed reproduces the exact scene.

export function createWorld(THREE, road, params) {
  const scene = new THREE.Scene();
  const sky = skyColor(params);
  scene.background = new THREE.Color(sky);
  // Fog hides the far edge of the finite road and adds depth; tint matches sky
  // so the horizon reads as distance rather than a wall.
  scene.fog = new THREE.Fog(sky, 60, 240);

  const camera = new THREE.PerspectiveCamera(72, 1, 0.1, 600);

  const hemi = new THREE.HemisphereLight(0xcfe0ff, 0x2c3a1e, 0.85);
  scene.add(hemi);
  const sun = new THREE.DirectionalLight(0xfff2d8, 1.15);
  sun.position.set(60, 100, 30);
  scene.add(sun);
  scene.add(new THREE.AmbientLight(0x404040, 0.4));

  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(4000, 4000),
    new THREE.MeshLambertMaterial({ color: 0x3f5f2e })
  );
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = -0.02;
  scene.add(ground);

  scene.add(buildRoadMesh(THREE, road));
  scene.add(buildLaneMarkings(THREE, road));
  scene.add(buildRoadsideProps(THREE, road));

  const car = buildCar(THREE);
  scene.add(car);

  return { scene, camera, carMesh: car, wheels: car.userData.wheels };
}

// Places the car and swings a chase camera behind its heading. Wheels spin and
// the body banks slightly into turns for a sense of motion.
export function updateWorld(world, carState) {
  const { carMesh, camera, wheels } = world;
  carMesh.position.set(carState.x, 0.35, carState.z);
  carMesh.rotation.y = carState.heading;

  // Spin wheels by accumulated distance so it tracks actual speed. Radius 0.36,
  // DT 1/30; kept on the world so it persists across frames without sim state.
  world._wheelSpin = (world._wheelSpin || 0) + (carState.speed / 0.36) * (1 / 30);
  for (const w of wheels) w.rotation.x = world._wheelSpin;

  const back = 8;
  const height = 3.4;
  camera.position.set(
    carState.x - Math.sin(carState.heading) * back,
    height,
    carState.z - Math.cos(carState.heading) * back
  );
  camera.lookAt(
    carState.x + Math.sin(carState.heading) * 10,
    0.8,
    carState.z + Math.cos(carState.heading) * 10
  );
}

function buildRoadMesh(THREE, road) {
  const half = road.width / 2;
  const n = road.points.length;
  const positions = new Float32Array(n * 2 * 3);

  for (let i = 0; i < n; i++) {
    const p = road.points[i];
    const h = road.headings[i];
    const nx = Math.cos(h);
    const nz = -Math.sin(h);
    positions[i * 6 + 0] = p.x - nx * half;
    positions[i * 6 + 1] = 0;
    positions[i * 6 + 2] = p.z - nz * half;
    positions[i * 6 + 3] = p.x + nx * half;
    positions[i * 6 + 4] = 0;
    positions[i * 6 + 5] = p.z + nz * half;
  }

  const indices = [];
  for (let i = 0; i < n - 1; i++) {
    const a = i * 2;
    indices.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geom.setIndex(indices);
  geom.computeVertexNormals();

  return new THREE.Mesh(geom, new THREE.MeshLambertMaterial({ color: 0x35353b }));
}

// Dashed centerline: short quads laid along the road, every few samples.
function buildLaneMarkings(THREE, road) {
  const dashLen = 2.4;
  const gap = 4;
  const wHalf = 0.14;
  const positions = [];
  const indices = [];
  let v = 0;

  const step = Math.round((dashLen + gap) / road.segmentLen);
  for (let i = 0; i < road.points.length - 2; i += step) {
    const a = road.points[i];
    const b = road.points[i + 1];
    const h = road.headings[i];
    const nx = Math.cos(h) * wHalf;
    const nz = -Math.sin(h) * wHalf;
    const y = 0.02;
    positions.push(
      a.x - nx, y, a.z - nz,
      a.x + nx, y, a.z + nz,
      b.x - nx, y, b.z - nz,
      b.x + nx, y, b.z + nz
    );
    indices.push(v, v + 1, v + 2, v + 1, v + 3, v + 2);
    v += 4;
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(positions), 3));
  geom.setIndex(indices);
  return new THREE.Mesh(geom, new THREE.MeshBasicMaterial({ color: 0xf4e6b0 }));
}

// Trees along both shoulders, instanced for cheapness. Placement is a pure
// function of road index (deterministic), so the scenery is reproducible.
function buildRoadsideProps(THREE, road) {
  const group = new THREE.Group();
  const trunkGeom = new THREE.CylinderGeometry(0.18, 0.25, 1.6, 5);
  const trunkMat = new THREE.MeshLambertMaterial({ color: 0x5a3d24 });
  const foliageGeom = new THREE.ConeGeometry(1.5, 3.2, 6);
  const foliageMat = new THREE.MeshLambertMaterial({ color: 0x2f5a2a });

  const spacing = 14; // metres between trees along the road
  const step = Math.round(spacing / road.segmentLen);
  const offset = road.width / 2 + 3;

  const slots = [];
  for (let i = 0; i < road.points.length; i += step) {
    const p = road.points[i];
    const h = road.headings[i];
    const nx = Math.cos(h);
    const nz = -Math.sin(h);
    // Jitter distance from the road by an index hash so the treeline isn't a
    // ruler-straight wall, while staying fully deterministic.
    const j = hash01(i) * 4;
    for (const side of [-1, 1]) {
      slots.push({
        x: p.x + nx * side * (offset + j),
        z: p.z + nz * side * (offset + j),
        s: 0.7 + hash01(i * 7 + side) * 0.8,
      });
    }
  }

  const trunks = new THREE.InstancedMesh(trunkGeom, trunkMat, slots.length);
  const foliage = new THREE.InstancedMesh(foliageGeom, foliageMat, slots.length);
  const m = new THREE.Matrix4();
  slots.forEach((t, i) => {
    m.makeScale(t.s, t.s, t.s);
    m.setPosition(t.x, 0.8 * t.s, t.z);
    trunks.setMatrixAt(i, m);
    m.makeScale(t.s, t.s, t.s);
    m.setPosition(t.x, (1.6 + 1.6) * t.s, t.z);
    foliage.setMatrixAt(i, m);
  });
  group.add(trunks, foliage);
  return group;
}

// A recognizable car: lower body, upper cabin, four wheels. Grouped so the sim
// moves one object; wheels are kept on userData so they can spin.
function buildCar(THREE) {
  const car = new THREE.Group();

  const body = new THREE.Mesh(
    new THREE.BoxGeometry(1.8, 0.5, 4),
    new THREE.MeshLambertMaterial({ color: 0xcc2b2b })
  );
  body.position.y = 0.45;
  car.add(body);

  const cabin = new THREE.Mesh(
    new THREE.BoxGeometry(1.5, 0.5, 1.9),
    new THREE.MeshLambertMaterial({ color: 0x8f1f1f })
  );
  cabin.position.set(0, 0.9, -0.2);
  car.add(cabin);

  const wheelGeom = new THREE.CylinderGeometry(0.36, 0.36, 0.3, 12);
  const wheelMat = new THREE.MeshLambertMaterial({ color: 0x1a1a1a });
  const wheels = [];
  const wx = 0.95;
  const wz = 1.3;
  for (const [sx, sz] of [[-wx, wz], [wx, wz], [-wx, -wz], [wx, -wz]]) {
    const w = new THREE.Mesh(wheelGeom, wheelMat);
    w.rotation.z = Math.PI / 2; // lay the cylinder on its side
    w.position.set(sx, 0.36, sz);
    car.add(w);
    wheels.push(w);
  }
  car.userData.wheels = wheels;
  return car;
}

function skyColor(params) {
  const noon = 0x87b7e0;
  const night = 0x0a1020;
  const t = 1 - Math.abs(params.timeOfDay - 0.5) * 2; // 1 at noon, 0 at midnight
  return lerpColor(night, noon, t);
}

function lerpColor(a, b, t) {
  const ar = (a >> 16) & 255, ag = (a >> 8) & 255, ab = a & 255;
  const br = (b >> 16) & 255, bg = (b >> 8) & 255, bb = b & 255;
  const r = Math.round(ar + (br - ar) * t);
  const g = Math.round(ag + (bg - ag) * t);
  const bl = Math.round(ab + (bb - ab) * t);
  return (r << 16) | (g << 8) | bl;
}

// Deterministic [0,1) hash of an integer — stand-in for a placement RNG that
// needs no state and always reproduces.
function hash01(n) {
  let x = (n | 0) * 374761393 + 668265263;
  x = (x ^ (x >>> 13)) * 1274126177;
  x = x ^ (x >>> 16);
  return ((x >>> 0) % 100000) / 100000;
}
