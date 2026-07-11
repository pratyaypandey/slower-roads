// Scene assembly. Builds the Three.js scene graph from the road geometry and a
// car mesh, and exposes a chase camera that tracks the car. Renderer-agnostic:
// the caller passes in the THREE module and owns the actual renderer (WebGL in
// the browser, headless-gl in Node), so this file never touches a canvas.

export function createWorld(THREE, road, params) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(skyColor(params));

  const camera = new THREE.PerspectiveCamera(70, 1, 0.1, 500);

  const hemi = new THREE.HemisphereLight(0xbfd4ff, 0x354022, 0.9);
  scene.add(hemi);
  const sun = new THREE.DirectionalLight(0xffffff, 1.1);
  sun.position.set(40, 80, 20);
  scene.add(sun);

  // Ground plane, colored per (later) biome; flat for the skeleton.
  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(2000, 2000),
    new THREE.MeshLambertMaterial({ color: 0x4a6b34 })
  );
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = -0.01;
  scene.add(ground);

  scene.add(buildRoadMesh(THREE, road));

  const car = buildCarMesh(THREE);
  scene.add(car);

  return { scene, camera, carMesh: car };
}

// Places the car mesh and swings a chase camera behind its heading.
export function updateWorld(world, carState) {
  const { carMesh, camera } = world;
  carMesh.position.set(carState.x, 0.4, carState.z);
  carMesh.rotation.y = carState.heading;

  const back = 7;
  const height = 3;
  camera.position.set(
    carState.x - Math.sin(carState.heading) * back,
    height,
    carState.z - Math.cos(carState.heading) * back
  );
  camera.lookAt(
    carState.x + Math.sin(carState.heading) * 6,
    0.6,
    carState.z + Math.cos(carState.heading) * 6
  );
}

function buildRoadMesh(THREE, road) {
  const half = road.width / 2;
  const n = road.points.length;
  const positions = new Float32Array(n * 2 * 3);

  for (let i = 0; i < n; i++) {
    const p = road.points[i];
    const h = road.headings[i];
    // Left/right offsets are perpendicular to the heading.
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

  return new THREE.Mesh(geom, new THREE.MeshLambertMaterial({ color: 0x2b2b30 }));
}

function buildCarMesh(THREE) {
  const geom = new THREE.BoxGeometry(1.6, 0.8, 3.2);
  return new THREE.Mesh(geom, new THREE.MeshLambertMaterial({ color: 0xd23b3b }));
}

function skyColor(params) {
  // Placeholder tie to timeOfDay so the schema field is exercised; real
  // day/night visuals come with the weather/tod pass.
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
