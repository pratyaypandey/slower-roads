// White hatchback for the v2 engine. Proportions are solved from the reference's
// median car sprite (the chase camera is rigid, so the car is a near-constant image;
// see eval/calibrate_view.py): long roof, big raked rear screen, high deck lip, a
// vertical rear panel, a full-width black lower bumper. One merged vertex-coloured
// geometry for the body plus separate wheels.

import * as THREE from 'three';
import { mergeGeometries } from '../vendor/jsm/utils/BufferGeometryUtils.js';

const PAINT = 0xffffff, GLASS = 0x6a7480, BLACK = 0x111213, TAIL = 0x3a1a1c;

function paint(geo, hex) {
  const c = new THREE.Color(hex);
  const g = geo.index ? geo.toNonIndexed() : geo;
  const n = g.attributes.position.count;
  const col = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) { col[i * 3] = c.r; col[i * 3 + 1] = c.g; col[i * 3 + 2] = c.b; }
  g.setAttribute('color', new THREE.BufferAttribute(col, 3));
  return g;
}

/** Extrude a side profile [(z, y), ...] across `width`, centred on x = 0. The bevel
 *  grows the outline by `bevel`, so profiles are given `bevel` inside the target. */
function extrudeProfile(pts, width, bevel) {
  const s = new THREE.Shape();
  s.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) s.lineTo(pts[i][0], pts[i][1]);
  s.closePath();
  const g = new THREE.ExtrudeGeometry(s, {
    depth: width - 2 * bevel, bevelEnabled: true, bevelSize: bevel, bevelThickness: bevel,
    bevelSegments: 3, steps: 1,
  });
  g.rotateY(-Math.PI / 2);           // profile length -> +z (forward), extrusion -> x
  g.translate(width / 2 - bevel, 0, 0);
  g.deleteAttribute('uv');
  g.computeVertexNormals();
  return g;
}

/** Narrow the greenhouse slightly towards the roof. */
function taper(geo, y0, k) {
  const p = geo.attributes.position;
  for (let i = 0; i < p.count; i++) {
    const y = p.getY(i);
    if (y > y0) p.setX(i, p.getX(i) * (1 - (y - y0) * k));
  }
  geo.computeVertexNormals();
  return geo;
}

function box(w, h, d, x, y, z) {
  const g = new THREE.BoxGeometry(w, h, d);
  g.translate(x, y, z);
  g.deleteAttribute('uv');
  return g;
}

function quad(a, b, c, d) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute([...a, ...b, ...c, ...a, ...c, ...d], 3));
  g.computeVertexNormals();
  return g;
}

export function buildCarGeometry() {
  const W = 2.3, B = 0.12;
  // Body (outline B inside the target surface): rear panel at z = -2.34, deck lip
  // at y = 1.03, black bumper zone below y = 0.66.
  const body = extrudeProfile([
    [-2.02, 0.32], [-2.12, 0.44], [-2.17, 0.62], [-2.18, 0.8], [-2.13, 0.9], [-1.8, 0.93],
    [1.3, 0.87], [2.08, 0.66], [2.16, 0.42], [2.04, 0.3],
  ], W, B);
  // Cabin: long roof, raked rear screen down to the deck.
  const CW = 1.84, CB = 0.08;
  const cabin = taper(extrudeProfile([
    [-1.86, 0.97], [-0.5, 1.43], [1.15, 1.46], [1.72, 0.92],
  ], CW, CB), 1.0, 0.12);

  // Rear screen, just proud of the cabin's raked surface.
  const along = (z0, y0, z1, y1) => { const l = Math.hypot(z1 - z0, y1 - y0); return [-(y1 - y0) / l, (z1 - z0) / l]; };
  const [nz, ny] = along(-1.86, 0.97, -0.5, 1.43);
  const off = CB + 0.012, rw = CW / 2 - 0.1;
  const rear = taper(quad([-rw, 1.09, -1.55], [rw, 1.09, -1.55], [rw - 0.04, 1.34, -0.64], [-rw + 0.04, 1.34, -0.64]), 1.0, 0.12);
  rear.translate(0, ny * off, nz * off);
  const [fz, fy] = along(1.15, 1.39, 1.72, 0.92);
  const front = taper(quad([rw, 0.98, 1.66], [-rw, 0.98, 1.66], [-rw, 1.36, 1.2], [rw, 1.36, 1.2]), 1.0, 0.12);
  front.translate(0, fy * off, fz * off);
  const sides = [-1, 1].map((sx) => {
    const x = sx * (CW / 2 + 0.012);
    return taper(quad([x, 1.02, -1.6], [x, 1.02, 1.55], [x, 1.33, 1.1], [x, 1.33, -0.6]), 1.0, 0.12);
  });

  // Rear: dark wedge lamps at the corners under the lip, full-width black bumper.
  const zr = -2.3;    // rear surface (profile + bevel) around lamp height
  const lamps = [-1, 1].map((sx) => box(0.5, 0.07, 0.03, sx * 0.72, 0.86, zr + 0.03));
  const bumper = box(W - 0.3, 0.34, 0.05, 0, 0.45, zr - 0.03);
  const under = box(W - 0.16, 0.12, 4.2, 0, 0.26, 0);

  return mergeGeometries([
    paint(body, PAINT), paint(cabin, PAINT),
    paint(rear, GLASS), paint(front, GLASS), ...sides.map((q) => paint(q, GLASS)),
    ...lamps.map((l) => paint(l, TAIL)),
    paint(bumper, BLACK), paint(under, BLACK),
  ]);
}

export function buildWheelGeometry(radius = 0.36, width = 0.26) {
  const g = new THREE.CylinderGeometry(radius, radius, width, 18);
  g.rotateZ(Math.PI / 2);
  g.deleteAttribute('uv');
  return paint(g, 0x141516);
}
