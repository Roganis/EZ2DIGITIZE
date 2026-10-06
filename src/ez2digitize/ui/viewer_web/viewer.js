// SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
// SPDX-License-Identifier: GPL-3.0-or-later
//
// The app's 3D viewer: a textured mesh (GLB), a point cloud (PLY), Gaussian
// splats (Spark) or the camera placement (sparse points and a frustum per
// photo). The Qt side calls `ez2d.show(spec)` (see ez2digitize.views.spec)
// and reads progress from console lines "EZ2D " + JSON (ui/viewer.py).

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";

const hud = document.getElementById("hud");

function report(event, data = {}) {
  console.log("EZ2D " + JSON.stringify({ event, ...data }));
}

window.addEventListener("error", (e) => report("error", { message: String(e.message) }));

// Without WebGL (no GPU driver Chromium accepts) there is nothing to draw
// with: say so, and still report "ready" so the app knows the page is up.
let renderer;
try {
  renderer = new THREE.WebGLRenderer({ antialias: true });
} catch (err) {
  hud.textContent = "The 3D view needs WebGL, which this graphics driver doesn't offer.";
  window.ez2d = { show() {}, clear() {} };
  report("ready", { webgl: false, gpu: null, error: String(err && err.message ? err.message : err) });
  throw err;
}
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x26282b);
scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 2.5));
const sun = new THREE.DirectionalLight(0xffffff, 1.0);
sun.position.set(1, 2, 1.5);
scene.add(sun);
scene.add(new SparkRenderer({ renderer }));

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.01, 1000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// What is shown: the model in `content`, turned upright by `upright`.
const upright = new THREE.Group();
scene.add(upright);
let content = null;
let grid = null;
let framedBox = null;
let showing = 0; // the latest show() call; older loads finishing late are dropped

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
renderer.domElement.addEventListener("dblclick", () => framedBox && frame(framedBox));

function frame(box) {
  framedBox = box;
  const center = box.getCenter(new THREE.Vector3());
  const radius = box.getSize(new THREE.Vector3()).length() / 2 || 1;
  const distance = radius / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2));
  const direction = new THREE.Vector3(0.6, 0.45, 0.8).normalize();
  camera.position.copy(center).addScaledVector(direction, distance);
  camera.near = distance / 200;
  camera.far = distance * 20;
  camera.updateProjectionMatrix();
  controls.target.copy(center);
  controls.update();
}

// A grid under the model, so up and size read at a glance.
function placeGrid(box) {
  if (grid) scene.remove(grid);
  const size = box.getSize(new THREE.Vector3());
  const span = Math.max(size.x, size.z) * 2 || 1;
  grid = new THREE.GridHelper(span, 10, 0x5f6368, 0x3c4043);
  grid.position.set(box.getCenter(new THREE.Vector3()).x, box.min.y, box.getCenter(new THREE.Vector3()).z);
  scene.add(grid);
}

// Bounds that ignore stray points: SfM and MVS clouds always have outliers
// far from the object, and framing on the full box makes the object tiny.
function robustBox(positions, matrix, lo = 0.02, hi = 0.98) {
  const n = positions.count;
  const step = Math.max(1, Math.floor(n / 100000));
  const axes = [[], [], []];
  const v = new THREE.Vector3();
  for (let i = 0; i < n; i += step) {
    v.fromBufferAttribute(positions, i).applyMatrix4(matrix);
    axes[0].push(v.x);
    axes[1].push(v.y);
    axes[2].push(v.z);
  }
  const pick = (values, q) => values[Math.min(values.length - 1, Math.floor(q * values.length))];
  const min = [];
  const max = [];
  for (const values of axes) {
    values.sort((a, b) => a - b);
    min.push(pick(values, lo));
    max.push(pick(values, hi));
  }
  return new THREE.Box3(new THREE.Vector3(...min), new THREE.Vector3(...max));
}

function clear() {
  if (!content) return;
  upright.remove(content);
  content.traverse((object) => {
    object.geometry?.dispose?.();
    const materials = Array.isArray(object.material) ? object.material : [object.material];
    for (const material of materials) {
      material?.map?.dispose?.();
      material?.dispose?.();
    }
    object.dispose?.(); // SplatMesh
  });
  content = null;
}

function setUpright(rows) {
  const m = new THREE.Matrix4();
  if (rows) {
    m.set(
      rows[0][0], rows[0][1], rows[0][2], 0,
      rows[1][0], rows[1][1], rows[1][2], 0,
      rows[2][0], rows[2][1], rows[2][2], 0,
      0, 0, 0, 1,
    );
  }
  upright.matrixAutoUpdate = false;
  upright.matrix.copy(m);
  upright.updateMatrixWorld(true);
  return m;
}

async function loadGlb(spec) {
  const gltf = await new GLTFLoader().loadAsync(spec.model);
  let faces = 0;
  gltf.scene.traverse((object) => {
    if (object.isMesh) {
      const index = object.geometry.getIndex();
      faces += index ? index.count / 3 : object.geometry.getAttribute("position").count / 3;
    }
  });
  return { object: gltf.scene, box: null, count: faces, unit: "faces" };
}

function pointsMaterial(geometry, size) {
  return new THREE.PointsMaterial({
    size, sizeAttenuation: false, vertexColors: geometry.hasAttribute("color"),
  });
}

async function loadPoints(spec, matrix) {
  const geometry = await new PLYLoader().loadAsync(spec.model);
  const positions = geometry.getAttribute("position");
  const object = new THREE.Points(geometry, pointsMaterial(geometry, 2));
  return { object, box: robustBox(positions, matrix), count: positions.count, unit: "points" };
}

async function loadSplat(spec) {
  const splats = new SplatMesh({ url: spec.model });
  await splats.initialized;
  return { object: splats, box: null, count: splats.packedSplats?.numSplats ?? 0, unit: "splats" };
}

// The sparse points and a small pyramid per camera, pointing where it looked.
async function loadCameras(spec, matrix) {
  const [geometry, data] = await Promise.all([
    new PLYLoader().loadAsync(spec.model),
    fetch(spec.cameras).then((r) => r.json()),
  ]);
  const group = new THREE.Group();
  const positions = geometry.getAttribute("position");
  group.add(new THREE.Points(geometry, pointsMaterial(geometry, 3)));
  const box = positions.count ? robustBox(positions, matrix) : new THREE.Box3();
  for (const cam of data.cameras) {
    box.expandByPoint(new THREE.Vector3(...cam.centre).applyMatrix4(matrix));
  }
  // Frustum depth: a tenth of the median camera distance to the scene centre.
  const centre = box.getCenter(new THREE.Vector3());
  const distances = data.cameras
    .map((c) => new THREE.Vector3(...c.centre).applyMatrix4(matrix).distanceTo(centre))
    .sort((a, b) => a - b);
  const depth = (distances[Math.floor(distances.length / 2)] || 1) * 0.1;
  const lines = [];
  for (const cam of data.cameras) {
    const h = depth * Math.tan(THREE.MathUtils.degToRad(cam.fov / 2));
    const w = h * cam.aspect;
    // Camera coordinates: x right, y down, z forward (COLMAP).
    const corners = [[-w, -h, depth], [w, -h, depth], [w, h, depth], [-w, h, depth]];
    const r = cam.rotation;
    const toWorld = ([x, y, z]) => [
      cam.centre[0] + r[0][0] * x + r[0][1] * y + r[0][2] * z,
      cam.centre[1] + r[1][0] * x + r[1][1] * y + r[1][2] * z,
      cam.centre[2] + r[2][0] * x + r[2][1] * y + r[2][2] * z,
    ];
    const world = corners.map(toWorld);
    for (let i = 0; i < 4; i++) {
      lines.push(...cam.centre, ...world[i], ...world[i], ...world[(i + 1) % 4]);
    }
    // The top edge's midpoint, raised: which way is up in the photo.
    const top = toWorld([0, -h * 1.4, depth]);
    lines.push(...world[0], ...top, ...top, ...world[1]);
  }
  const frustums = new THREE.BufferGeometry();
  frustums.setAttribute("position", new THREE.Float32BufferAttribute(lines, 3));
  group.add(new THREE.LineSegments(frustums, new THREE.LineBasicMaterial({ color: 0x8ab4f8 })));
  const unit = `points, ${data.cameras.length} cameras`;
  return { object: group, box, count: positions.count, unit };
}

const LOADERS = { glb: loadGlb, points: loadPoints, splat: loadSplat, cameras: loadCameras };

async function show(spec) {
  const call = ++showing;
  const loader = LOADERS[spec.kind];
  if (!loader) throw new Error(`unknown kind ${spec.kind}`);
  hud.textContent = `${spec.label}: loading…`;
  report("loading", { kind: spec.kind });
  const t0 = performance.now();
  const matrix = setUpright(spec.upright);
  const loaded = await loader(spec, matrix);
  if (call !== showing) return; // another view was asked for meanwhile
  clear();
  content = loaded.object;
  upright.add(content);
  upright.updateMatrixWorld(true);
  const box = loaded.box ?? new THREE.Box3().setFromObject(upright, true);
  if (spec.kind === "splat" && loaded.box === null) {
    box.copy(content.getBoundingBox(true)).applyMatrix4(matrix);
  }
  frame(box);
  placeGrid(box);
  const loadMs = Math.round(performance.now() - t0);
  hud.textContent = `${spec.label}: ${loaded.count.toLocaleString()} ${loaded.unit}`;
  report("loaded", { kind: spec.kind, count: loaded.count, unit: loaded.unit, load_ms: loadMs });
}

window.ez2d = {
  show(spec) {
    show(spec).catch((err) => {
      hud.textContent = `${spec.label}: could not be shown`;
      report("error", { message: String(err && err.stack ? err.stack : err) });
    });
  },
  clear() {
    showing++;
    clear();
    if (grid) scene.remove(grid);
    grid = null;
    hud.textContent = "Nothing to show yet";
  },
};

renderer.setAnimationLoop(() => {
  controls.update();
  renderer.render(scene, camera);
});

const gl = renderer.getContext();
const debug = gl.getExtension("WEBGL_debug_renderer_info");
report("ready", {
  webgl: true,
  gpu: debug ? gl.getParameter(debug.UNMASKED_RENDERER_WEBGL) : "unknown",
  webgl2: renderer.capabilities.isWebGL2,
});
