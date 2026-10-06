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
  const nothing = () => {};
  window.ez2d = {
    show: nothing, clear: nothing, setCropBox: nothing, frameCropBox: nothing,
    setPicking: nothing, setMeasuring: nothing, setMeasure: nothing, setCoverage: nothing,
  };
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

// Camera colours: placed well, matched few other photos, placed far off.
const CAMERA_COLORS = { ok: 0x8ab4f8, weak: 0xfcad70, far: 0xf28b82 };

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
  const lines = { ok: [], weak: [], far: [] };
  for (const cam of data.cameras) {
    const own = lines[cam.flag] ?? lines.ok;
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
      own.push(...cam.centre, ...world[i], ...world[i], ...world[(i + 1) % 4]);
    }
    // The top edge's midpoint, raised: which way is up in the photo.
    const top = toWorld([0, -h * 1.4, depth]);
    own.push(...world[0], ...top, ...top, ...world[1]);
  }
  for (const [flag, segments] of Object.entries(lines)) {
    if (!segments.length) continue;
    const frustums = new THREE.BufferGeometry();
    frustums.setAttribute("position", new THREE.Float32BufferAttribute(segments, 3));
    const material = new THREE.LineBasicMaterial({ color: CAMERA_COLORS[flag] });
    group.add(new THREE.LineSegments(frustums, material));
  }
  const unit = `points, ${data.cameras.length} cameras`;
  return { object: group, box, count: positions.count, unit, coverage: data.coverage ?? null };
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
  layoutCoverage(loaded.coverage ?? null);
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

// --- Coverage: the camera rings --------------------------------------------------
//
// On the camera placement, the cameras grouped into rings by height
// (ez2digitize.coverage.rings, upright frame): each ring a circle at its
// cameras' height and distance from the object, its gaps shaded and
// labelled, orange, or red when wider than `max_gap`. A point at azimuth a
// is (cos a, 0, -sin a) from the centre.

const coverageGroup = new THREE.Group();
scene.add(coverageGroup);
const coverageLabels = document.getElementById("coverage");
const RING_COLORS = { covered: 0x81c995, gap: 0xfcad70, wide: 0xf28b82 };
let coverageOn = true;
let gapLabels = []; // [{element, position}]

function clearCoverage() {
  for (const child of [...coverageGroup.children]) {
    coverageGroup.remove(child);
    child.geometry.dispose();
    child.material.dispose();
  }
  coverageLabels.replaceChildren();
  gapLabels = [];
}

function layoutCoverage(coverage) {
  clearCoverage();
  if (!coverage) return;
  const centre = new THREE.Vector3(...coverage.centre);
  const at = (ring, degrees) => {
    const a = THREE.MathUtils.degToRad(degrees);
    return new THREE.Vector3(
      centre.x + ring.radius * Math.cos(a),
      centre.y + ring.height,
      centre.z - ring.radius * Math.sin(a),
    );
  };
  const inGap = (ring, degrees) => ring.gaps.find((gap) => {
    const from = (degrees - gap.start + 360) % 360;
    return from < gap.degrees;
  });
  const colour = new THREE.Color();
  for (const ring of coverage.rings) {
    // The circle in 2° pieces, each coloured by whether it lies in a gap.
    const positions = [];
    const colours = [];
    for (let d = 0; d < 360; d += 2) {
      const gap = inGap(ring, d + 1);
      colour.set(!gap ? RING_COLORS.covered : gap.degrees > coverage.max_gap ? RING_COLORS.wide : RING_COLORS.gap);
      positions.push(...at(ring, d).toArray(), ...at(ring, d + 2).toArray());
      colours.push(...colour.toArray(), ...colour.toArray());
    }
    const circle = new THREE.BufferGeometry();
    circle.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    circle.setAttribute("color", new THREE.Float32BufferAttribute(colours, 3));
    coverageGroup.add(new THREE.LineSegments(circle, new THREE.LineBasicMaterial({ vertexColors: true })));
    for (const gap of ring.gaps) {
      const color = gap.degrees > coverage.max_gap ? RING_COLORS.wide : RING_COLORS.gap;
      const wedge = new THREE.Mesh(
        new THREE.CircleGeometry(
          ring.radius,
          Math.max(2, Math.ceil(gap.degrees / 4)),
          THREE.MathUtils.degToRad(gap.start),
          THREE.MathUtils.degToRad(gap.degrees),
        ),
        new THREE.MeshBasicMaterial({
          color, transparent: true, opacity: 0.16, side: THREE.DoubleSide, depthWrite: false,
        }),
      );
      wedge.rotation.x = -Math.PI / 2; // the circle's plane (XY) onto the ground plane
      wedge.position.set(centre.x, centre.y + ring.height, centre.z);
      coverageGroup.add(wedge);
      const element = document.createElement("div");
      element.className = "overlay gap";
      element.style.color = `#${new THREE.Color(color).getHexString()}`;
      element.textContent = `${Math.round(gap.degrees)}° gap`;
      coverageLabels.appendChild(element);
      gapLabels.push({ element, position: at(ring, gap.start + gap.degrees / 2) });
    }
  }
  setCoverage(coverageOn);
}

function setCoverage(on) {
  coverageOn = on;
  coverageGroup.visible = on;
  coverageLabels.style.display = on ? "block" : "none";
}

function placeGapLabels() {
  if (!coverageOn || !gapLabels.length) return;
  const rect = renderer.domElement.getBoundingClientRect();
  for (const { element, position } of gapLabels) {
    const p = position.clone().project(camera);
    element.style.display = p.z > 1 ? "none" : "block"; // behind the view
    element.style.left = `${rect.left + ((p.x + 1) / 2) * rect.width}px`;
    element.style.top = `${rect.top + ((1 - p.y) / 2) * rect.height}px`;
  }
}

// --- The crop box -------------------------------------------------------------
//
// A box in the upright frame (Y up), turned about Y by `yaw` degrees: what
// the dense reconstruction keeps (ez2digitize.crop). Dragging one of its six
// handles moves that face along its axis, the opposite face staying put;
// the new box is reported when the drag ends ("cropbox" event).

const cropGroup = new THREE.Group();
cropGroup.visible = false;
scene.add(cropGroup);
const cropBox = { centre: new THREE.Vector3(), half: new THREE.Vector3(1, 1, 1), yaw: 0 };
let cropEditable = false;
const unitBox = new THREE.BoxGeometry(2, 2, 2);
const boxFaces = new THREE.Mesh(
  unitBox,
  new THREE.MeshBasicMaterial({ color: 0xfbbc04, transparent: true, opacity: 0.06, depthWrite: false }),
);
const boxEdges = new THREE.LineSegments(
  new THREE.EdgesGeometry(unitBox),
  new THREE.LineBasicMaterial({ color: 0xfbbc04 }),
);
cropGroup.add(boxFaces, boxEdges);
const handleGeometry = new THREE.SphereGeometry(1, 16, 12);
const handles = [];
for (let axis = 0; axis < 3; axis++) {
  for (const sign of [-1, 1]) {
    const handle = new THREE.Mesh(handleGeometry, new THREE.MeshBasicMaterial({ color: 0xfbbc04 }));
    handle.userData = { axis, sign };
    handles.push(handle);
    cropGroup.add(handle);
  }
}

function layoutCropBox() {
  cropGroup.position.copy(cropBox.centre);
  cropGroup.rotation.set(0, THREE.MathUtils.degToRad(cropBox.yaw), 0);
  boxFaces.scale.copy(cropBox.half);
  boxEdges.scale.copy(cropBox.half);
  for (const handle of handles) {
    const { axis, sign } = handle.userData;
    handle.position.set(0, 0, 0).setComponent(axis, sign * cropBox.half.getComponent(axis));
    handle.visible = cropEditable;
  }
}

// Handles keep the same size on screen (about 7 px), however far the view is.
const HANDLE_PX = 7;
function scaleHandles() {
  if (!cropGroup.visible) return;
  const perPixel = (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2))) / renderer.domElement.clientHeight;
  const position = new THREE.Vector3();
  for (const handle of handles) {
    handle.getWorldPosition(position);
    handle.scale.setScalar(position.distanceTo(camera.position) * perPixel * HANDLE_PX);
  }
}

function cropReport() {
  report("cropbox", {
    centre: cropBox.centre.toArray(),
    half_size: cropBox.half.toArray(),
    yaw: cropBox.yaw,
  });
}

const pointer = new THREE.Vector2();
const raycaster = new THREE.Raycaster();
let drag = null;

function setPointer(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.set(
    ((event.clientX - rect.left) / rect.width) * 2 - 1,
    -((event.clientY - rect.top) / rect.height) * 2 + 1,
  );
  raycaster.setFromCamera(pointer, camera);
}

// Where the mouse ray passes closest to the line `origin + t * direction`: t.
function closestOnLine(origin, direction) {
  const ray = raycaster.ray;
  const w0 = origin.clone().sub(ray.origin);
  const b = direction.dot(ray.direction);
  const d = direction.dot(w0);
  const e = ray.direction.dot(w0);
  const denom = 1 - b * b; // both unit vectors
  return Math.abs(denom) < 1e-9 ? 0 : (b * e - d) / denom;
}

renderer.domElement.addEventListener("pointerdown", (event) => {
  if (!cropGroup.visible || !cropEditable || event.button !== 0) return;
  setPointer(event);
  const hit = raycaster.intersectObjects(handles, false)[0];
  if (!hit) return;
  const { axis, sign } = hit.object.userData;
  const direction = new THREE.Vector3().setComponent(axis, sign).applyQuaternion(cropGroup.quaternion);
  const face = cropBox.centre.clone().addScaledVector(direction, cropBox.half.getComponent(axis));
  const opposite = cropBox.centre.clone().addScaledVector(direction, -cropBox.half.getComponent(axis));
  drag = { axis, direction, face, opposite, start: closestOnLine(face, direction) };
  controls.enabled = false;
  try {
    renderer.domElement.setPointerCapture(event.pointerId);
  } catch {
    // synthetic events (tests) have no capturable pointer
  }
});

renderer.domElement.addEventListener("pointermove", (event) => {
  if (!drag) return;
  setPointer(event);
  const moved = closestOnLine(drag.face, drag.direction) - drag.start;
  const face = drag.face.clone().addScaledVector(drag.direction, moved);
  const minimum = (framedBox ? framedBox.getSize(new THREE.Vector3()).length() : 1) * 0.005;
  const extent = Math.max(face.clone().sub(drag.opposite).dot(drag.direction), minimum);
  cropBox.half.setComponent(drag.axis, extent / 2);
  cropBox.centre.copy(drag.opposite).addScaledVector(drag.direction, extent / 2);
  layoutCropBox();
});

function endDrag(event) {
  if (!drag) return;
  drag = null;
  controls.enabled = true;
  try {
    renderer.domElement.releasePointerCapture(event.pointerId);
  } catch {
    // see pointerdown
  }
  cropReport();
}
renderer.domElement.addEventListener("pointerup", endDrag);
renderer.domElement.addEventListener("pointercancel", endDrag);

// --- Picking points: two for the scale, three to level -----------------------
//
// While picking, a click (not a drag, which turns the view) picks the point
// of the cloud nearest the mouse ray. Once `count` points are picked they
// are reported (a "measure" or "level" event, upright frame) and picking
// ends. For the scale the app answers with setMeasure(points, label) to keep
// showing the two points with their distance.

const measureGroup = new THREE.Group();
measureGroup.visible = false;
scene.add(measureGroup);
const measureColor = 0x81c995;
const markers = [0, 1, 2].map(() => {
  const marker = new THREE.Mesh(handleGeometry, new THREE.MeshBasicMaterial({ color: measureColor, depthTest: false }));
  marker.renderOrder = 2;
  marker.visible = false;
  measureGroup.add(marker);
  return marker;
});
const measureLine = new THREE.Line(
  new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
  new THREE.LineBasicMaterial({ color: measureColor, depthTest: false }),
);
measureLine.renderOrder = 2;
measureGroup.add(measureLine);
const measureLabel = document.getElementById("measure");
const help = document.getElementById("help");
const HELP = help.textContent;
let pickCount = 0; // 0: not picking
let pickKind = "";
let picks = [];
let pressedAt = null;

function layoutMeasure(label = "") {
  measureGroup.visible = picks.length > 0;
  markers.forEach((marker, i) => {
    marker.visible = i < picks.length;
    if (marker.visible) marker.position.copy(picks[i]);
  });
  // Two points: the distance; three: the triangle of the ground plane.
  measureLine.visible = picks.length >= 2;
  if (picks.length >= 2) {
    measureLine.geometry.setFromPoints(picks.length === 3 ? [...picks, picks[0]] : picks);
  }
  measureLabel.textContent = label;
  measureLabel.style.display = picks.length === 2 && label ? "block" : "none";
}

function placeMeasureLabel() {
  if (picks.length !== 2 || measureLabel.style.display !== "block") return;
  const middle = picks[0].clone().add(picks[1]).multiplyScalar(0.5).project(camera);
  const rect = renderer.domElement.getBoundingClientRect();
  measureLabel.style.left = `${rect.left + ((middle.x + 1) / 2) * rect.width}px`;
  measureLabel.style.top = `${rect.top + ((1 - middle.y) / 2) * rect.height}px`;
}

// Markers keep the same size on screen (about 5 px), like the crop handles.
function scaleMarkers() {
  if (!measureGroup.visible) return;
  const perPixel = (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2))) / renderer.domElement.clientHeight;
  for (const marker of markers) {
    marker.scale.setScalar(marker.position.distanceTo(camera.position) * perPixel * 5);
  }
}

// The cloud's point nearest the mouse ray, in the upright (world) frame.
function pickPoint(event) {
  if (!content) return null;
  setPointer(event);
  const size = framedBox ? framedBox.getSize(new THREE.Vector3()).length() : 1;
  raycaster.params.Points.threshold = size * 0.004;
  const hits = raycaster
    .intersectObject(content, true)
    .filter((hit) => hit.object.isPoints || hit.object.isMesh);
  if (!hits.length) return null;
  // Of the points near the front-most hit, the one closest to the ray: the
  // click lands on the visible surface, not on points behind it.
  const front = hits[0].distance;
  const near = hits.filter((hit) => hit.distance <= front + size * 0.01);
  near.sort((a, b) => (a.distanceToRay ?? 0) - (b.distanceToRay ?? 0));
  return near[0].point.clone();
}

renderer.domElement.addEventListener("pointerdown", (event) => {
  if (pickCount && event.button === 0) pressedAt = [event.clientX, event.clientY];
});

renderer.domElement.addEventListener("pointerup", (event) => {
  if (!pickCount || !pressedAt || event.button !== 0) return;
  const moved = Math.hypot(event.clientX - pressedAt[0], event.clientY - pressedAt[1]);
  pressedAt = null;
  if (moved > 4) return; // a drag turned the view
  const point = pickPoint(event);
  if (!point) return;
  picks = [...picks, point];
  layoutMeasure();
  if (picks.length === pickCount) {
    const kind = pickKind;
    setPicking(0);
    report(kind, { points: picks.map((p) => p.toArray()) });
  }
});

const PICK_HELP = {
  measure: "Click two points of the model whose real distance you know",
  level: "Click three points on the surface the object stands on, far apart",
};

// count: how many points to pick (0 stops); kind: "measure" or "level".
function setPicking(count, kind = "") {
  pickCount = count;
  pickKind = kind;
  renderer.domElement.style.cursor = count ? "crosshair" : "";
  help.textContent = count ? PICK_HELP[kind] ?? `Click ${count} points` : HELP;
  if (count) {
    picks = [];
    layoutMeasure();
  }
}

function setMeasuring(on) {
  setPicking(on ? 2 : 0, "measure");
}

window.ez2d = {
  // Show or hide the coverage rings of the camera placement.
  setCoverage,
  // The gaps' labels as shown, for tests.
  gapLabels() {
    return gapLabels.map(({ element }) => element.textContent);
  },
  // Pick points (see above); setPicking(0) or setMeasuring(false) stops.
  setPicking,
  setMeasuring,
  // points: two [x, y, z] in the upright frame, or null to hide them.
  setMeasure(points, label = "") {
    picks = points ? points.map((p) => new THREE.Vector3(...p)) : [];
    layoutMeasure(label);
  },
  // Where a point of the upright frame is on screen (client pixels), for tests.
  pointOnScreen(point) {
    scene.updateMatrixWorld(true);
    const p = new THREE.Vector3(...point).project(camera);
    const rect = renderer.domElement.getBoundingClientRect();
    return [rect.left + ((p.x + 1) / 2) * rect.width, rect.top + ((1 - p.y) / 2) * rect.height];
  },
  // box: {centre, half_size, yaw} in the upright frame, or null to hide it.
  setCropBox(box, editable = true) {
    cropGroup.visible = Boolean(box);
    cropEditable = Boolean(box) && editable;
    if (box) {
      cropBox.centre.fromArray(box.centre);
      cropBox.half.fromArray(box.half_size);
      cropBox.yaw = box.yaw ?? 0;
      layoutCropBox();
    }
  },
  // Where a handle is on screen (client pixels), for tests that drag it.
  handleOnScreen(axis, sign) {
    const handle = handles.find((h) => h.userData.axis === axis && h.userData.sign === sign);
    scene.updateMatrixWorld(true);
    const p = handle.getWorldPosition(new THREE.Vector3()).project(camera);
    const rect = renderer.domElement.getBoundingClientRect();
    return [rect.left + ((p.x + 1) / 2) * rect.width, rect.top + ((1 - p.y) / 2) * rect.height];
  },
  // Frame the view on the crop box (after "Fit to the points").
  frameCropBox() {
    if (!cropGroup.visible) return;
    cropGroup.updateMatrixWorld(true);
    frame(new THREE.Box3().setFromObject(boxEdges).expandByScalar(cropBox.half.length() * 0.3));
  },
  show(spec) {
    show(spec).catch((err) => {
      hud.textContent = `${spec.label}: could not be shown`;
      report("error", { message: String(err && err.stack ? err.stack : err) });
    });
  },
  clear() {
    showing++;
    clear();
    clearCoverage();
    setPicking(0);
    picks = [];
    layoutMeasure();
    if (grid) scene.remove(grid);
    grid = null;
    hud.textContent = "Nothing to show yet";
  },
};

renderer.setAnimationLoop(() => {
  controls.update();
  scaleHandles();
  scaleMarkers();
  renderer.render(scene, camera);
  placeMeasureLabel();
  placeGapLabels();
});

const gl = renderer.getContext();
const debug = gl.getExtension("WEBGL_debug_renderer_info");
report("ready", {
  webgl: true,
  gpu: debug ? gl.getParameter(debug.UNMASKED_RENDERER_WEBGL) : "unknown",
  webgl2: renderer.capabilities.isWebGL2,
});
