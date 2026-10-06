// SPDX-FileCopyrightText: 2026 EZ2DIGITIZE contributors
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Viewer spike: show a Gaussian splat, a textured mesh or a point cloud.
// Query parameters: model=<url>, kind=splat|mesh|points, texture=<url> (mesh).
// Progress is reported as console lines starting with "EZ2D " + JSON, which
// the Qt host prints, so the spike can be checked without looking at it.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";

const params = new URLSearchParams(location.search);
const modelUrl = params.get("model");
const kind = params.get("kind") ?? "mesh";
const textureUrl = params.get("texture");
const hud = document.getElementById("hud");

function report(event, data = {}) {
  console.log("EZ2D " + JSON.stringify({ event, ...data }));
}

// Errors outside main() (e.g. no WebGL context) must also reach the host.
window.addEventListener("error", (e) => report("error", { message: String(e.message) }));

const renderer = new THREE.WebGLRenderer({ antialias: false });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x202124);
const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.01, 1000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 2.5));

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

function frame(box) {
  const center = box.getCenter(new THREE.Vector3());
  const radius = box.getSize(new THREE.Vector3()).length() / 2 || 1;
  const distance = radius / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2));
  camera.position.copy(center).add(new THREE.Vector3(0.6, 0.5, 0.6).normalize().multiplyScalar(distance));
  camera.near = distance / 100;
  camera.far = distance * 10;
  camera.updateProjectionMatrix();
  controls.target.copy(center);
  controls.update();
}

// Bounds that ignore stray points: SfM and MVS clouds always have outliers
// far from the object, and framing on the full box makes the object tiny.
function robustBox(positions, lo = 0.02, hi = 0.98) {
  const n = positions.count;
  const step = Math.max(1, Math.floor(n / 100000));
  const axes = [[], [], []];
  for (let i = 0; i < n; i += step) {
    axes[0].push(positions.getX(i));
    axes[1].push(positions.getY(i));
    axes[2].push(positions.getZ(i));
  }
  const pick = (values, q) => values[Math.min(values.length - 1, Math.floor(q * values.length))];
  const min = [], max = [];
  for (const values of axes) {
    values.sort((a, b) => a - b);
    min.push(pick(values, lo));
    max.push(pick(values, hi));
  }
  return new THREE.Box3(new THREE.Vector3(...min), new THREE.Vector3(...max));
}

async function loadSplat() {
  scene.add(new SparkRenderer({ renderer }));
  const splats = new SplatMesh({ url: modelUrl });
  scene.add(splats);
  await splats.initialized;
  frame(splats.getBoundingBox(true));
  return splats.packedSplats?.numSplats ?? 0;
}

async function loadPly() {
  const geometry = await new PLYLoader().loadAsync(modelUrl);
  const box = robustBox(geometry.getAttribute("position"));
  if (kind === "points") {
    const material = new THREE.PointsMaterial({
      size: 2, sizeAttenuation: false, vertexColors: geometry.hasAttribute("color"),
    });
    scene.add(new THREE.Points(geometry, material));
    frame(box);
    return geometry.getAttribute("position").count;
  }
  geometry.computeVertexNormals();
  const material = new THREE.MeshStandardMaterial({ side: THREE.DoubleSide });
  if (textureUrl && geometry.hasAttribute("uv")) {
    const texture = await new THREE.TextureLoader().loadAsync(textureUrl);
    texture.colorSpace = THREE.SRGBColorSpace;
    material.map = texture;
    material.roughness = 1;
  } else if (geometry.hasAttribute("color")) {
    material.vertexColors = true;
  }
  scene.add(new THREE.Mesh(geometry, material));
  frame(box);
  const index = geometry.getIndex();
  return index ? index.count / 3 : geometry.getAttribute("position").count / 3;
}

// Frame-time measurement: average over a fixed window after loading.
let frames = 0;
let measureStart = null;
const MEASURE_MS = 3000;

// While measuring, orbit the camera: a still camera hides the cost of
// re-sorting splats, which happens whenever the view changes.
const ORBIT_RAD_PER_MS = (2 * Math.PI) / 6000; // one turn per 6 s
let lastTime = null;

function animate(time) {
  if (measureStart !== null && lastTime !== null) {
    const offset = camera.position.clone().sub(controls.target);
    offset.applyAxisAngle(camera.up, ORBIT_RAD_PER_MS * (time - lastTime));
    camera.position.copy(controls.target).add(offset);
  }
  lastTime = time;
  controls.update();
  renderer.render(scene, camera);
  if (measureStart !== null) {
    frames += 1;
    if (time - measureStart >= MEASURE_MS) {
      const fps = (frames * 1000) / (time - measureStart);
      report("fps", { fps: Math.round(fps * 10) / 10, orbiting: true });
      hud.textContent += `\n${fps.toFixed(1)} fps`;
      measureStart = null;
    }
  }
}

async function main() {
  const gl = renderer.getContext();
  const debug = gl.getExtension("WEBGL_debug_renderer_info");
  const gpu = debug ? gl.getParameter(debug.UNMASKED_RENDERER_WEBGL) : "unknown";
  report("start", { kind, model: modelUrl, gpu, webgl2: renderer.capabilities.isWebGL2 });

  const t0 = performance.now();
  const count = kind === "splat" ? await loadSplat() : await loadPly();
  const loadMs = Math.round(performance.now() - t0);
  const unit = kind === "splat" ? "splats" : kind === "points" ? "points" : "faces";
  hud.textContent = `${kind}: ${count.toLocaleString()} ${unit}, loaded in ${loadMs} ms\n${gpu}`;
  report("loaded", { count, unit, load_ms: loadMs });

  renderer.setAnimationLoop(animate);
  // Let the first frames (shader compilation, sorting) settle before measuring.
  setTimeout(() => { measureStart = performance.now(); frames = 0; }, 1000);
}

main().catch((err) => {
  hud.textContent = `error: ${err}`;
  report("error", { message: String(err && err.stack ? err.stack : err) });
});
