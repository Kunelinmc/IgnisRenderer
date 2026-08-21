import {
	AmbientLight,
	DirectionalLight,
	FastApproximateAntiAliasingPass,
	GLTFLoader,
	MeshInstance,
	OrbitCamera,
	Platform,
	Renderer,
	Scene,
	SoftwareBackend,
	Vector3,
	WebGLBackend,
	WebGPUBackend,
} from "../../src/index";
import type { AnimationMixer, Node } from "../../src/index";
import { deformPrimitiveGeometry } from "../../src/simulation/animation/SoftwareAnimationDeformer";

const canvas = getElement<HTMLCanvasElement>("canvas3d");
const status = getElement<HTMLParagraphElement>("status");
const form = getElement<HTMLFormElement>("url-form");
const urlInput = getElement<HTMLInputElement>("model-url");
const fileInput = getElement<HTMLInputElement>("model-file");
const meshCount = getElement<HTMLElement>("mesh-count");
const materialCount = getElement<HTMLElement>("material-count");
const triangleCount = getElement<HTMLElement>("triangle-count");
const fps = getElement<HTMLElement>("fps");

const MODEL_TARGET_SIZE = 2;

const scene = new Scene();
scene.setSpatialIndexMode("hybrid");

const camera = createCamera();
scene.add(camera);
scene.add(new AmbientLight({ intensity: 1 }));
const sun = new DirectionalLight({
	intensity: 2.5,
	direction: { x: 0.4, y: -1, z: -0.5 },
});
scene.add(sun);

const shadowMap = scene.shadows.createCascaded({
	size: 2048,
	lambda: 0.65,
	maxDistance: 100,
	blendRatio: 0.1,
	stabilize: true,
	sampling: {
		filterMode: "pcf",
		pcfRadius: 1,
		radius: 0,
		samples: 1,
		searchSamples: 1,
		strength: 1,
	},
});
scene.shadows.bind(sun, shadowMap);

const query = new URLSearchParams(window.location.search);
const backendPreference = getBackendPreference(query.get("backend"));

const renderer = createRenderer(canvas, camera, backendPreference);
renderer.setScene(scene);

renderer.postProcess.registerPass(new FastApproximateAntiAliasingPass({ enabled: true }));

let loadedModel: Node | null = null;
let loadedMixer: AnimationMixer | null = null;
let loading = false;

await renderer.initialize();
renderer.renderLoop();
bindFrameRateCounter();
bindControls();

const initialModelURL = query.get("model");
if (initialModelURL) {
	urlInput.value = initialModelURL;
	void loadURL(initialModelURL);
} else {
	setStatus(`Ready (${renderer.getBackendDebugInfo().backend}).`);
}

function getElement<T extends HTMLElement>(id: string): T {
	const element = document.getElementById(id);
	if (!element) {
		throw new Error(`Missing element #${id}.`);
	}
	return element as T;
}

function createCamera(): OrbitCamera {
	const result = new OrbitCamera(new Vector3(0, 0, 0), 4);
	result.fov = 45;
	result.near = 0.01;
	result.far = 1000;
	result.theta = 0.65;
	result.phi = 1.1;
	result.minDistance = 0.01;
	result.maxDistance = 100_000;
	result.updatePosition();
	return result;
}

type BackendPreference = "webgpu" | "webgl" | "software";

function getBackendPreference(value: string | null): BackendPreference {
	if (value === "webgpu" || value === "webgl" || value === "software") {
		return value;
	}
	return "webgpu";
}

function createRenderer(
	target: HTMLCanvasElement,
	viewCamera: OrbitCamera,
	preference: BackendPreference,
): Renderer {
	const platform = Platform.detect();
	if (preference === "webgpu") {
		if (!platform.hasWebGPU) {
			throw new Error("WebGPU was requested but is unavailable in this browser.");
		}
		const backend = new WebGPUBackend();
		return new Renderer(target, backend, viewCamera);
	}
	if (preference === "webgl") {
		if (!platform.hasWebGL2) {
			throw new Error("WebGL2 was requested but is unavailable in this browser.");
		}
		return new Renderer(target, new WebGLBackend(), viewCamera);
	}
	if (preference === "software") {
		return new Renderer(target, new SoftwareBackend(), viewCamera);
	}
	throw new Error(`Unsupported backend preference: ${preference}.`);
}

function bindControls(): void {
	form.addEventListener("submit", (event) => {
		event.preventDefault();
		const url = urlInput.value.trim();
		if (url) {
			void loadURL(url);
		}
	});

	fileInput.addEventListener("change", () => {
		const file = fileInput.files?.[0];
		if (file) {
			void loadFile(file);
		}
	});

	canvas.addEventListener("dragover", (event) => {
		event.preventDefault();
	});
	canvas.addEventListener("drop", (event) => {
		event.preventDefault();
		const file = event.dataTransfer?.files[0];
		if (file) {
			void loadFile(file);
		}
	});

	bindOrbitControls();
	window.addEventListener("resize", () => {
		renderer.resizeCanvas();
	});
}

async function loadURL(url: string): Promise<void> {
	await loadModel(url, (loader) => loader.load(url));
}

async function loadFile(file: File): Promise<void> {
	await loadModel(file.name, async (loader) => {
		const isGLB = file.name.toLowerCase().endsWith(".glb");
		if (!isGLB) {
			setStatus("Local .gltf requires embedded resources; use a served URL otherwise.");
		}
		return loader.parse(await file.arrayBuffer());
	});
}

async function loadModel(
	label: string,
	load: (loader: GLTFLoader) => Promise<Node>,
): Promise<void> {
	if (loading) {
		return;
	}
	loading = true;
	setStatus(`Loading ${label}…`);
	const loader = new GLTFLoader();
	loader.on("progress", ({ loaded, total }) => {
		if (Number.isFinite(total) && total > 0) {
			setStatus(`Loading ${label}: ${Math.round((loaded / total) * 100)}%`);
		}
	});
	loader.on("parseprogress", ({ current, total, message }) => {
		setStatus(`Parsing ${message} (${current}/${total})`);
	});

	try {
		const root = await load(loader);
		if (loadedModel) {
			scene.remove(loadedModel);
		}
		if (loadedMixer) {
			renderer.animationSystem.removeMixer(loadedMixer);
		}
		normalizeModel(root);
		loadedModel = scene.add(root);
		scene.updateWorldMatrices();
		scene.syncNodeToECS();
		scene.updateWorldMatrices();
		loadedMixer = configureAnimations(root, loader);
		updateModelStats(root);
		frameCameraToScene();
		renderer.requestRender("scene");
		setStatus(`Loaded ${label}.`);
	} catch (error) {
		console.error("Failed to load glTF model.", error);
		setStatus(`Failed to load ${label}: ${formatError(error)}`);
	} finally {
		loading = false;
	}
}

function updateModelStats(root: Node): void {
	let meshes = 0;
	let triangles = 0;
	const materials = new Set();
	root.traverse((node) => {
		if (!(node instanceof MeshInstance)) return;
		meshes++;
		for (const primitive of node.mesh.primitives) {
			materials.add(primitive.material);
			if ((primitive.topology ?? "triangle-list") === "triangle-list") {
				triangles += Math.floor(primitive.geometry.indices.length / 3);
			}
		}
	});
	meshCount.textContent = String(meshes);
	materialCount.textContent = String(materials.size);
	triangleCount.textContent = triangles.toLocaleString();
}

function normalizeModel(root: Node): void {
	root.updateWorldMatrix();
	const bounds = getRenderableBounds(root);
	const center = {
		x: (bounds.min.x + bounds.max.x) * 0.5,
		y: (bounds.min.y + bounds.max.y) * 0.5,
		z: (bounds.min.z + bounds.max.z) * 0.5,
	};
	const largestDimension = Math.max(
		bounds.max.x - bounds.min.x,
		bounds.max.y - bounds.min.y,
		bounds.max.z - bounds.min.z,
	);
	if (!Number.isFinite(largestDimension) || largestDimension <= 0) {
		return;
	}

	// The glTF loader's root has no author transform, so normalize it directly.
	const scale = MODEL_TARGET_SIZE / largestDimension;
	root.setScale(scale, scale, scale);
	root.setPosition(-center.x * scale, -center.y * scale, -center.z * scale);
	root.updateWorldMatrix();
}

function bindFrameRateCounter(): void {
	let frameCount = 0;
	let sampleStart: number | null = null;

	renderer.on("tick", ({ now }) => {
		if (sampleStart === null) {
			sampleStart = now;
		}
		frameCount++;
		const elapsedMs = now - sampleStart;
		if (elapsedMs >= 500) {
			fps.textContent = String(Math.round((frameCount * 1000) / elapsedMs));
			frameCount = 0;
			sampleStart = now;
		}
	});
}

function frameCameraToScene(): void {
	if (!loadedModel) return;
	const bounds = getRenderableBounds(loadedModel);
	const center = {
		x: (bounds.min.x + bounds.max.x) * 0.5,
		y: (bounds.min.y + bounds.max.y) * 0.5,
		z: (bounds.min.z + bounds.max.z) * 0.5,
	};
	const radius = Math.max(
		Math.hypot(bounds.max.x - center.x, bounds.max.y - center.y, bounds.max.z - center.z),
		0.01,
	);
	camera.setTarget(center);
	camera.distance = radius * 2.6;
	camera.near = Math.max(radius / 1000, 0.001);
	camera.far = Math.max(radius * 100, 100);
	camera.updatePosition();
	shadowMap.maxDistance = Math.max(radius * 20, 20);
}

function configureAnimations(root: Node, loader: GLTFLoader): AnimationMixer | null {
	const bundle = loader.getLastAnimationBundle();
	if (!bundle || (bundle.clips.length === 0 && bundle.morphBindings.length === 0)) {
		return null;
	}

	const mixer = renderer.animationSystem.createMixer(root);
	const nodesById = new Map<string, Node>();
	root.traverse((node) => nodesById.set(node.id, node));
	for (const [path, nodeId] of Object.entries(bundle.nodePathMap)) {
		const node = nodesById.get(nodeId);
		if (node) mixer.bindNode(path, node);
	}
	for (const binding of bundle.morphBindings) {
		mixer.bindMorph(binding.path, binding.instance);
	}
	for (const clip of bundle.clips) {
		mixer.addClip(clip);
	}
	if (bundle.clips[0]) {
		mixer.clipAction(bundle.clips[0].name).play();
	}
	return mixer;
}

function getRenderableBounds(root: Node): ReturnType<Node["getWorldBoundingBox"]> {
	const bounds = {
		min: { x: Infinity, y: Infinity, z: Infinity },
		max: { x: -Infinity, y: -Infinity, z: -Infinity },
	};
	let hasGeometry = false;

	root.traverse((node) => {
		if (!(node instanceof MeshInstance)) return;
		for (
			let primitiveIndex = 0;
			primitiveIndex < node.mesh.primitives.length;
			primitiveIndex++
		) {
			const primitive = node.mesh.primitives[primitiveIndex];
			const morphWeights = node.morphWeights[primitiveIndex] ?? new Float32Array(0);
			const geometry =
				node.skeleton || morphWeights.length > 0
					? deformPrimitiveGeometry({
							geometry: primitive.geometry,
							morphWeights,
							skeleton: node.skeleton,
							meshWorldMatrix: node.worldMatrix,
						})
					: primitive.geometry;
			const positions = geometry.positions;
			for (let i = 0; i < positions.length; i += 3) {
				const point = node.worldMatrix.transformPoint({
					x: positions[i],
					y: positions[i + 1],
					z: positions[i + 2],
				});
				bounds.min.x = Math.min(bounds.min.x, point.x);
				bounds.min.y = Math.min(bounds.min.y, point.y);
				bounds.min.z = Math.min(bounds.min.z, point.z);
				bounds.max.x = Math.max(bounds.max.x, point.x);
				bounds.max.y = Math.max(bounds.max.y, point.y);
				bounds.max.z = Math.max(bounds.max.z, point.z);
				hasGeometry = true;
			}
		}
	});

	return hasGeometry ? bounds : root.getWorldBoundingBox();
}

function bindOrbitControls(): void {
	const rotationSensitivity = 0.004;
	const zoomSensitivity = 0.002;
	let activePointerId: number | null = null;
	let lastX = 0;
	let lastY = 0;

	canvas.addEventListener("pointerdown", (event) => {
		if (event.button !== 0) return;
		activePointerId = event.pointerId;
		lastX = event.clientX;
		lastY = event.clientY;
		canvas.setPointerCapture(event.pointerId);
	});
	canvas.addEventListener("pointermove", (event) => {
		if (activePointerId !== event.pointerId) return;
		camera.rotate(
			(event.clientX - lastX) * rotationSensitivity,
			(event.clientY - lastY) * rotationSensitivity,
		);
		lastX = event.clientX;
		lastY = event.clientY;
		renderer.requestRender("camera");
	});
	canvas.addEventListener("pointerup", releasePointer);
	canvas.addEventListener("pointercancel", releasePointer);
	canvas.addEventListener(
		"wheel",
		(event) => {
			event.preventDefault();
			camera.zoom(event.deltaY * camera.distance * zoomSensitivity);
			renderer.requestRender("camera");
		},
		{ passive: false },
	);

	function releasePointer(event: PointerEvent): void {
		if (activePointerId !== event.pointerId) return;
		activePointerId = null;
		if (canvas.hasPointerCapture(event.pointerId)) {
			canvas.releasePointerCapture(event.pointerId);
		}
	}
}

function setStatus(message: string): void {
	status.textContent = message;
}

function formatError(error: unknown): string {
	return error instanceof Error ? error.message : String(error);
}
