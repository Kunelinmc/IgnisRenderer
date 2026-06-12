import {
	AmbientLight,
	DirectionalLight,
	MeshFactory,
	OrbitCamera,
	PBRMaterial,
	Platform,
	PointLight,
	Renderer,
	Scene,
	Vector3,
	WebGLBackend,
	WebGPUBackend,
} from "../../src/index";
import type { RGB } from "../../src/index";

const TWEAKPANE_CDN_URL =
	"https://cdn.jsdelivr.net/npm/tweakpane@4.0.5/dist/tweakpane.min.js";
const MAX_LIGHTS = 512;
const DEFAULT_LIGHT_COUNT = 320;
const LEGACY_POINT_LIGHT_CAP = 16;

const LIGHT_PALETTE: RGB[] = [
	{ r: 255, g: 92, b: 88 },
	{ r: 255, g: 178, b: 70 },
	{ r: 86, g: 218, b: 148 },
	{ r: 78, g: 177, b: 255 },
	{ r: 176, g: 116, b: 255 },
	{ r: 255, g: 126, b: 199 },
	{ r: 127, g: 232, b: 224 },
	{ r: 255, g: 235, b: 130 },
];

interface TweakpaneBinding {
	refresh?: () => void;
	on: (
		eventName: "change" | "click",
		handler: (event: { value: unknown }) => void
	) => TweakpaneBinding;
}

interface TweakpanePane {
	addBinding: (
		target: Record<string, unknown>,
		key: string,
		options?: Record<string, unknown>
	) => TweakpaneBinding;
	addButton: (options: Record<string, unknown>) => TweakpaneBinding;
	addFolder: (options: Record<string, unknown>) => TweakpanePane;
	refresh?: () => void;
}

interface TweakpaneModule {
	Pane: new (options?: Record<string, unknown>) => TweakpanePane;
}

interface DemoLight {
	light: PointLight;
	marker: ReturnType<typeof MeshFactory.createSphere>;
	angle: number;
	radiusFactor: number;
	heightFactor: number;
	speed: number;
}

interface DemoSettings {
	lightCount: number;
	lightRange: number;
	lightIntensity: number;
	clustered: boolean;
	tileSizePx: number;
	zSlices: number;
	maxLightsPerCluster: number;
	animateLights: boolean;
	showMarkers: boolean;
	markerBudget: number;
	fieldRadius: number;
}

interface DemoPerformance {
	frameCount: number;
	accumulatedMs: number;
	lastUpdateMs: number;
	fps: number;
	frameMs: number;
}

interface DemoState {
	renderer: Renderer;
	scene: Scene;
	camera: OrbitCamera;
	lights: DemoLight[];
	settings: DemoSettings;
	performance: DemoPerformance;
	startedAt: number;
}

const statusDot = getElement<HTMLElement>("statusDot");
const statusText = getElement<HTMLElement>("statusText");
const backendMetric = getElement<HTMLElement>("backendMetric");
const modeMetric = getElement<HTMLElement>("modeMetric");
const fpsMetric = getElement<HTMLElement>("fpsMetric");
const frameMetric = getElement<HTMLElement>("frameMetric");
const lightMetric = getElement<HTMLElement>("lightMetric");
const gridMetric = getElement<HTMLElement>("gridMetric");
const errorMessage = getElement<HTMLElement>("errorMessage");
const errorText = getElement<HTMLElement>("errorText");
const canvas = getElement<HTMLCanvasElement>("canvas3d");

const state = await bootDemo();
bindOrbitControls(state);
void createTweakpane(state);

function getElement<T extends HTMLElement>(id: string): T {
	const element = document.getElementById(id);
	if (!element) {
		throw new Error(`Missing element #${id}.`);
	}
	return element as T;
}

async function bootDemo(): Promise<DemoState> {
	setStatus("Starting", "pending");
	try {
		const settings = createDefaultSettings();
		const scene = new Scene();
		scene.spatialIndexMode = "hybrid";

		const camera = createCamera();
		scene.add(camera);
		populateScene(scene);
		const lights = createLightPool(settings);
		const performance = createPerformanceState();

		const renderer = await createRenderer(canvas, camera);
		renderer.setScene(scene);
		renderer.features.enableShadows = false;
		renderer.features.enableEnvironment = false;
		renderer.features.enableReflection = false;
		renderer.features.enableOIT = false;
		renderer.features.enableClusteredLighting = settings.clustered;
		renderer.features.clusteredLightingOptions = readClusterOptions(settings);
		renderer.on("frameend", ({ now, deltaTime }) => {
			samplePerformance(performance, now, deltaTime);
			updateMetrics(demo);
		});

		const demo: DemoState = {
			renderer,
			scene,
			camera,
			lights,
			settings,
			performance,
			startedAt: performanceNow(),
		};

		applyLightMembership(demo);
		updateLights(demo, 0);

		await renderer.init();
		await renderer.warmup({ includeCorePasses: true });

		scene.syncNodeToECS();
		scene.updateWorldMatrices();
		renderer.requestRender("unknown");
		setStatus("Running", "ready");
		updateMetrics(demo);
		startAnimationLoop(demo);
		return demo;
	} catch (error) {
		const message = error instanceof Error ? error.message : String(error);
		setStatus("Failed", "error");
		errorText.textContent = message;
		errorMessage.classList.add("visible");
		throw error;
	}
}

function createDefaultSettings(): DemoSettings {
	return {
		lightCount: DEFAULT_LIGHT_COUNT,
		lightRange: 7.4,
		lightIntensity: 18,
		clustered: true,
		tileSizePx: 64,
		zSlices: 24,
		maxLightsPerCluster: MAX_LIGHTS,
		animateLights: true,
		showMarkers: true,
		markerBudget: 128,
		fieldRadius: 16,
	};
}

function createPerformanceState(): DemoPerformance {
	return {
		frameCount: 0,
		accumulatedMs: 0,
		lastUpdateMs: performanceNow(),
		fps: 0,
		frameMs: 0,
	};
}

function createCamera(): OrbitCamera {
	const camera = new OrbitCamera(new Vector3(0, 1.8, -0.6), 25);
	camera.fov = 52;
	camera.near = 0.08;
	camera.far = 120;
	camera.theta = -0.66;
	camera.phi = 1.08;
	camera.minDistance = 7;
	camera.maxDistance = 58;
	camera.lookSensitivity = 0.0042;
	camera.zoomSensitivity = 0.018;
	camera.updatePosition();
	return camera;
}

function resetCamera(demo: DemoState): void {
	demo.camera.setTarget({ x: 0, y: 1.8, z: -0.6 });
	demo.camera.distance = 25;
	demo.camera.theta = -0.66;
	demo.camera.phi = 1.08;
	demo.camera.updatePosition();
	demo.renderer.requestRender("camera");
}

function populateScene(scene: Scene): void {
	const floorMaterial = new PBRMaterial({
		name: "stress-floor",
		albedo: { r: 50, g: 52, b: 49 },
		roughness: 0.78,
		metalness: 0,
		doubleSided: true,
	});
	const floor = MeshFactory.createPlane(
		{ x: 0, y: -0.02, z: 0 },
		42,
		30,
		floorMaterial
	);
	scene.add(floor);

	const blockMaterials = [
		new PBRMaterial({
			name: "matte-basalt",
			albedo: { r: 96, g: 100, b: 96 },
			roughness: 0.68,
			metalness: 0.03,
		}),
		new PBRMaterial({
			name: "warm-concrete",
			albedo: { r: 158, g: 122, b: 90 },
			roughness: 0.72,
			metalness: 0.02,
		}),
		new PBRMaterial({
			name: "brushed-alloy",
			albedo: { r: 140, g: 150, b: 158 },
			roughness: 0.35,
			metalness: 0.52,
		}),
		new PBRMaterial({
			name: "dark-ceramic",
			albedo: { r: 66, g: 68, b: 76 },
			roughness: 0.44,
			metalness: 0.12,
		}),
	];

	for (let z = -4; z <= 4; z++) {
		for (let x = -6; x <= 6; x++) {
			const wave = Math.sin(x * 0.77 + z * 0.43);
			const height = 0.38 + ((Math.abs(x * 3 + z * 5) % 7) * 0.2) +
				Math.max(0, wave) * 0.36;
			const material =
				blockMaterials[Math.abs(x + z * 2) % blockMaterials.length];
			const block = MeshFactory.createBox(
				{ x: x * 2.35, y: height * 0.5, z: z * 2.45 - 0.2 },
				0.82 + (Math.abs(x) % 3) * 0.14,
				0.82 + (Math.abs(z) % 3) * 0.12,
				height,
				material
			);
			block.setRotationFromEuler(0, (x * 0.11 - z * 0.07), 0);
			scene.add(block);
		}
	}

	const centerMaterial = new PBRMaterial({
		name: "central-polished-alloy",
		albedo: { r: 190, g: 194, b: 184 },
		roughness: 0.22,
		metalness: 0.72,
	});
	const center = MeshFactory.createTorus(
		{ x: 0, y: 1.26, z: -0.7 },
		2.35,
		0.2,
		18,
		56,
		centerMaterial
	);
	center.setRotationFromEuler(Math.PI * 0.5, 0.18, 0);
	scene.add(center);

	const towerMaterial = new PBRMaterial({
		name: "receiver-towers",
		albedo: { r: 124, g: 127, b: 132 },
		roughness: 0.4,
		metalness: 0.38,
	});
	for (let index = 0; index < 18; index++) {
		const angle = (index / 18) * Math.PI * 2;
		const radius = 10.5 + (index % 3) * 2.2;
		const height = 1.5 + (index % 4) * 0.42;
		const tower = MeshFactory.createBox(
			{
				x: Math.cos(angle) * radius,
				y: height * 0.5,
				z: Math.sin(angle) * radius - 0.4,
			},
			0.42,
			0.42,
			height,
			towerMaterial
		);
		tower.setRotationFromEuler(0, -angle, 0);
		scene.add(tower);
	}

	scene.add(new AmbientLight({
		color: { r: 115, g: 122, b: 132 },
		intensity: 0.08,
	}));
	scene.add(new DirectionalLight({
		color: { r: 255, g: 244, b: 226 },
		intensity: 0.2,
		direction: { x: -0.28, y: -1, z: -0.36 },
	}));
}

function createLightPool(settings: DemoSettings): DemoLight[] {
	const lights: DemoLight[] = [];
	for (let index = 0; index < MAX_LIGHTS; index++) {
		const color = LIGHT_PALETTE[index % LIGHT_PALETTE.length];
		const angle = (index / MAX_LIGHTS) * Math.PI * 2;
		const radiusFactor = 0.36 + ((index * 37) % 100) / 100 * 0.72;
		const heightFactor = ((index * 17) % 100) / 100;
		const light = new PointLight({
			color,
			intensity: resolveLightIntensity(settings, index),
			position: { x: 0, y: 0, z: 0 },
			range: settings.lightRange,
		});
		const marker = MeshFactory.createSphere(
			{ x: 0, y: 0, z: 0 },
			0.065,
			6,
			4,
			new PBRMaterial({
				name: `stress-light-marker-${index}`,
				albedo: color,
				emissive: color,
				emissiveIntensity: 2.8,
				roughness: 0.2,
			})
		);
		lights.push({
			light,
			marker,
			angle,
			radiusFactor,
			heightFactor,
			speed: 0.18 + (index % 11) * 0.014,
		});
	}
	return lights;
}

async function createRenderer(
	target: HTMLCanvasElement,
	camera: OrbitCamera
): Promise<Renderer> {
	const platform = Platform.detect();
	if (platform.hasWebGPU) {
		return new Renderer(
			new WebGPUBackend({
				enableDeferredLighting: true,
				enableEarlyZPrepass: true,
				enableOcclusionCulling: false,
			}),
			target,
			camera
		);
	}
	if (platform.hasWebGL2) {
		return new Renderer(new WebGLBackend(), target, camera);
	}
	throw new Error("This browser does not expose WebGPU or WebGL2.");
}

async function createTweakpane(demo: DemoState): Promise<void> {
	try {
		const { Pane } = await import(
			/* @vite-ignore */ TWEAKPANE_CDN_URL
		) as TweakpaneModule;
		const pane = new Pane({ title: "Clustered Lighting" });
		const load = pane.addFolder({ title: "Light Load", expanded: true });
		load.addBinding(demo.settings as unknown as Record<string, unknown>, "lightCount", {
			label: "Lights",
			min: 16,
			max: MAX_LIGHTS,
			step: 16,
		}).on("change", () => {
			demo.settings.lightCount = clampInteger(
				demo.settings.lightCount,
				16,
				MAX_LIGHTS
			);
			applyLightMembership(demo);
			applyClusterOptions(demo);
			updateLights(demo, getElapsedSeconds(demo));
			requestSceneRender(demo, "unknown");
		});
		load.addBinding(demo.settings as unknown as Record<string, unknown>, "lightIntensity", {
			label: "Intensity",
			min: 4,
			max: 36,
			step: 1,
		}).on("change", () => {
			applyLightIntensity(demo);
			requestSceneRender(demo, "unknown");
		});
		load.addBinding(demo.settings as unknown as Record<string, unknown>, "lightRange", {
			label: "Range",
			min: 2.5,
			max: 12,
			step: 0.1,
		}).on("change", () => {
			applyLightRange(demo);
			requestSceneRender(demo, "unknown");
		});
		load.addBinding(demo.settings as unknown as Record<string, unknown>, "fieldRadius", {
			label: "Spread",
			min: 9,
			max: 24,
			step: 0.5,
		}).on("change", () => {
			updateLights(demo, getElapsedSeconds(demo));
			requestSceneRender(demo, "unknown");
		});

		const cluster = pane.addFolder({ title: "Cluster Grid", expanded: true });
		cluster.addBinding(demo.settings as unknown as Record<string, unknown>, "clustered", {
			label: "Clustered",
		}).on("change", () => {
			applyClusterOptions(demo);
			requestSceneRender(demo, "unknown");
		});
		cluster.addBinding(demo.settings as unknown as Record<string, unknown>, "tileSizePx", {
			label: "Tile",
			options: {
				"32 px": 32,
				"48 px": 48,
				"64 px": 64,
				"96 px": 96,
			},
		}).on("change", () => {
			applyClusterOptions(demo);
			requestSceneRender(demo, "unknown");
		});
		cluster.addBinding(demo.settings as unknown as Record<string, unknown>, "zSlices", {
			label: "Z slices",
			options: {
				"16": 16,
				"24": 24,
				"32": 32,
				"40": 40,
			},
		}).on("change", () => {
			applyClusterOptions(demo);
			requestSceneRender(demo, "unknown");
		});
		cluster.addBinding(
			demo.settings as unknown as Record<string, unknown>,
			"maxLightsPerCluster",
			{
				label: "Cluster cap",
				min: 16,
				max: MAX_LIGHTS,
				step: 8,
			}
		).on("change", () => {
			applyClusterOptions(demo);
			requestSceneRender(demo, "unknown");
		});

		const view = pane.addFolder({ title: "View", expanded: false });
		view.addBinding(demo.settings as unknown as Record<string, unknown>, "animateLights", {
			label: "Animate",
		}).on("change", () => requestSceneRender(demo, "unknown"));
		view.addBinding(demo.settings as unknown as Record<string, unknown>, "showMarkers", {
			label: "Markers",
		}).on("change", () => {
			applyLightMembership(demo);
			requestSceneRender(demo, "unknown");
		});
		view.addBinding(demo.settings as unknown as Record<string, unknown>, "markerBudget", {
			label: "Marker cap",
			min: 0,
			max: MAX_LIGHTS,
			step: 16,
		}).on("change", () => {
			demo.settings.markerBudget = clampInteger(
				demo.settings.markerBudget,
				0,
				MAX_LIGHTS
			);
			applyLightMembership(demo);
			requestSceneRender(demo, "unknown");
		});
		view.addButton({ title: "Reset Camera" }).on("click", () => {
			resetCamera(demo);
			updateMetrics(demo);
		});
		pane.refresh?.();
	} catch (error) {
		const detail = error instanceof Error ? error.message : String(error);
		setStatus("Controls unavailable", "error");
		errorText.textContent = `Tweakpane CDN failed to load: ${detail}`;
		errorMessage.classList.add("visible");
	}
}

function bindOrbitControls(demo: DemoState): void {
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
		const dx = event.clientX - lastX;
		const dy = event.clientY - lastY;
		lastX = event.clientX;
		lastY = event.clientY;
		if (dx === 0 && dy === 0) return;
		demo.camera.rotate(dx, dy);
		requestSceneRender(demo, "camera");
	});

	const clearPointer = (event: PointerEvent) => {
		if (activePointerId !== event.pointerId) return;
		activePointerId = null;
		if (canvas.hasPointerCapture(event.pointerId)) {
			canvas.releasePointerCapture(event.pointerId);
		}
	};
	canvas.addEventListener("pointerup", clearPointer);
	canvas.addEventListener("pointercancel", clearPointer);

	canvas.addEventListener(
		"wheel",
		(event) => {
			event.preventDefault();
			demo.camera.zoom(event.deltaY);
			requestSceneRender(demo, "camera");
		},
		{ passive: false }
	);

	window.addEventListener("resize", () => {
		demo.renderer.resizeCanvas();
		requestSceneRender(demo, "resize");
		updateMetrics(demo);
	});
}

function startAnimationLoop(demo: DemoState): void {
	const animate = () => {
		if (demo.settings.animateLights) {
			updateLights(demo, getElapsedSeconds(demo));
			requestSceneRender(demo, "unknown");
		}
		requestAnimationFrame(animate);
	};
	requestAnimationFrame(animate);
}

function updateLights(demo: DemoState, timeSeconds: number): void {
	const activeCount = getActiveLightCount(demo);
	for (let index = 0; index < activeCount; index++) {
		const item = demo.lights[index];
		const ringPhase = item.angle + timeSeconds * item.speed;
		const radius = demo.settings.fieldRadius * item.radiusFactor;
		const height = 0.8 + item.heightFactor * 4.7;
		const wobble = Math.sin(timeSeconds * 1.3 + index * 0.31) * 0.62;
		const x = Math.cos(ringPhase) * radius;
		const z = Math.sin(ringPhase * 1.17 + index * 0.019) * radius * 0.64 - 0.5;
		const y = height + wobble;
		item.light.position.set(x, y, z);
		item.marker.position.set(x, y, z);
		item.light.updateLocalMatrix();
		item.marker.updateLocalMatrix();
	}
}

function applyLightMembership(demo: DemoState): void {
	const activeCount = getActiveLightCount(demo);
	const markerCount = getVisibleMarkerCount(demo);
	for (let index = 0; index < demo.lights.length; index++) {
		const item = demo.lights[index];
		setNodeAttached(demo.scene, item.light, index < activeCount);
		setNodeAttached(demo.scene, item.marker, index < markerCount);
	}
	updateMetrics(demo);
}

function setNodeAttached(
	scene: Scene,
	node: DemoLight["light"] | DemoLight["marker"],
	attached: boolean
): void {
	const currentlyAttached = scene.contains(node);
	if (attached && !currentlyAttached) {
		scene.add(node);
		return;
	}
	if (!attached && currentlyAttached) {
		scene.remove(node);
	}
}

function applyLightIntensity(demo: DemoState): void {
	for (let index = 0; index < demo.lights.length; index++) {
		demo.lights[index].light.intensity = resolveLightIntensity(
			demo.settings,
			index
		);
	}
}

function applyLightRange(demo: DemoState): void {
	for (const item of demo.lights) {
		item.light.range = demo.settings.lightRange;
	}
}

function applyClusterOptions(demo: DemoState): void {
	demo.renderer.features.enableClusteredLighting = demo.settings.clustered;
	demo.renderer.features.clusteredLightingOptions = readClusterOptions(
		demo.settings
	);
	updateMetrics(demo);
}

function readClusterOptions(settings: DemoSettings): {
	tileSizePx: number;
	zSlices: number;
	maxLights: number;
	maxLightsPerCluster: number;
} {
	return {
		tileSizePx: settings.tileSizePx,
		zSlices: settings.zSlices,
		maxLights: settings.lightCount,
		maxLightsPerCluster: settings.maxLightsPerCluster,
	};
}

function requestSceneRender(
	demo: DemoState,
	reason: Parameters<Renderer["requestRender"]>[0]
): void {
	demo.scene.updateWorldMatrices();
	demo.renderer.requestRender(reason);
	updateMetrics(demo);
}

function updateMetrics(demo: DemoState): void {
	const options = readClusterOptions(demo.settings);
	const width = Math.max(1, demo.renderer.canvas.width);
	const height = Math.max(1, demo.renderer.canvas.height);
	const tilesX = Math.ceil(width / options.tileSizePx);
	const tilesY = Math.ceil(height / options.tileSizePx);
	const activeLights = getActiveLightCount(demo);
	const visibleMarkers = getVisibleMarkerCount(demo);
	backendMetric.textContent = demo.renderer.backend.type;
	modeMetric.textContent = demo.settings.clustered ?
		"clustered"
	:	`forward cap ${LEGACY_POINT_LIGHT_CAP}`;
	fpsMetric.textContent = demo.performance.fps > 0 ?
		demo.performance.fps.toFixed(1)
	:	"-";
	frameMetric.textContent = demo.performance.frameMs > 0 ?
		`${demo.performance.frameMs.toFixed(1)} ms`
	:	"-";
	lightMetric.textContent = `${activeLights} lights, ${visibleMarkers} markers`;
	gridMetric.textContent = demo.settings.clustered ?
		`${tilesX} x ${tilesY} x ${options.zSlices}`
	:	"disabled";
}

function samplePerformance(
	performanceState: DemoPerformance,
	now: number,
	deltaTimeMs: number
): void {
	if (deltaTimeMs <= 0) return;
	performanceState.frameCount++;
	performanceState.accumulatedMs += deltaTimeMs;
	if (now - performanceState.lastUpdateMs < 500) return;
	performanceState.frameMs =
		performanceState.accumulatedMs /
		Math.max(1, performanceState.frameCount);
	performanceState.fps = 1000 / Math.max(performanceState.frameMs, 0.001);
	performanceState.frameCount = 0;
	performanceState.accumulatedMs = 0;
	performanceState.lastUpdateMs = now;
}

function resolveLightIntensity(settings: DemoSettings, index: number): number {
	return settings.lightIntensity * (0.78 + (index % 9) * 0.045);
}

function getActiveLightCount(demo: DemoState): number {
	return clampInteger(demo.settings.lightCount, 1, demo.lights.length);
}

function getVisibleMarkerCount(demo: DemoState): number {
	if (!demo.settings.showMarkers) return 0;
	return Math.min(getActiveLightCount(demo), demo.settings.markerBudget);
}

function clampInteger(value: number, min: number, max: number): number {
	if (!Number.isFinite(value)) return min;
	return Math.max(min, Math.min(max, Math.floor(value)));
}

function getElapsedSeconds(demo: DemoState): number {
	return (performanceNow() - demo.startedAt) * 0.001;
}

function performanceNow(): number {
	return typeof performance !== "undefined" ? performance.now() : Date.now();
}

function setStatus(
	label: string,
	mode: "pending" | "ready" | "error"
): void {
	statusText.textContent = label;
	statusDot.classList.toggle("ready", mode === "ready");
	statusDot.classList.toggle("error", mode === "error");
}
