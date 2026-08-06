import { Pane } from "tweakpane";

import {
	MeshAsset,
	MeshFactory,
	MeshInstance,
	OrbitCamera,
	PBRMaterial,
	Platform,
	PointLight,
	ReflectionProbe,
	Renderer,
	Scene,
	ScreenSpaceGlobalIlluminationPass,
	Vector3,
	WebGPUBackend,
} from "../../src/index";
import type { SSGIOptions } from "../../src/index";

const canvas = getElement<HTMLCanvasElement>("canvas3d");
const status = getElement<HTMLParagraphElement>("status");

interface TweakpaneBinding {
	on: (
		eventName: "change" | "click",
		handler: (event: { value: unknown }) => void,
	) => TweakpaneBinding;
}

interface TweakpanePane {
	addBinding: (
		target: Record<string, unknown>,
		key: string,
		options?: Record<string, unknown>,
	) => TweakpaneBinding;
	addButton: (options: Record<string, unknown>) => TweakpaneBinding;
	addFolder: (options: Record<string, unknown>) => TweakpanePane;
	refresh?: () => void;
}

interface ColorSetting {
	r: number;
	g: number;
	b: number;
}

interface DemoSettings {
	enabled: boolean;
	downsample: 1 | 2 | 4;
	raysPerPixel: number;
	maxSteps: number;
	binarySearchSteps: number;
	maxDistance: number;
	thickness: number;
	normalBias: number;
	distanceFalloffExponent: number;
	edgeFade: number;
	intensity: number;
	historyWeight: number;
	disocclusionDepthThreshold: number;
	historyClamp: number;
	denoiseRadius: number;
	denoiseDepthPhi: number;
	denoiseNormalPhi: number;
	pointLightIntensity: number;
	pointLightRange: number;
	emitterIntensity: number;
	floorRoughness: number;
	receiverRoughness: number;
	wallRoughness: number;
	redWallColor: ColorSetting;
	blueWallColor: ColorSetting;
	emitterColor: ColorSetting;
	pointLightColor: ColorSetting;
}

interface SceneControls {
	floor: PBRMaterial;
	white: PBRMaterial;
	redWall: PBRMaterial;
	blueWall: PBRMaterial;
	emitter: PBRMaterial;
	pointLight: PointLight;
}

interface DemoState {
	renderer: Renderer;
	scene: Scene;
	camera: OrbitCamera;
	ssgi: ScreenSpaceGlobalIlluminationPass;
	settings: DemoSettings;
	sceneControls: SceneControls;
}

const DEFAULT_SETTINGS: DemoSettings = {
	enabled: true,
	downsample: 2,
	raysPerPixel: 4,
	maxSteps: 64,
	binarySearchSteps: 4,
	maxDistance: 5.5,
	thickness: 0.24,
	normalBias: 0.05,
	distanceFalloffExponent: 1.25,
	edgeFade: 0.08,
	intensity: 1.2,
	historyWeight: 0.92,
	disocclusionDepthThreshold: 0.02,
	historyClamp: 3,
	denoiseRadius: 3,
	denoiseDepthPhi: 24,
	denoiseNormalPhi: 16,
	pointLightIntensity: 14,
	pointLightRange: 7.5,
	emitterIntensity: 6,
	floorRoughness: 0.9,
	receiverRoughness: 0.82,
	wallRoughness: 0.88,
	redWallColor: { r: 210, g: 38, b: 30 },
	blueWallColor: { r: 32, g: 72, b: 220 },
	emitterColor: { r: 255, g: 240, b: 205 },
	pointLightColor: { r: 255, g: 244, b: 220 },
};

void startDemo();

async function startDemo(): Promise<void> {
	if (!Platform.detect().hasWebGPU) {
		status.textContent = "WebGPU is unavailable: this demo cannot run SSGI.";
		return;
	}

	const settings = createDefaultSettings();
	const scene = new Scene();
	const camera = createCamera();
	scene.add(camera);
	const sceneControls = populateScene(scene, settings);

	const backend = new WebGPUBackend({
		enableDeferredLighting: false,
		sampleCount: 1,
	});
	const renderer = new Renderer({
		backend,
		canvas,
		camera,
	});
	renderer.setScene(scene);
	renderer.features.enableShadows = false;

	const ssgi = new ScreenSpaceGlobalIlluminationPass({
		enabled: settings.enabled,
		options: getSSGIOptions(settings),
	});
	renderer.postProcess.registerPass(ssgi);

	await renderer.initialize();
	if (!supportsSSGI(renderer.getBackendDebugInfo().limits)) {
		status.textContent =
			"SSGI requires a WebGPU device with at least 5 color attachments and " +
			"40 color-attachment bytes per sample.";
		return;
	}

	await renderer.warmup({ includeCorePasses: true });
	bindOrbitControls(renderer, camera);
	createTweakpane({
		renderer,
		scene,
		camera,
		ssgi,
		settings,
		sceneControls,
	});

	window.addEventListener("resize", () => {
		renderer.resizeCanvas();
		renderer.requestRender("resize");
	});

	updateStatus(settings.enabled);
	renderer.renderLoop();
}

function createCamera(): OrbitCamera {
	const camera = new OrbitCamera(new Vector3(0, 1.65, -0.8), 7.4);
	resetCamera(camera);
	return camera;
}

function resetCamera(camera: OrbitCamera): void {
	camera.target.set(0, 1.65, -0.8);
	camera.distance = 7.4;
	camera.fov = 45;
	camera.near = 0.1;
	camera.far = 50;
	camera.theta = 0;
	camera.phi = 1.18;
	camera.minDistance = 3;
	camera.maxDistance = 12;
	camera.updatePosition();
}

function bindOrbitControls(renderer: Renderer, camera: OrbitCamera): void {
	const rotationSensitivity = 0.004;
	const zoomSensitivity = 0.015;
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
		const deltaX = event.clientX - lastX;
		const deltaY = event.clientY - lastY;
		lastX = event.clientX;
		lastY = event.clientY;
		if (deltaX === 0 && deltaY === 0) return;
		camera.rotate(deltaX * rotationSensitivity, deltaY * rotationSensitivity);
		renderer.requestRender("camera");
	});

	const releasePointer = (event: PointerEvent) => {
		if (activePointerId !== event.pointerId) return;
		activePointerId = null;
		if (canvas.hasPointerCapture(event.pointerId)) {
			canvas.releasePointerCapture(event.pointerId);
		}
	};
	canvas.addEventListener("pointerup", releasePointer);
	canvas.addEventListener("pointercancel", releasePointer);

	canvas.addEventListener(
		"wheel",
		(event) => {
			event.preventDefault();
			camera.zoom(event.deltaY * zoomSensitivity);
			renderer.requestRender("camera");
		},
		{ passive: false },
	);
}

function supportsSSGI(limits: Record<string, number> | undefined): boolean {
	return (
		(limits?.maxColorAttachments ?? 0) >= 5 &&
		(limits?.maxColorAttachmentBytesPerSample ?? 0) >= 40
	);
}

function populateScene(scene: Scene, settings: DemoSettings): SceneControls {
	const floor = new PBRMaterial({
		name: "receiver-floor",
		albedo: { r: 225, g: 225, b: 220 },
		roughness: settings.floorRoughness,
	});
	const white = new PBRMaterial({
		name: "receiver-white",
		albedo: { r: 235, g: 232, b: 224 },
		roughness: settings.receiverRoughness,
	});
	const redWall = new PBRMaterial({
		name: "red-bounce-wall",
		albedo: copyColor(settings.redWallColor),
		roughness: settings.wallRoughness,
	});
	const blueWall = new PBRMaterial({
		name: "blue-bounce-wall",
		albedo: copyColor(settings.blueWallColor),
		roughness: settings.wallRoughness,
	});
	const ceilingEmitter = new PBRMaterial({
		name: "ceiling-emitter",
		albedo: { r: 255, g: 248, b: 225 },
		emissive: copyColor(settings.emitterColor),
		emissiveIntensity: settings.emitterIntensity,
		roughness: 1,
	});
	const glass = new PBRMaterial({
		name: "glass-sphere",
		albedo: { r: 235, g: 248, b: 255 },
		roughness: 0.02,
		transmissionFactor: 0.82,
		ior: 1.6,
		clearcoat: 1,
		clearcoatRoughness: 0.02,
		thicknessFactor: 1.1,
		attenuationDistance: 2.5,
		attenuationColor: { r: 220, g: 244, b: 255 },
	});

	// A compact Cornell-box layout keeps the colored source surfaces on screen,
	// where SSGI can trace them and carry their radiance onto neutral receivers.
	scene.add(createRoomMesh(floor, white, redWall, blueWall, ceilingEmitter));

	scene.add(MeshFactory.createBox({ x: -0.9, y: 0.95, z: -0.9 }, 1.25, 1.4, 1.9, white));
	scene.add(MeshFactory.createBox({ x: 0.9, y: 0.58, z: 0.25 }, 1.45, 1.3, 1.16, white));
	scene.add(MeshFactory.createSphere({ x: 0.9, y: 1.71, z: 0.25 }, 0.55, 40, 24, glass));

	const reflectionProbe = new ReflectionProbe({
		shape: "box",
		halfExtents: { x: 3, y: 2.3, z: 3 },
		blendDistance: 0.2,
		parallaxMode: "box",
		source: "capturedScene",
		captureUpdateMode: "manual",
		captureResolution: { width: 128, height: 64 },
		captureFar: 10,
		includeEnvironment: false,
		includeMeshes: true,
		includeTransparent: false,
		includeParticles: false,
		includeShadows: false,
	});
	reflectionProbe.position.set(0, 2.3, 0);
	scene.add(reflectionProbe);
	reflectionProbe.requestCapture();

	const pointLight = new PointLight({
		color: copyColor(settings.pointLightColor),
		intensity: settings.pointLightIntensity,
		position: { x: 0, y: 4.05, z: -0.7 },
		range: settings.pointLightRange,
	});
	scene.add(pointLight);

	return {
		floor,
		white,
		redWall,
		blueWall,
		emitter: ceilingEmitter,
		pointLight,
	};
}

function createRoomMesh(
	floor: PBRMaterial,
	white: PBRMaterial,
	redWall: PBRMaterial,
	blueWall: PBRMaterial,
	ceilingEmitter: PBRMaterial,
): MeshInstance {
	const halfWidth = 3.05;
	const halfDepth = 3.05;
	const height = 4.7;
	const faces = [
		// Counter-clockwise from inside the room: +Y.
		createRoomFace(
			[
				{ x: -halfWidth, y: 0, z: -halfDepth },
				{ x: -halfWidth, y: 0, z: halfDepth },
				{ x: halfWidth, y: 0, z: halfDepth },
				{ x: halfWidth, y: 0, z: -halfDepth },
			],
			{ x: 0, y: 1, z: 0 },
			floor,
		),
		// Back wall faces the open front of the room: +Z.
		createRoomFace(
			[
				{ x: -halfWidth, y: 0, z: -halfDepth },
				{ x: halfWidth, y: 0, z: -halfDepth },
				{ x: halfWidth, y: height, z: -halfDepth },
				{ x: -halfWidth, y: height, z: -halfDepth },
			],
			{ x: 0, y: 0, z: 1 },
			white,
		),
		// Side walls point toward the room center: +X and -X.
		createRoomFace(
			[
				{ x: -halfWidth, y: 0, z: halfDepth },
				{ x: -halfWidth, y: 0, z: -halfDepth },
				{ x: -halfWidth, y: height, z: -halfDepth },
				{ x: -halfWidth, y: height, z: halfDepth },
			],
			{ x: 1, y: 0, z: 0 },
			redWall,
		),
		createRoomFace(
			[
				{ x: halfWidth, y: 0, z: -halfDepth },
				{ x: halfWidth, y: 0, z: halfDepth },
				{ x: halfWidth, y: height, z: halfDepth },
				{ x: halfWidth, y: height, z: -halfDepth },
			],
			{ x: -1, y: 0, z: 0 },
			blueWall,
		),
		// Ceiling and light panel both face downward: -Y.
		createRoomFace(
			[
				{ x: -halfWidth, y: height, z: -halfDepth },
				{ x: halfWidth, y: height, z: -halfDepth },
				{ x: halfWidth, y: height, z: halfDepth },
				{ x: -halfWidth, y: height, z: halfDepth },
			],
			{ x: 0, y: -1, z: 0 },
			white,
		),
		createRoomFace(
			[
				{ x: -0.9, y: 4.68, z: -1.3 },
				{ x: 0.9, y: 4.68, z: -1.3 },
				{ x: 0.9, y: 4.68, z: -0.1 },
				{ x: -0.9, y: 4.68, z: -0.1 },
			],
			{ x: 0, y: -1, z: 0 },
			ceilingEmitter,
		),
	];

	return new MeshInstance({
		name: "gi-room",
		mesh: MeshAsset.fromFaces(faces),
	});
}

function createRoomFace(
	positions: Array<{ x: number; y: number; z: number }>,
	normal: { x: number; y: number; z: number },
	material: PBRMaterial,
) {
	const uvs = [
		{ u: 0, v: 0 },
		{ u: 1, v: 0 },
		{ u: 1, v: 1 },
		{ u: 0, v: 1 },
	];
	return {
		vertices: positions.map((position, index) => ({
			...position,
			...uvs[index],
			normal: { ...normal },
		})),
		normal,
		material,
	};
}

function createTweakpane(demo: DemoState): void {
	const pane = asTweakpanePane(
		new Pane({
			title: "Screen-Space GI",
			expanded: true,
		}),
	);
	const target = demo.settings as unknown as Record<string, unknown>;
	const bindSSGI = (
		folder: TweakpanePane,
		key: keyof DemoSettings,
		options: Record<string, unknown>,
	) => {
		folder.addBinding(target, key, options).on("change", () => {
			applySSGISettings(demo);
		});
	};
	const bindScene = (
		folder: TweakpanePane,
		key: keyof DemoSettings,
		options: Record<string, unknown>,
	) => {
		folder.addBinding(target, key, options).on("change", () => {
			applySceneSettings(demo);
		});
	};

	const gi = pane.addFolder({ title: "GI", expanded: true });
	bindSSGI(gi, "enabled", { label: "Enabled" });
	bindSSGI(gi, "intensity", {
		label: "Intensity",
		min: 0,
		max: 3,
		step: 0.05,
	});
	bindSSGI(gi, "downsample", {
		label: "Resolution",
		options: {
			Full: 1,
			Half: 2,
			Quarter: 4,
		},
	});
	bindSSGI(gi, "raysPerPixel", {
		label: "Rays / pixel",
		min: 1,
		max: 4,
		step: 1,
	});
	bindSSGI(gi, "maxDistance", {
		label: "Bounce distance",
		min: 0.5,
		max: 12,
		step: 0.1,
	});

	const scene = pane.addFolder({ title: "Scene", expanded: false });
	bindScene(scene, "pointLightIntensity", {
		label: "Light intensity",
		min: 0,
		max: 30,
		step: 0.5,
	});
	bindScene(scene, "emitterIntensity", {
		label: "Emitter",
		min: 0,
		max: 16,
		step: 0.25,
	});
	const colorOptions = {
		view: "color",
		color: { type: "int" },
	};
	bindScene(scene, "redWallColor", {
		...colorOptions,
		label: "Left wall",
	});
	bindScene(scene, "blueWallColor", {
		...colorOptions,
		label: "Right wall",
	});

	const actions = pane.addFolder({ title: "Actions", expanded: true });
	actions.addButton({ title: "Reset camera" }).on("click", () => {
		resetCamera(demo.camera);
		demo.renderer.requestRender("camera");
	});
	actions.addButton({ title: "Reset all settings" }).on("click", () => {
		Object.assign(demo.settings, createDefaultSettings());
		applySSGISettings(demo);
		applySceneSettings(demo);
		resetCamera(demo.camera);
		pane.refresh?.();
	});
}

function applySSGISettings(demo: DemoState): void {
	demo.ssgi.setEnabled(demo.settings.enabled);
	demo.ssgi.setOptions(getSSGIOptions(demo.settings));
	updateStatus(demo.settings.enabled);
	demo.renderer.requestRender("unknown");
}

function applySceneSettings(demo: DemoState): void {
	const controls = demo.sceneControls;
	controls.floor.roughness = demo.settings.floorRoughness;
	controls.white.roughness = demo.settings.receiverRoughness;
	controls.redWall.roughness = demo.settings.wallRoughness;
	controls.blueWall.roughness = demo.settings.wallRoughness;
	controls.redWall.albedo = copyColor(demo.settings.redWallColor);
	controls.blueWall.albedo = copyColor(demo.settings.blueWallColor);
	controls.emitter.emissive = copyColor(demo.settings.emitterColor);
	controls.emitter.emissiveIntensity = demo.settings.emitterIntensity;
	controls.pointLight.color = copyColor(demo.settings.pointLightColor);
	controls.pointLight.intensity = demo.settings.pointLightIntensity;
	controls.pointLight.range = demo.settings.pointLightRange;
	demo.scene.invalidate("unknown");
	demo.renderer.requestRender("unknown");
}

function getSSGIOptions(settings: DemoSettings): SSGIOptions {
	return {
		downsample: settings.downsample,
		raysPerPixel: settings.raysPerPixel,
		maxSteps: settings.maxSteps,
		binarySearchSteps: settings.binarySearchSteps,
		maxDistance: settings.maxDistance,
		thickness: settings.thickness,
		normalBias: settings.normalBias,
		distanceFalloffExponent: settings.distanceFalloffExponent,
		edgeFade: settings.edgeFade,
		intensity: settings.intensity,
		historyWeight: settings.historyWeight,
		disocclusionDepthThreshold: settings.disocclusionDepthThreshold,
		historyClamp: settings.historyClamp,
		denoiseRadius: settings.denoiseRadius,
		denoiseDepthPhi: settings.denoiseDepthPhi,
		denoiseNormalPhi: settings.denoiseNormalPhi,
	};
}

function createDefaultSettings(): DemoSettings {
	return {
		...DEFAULT_SETTINGS,
		redWallColor: copyColor(DEFAULT_SETTINGS.redWallColor),
		blueWallColor: copyColor(DEFAULT_SETTINGS.blueWallColor),
		emitterColor: copyColor(DEFAULT_SETTINGS.emitterColor),
		pointLightColor: copyColor(DEFAULT_SETTINGS.pointLightColor),
	};
}

function copyColor(color: ColorSetting): ColorSetting {
	return { r: color.r, g: color.g, b: color.b };
}

function updateStatus(enabled: boolean): void {
	status.textContent = enabled
		? "SSGI enabled — drag to orbit and use the panel to tune the effect."
		: "SSGI disabled — only direct and emissive lighting remains.";
}

function asTweakpanePane(value: unknown): TweakpanePane {
	return value as TweakpanePane;
}

function getElement<T extends HTMLElement>(id: string): T {
	const element = document.getElementById(id);
	if (!element) {
		throw new Error(`Missing element #${id}.`);
	}
	return element as T;
}
