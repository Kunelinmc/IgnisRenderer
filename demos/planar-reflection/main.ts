import { Pane } from "tweakpane";

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
	UnlitMaterial,
	Vector3,
	WebGPUBackend,
} from "../../src/index";
import type { MeshInstance, RGB } from "../../src/index";

const MIRROR_Y = 0.03;
const OBJECT_COUNT = 13;

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

function asTweakpanePane(value: unknown): TweakpanePane {
	return value as TweakpanePane;
}

interface AnimatedObject {
	node: MeshInstance;
	baseY: number;
	phase: number;
	spinSpeed: number;
	floatAmplitude: number;
}

interface DemoSettings {
	reflections: boolean;
	reflectivity: number;
	animateObjects: boolean;
	animateLights: boolean;
	showLightMarkers: boolean;
}

interface DemoPerformance {
	frameCount: number;
	accumulatedMs: number;
	lastUpdateMs: number;
	fps: number;
	frameMs: number;
	backend: string;
	reflection: string;
	mirror: string;
	objects: string;
}

interface DemoPaneBindings {
	backend?: TweakpaneBinding;
	reflection?: TweakpaneBinding;
	fps?: TweakpaneBinding;
	frameMs?: TweakpaneBinding;
	mirror?: TweakpaneBinding;
	objects?: TweakpaneBinding;
}

interface DemoState {
	renderer: Renderer;
	scene: Scene;
	camera: OrbitCamera;
	mirrorMaterial: PBRMaterial;
	objects: AnimatedObject[];
	lights: Array<{ light: PointLight; marker: MeshInstance; phase: number }>;
	settings: DemoSettings;
	performance: DemoPerformance;
	paneBindings: DemoPaneBindings;
	startedAt: number;
}

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
	try {
		const platform = Platform.detect();
		if (!platform.hasWebGPU) {
			throw new Error("This demo requires WebGPU planar reflection support.");
		}

		const settings = createDefaultSettings();
		const scene = new Scene();
		scene.spatialIndexMode = "hybrid";

		const camera = createCamera();
		scene.add(camera);

		const mirrorMaterial = createMirrorMaterial(settings.reflectivity);
		const objects = populateScene(scene, mirrorMaterial);
		const lights = createLights(scene);
		const performance = createPerformanceState();
		const renderer = await createRenderer(canvas, camera);

		renderer.setScene(scene);
		renderer.features.enableLighting = true;
		renderer.features.enableShadows = false;
		renderer.features.enableEnvironment = false;
		renderer.features.enableReflection = settings.reflections;
		renderer.features.enableOIT = false;
		renderer.features.enableClusteredLighting = true;
		renderer.features.clusteredLightingOptions = {
			tileSizePx: 64,
			zSlices: 24,
			maxLights: 32,
			maxLightsPerCluster: 32,
		};
		renderer.on("frameend", ({ now, deltaTime }) => {
			samplePerformance(performance, now, deltaTime);
			updateMetrics(demo);
		});

		const demo: DemoState = {
			renderer,
			scene,
			camera,
			mirrorMaterial,
			objects,
			lights,
			settings,
			performance,
			paneBindings: {},
			startedAt: performanceNow(),
		};

		applyLightMarkerVisibility(demo);
		updateAnimation(demo, 0);

		await renderer.initialize();
		await renderer.warmup();

		scene.syncNodeToECS();
		scene.updateWorldMatrices();
		renderer.requestRender("unknown");
		await renderer.renderFrame(performanceNow());
		updateMetrics(demo);
		startAnimationLoop(demo);
		return demo;
	} catch (error) {
		console.error("Planar reflection demo failed to start.", error);
		throw error;
	}
}

function createDefaultSettings(): DemoSettings {
	return {
		reflections: true,
		reflectivity: 0.82,
		animateObjects: true,
		animateLights: true,
		showLightMarkers: true,
	};
}

function createPerformanceState(): DemoPerformance {
	return {
		frameCount: 0,
		accumulatedMs: 0,
		lastUpdateMs: performanceNow(),
		fps: 0,
		frameMs: 0,
		backend: "-",
		reflection: "-",
		mirror: "-",
		objects: "-",
	};
}

function createCamera(): OrbitCamera {
	const camera = new OrbitCamera(new Vector3(0, 1.25, -0.4), 16);
	camera.fov = 50;
	camera.near = 0.06;
	camera.far = 90;
	camera.theta = -0.72;
	camera.phi = 1.03;
	camera.minDistance = 6;
	camera.maxDistance = 34;
	camera.updatePosition();
	return camera;
}

function resetCamera(demo: DemoState): void {
	demo.camera.setTarget({ x: 0, y: 1.25, z: -0.4 });
	demo.camera.distance = 16;
	demo.camera.theta = -0.72;
	demo.camera.phi = 1.03;
	demo.camera.updatePosition();
	requestSceneRender(demo, "camera");
}

function createMirrorMaterial(reflectivity: number): PBRMaterial {
	const material = new PBRMaterial({
		name: "planar-reflection-mirror",
		albedo: { r: 36, g: 42, b: 46 },
		roughness: 0.08,
		metalness: 0.64,
		doubleSided: true,
	});
	material.reflectivity = reflectivity;
	material.mirrorPlane = {
		normal: { x: 0, y: 1, z: 0 },
		constant: -MIRROR_Y,
	};
	return material;
}

function populateScene(
	scene: Scene,
	mirrorMaterial: PBRMaterial
): AnimatedObject[] {
	const objects: AnimatedObject[] = [];
	const floorMaterial = new PBRMaterial({
		name: "matte-gallery-floor",
		albedo: { r: 82, g: 86, b: 78 },
		roughness: 0.76,
		metalness: 0,
		doubleSided: true,
	});
	const floor = MeshFactory.createPlane(
		{ x: 0, y: 0, z: 0 },
		30,
		22,
		floorMaterial
	);
	scene.add(floor);

	const mirror = MeshFactory.createPlane(
		{ x: 0, y: MIRROR_Y, z: -0.25 },
		12.2,
		7.4,
		mirrorMaterial
	);
	scene.add(mirror);

	const trimMaterial = new PBRMaterial({
		name: "mirror-brushed-trim",
		albedo: { r: 152, g: 153, b: 146 },
		roughness: 0.32,
		metalness: 0.76,
	});
	addMirrorTrim(scene, trimMaterial);
	addBackWall(scene);

	const palette: RGB[] = [
		{ r: 235, g: 79, b: 74 },
		{ r: 245, g: 169, b: 64 },
		{ r: 76, g: 188, b: 125 },
		{ r: 71, g: 150, b: 224 },
		{ r: 169, g: 101, b: 222 },
		{ r: 237, g: 120, b: 185 },
	];

	for (let index = 0; index < OBJECT_COUNT; index++) {
		const angle = (index / OBJECT_COUNT) * Math.PI * 2;
		const radius = 3.2 + (index % 3) * 1.2;
		const x = Math.cos(angle) * radius;
		const z = Math.sin(angle) * radius * 0.72 - 0.35;
		const color = palette[index % palette.length];
		const material = new PBRMaterial({
			name: `reflection-object-${index}`,
			albedo: color,
			roughness: 0.28 + (index % 4) * 0.08,
			metalness: index % 3 === 0 ? 0.48 : 0.08,
			clearcoat: index % 2 === 0 ? 0.35 : 0,
		});
		const node =
			index % 3 === 0 ?
				MeshFactory.createSphere(
					{ x, y: 0.82, z },
					0.48 + (index % 2) * 0.12,
					28,
					14,
					material
				)
			:	index % 3 === 1 ?
				MeshFactory.createBox(
					{ x, y: 0.62, z },
					0.8,
					0.8,
					1.2,
					material
				)
			:	MeshFactory.createTorus(
					{ x, y: 0.92, z },
					0.52,
					0.15,
					16,
					36,
					material
				);
		node.setRotationFromEuler(0.18 * index, angle * 0.35, 0.1 * index);
		scene.add(node);
		objects.push({
			node,
			baseY: node.position.y,
			phase: angle,
			spinSpeed: 0.22 + index * 0.017,
			floatAmplitude: 0.06 + (index % 4) * 0.018,
		});
	}

	scene.add(new AmbientLight({
		color: { r: 190, g: 202, b: 205 },
		intensity: 0.34,
	}));
	scene.add(new DirectionalLight({
		color: { r: 255, g: 248, b: 232 },
		intensity: 0.78,
		direction: { x: -0.32, y: -1, z: -0.18 },
	}));
	return objects;
}

function addMirrorTrim(scene: Scene, material: PBRMaterial): void {
	const width = 12.55;
	const depth = 7.75;
	const thickness = 0.12;
	const y = MIRROR_Y + 0.035;
	const z = -0.25;
	const rails = [
		MeshFactory.createBox({ x: 0, y, z: z - depth * 0.5 }, width, thickness, thickness, material),
		MeshFactory.createBox({ x: 0, y, z: z + depth * 0.5 }, width, thickness, thickness, material),
		MeshFactory.createBox({ x: -width * 0.5, y, z }, thickness, depth, thickness, material),
		MeshFactory.createBox({ x: width * 0.5, y, z }, thickness, depth, thickness, material),
	];
	for (const rail of rails) {
		scene.add(rail);
	}
}

function addBackWall(scene: Scene): void {
	const wallMaterial = new PBRMaterial({
		name: "gallery-back-wall",
		albedo: { r: 94, g: 101, b: 103 },
		roughness: 0.72,
		metalness: 0.02,
	});
	const wall = MeshFactory.createBox(
		{ x: 0, y: 2.1, z: 5.0 },
		22,
		0.28,
		4.2,
		wallMaterial
	);
	scene.add(wall);

	const plinthMaterial = new PBRMaterial({
		name: "gallery-plinths",
		albedo: { r: 132, g: 137, b: 130 },
		roughness: 0.62,
		metalness: 0.06,
	});
	for (let index = 0; index < 7; index++) {
		const x = (index - 3) * 2.1;
		const height = 0.28 + (index % 2) * 0.22;
		const plinth = MeshFactory.createBox(
			{ x, y: height * 0.5, z: 3.35 },
			1.1,
			1.0,
			height,
			plinthMaterial
		);
		scene.add(plinth);
	}
}

function createLights(
	scene: Scene
): Array<{ light: PointLight; marker: MeshInstance; phase: number }> {
	const colors: RGB[] = [
		{ r: 255, g: 120, b: 102 },
		{ r: 112, g: 190, b: 255 },
		{ r: 139, g: 235, b: 172 },
		{ r: 255, g: 215, b: 112 },
	];
	const result: Array<{ light: PointLight; marker: MeshInstance; phase: number }> = [];
	for (let index = 0; index < colors.length; index++) {
		const color = colors[index];
		const phase = (index / colors.length) * Math.PI * 2;
		const light = new PointLight({
			color,
			intensity: 14 + index * 2.1,
			position: { x: 0, y: 2.4, z: 0 },
			range: 10,
		});
		const marker = MeshFactory.createSphere(
			{ x: 0, y: 2.4, z: 0 },
			0.08,
			10,
			5,
			new UnlitMaterial({
				name: `planar-light-marker-${index}`,
				diffuse: color,
			}),
		);
		scene.add(light);
		scene.add(marker);
		result.push({ light, marker, phase });
	}
	return result;
}

async function createRenderer(
	target: HTMLCanvasElement,
	camera: OrbitCamera
): Promise<Renderer> {
	return new Renderer(
		target,
		new WebGPUBackend({
			enableDeferredLighting: true,
			enableEarlyZPrepass: true,
			enableOcclusionCulling: false,
		}),
		camera,
	);
}

async function createTweakpane(demo: DemoState): Promise<void> {
	try {
		const pane = asTweakpanePane(new Pane({ title: "Planar Reflection" }));
		const mirror = pane.addFolder({ title: "Mirror", expanded: true });

		mirror
			.addBinding(demo.settings as unknown as Record<string, unknown>, "reflections", {
				label: "Enabled",
			})
			.on("change", () => {
				applyReflectionSettings(demo);
				requestSceneRender(demo, "unknown");
			});

		mirror
			.addBinding(demo.settings as unknown as Record<string, unknown>, "reflectivity", {
				label: "Reflectivity",
				min: 0,
				max: 1,
				step: 0.01,
			})
			.on("change", () => {
				applyReflectionSettings(demo);
				requestSceneRender(demo, "unknown");
			});

		const motion = pane.addFolder({ title: "Motion", expanded: true });
		motion
			.addBinding(demo.settings as unknown as Record<string, unknown>, "animateObjects", {
				label: "Objects",
			})
			.on("change", () => requestSceneRender(demo, "unknown"));
		motion
			.addBinding(demo.settings as unknown as Record<string, unknown>, "animateLights", {
				label: "Lights",
			})
			.on("change", () => requestSceneRender(demo, "unknown"));
		motion
			.addBinding(demo.settings as unknown as Record<string, unknown>, "showLightMarkers", {
				label: "Markers",
			})
			.on("change", () => {
				applyLightMarkerVisibility(demo);
				requestSceneRender(demo, "unknown");
			});

		const view = pane.addFolder({ title: "View", expanded: false });
		view.addButton({ title: "Reset Camera" }).on("click", () => {
			resetCamera(demo);
			updateMetrics(demo);
		});

		const stats = pane.addFolder({ title: "Stats", expanded: true });
		demo.paneBindings.backend = stats.addBinding(
			demo.performance as unknown as Record<string, unknown>,
			"backend",
			{ label: "Backend", readonly: true },
		);
		demo.paneBindings.reflection = stats.addBinding(
			demo.performance as unknown as Record<string, unknown>,
			"reflection",
			{ label: "Reflection", readonly: true },
		);
		demo.paneBindings.fps = stats.addBinding(
			demo.performance as unknown as Record<string, unknown>,
			"fps",
			{
				label: "FPS",
				readonly: true,
				format: (value: number) => (value > 0 ? value.toFixed(1) : "-"),
			},
		);
		demo.paneBindings.frameMs = stats.addBinding(
			demo.performance as unknown as Record<string, unknown>,
			"frameMs",
			{
				label: "Frame Time",
				readonly: true,
				format: (value: number) => (value > 0 ? `${value.toFixed(1)} ms` : "-"),
			},
		);
		demo.paneBindings.mirror = stats.addBinding(
			demo.performance as unknown as Record<string, unknown>,
			"mirror",
			{ label: "Mirror", readonly: true },
		);
		demo.paneBindings.objects = stats.addBinding(
			demo.performance as unknown as Record<string, unknown>,
			"objects",
			{ label: "Objects", readonly: true },
		);
		pane.refresh?.();
		updateMetrics(demo);
	} catch (error) {
		console.error("Planar reflection controls failed to load.", error);
	}
}

function bindOrbitControls(demo: DemoState): void {
	const lookSensitivity = 0.0042;
	const zoomSensitivity = 0.018;
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
		demo.camera.rotate(dx * lookSensitivity, dy * lookSensitivity);
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
			demo.camera.zoom(event.deltaY * zoomSensitivity);
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
		if (demo.settings.animateObjects || demo.settings.animateLights) {
			updateAnimation(demo, getElapsedSeconds(demo));
			requestSceneRender(demo, "unknown");
		}
		requestAnimationFrame(animate);
	};
	requestAnimationFrame(animate);
}

function updateAnimation(demo: DemoState, timeSeconds: number): void {
	if (demo.settings.animateObjects) {
		for (const item of demo.objects) {
			const bob = Math.sin(timeSeconds * 1.35 + item.phase) *
				item.floatAmplitude;
			item.node.position.y = item.baseY + bob;
			item.node.setRotationFromEuler(
				0.18 + Math.sin(item.phase) * 0.12,
				item.phase + timeSeconds * item.spinSpeed,
				Math.cos(item.phase) * 0.18
			);
		}
	}
	if (demo.settings.animateLights) {
		for (let index = 0; index < demo.lights.length; index++) {
			const item = demo.lights[index];
			const phase = item.phase + timeSeconds * (0.36 + index * 0.055);
			const x = Math.cos(phase) * (4.2 + index * 0.35);
			const z = Math.sin(phase * 1.13) * 3.1 - 0.15;
			const y = 2.0 + Math.sin(timeSeconds * 0.9 + index) * 0.55;
			item.light.position.set(x, y, z);
			item.marker.position.set(x, y, z);
			item.light.updateLocalMatrix();
			item.marker.updateLocalMatrix();
		}
	}
}

function applyReflectionSettings(demo: DemoState): void {
	demo.renderer.features.enableReflection = demo.settings.reflections;
	demo.mirrorMaterial.reflectivity = demo.settings.reflectivity;
	demo.scene.invalidate("unknown");
	updateMetrics(demo);
}

function applyLightMarkerVisibility(demo: DemoState): void {
	for (const item of demo.lights) {
		const attached = demo.scene.contains(item.marker);
		if (demo.settings.showLightMarkers && !attached) {
			demo.scene.add(item.marker);
		} else if (!demo.settings.showLightMarkers && attached) {
			demo.scene.remove(item.marker);
		}
	}
	updateMetrics(demo);
}

function requestSceneRender(
	demo: DemoState,
	reason: Parameters<Renderer["requestRender"]>[0]
): void {
	demo.scene.updateWorldMatrices();
	demo.renderer.requestRender(reason);
	void demo.renderer.renderFrame(performanceNow()).catch(() => {});
	updateMetrics(demo);
}

function updateMetrics(demo: DemoState): void {
	demo.performance.backend = demo.renderer.backendProfile.id;
	demo.performance.reflection = demo.settings.reflections ? "planar" : "off";
	demo.performance.mirror =
		`${Math.round(demo.settings.reflectivity * 100)}% at y=${MIRROR_Y}`;
	demo.performance.objects =
		`${demo.objects.length} meshes, ${demo.lights.length} lights`;
	demo.paneBindings.backend?.refresh?.();
	demo.paneBindings.reflection?.refresh?.();
	demo.paneBindings.fps?.refresh?.();
	demo.paneBindings.frameMs?.refresh?.();
	demo.paneBindings.mirror?.refresh?.();
	demo.paneBindings.objects?.refresh?.();
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

function getElapsedSeconds(demo: DemoState): number {
	return (performanceNow() - demo.startedAt) * 0.001;
}

function performanceNow(): number {
	return globalThis.performance?.now?.() ?? Date.now();
}
