import { Pane } from "tweakpane";
import { MeshAsset, MeshInstance } from "../../src/meshes";
import { OrbitCamera } from "../../src/cameras/OrbitCamera";
import { Platform } from "../../src/foundation/Platform";
import { Renderer } from "../../src/rendering/Renderer";
import { Scene } from "../../src/core/Scene";
import { ShaderMaterial } from "../../src/materials";
import { TextureLoader } from "../../src/loaders/TextureLoader";
import { Vector3 } from "../../src/maths/Vector3";
import { WebGLBackend } from "../../src/backends/webgl/WebGLBackend";
import { WebGPUBackend } from "../../src/backends/webgpu/WebGPUBackend";
import type { Texture } from "../../src/core/Texture";

import {
	INTERIOR_MAPPING_FRAGMENT_GLSL,
	INTERIOR_MAPPING_FRAGMENT_WGSL,
	INTERIOR_MAPPING_VERTEX_GLSL,
	INTERIOR_MAPPING_VERTEX_WGSL,
} from "./shaders";

interface TweakpaneBinding {
	refresh?: () => void;
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

function asTweakpanePane(value: unknown): TweakpanePane {
	return value as TweakpanePane;
}

interface DemoSettings {
	roomTilingX: number;
	roomTilingY: number;
	roomDepth: number;
	roomAspect: number;
}

interface DemoPerformance {
	frameCount: number;
	accumulatedMs: number;
	lastUpdateMs: number;
	fps: number;
	frameMs: number;
	backend: string;
	viewVector: string;
}

interface DemoPaneBindings {
	fps?: TweakpaneBinding;
	frameMs?: TweakpaneBinding;
	backend?: TweakpaneBinding;
	viewVector?: TweakpaneBinding;
}

interface DemoState {
	renderer: Renderer;
	scene: Scene;
	camera: OrbitCamera;
	material: ShaderMaterial;
	settings: DemoSettings;
	performance: DemoPerformance;
	paneBindings: DemoPaneBindings;
	startedAt: number;
}

type DemoBackendPreference = "auto" | "webgpu" | "webgl";

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

		const settings = createDefaultSettings();
		const scene = new Scene();
		scene.spatialIndexMode = "hybrid";

		const camera = createCamera();
		scene.add(camera);

		const roomTexture = await loadRoomTexture();
		const material = createInteriorMaterial(settings, roomTexture);
		const w2 = 2.25;
		const h2 = 2.25;
		const d2 = 2.25;
		const faces = [
			// Front Face (z = +2.25)
			{
				vertices: [
					{
						x: -w2,
						y: -h2,
						z: d2,
						u: 0,
						v: 1,
						normal: { x: 0, y: 0, z: 1 },
						tangent: { x: 1, y: 0, z: 0, w: -1 },
					},
					{
						x: w2,
						y: -h2,
						z: d2,
						u: 1,
						v: 1,
						normal: { x: 0, y: 0, z: 1 },
						tangent: { x: 1, y: 0, z: 0, w: -1 },
					},
					{
						x: w2,
						y: h2,
						z: d2,
						u: 1,
						v: 0,
						normal: { x: 0, y: 0, z: 1 },
						tangent: { x: 1, y: 0, z: 0, w: -1 },
					},
					{
						x: -w2,
						y: h2,
						z: d2,
						u: 0,
						v: 0,
						normal: { x: 0, y: 0, z: 1 },
						tangent: { x: 1, y: 0, z: 0, w: -1 },
					},
				],
				normal: { x: 0, y: 0, z: 1 },
				material,
			},
			// Back Face (z = -2.25)
			{
				vertices: [
					{
						x: w2,
						y: -h2,
						z: -d2,
						u: 0,
						v: 1,
						normal: { x: 0, y: 0, z: -1 },
						tangent: { x: -1, y: 0, z: 0, w: -1 },
					},
					{
						x: -w2,
						y: -h2,
						z: -d2,
						u: 1,
						v: 1,
						normal: { x: 0, y: 0, z: -1 },
						tangent: { x: -1, y: 0, z: 0, w: -1 },
					},
					{
						x: -w2,
						y: h2,
						z: -d2,
						u: 1,
						v: 0,
						normal: { x: 0, y: 0, z: -1 },
						tangent: { x: -1, y: 0, z: 0, w: -1 },
					},
					{
						x: w2,
						y: h2,
						z: -d2,
						u: 0,
						v: 0,
						normal: { x: 0, y: 0, z: -1 },
						tangent: { x: -1, y: 0, z: 0, w: -1 },
					},
				],
				normal: { x: 0, y: 0, z: -1 },
				material,
			},
			// Left Face (x = -2.25)
			{
				vertices: [
					{
						x: -w2,
						y: -h2,
						z: -d2,
						u: 0,
						v: 1,
						normal: { x: -1, y: 0, z: 0 },
						tangent: { x: 0, y: 0, z: 1, w: -1 },
					},
					{
						x: -w2,
						y: -h2,
						z: d2,
						u: 1,
						v: 1,
						normal: { x: -1, y: 0, z: 0 },
						tangent: { x: 0, y: 0, z: 1, w: -1 },
					},
					{
						x: -w2,
						y: h2,
						z: d2,
						u: 1,
						v: 0,
						normal: { x: -1, y: 0, z: 0 },
						tangent: { x: 0, y: 0, z: 1, w: -1 },
					},
					{
						x: -w2,
						y: h2,
						z: -d2,
						u: 0,
						v: 0,
						normal: { x: -1, y: 0, z: 0 },
						tangent: { x: 0, y: 0, z: 1, w: -1 },
					},
				],
				normal: { x: -1, y: 0, z: 0 },
				material,
			},
			// Right Face (x = 2.25)
			{
				vertices: [
					{
						x: w2,
						y: -h2,
						z: d2,
						u: 0,
						v: 1,
						normal: { x: 1, y: 0, z: 0 },
						tangent: { x: 0, y: 0, z: -1, w: -1 },
					},
					{
						x: w2,
						y: -h2,
						z: -d2,
						u: 1,
						v: 1,
						normal: { x: 1, y: 0, z: 0 },
						tangent: { x: 0, y: 0, z: -1, w: -1 },
					},
					{
						x: w2,
						y: h2,
						z: -d2,
						u: 1,
						v: 0,
						normal: { x: 1, y: 0, z: 0 },
						tangent: { x: 0, y: 0, z: -1, w: -1 },
					},
					{
						x: w2,
						y: h2,
						z: d2,
						u: 0,
						v: 0,
						normal: { x: 1, y: 0, z: 0 },
						tangent: { x: 0, y: 0, z: -1, w: -1 },
					},
				],
				normal: { x: 1, y: 0, z: 0 },
				material,
			},
		];
		const mesh = MeshAsset.fromFaces(faces);
		const wallBox = new MeshInstance({ mesh, position: new Vector3(0, 0, 0) });
		wallBox.updateLocalMatrix();
		scene.add(wallBox);

		const backend = createBackend(platform, resolveBackendPreference());

		const renderer = new Renderer({
			canvas,
			backend,
			camera,
		});

		renderer.setScene(scene);
		renderer.features.enableLighting = false;
		renderer.features.enableShadows = false;
		renderer.features.enableEnvironment = false;
		renderer.features.enableOIT = false;

		const performanceState = createPerformanceState();
		const demo: DemoState = {
			renderer,
			scene,
			camera,
			material,
			settings,
			performance: performanceState,
			paneBindings: {},
			startedAt: performanceNow(),
		};

		renderer.on("tick", ({ now, deltaTime }) => {
			samplePerformance(performanceState, now, deltaTime);
			updateMetrics(demo);
		});

		await renderer.initialize();
		await renderer.warmup();

		scene.syncNodeToECS();
		scene.updateWorldMatrices();
		renderer.requestRender("unknown");
		renderer.renderLoop();

		return demo;
	} catch (error) {
		console.error("Interior mapping demo failed to start.", error);
		throw error;
	}
}

function createDefaultSettings(): DemoSettings {
	return {
		roomTilingX: 3,
		roomTilingY: 3,
		roomDepth: 1,
		roomAspect: 1,
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
		viewVector: "-",
	};
}

function resolveBackendPreference(): DemoBackendPreference {
	const value = new URLSearchParams(window.location.search)
		.get("backend")
		?.trim()
		.toLowerCase();
	if (value === "webgpu" || value === "webgl" || value === "auto") {
		return value;
	}
	if (value) {
		console.warn(
			`Unknown backend search parameter "${value}"; using automatic backend selection.`,
		);
	}
	return "auto";
}

function createBackend(
	platform: ReturnType<typeof Platform.detect>,
	preference: DemoBackendPreference,
): WebGPUBackend | WebGLBackend {
	if (preference === "webgpu") {
		if (!platform.hasWebGPU) {
			throw new Error(
				"The requested WebGPU backend is not supported by this browser.",
			);
		}
		return createWebGPUBackend();
	}
	if (preference === "webgl") {
		if (!platform.hasWebGL2) {
			throw new Error(
				"The requested WebGL backend is not supported by this browser.",
			);
		}
		return createWebGLBackend();
	}
	if (platform.hasWebGPU) {
		return createWebGPUBackend();
	}
	if (platform.hasWebGL2) {
		return createWebGLBackend();
	}
	throw new Error("This browser does not support WebGPU or WebGL.");
}

function createWebGPUBackend(): WebGPUBackend {
	return new WebGPUBackend({
		enableDeferredLighting: false,
		enableEarlyZPrepass: false,
		enableOcclusionCulling: false,
	});
}

function createWebGLBackend(): WebGLBackend {
	return new WebGLBackend({
		enableEarlyZPrepass: false,
	});
}

async function loadRoomTexture(): Promise<Texture> {
	const texture = await new TextureLoader().load(
		"../../assets/textures/interior-mapping-texture.png",
	);
	texture.label = "InteriorMappingRoomAtlas";
	texture.wrapS = "Clamp";
	texture.wrapT = "Clamp";
	texture.minFilter = "Linear";
	texture.magFilter = "Linear";
	return texture;
}

function createCamera(): OrbitCamera {
	const camera = new OrbitCamera(new Vector3(0, 0, 0), 10);
	camera.fov = 45;
	camera.near = 0.1;
	camera.far = 100;
	camera.theta = 0;
	camera.phi = Math.PI / 2;
	camera.minDistance = 2;
	camera.maxDistance = 20;
	camera.zoomSensitivity = 0.01;
	camera.updatePosition();
	return camera;
}

function createInteriorMaterial(settings: DemoSettings, roomTexture: Texture): ShaderMaterial {
	return new ShaderMaterial({
		name: "interior-mapping-shader",
		doubleSided: true,
		textureBindings: [
			{
				name: "roomAtlas",
				texture: roomTexture,
				slot: 0,
				uvSet: 0,
			},
		],
		uniformBindings: [
			{
				name: "roomTiling",
				type: "vec2f",
				value: [settings.roomTilingX, settings.roomTilingY],
				wgslField: "roomTiling",
			},
			{
				name: "roomDepth",
				type: "f32",
				value: settings.roomDepth,
				wgslField: "roomDepth",
			},
			{
				name: "roomAspect",
				type: "f32",
				value: settings.roomAspect,
				wgslField: "roomAspect",
			},
		],
		chunks: [
			{
				language: "wgsl",
				stage: "vertex",
				code: INTERIOR_MAPPING_VERTEX_WGSL,
			},
			{
				language: "wgsl",
				stage: "fragment",
				mode: "single",
				code: INTERIOR_MAPPING_FRAGMENT_WGSL,
			},
			{
				backend: "webgl",
				language: "glsl",
				stage: "vertex",
				code: INTERIOR_MAPPING_VERTEX_GLSL,
			},
			{
				backend: "webgl",
				language: "glsl",
				stage: "fragment",
				mode: "single",
				code: INTERIOR_MAPPING_FRAGMENT_GLSL,
			},
		],
	});
}

async function createTweakpane(demo: DemoState): Promise<void> {
	try {
		const pane = asTweakpanePane(new Pane({ title: "Interior Mapping" }));

		const room = pane.addFolder({ title: "Room", expanded: true });
		room.addBinding(demo.settings as unknown as Record<string, unknown>, "roomTilingX", {
			label: "Tiles X",
			min: 1,
			max: 10,
			step: 1,
		}).on("change", () => applyUniforms(demo));
		room.addBinding(demo.settings as unknown as Record<string, unknown>, "roomTilingY", {
			label: "Tiles Y",
			min: 1,
			max: 10,
			step: 1,
		}).on("change", () => applyUniforms(demo));
		room.addBinding(demo.settings as unknown as Record<string, unknown>, "roomDepth", {
			label: "Depth",
			min: 0.1,
			max: 3,
			step: 0.05,
		}).on("change", () => applyUniforms(demo));
		room.addBinding(demo.settings as unknown as Record<string, unknown>, "roomAspect", {
			label: "Aspect",
			min: 0.4,
			max: 3,
			step: 0.05,
		}).on("change", () => applyUniforms(demo));

		const stats = pane.addFolder({ title: "Stats", expanded: true });
		demo.paneBindings.backend = stats.addBinding(
			demo.performance as unknown as Record<string, unknown>,
			"backend",
			{ label: "Backend", readonly: true },
		);
		demo.paneBindings.fps = stats.addBinding(
			demo.performance as unknown as Record<string, unknown>,
			"fps",
			{
				label: "FPS",
				readonly: true,
				format: (value: number) => value.toFixed(1),
			},
		);
		demo.paneBindings.frameMs = stats.addBinding(
			demo.performance as unknown as Record<string, unknown>,
			"frameMs",
			{
				label: "Frame Time",
				readonly: true,
				format: (value: number) => `${value.toFixed(1)} ms`,
			},
		);
		demo.paneBindings.viewVector = stats.addBinding(
			demo.performance as unknown as Record<string, unknown>,
			"viewVector",
			{ label: "TBN View", readonly: true },
		);

		pane.refresh?.();
		updateMetrics(demo);
	} catch (error) {
		console.error("Interior mapping controls failed to load.", error);
	}
}

function bindOrbitControls(demo: DemoState): void {
	const activePointers = new Map<number, PointerEvent>();
	let lastX = 0;
	let lastY = 0;
	let lastPinchDistance = 0;

	canvas.addEventListener("pointerdown", (event) => {
		activePointers.set(event.pointerId, event);
		canvas.setPointerCapture(event.pointerId);

		if (activePointers.size === 1) {
			lastX = event.clientX;
			lastY = event.clientY;
		} else if (activePointers.size === 2) {
			const pts = Array.from(activePointers.values());
			lastPinchDistance = Math.hypot(
				pts[0].clientX - pts[1].clientX,
				pts[0].clientY - pts[1].clientY,
			);
		}
	});

	canvas.addEventListener("pointermove", (event) => {
		if (!activePointers.has(event.pointerId)) return;
		activePointers.set(event.pointerId, event);

		if (activePointers.size === 1) {
			const dx = event.clientX - lastX;
			const dy = event.clientY - lastY;
			lastX = event.clientX;
			lastY = event.clientY;
			if (dx === 0 && dy === 0) return;
			demo.camera.rotate(dx, dy);
			requestSceneRender(demo);
		} else if (activePointers.size === 2) {
			const pts = Array.from(activePointers.values());
			const currentDistance = Math.hypot(
				pts[0].clientX - pts[1].clientX,
				pts[0].clientY - pts[1].clientY,
			);
			const diff = lastPinchDistance - currentDistance;
			if (Math.abs(diff) > 0.5) {
				demo.camera.zoom(diff * 1.5);
				requestSceneRender(demo);
				lastPinchDistance = currentDistance;
			}
		}
	});

	const clearPointer = (event: PointerEvent) => {
		activePointers.delete(event.pointerId);
		try {
			if (canvas.hasPointerCapture(event.pointerId)) {
				canvas.releasePointerCapture(event.pointerId);
			}
		} catch (e) {
			// ignore
		}

		if (activePointers.size === 1) {
			const remaining = Array.from(activePointers.values())[0];
			lastX = remaining.clientX;
			lastY = remaining.clientY;
		}
	};

	canvas.addEventListener("pointerup", clearPointer);
	canvas.addEventListener("pointercancel", clearPointer);
	canvas.addEventListener("pointerleave", clearPointer);

	canvas.addEventListener(
		"wheel",
		(event) => {
			event.preventDefault();
			demo.camera.zoom(event.deltaY);
			requestSceneRender(demo);
		},
		{ passive: false },
	);

	window.addEventListener("resize", () => {
		demo.renderer.resizeCanvas();
		requestSceneRender(demo);
	});
}

function applyUniforms(demo: DemoState): void {
	demo.settings.roomTilingX = clampInteger(demo.settings.roomTilingX, 1, 10);
	demo.settings.roomTilingY = clampInteger(demo.settings.roomTilingY, 1, 10);
	demo.settings.roomDepth = clamp(demo.settings.roomDepth, 0.1, 3);
	demo.settings.roomAspect = clamp(demo.settings.roomAspect, 0.4, 3);
	demo.material.setUniform("roomTiling", [demo.settings.roomTilingX, demo.settings.roomTilingY]);
	demo.material.setUniform("roomDepth", demo.settings.roomDepth);
	demo.material.setUniform("roomAspect", demo.settings.roomAspect);
	requestSceneRender(demo);
}

function requestSceneRender(demo: DemoState): void {
	demo.scene.updateWorldMatrices();
	demo.renderer.requestRender("camera");
}

function updateMetrics(demo: DemoState): void {
	demo.performance.backend = demo.renderer.backendProfile.id;
	const cameraPos = demo.camera.position;
	const viewDirWorld = new Vector3().copy(cameraPos).scale(-1).normalize();
	demo.performance.viewVector = `[${viewDirWorld.x.toFixed(2)}, ${viewDirWorld.y.toFixed(2)}, ${viewDirWorld.z.toFixed(2)}]`;
	demo.paneBindings.backend?.refresh?.();
	demo.paneBindings.fps?.refresh?.();
	demo.paneBindings.frameMs?.refresh?.();
	demo.paneBindings.viewVector?.refresh?.();
}

function samplePerformance(
	performanceState: DemoState["performance"],
	now: number,
	deltaTimeMs: number,
): void {
	if (deltaTimeMs <= 0) return;
	performanceState.frameCount++;
	performanceState.accumulatedMs += deltaTimeMs;
	if (now - performanceState.lastUpdateMs < 500) return;
	performanceState.frameMs =
		performanceState.accumulatedMs / Math.max(1, performanceState.frameCount);
	performanceState.fps = 1000 / Math.max(performanceState.frameMs, 0.001);
	performanceState.frameCount = 0;
	performanceState.accumulatedMs = 0;
	performanceState.lastUpdateMs = now;
}

function clamp(value: number, min: number, max: number): number {
	return Math.min(max, Math.max(min, value));
}

function clampInteger(value: number, min: number, max: number): number {
	return Math.trunc(clamp(value, min, max));
}

function performanceNow(): number {
	return globalThis.performance?.now?.() ?? Date.now();
}
