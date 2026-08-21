import { Pane } from "tweakpane";

import {
	AmbientLight,
	DirectionalLight,
	MeshFactory,
	OrbitCamera,
	PBRMaterial,
	Platform,
	Renderer,
	Scene,
	Vector3,
	WebGLBackend,
	WebGPUBackend,
	PhysicsSystem,
	RapierWorkerPhysicsAdapter,
	CameraShakePlugin,
	FastApproximateAntiAliasingPass,
	UnlitMaterial,
} from "../../src/index";
import type { MeshInstance } from "../../src/index";

// ----------------------------------------------------
// Core Types & State
// ----------------------------------------------------
type DemoBackendPreference = "auto" | "webgpu" | "webgl";

interface DemoSettings {
	gravity: number;
	restitution: number;
	friction: number;
	backend: string;
	maxObjects: number;
	fxaa: boolean;
}

interface DemoDiagnostics {
	fps: string;
	ms: string;
	active: string;
	sleeping: string;
	activeBackend: string;
}

interface DemoState {
	renderer: Renderer;
	scene: Scene;
	camera: OrbitCamera;
	physics: PhysicsSystem;
	spawnedItems: Array<{ node: MeshInstance; shapeKind: string }>;
	lastTime: number;
	frameCount: number;
	lastFpsUpdate: number;
	isRendering: boolean;
	settings: DemoSettings;
	diagnostics: DemoDiagnostics;
	paneBindings: {
		fps?: any;
		ms?: any;
		active?: any;
		sleeping?: any;
		activeBackend?: any;
	};
	cameraShake: CameraShakePlugin;
}

const canvas = document.getElementById("canvas3d") as HTMLCanvasElement;

// ----------------------------------------------------
// Dynamic Backend Helper Functions
// ----------------------------------------------------
function resolveBackendPreference(): DemoBackendPreference {
	const val = new URLSearchParams(window.location.search).get("backend")?.trim().toLowerCase();
	if (val === "webgpu" || val === "webgl" || val === "auto") {
		return val as DemoBackendPreference;
	}
	return "auto";
}

function createBackend(
	platform: ReturnType<typeof Platform.detect>,
	preference: DemoBackendPreference,
): WebGPUBackend | WebGLBackend {
	if (preference === "webgpu") {
		if (!platform.hasWebGPU) {
			throw new Error("WebGPU is not supported by this browser.");
		}
		return new WebGPUBackend({
			enableDeferredLighting: true,
			enableEarlyZPrepass: true,
			enableOcclusionCulling: true,
		});
	}
	if (preference === "webgl") {
		if (!platform.hasWebGL2) {
			throw new Error("WebGL2 is not supported by this browser.");
		}
		return new WebGLBackend({ enableEarlyZPrepass: true });
	}
	if (platform.hasWebGPU) {
		return new WebGPUBackend({
			enableDeferredLighting: true,
			enableEarlyZPrepass: true,
			enableOcclusionCulling: true,
		});
	}
	if (platform.hasWebGL2) {
		return new WebGLBackend({ enableEarlyZPrepass: true });
	}
	throw new Error("This browser does not support WebGPU or WebGL.");
}

// ----------------------------------------------------
// Demo Boot Sequence
// ----------------------------------------------------
async function bootDemo(): Promise<DemoState> {
	const platform = Platform.detect();
	const preference = resolveBackendPreference();
	const backend = createBackend(platform, preference);

	// Setup Camera
	const camera = new OrbitCamera(new Vector3(0, 0, 0), 20);
	camera.near = 0.1;
	camera.far = 100;
	camera.theta = -0.78; // Angle around Y
	camera.phi = 1.05; // Elevation angle
	camera.minDistance = 6;
	camera.maxDistance = 35;
	camera.updatePosition();

	// Setup Scene
	const scene = new Scene();
	scene.spatialIndexMode = "hybrid";
	scene.add(camera);

	// Add Lights
	const ambientLight = new AmbientLight({
		color: { r: 50, g: 60, b: 85 },
		intensity: 0.7,
	});
	scene.add(ambientLight);

	const dirLight1 = new DirectionalLight({
		intensity: 4.5,
		direction: { x: 0.6, y: -1.0, z: -0.4 },
	});
	scene.add(dirLight1);

	// Setup CSM Shadow Map on dirLight1
	const shadowMap = scene.shadows.createCascaded({
		size: 2048,
		lambda: 0.65,
		maxDistance: 60,
		blendRatio: 0.1,
		stabilize: true,
		sampling: {
			filterMode: "pcf",
			radius: 0.1,
			samples: 8,
			searchSamples: 4,
			strength: 1,
		},
		bias: {
			constant: 0.0005,
			slope: 0.001,
			normal: 0.005,
			normalMin: 0.005,
			texel: 1,
			max: 0.01,
		},
	});
	scene.shadows.bind(dirLight1, shadowMap);

	const dirLight2 = new DirectionalLight({
		intensity: 1.5,
		direction: { x: -0.6, y: -0.5, z: 0.4 },
	});
	scene.add(dirLight2);

	// Initialize Renderer
	const renderer = new Renderer(canvas, backend, camera);
	renderer.setScene(scene);
	renderer.features.enableLighting = true;
	renderer.features.enableShadows = true;
	renderer.features.enableEnvironment = false;
	renderer.features.enableReflection = false;
	renderer.features.enableOIT = false;

	await renderer.initialize();
	await renderer.warmup();

	// Register FXAA pass
	renderer.postProcess.registerPass(new FastApproximateAntiAliasingPass({ enabled: true }));

	// Initialize Physics
	const physics = new PhysicsSystem({
		adapter: new RapierWorkerPhysicsAdapter({
			strict: true,
		}),
	});
	await physics.init();

	const initialGravity = -9.81;
	physics.createWorld({
		worldId: "main",
		gravity: { x: 0, y: initialGravity, z: 0 },
		mode: "fixed",
		fixedDeltaSeconds: 1 / 60,
		maxSubsteps: 4,
		maxDeltaSeconds: 1 / 10,
	});

	// Bind physics resolver to ECS nodes
	physics.setEntityNodeResolver((entityId) => {
		return scene.ecs.getNodeByEntity(entityId);
	});
	physics.bindSceneSpatial(scene);

	const settings: DemoSettings = {
		gravity: initialGravity,
		restitution: 0.6,
		friction: 0.3,
		backend: preference,
		maxObjects: 150,
		fxaa: true,
	};

	const diagnostics: DemoDiagnostics = {
		fps: "-",
		ms: "- ms",
		active: "-",
		sleeping: "-",
		activeBackend: renderer.backendProfile.id,
	};

	const state: DemoState = {
		renderer,
		scene,
		camera,
		physics,
		spawnedItems: [],
		lastTime: performance.now(),
		frameCount: 0,
		lastFpsUpdate: performance.now(),
		isRendering: false,
		settings,
		diagnostics,
		paneBindings: {},
		cameraShake: new CameraShakePlugin().attach(renderer),
	};

	// Create static boundary container (Ground and 4 walls)
	createContainer(state);

	// Spawn a few initial objects to show off immediately
	for (let i = 0; i < 8; i++) {
		const types = ["box", "sphere", "cylinder"] as const;
		spawnObject(state, types[i % 3]);
	}

	return state;
}

// ----------------------------------------------------
// Physics Container Creation
// ----------------------------------------------------
function createContainer(state: DemoState): void {
	const groundMat = new UnlitMaterial({ diffuse: { r: 35, g: 39, b: 50 } });

	const wallMat = new UnlitMaterial({ diffuse: { r: 48, g: 52, b: 67 } });

	const size = 16;
	const height = 5;
	const thickness = 0.5;

	// Ground Plane
	const ground = MeshFactory.createBox(
		{ x: 0, y: -thickness / 2, z: 0 },
		size,
		size,
		thickness,
		groundMat,
	);
	state.scene.add(ground);
	const groundBody = state.physics.attachBody(ground, {
		worldId: "main",
		body: { type: "fixed" },
		authority: "physics",
	});
	state.physics.addCollider(groundBody, {
		mode: "explicit",
		shape: { kind: "box", halfExtents: { x: size / 2, y: thickness / 2, z: size / 2 } },
		material: { restitution: 0.5, friction: 0.5 },
	});

	// Front Wall (Z = -size/2)
	const frontWall = MeshFactory.createBox(
		{ x: 0, y: height / 2, z: -size / 2 - thickness / 2 },
		size + thickness * 2,
		thickness,
		height,
		wallMat,
	);
	state.scene.add(frontWall);
	const frontBody = state.physics.attachBody(frontWall, {
		worldId: "main",
		body: { type: "fixed" },
		authority: "physics",
	});
	state.physics.addCollider(frontBody, {
		mode: "explicit",
		shape: {
			kind: "box",
			halfExtents: { x: (size + thickness * 2) / 2, y: height / 2, z: thickness / 2 },
		},
		material: { restitution: 0.5, friction: 0.5 },
	});

	// Back Wall (Z = size/2)
	const backWall = MeshFactory.createBox(
		{ x: 0, y: height / 2, z: size / 2 + thickness / 2 },
		size + thickness * 2,
		thickness,
		height,
		wallMat,
	);
	state.scene.add(backWall);
	const backBody = state.physics.attachBody(backWall, {
		worldId: "main",
		body: { type: "fixed" },
		authority: "physics",
	});
	state.physics.addCollider(backBody, {
		mode: "explicit",
		shape: {
			kind: "box",
			halfExtents: { x: (size + thickness * 2) / 2, y: height / 2, z: thickness / 2 },
		},
		material: { restitution: 0.5, friction: 0.5 },
	});

	// Left Wall (X = -size/2)
	const leftWall = MeshFactory.createBox(
		{ x: -size / 2 - thickness / 2, y: height / 2, z: 0 },
		thickness,
		size,
		height,
		wallMat,
	);
	state.scene.add(leftWall);
	const leftBody = state.physics.attachBody(leftWall, {
		worldId: "main",
		body: { type: "fixed" },
		authority: "physics",
	});
	state.physics.addCollider(leftBody, {
		mode: "explicit",
		shape: { kind: "box", halfExtents: { x: thickness / 2, y: height / 2, z: size / 2 } },
		material: { restitution: 0.5, friction: 0.5 },
	});

	// Right Wall (X = size/2)
	const rightWall = MeshFactory.createBox(
		{ x: size / 2 + thickness / 2, y: height / 2, z: 0 },
		thickness,
		size,
		height,
		wallMat,
	);
	state.scene.add(rightWall);
	const rightBody = state.physics.attachBody(rightWall, {
		worldId: "main",
		body: { type: "fixed" },
		authority: "physics",
	});
	state.physics.addCollider(rightBody, {
		mode: "explicit",
		shape: { kind: "box", halfExtents: { x: thickness / 2, y: height / 2, z: size / 2 } },
		material: { restitution: 0.5, friction: 0.5 },
	});
}

// ----------------------------------------------------
// Spawning Dynamic Shapes
// ----------------------------------------------------
function spawnObject(state: DemoState, shapeKind: "box" | "sphere" | "cylinder"): void {
	// Restrict to max objects configured in settings to prevent performance lag
	if (state.spawnedItems.length >= state.settings.maxObjects) {
		const oldest = state.spawnedItems.shift();
		if (oldest) {
			state.physics.detachBody(oldest.node);
			state.scene.remove(oldest.node);
		}
	}

	const x = (Math.random() - 0.5) * 10;
	const y = 8 + Math.random() * 6;
	const z = (Math.random() - 0.5) * 10;

	// Vibrant curated color palette
	const colors = [
		{ r: 239, g: 68, b: 68 }, // Red
		{ r: 16, g: 185, b: 129 }, // Emerald Green
		{ r: 59, g: 130, b: 246 }, // Blue
		{ r: 245, g: 158, b: 11 }, // Amber/Yellow
		{ r: 139, g: 92, b: 246 }, // Violet
		{ r: 6, g: 182, b: 212 }, // Cyan
		{ r: 236, g: 72, b: 153 }, // Pink
		{ r: 249, g: 115, b: 22 }, // Orange
	];
	const albedo = colors[Math.floor(Math.random() * colors.length)];

	const material = new PBRMaterial();
	material.albedo = albedo;
	material.roughness = 0.2 + Math.random() * 0.4;
	material.metalness = Math.random() > 0.65 ? 0.8 : 0.05;

	let node: MeshInstance;
	let halfExtents: { x: number; y: number; z: number } | null = null;
	let radius = 0;
	let halfHeight = 0;

	if (shapeKind === "box") {
		const w = 0.8 + Math.random() * 0.8;
		const h = 0.8 + Math.random() * 0.8;
		const d = 0.8 + Math.random() * 0.8;
		node = MeshFactory.createBox({ x, y, z }, w, d, h, material);
		halfExtents = { x: w / 2, y: h / 2, z: d / 2 };
	} else if (shapeKind === "sphere") {
		radius = 0.45 + Math.random() * 0.45;
		node = MeshFactory.createSphere({ x, y, z }, radius, 18, 14, material);
	} else {
		radius = 0.4 + Math.random() * 0.3;
		const h = 1.0 + Math.random() * 0.8;
		node = MeshFactory.createCylinder({ x, y, z }, radius, h, 16, material);
		halfHeight = h / 2;
	}

	// Add random initial orientation
	node.setRotationFromEuler(
		Math.random() * Math.PI,
		Math.random() * Math.PI,
		Math.random() * Math.PI,
	);

	state.scene.add(node);

	const restitution = state.settings.restitution;
	const friction = state.settings.friction;

	// Attach dynamic physics body
	const body = state.physics.attachBody(node, {
		worldId: "main",
		body: {
			type: "dynamic",
			linearVelocity: {
				x: (Math.random() - 0.5) * 3,
				y: -2,
				z: (Math.random() - 0.5) * 3,
			},
			angularVelocity: {
				x: (Math.random() - 0.5) * 6,
				y: (Math.random() - 0.5) * 6,
				z: (Math.random() - 0.5) * 6,
			},
		},
		authority: "physics",
	});

	// Attach matching physical collider
	if (shapeKind === "box" && halfExtents) {
		state.physics.addCollider(body, {
			mode: "explicit",
			shape: { kind: "box", halfExtents },
			material: { restitution, friction },
		});
	} else if (shapeKind === "sphere") {
		state.physics.addCollider(body, {
			mode: "explicit",
			shape: { kind: "sphere", radius },
			material: { restitution, friction },
		});
	} else if (shapeKind === "cylinder") {
		state.physics.addCollider(body, {
			mode: "explicit",
			shape: { kind: "cylinder", radius, halfHeight },
			material: { restitution, friction },
		});
	}

	state.spawnedItems.push({ node, shapeKind });
}

// ----------------------------------------------------
// Physics Actions & Helpers
// ----------------------------------------------------
function applyExplosion(state: DemoState): void {
	// Push all dynamic objects outward and upward
	for (const item of state.spawnedItems) {
		const pos = item.node.position;
		// Distance vector from center (0, 0, 0)
		const dx = pos.x;
		const dz = pos.z;
		const dist = Math.sqrt(dx * dx + dz * dz) || 0.1;

		// Calculate outward impulse force
		const forceX = (dx / dist) * 4.5;
		const forceZ = (dz / dist) * 4.5;
		const forceY = 10.0 + Math.random() * 4.0; // strong upward boost

		state.physics.applyImpulse(item.node, { x: forceX, y: forceY, z: forceZ });
	}

	// Trigger camera shake effect
	state.cameraShake.trigger({
		intensity: 0.8,
		durationSeconds: 0.4,
		frequencyHz: 10,
	});
}

function clearSpawnedObjects(state: DemoState): void {
	for (const item of state.spawnedItems) {
		state.physics.detachBody(item.node);
		state.scene.remove(item.node);
	}
	state.spawnedItems = [];
}

// ----------------------------------------------------
// UI Sync & HUD Updates
// ----------------------------------------------------
function updateHUD(state: DemoState, deltaTimeSeconds: number): void {
	state.frameCount++;
	const now = performance.now();
	const elapsed = now - state.lastFpsUpdate;

	if (elapsed >= 500) {
		const fps = Math.round((state.frameCount * 1000) / elapsed);
		state.diagnostics.fps = String(fps);
		state.diagnostics.ms = `${(deltaTimeSeconds * 1000).toFixed(1)} ms`;
		state.frameCount = 0;
		state.lastFpsUpdate = now;

		state.paneBindings.fps?.refresh();
		state.paneBindings.ms?.refresh();
	}

	// Read raw counts from Rapier adapter cache
	// @ts-expect-error - Accessing private world runtime state
	const runtime = state.physics._runtimeByWorldId.get("main");
	if (runtime) {
		state.diagnostics.active = String(runtime.cachedStats.activeBodies);
		state.diagnostics.sleeping = String(runtime.cachedStats.sleepingBodies);

		state.paneBindings.active?.refresh();
		state.paneBindings.sleeping?.refresh();
	}
}

// ----------------------------------------------------
// Event Bindings & Orbit Controls
// ----------------------------------------------------
function bindOrbitControls(state: DemoState): void {
	const lookSensitivity = 0.005;
	const zoomSensitivity = 0.02;
	const activePointers: Map<number, { x: number; y: number }> = new Map();
	let prevPinchDist = 0;

	canvas.addEventListener("pointerdown", (event) => {
		activePointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
		canvas.setPointerCapture(event.pointerId);

		if (activePointers.size === 2) {
			const pts = Array.from(activePointers.values());
			prevPinchDist = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
		}
	});

	canvas.addEventListener("pointermove", (event) => {
		const prev = activePointers.get(event.pointerId);
		if (!prev) return;

		if (activePointers.size === 1) {
			const dx = event.clientX - prev.x;
			const dy = event.clientY - prev.y;
			if (dx !== 0 || dy !== 0) {
				state.camera.rotate(dx * lookSensitivity, dy * lookSensitivity);
				state.scene.updateWorldMatrices();
				state.renderer.requestRender("camera");
			}
			activePointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
		} else if (activePointers.size === 2) {
			activePointers.set(event.pointerId, { x: event.clientX, y: event.clientY });

			const pts = Array.from(activePointers.values());
			const dist = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
			const delta = prevPinchDist - dist;

			state.camera.zoom(delta * zoomSensitivity);
			state.scene.updateWorldMatrices();
			state.renderer.requestRender("camera");

			prevPinchDist = dist;
		}
	});

	const clearPointer = (event: PointerEvent) => {
		activePointers.delete(event.pointerId);
		if (canvas.hasPointerCapture(event.pointerId)) {
			canvas.releasePointerCapture(event.pointerId);
		}
		if (activePointers.size < 2) {
			prevPinchDist = 0;
		}
	};
	canvas.addEventListener("pointerup", clearPointer);
	canvas.addEventListener("pointercancel", clearPointer);

	canvas.addEventListener(
		"wheel",
		(event) => {
			event.preventDefault();
			state.camera.zoom(event.deltaY * zoomSensitivity);
			state.scene.updateWorldMatrices();
			state.renderer.requestRender("camera");
		},
		{ passive: false },
	);

	window.addEventListener("resize", () => {
		state.renderer.resizeCanvas();
		state.scene.updateWorldMatrices();
		state.renderer.requestRender("resize");
	});
}

// ----------------------------------------------------
// Tweakpane UI Initialization
// ----------------------------------------------------
function createTweakpane(state: DemoState): void {
	const pane = new Pane({ title: "Physics Playground", expanded: true });

	// Spawn Objects Folder
	const spawnFolder = pane.addFolder({ title: "Spawn Objects", expanded: true });

	// Single Spawners for precise touch spawning
	spawnFolder.addButton({ title: "Spawn Box (B)" }).on("click", () => {
		spawnObject(state, "box");
	});
	spawnFolder.addButton({ title: "Spawn Sphere (S)" }).on("click", () => {
		spawnObject(state, "sphere");
	});
	spawnFolder.addButton({ title: "Spawn Cylinder (C)" }).on("click", () => {
		spawnObject(state, "cylinder");
	});

	// Batch Spawners
	spawnFolder.addButton({ title: "Spawn 10 Boxes" }).on("click", () => {
		for (let i = 0; i < 10; i++) spawnObject(state, "box");
	});
	spawnFolder.addButton({ title: "Spawn 10 Spheres" }).on("click", () => {
		for (let i = 0; i < 10; i++) spawnObject(state, "sphere");
	});
	spawnFolder.addButton({ title: "Spawn 10 Cylinders" }).on("click", () => {
		for (let i = 0; i < 10; i++) spawnObject(state, "cylinder");
	});

	// Simulation Controls Folder
	const controlsFolder = pane.addFolder({ title: "Simulation Controls", expanded: true });
	controlsFolder
		.addButton({ title: "💥 Explosion Impulse" })
		.on("click", () => applyExplosion(state));
	controlsFolder
		.addButton({ title: "🗑️ Reset Scene" })
		.on("click", () => clearSpawnedObjects(state));

	// Settings Folder
	const settingsFolder = pane.addFolder({ title: "Physics Settings", expanded: true });
	settingsFolder
		.addBinding(state.settings, "gravity", {
			min: -25,
			max: 5,
			step: 0.5,
			label: "Gravity",
		})
		.on("change", (ev) => {
			const val = ev.value;
			// @ts-expect-error - Mutating private world gravity scale dynamically
			const worldState = state.physics._runtimeByWorldId.get("main");
			if (worldState) {
				worldState.world.gravity.y = val;
				worldState.config.gravity.y = val;
			}
		});

	settingsFolder.addBinding(state.settings, "restitution", {
		min: 0,
		max: 1,
		step: 0.05,
		label: "Bounciness",
	});

	settingsFolder.addBinding(state.settings, "friction", {
		min: 0,
		max: 1,
		step: 0.05,
		label: "Friction",
	});

	settingsFolder.addBinding(state.settings, "maxObjects", {
		min: 10,
		max: 500,
		step: 10,
		label: "Max Objects",
	});

	settingsFolder
		.addBinding(state.settings, "fxaa", {
			label: "FXAA Anti-Aliasing",
		})
		.on("change", (ev) => {
			const pass = state.renderer.postProcess.getPass("fxaa");
			if (pass) {
				pass.setEnabled(ev.value);
			}
		});

	settingsFolder
		.addBinding(state.settings, "backend", {
			label: "Renderer Backend",
			options: {
				"Auto (WebGPU/WebGL)": "auto",
				"Force WebGPU": "webgpu",
				"Force WebGL2": "webgl",
			},
		})
		.on("change", (ev) => {
			const url = new URL(window.location.href);
			url.searchParams.set("backend", ev.value);
			window.location.href = url.toString();
		});

	// Diagnostics / Stats Folder
	const statsFolder = pane.addFolder({ title: "Diagnostics", expanded: true });
	state.paneBindings.fps = statsFolder.addBinding(state.diagnostics, "fps", {
		readonly: true,
		label: "FPS",
	});
	state.paneBindings.ms = statsFolder.addBinding(state.diagnostics, "ms", {
		readonly: true,
		label: "Frame Time",
	});
	state.paneBindings.active = statsFolder.addBinding(state.diagnostics, "active", {
		readonly: true,
		label: "Active Bodies",
	});
	state.paneBindings.sleeping = statsFolder.addBinding(state.diagnostics, "sleeping", {
		readonly: true,
		label: "Sleeping Bodies",
	});
	state.paneBindings.activeBackend = statsFolder.addBinding(state.diagnostics, "activeBackend", {
		readonly: true,
		label: "Active Backend",
	});

	// Add Keyboard Shortcuts
	window.addEventListener("keydown", (event) => {
		const key = event.key.toLowerCase();
		if (key === "b") {
			spawnObject(state, "box");
		} else if (key === "s") {
			spawnObject(state, "sphere");
		} else if (key === "c") {
			spawnObject(state, "cylinder");
		} else if (key === " ") {
			event.preventDefault(); // prevent scroll
			applyExplosion(state);
		} else if (key === "r") {
			clearSpawnedObjects(state);
		}
	});
}

// ----------------------------------------------------
// ----------------------------------------------------
// Out of Bounds Cleanup
// ----------------------------------------------------
function cleanOutOfBoundsObjects(state: DemoState): void {
	for (let i = state.spawnedItems.length - 1; i >= 0; i--) {
		const item = state.spawnedItems[i];
		const pos = item.node.position;
		// If object falls below Y = -10, or flies too far out (X or Z beyond 35 units)
		if (pos.y < -10 || Math.abs(pos.x) > 35 || Math.abs(pos.z) > 35) {
			state.physics.detachBody(item.node);
			state.scene.remove(item.node);
			state.spawnedItems.splice(i, 1);
		}
	}
}

// ----------------------------------------------------
// Frame Render Loop
// ----------------------------------------------------
function startRenderLoop(state: DemoState): void {
	const loop = async (now: number) => {
		requestAnimationFrame(loop);

		if (state.isRendering) {
			return;
		}

		state.isRendering = true;
		const deltaTime = Math.min((now - state.lastTime) / 1000, 0.1);
		state.lastTime = now;

		try {
			// 1. Run physics step manually (async)
			await state.physics.stepAsync(deltaTime);

			// Clean up out-of-bounds objects
			cleanOutOfBoundsObjects(state);

			// 2. Refresh node world matrices
			state.scene.updateWorldMatrices();

			// 3. Render the IgnisRenderer frame
			state.renderer.requestRender("unknown");
			await state.renderer.renderFrame(now);

			// 4. Tick HUD diagnostics
			updateHUD(state, deltaTime);
		} catch (err) {
			console.error("Frame loop execution failed", err);
		} finally {
			state.isRendering = false;
		}
	};
	requestAnimationFrame(loop);
}

// ----------------------------------------------------
// Main Initialization Entry
// ----------------------------------------------------
async function main() {
	try {
		const state = await bootDemo();
		bindOrbitControls(state);
		createTweakpane(state);
		startRenderLoop(state);
	} catch (error) {
		console.error("Failed to start Rigid Body Collision Demo:", error);
		const overlay = document.createElement("div");
		overlay.style.position = "absolute";
		overlay.style.top = "50%";
		overlay.style.left = "50%";
		overlay.style.transform = "translate(-50%, -50%)";
		overlay.style.background = "rgba(239, 68, 68, 0.9)";
		overlay.style.padding = "24px";
		overlay.style.borderRadius = "12px";
		overlay.style.border = "1px solid rgba(255,255,255,0.2)";
		overlay.style.color = "#fff";
		overlay.style.textAlign = "center";
		overlay.style.fontSize = "16px";
		overlay.style.fontFamily = "sans-serif";
		overlay.innerHTML = `<h3>Demo Error</h3><p>${String(error)}</p><p style='font-size:12px;opacity:0.8'>Make sure you are running in a supported WebGL2/WebGPU browser.</p>`;
		document.body.appendChild(overlay);
	}
}

void main();
