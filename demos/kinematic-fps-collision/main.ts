import { Pane } from "tweakpane";

import {
	AmbientLight,
	DirectionalLight,
	MeshFactory,
	OrbitCamera,
	Platform,
	Renderer,
	Scene,
	Vector3,
	WebGLBackend,
	WebGPUBackend,
	PhysicsSystem,
	RapierPhysicsAdapter,
	PhongMaterial,
	FastApproximateAntiAliasingPass,
} from "../../src/index";
import type { MeshInstance } from "../../src/index";

// ----------------------------------------------------
// Core Types & State
// ----------------------------------------------------
type DemoBackendPreference = "auto" | "webgpu" | "webgl";

interface DemoSettings {
	gravity: number;
	playerSpeed: number;
	jumpForce: number;
}

interface DemoDiagnostics {
	fps: string;
	ms: string;
	active: string;
	sleeping: string;
	activeBackend: string;
	physicsAdapter: string;
}

interface DemoState {
	renderer: Renderer;
	scene: Scene;
	camera: OrbitCamera;
	physics: PhysicsSystem;
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
		physicsAdapter?: any;
	};
	keysPressed: Set<string>;
	// Player Dynamic Physics Object
	playerNode: MeshInstance;
	playerBody: any;
	// Kinematic Moving Platform
	platformNode: MeshInstance;
	platformBody: any;
	// Kinematic Moving Elevator
	elevatorNode: MeshInstance;
	elevatorBody: any;
	platformInertia: { x: number; z: number };
	pushableItems: Array<{ node: MeshInstance; spawnPos: { x: number; y: number; z: number } }>;
	spawnedSpheres: MeshInstance[];
	virtualTime: number;
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

	// Setup Orbit Camera targeting the player's initial position
	const camera = new OrbitCamera(new Vector3(0, 1.0, 0), 12);
	camera.near = 0.1;
	camera.far = 100;
	camera.theta = -0.78; // Angle around Y
	camera.phi = 1.25; // Look slightly downwards
	camera.minDistance = 3;
	camera.maxDistance = 25;
	camera.updatePosition();

	// Setup Scene
	const scene = new Scene();
	scene.spatialIndexMode = "hybrid";
	scene.add(camera);

	// Add Ambient Light (adjusted for higher contrast shadows)
	const ambientLight = new AmbientLight({
		color: { r: 80, g: 85, b: 100 },
		intensity: 0.6,
	});
	scene.add(ambientLight);

	// Add Sun Light (Directional, Shadow casting - slightly stronger)
	const sun = new DirectionalLight({
		intensity: 5.5,
		direction: { x: 0.6, y: -1.0, z: -0.4 },
	});
	scene.add(sun);

	// Setup CSM Shadow Map on Sun
	const shadowMap = scene.shadows.createCascaded({
		size: 2048,
		lambda: 0.65,
		maxDistance: 60,
		blendRatio: 0.1,
		stabilize: true,
		sampling: {
			filterMode: "pcf",
			radius: 0.15,
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
	scene.shadows.bind(sun, shadowMap);

	// Initialize Renderer
	const renderer = new Renderer({
		canvas,
		backend,
		camera,
	});
	renderer.setScene(scene);
	renderer.features.enableLighting = true;
	renderer.features.enableShadows = true;
	renderer.features.enableEnvironment = false;
	renderer.features.enableReflection = false;
	renderer.features.enableOIT = false;

	await renderer.initialize();
	console.log("[Warmup] Starting renderer warmup...");
	const report = await renderer.warmup({
		onProgress: (progress) => {
			console.log(
				`[Warmup Progress] Phase: ${progress.phase} | ${progress.completed}/${progress.total}` +
					(progress.detail ? ` (${progress.detail})` : ""),
			);
		},
	});
	console.log(
		`[Warmup Complete] Backend: ${report.backend} | Total: ${report.total} | Compiled: ${report.compiled} | Skipped: ${report.skipped} | Failed: ${report.failed} | Duration: ${report.durationMs.toFixed(1)}ms`,
	);

	// Register FAAA pass
	renderer.postProcess.registerPass(new FastApproximateAntiAliasingPass({ enabled: true }));

	// Initialize Physics System
	const physics = new PhysicsSystem({
		adapter: new RapierPhysicsAdapter({
			strict: true,
		}),
	});
	await physics.init();

	const initialGravity = -15.0; // Slightly stronger gravity for responsive gameplay jumping/falling
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
		playerSpeed: 8,
		jumpForce: 8,
	};

	const diagnostics: DemoDiagnostics = {
		fps: "-",
		ms: "- ms",
		active: "-",
		sleeping: "-",
		activeBackend: renderer.backendProfile.id,
		physicsAdapter: "-",
	};

	// Create Floor Ground
	createFloor(physics, scene);

	// Create static obstacles
	createStaticObstacles(physics, scene);

	// Create dynamic player character (Vibrant electric blue using PhongMaterial)
	const playerMat = new PhongMaterial({
		diffuse: { r: 0, g: 140, b: 255 },
		specular: { r: 50, g: 50, b: 50 },
		shininess: 16,
	});

	// Tall player box (w: 1.0, d: 1.0, h: 2.0) starting at (0, 1.0, 3)
	const playerNode = MeshFactory.createBox({ x: 0, y: 1.0, z: 3.0 }, 1.0, 1.0, 2.0, playerMat);
	scene.add(playerNode);

	const playerBody = physics.attachBody(playerNode, {
		worldId: "main",
		body: {
			type: "dynamic",
			lockRotations: [true, true, true], // lock all rotations so the character stands straight
		},
		authority: "physics",
	});

	physics.addCollider(playerBody, {
		mode: "explicit",
		shape: { kind: "box", halfExtents: { x: 0.5, y: 1.0, z: 0.5 } },
		material: { restitution: 0.05, friction: 0.9 }, // High friction, low bounce for stable control
	});

	// Create kinematic moving platform (Vibrant neon orange using PhongMaterial) - moved further away and height lowered to 0.4
	const platformMat = new PhongMaterial({
		diffuse: { r: 255, g: 85, b: 0 },
		specular: { r: 50, g: 50, b: 50 },
		shininess: 16,
	});

	// Flat box (w: 3.0, d: 3.0, h: 0.4) starting at (-8, 0.4, 6.0)
	const platformNode = MeshFactory.createBox(
		{ x: -8, y: 0.4, z: 6.0 },
		3.0,
		3.0,
		0.4,
		platformMat,
	);
	scene.add(platformNode);

	const platformBody = physics.attachBody(platformNode, {
		worldId: "main",
		body: {
			type: "kinematic",
		},
		authority: "animation",
	});

	physics.addCollider(platformBody, {
		mode: "explicit",
		shape: { kind: "box", halfExtents: { x: 1.5, y: 0.2, z: 1.5 } },
		material: { restitution: 0.1, friction: 0.5 },
	});

	// Create vertical elevator platform (Vibrant gold yellow using PhongMaterial)
	const elevatorMat = new PhongMaterial({
		diffuse: { r: 255, g: 215, b: 0 },
		specular: { r: 50, g: 50, b: 50 },
		shininess: 16,
	});

	// Flat box (w: 3.0, d: 3.0, h: 0.4) starting at (5.0, 0.4, -5.0)
	const elevatorNode = MeshFactory.createBox(
		{ x: 5.0, y: 0.4, z: -5.0 },
		3.0,
		3.0,
		0.4,
		elevatorMat,
	);
	scene.add(elevatorNode);

	const elevatorBody = physics.attachBody(elevatorNode, {
		worldId: "main",
		body: {
			type: "kinematic",
		},
		authority: "animation",
	});

	physics.addCollider(elevatorBody, {
		mode: "explicit",
		shape: { kind: "box", halfExtents: { x: 1.5, y: 0.2, z: 1.5 } },
		material: { restitution: 0.1, friction: 0.5 },
	});

	// Spawn pushable dynamic objects (Vibrant green & pink-red)
	const pushableMat1 = new PhongMaterial({
		diffuse: { r: 0, g: 220, b: 100 },
		specular: { r: 40, g: 40, b: 40 },
		shininess: 16,
	});
	const pushableMat2 = new PhongMaterial({
		diffuse: { r: 255, g: 40, b: 100 },
		specular: { r: 40, g: 40, b: 40 },
		shininess: 16,
	});

	const pushableItems: Array<{
		node: MeshInstance;
		spawnPos: { x: number; y: number; z: number };
	}> = [];

	// Pushable Box 1
	const pos1 = { x: 2.0, y: 0.6, z: 2.0 };
	const box1 = MeshFactory.createBox(pos1, 1.2, 1.2, 1.2, pushableMat1);
	scene.add(box1);
	const body1 = physics.attachBody(box1, {
		worldId: "main",
		body: { type: "dynamic" },
		authority: "physics",
	});
	physics.addCollider(body1, {
		mode: "explicit",
		shape: { kind: "box", halfExtents: { x: 0.6, y: 0.6, z: 0.6 } },
		material: { restitution: 0.3, friction: 0.5 },
	});
	pushableItems.push({ node: box1, spawnPos: pos1 });

	// Pushable Box 2
	const pos2 = { x: -2.0, y: 0.5, z: 1.0 };
	const box2 = MeshFactory.createBox(pos2, 1.0, 1.0, 1.0, pushableMat2);
	scene.add(box2);
	const body2 = physics.attachBody(box2, {
		worldId: "main",
		body: { type: "dynamic" },
		authority: "physics",
	});
	physics.addCollider(body2, {
		mode: "explicit",
		shape: { kind: "box", halfExtents: { x: 0.5, y: 0.5, z: 0.5 } },
		material: { restitution: 0.3, friction: 0.5 },
	});
	pushableItems.push({ node: box2, spawnPos: pos2 });

	const state: DemoState = {
		renderer,
		scene,
		camera,
		physics,
		lastTime: performance.now(),
		frameCount: 0,
		lastFpsUpdate: performance.now(),
		isRendering: false,
		settings,
		diagnostics,
		paneBindings: {},
		keysPressed: new Set(),
		playerNode,
		playerBody,
		platformNode,
		platformBody,
		elevatorNode,
		elevatorBody,
		platformInertia: { x: 0, z: 0 },
		pushableItems,
		spawnedSpheres: [],
		virtualTime: 0,
	};

	return state;
}

// ----------------------------------------------------
// Floor Ground Creation
// ----------------------------------------------------
function createFloor(physics: PhysicsSystem, scene: Scene): void {
	const groundMat = new PhongMaterial({
		diffuse: { r: 250, g: 250, b: 250 },
		specular: { r: 20, g: 20, b: 20 },
		shininess: 8,
	});

	const size = 30;
	const thickness = 0.3;

	const ground = MeshFactory.createBox(
		{ x: 0, y: -thickness / 2, z: 0 },
		size,
		size,
		thickness,
		groundMat,
	);
	scene.add(ground);
	const groundBody = physics.attachBody(ground, {
		worldId: "main",
		body: { type: "fixed" },
		authority: "physics",
	});
	physics.addCollider(groundBody, {
		mode: "explicit",
		shape: { kind: "box", halfExtents: { x: size / 2, y: thickness / 2, z: size / 2 } },
		material: { restitution: 0.5, friction: 0.5 },
	});
}

// ----------------------------------------------------
// Static Obstacles Creation
// ----------------------------------------------------
function createStaticObstacles(physics: PhysicsSystem, scene: Scene): void {
	const obstacleMat = new PhongMaterial({
		diffuse: { r: 140, g: 0, b: 255 },
		specular: { r: 30, g: 30, b: 30 },
		shininess: 12,
	});

	// 1. Tall Cylinder Pillar
	const pillar = MeshFactory.createCylinder({ x: 3, y: 2.0, z: 3 }, 0.6, 4.0, 12, obstacleMat);
	scene.add(pillar);
	const pillarBody = physics.attachBody(pillar, {
		worldId: "main",
		body: { type: "fixed" },
		authority: "physics",
	});
	physics.addCollider(pillarBody, {
		mode: "explicit",
		shape: { kind: "cylinder", radius: 0.6, halfHeight: 2.0 },
		material: { restitution: 0.4, friction: 0.5 },
	});

	// 2. Long low barrier block
	const barrier = MeshFactory.createBox({ x: -4, y: 0.75, z: 0 }, 4.0, 1.0, 1.5, obstacleMat);
	scene.add(barrier);
	const barrierBody = physics.attachBody(barrier, {
		worldId: "main",
		body: { type: "fixed" },
		authority: "physics",
	});
	physics.addCollider(barrierBody, {
		mode: "explicit",
		shape: { kind: "box", halfExtents: { x: 2.0, y: 0.75, z: 0.5 } },
		material: { restitution: 0.4, friction: 0.5 },
	});

	// 3. Central big box obstacle
	const bigBox = MeshFactory.createBox({ x: 0, y: 1.0, z: -4 }, 2.0, 2.0, 2.0, obstacleMat);
	scene.add(bigBox);
	const bigBoxBody = physics.attachBody(bigBox, {
		worldId: "main",
		body: { type: "fixed" },
		authority: "physics",
	});
	physics.addCollider(bigBoxBody, {
		mode: "explicit",
		shape: { kind: "box", halfExtents: { x: 1.0, y: 1.0, z: 1.0 } },
		material: { restitution: 0.4, friction: 0.5 },
	});
}

// ----------------------------------------------------
// Spawn Pushable Sphere Above Player
// ----------------------------------------------------
function spawnSphereAbovePlayer(state: DemoState): void {
	const playerPos = state.playerNode.position;
	// Spawn 3.0 units above player center (since player is box height 2.0, this is well above the player)
	const spawnPos = {
		x: playerPos.x,
		y: playerPos.y + 3.0,
		z: playerPos.z,
	};

	// Curated vibrant color palette using PhongMaterial
	const colors = [
		{ r: 255, g: 0, b: 128 }, // Hot Pink
		{ r: 0, g: 255, b: 255 }, // Cyan
		{ r: 255, g: 215, b: 0 }, // Gold
		{ r: 128, g: 0, b: 255 }, // Violet
		{ r: 255, g: 85, b: 0 }, // Neon Orange
		{ r: 0, g: 220, b: 100 }, // Vibrant Green
	];
	const color = colors[Math.floor(Math.random() * colors.length)];

	const material = new PhongMaterial({
		diffuse: color,
		specular: { r: 50, g: 50, b: 50 },
		shininess: 16,
	});

	const radius = 0.5;
	const sphereNode = MeshFactory.createSphere(spawnPos, radius, 16, 12, material);
	state.scene.add(sphereNode);

	const body = state.physics.attachBody(sphereNode, {
		worldId: "main",
		body: { type: "dynamic" },
		authority: "physics",
	});

	state.physics.addCollider(body, {
		mode: "explicit",
		shape: { kind: "sphere", radius },
		material: { restitution: 0.5, friction: 0.5 },
	});

	state.spawnedSpheres.push(sphereNode);
}

// ----------------------------------------------------
// Orbit Controls Binding
// ----------------------------------------------------
function bindOrbitControls(state: DemoState): void {
	const lookSensitivity = 0.005;
	const zoomSensitivity = 0.02;
	const activePointers: Map<number, { x: number; y: number }> = new Map();
	let prevPinchDist = 0;

	// Request Pointer Lock on clicking the canvas to hide cursor
	canvas.addEventListener("click", () => {
		if (document.pointerLockElement !== canvas) {
			canvas.requestPointerLock();
		}
	});

	// Listen to mouse movement for Pointer Lock camera control
	window.addEventListener("mousemove", (event) => {
		if (document.pointerLockElement === canvas) {
			const dx = event.movementX;
			const dy = event.movementY;
			if (dx !== 0 || dy !== 0) {
				state.camera.rotate(dx * lookSensitivity, dy * lookSensitivity);
				state.scene.updateWorldMatrices();
				state.renderer.requestRender("camera");
			}
		}
	});

	// Camera rotates when dragging, targets the player (active only if not pointer locked)
	canvas.addEventListener("pointerdown", (event) => {
		if (document.pointerLockElement === canvas) return;
		activePointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
		canvas.setPointerCapture(event.pointerId);

		if (activePointers.size === 2) {
			const pts = Array.from(activePointers.values());
			prevPinchDist = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
		}
	});

	canvas.addEventListener("pointermove", (event) => {
		if (document.pointerLockElement === canvas) return;
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

	// WASD & Jump capture
	window.addEventListener("keydown", (event) => {
		state.keysPressed.add(event.code);
		if (event.code === "KeyQ" && !event.repeat) {
			spawnSphereAbovePlayer(state);
		}
	});

	window.addEventListener("keyup", (event) => {
		state.keysPressed.delete(event.code);
	});
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

	const adapterId = state.physics.getAdapterId();
	state.diagnostics.physicsAdapter = adapterId;
	state.paneBindings.physicsAdapter?.refresh();

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
// Tweakpane UI Initialization
// ----------------------------------------------------
function createTweakpane(state: DemoState): void {
	const pane = new Pane({ title: "Third-Person Playground", expanded: true });

	const playerFolder = pane.addFolder({ title: "Player Settings", expanded: true });
	playerFolder.addBinding(state.settings, "playerSpeed", {
		min: 2.0,
		max: 12.0,
		step: 0.5,
		label: "Move Speed",
	});
	playerFolder.addBinding(state.settings, "jumpForce", {
		min: 3.0,
		max: 12.0,
		step: 0.5,
		label: "Jump Force",
	});

	const physicsFolder = pane.addFolder({ title: "Physics Settings", expanded: false });
	physicsFolder
		.addBinding(state.settings, "gravity", {
			min: -30,
			max: 0,
			step: 1.0,
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
	state.paneBindings.physicsAdapter = statsFolder.addBinding(
		state.diagnostics,
		"physicsAdapter",
		{
			readonly: true,
			label: "Physics Adapter",
		},
	);
}

// ----------------------------------------------------
// Out of Bounds Cleanup
// ----------------------------------------------------
// If the player or pushable objects fall out of bounds, reset their positions
function cleanOutOfBoundsObjects(state: DemoState): void {
	if (state.playerNode.position.y < -10) {
		// Teleport player node
		state.playerNode.position.set(0, 4.0, 0);
		// Teleport player physics body as well
		state.physics.setLinearVelocity(state.playerNode, { x: 0, y: 0, z: 0 });
		state.physics.setBodyTransform(state.playerNode, {
			position: { x: 0, y: 4.0, z: 0 },
			rotation: [0, 0, 0, 1],
		});
	}

	for (const item of state.pushableItems) {
		if (item.node.position.y < -10) {
			state.physics.setLinearVelocity(item.node, { x: 0, y: 0, z: 0 });
			state.physics.setAngularVelocity(item.node, { x: 0, y: 0, z: 0 });
			state.physics.setBodyTransform(item.node, {
				position: item.spawnPos,
				rotation: [0, 0, 0, 1],
			});
		}
	}

	// Clean up out of bounds spawned spheres completely (one-time / disposable)
	for (let i = state.spawnedSpheres.length - 1; i >= 0; i--) {
		const sphere = state.spawnedSpheres[i];
		if (sphere.position.y < -10) {
			state.physics.detachBody(sphere);
			state.scene.remove(sphere);
			state.spawnedSpheres.splice(i, 1);
		}
	}
}

// ----------------------------------------------------
// Frame Render Loop
// ----------------------------------------------------
function setupRenderLoop(state: DemoState): void {
	let isUpdating = false;

	state.renderer.on("tick", async ({ now, deltaTime }) => {
		if (isUpdating) {
			return;
		}
		isUpdating = true;

		// Convert ms to seconds and cap to avoid spikes
		const deltaTimeSeconds = Math.min(deltaTime / 1000, 0.1);

		try {
			state.virtualTime += deltaTimeSeconds;

			// 1. Process WASD relative to the camera horizontal yaw orientation
			const theta = state.camera.theta;

			// Camera look directions (horizontal plane)
			const forwardX = -Math.sin(theta);
			const forwardZ = -Math.cos(theta);

			// Right vector perpendicular to look direction
			const rightX = Math.cos(theta);
			const rightZ = -Math.sin(theta);

			let moveX = 0;
			let moveZ = 0;

			if (state.keysPressed.has("KeyW")) {
				moveX += forwardX;
				moveZ += forwardZ;
			}
			if (state.keysPressed.has("KeyS")) {
				moveX -= forwardX;
				moveZ -= forwardZ;
			}
			if (state.keysPressed.has("KeyA")) {
				moveX -= rightX;
				moveZ -= rightZ;
			}
			if (state.keysPressed.has("KeyD")) {
				moveX += rightX;
				moveZ += rightZ;
			}

			// Normalize movement vector
			const moveLen = Math.hypot(moveX, moveZ);
			let vx = 0;
			let vz = 0;
			if (moveLen > 0) {
				vx = (moveX / moveLen) * state.settings.playerSpeed;
				vz = (moveZ / moveLen) * state.settings.playerSpeed;
			}

			// Read current velocity to maintain gravitational Y velocity
			const currentVel = state.physics.getLinearVelocity(state.playerNode) || {
				x: 0,
				y: 0,
				z: 0,
			};
			let vy = currentVel.y;

			// Raycast downwards from the player's center to check if we are grounded
			const rayHit = await state.physics.raycastAsync({
				worldId: "main",
				origin: state.playerNode.position,
				direction: { x: 0, y: -1, z: 0 },
				maxDistance: 1.1,
				filter: {
					excludeBodyIds: [state.playerBody.id],
				},
			});
			// Grounded only if ray hits and the player is not rising (jumping)
			const isGrounded = rayHit !== null && currentVel.y <= 0.05;
			const standingOnPlatform =
				isGrounded && rayHit && rayHit.bodyId === state.platformBody.id;
			const standingOnElevator =
				isGrounded && rayHit && rayHit.bodyId === state.elevatorBody.id;

			// Compute moving platform translation delta (repositioned around X = -8, Y = 0.4, Z = 6)
			const platformX = -8.0 + Math.sin(state.virtualTime * 1.2) * 4.0;
			const deltaX = platformX - state.platformNode.position.x;
			const platformVel = { x: 4.8 * Math.cos(state.virtualTime * 1.2), y: 0, z: 0 };

			// Compute moving elevator translation delta
			const elevatorY = 2.7 + Math.sin(state.virtualTime * 1.0) * 2.3;
			const deltaY = elevatorY - state.elevatorNode.position.y;
			const elevatorVel = { x: 0, y: 2.3 * Math.cos(state.virtualTime * 1.0), z: 0 };

			let finalVx = vx;
			let finalVz = vz;

			if (standingOnPlatform) {
				// 1. Teleport player directly by the platform delta to prevent sliding/falling off due to thread lag
				const playerPos = state.playerNode.position;
				state.playerNode.position.set(playerPos.x + deltaX, playerPos.y, playerPos.z);
				state.physics.setBodyTransform(state.playerNode, {
					position: state.playerNode.position,
					rotation: [0, 0, 0, 1],
				});

				// 2. Store platform velocity as inertia for when they leave the platform
				state.platformInertia.x = platformVel.x;
				state.platformInertia.z = platformVel.z;
			} else if (standingOnElevator) {
				// 1. Teleport player vertically by the elevator delta
				const playerPos = state.playerNode.position;
				state.playerNode.position.set(playerPos.x, playerPos.y + deltaY, playerPos.z);
				state.physics.setBodyTransform(state.playerNode, {
					position: state.playerNode.position,
					rotation: [0, 0, 0, 1],
				});

				// 2. Sync Y velocity with elevator movement when not jumping
				if (!state.keysPressed.has("Space")) {
					vy = elevatorVel.y;
				}
				state.platformInertia.x = 0;
				state.platformInertia.z = 0;
			} else {
				if (isGrounded) {
					// Landed on another surface, reset inertia immediately
					state.platformInertia.x = 0;
					state.platformInertia.z = 0;
				} else {
					// Gradually damp the platform inertia while in mid-air
					state.platformInertia.x *= Math.max(0, 1 - deltaTimeSeconds * 2.0);
					state.platformInertia.z *= Math.max(0, 1 - deltaTimeSeconds * 2.0);
				}
				finalVx += state.platformInertia.x;
				finalVz += state.platformInertia.z;
			}

			if (state.keysPressed.has("Space") && isGrounded) {
				vy = state.settings.jumpForce;
			}

			// Update kinematic platform position
			state.platformNode.position.set(platformX, 0.4, 6.0);
			state.platformNode.updateLocalMatrix();

			// Update kinematic elevator position
			state.elevatorNode.position.set(5.0, elevatorY, -5.0);
			state.elevatorNode.updateLocalMatrix();

			// Update player linear velocity in physics adapter
			state.physics.setLinearVelocity(state.playerNode, {
				x: finalVx,
				y: vy,
				z: finalVz,
			});
			// Step physics simulation using exact frame deltaTimeSeconds
			await state.physics.stepAsync(deltaTimeSeconds);

			// Clean up out-of-bounds bodies
			cleanOutOfBoundsObjects(state);

			// 2. Make Camera target follow the player's new position
			state.camera.target.copy(state.playerNode.position);
			state.camera.target.y += 0.5; // Offset camera target slightly up to look at player chest height
			state.camera.updatePosition();

			// 3. Update scene graph matrix hierarchy
			state.scene.updateWorldMatrices();

			// Request renderer to draw the updated scene
			state.renderer.requestRender("camera");

			// 4. Refresh HUD metrics
			updateHUD(state, deltaTimeSeconds);
		} catch (err) {
			console.error("Frame update failed", err);
		} finally {
			isUpdating = false;
		}
	});

	// Start the automatic render loop
	state.renderer.renderLoop();
}

// ----------------------------------------------------
// Main Initialization Entry
// ----------------------------------------------------
async function main() {
	try {
		const state = await bootDemo();
		bindOrbitControls(state);
		createTweakpane(state);
		setupRenderLoop(state);
	} catch (error) {
		console.error("Failed to start Orbit Physics Controller Demo:", error);
	}
}

void main();
