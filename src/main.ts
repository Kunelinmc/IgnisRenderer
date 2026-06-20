import {
	FPSCamera,
	Platform,
	Renderer,
	Scene,
	SoftwareBackend,
	WebGPUBackend,
	WebGLBackend,
	GLTFLoader,
	InteractionController,
	MeshInstance,
	Logger,
	DirectionalLight,
	FastApproximateAntiAliasingPass,
	type Camera,
} from "./index";

async function init(): Promise<void> {
	const platform = Platform.detect();

	if (!platform.isBrowserRuntime) {
		throw new Error(`Main entry requires browser runtime, got "${platform.runtime}".`);
	}

	const canvasElement = document.getElementById("canvas3d");
	if (!(canvasElement instanceof HTMLCanvasElement)) {
		throw new Error('Missing required canvas element with id "canvas3d".');
	}

	let canvas = canvasElement;

	const camera = new FPSCamera();
	camera.near = 0.1;
	camera.far = 100;
	camera.moveSpeed = 5;
	camera.position.set(0, 5, 10);
	camera.updateMatrices();

	const scene = new Scene();
	await setupScene(scene, camera);

	const bootstrap = await createRenderer(canvas, camera, scene);
	canvas = bootstrap.canvas;

	const renderer = bootstrap.renderer;
	renderer
		.warmup({ scheduling: "idle" })
		.then((report) => {
			if (report.failed > 0) {
				Logger.warn(`Renderer warmup completed with ${report.failed} failure(s).`, {
					scope: "Main",
				});
			}
		})
		.catch((error) => {
			Logger.warn(["Renderer warmup failed.", error], {
				scope: "Main",
			});
		});

	renderer.renderLoop();

	bindControls(canvas, camera, renderer);
	setupInteraction(renderer, scene, camera);

	window.addEventListener("resize", () => {
		renderer.resizeCanvas();
		renderer.requestRender("camera");
	});
}

/**
 * Configures the scene components, including spatial indexing, loaded models,
 * directional lights, and CSM shadow mapping.
 */
async function setupScene(scene: Scene, camera: Camera): Promise<void> {
	// Set spatial index mode to hybrid
	scene.spatialIndexMode = "hybrid";

	scene.add(camera);

	// Load model from assets/models/model.glb
	const gltfLoader = new GLTFLoader();

	gltfLoader.on("progress", (event) => {
		const total = event.total || 0;
		const percent = total > 0 ? (event.loaded / total) * 100 : 0;
		Logger.info(`Loading model: ${percent.toFixed(2)}% (${event.loaded}/${total} bytes)`);
	});

	const model = await gltfLoader.load("assets/models/model.glb");
	scene.add(model);

	const sun = new DirectionalLight({
		intensity: 5.0,
		direction: { x: 0.5, y: -1, z: -1 },
	});
	scene.add(sun);

	const shadowMap = scene.shadows.createCascaded({
		size: 2048,
		lambda: 0.65,
		maxDistance: 80,
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
			constant: 0,
			slope: 0.001,
			normal: 0.01,
			normalMin: 0.01,
			texel: 1,
			max: 0.01,
		},
	});

	scene.shadows.bind(sun, shadowMap);
}

function getSelectedBackend(requested?: string): SoftwareBackend | WebGPUBackend | WebGLBackend {
	if (requested) {
		if (requested === "webgpu") {
			return new WebGPUBackend({
				enableDeferredLighting: true,
				enableEarlyZPrepass: true,
				enableOcclusionCulling: true,
			});
		}
		if (requested === "webgl") {
			return new WebGLBackend({
				enableEarlyZPrepass: true,
			});
		}
		return new SoftwareBackend({
			rasterMode: "tile",
			enableEarlyZPrepass: true,
		});
	}

	const platform = Platform.detect();

	if (platform.hasWebGPU) {
		return new WebGPUBackend({
			enableDeferredLighting: true,
			enableEarlyZPrepass: true,
			enableOcclusionCulling: true,
		});
	}

	if (platform.hasWebGL2) {
		return new WebGLBackend({
			enableEarlyZPrepass: true,
		});
	}

	return new SoftwareBackend({
		rasterMode: "tile",
		enableEarlyZPrepass: true,
	});
}

/**
 * Initializes the renderer with the selected backend, sets up the scene, and registers post-processing passes. The backend can be specified via the "backend" URL query parameter.
 */
async function createRenderer(
	canvas: HTMLCanvasElement,
	camera: Camera,
	scene: Scene,
): Promise<{ canvas: HTMLCanvasElement; renderer: Renderer }> {
	const params = new URLSearchParams(window.location.search);
	const requested = params.get("backend")?.toLowerCase() || undefined;

	const backend = getSelectedBackend(requested);
	const renderer = new Renderer({
		backend,
		canvas,
		camera,
	});
	renderer.setScene(scene);

	await renderer.initialize();

	renderer.postProcess.registerPass(new FastApproximateAntiAliasingPass({ enabled: true }));
	renderer.features.enableOIT = true;

	return { canvas, renderer };
}

/**
 * Binds keyboard and mouse controls for FPS-style camera movement and looking around.
 */
function bindControls(canvas: HTMLCanvasElement, camera: FPSCamera, renderer: Renderer): void {
	const keys = new Set<string>();

	// Keyboard Events
	window.addEventListener("keydown", (event: KeyboardEvent) => {
		keys.add(event.code);
	});

	window.addEventListener("keyup", (event: KeyboardEvent) => {
		keys.delete(event.code);
	});

	// Mouse Events
	canvas.addEventListener("mousedown", () => {
		canvas.requestPointerLock();
	});

	const onMouseMove = (event: MouseEvent) => {
		if (event.movementX === 0 && event.movementY === 0) return;
		camera.rotate(event.movementX, event.movementY);
		renderer.requestRender("camera");
	};

	document.addEventListener("pointerlockchange", () => {
		if (document.pointerLockElement === canvas) {
			window.addEventListener("mousemove", onMouseMove);
		} else {
			window.removeEventListener("mousemove", onMouseMove);
		}
	});

	let accumulator = 0;
	const fixedTimeStep = 1 / 60; // 60Hz fixed update

	renderer.on("tick", ({ deltaTime }) => {
		// If no keys are active, we don't need to simulate movement.
		// Simply discard accumulated time or keep the fractional remainder.
		if (keys.size === 0) {
			accumulator %= fixedTimeStep;
			return;
		}

		// Convert ms to seconds and cap to avoid spiral of death
		accumulator += Math.min(deltaTime / 1000, 0.25);

		let cameraMoved = false;
		const stepDistance = camera.moveSpeed * fixedTimeStep;

		while (accumulator >= fixedTimeStep) {
			let forwardInput = 0;
			let rightInput = 0;
			let upInput = 0;

			if (keys.has("KeyW")) {
				forwardInput += 1;
			}
			if (keys.has("KeyS")) {
				forwardInput -= 1;
			}
			if (keys.has("KeyA")) {
				rightInput -= 1;
			}
			if (keys.has("KeyD")) {
				rightInput += 1;
			}

			const planarLength = Math.hypot(forwardInput, rightInput);
			if (planarLength > 0) {
				camera.moveForward((forwardInput / planarLength) * stepDistance);
				camera.moveRight((rightInput / planarLength) * stepDistance);
				cameraMoved = true;
			}

			if (keys.has("Space")) {
				upInput += 1;
			}

			if (keys.has("ShiftLeft")) {
				upInput -= 1;
			}

			if (upInput !== 0) {
				camera.moveUp(upInput * stepDistance);
				cameraMoved = true;
			}

			accumulator -= fixedTimeStep;
		}

		if (cameraMoved) {
			renderer.requestRender("camera");
		}
	});
}

/**
 * Setup real-time click interaction
 */
export function setupInteraction(renderer: Renderer, scene: Scene, camera: Camera) {
	const interaction = new InteractionController();

	for (const meshInstance of scene.getMeshInstances()) {
		interaction.interactables.set(meshInstance, {});
	}
	interaction.attach(scene, camera);

	// Listen for selection events
	interaction.on("selectionChanged", ({ node }) => {
		if (node instanceof MeshInstance) {
			Logger.info(`Clicked mesh: ${node.name}`);
		}
	});

	// Handle mouse input
	const canvas = renderer.canvas;
	const handlePointer = (type: "down" | "move" | "up", e: MouseEvent) => {
		interaction.updatePointer({
			type,
			screenX: e.clientX,
			screenY: e.clientY,
			viewportWidth: canvas.clientWidth,
			viewportHeight: canvas.clientHeight,
		});
	};

	canvas.addEventListener("mousedown", (e) => handlePointer("down", e));
	canvas.addEventListener("mousemove", (e) => handlePointer("move", e));
	canvas.addEventListener("mouseup", (e) => handlePointer("up", e));
}

init().catch((error) => {
	Logger.error(["Failed to initialize scene:", error], {
		scope: "Main",
	});
});
