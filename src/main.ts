import {
	FPSCamera,
	Camera,
	Platform,
	Renderer,
	Scene,
	SoftwareBackend,
	WebGPUBackend,
	WebGLBackend,
	GLTFLoader,
	Quaternion,
	InteractionManager,
	MeshInstance,
	DirectionalLight,
	Logger,
	AmbientLight,
} from "./index";

async function init() {
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
	camera.position.set(5, 4, 7);
	camera.updateMatrices();

	const scene = new Scene();

	// Set spatial index mode to hybrid
	scene.spatialIndexMode = "hybrid";

	scene.add(camera);

	// Load model from public/test.glb
	const gltfLoader = new GLTFLoader();
	gltfLoader.on("progress", (event) => {
		const percent = (event.loaded / event.total) * 100;
		Logger.info(`Loading model: ${percent.toFixed(2)}% (${event.loaded}/${event.total} bytes)`);
	});
	const model = await gltfLoader.load("test.glb");
	scene.add(model);

	// Add some light
	scene.add(
		new AmbientLight({
			intensity: 0.1,
			color: {
				r: 250,
				g: 254,
				b: 255,
			},
		}),
	);

	const sun = new DirectionalLight({
		intensity: 5.0,
		direction: { x: 0.5, y: -1, z: -1 },
	});
	scene.add(sun);

	const shadowMap = scene.shadows.createSingle({
		size: 4096,
		sampling: {
			filterMode: "pcf",
			radius: 0.1,
			samples: 16,
			searchSamples: 8,
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

	const bootstrap = await createRenderer(canvas, camera, scene);
	canvas = bootstrap.canvas;
	const renderer = bootstrap.renderer;

	await renderer.warmup({
		includeEnvironmentIBLBake: false,
	});

	renderer.updateSH();
	renderer.requestRender("unknown");

	bindControls(canvas, camera, renderer, scene);
	setupInteraction(renderer, scene, camera);

	window.addEventListener("resize", () => {
		renderer.resizeCanvas();
		renderer.requestRender("camera");
	});
}

async function createRenderer(
	canvas: HTMLCanvasElement,
	camera: Camera,
	scene: Scene,
): Promise<{ canvas: HTMLCanvasElement; renderer: Renderer }> {
	let renderer: Renderer;

	if (Platform.hasWebGPU()) {
		renderer = new Renderer(new WebGPUBackend(), canvas, camera);
		renderer.setScene(scene);
		renderer.features.enableFXAA = true;
		renderer.features.enableOIT = true;

		try {
			await renderer.init();

			Logger.info("Using WebGPU backend");
			return { canvas, renderer };
		} catch (error) {
			Logger.warn(["WebGPU initialization failed, trying WebGL.", error], {
				scope: "Main",
			});
		}
	}

	try {
		renderer = new Renderer(new WebGLBackend(), canvas, camera);
		renderer.setScene(scene);
		renderer.features.enableFXAA = true;
		renderer.features.enableOIT = true;

		await renderer.init();

		Logger.info("Using WebGL backend");
		return { canvas, renderer };
	} catch (error) {
		Logger.warn(["WebGL initialization failed, fallback to software.", error], {
			scope: "Main",
		});
	}

	renderer = new Renderer(
		new SoftwareBackend({
			rasterMode: "tile",
		}),
		canvas,
		camera,
	);
	renderer.setScene(scene);

	await renderer.init();

	Logger.info("Using software backend");
	return { canvas, renderer };
}

function bindControls(
	canvas: HTMLCanvasElement,
	camera: FPSCamera,
	renderer: Renderer,
	scene: Scene,
): void {
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

	window.addEventListener("mousemove", (event: MouseEvent) => {
		if (document.pointerLockElement === canvas) {
			if (event.movementX === 0 && event.movementY === 0) return;
			camera.rotate(event.movementX, event.movementY);
			renderer.requestRender("camera");
		}
	});

	let accumulator = 0;
	const fixedTimeStep = 1 / 60; // 60Hz fixed update

	renderer.on("tick", async ({ deltaTime }) => {
		// Convert ms to seconds and cap to avoid spiral of death
		accumulator += Math.min(deltaTime / 1000, 0.25);

		const moveSpeed = camera.moveSpeed;
		let cameraMoved = false;

		while (accumulator >= fixedTimeStep) {
			const qYaw = Quaternion.fromAxisAngle({ x: 0, y: 1, z: 0 }, camera.yaw);
			const forward = qYaw.rotatePoint({ x: 0, y: 0, z: -1 });
			const right = qYaw.rotatePoint({ x: 1, y: 0, z: 0 });

			const moveDir = { x: 0, y: 0, z: 0 };

			if (keys.has("KeyW")) {
				moveDir.x += forward.x;
				moveDir.z += forward.z;
			}
			if (keys.has("KeyS")) {
				moveDir.x -= forward.x;
				moveDir.z -= forward.z;
			}
			if (keys.has("KeyA")) {
				moveDir.x -= right.x;
				moveDir.z -= right.z;
			}
			if (keys.has("KeyD")) {
				moveDir.x += right.x;
				moveDir.z += right.z;
			}

			// Normalize move direction if moving
			const length = Math.sqrt(moveDir.x * moveDir.x + moveDir.z * moveDir.z);
			if (length > 0) {
				moveDir.x = (moveDir.x / length) * moveSpeed;
				moveDir.z = (moveDir.z / length) * moveSpeed;
			}

			if (keys.has("Space")) {
				moveDir.y = moveSpeed;
			}

			if (keys.has("ShiftLeft")) {
				moveDir.y = -moveSpeed;
			}

			if (moveDir.x !== 0 || moveDir.y !== 0 || moveDir.z !== 0) {
				camera.position.x += moveDir.x * fixedTimeStep;
				camera.position.y += moveDir.y * fixedTimeStep;
				camera.position.z += moveDir.z * fixedTimeStep;
				cameraMoved = true;
			}

			accumulator -= fixedTimeStep;
		}

		if (cameraMoved) {
			camera.updateMatrices();
			renderer.requestRender("camera");
		}
	});
}

/**
 * Setup real-time click interaction
 */
export function setupInteraction(renderer: Renderer, scene: Scene, camera: Camera) {
	const interaction = new InteractionManager();
	interaction.attach(renderer, scene, camera);

	// Listen for selection events
	interaction.on("selectionChanged", ({ node }) => {
		if (node instanceof MeshInstance) {
			Logger.info(`Clicked mesh: ${node.name}`);
		}
	});

	// Handle mouse input
	const canvas = renderer.canvas;
	const handlePointer = (type: any, e: MouseEvent) => {
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
