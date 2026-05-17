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
	InteractionManager,
	MeshInstance,
	Logger,
	AreaLight,
	DirectionalLight,
} from "./index";

function colorTemperatureToRGB(kelvin) {
	const temp = kelvin / 100;
	let r, g, b;

	// Red
	if (temp <= 66) {
		r = 255;
	} else {
		r = 329.698727446 * Math.pow(temp - 60, -0.1332047592);
		r = Math.max(0, Math.min(255, r));
	}

	// Green
	if (temp <= 66) {
		g = 99.4708025861 * Math.log(temp) - 161.1195681661;
	} else {
		g = 288.1221695283 * Math.pow(temp - 60, -0.0755148492);
	}
	g = Math.max(0, Math.min(255, g));

	// Blue
	if (temp >= 66) {
		b = 255;
	} else if (temp <= 19) {
		b = 0;
	} else {
		b = 138.5177312231 * Math.log(temp - 10) - 305.0447927307;
		b = Math.max(0, Math.min(255, b));
	}

	return {
		r: Math.round(r),
		g: Math.round(g),
		b: Math.round(b),
	};
}

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

	// Load model from assets/model.glb
	const gltfLoader = new GLTFLoader();
	gltfLoader.on("progress", (event) => {
		const percent = (event.loaded / event.total) * 100;
		Logger.info(`Loading model: ${percent.toFixed(2)}% (${event.loaded}/${event.total} bytes)`);
	});
	const model = await gltfLoader.load("assets/model.glb");
	scene.add(model);

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

	const bootstrap = await createRenderer(canvas, camera, scene);
	canvas = bootstrap.canvas;
	const renderer = bootstrap.renderer;

	await renderer.warmup({
		includeEnvironmentIBLBake: false,
	});

	renderer.updateSH();
	renderer.requestRender("unknown");

	bindControls(canvas, camera, renderer);
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
		renderer.postProcess.enable("fxaa");
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
		renderer.postProcess.enable("fxaa");
		renderer.postProcess.enable("ssgi");
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

		let cameraMoved = false;

		while (accumulator >= fixedTimeStep) {
			const stepDistance = camera.moveSpeed * fixedTimeStep;
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
