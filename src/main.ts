import {
	AmbientLight,
	DirectionalLight,
	MeshFactory,
	OrbitCamera,
	PBRMaterial,
	Platform,
	Renderer,
	Scene,
	SoftwareBackend,
	WebGPUBackend,
	WebGLBackend,
	Vector3,
	HDRLoader,
} from "./index";

async function init() {
	const platform = Platform.detect();
	if (!platform.isBrowserRuntime) {
		throw new Error(
			`Main entry requires browser runtime, got "${platform.runtime}".`
		);
	}

	const canvasElement = document.getElementById("canvas3d");
	if (!(canvasElement instanceof HTMLCanvasElement)) {
		throw new Error('Missing required canvas element with id "canvas3d".');
	}

	let canvas = canvasElement;
	const camera = new OrbitCamera(new Vector3(0, 0, 0), 1200);
	camera.phi = Math.PI / 4;
	camera.theta = Math.PI / 4;
	camera.minDistance = 100;
	camera.maxDistance = 5000;
	camera.updatePosition();

	const scene = new Scene();
	scene.add(camera);

	buildPlayground(scene);

	const bootstrap = await createRenderer(canvas, camera, scene);
	canvas = bootstrap.canvas;
	const renderer = bootstrap.renderer;

	// Load environment map for skybox and light probe baking
	const hdrLoader = new HDRLoader();
	const skybox = await hdrLoader.load("puresky_1k.hdr");
	scene.skybox = skybox;

	// Bakes environment IBL data from the skybox.
	await renderer.warmup({ includeEnvironmentIBLBake: true });

	renderer.updateSH();
	renderer.requestRender();

	bindControls(canvas, camera, renderer);

	window.addEventListener("resize", () => {
		renderer.resizeCanvas();
		renderer.requestRender();
	});
}

function buildPlayground(scene: Scene): void {
	// Create a grounded plane and a single metallic cube to show off the renderer's capabilities, along with a directional light and some ambient light for basic illumination.
	const ground = MeshFactory.createPlane(
		{ x: 0, y: 0, z: 0 },
		2000,
		2000,
		new PBRMaterial({
			albedo: { r: 60, g: 65, b: 70 },
			roughness: 0.2,
			metalness: 0.1,
			doubleSided: true,
		})
	);
	ground.name = "ground-plane";
	scene.add(ground);

	// Create a single prominent metallic cube
	const size = 150;
	const cube = MeshFactory.createBox(
		{ x: 0, y: size / 2, z: 0 },
		size,
		size,
		size,
		new PBRMaterial({
			albedo: { r: 255, g: 200, b: 50 },
			roughness: 0.1,
			metalness: 1.0,
		})
	);
	cube.name = "metallic-cube";
	scene.add(cube);

	// Add lights
	scene.add(
		new DirectionalLight({
			color: { r: 255, g: 255, b: 245 },
			intensity: 4,
			direction: { x: -1, y: -2, z: -1 },
		})
	);

	scene.add(
		new AmbientLight({
			color: { r: 180, g: 200, b: 255 },
			intensity: 0.1,
		})
	);
}

async function createRenderer(
	canvas: HTMLCanvasElement,
	camera: OrbitCamera,
	scene: Scene
): Promise<{ canvas: HTMLCanvasElement; renderer: Renderer }> {
	if (Platform.hasWebGPU()) {
		const webgpuRenderer = new Renderer(new WebGPUBackend(), canvas, camera);
		webgpuRenderer.setScene(scene);
		webgpuRenderer.features.enableTAA = true;

		try {
			await webgpuRenderer.init();

			console.info("Using WebGPU backend");
			return { canvas, renderer: webgpuRenderer };
		} catch (error) {
			console.warn(["WebGPU initialization failed, trying WebGL.", error], {
				scope: "Main",
			});
		}
	}

	try {
		const webglRenderer = new Renderer(new WebGLBackend(), canvas, camera);
		webglRenderer.setScene(scene);
		webglRenderer.features.enableTAA = true;

		await webglRenderer.init();

		console.info("Using WebGL backend");
		return { canvas, renderer: webglRenderer };
	} catch (error) {
		console.warn(["WebGL initialization failed, fallback to software.", error], {
			scope: "Main",
		});
	}

	const softwareRenderer = new Renderer(new SoftwareBackend(), canvas, camera);
	softwareRenderer.setScene(scene);

	await softwareRenderer.init();

	console.info("Using software backend");
	return { canvas, renderer: softwareRenderer };
}

function bindControls(
	canvas: HTMLCanvasElement,
	camera: OrbitCamera,
	renderer: Renderer
): void {
	let isDraggingCamera = false;
	let lastMouse = { x: 0, y: 0 };

	canvas.addEventListener("mousedown", (event: MouseEvent) => {
		isDraggingCamera = true;
		lastMouse = { x: event.clientX, y: event.clientY };
	});

	window.addEventListener("mousemove", (event: MouseEvent) => {
		if (isDraggingCamera) {
			camera.rotate(event.clientX - lastMouse.x, event.clientY - lastMouse.y);
			lastMouse = { x: event.clientX, y: event.clientY };
			renderer.requestRender();
		}
	});

	window.addEventListener("mouseup", () => {
		isDraggingCamera = false;
	});

	canvas.addEventListener(
		"wheel",
		(event) => {
			event.preventDefault();
			camera.zoom(event.deltaY);
			renderer.requestRender();
		},
		{ passive: false }
	);
}

init().catch((error) => {
	console.error(["Failed to initialize scene:", error], {
		scope: "Main",
	});
});
