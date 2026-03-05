import {
	Renderer,
	Scene,
	AmbientLight,
	DirectionalLight,
	OrbitCamera,
	GLTFLoader,
	PBRMaterial,
	ModelFactory,
} from "./index";
import { SoftwareBackend } from "./core/backend/SoftwareBackend";
import { WebGPUBackend } from "./core/backend/WebGPUBackend";

interface RendererBootstrap {
	canvas: HTMLCanvasElement;
	renderer: Renderer;
}

async function init() {
	let canvas = document.getElementById("canvas3d") as HTMLCanvasElement;
	const camera = new OrbitCamera({ x: 0, y: 0, z: 0 }, 500);
	const scene = new Scene();

	scene.addLight(
		new AmbientLight({
			color: { r: 255, g: 255, b: 255 },
			intensity: 0.5,
		})
	);

	scene.addLight(
		new DirectionalLight({
			color: { r: 255, g: 255, b: 255 },
			dir: { x: -1, y: -1, z: -1 },
			intensity: 2.5,
		})
	);

	const loader = new GLTFLoader();
	const model = await loader.load("./assets/duck.glb");

	const targetRadius = 120;
	const scale = targetRadius / model.boundingSphere.radius;
	model.transform.scale.set(scale, scale, scale);
	model.transform.position.y = -model.getWorldBoundingBox().min.y;
	scene.addModel(model);

	scene.addModel(
		ModelFactory.createPlane(
			{ x: 0, y: 0, z: 0 },
			400,
			400,
			new PBRMaterial({
				albedo: { r: 255, g: 255, b: 255 },
				doubleSided: true,
				mirrorPlane: { normal: { x: 0, y: 1, z: 0 }, constant: 0 },
				reflectivity: 0.5,
			})
		)
	);

	const bootstrap = await createRenderer(canvas, camera, scene);
	canvas = bootstrap.canvas;
	const renderer = bootstrap.renderer;

	renderer.updateSH();
	renderer.requestRender();

	bindControls(canvas, camera, renderer);
	window.addEventListener("resize", () => {
		renderer.resizeCanvas();
		renderer.requestRender();
	});
}

async function createRenderer(
	canvas: HTMLCanvasElement,
	camera: OrbitCamera,
	scene: Scene
): Promise<RendererBootstrap> {
	if (navigator.gpu) {
		const webgpuRenderer = new Renderer(new WebGPUBackend(), canvas, camera);
		webgpuRenderer.scene = scene;
		configureRenderer(webgpuRenderer);

		try {
			await webgpuRenderer.init();
			console.info("Using WebGPU backend");
			return {
				canvas,
				renderer: webgpuRenderer,
			};
		} catch (error) {
			console.warn(
				"WebGPU initialization failed, falling back to software.",
				error
			);
		}
	}

	const softwareRenderer = new Renderer(new SoftwareBackend(), canvas, camera);
	softwareRenderer.scene = scene;
	configureRenderer(softwareRenderer);
	await softwareRenderer.init();
	console.info("Using software backend");

	return {
		canvas,
		renderer: softwareRenderer,
	};
}

function configureRenderer(renderer: Renderer): void {
	renderer.features.enableLighting = true;
	renderer.features.enableGamma = true;

	if (renderer.backendType === "webgpu") {
		renderer.features.enableSH = true;
		renderer.features.enableShadows = true;
		renderer.features.enableReflection = false;
		renderer.features.enableSkybox = true;
		renderer.features.enableSSAO = false;
		renderer.features.enableSSR = false;
		renderer.features.enableVolumetric = false;
		return;
	}

	renderer.features.enableSH = true;
	renderer.features.enableShadows = true;
	renderer.features.enableReflection = true;
}

function bindControls(
	canvas: HTMLCanvasElement,
	camera: OrbitCamera,
	renderer: Renderer
): void {
	let isDragging = false;
	let lastMouse = { x: 0, y: 0 };

	canvas.addEventListener("mousedown", (event) => {
		isDragging = true;
		lastMouse = { x: event.clientX, y: event.clientY };
	});

	window.addEventListener("mousemove", (event) => {
		if (!isDragging) return;
		camera.rotate(event.clientX - lastMouse.x, event.clientY - lastMouse.y);
		lastMouse = { x: event.clientX, y: event.clientY };
		renderer.requestRender();
	});

	window.addEventListener("mouseup", () => {
		isDragging = false;
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

	canvas.addEventListener(
		"touchstart",
		(event) => {
			if (event.touches.length !== 1) return;
			isDragging = true;
			lastMouse = {
				x: event.touches[0].clientX,
				y: event.touches[0].clientY,
			};
		},
		{ passive: false }
	);

	canvas.addEventListener(
		"touchmove",
		(event) => {
			if (!isDragging || event.touches.length !== 1) return;
			const touch = event.touches[0];
			camera.rotate(touch.clientX - lastMouse.x, touch.clientY - lastMouse.y);
			lastMouse = { x: touch.clientX, y: touch.clientY };
			renderer.requestRender();
		},
		{ passive: false }
	);

	canvas.addEventListener("touchend", () => {
		isDragging = false;
	});
}

init().catch((error) => {
	console.error("Failed to initialize scene:", error);
});
