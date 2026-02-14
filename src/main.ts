import {
	Renderer,
	Scene,
	AmbientLight,
	DirectionalLight,
	OrbitCamera,
	GLTFLoader,
	PhongMaterial,
	ModelFactory,
	UnlitMaterial,
} from "./index";

async function init() {
	const canvas = document.getElementById("canvas3d") as HTMLCanvasElement;

	const camera = new OrbitCamera({ x: 0, y: 0, z: 0 }, 500);

	const renderer = new Renderer(canvas, camera);
	const scene = new Scene();

	renderer.scene = scene;

	renderer.params.enableLighting = true;
	renderer.params.enableSH = true;
	renderer.params.enableShadows = true;
	renderer.params.enableReflection = true;
	renderer.params.enableVolumetric = false;

	scene.addLight(
		new AmbientLight({
			color: { r: 255, g: 255, b: 255 },
			intensity: 0.3,
		})
	);

	scene.addLight(
		new DirectionalLight({
			color: { r: 255, g: 255, b: 255 },
			dir: { x: -1, y: -1, z: -1 },
			intensity: 1.4,
		})
	);

	const loader = new GLTFLoader();

	loader.on(
		"progress",
		(event: { loaded: number; total: number; url: string }) => {
			const { loaded, total, url } = event;
			if (!total) return;
			const percent = ((loaded / total) * 100).toFixed(1);
			console.log(`[Loading] ${url}: ${percent}%`);
		}
	);

	const model = await loader.load("./assets/truck.glb");

	const targetRadius = 120;
	const scale = targetRadius / model.boundingSphere.radius;

	model.transform.scale.set(scale, scale, scale);
	model.transform.position.y = -model.getWorldBoundingBox().min.y;

	scene.addModel(model);

	const plane = ModelFactory.createPlane(
		{
			x: 0,
			y: 0,
			z: 0,
		},
		400,
		400,
		new PhongMaterial({
			diffuse: { r: 255, g: 255, b: 255 },
			doubleSided: true,
			mirrorPlane: {
				normal: { x: 0, y: 1, z: 0 },
				constant: 0,
			},
			reflectivity: 0.5,
			reflectionBlur: 2,
		})
	);

	scene.addModel(plane);

	renderer.updateSH();
	renderer.requestRender();
	renderer.init();

	let isDragging = false;
	let lastMouse = { x: 0, y: 0 };

	canvas.addEventListener("mousedown", (e) => {
		isDragging = true;
		lastMouse = { x: e.clientX, y: e.clientY };
	});

	window.addEventListener("mousemove", (e) => {
		if (!isDragging) return;
		camera.rotate(e.clientX - lastMouse.x, e.clientY - lastMouse.y);
		lastMouse = { x: e.clientX, y: e.clientY };
		renderer.requestRender();
	});

	window.addEventListener("mouseup", () => {
		isDragging = false;
	});

	canvas.addEventListener(
		"wheel",
		(e) => {
			e.preventDefault();
			camera.zoom(e.deltaY);
			renderer.requestRender();
		},
		{ passive: false }
	);

	canvas.addEventListener(
		"touchstart",
		(e) => {
			if (e.touches.length !== 1) return;
			isDragging = true;
			lastMouse = { x: e.touches[0].clientX, y: e.touches[0].clientY };
		},
		{ passive: false }
	);

	canvas.addEventListener(
		"touchmove",
		(e) => {
			if (!isDragging || e.touches.length !== 1) return;
			const touch = e.touches[0];
			camera.rotate(touch.clientX - lastMouse.x, touch.clientY - lastMouse.y);
			lastMouse = { x: touch.clientX, y: touch.clientY };
			renderer.requestRender();
		},
		{ passive: false }
	);

	canvas.addEventListener("touchend", () => {
		isDragging = false;
	});

	window.addEventListener("resize", () => {
		renderer.resizeCanvas();
	});
}

init().catch((error) => {
	console.error("Failed to initialize scene:", error);
});
