import {
	Renderer,
	Scene,
	AmbientLight,
	DirectionalLight,
	OrbitCamera,
	GLTFLoader,
	PhongMaterial,
	ModelFactory,
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
	renderer.params.enableVolumetric = false;
	renderer.params.enableReflection = true;

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
			intensity: 2,
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

	const duck = await loader.load("./assets/duck.glb");

	const targetRadius = 120;
	const scale = targetRadius / duck.boundingSphere.radius;

	duck.transform.scale.set(scale, scale, scale);

	const distance = targetRadius / 2;

	const localBottom = duck.getWorldBoundingBox().min.y;

	duck.transform.position.y = -localBottom - distance;

	scene.addModel(duck);

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
				constant: distance,
			},
			reflectivity: 0.5,
			reflectionBlur: 2,
		})
	);

	plane.transform.position.y = -distance;

	scene.addModel(plane);

	renderer.updateSH();
	renderer.invalidate();
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
		renderer.invalidate();
	});

	window.addEventListener("mouseup", () => {
		isDragging = false;
	});

	canvas.addEventListener(
		"wheel",
		(e) => {
			e.preventDefault();
			camera.zoom(e.deltaY);
			renderer.invalidate();
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
			renderer.invalidate();
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
