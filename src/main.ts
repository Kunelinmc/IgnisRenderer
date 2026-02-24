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

	loader.on("progress", (event) => {
		const { loaded, total, url } = event;
		if (!total) return;
		const percent = ((loaded / total) * 100).toFixed(1);
		console.log(`[Loading] ${url}: ${percent}%`);
	});

	const model = await loader.load("./assets/duck.glb");

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
		})
	);

	scene.addModel(plane);

	renderer.updateSH();
	renderer.requestRender();
	renderer.init();

	let isDragging = false;
	let lastMouse = { x: 0, y: 0 };
	let lastPinchDistance = 0;
	let idleTimeout: any = null;
	let isInteracting = false;

	const startInteraction = () => {
		if (!isInteracting) {
			isInteracting = true;
			renderer.params.enableLighting = false;
			renderer.params.enableShadows = false;
			renderer.params.enableReflection = false;
		}
		if (idleTimeout) clearTimeout(idleTimeout);
	};

	const stopInteraction = () => {
		if (idleTimeout) clearTimeout(idleTimeout);
		idleTimeout = setTimeout(() => {
			isInteracting = false;
			renderer.params.enableLighting = true;
			renderer.params.enableShadows = true;
			renderer.params.enableReflection = true;
			renderer.requestRender();
			idleTimeout = null;
		}, 300);
	};

	canvas.addEventListener("mousedown", (e) => {
		startInteraction();
		isDragging = true;
		lastMouse = { x: e.clientX, y: e.clientY };
	});

	window.addEventListener("mousemove", (e) => {
		if (!isDragging) return;
		startInteraction();
		camera.rotate(e.clientX - lastMouse.x, e.clientY - lastMouse.y);
		lastMouse = { x: e.clientX, y: e.clientY };
		renderer.requestRender();
	});

	window.addEventListener("mouseup", () => {
		isDragging = false;
		stopInteraction();
	});

	canvas.addEventListener(
		"wheel",
		(e) => {
			e.preventDefault();
			startInteraction();
			camera.zoom(e.deltaY);
			renderer.requestRender();
			stopInteraction();
		},
		{ passive: false }
	);

	canvas.addEventListener(
		"touchstart",
		(e) => {
			e.preventDefault();
			startInteraction();
			if (e.touches.length === 1) {
				isDragging = true;
				lastMouse = { x: e.touches[0].clientX, y: e.touches[0].clientY };
			} else if (e.touches.length === 2) {
				isDragging = false;
				const dx = e.touches[0].clientX - e.touches[1].clientX;
				const dy = e.touches[0].clientY - e.touches[1].clientY;
				lastPinchDistance = Math.sqrt(dx * dx + dy * dy);
			}
		},
		{ passive: false }
	);

	canvas.addEventListener(
		"touchmove",
		(e) => {
			e.preventDefault();
			startInteraction();
			if (e.touches.length === 1 && isDragging) {
				const touch = e.touches[0];
				camera.rotate(touch.clientX - lastMouse.x, touch.clientY - lastMouse.y);
				lastMouse = { x: touch.clientX, y: touch.clientY };
				renderer.requestRender();
			} else if (e.touches.length === 2) {
				const dx = e.touches[0].clientX - e.touches[1].clientX;
				const dy = e.touches[0].clientY - e.touches[1].clientY;
				const distance = Math.sqrt(dx * dx + dy * dy);

				if (lastPinchDistance > 0) {
					const delta = distance - lastPinchDistance;
					camera.zoom(-delta);
					renderer.requestRender();
				}
				lastPinchDistance = distance;
			}
		},
		{ passive: false }
	);

	canvas.addEventListener("touchend", (e) => {
		isDragging = false;
		lastPinchDistance = 0;
		if (e.touches.length === 1) {
			isDragging = true;
			lastMouse = { x: e.touches[0].clientX, y: e.touches[0].clientY };
		}
		stopInteraction();
	});

	window.addEventListener("resize", () => {
		renderer.resizeCanvas();
	});
}

init().catch((error) => {
	console.error("Failed to initialize scene:", error);
});
