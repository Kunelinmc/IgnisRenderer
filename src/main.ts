import {
	AmbientLight,
	MeshFactory,
	OrbitCamera,
	PBRMaterial,
	Platform,
	PointLight,
	Renderer,
	Scene,
	SoftwareBackend,
	WebGPUBackend,
	WebGLBackend,
	Vector3,
} from "./index";

interface RendererBootstrap {
	canvas: HTMLCanvasElement;
	renderer: Renderer;
}

async function init() {
	const platform = Platform.detect();
	if (!platform.isBrowserRuntime) {
		throw new Error(
			`Main entry requires browser runtime, got "${platform.runtime}".`
		);
	}

	const canvasElement = document.getElementById("canvas3d");
	if (!(canvasElement instanceof HTMLCanvasElement)) {
		throw new Error("Missing required canvas element with id \"canvas3d\".");
	}

	let canvas = canvasElement;
	const camera = new OrbitCamera(new Vector3(0, 280, 0), 860);
	camera.phi = Math.PI / 2;
	camera.theta = 0;
	camera.minDistance = 220;
	camera.maxDistance = 2400;
	camera.updatePosition();

	const scene = new Scene();
	scene.add(camera);

	buildCornellBox(scene);

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

function buildCornellBox(scene: Scene): void {
	const roomWidth = 555;
	const roomHeight = 555;
	const roomDepth = 555;
	const halfWidth = roomWidth / 2;
	const halfDepth = roomDepth / 2;

	const whiteWallMaterial = new PBRMaterial({
		albedo: { r: 205, g: 205, b: 205 },
		roughness: 0.96,
		metalness: 0,
		doubleSided: true,
	});
	const leftWallMaterial = new PBRMaterial({
		albedo: { r: 182, g: 43, b: 38 },
		roughness: 0.96,
		metalness: 0,
		doubleSided: true,
	});
	const rightWallMaterial = new PBRMaterial({
		albedo: { r: 54, g: 170, b: 66 },
		roughness: 0.96,
		metalness: 0,
		doubleSided: true,
	});
	const blockMaterial = new PBRMaterial({
		albedo: { r: 220, g: 220, b: 220 },
		roughness: 0.88,
		metalness: 0,
	});

	const floor = MeshFactory.createPlane(
		{ x: 0, y: 0, z: 0 },
		roomWidth,
		roomDepth,
		whiteWallMaterial
	);
	floor.name = "cornell-floor";
	scene.add(floor);

	const ceiling = MeshFactory.createPlane(
		{ x: 0, y: roomHeight, z: 0 },
		roomWidth,
		roomDepth,
		whiteWallMaterial
	);
	ceiling.name = "cornell-ceiling";
	ceiling.setRotationFromEuler(Math.PI, 0, 0);
	scene.add(ceiling);

	const backWall = MeshFactory.createPlane(
		{ x: 0, y: roomHeight / 2, z: -halfDepth },
		roomWidth,
		roomHeight,
		whiteWallMaterial
	);
	backWall.name = "cornell-back-wall";
	backWall.setRotationFromEuler(Math.PI / 2, 0, 0);
	scene.add(backWall);

	const redWall = MeshFactory.createPlane(
		{ x: -halfWidth, y: roomHeight / 2, z: 0 },
		roomHeight,
		roomDepth,
		leftWallMaterial
	);
	redWall.name = "cornell-left-wall-red";
	redWall.setRotationFromEuler(0, 0, -Math.PI / 2);
	scene.add(redWall);

	const greenWall = MeshFactory.createPlane(
		{ x: halfWidth, y: roomHeight / 2, z: 0 },
		roomHeight,
		roomDepth,
		rightWallMaterial
	);
	greenWall.name = "cornell-right-wall-green";
	greenWall.setRotationFromEuler(0, 0, Math.PI / 2);
	scene.add(greenWall);

	const shortBlock = MeshFactory.createBox(
		{ x: -112, y: 82.5, z: 95 },
		165,
		165,
		165,
		blockMaterial
	);
	shortBlock.name = "cornell-short-block";
	shortBlock.setRotationFromEuler(0, (-20 * Math.PI) / 180, 0);
	scene.add(shortBlock);

	const tallBlock = MeshFactory.createBox(
		{ x: 118, y: 165, z: -88 },
		165,
		165,
		330,
		blockMaterial
	);
	tallBlock.name = "cornell-tall-block";
	tallBlock.setRotationFromEuler(0, (16 * Math.PI) / 180, 0);
	scene.add(tallBlock);

	const ceilingLightMaterial = new PBRMaterial({
		albedo: { r: 255, g: 248, b: 225 },
		emissive: { r: 255, g: 245, b: 218 },
		emissiveIntensity: 14,
		roughness: 0.35,
		metalness: 0,
		doubleSided: true,
	});
	const ceilingLightPanel = MeshFactory.createPlane(
		{ x: 0, y: roomHeight - 1, z: 0 },
		130,
		105,
		ceilingLightMaterial
	);
	ceilingLightPanel.name = "cornell-ceiling-light-panel";
	ceilingLightPanel.setRotationFromEuler(Math.PI, 0, 0);
	scene.add(ceilingLightPanel);

	const ceilingLightY = roomHeight - 18;
	const pointLightIntensity = 42000;
	const pointLightRange = 820;
	const warmWhite = { r: 255, g: 244, b: 214 };
	const pointLightOffsets = [
		{ x: -38, z: -27 },
		{ x: 38, z: -27 },
		{ x: -38, z: 27 },
		{ x: 38, z: 27 },
	];

	for (let i = 0; i < pointLightOffsets.length; i++) {
		const offset = pointLightOffsets[i];
		scene.add(
			new PointLight({
				color: warmWhite,
				intensity: pointLightIntensity,
				range: pointLightRange,
				position: {
					x: offset.x,
					y: ceilingLightY,
					z: offset.z,
				},
			})
		);
	}

	scene.add(
		new AmbientLight({
			color: { r: 255, g: 240, b: 220 },
			intensity: 0.03,
		})
	);
}

async function createRenderer(
	canvas: HTMLCanvasElement,
	camera: OrbitCamera,
	scene: Scene
): Promise<RendererBootstrap> {
	if (Platform.hasWebGPU()) {
		const webgpuRenderer = new Renderer(new WebGPUBackend(), canvas, camera);
		webgpuRenderer.setScene(scene);
		webgpuRenderer.features.enableTAA = true;

		try {
			await webgpuRenderer.init();
			console.info("Using WebGPU backend");
			return {
				canvas,
				renderer: webgpuRenderer,
			};
		} catch (error) {
			console.warn("WebGPU initialization failed, trying WebGL.", error);
		}
	}

	try {
		const webglRenderer = new Renderer(new WebGLBackend(), canvas, camera);
		webglRenderer.setScene(scene);
		await webglRenderer.init();
		webglRenderer.features.enableTAA = true;
		console.info("Using WebGL backend");
		return {
			canvas,
			renderer: webglRenderer,
		};
	} catch (error) {
		console.warn(
			"WebGL initialization failed, falling back to software.",
			error
		);
	}

	const softwareRenderer = new Renderer(new SoftwareBackend(), canvas, camera);
	softwareRenderer.setScene(scene);
	await softwareRenderer.init();
	console.info("Using software backend");

	return {
		canvas,
		renderer: softwareRenderer,
	};
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
