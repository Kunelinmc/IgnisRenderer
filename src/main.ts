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
	screenToWorldRay,
} from "./index";
import { INTERACTION_TRANSIENT_STATE_KEY } from "./pipeline/types";
import type { InteractionTransientState } from "./pipeline/types";

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

	renderer.updateSH();
	renderer.requestRender();

	const selectedEntityIds: number[] = [];

	renderer.registerFrameTransientContributor((context) => {
		const interactionState: InteractionTransientState = {
			selectedEntityIds: [...selectedEntityIds],
			hoveredEntityId: null,
			outline: {
				color: { r: 64, g: 196, b: 255, a: 1 }, // Cyan outline
				thickness: 4,
				opacity: 0.9,
				xray: true,
			},
			gizmo: null,
			dragRect: null,
		};
		context.transient.set(INTERACTION_TRANSIENT_STATE_KEY, interactionState);
	});

	bindControls(canvas, camera, renderer, scene, selectedEntityIds);

	window.addEventListener("resize", () => {
		renderer.resizeCanvas();
		renderer.requestRender();
	});
}

function buildPlayground(scene: Scene): void {
	// Create a huge ground plane
	const groundMaterial = new PBRMaterial({
		albedo: { r: 40, g: 45, b: 50 },
		roughness: 0.8,
		metalness: 0.1,
		doubleSided: true,
	});

	const ground = MeshFactory.createPlane(
		{ x: 0, y: 0, z: 0 },
		1000,
		1000,
		groundMaterial
	);
	ground.name = "ground-plane";
	scene.add(ground);

	// Create some interactive cubes
	const cubeColors = [
		{ r: 255, g: 80, b: 80 }, // Red
		{ r: 80, g: 255, b: 80 }, // Green
		{ r: 80, g: 80, b: 255 }, // Blue
		{ r: 255, g: 255, b: 80 }, // Yellow
		{ r: 255, g: 80, b: 255 }, // Magenta
		{ r: 80, g: 255, b: 255 }, // Cyan
	];

	for (let i = 0; i < 12; i++) {
		const color = cubeColors[i % cubeColors.length];
		const material = new PBRMaterial({
			albedo: color,
			roughness: 0.4,
			metalness: 0.2,
		});

		const size = 50 + Math.random() * 50;
		const x = (Math.random() - 0.5) * 800;
		const z = (Math.random() - 0.5) * 800;
		const y = size / 2; // Bottom on ground

		const cube = MeshFactory.createBox({ x, y, z }, size, size, size, material);
		cube.name = `interactive-cube-${i}`;
		scene.add(cube);
	}

	// Add lights
	scene.add(
		new DirectionalLight({
			color: { r: 255, g: 255, b: 240 },
			intensity: 4,
			direction: { x: -1, y: -2, z: -1 },
		})
	);

	scene.add(
		new AmbientLight({
			color: { r: 200, g: 220, b: 255 },
			intensity: 0.05,
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
			console.warn("WebGPU initialization failed, trying WebGL.", error);
		}
	}

	try {
		const webglRenderer = new Renderer(new WebGLBackend(), canvas, camera);
		webglRenderer.setScene(scene);
		await webglRenderer.init();
		webglRenderer.features.enableTAA = true;
		console.info("Using WebGL backend");
		return { canvas, renderer: webglRenderer };
	} catch (error) {
		console.warn("WebGL initialization failed, fallback to software.", error);
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
	renderer: Renderer,
	scene: Scene,
	selectedEntityIds: number[]
): void {
	let isDraggingCamera = false;
	let draggedNode: any = null;
	let lastMouse = { x: 0, y: 0 };
	let dragPlaneY = 0;
	let dragOffset = new Vector3();

	canvas.addEventListener("mousedown", (event: MouseEvent) => {
		const rect = canvas.getBoundingClientRect();
		const mouseX = event.clientX - rect.left;
		const mouseY = event.clientY - rect.top;

		lastMouse = { x: event.clientX, y: event.clientY };

		// Try picking
		const ray = screenToWorldRay(camera, {
			screenX: mouseX,
			screenY: mouseY,
			viewportWidth: canvas.width / (window.devicePixelRatio || 1),
			viewportHeight: canvas.height / (window.devicePixelRatio || 1),
		});

		const meshInstances = scene.getMeshInstances();
		const spatial = scene.rebuildSpatialIndex(meshInstances);
		const hits = spatial.queryRayDetailed(ray.origin, ray.direction, {
			maxDistance: 5000,
			includeInvisible: false,
		});

		if (hits.length > 0) {
			// Find first interactive cube
			const hit = hits.find((h) =>
				h.meshInstance.name.startsWith("interactive-cube")
			);
			if (hit) {
				const entityId = hit.meshInstance.entityId;
				if (typeof entityId === "number") {
					if (!event.shiftKey) {
						selectedEntityIds.length = 0;
					}
					if (!selectedEntityIds.includes(entityId)) {
						selectedEntityIds.push(entityId);
					}
					renderer.requestRender();
				}

				draggedNode = hit.meshInstance;
				dragPlaneY = draggedNode.position.y;

				// Calculate offset for smoother dragging
				const intersectPoint = intersectRayPlaneY(
					ray.origin,
					ray.direction,
					dragPlaneY
				);
				if (intersectPoint) {
					dragOffset.copy(draggedNode.position).sub(intersectPoint);
				}

				isDraggingCamera = false;
				return;
			}
		}

		if (!event.shiftKey) {
			selectedEntityIds.length = 0;
			renderer.requestRender();
		}
		isDraggingCamera = true;
	});

	window.addEventListener("mousemove", (event: MouseEvent) => {
		const rect = canvas.getBoundingClientRect();
		const mouseX = event.clientX - rect.left;
		const mouseY = event.clientY - rect.top;

		if (draggedNode) {
			const ray = screenToWorldRay(camera, {
				screenX: mouseX,
				screenY: mouseY,
				viewportWidth: canvas.width / (window.devicePixelRatio || 1),
				viewportHeight: canvas.height / (window.devicePixelRatio || 1),
			});

			const intersectPoint = intersectRayPlaneY(
				ray.origin,
				ray.direction,
				dragPlaneY
			);
			if (intersectPoint) {
				draggedNode.position.set(
					intersectPoint.x + dragOffset.x,
					dragPlaneY, // Keep height
					intersectPoint.z + dragOffset.z
				);
				draggedNode.updateLocalMatrix();
				scene.invalidate();
				renderer.requestRender();
			}
			return;
		}

		if (isDraggingCamera) {
			camera.rotate(event.clientX - lastMouse.x, event.clientY - lastMouse.y);
			lastMouse = { x: event.clientX, y: event.clientY };
			renderer.requestRender();
		}
	});

	window.addEventListener("mouseup", () => {
		isDraggingCamera = false;
		draggedNode = null;
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

function intersectRayPlaneY(
	origin: { x: number; y: number; z: number },
	direction: { x: number; y: number; z: number },
	planeY: number
): Vector3 | null {
	if (Math.abs(direction.y) < 1e-6) return null;
	const t = (planeY - origin.y) / direction.y;
	if (t < 0) return null;
	return new Vector3(
		origin.x + t * direction.x,
		planeY,
		origin.z + t * direction.z
	);
}

init().catch((error) => {
	console.error("Failed to initialize scene:", error);
});
