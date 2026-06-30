import { Pane } from "tweakpane";

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
	WebGLBackend,
	WebGPUBackend,
	Vector3,
	Logger,
	type RGB,
} from "../../src/index";

import { CustomGaussianBlurPass } from "./CustomGaussianBlurPass";

// Helper cast function for Tweakpane
interface TweakpaneBinding {
	on: (
		eventName: "change" | "click",
		handler: (event: { value: unknown }) => void
	) => TweakpaneBinding;
}

interface TweakpanePane {
	addBinding: (
		target: Record<string, unknown> | object,
		key: string,
		options?: Record<string, unknown>
	) => TweakpaneBinding;
	addButton: (options: Record<string, unknown>) => TweakpaneBinding;
	addFolder: (options: Record<string, unknown>) => TweakpanePane;
	refresh?: () => void;
}

function asTweakpanePane(value: unknown): TweakpanePane {
	return value as TweakpanePane;
}

interface AnimatedObject {
	mesh: any;
	baseY: number;
	spinSpeed: number;
	bobSpeed: number;
	phase: number;
	rotationY: number;
	rotationX: number;
}

async function init() {
	const platform = Platform.detect();

	// Parse URL params for selected backend
	const params = new URLSearchParams(window.location.search);
	const requestedBackend = params.get("backend")?.toLowerCase() || undefined;

	let backend: any;
	let backendLabel = "";

	if (requestedBackend === "webgpu" && platform.hasWebGPU) {
		backend = new WebGPUBackend({ enableDeferredLighting: true });
		backendLabel = "WebGPU";
	} else if (requestedBackend === "webgl" && platform.hasWebGL2) {
		backend = new WebGLBackend();
		backendLabel = "WebGL 2";
	} else if (requestedBackend === "software") {
		backend = new SoftwareBackend({ rasterMode: "tile" });
		backendLabel = "Software (CPU)";
	} else {
		// Auto fallback based on platform support
		if (platform.hasWebGPU) {
			backend = new WebGPUBackend({ enableDeferredLighting: true });
			backendLabel = "WebGPU (Auto)";
		} else if (platform.hasWebGL2) {
			backend = new WebGLBackend();
			backendLabel = "WebGL 2 (Auto)";
		} else {
			backend = new SoftwareBackend({ rasterMode: "tile" });
			backendLabel = "Software (Auto)";
		}
	}

	const canvas = document.getElementById("canvas3d") as HTMLCanvasElement;
	if (!canvas) {
		throw new Error("Missing canvas #canvas3d");
	}

	// Create Scene and Camera
	const scene = new Scene();
	scene.spatialIndexMode = "hybrid";

	const camera = new OrbitCamera(new Vector3(0, 3, 6), 10);
	camera.fov = 45;
	camera.near = 0.1;
	camera.far = 100;
	camera.theta = 0.5;
	camera.phi = 1.2;
	camera.updatePosition();
	scene.add(camera);

	// Add basic lighting
	const ambient = new AmbientLight({ intensity: 0.5 });
	scene.add(ambient);

	const sun = new DirectionalLight({
		intensity: 3.0,
		direction: { x: 0.5, y: -1.0, z: -0.5 },
	});
	scene.add(sun);

	// Configure and bind shadow map to the directional light
	// This ensures shadow functions are compiled in WebGL shaders
	const shadowMap = scene.shadows.createCascaded({
		size: 1024,
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

	// Populate scene with colorful geometric shapes
	const objects: AnimatedObject[] = [];
	const colors: RGB[] = [
		{ r: 255, g: 90, b: 95 },
		{ r: 90, g: 255, b: 120 },
		{ r: 90, g: 150, b: 255 },
		{ r: 255, g: 200, b: 90 },
	];

	for (let i = 0; i < 4; i++) {
		const mat = new PBRMaterial({
			name: `geo-mat-${i}`,
			albedo: colors[i],
			roughness: 0.2,
			metalness: 0.8,
		});

		let mesh: any;
		if (i % 2 === 0) {
			mesh = MeshFactory.createBox({ x: 0, y: 0, z: 0 }, 1.0, 1.0, 1.0, mat);
		} else {
			mesh = MeshFactory.createSphere({ x: 0, y: 0, z: 0 }, 0.6, 24, 12, mat);
		}

		// Arrange them tightly in a circle at the center of the scene
		const angle = (i / 4) * Math.PI * 2;
		const radius = 1.0;
		mesh.position.set(Math.cos(angle) * radius, 1.0, Math.sin(angle) * radius);
		scene.add(mesh);

		objects.push({
			mesh,
			baseY: 1.0,
			spinSpeed: 0.5 + Math.random() * 0.5,
			bobSpeed: 1.0 + Math.random() * 1.0,
			phase: Math.random() * Math.PI * 2,
			rotationY: 0,
			rotationX: 0,
		});
	}

	// Also add a floor plane
	const floorMat = new PBRMaterial({
		name: "floor-mat",
		albedo: { r: 50, g: 50, b: 50 },
		roughness: 0.8,
		metalness: 0.1,
	});
	const floor = MeshFactory.createPlane({ x: 0, y: 0, z: 0 }, 15, 15, floorMat);
	scene.add(floor);

	// Scene is populated entirely with primitive shapes centered at the origin

	// Initialize Renderer
	const renderer = new Renderer({
		backend,
		canvas,
		camera,
	});
	renderer.setScene(scene);
	renderer.features.enableShadows = true;
	await renderer.initialize();

	// Register Custom Gaussian Blur Pass
	const blurPass = new CustomGaussianBlurPass({ enabled: true });
	renderer.postProcess.registerPass(blurPass);

	// Warmup renderer
	await renderer.warmup({ includeCorePasses: true });

	// Start the render loop
	renderer.renderLoop();

	// Tick animation update
	let lastTime = performance.now();
	renderer.on("tick", ({ deltaTime }) => {
		const time = performance.now() / 1000;
		
		// Animate shapes
		for (const obj of objects) {
			obj.rotationY += obj.spinSpeed * (deltaTime / 1000);
			obj.rotationX += obj.spinSpeed * 0.5 * (deltaTime / 1000);
			obj.mesh.setRotationFromEuler(obj.rotationX, obj.rotationY, 0);
			obj.mesh.position.y = obj.baseY + Math.sin(time * obj.bobSpeed + obj.phase) * 0.4;
		}

		scene.syncNodeToECS();
		renderer.requestRender("tick");
	});

	// Window resize handler
	window.addEventListener("resize", () => {
		renderer.resizeCanvas();
		renderer.requestRender("resize");
	});

	// Build Tweakpane UI controls
	const pane = asTweakpanePane(new Pane({ title: "Custom Blur Demo" }));

	const infoFolder = pane.addFolder({ title: "System Info", expanded: true });
	infoFolder.addBinding({ backend: backendLabel }, "backend", {
		label: "Backend",
		disabled: true,
	});

	const backendSelect = {
		active: requestedBackend || (platform.hasWebGPU ? "webgpu" : platform.hasWebGL2 ? "webgl" : "software"),
	};
	infoFolder.addBinding(backendSelect, "active", {
		label: "Switch Backend",
		options: {
			WebGPU: "webgpu",
			WebGL2: "webgl",
			Software: "software",
		},
	}).on("change", (ev) => {
		window.location.search = `?backend=${ev.value}`;
	});

	const blurFolder = pane.addFolder({ title: "Gaussian Blur", expanded: true });

	// Controls for Radius, Sigma and Enabled state
	const blurSettings = {
		enabled: blurPass.enabled,
		radius: 3,
		sigma: 2.0,
	};

	blurFolder.addBinding(blurSettings, "enabled", {
		label: "Enabled",
	}).on("change", (ev) => {
		blurPass.setEnabled(ev.value as boolean);
		renderer.requestRender("blur-toggle");
	});

	blurFolder.addBinding(blurSettings, "radius", {
		label: "Radius",
		min: 1,
		max: 5,
		step: 1,
	}).on("change", (ev) => {
		blurPass.enable({ radius: ev.value as number });
		renderer.requestRender("blur-radius");
	});

	blurFolder.addBinding(blurSettings, "sigma", {
		label: "Sigma",
		min: 0.5,
		max: 5.0,
		step: 0.1,
	}).on("change", (ev) => {
		blurPass.enable({ sigma: ev.value as number });
		renderer.requestRender("blur-sigma");
	});
}

init().catch((err) => {
	console.error("Demo failed to initialize:", err);
});
