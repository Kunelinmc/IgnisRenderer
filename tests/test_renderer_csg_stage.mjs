import assert from "node:assert/strict";
import { Camera } from "../src/cameras/Camera.ts";
import { Renderer } from "../src/renderers/Renderer.ts";
import { CSG } from "../src/csg/CSGBuilder.ts";
import { CSGMeshInstance } from "../src/meshes/CSGMeshInstance.ts";
import { MeshFactory } from "../src/meshes/MeshFactory.ts";
import { Material } from "../src/materials/Material.ts";

class StubBackend {
	constructor() {
		this.type = "stub";
		this.capabilities = {
			sh: false,
			shadows: false,
			reflection: false,
			environment: false,
			ssao: false,
			taa: false,
			ssr: false,
			volumetric: false,
			fog: false,
		};
		this.frameScheduling = "on-demand";
		this.beginFrameCount = 0;
	}

	async init() {}

	resize() {}

	getAttachments(width, height) {
		return {
			width,
			height,
			pixels: new Uint8ClampedArray(width * height * 4),
			depthBuffer: new Float32Array(width * height),
			normalBuffer: new Float32Array(width * height * 3),
		};
	}

	beginFrame() {
		this.beginFrameCount++;
	}

	executePass() {}

	endFrame() {}
}

class CountingCSGMeshInstance extends CSGMeshInstance {
	constructor(params) {
		super(params);
		this.flushCount = 0;
	}

	flushCSG(options) {
		this.flushCount++;
		return super.flushCSG(options);
	}
}

function createOperands() {
	const left = MeshFactory.createBox(
		{ x: -0.25, y: 0, z: 0 },
		2,
		2,
		2,
		new Material({ name: "RendererCSGLeft" })
	);
	const right = MeshFactory.createBox(
		{ x: 0.3, y: 0, z: 0 },
		2,
		2,
		2,
		new Material({ name: "RendererCSGRight" })
	);
	return { left, right };
}

async function run() {
	const originalWindow = globalThis.window;
	const originalRAF = globalThis.requestAnimationFrame;

	try {
		globalThis.window = { devicePixelRatio: 1 };
		globalThis.requestAnimationFrame = () => 0;

		const backend = new StubBackend();
		const camera = new Camera();
		camera.position.set(0, 0, 6);
		const canvas = {
			width: 320,
			height: 180,
			getBoundingClientRect() {
				return { width: 320, height: 180 };
			},
		};
		const renderer = new Renderer(backend, canvas, camera);
		renderer.features.enableShadows = false;
		renderer.features.enableReflection = false;
		renderer.features.enableGamma = false;

		const { left, right } = createOperands();
		const dirty = renderer.scene.add(
			new CountingCSGMeshInstance({
				graph: CSG.from(left).union(right),
				name: "dirty-csg",
			})
		);
		const clean = renderer.scene.add(
			new CountingCSGMeshInstance({
				graph: CSG.from(left).subtract(right),
				name: "clean-csg",
			})
		);
		clean.flushCSG();
		clean.flushCount = 0;

		await renderer.renderScene(0);
		assert.equal(backend.beginFrameCount, 1);
		assert.equal(dirty.isCSGDirty, false);
		assert.equal(dirty.flushCount, 1);
		assert.equal(clean.flushCount, 0);

		dirty.markCSGDirty();
		await renderer.renderScene(16);
		assert.equal(backend.beginFrameCount, 2);
		assert.equal(dirty.flushCount, 2);

		console.log("Renderer CSG stage tests passed");
	} finally {
		globalThis.window = originalWindow;
		globalThis.requestAnimationFrame = originalRAF;
	}
}

await run();
