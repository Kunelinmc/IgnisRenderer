import assert from "node:assert/strict";
import { Camera } from "../src/cameras/Camera.ts";
import { Material } from "../src/materials/Material.ts";
import { LODMeshInstance } from "../src/meshes/LODMeshInstance.ts";
import { MeshFactory } from "../src/meshes/MeshFactory.ts";
import { Renderer } from "../src/renderers/Renderer.ts";
import {
	installNoopPostProcessSupport,
} from "./helpers/postprocess.mjs";

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
		installNoopPostProcessSupport(
			this,
			"stub"
		);
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

function createLODLevels() {
	const high = MeshFactory.createBox(
		{ x: 0, y: 0, z: 0 },
		2,
		2,
		2,
		new Material({ name: "LODHigh" })
	).mesh;
	const medium = MeshFactory.createBox(
		{ x: 0, y: 0, z: 0 },
		1.4,
		1.4,
		1.4,
		new Material({ name: "LODMedium" })
	).mesh;
	const low = MeshFactory.createPlane(
		{ x: 0, y: 0, z: 0 },
		1,
		1,
		new Material({ name: "LODLow" })
	).mesh;
	return { high, medium, low };
}

function testLODSelectionAndHysteresis() {
	const { high, medium, low } = createLODLevels();
	const lod = new LODMeshInstance({
		name: "lodSelection",
		levels: [
			{
				mesh: high,
				distance: 6,
			},
			{
				mesh: medium,
				distance: 16,
			},
			{
				mesh: low,
				distance: Number.POSITIVE_INFINITY,
			},
		],
		hysteresis: 1,
	});

	assert.equal(lod.activeLevelIndex, 0);
	assert.equal(lod.mesh, high);

	const switchedAtBoundary = lod.updateLODByDistance(7, {
		notifyScene: false,
	});
	assert.equal(switchedAtBoundary, false);
	assert.equal(lod.activeLevelIndex, 0);

	const switchedFarther = lod.updateLODByDistance(7.1, {
		notifyScene: false,
	});
	assert.equal(switchedFarther, true);
	assert.equal(lod.activeLevelIndex, 1);
	assert.equal(lod.mesh, medium);

	const switchedNearBoundary = lod.updateLODByDistance(5.5, {
		notifyScene: false,
	});
	assert.equal(switchedNearBoundary, false);
	assert.equal(lod.activeLevelIndex, 1);

	const switchedNearer = lod.updateLODByDistance(4.9, {
		notifyScene: false,
	});
	assert.equal(switchedNearer, true);
	assert.equal(lod.activeLevelIndex, 0);
	assert.equal(lod.mesh, high);

	const switchedToLowest = lod.updateLODByDistance(96, {
		notifyScene: false,
	});
	assert.equal(switchedToLowest, true);
	assert.equal(lod.activeLevelIndex, 2);
	assert.equal(lod.mesh, low);
}

async function testRendererResolvesLODStage() {
	const originalWindow = globalThis.window;
	const originalRAF = globalThis.requestAnimationFrame;

	try {
		globalThis.window = { devicePixelRatio: 1 };
		globalThis.requestAnimationFrame = () => 0;

		const { high, low } = createLODLevels();
		const backend = new StubBackend();
		const camera = new Camera();
		camera.position.set(0, 0, 0);
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
		renderer.postProcess.getPass("gamma")?.disable();

		const lodMesh = renderer.scene.add(
			new LODMeshInstance({
				name: "lodRendererStage",
				levels: [
					{
						mesh: high,
						distance: 4,
					},
					{
						mesh: low,
						distance: Number.POSITIVE_INFINITY,
					},
				],
			})
		);
		lodMesh.position.z = -10;

		await renderer.renderScene(0);
		assert.equal(backend.beginFrameCount, 1);
		assert.equal(lodMesh.activeLevelIndex, 1);
		assert.equal(lodMesh.mesh, low);

		camera.position.z = -9;
		renderer.requestRender("camera");
		await renderer.renderScene(16);
		assert.equal(backend.beginFrameCount, 2);
		assert.equal(lodMesh.activeLevelIndex, 0);
		assert.equal(lodMesh.mesh, high);
	} finally {
		globalThis.window = originalWindow;
		globalThis.requestAnimationFrame = originalRAF;
	}
}

async function run() {
	testLODSelectionAndHysteresis();
	await testRendererResolvesLODStage();
	console.log("LOD mesh instance tests passed");
}

await run();
