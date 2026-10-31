import assert from "node:assert/strict";

import { Camera } from "../../../src/cameras/Camera.ts";
import { Texture } from "../../../src/core/Texture.ts";
import { BasicMaterial } from "../../../src/materials/BasicMaterial.ts";
import { MeshAsset } from "../../../src/meshes/MeshAsset.ts";
import { MeshInstance } from "../../../src/meshes/MeshInstance.ts";
import { Renderer } from "../../../src/rendering/Renderer.ts";
import { SoftwareBackend } from "../../../src/backends/software/SoftwareBackend.ts";

const WIDTH = 256;
const HEIGHT = 128;

function createScene({ background = false, secondMesh = false } = {}) {
	const camera = new Camera();
	camera.position.set(0, 0, 4);
	const backend = new SoftwareBackend();
	const renderer = new Renderer({
		width: WIDTH, height: HEIGHT,
		getBoundingClientRect: () => ({ width: WIDTH, height: HEIGHT }),
	}, backend, camera);
	renderer.features.enableLighting = false;
	renderer.features.enableReflection = false;
	renderer.features.enableEnvironment = background;
	if (background) {
		renderer.scene.environment.backgroundTexture = new Texture({
			width: 1, height: 1, data: new Uint8Array([0, 0, 255, 255]),
		});
	}
	const meshes = (secondMesh ? [-1.5, 1.5] : [-1.5]).map((x) => {
		const material = new BasicMaterial({
			diffuse: { r: 255, g: 0, b: 0 }, doubleSided: true,
		});
		const mesh = MeshAsset.fromFaces([{
			material,
			vertices: [
				{ x: -0.25, y: -0.25, z: 0, u: 0, v: 0, normal: { x: 0, y: 0, z: 1 } },
				{ x: 0.25, y: -0.25, z: 0, u: 1, v: 0, normal: { x: 0, y: 0, z: 1 } },
				{ x: 0, y: 0.25, z: 0, u: 0.5, v: 1, normal: { x: 0, y: 0, z: 1 } },
			],
		}]);
		const instance = renderer.scene.add(new MeshInstance({ mesh }));
		instance.position.x = x;
		return instance;
	});
	const pixels = () => backend.getAttachments({ width: WIDTH, height: HEIGHT }).pixels.slice();
	return { renderer, backend, meshes, pixels };
}

async function testPostProcessReplayPreservesScenePixels() {
	const { renderer, pixels } = createScene();
	try {
		await renderer.renderFrame(0);
		const initial = pixels();
		assert.ok(initial.some((value, index) => index % 4 === 0 && value > 0));
		renderer.requestRender("postfx");
		const replay = await renderer.renderFrame(16);
		assert.equal(replay.incremental.plan.forceFullFrame, true);
		assert.equal(replay.incremental.plan.firstPass, null);
		assert.equal(replay.incremental.plan.postProcessStartPass, null);
		assert.equal(replay.incremental.plan.temporalHistoryReset, true);
		assert.deepEqual(pixels(), initial,
			"postprocess replay must reconstruct scene input rather than clear it or convert it twice");

		renderer.postProcess.getPass("gamma").disable();
		await renderer.renderFrame(32);
		const changed = pixels();
		assert.notDeepEqual(changed, initial);
		renderer.requestRender("unknown");
		await renderer.renderFrame(48);
		assert.deepEqual(pixels(), changed, "pass toggles must match a complete replay");
	} finally {
		await renderer.destroy();
	}
}

async function testGlobalExposureAndLocalEditMatchFullReplay() {
	const { renderer, meshes, pixels } = createScene({ secondMesh: true });
	try {
		await renderer.renderFrame(0);
		const initial = pixels();
		meshes[0].mesh.primitives[0].material.opacity = 0.5;
		await renderer.setDisplayOutput({ exposure: 0.2 });
		const frame = await renderer.renderFrame(16);
		const changed = pixels();
		assert.notDeepEqual(changed, initial);
		renderer.requestRender("unknown");
		await renderer.renderFrame(32);
		assert.deepEqual(changed, pixels(),
			"a local dirty rectangle must not restrict a global exposure change");
		assert.equal(frame.incremental.plan.dirtyAreaRatio, 1);
	} finally {
		await renderer.destroy();
	}
}

async function testPartialSceneReconstructsExposedEnvironment() {
	const { renderer, meshes, pixels } = createScene({ background: true });
	try {
		await renderer.renderFrame(0);
		const initial = pixels();
		meshes[0].position.x += 0.3;
		renderer.requestRender("transform");
		const frame = await renderer.renderFrame(16);
		assert.equal(frame.incremental.plan.forceFullFrame, false);
		assert.ok(frame.incremental.plan.dirtyAreaRatio < 1);
		const partial = pixels();
		assert.deepEqual(partial.slice(0, 4), initial.slice(0, 4),
			"background outside the dirty region must remain unchanged");
		renderer.requestRender("unknown");
		await renderer.renderFrame(32);
		assert.deepEqual(partial, pixels(),
			"newly exposed background must match a complete frame");
	} finally {
		await renderer.destroy();
	}
}

const originalWindow = globalThis.window;
try {
	globalThis.window = { devicePixelRatio: 1 };
	await testPostProcessReplayPreservesScenePixels();
	await testGlobalExposureAndLocalEditMatchFullReplay();
	await testPartialSceneReconstructsExposedEnvironment();
	console.log("Renderer incremental pixel tests passed");
} finally {
	globalThis.window = originalWindow;
}
