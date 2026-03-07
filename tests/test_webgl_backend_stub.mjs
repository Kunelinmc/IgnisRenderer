import assert from "node:assert/strict";
import { WebGLBackend } from "../src/renderers/webgl/WebGLBackend.ts";

const STUB_MESSAGE = "WebGLBackend is a stub and is not implemented yet";

async function run() {
	const backend = new WebGLBackend();

	assert.equal(backend.type, "webgl");
	assert.equal(backend.frameScheduling, "on-demand");
	assert.deepEqual(backend.passExecutors, {});
	assert.deepEqual(backend.capabilities, {
		sh: false,
		shadows: false,
		reflection: false,
		skybox: false,
		ssao: false,
		taa: false,
		ssr: false,
		volumetric: false,
	});

	backend.setRenderer({
		canvas: { width: 1, height: 1 },
		camera: {},
		scene: { lights: [] },
		features: { enableShadows: false },
		warnOnce() {},
	});
	backend.resize(800, 600);

	assert.deepEqual(backend.getAttachments(320, 240), {
		width: 320,
		height: 240,
	});

	await assert.rejects(() => backend.init({}), /WebGLBackend is a stub/);
	assert.throws(() => backend.beginFrame({}), /WebGLBackend is a stub/);
	assert.throws(() => backend.executePass({}, {}), /WebGLBackend is a stub/);
	assert.throws(() => backend.endFrame(), /WebGLBackend is a stub/);

	console.log("WebGL backend stub tests passed");
}

await run();
