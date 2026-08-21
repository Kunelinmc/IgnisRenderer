import assert from "node:assert/strict";
import { SobelNormalMapper } from "../../../src/addons/SobelNormalMapper.ts";
import { Texture } from "../../../src/core/Texture.ts";
import { ShaderSource } from "../../../src/shaders/ShaderSource.ts";
import { createWebGPUComputeFacade } from "../../../src/backends/webgpu/ComputeFacade.ts";

import { FakeWebGPUBackend, FakeRenderer } from "../../helpers/fakes.mjs";

function createRenderer(backend) {
	backend.computeFacade = createWebGPUComputeFacade(backend);
	return new FakeRenderer(backend);
}

function getLastBufferWriteValues(backend) {
	const write = backend.writeCalls
		.filter((call) => call[0] === "writeBuffer")
		.at(-1);
	assert.ok(write, "Expected a buffer write.");
	return Array.from(write[2]);
}

async function testInitUpdateAndInvalidationFlow() {
	const backend = new FakeWebGPUBackend();
	const renderer = createRenderer(backend);
	const source = new Texture({
		data: new Uint8ClampedArray(4),
		width: 15,
		height: 9,
		colorSpace: "sRGB",
	});
	const mapper = new SobelNormalMapper(source);

	await mapper.init(renderer);
	assert.equal(mapper.isInitialized, true);
	assert.equal(backend.registeredExternalTextures.length, 1);
	assert.equal(backend.registeredExternalTextures[0].texture, mapper.normalMap);

	assert.equal(mapper.update(), true);
	assert.equal(backend.submits, 1);
	assert.deepEqual(backend.dispatches[0], [2, 2, 1]);
	assert.equal(backend.createBindingGroupCalls, 1);
	assert.equal(mapper.update(), false);
	assert.equal(backend.submits, 1);

	source.markNeedsUpdate();
	assert.equal(mapper.update(), true);
	assert.equal(backend.submits, 2);

	mapper.strength = 3.5;
	assert.equal(mapper.update(), true);
	assert.equal(backend.submits, 3);
	assert.equal(backend.bufferWrites.length >= 3, true);
}

async function testAttachRunsOnPostAnimationAndDetachStops() {
	const backend = new FakeWebGPUBackend();
	const renderer = createRenderer(backend);
	const source = new Texture({
		data: new Uint8ClampedArray(4),
		width: 8,
		height: 8,
		colorSpace: "sRGB",
	});
	const mapper = new SobelNormalMapper(source);

	await mapper.attach(renderer);
	assert.equal(mapper.isAttached, true);
	assert.equal(renderer.requestRenderCalls, 1);

	source.markNeedsUpdate();
	renderer.emit("postanimation", {
		now: 16,
		deltaTime: 16,
		scene: null,
		transient: new Map(),
	});
	assert.equal(backend.submits, 1);

	mapper.detach();
	assert.equal(mapper.isAttached, false);
	source.markNeedsUpdate();
	renderer.emit("postanimation", {
		now: 32,
		deltaTime: 16,
		scene: null,
		transient: new Map(),
	});
	assert.equal(backend.submits, 1);
}

async function testDestroyReleasesResources() {
	const backend = new FakeWebGPUBackend();
	const renderer = createRenderer(backend);
	const source = new Texture({
		data: new Uint8ClampedArray(4),
		width: 4,
		height: 4,
		colorSpace: "sRGB",
	});
	const mapper = new SobelNormalMapper(source);

	await mapper.init(renderer);
	mapper.update();
	mapper.destroy();
	await Promise.resolve();

	assert.equal(mapper.isInitialized, false);
	assert.equal(backend.unregisteredExternalTextures.length >= 1, true);
	assert.equal(backend.destroyCalls >= 4, true);
	assert.equal(mapper.update(), false);
}

async function testInjectedComputeFacadeSupportsNonWebGPUBackend() {
	const computeHost = new FakeWebGPUBackend();
	const computeFacade = createWebGPUComputeFacade(computeHost);
	const renderer = new FakeRenderer({ type: "webgl" });
	const source = new Texture({
		data: new Uint8ClampedArray(4),
		width: 6,
		height: 5,
		colorSpace: "sRGB",
	});
	const mapper = new SobelNormalMapper(source, {
		computeFacade,
		strength: 2,
		invertX: true,
	});

	await mapper.init(renderer);
	assert.equal(mapper.isInitialized, true);
	assert.equal(computeHost.registeredExternalTextures.length, 1);
	assert.equal(computeHost.registeredExternalTextures[0].texture, mapper.normalMap);

	assert.equal(mapper.update(), true);
	assert.equal(computeHost.submits, 1);
	assert.deepEqual(computeHost.dispatches[0], [1, 1, 1]);
	assert.equal(computeHost.bufferWrites.length, 1);

	mapper.destroy();
	assert.equal(computeHost.unregisteredExternalTextures.length >= 1, true);
}

async function testHeightSourceControlsUniformAndInvalidation() {
	const backend = new FakeWebGPUBackend();
	const renderer = createRenderer(backend);
	const source = new Texture({
		data: new Uint8ClampedArray(4),
		width: 8,
		height: 8,
		colorSpace: "sRGB",
	});
	const mapper = new SobelNormalMapper(source, {
		heightSource: "blue",
	});

	assert.equal(mapper.heightSource, "blue");

	await mapper.init(renderer);
	assert.equal(mapper.update(), true);
	assert.deepEqual(getLastBufferWriteValues(backend), [2, 1, 1, 3]);

	assert.equal(mapper.update(), false);
	mapper.heightSource = "alpha";
	assert.equal(mapper.update(), true);
	assert.deepEqual(getLastBufferWriteValues(backend), [2, 1, 1, 4]);

	mapper.heightSource = "red";
	assert.equal(mapper.update(), true);
	assert.deepEqual(getLastBufferWriteValues(backend), [2, 1, 1, 1]);
}

async function testSobelShaderSupportsChannelHeightSources() {
	ShaderSource.clearCache();
	const shader = await ShaderSource.load("webgpu.postprocess.sobelNormal");

	assert.match(shader.source.code, /heightSource: f32/);
	assert.match(shader.source.code, /return color\.r;/);
	assert.match(shader.source.code, /return color\.g;/);
	assert.match(shader.source.code, /return color\.b;/);
	assert.match(shader.source.code, /return color\.a;/);
	assert.match(
		shader.source.code,
		/dot\(color\.rgb, vec3<f32>\(0\.2126, 0\.7152, 0\.0722\)\)/
	);
}

await testInitUpdateAndInvalidationFlow();
await testAttachRunsOnPostAnimationAndDetachStops();
await testDestroyReleasesResources();
await testInjectedComputeFacadeSupportsNonWebGPUBackend();
await testHeightSourceControlsUniformAndInvalidation();
await testSobelShaderSupportsChannelHeightSources();
console.log("Sobel normal mapper tests passed");
