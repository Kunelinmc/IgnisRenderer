import assert from "node:assert/strict";
import {
	createWebGPUComputeFacade,
	getWebGPUComputeFacadeCacheStats,
	invalidateWebGPUComputeFacade,
	resetWebGPUComputeFacadeCacheForTesting,
	resolveWebGPUComputeFacade,
} from "../../../src/renderers/webgpu/computeFacade.ts";
import { FakeWebGPUBackend } from "../../helpers/fakes.mjs";

async function testFacadeDelegatesAndCaches() {
	resetWebGPUComputeFacadeCacheForTesting();
	assert.equal(getWebGPUComputeFacadeCacheStats().entryCount, 0);
	const backend = new FakeWebGPUBackend();
	const facadeA = createWebGPUComputeFacade(backend);
	const facadeB = createWebGPUComputeFacade(backend);
	assert.equal(facadeA, facadeB);
	assert.equal(getWebGPUComputeFacadeCacheStats().entryCount, 1);

	const sampler = facadeA.createSampler({ label: "sampler" });
	assert.equal(sampler.kind, "sampler");

	const shader = await facadeA.createShaderModule({
		label: "shader",
		code: "shader-code",
	});
	assert.equal(shader.kind, "shader-module");

	const pipeline = facadeA.createComputePipeline({
		label: "pipeline",
		compute: { module: shader, entryPoint: "csMain" },
	});
	assert.equal(pipeline.kind, "compute-pipeline");

	const buffer = facadeA.createBuffer({
		label: "buffer",
		size: 16,
		usage: 0,
	});
	facadeA.writeBuffer(buffer, new Float32Array([1, 2, 3, 4]), 4);
	assert.equal(buffer.lastWrite.offset, 4);

	const texture = facadeA.createTexture({
		label: "texture",
		width: 4,
		height: 4,
		format: "rgba8unorm",
		usage: 0,
	});
	assert.equal(texture.kind, "texture");

	const slotResource = facadeA.resolveTextureForSlot({ id: "source" }, 2);
	assert.equal(slotResource.kind, "slot-texture");
	assert.equal(slotResource.slotIndex, 2);

	facadeA.registerExternalTexture(
		{ id: "tex" },
		{ id: "res" },
		12,
		3
	);
	facadeA.unregisterExternalTexture({ id: "tex" });

	const encoded = facadeA.createCommandEncoder();
	facadeA.submit([encoded.finish()]);

	const textureWithView = {
		width: 1,
		height: 1,
		destroy() {},
		_webgpuTexture: {
			texture: {
				createView(desc) {
					return { kind: "custom-view", desc };
				},
			},
			view: { kind: "default-view" },
		},
	};
	assert.equal(
		facadeA.createTextureView(textureWithView).kind,
		"default-view"
	);
	assert.equal(
		facadeA.createTextureView(textureWithView, { baseMipLevel: 1 }).kind,
		"custom-view"
	);

	const layout = facadeA.createBindGroupLayout({
		label: "layout",
		entries: [],
	});
	assert.equal(layout.kind, "bind-group-layout");
	const pipelineLayout = facadeA.createPipelineLayout({
		label: "pipeline-layout",
		bindGroupLayouts: [],
	});
	assert.equal(pipelineLayout.kind, "pipeline-layout");

	const trackedTexture = { id: "tracked", version: 7 };
	facadeA.registerExternalTexture(trackedTexture, texture, 7, 1);
	facadeA.destroy();
	assert.equal(getWebGPUComputeFacadeCacheStats().entryCount, 0);
	assert.throws(
		() => facadeA.createCommandEncoder(),
		/WebGPU compute facade is destroyed/
	);
	const unregisterCalls = backend.calls.filter(
		([name, value]) =>
			name === "unregisterExternalTexture" && value === trackedTexture
	);
	assert.equal(unregisterCalls.length, 1);
}

function testResolverSupportsRendererAndBackend() {
	const backend = new FakeWebGPUBackend();
	const facade = createWebGPUComputeFacade(backend);
	backend.getComputeFacade = () => facade;

	const fromBackend = resolveWebGPUComputeFacade(backend);
	assert.equal(fromBackend, facade);

	const renderer = { backend };
	const fromRenderer = resolveWebGPUComputeFacade(renderer);
	assert.equal(fromRenderer, facade);
}

function testResolverRejectsNonWebGPUBackend() {
	assert.throws(
		() => resolveWebGPUComputeFacade({ type: "webgl" }),
		/WebGPU compute facade requires WebGPU backend/
	);
}

function testCacheInvalidationRecreatesFacade() {
	resetWebGPUComputeFacadeCacheForTesting();
	const backend = new FakeWebGPUBackend();
	const facadeA = createWebGPUComputeFacade(backend);
	assert.equal(getWebGPUComputeFacadeCacheStats().entryCount, 1);
	invalidateWebGPUComputeFacade(backend);
	assert.equal(getWebGPUComputeFacadeCacheStats().entryCount, 0);
	const facadeB = createWebGPUComputeFacade(backend);
	assert.notEqual(facadeA, facadeB);
	assert.equal(getWebGPUComputeFacadeCacheStats().entryCount, 1);
}

function testResolverRejectsCycles() {
	const cyclicA = {};
	const cyclicB = {};
	cyclicA.backend = cyclicB;
	cyclicB.backend = cyclicA;
	assert.throws(
		() => resolveWebGPUComputeFacade(cyclicA),
		/cyclic source references detected/
	);
}

function testResolverRejectsIncompleteBackendLike() {
	const missingSampler = {
		type: "webgpu",
		createShaderModule: async () => ({}),
		createComputePipeline: () => ({}),
		createBuffer: () => ({ size: 1, destroy() {} }),
		createTexture: () => ({ width: 1, height: 1, destroy() {} }),
		createBindingGroup: () => ({}),
		createTextureView: () => ({}),
		createCommandEncoder: () => ({ finish: () => ({}) }),
		submit() {},
		writeBuffer() {},
		getTextureForSlot: () => ({ width: 1, height: 1, destroy() {} }),
		registerExternalTexture() {},
		unregisterExternalTexture() {},
	};
	assert.throws(
		() => resolveWebGPUComputeFacade(missingSampler),
		/Failed to resolve WebGPU compute facade/
	);

	const missingTextureView = {
		type: "webgpu",
		createSampler: () => ({}),
		createShaderModule: async () => ({}),
		createComputePipeline: () => ({}),
		createBuffer: () => ({ size: 1, destroy() {} }),
		createTexture: () => ({ width: 1, height: 1, destroy() {} }),
		createBindingGroup: () => ({}),
		createCommandEncoder: () => ({ finish: () => ({}) }),
		submit() {},
		writeBuffer() {},
		getTextureForSlot: () => ({ width: 1, height: 1, destroy() {} }),
		registerExternalTexture() {},
		unregisterExternalTexture() {},
	};
	assert.throws(
		() => resolveWebGPUComputeFacade(missingTextureView),
		/Failed to resolve WebGPU compute facade/
	);
}

export async function run() {
	await testFacadeDelegatesAndCaches();
	testResolverSupportsRendererAndBackend();
	testResolverRejectsNonWebGPUBackend();
	testCacheInvalidationRecreatesFacade();
	testResolverRejectsCycles();
	testResolverRejectsIncompleteBackendLike();
	console.log("WebGPU compute facade tests passed");
}

if (import.meta.url === `file://${process.argv[1]}`) {
	await run();
}
