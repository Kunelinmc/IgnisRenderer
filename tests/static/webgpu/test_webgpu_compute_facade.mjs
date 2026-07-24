import assert from "node:assert/strict";
import {
	createWebGPUComputeFacade,
	getWebGPUComputeFacadeCacheStats,
	invalidateWebGPUComputeFacade,
	resetWebGPUComputeFacadeCacheForTesting,
	resolveWebGPUComputeFacade,
} from "../../../src/backends/webgpu/computeFacade.ts";
import {
	createRenderBackendExtensionRegistry,
	WEBGPU_COMPUTE_EXTENSION,
} from "../../../src/backends/BackendExtensions.ts";
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

	const pipeline = await facadeA.createComputePipeline({
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
	facadeA.writeTexture(
		texture,
		new Uint8Array(256),
		{ bytesPerRow: 256, rowsPerImage: 1 },
		{ width: 1, height: 1 },
	);
	assert.equal(backend.textureWrites.length, 1);

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

function testResolverSupportsBackendAndFacade() {
	const host = new FakeWebGPUBackend();
	const facade = createWebGPUComputeFacade(host);
	const backend = {
		extensions: createRenderBackendExtensionRegistry([
			{
				id: WEBGPU_COMPUTE_EXTENSION.id,
				insertionPoints: ["application:webgpu-compute"],
				api: facade,
			},
		]),
	};

	const fromBackend = resolveWebGPUComputeFacade(backend);
	assert.equal(fromBackend, facade);

	const fromFacade = resolveWebGPUComputeFacade(facade);
	assert.equal(fromFacade, facade);
}

function testResolverRejectsRendererLikeSource() {
	const backend = new FakeWebGPUBackend();
	assert.throws(
		() => resolveWebGPUComputeFacade({ backend }),
		/neither a WebGPU compute facade nor an IRenderBackend/
	);
}

function testResolverRejectsNonWebGPUBackend() {
	assert.throws(
		() => resolveWebGPUComputeFacade({ type: "webgl" }),
		/neither a WebGPU compute facade nor an IRenderBackend/
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

function testResolverRejectsBackendLikeDuckTyping() {
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
		/neither a WebGPU compute facade nor an IRenderBackend/
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
		/neither a WebGPU compute facade nor an IRenderBackend/
	);
}

export async function run() {
	await testFacadeDelegatesAndCaches();
	testResolverSupportsBackendAndFacade();
	testResolverRejectsRendererLikeSource();
	testResolverRejectsNonWebGPUBackend();
	testCacheInvalidationRecreatesFacade();
	testResolverRejectsBackendLikeDuckTyping();
	console.log("WebGPU compute facade tests passed");
}

if (import.meta.url === `file://${process.argv[1]}`) {
	await run();
}
