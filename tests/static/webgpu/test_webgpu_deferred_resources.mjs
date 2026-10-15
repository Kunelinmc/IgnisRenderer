import assert from "node:assert/strict";

import { WebGPUDeferredResources } from
	"../../../src/backends/webgpu/WebGPUDeferredResources.ts";
import { TextureFormat } from "../../../src/core/TextureFormat.ts";

import { FakeWebGPUBackend } from "../../helpers/fakes.mjs";

const backend = new FakeWebGPUBackend();
const layouts = {
	gbufferWriteBindGroupLayout: {},
	gbufferReadBindGroupLayout: {},
	decalBindGroupLayout: {},
	decalOutputBindGroupLayout: {},
	decalBatchBindGroupLayout: {},
	deferredUnusedBindGroupLayout: {},
	decalPipelineLayout: {},
	decalBatchPipelineLayout: {},
};
const resources = new WebGPUDeferredResources(
	backend,
	layouts,
	async () => ({ id: "deferred-lighting" }),
);

const first = resources.getDeferredPlaceholderTextures();
const reused = resources.getDeferredPlaceholderTextures();
assert.strictEqual(first, reused);
assert.deepEqual(
	[first.rgba16Float, first.rgba8Unorm, first.rgba16Uint].map(
		(texture) => [texture.width, texture.height, texture.format]
	),
	[
		[1, 1, TextureFormat.RGBA16Float],
		[1, 1, TextureFormat.RGBA8Unorm],
		[1, 1, TextureFormat.RGBA16Uint],
	]
);

resources.onShaderRuntimeChanged();
assert.equal(first.rgba16Float.destroyed, true);
assert.equal(first.rgba8Unorm.destroyed, true);
assert.equal(first.rgba16Uint.destroyed, true);
const rebuilt = resources.getDeferredPlaceholderTextures();
assert.notStrictEqual(rebuilt.rgba16Float, first.rgba16Float);

resources.destroy();
assert.equal(rebuilt.rgba16Float.destroyed, true);

const lighting = new WebGPUDeferredResources(backend, layouts);
const [pipeline, concurrent] = await Promise.all([
	lighting.getDeferredLightingPipeline(),
	lighting.getDeferredLightingPipeline(),
]);
assert.strictEqual(concurrent, pipeline);
assert.equal(backend.shaderModules.length, 1, "concurrent lighting requests must compile once");
assert.equal(backend.pipelines.length, 1);
assert.ok(pipeline.desc.fragment.module.desc.sourceMap.segments.some(segment =>
	segment.sourcePath === "./webgpu/common/shadowVisibility.wgsl"));
assert.strictEqual(await lighting.getDeferredLightingPipeline(), pipeline);
lighting.onShaderRuntimeChanged();
assert.equal(pipeline.destroyed, true);
assert.equal(pipeline.desc.fragment.module.destroyed, true);
const refreshed = await lighting.getDeferredLightingPipeline();
assert.notStrictEqual(refreshed, pipeline);
lighting.destroy();
assert.equal(refreshed.destroyed, true);
await assert.rejects(lighting.getDeferredLightingPipeline(), /destroyed/);

const pendingBackend = new FakeWebGPUBackend();
const pendingResources = new WebGPUDeferredResources(pendingBackend, layouts);
const create = pendingBackend.createPipeline.bind(pendingBackend);
let entered;
let release;
let createCalls = 0;
const started = new Promise(resolve => { entered = resolve; });
const gate = new Promise(resolve => { release = resolve; });
pendingBackend.createPipeline = async descriptor => {
	const pipeline = await create(descriptor);
	if (++createCalls === 1) {
		entered();
		await gate;
	}
	return pipeline;
};
const stale = pendingResources.getDeferredLightingPipeline();
await started;
pendingResources.onShaderRuntimeChanged();
const fresh = pendingResources.getDeferredLightingPipeline();
release();
const [recovered, current] = await Promise.all([stale, fresh]);
assert.strictEqual(recovered, current);
assert.equal(createCalls, 2, "invalidated compilation must join the fresh request");
assert.equal(pendingBackend.pipelines[0].destroyed, true);
assert.equal(pendingBackend.shaderModules[0].destroyed, true);
pendingResources.destroy();
assert.equal(current.destroyed, true);

console.log("WebGPU deferred resource lifecycle tests passed");
