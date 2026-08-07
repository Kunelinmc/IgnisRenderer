import assert from "node:assert/strict";

import { WebGPUDeferredResources } from
	"../../../src/backends/webgpu/WebGPUDeferredResources.ts";
import { TextureFormat } from "../../../src/backends/types.ts";

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

console.log("WebGPU deferred resource lifecycle tests passed");
