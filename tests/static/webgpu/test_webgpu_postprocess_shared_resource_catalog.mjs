import assert from "node:assert/strict";

import { POST_PROCESS_SHARED_RESOURCE_IDS } from "../../../src/postprocess/executionDeclarations.ts";
import { WEBGPU_FRAME_GRAPH_RESOURCES } from "../../../src/backends/webgpu/rendergraph/WebGPUFrameGraphResourceCatalog.ts";
import {
	getWebGPUPostProcessSharedResourceDescriptor,
	listWebGPUPostProcessSharedResourceDescriptors,
} from "../../../src/backends/webgpu/rendergraph/WebGPUPostProcessSharedResourceCatalog.ts";

const ids = POST_PROCESS_SHARED_RESOURCE_IDS;
const graph = WEBGPU_FRAME_GRAPH_RESOURCES;
const expectedGraphResources = new Map([
	[ids.frameHiZ, graph.frameHiZ],
	[ids.planarReflectionMask, graph.planarReflectionMask],
	[ids.transmissionSceneColor, graph.transmissionSceneColorCopy],
	[ids.transmissionLighting, graph.transmissionLighting],
	[ids.transmissionSurface1, graph.transmissionSurface1],
	[ids.transmissionSurface2, graph.transmissionSurface2],
]);

assert.equal(listWebGPUPostProcessSharedResourceDescriptors().length, 6);
for (const [id, graphResourceId] of expectedGraphResources) {
	const descriptor = getWebGPUPostProcessSharedResourceDescriptor(id);
	assert.ok(descriptor);
	assert.equal(descriptor.graphResourceId, graphResourceId);
}
assert.equal(getWebGPUPostProcessSharedResourceDescriptor("custom:unknown"), null);

const hiZ = { id: "hiz" };
const mask = { id: "mask" };
const targets = { hiZ, planarReflectionMask: mask };
const hiZDescriptor = getWebGPUPostProcessSharedResourceDescriptor(ids.frameHiZ);
assert.equal(hiZDescriptor.isAllocated(targets), true);
assert.equal(
	hiZDescriptor.resolveTexture({ targets, isHiZReady: false }),
	null,
);
assert.strictEqual(
	hiZDescriptor.resolveTexture({ targets, isHiZReady: true }),
	hiZ,
);
const maskDescriptor = getWebGPUPostProcessSharedResourceDescriptor(
	ids.planarReflectionMask,
);
assert.equal(maskDescriptor.allocateWhenOptional, true);
assert.strictEqual(
	maskDescriptor.resolveTexture({ targets, isHiZReady: false }),
	mask,
);

console.log("WebGPU post-process shared-resource catalog tests passed");
