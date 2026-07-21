import assert from "node:assert/strict";

import {
	WEBGPU_FRAME_GRAPH_RESOURCES,
	collectActiveWebGPUFrameGraphResources,
	collectWebGPUFrameGraphResourceCatalog,
} from "../../../src/backends/webgpu/rendergraph/WebGPUFrameGraphResourceCatalog.ts";

const resources = collectActiveWebGPUFrameGraphResources({
	sceneColor: {},
	depth: {},
	postPing: {},
	postPong: null,
	hiZ: {},
	gAlbedoAlpha: {},
	gNormalRoughMetal: null,
	gEmissiveOcclusion: null,
	gMotionDepth: null,
	gSpecular: null,
	gCoatSheen: null,
	gSheenReflectance: null,
	gMaterialExt0: null,
	gMaterialExt1: null,
	gMaterialExt2: null,
	gMaterialExt3: null,
	oitAccum: null,
	oitReveal: null,
	oitSceneColorCopy: null,
	transmissionSceneColorCopy: null,
	transmissionLighting: null,
	gTransmissionSurface0: null,
	gTransmissionSurface1: null,
	gTransmissionSurface2: null,
	transmissionDepth: null,
	planarReflectionMask: {},
}, null);

assert.ok(resources.includes(WEBGPU_FRAME_GRAPH_RESOURCES.canvasColor));
assert.ok(resources.includes(WEBGPU_FRAME_GRAPH_RESOURCES.frameColor));
assert.ok(resources.includes(WEBGPU_FRAME_GRAPH_RESOURCES.postPing));
assert.ok(resources.includes(WEBGPU_FRAME_GRAPH_RESOURCES.frameHiZ));
assert.ok(resources.includes(WEBGPU_FRAME_GRAPH_RESOURCES.gbufferAlbedoAlpha));
assert.ok(resources.includes(WEBGPU_FRAME_GRAPH_RESOURCES.planarReflectionMask));
assert.equal(new Set(Object.values(WEBGPU_FRAME_GRAPH_RESOURCES)).size,
	Object.keys(WEBGPU_FRAME_GRAPH_RESOURCES).length);

const sceneColor = {
	format: "rgba16float",
	width: 128,
	height: 64,
};
const msaaColor = {
	format: "rgba16float",
	width: 128,
	height: 64,
};
const resolver = new Map();
const catalog = collectWebGPUFrameGraphResourceCatalog(
	{ sceneColorMain: sceneColor },
	{ sceneColorMain: msaaColor },
	128,
	64,
	8,
	resolver,
);
const sceneDescriptor = catalog.resources.find((entry) =>
	entry.id === WEBGPU_FRAME_GRAPH_RESOURCES.frameColor);
const msaaDescriptor = catalog.resources.find((entry) =>
	entry.id === WEBGPU_FRAME_GRAPH_RESOURCES.msaaColor);
assert.equal(sceneDescriptor.kind, "texture");
assert.equal(sceneDescriptor.width, 128);
assert.equal(msaaDescriptor.sampleCount, 8);
assert.equal(
	resolver.get(`webgpu:${WEBGPU_FRAME_GRAPH_RESOURCES.frameColor}`),
	sceneColor,
);
assert.ok(catalog.bindings.some((entry) =>
	entry.resourceId === WEBGPU_FRAME_GRAPH_RESOURCES.frameColor &&
	entry.kind === "texture"));

console.log("WebGPU frame resource catalog tests passed");
