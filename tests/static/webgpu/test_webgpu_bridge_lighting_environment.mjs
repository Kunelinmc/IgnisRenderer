import assert from "node:assert/strict";
import { DirectionalLight } from "../../../src/lights/DirectionalLight.ts";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import { collectWebGPULightingCatalog } from "../../../src/backends/webgpu/lights.ts";

const light = new DirectionalLight();
const prepared = {
	light,
	lightId: light.id,
	definition: {
		bias: {},
		filterMode: "pcf",
		sampling: { quality: "medium" },
		strength: 1,
		projection: {},
	},
	requestedTechnique: "single",
	effectiveTechnique: "single",
	requestedCascadeCount: 1,
	effectiveCascadeCount: 1,
	requestedResolution: 512,
	effectiveResolution: 512,
	sampling: { quality: "medium" },
	requestedFilterMode: "pcf",
	effectiveFilterMode: "pcf",
	priority: 0,
	cost: 1,
	score: 1,
	slices: [{
		index: 0,
		resolution: 512,
		view: Matrix4.identity(),
		projection: Matrix4.identity(),
		viewProjection: Matrix4.identity(),
		lightDirection: { x: 0, y: -1, z: 0 },
		splitNear: 0,
		splitFar: 1,
	}],
};
const plan = {
	revision: 1,
	lights: [prepared],
	diagnostics: [],
	hasRasterWork: true,
	hasTransmissionWork: false,
};
const catalog = collectWebGPULightingCatalog([light], true, false, true, plan);
assert.equal(catalog.lights[0].shadow.enabled, true);
assert.equal(catalog.lights[0].shadow.storageMode, "atlas");

const pagedCatalog = collectWebGPULightingCatalog([light], true, false, true, plan, {
	prepared,
	settings: {
		pageSize: 128,
		pageGridSize: 128,
		physicalPageCount: 2048,
		maxPagesPerFrame: 256,
		cacheFrames: 120,
		feedbackMode: "conservative",
	},
});
assert.equal(pagedCatalog.lights[0].shadow.storageMode, "paged");
assert.equal(pagedCatalog.lights[0].shadow.pagedPageGridSize, 128);
assert.equal(pagedCatalog.lights[0].shadow.pagedPhysicalAtlasSize, 5888);
console.log("WebGPU lighting bridge reads prepared shadow data");
