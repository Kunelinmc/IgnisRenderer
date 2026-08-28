import assert from "node:assert/strict";
import { WebGPUShadowFrameModule } from "../../../src/backends/webgpu/rendergraph/WebGPUShadowFrameModule.ts";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";

const pass = { stage: "shadow", executor: "backend", enabled: true, dependsOn: [] };
const prepared = {
	light: { id: "sun", type: "directional" },
	lightId: "sun",
	definition: {},
	effectiveTechnique: "cascaded",
	effectiveFilterMode: "pcf",
	slices: [{ viewProjection: Matrix4.identity() }],
};
const context = {
	scene: { decalPackets: [] },
	shadowPlan: {
		lights: [prepared],
		diagnostics: [],
		revision: 1,
		hasRasterWork: true,
		hasTransmissionWork: false,
	},
};
const pagedFrame = {
	prepared,
	settings: {
		pageSize: 128,
		pageGridSize: 128,
		physicalPageCount: 2048,
		maxPagesPerFrame: 256,
		cacheFrames: 120,
		feedbackMode: "screen-feedback",
	},
};
const disabledModule = new WebGPUShadowFrameModule({
	resolvePagedShadowFrame: () => null,
});
const disabledContributions = disabledModule.planStage({
	pass,
	context,
	state: { sceneTargetMode: "mrt" },
	messages: {},
});
assert.ok(disabledContributions[0].nodes.some((node) => node.kind === "shadow"));
assert.equal(
	disabledContributions[0].nodes.some((node) => node.kind === "paged-shadow-page-mark"),
	false,
);

const enabledModule = new WebGPUShadowFrameModule({
	resolvePagedShadowFrame: () => pagedFrame,
});
const contributions = enabledModule.planStage({
	pass,
	context,
	state: { sceneTargetMode: "mrt" },
	messages: {},
});
assert.ok(contributions[0].nodes.some((node) => node.kind === "paged-shadow-page-mark"));
assert.ok(contributions[0].nodes.some((node) => node.kind === "shadow"));
console.log("WebGPU frame graph uses private paged shadow state");
