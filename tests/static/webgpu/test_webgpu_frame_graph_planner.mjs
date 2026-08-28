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

const shadowRenderer = {
	renderShadows: async () => {},
};
const shadowModule = new WebGPUShadowFrameModule(shadowRenderer);

const contributions = shadowModule.planStage({
	pass,
	context,
	state: { sceneTargetMode: "mrt" },
	messages: {},
});

assert.equal(contributions.length, 1);
assert.equal(contributions[0].lane, "geometry");
assert.ok(contributions[0].nodes.some((node) => node.kind === "shadow"));
const shadowNode = contributions[0].nodes.find((node) => node.kind === "shadow");
assert.ok(shadowNode.writes.some((write) => write.id === "shadow-atlas"));
assert.ok(shadowNode.writes.some((write) => write.id === "shadow-transmittance-atlas"));

const noRasterContext = {
	...context,
	shadowPlan: {
		...context.shadowPlan,
		hasRasterWork: false,
	},
};
const noWorkContributions = shadowModule.planStage({
	pass,
	context: noRasterContext,
	state: { sceneTargetMode: "mrt" },
	messages: {},
});
assert.equal(noWorkContributions.length, 0);

const nonShadowContributions = shadowModule.planStage({
	pass: { stage: "main-opaque", executor: "backend", enabled: true, dependsOn: [] },
	context,
	state: { sceneTargetMode: "mrt" },
	messages: {},
});
assert.equal(nonShadowContributions.length, 0);

console.log("WebGPU frame graph shadow planner tests passed");
