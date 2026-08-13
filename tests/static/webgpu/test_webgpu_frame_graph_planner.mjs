import assert from "node:assert/strict";
import { WebGPUShadowFrameModule } from "../../../src/backends/webgpu/rendergraph/WebGPUShadowFrameModule.ts";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";

const module = new WebGPUShadowFrameModule({}, {});
const pass = { stage: "shadow", executor: "backend", enabled: true, dependsOn: [] };
const prepared = { light: { id: "sun", type: "directional" }, lightId: "sun", definition: {}, effectiveTechnique: "cascaded", storage: "paged", pagedSettings: { feedbackMode: "screen-feedback" }, slices: [{ viewProjection: Matrix4.identity() }] };
const context = { scene: { decalPackets: [] }, shadowPlan: { lights: [prepared], jobs: [{ technique: "paged", lightIndex: 0, sliceIndices: [0] }], diagnostics: [], revision: 1, hasRasterWork: true, hasTransmissionWork: false, hasPagedWork: true } };
const contributions = module.planStage({
	pass,
	context,
	state: { sceneTargetMode: "mrt" },
	messages: {},
});
assert.ok(contributions[0].nodes.some((node) => node.kind === "paged-shadow-page-mark"));
console.log("WebGPU frame graph uses prepared shadow plans");
