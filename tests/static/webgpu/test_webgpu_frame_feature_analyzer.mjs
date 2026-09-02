import assert from "node:assert/strict";

import { PBRMaterial } from "../../../src/materials/PBRMaterial.ts";
import {
	analyzeWebGPUDeferredFeatures,
} from "../../../src/backends/webgpu/rendergraph/WebGPUDeferredFrameModule.ts";
import {
	analyzeWebGPUPostProcessFeatures,
} from "../../../src/backends/webgpu/rendergraph/WebGPUPostProcessFrameModule.ts";
import {
	analyzeWebGPUReflectionFeatures,
} from "../../../src/backends/webgpu/rendergraph/WebGPUReflectionFrameModule.ts";
import {
	analyzeWebGPUTransparency,
} from "../../../src/backends/webgpu/rendergraph/WebGPUTransparencyRuntime.ts";
import {
	analyzeWebGPUVisibilityFeatures,
} from "../../../src/backends/webgpu/rendergraph/WebGPUVisibilityFrameModule.ts";
import { POST_PROCESS_SHARED_RESOURCE_IDS } from "../../../src/postprocess/executionDeclarations.ts";
import { createTestDrawPacket } from "../helpers/drawPacket.mjs";

function createContext() {
	return {
		features: {
			enableOIT: true,
			enableReflection: true,
			enableOcclusionCulling: true,
		},
		scene: {
			decalPackets: [],
			particleSystems: [],
			reflectivePackets: [{}],
			occlusion: { eligibleCandidateCount: 1 },
		},
	};
}

const context = createContext();
const base = analyzeWebGPUDeferredFeatures(context, {
	all: [],
	opaque: [createTestDrawPacket({ material: new PBRMaterial() })],
	transparent: [],
	shadowCasterSubmissions: [],
	shadowTransmitterSubmissions: [],
	reflective: [],
});
assert.equal(base.hasDeferredLightingWork, true);
assert.equal(base.deferredGBufferLayout, "base");

context.scene.decalPackets.push({});
const extended = analyzeWebGPUDeferredFeatures(context, {
	all: [],
	opaque: [createTestDrawPacket({ material: new PBRMaterial() })],
	transparent: [],
	shadowCasterSubmissions: [],
	shadowTransmitterSubmissions: [],
	reflective: [],
});
assert.equal(extended.deferredGBufferLayout, "extended");

let transmissionReads = 0;
const transparency = analyzeWebGPUTransparency(context, [createTestDrawPacket({
	material: {
		get transmissionFactor() {
			transmissionReads++;
			return 1;
		},
	},
}), createTestDrawPacket({ material: {} })]);
assert.equal(transmissionReads, 1);
assert.equal(transparency.transmissionPackets.length, 1);
assert.equal(transparency.oitPackets.length, 1);

assert.equal(analyzeWebGPUReflectionFeatures(context).needsPlanarReflection, true);
assert.equal(analyzeWebGPUVisibilityFeatures(context).needsOcclusionTargets, true);

const colorOnly = analyzeWebGPUPostProcessFeatures([{
	id: "bloom",
	declaration: { color: { access: "read", output: "new-version" } },
}]);
assert.equal(colorOnly.needsPostProcessTargets, true);
assert.equal(colorOnly.needsPostProcessGBuffer, false);

const gBufferConsumer = analyzeWebGPUPostProcessFeatures([{
	id: "ssgi",
	declaration: {
		color: { access: "read", output: "new-version" },
		gBuffer: [{ semantic: "normal", access: "read" }],
	},
}]);
assert.equal(gBufferConsumer.needsPostProcessGBuffer, true);

const sharedConsumers = analyzeWebGPUPostProcessFeatures([{
	id: "shared",
	declaration: {
		color: { access: "read", output: "new-version" },
		shared: [{
			id: POST_PROCESS_SHARED_RESOURCE_IDS.planarReflectionMask,
			access: "read",
			usage: "sampled",
		}, {
			id: "backend:frame-hiz",
			access: "read",
			usage: "sampled",
		}],
	},
}]);
assert.equal(sharedConsumers.needsPlanarReflectionMask, true);
assert.equal(sharedConsumers.needsHiZTarget, true);

console.log("WebGPU feature-local analysis tests passed");
