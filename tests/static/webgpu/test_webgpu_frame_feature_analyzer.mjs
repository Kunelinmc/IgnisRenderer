import assert from "node:assert/strict";

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
import { AlphaMode, ShadingModel } from "../../../src/materials/Material.ts";
import { PBRMaterial } from "../../../src/materials/PBRMaterial.ts";
import { createTestDrawPacket } from "../helpers/drawPacket.mjs";

function createContext() {
	return {
		features: {
			enableOIT: true,
			enableReflection: true,
			enableOcclusionCulling: true,
		},
		scene: {
			particleSystems: [],
			reflectivePackets: [{}],
			occlusion: { eligibleCandidateCount: 1 },
		},
	};
}

const context = createContext();
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

const material = new PBRMaterial({ transmissionFactor: 0.75, alphaMode: AlphaMode.Blend });
const packet = createTestDrawPacket({ material });
material.shading = ShadingModel.Unlit;
assert.deepEqual(analyzeWebGPUTransparency(context, [packet]).oitPackets, [packet]);
assert.equal(analyzeWebGPUTransparency(context, [packet]).transmissionPackets.length, 0);
material.shading = ShadingModel.PBR;
assert.deepEqual(analyzeWebGPUTransparency(context, [packet]).transmissionPackets, [packet]);
assert.equal(material.transmissionFactor, 0.75);

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
