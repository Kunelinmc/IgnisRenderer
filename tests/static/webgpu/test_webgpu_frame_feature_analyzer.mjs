import assert from "node:assert/strict";

import { PBRMaterial } from "../../../src/materials/PBRMaterial.ts";
import { WebGPUFrameFeatureAnalyzer } from "../../../src/renderers/webgpu/rendergraph/WebGPUFrameFeatureAnalyzer.ts";
import { createResolvedPostProcess } from "../../helpers/postprocess.mjs";

function createContext() {
	return {
		features: {
			enableOIT: true,
			enableReflection: true,
			enableOcclusionCulling: true,
		},
		postProcess: createResolvedPostProcess({ ssr: true }, "webgpu"),
		scene: {
			opaquePackets: [{ material: new PBRMaterial({ anisotropyStrength: 1 }) }],
			transparentPackets: [{ material: {} }],
			particleSystems: [],
			reflectivePackets: [{}],
			occlusion: { eligibleCandidateCount: 1 },
		},
	};
}

const analysis = new WebGPUFrameFeatureAnalyzer().analyze(createContext());
assert.equal(analysis.hasDeferredLightingWork, true);
assert.equal(analysis.oitRequested, true);
assert.equal(analysis.hasOITWork, true);
assert.equal(analysis.needsPlanarReflection, true);
assert.equal(analysis.needsPlanarReflectionMask, true);
assert.equal(analysis.needsOcclusionTargets, true);
assert.equal(analysis.needsHiZTarget, true);

console.log("test_webgpu_frame_feature_analyzer: ok");
