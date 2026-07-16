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
assert.equal(analysis.transparency.oitPackets.length, 1);
assert.equal(analysis.transparency.transmissionPackets.length, 0);
assert.equal(analysis.needsPlanarReflection, true);
assert.equal(analysis.needsPlanarReflectionMask, true);
assert.equal(analysis.needsOcclusionTargets, true);
assert.equal(analysis.needsHiZTarget, true);

const transmissionOnly = createContext();
transmissionOnly.scene.transparentPackets = [{ material: { transmissionFactor: 1 } }];
const transmissionAnalysis = new WebGPUFrameFeatureAnalyzer().analyze(transmissionOnly);
assert.equal(transmissionAnalysis.transparency.hasOITContributors, false);
assert.equal(transmissionAnalysis.transparency.transmissionPackets.length, 1);

const additiveOnly = createContext();
additiveOnly.scene.transparentPackets = [];
additiveOnly.scene.particleSystems = [{
	visible: true,
	templates: [{ shape: { kind: "billboard", blendMode: "additive" } }],
}];
const additiveAnalysis = new WebGPUFrameFeatureAnalyzer().analyze(additiveOnly);
assert.equal(additiveAnalysis.transparency.hasOITContributors, false);
assert.equal(additiveAnalysis.transparency.hasAdditiveBillboardParticles, true);

console.log("test_webgpu_frame_feature_analyzer: ok");
