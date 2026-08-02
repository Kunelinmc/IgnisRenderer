import assert from "node:assert/strict";

import { PBRMaterial } from "../../../src/materials/PBRMaterial.ts";
import { WebGPUFrameFeatureAnalyzer } from "../../../src/backends/webgpu/rendergraph/WebGPUFrameFeatureAnalyzer.ts";
import { POST_PROCESS_SHARED_RESOURCE_IDS } from "../../../src/postprocess/executionDeclarations.ts";
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

function postProcessPasses(context) {
	return Array.from(context.postProcess.getEnabledPasses()).map((resolved) => {
		const implementation = resolved.pass.getImplementation("webgpu");
		const declaration = implementation.describeExecution({
			backend: "webgpu",
			frameContext: context,
			postProcess: context.postProcess,
			width: context.attachments?.width ?? 1,
			height: context.attachments?.height ?? 1,
			options: resolved.options,
		});
		return {
			...resolved,
			implementation,
			declaration,
			historyIds: (declaration.histories ?? []).map(
				(entry) => entry.descriptor.id,
			),
			transientIds: (declaration.transients ?? []).map(
				(entry) => entry.descriptor.id,
			),
		};
	});
}

function analyze(context, passes = postProcessPasses(context)) {
	return new WebGPUFrameFeatureAnalyzer().analyze(context, {
		framePackets: createFramePackets(context),
		postProcessPasses: passes,
	});
}

function createFramePackets(context) {
	return {
		all: [...context.scene.opaquePackets, ...context.scene.transparentPackets],
		opaque: context.scene.opaquePackets,
		transparent: context.scene.transparentPackets,
		shadowCasters: context.scene.shadowCasterPackets ?? [],
		shadowTransmitters: context.scene.shadowTransmitterPackets ?? [],
		reflective: context.scene.reflectivePackets,
	};
}

const analysis = analyze(createContext());
assert.equal(analysis.hasDeferredLightingWork, true);
assert.equal(analysis.oitRequested, true);
assert.equal(analysis.hasOITWork, true);
assert.equal(analysis.transparency.oitPackets.length, 1);
assert.equal(analysis.transparency.transmissionPackets.length, 0);
assert.equal(analysis.needsPlanarReflection, true);
assert.equal(analysis.needsPlanarReflectionMask, true);
assert.equal(analysis.needsOcclusionTargets, true);
assert.equal(analysis.needsHiZTarget, true);

const ssgiContext = createContext();
ssgiContext.features.enableOIT = false;
ssgiContext.features.enableReflection = false;
ssgiContext.features.enableOcclusionCulling = false;
ssgiContext.postProcess = createResolvedPostProcess({ ssgi: true }, "webgpu");
ssgiContext.scene.opaquePackets = [];
ssgiContext.scene.transparentPackets = [];
ssgiContext.scene.reflectivePackets = [];
ssgiContext.scene.occlusion = { eligibleCandidateCount: 0 };
const ssgiAnalysis = analyze(ssgiContext);
assert.equal(ssgiAnalysis.needsPostProcessGBuffer, true);
assert.equal(ssgiAnalysis.needsHiZTarget, true);

const customHiZContext = createContext();
customHiZContext.features.enableOcclusionCulling = false;
customHiZContext.scene.occlusion = { eligibleCandidateCount: 0 };
customHiZContext.postProcess = createResolvedPostProcess({}, "webgpu");
const customHiZAnalysis = analyze(customHiZContext, [{
	id: "custom-hiz-consumer",
	pass: {},
	options: {},
	implementation: {},
	declaration: {
		color: { access: "read", output: "new-version" },
		shared: [{
			id: "backend:frame-hiz",
			access: "read",
			usage: "sampled",
		}],
	},
	historyIds: [],
	transientIds: [],
}]);
assert.equal(customHiZAnalysis.needsHiZTarget, true);

const customMaskContext = createContext();
customMaskContext.features.enableReflection = false;
customMaskContext.scene.reflectivePackets = [];
const customMaskAnalysis = analyze(customMaskContext, [{
	id: "custom-mask-consumer",
	declaration: {
		color: { access: "read", output: "new-version" },
		shared: [{
			id: POST_PROCESS_SHARED_RESOURCE_IDS.planarReflectionMask,
			access: "read",
			usage: "sampled",
			optional: true,
		}],
	},
}]);
assert.equal(customMaskAnalysis.needsPlanarReflectionMask, true);

const passIdOnlyContext = createContext();
passIdOnlyContext.features.enableReflection = false;
passIdOnlyContext.scene.reflectivePackets = [];
const passIdOnlyAnalysis = analyze(passIdOnlyContext, [{
	id: "ssr",
	declaration: { color: { access: "read", output: "new-version" } },
}]);
assert.equal(passIdOnlyAnalysis.needsPlanarReflectionMask, false);

const transmissionOnly = createContext();
transmissionOnly.scene.transparentPackets = [{ material: { transmissionFactor: 1 } }];
const transmissionAnalysis = analyze(transmissionOnly);
assert.equal(transmissionAnalysis.transparency.hasOITContributors, false);
assert.equal(transmissionAnalysis.transparency.transmissionPackets.length, 1);

const customTransmissionPasses = [{
	id: "custom-transmission-consumer",
	declaration: {
		color: { access: "read", output: "new-version" },
		shared: [{
			id: POST_PROCESS_SHARED_RESOURCE_IDS.transmissionSceneColor,
			access: "read",
			usage: "sampled",
		}],
	},
}];
assert.equal(
	analyze(transmissionOnly, customTransmissionPasses).needsTransmissionTargets,
	true,
);
const noTransmissionPackets = createContext();
noTransmissionPackets.scene.transparentPackets = [];
assert.equal(
	analyze(noTransmissionPackets, customTransmissionPasses).needsTransmissionTargets,
	false,
);

const additiveOnly = createContext();
additiveOnly.scene.transparentPackets = [];
additiveOnly.scene.particleSystems = [{
	visible: true,
	templates: [{ shape: { kind: "billboard", blendMode: "additive" } }],
}];
const additiveAnalysis = analyze(additiveOnly);
assert.equal(additiveAnalysis.transparency.hasOITContributors, false);
assert.equal(additiveAnalysis.transparency.hasAdditiveBillboardParticles, true);

console.log("test_webgpu_frame_feature_analyzer: ok");
