import assert from "node:assert/strict";

import { PBRMaterial } from "../../../src/materials/PBRMaterial.ts";
import { ShaderMaterial } from "../../../src/materials/ShaderMaterial.ts";
import { WebGPUDeferredFrameModule } from "../../../src/backends/webgpu/rendergraph/WebGPUDeferredFrameModule.ts";
import {
	WebGPUFrameConfigurationModule,
} from "../../../src/backends/webgpu/rendergraph/WebGPUFrameConfigurationModule.ts";
import {
	WebGPUFrameConfigurationBuilder,
} from "../../../src/backends/webgpu/rendergraph/WebGPUFrameConfigurationContribution.ts";
import { WebGPUFrameGraphModuleRegistry } from "../../../src/backends/webgpu/rendergraph/WebGPUFrameGraphModuleRegistry.ts";
import {
	WEBGPU_DEFERRED_FEATURE_ANALYSIS,
	WEBGPU_POST_PROCESS_FEATURE_ANALYSIS,
	WEBGPU_REFLECTION_FEATURE_ANALYSIS,
	WEBGPU_TRANSPARENCY_FEATURE_ANALYSIS,
	WEBGPU_VISIBILITY_FEATURE_ANALYSIS,
} from "../../../src/backends/webgpu/rendergraph/WebGPUFrameModuleStateKeys.ts";
import { WebGPUPostProcessFrameModule } from "../../../src/backends/webgpu/rendergraph/WebGPUPostProcessFrameModule.ts";
import { WebGPUReflectionFrameModule } from "../../../src/backends/webgpu/rendergraph/WebGPUReflectionFrameModule.ts";
import { WebGPUTransparencyRuntime } from "../../../src/backends/webgpu/rendergraph/WebGPUTransparencyRuntime.ts";
import { WebGPUVisibilityFrameModule } from "../../../src/backends/webgpu/rendergraph/WebGPUVisibilityFrameModule.ts";
import { WEBGPU_FRAME_GRAPH_NODE_KINDS } from "../../../src/backends/webgpu/rendergraph/types.ts";
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

function prepareModuleAnalysis(context, passes = postProcessPasses(context)) {
	const unused = {};
	const modules = [
		new WebGPUDeferredFrameModule(unused, unused, unused, unused),
		new WebGPUTransparencyRuntime(unused, unused, unused, unused, unused, unused),
		new WebGPUReflectionFrameModule(unused, unused, unused),
		new WebGPUVisibilityFrameModule(unused, unused, unused),
		new WebGPUPostProcessFrameModule(unused, unused, unused, unused, unused),
	];
	const ownedKinds = new Set(
		modules.flatMap((module) => Object.keys(module.executors)),
	);
	const registry = new WebGPUFrameGraphModuleRegistry();
	registry.register({
		id: "remaining-executors",
		executors: Object.fromEntries(
			WEBGPU_FRAME_GRAPH_NODE_KINDS
				.filter((kind) => !ownedKinds.has(kind))
				.map((kind) => [kind, async () => {}]),
		),
		destroy() {},
	});
	for (const module of modules) registry.register(module);
	registry.seal();
	const framePackets = createFramePackets(context);
	const state = registry.analyze({
		context,
		framePackets,
		postProcessPasses: passes,
	});
	return { framePackets, registry, state };
}

function analyze(context, passes = postProcessPasses(context)) {
	const { state } = prepareModuleAnalysis(context, passes);
	const deferred = state.require(WEBGPU_DEFERRED_FEATURE_ANALYSIS);
	const transparency = state.require(WEBGPU_TRANSPARENCY_FEATURE_ANALYSIS);
	const reflection = state.require(WEBGPU_REFLECTION_FEATURE_ANALYSIS);
	const visibility = state.require(WEBGPU_VISIBILITY_FEATURE_ANALYSIS);
	const postProcess = state.require(WEBGPU_POST_PROCESS_FEATURE_ANALYSIS);
	return {
		...deferred,
		transparency,
		oitRequested: context.features.enableOIT === true,
		hasOITWork: transparency.hasOITContributors,
		needsPostProcessTargets: postProcess.needsPostProcessTargets,
		needsPostProcessGBuffer: postProcess.needsPostProcessGBuffer,
		needsPlanarReflection: reflection.needsPlanarReflection,
		needsPlanarReflectionMask:
			reflection.needsPlanarReflection || postProcess.needsPlanarReflectionMask,
		needsTransmissionTargets:
			postProcess.needsTransmissionTargets &&
			transparency.transmissionPackets.length > 0,
		needsOcclusionTargets: visibility.needsOcclusionTargets,
		needsHiZTarget:
			visibility.needsOcclusionTargets || postProcess.needsHiZTarget,
	};
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
assert.equal(analysis.deferredGBufferLayout, "extended");
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

const singleScanContext = createContext();
let transmissionFactorReads = 0;
singleScanContext.scene.transparentPackets = [{
	material: {
		get transmissionFactor() {
			transmissionFactorReads++;
			return 1;
		},
	},
}];
const preparedSingleScan = prepareModuleAnalysis(
	singleScanContext,
	customTransmissionPasses,
);
assert.equal(transmissionFactorReads, 1);
const contributions = preparedSingleScan.registry.collectConfigurationContributions({
	context: singleScanContext,
	state: preparedSingleScan.state,
});
const configurationModule = new WebGPUFrameConfigurationModule();
const incompleteBuilder = new WebGPUFrameConfigurationBuilder(
	preparedSingleScan.framePackets,
);
assert.throws(() => incompleteBuilder.build(), /requires a "deferred" contribution/);
const duplicateBuilder = new WebGPUFrameConfigurationBuilder(
	preparedSingleScan.framePackets,
);
const deferredContribution = preparedSingleScan.state.require(
	WEBGPU_DEFERRED_FEATURE_ANALYSIS,
);
duplicateBuilder.setDeferred(deferredContribution);
assert.throws(
	() => duplicateBuilder.setDeferred(deferredContribution),
	/duplicate "deferred" contributions/,
);
for (let attempt = 0; attempt < 2; attempt++) {
	const configuration = configurationModule.resolve(
		preparedSingleScan.framePackets,
		contributions,
		{
			maxColorAttachments: 8,
			maxColorAttachmentBytesPerSample: 64,
			maxStorageTexturesPerShaderStage: 4,
		},
		{
			enableEarlyZPrepass: true,
			enableDeferredLighting: true,
			samplePlan: {
				requestedSampleCount: 1,
				sampleCount: 1,
				selectionSignature: `test-${attempt}`,
				runtimeFallbackActive: false,
			},
			supportsInFrameTextureCopy: true,
		},
	);
	assert.equal(configuration.targetRequirements.needsTransmissionTargets, true);
}
assert.equal(transmissionFactorReads, 1);

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

const baseContext = createContext();
baseContext.scene.opaquePackets = [{ material: new PBRMaterial() }];
baseContext.scene.transparentPackets = [];
baseContext.scene.decalPackets = [];
assert.equal(analyze(baseContext).deferredGBufferLayout, "base");

const decalContext = createContext();
decalContext.scene.opaquePackets = [{ material: new PBRMaterial() }];
decalContext.scene.decalPackets = [{}];
assert.equal(analyze(decalContext).deferredGBufferLayout, "extended");

const customContext = createContext();
customContext.scene.opaquePackets = [{
	material: new ShaderMaterial({
		deferredLighting: true,
		chunks: [{
			language: "wgsl",
			stage: "fragment",
			mode: "deferred",
			code: "@fragment fn fsMainDeferred() {}",
		}],
	}),
}];
assert.equal(analyze(customContext).deferredGBufferLayout, "extended");

console.log("test_webgpu_frame_feature_analyzer: ok");
