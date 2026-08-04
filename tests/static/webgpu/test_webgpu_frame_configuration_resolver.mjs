import assert from "node:assert/strict";
import { WebGPUFrameConfigurationResolver } from "../../../src/backends/webgpu/rendergraph/WebGPUFrameConfigurationResolver.ts";
import { WebGPUFrameFeatureAnalyzer } from "../../../src/backends/webgpu/rendergraph/WebGPUFrameFeatureAnalyzer.ts";
import { PBRMaterial } from "../../../src/materials/PBRMaterial.ts";
import { createResolvedPostProcess } from "../../helpers/postprocess.mjs";

function createContext() {
	return {
		features: {
			enableOIT: false,
			enableReflection: false,
			enableOcclusionCulling: false,
		},
		postProcess: createResolvedPostProcess({}, "webgpu"),
		scene: {
			opaquePackets: [],
			transparentPackets: [],
			particleSystems: [],
			reflectivePackets: [],
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

function resolve(context, overrides = {}) {
	const sampleCount = overrides.sampleCount ?? 1;
	const { sampleCount: _sampleCount, ...rest } = overrides;
	const analysis = new WebGPUFrameFeatureAnalyzer().analyze(context, {
		framePackets: createFramePackets(context),
		postProcessPasses: postProcessPasses(context),
	});
	return new WebGPUFrameConfigurationResolver().resolve(analysis, {
		maxColorAttachments: 8,
		maxColorAttachmentBytesPerSample: 64,
		maxStorageTexturesPerShaderStage: 4,
	}, {
		enableEarlyZPrepass: true,
		enableDeferredLighting: true,
		samplePlan: {
			requestedSampleCount: sampleCount,
			sampleCount,
			selectionSignature: "test",
			runtimeFallbackActive: false,
		},
		supportsInFrameTextureCopy: true,
		...rest,
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

const noWork = resolve(createContext());
assert.equal(noWork.sceneTargetMode, "single");
assert.equal(noWork.targetRequirements, null);

const ssgiContext = createContext();
ssgiContext.postProcess = createResolvedPostProcess({ ssgi: true }, "webgpu");
const ssgi = resolve(ssgiContext);
assert.equal(ssgi.sceneTargetMode, "mrt");
assert.equal(ssgi.targetRequirements.needsPostProcessTargets, true);
assert.equal(ssgi.targetRequirements.needsHiZTarget, true);

const oitContext = createContext();
oitContext.features.enableOIT = true;
oitContext.scene.transparentPackets.push({ material: {} });
const oit = resolve(oitContext);
assert.equal(oit.oitActive, true);
assert.equal(oit.targetRequirements.needsOITTargets, true);
assert.equal(oit.transparencyMode, "oit");

const transmissionOnlyContext = createContext();
transmissionOnlyContext.features.enableOIT = true;
transmissionOnlyContext.scene.transparentPackets.push({ material: { transmissionFactor: 1 } });
const transmissionOnly = resolve(transmissionOnlyContext);
assert.equal(transmissionOnly.oitActive, false);
assert.equal(transmissionOnly.targetRequirements, null);

const additiveOnlyContext = createContext();
additiveOnlyContext.features.enableOIT = true;
additiveOnlyContext.scene.particleSystems.push({
	visible: true,
	templates: [{ shape: { kind: "billboard", blendMode: "additive" } }],
});
const additiveOnly = resolve(additiveOnlyContext);
assert.equal(additiveOnly.oitActive, false);
assert.equal(additiveOnly.targetRequirements, null);

const oitMsaa = resolve(oitContext, { sampleCount: 4 });
assert.equal(oitMsaa.oitActive, false);
assert.ok(oitMsaa.diagnostics.some((diagnostic) => diagnostic.code === "webgpu-oit-disabled-msaa"));

const deferredContext = createContext();
deferredContext.scene.opaquePackets.push({ material: new PBRMaterial({ anisotropyStrength: 1 }) });
const deferred = resolve(deferredContext);
assert.equal(deferred.sceneTargetMode, "gbuffer");
assert.equal(deferred.deferredActive, true);
const deferredMsaa = resolve(deferredContext, { sampleCount: 4 });
assert.equal(deferredMsaa.deferredActive, false);
assert.equal(deferredMsaa.sceneTargetMode, "color");

const sceneMsaa = resolve(createContext(), { sampleCount: 4 });
assert.equal(sceneMsaa.sceneTargetMode, "color");

const unsupportedAnalysis = new WebGPUFrameFeatureAnalyzer().analyze(deferredContext, {
	framePackets: createFramePackets(deferredContext),
	postProcessPasses: [],
});
const unsupported = new WebGPUFrameConfigurationResolver().resolve(unsupportedAnalysis, {
	maxColorAttachments: 1,
	maxColorAttachmentBytesPerSample: 16,
	maxStorageTexturesPerShaderStage: 0,
}, {
	enableEarlyZPrepass: true,
	enableDeferredLighting: true,
	samplePlan: {
		requestedSampleCount: 1,
		sampleCount: 1,
		selectionSignature: "test",
		runtimeFallbackActive: false,
	},
	supportsInFrameTextureCopy: true,
});
assert.equal(unsupported.sceneTargetMode, "single");
assert.ok(unsupported.diagnostics.some((diagnostic) => diagnostic.code === "webgpu-mrt-disabled-attachments"));
assert.ok(unsupported.diagnostics.some((diagnostic) => diagnostic.code === "webgpu-deferred-disabled-mrt"));

console.log("test_webgpu_frame_configuration_resolver: ok");
