import assert from "node:assert/strict";
import { WebGPUFrameConfigurationResolver } from "../../../src/renderers/webgpu/rendergraph/WebGPUFrameConfigurationResolver.ts";
import { WebGPUFrameFeatureAnalyzer } from "../../../src/renderers/webgpu/rendergraph/WebGPUFrameFeatureAnalyzer.ts";
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

function resolve(context, overrides = {}) {
	const analysis = new WebGPUFrameFeatureAnalyzer().analyze(context);
	return new WebGPUFrameConfigurationResolver().resolve(analysis, {
		maxColorAttachments: 8,
		maxColorAttachmentBytesPerSample: 64,
		maxStorageTexturesPerShaderStage: 4,
	}, {
		enableEarlyZPrepass: true,
		enableDeferredLighting: true,
		sampleCount: 1,
		supportsInFrameTextureCopy: true,
		...overrides,
	});
}

const noWork = resolve(createContext());
assert.equal(noWork.sceneTargetMode, "single");
assert.equal(noWork.targetRequirements, null);

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

const unsupportedAnalysis = new WebGPUFrameFeatureAnalyzer().analyze(deferredContext);
const unsupported = new WebGPUFrameConfigurationResolver().resolve(unsupportedAnalysis, {
	maxColorAttachments: 1,
	maxColorAttachmentBytesPerSample: 16,
	maxStorageTexturesPerShaderStage: 0,
}, {
	enableEarlyZPrepass: true,
	enableDeferredLighting: true,
	sampleCount: 1,
	supportsInFrameTextureCopy: true,
});
assert.equal(unsupported.sceneTargetMode, "single");
assert.ok(unsupported.diagnostics.some((diagnostic) => diagnostic.code === "webgpu-mrt-disabled-attachments"));
assert.ok(unsupported.diagnostics.some((diagnostic) => diagnostic.code === "webgpu-deferred-disabled-mrt"));

console.log("test_webgpu_frame_configuration_resolver: ok");
