import assert from "node:assert/strict";

import { PBRMaterial } from "../../../src/materials/PBRMaterial.ts";
import { WebGPUDeferredFrameModule } from "../../../src/backends/webgpu/rendergraph/WebGPUDeferredFrameModule.ts";
import { WebGPUFrameConfigurationModule } from "../../../src/backends/webgpu/rendergraph/WebGPUFrameConfigurationModule.ts";
import { WebGPUFrameMessageRegistry } from "../../../src/backends/webgpu/rendergraph/WebGPUFrameMessageRegistry.ts";
import {
	WEBGPU_FRAME_CONFIGURATION_MESSAGE,
	WEBGPU_FRAME_CONFIGURATION_DEMAND_MESSAGE,
	WEBGPU_FRAME_CONFIGURATION_REQUEST_MESSAGE,
	WEBGPU_FRAME_LOGICAL_RESOURCES,
	WEBGPU_FRAME_CONTEXT_MESSAGE,
	WEBGPU_FRAME_PACKETS_MESSAGE,
	WEBGPU_POST_PROCESS_PASSES_MESSAGE,
} from "../../../src/backends/webgpu/rendergraph/WebGPUFrameMessages.ts";
import { WebGPUPostProcessFrameModule } from "../../../src/backends/webgpu/rendergraph/WebGPUPostProcessFrameModule.ts";
import { WebGPUReflectionFrameModule } from "../../../src/backends/webgpu/rendergraph/WebGPUReflectionFrameModule.ts";
import { WebGPUTransparencyRuntime } from "../../../src/backends/webgpu/rendergraph/WebGPUTransparencyRuntime.ts";
import { WebGPUVisibilityFrameModule } from "../../../src/backends/webgpu/rendergraph/WebGPUVisibilityFrameModule.ts";

const unused = {};
const modules = [
	new WebGPUDeferredFrameModule(unused, unused, unused),
	new WebGPUTransparencyRuntime(unused, unused, unused, unused, unused, unused),
	new WebGPUReflectionFrameModule(unused, unused, unused),
	new WebGPUVisibilityFrameModule(unused, unused, unused),
	new WebGPUPostProcessFrameModule(unused, unused, unused, unused, unused),
];
const registry = new WebGPUFrameMessageRegistry();
for (const module of modules) {
	for (const handler of module.messageHandlers) registry.register(handler);
}
for (const handler of new WebGPUFrameConfigurationModule().messageHandlers) {
	registry.register(handler);
}
registry.seal();

function createContext() {
	return {
		features: {
			enableOIT: false,
			enableReflection: false,
			enableOcclusionCulling: false,
		},
		scene: {
			decalPackets: [],
			particleSystems: [],
			reflectivePackets: [],
			occlusion: { eligibleCandidateCount: 0 },
		},
	};
}

function packets(opaque = [], transparent = []) {
	return {
		all: [...opaque, ...transparent],
		opaque,
		transparent,
		shadowCasters: [],
		shadowTransmitters: [],
		reflective: [],
	};
}

async function analyze(context, framePackets, passes) {
	return registry.dispatch("analysis", {
		seeds: [
			{ descriptor: WEBGPU_FRAME_CONTEXT_MESSAGE, value: context },
			{ descriptor: WEBGPU_FRAME_PACKETS_MESSAGE, value: framePackets },
			{ descriptor: WEBGPU_POST_PROCESS_PASSES_MESSAGE, value: passes },
		],
	});
}

async function configure(analysis, context, overrides = {}) {
	const sampleCount = overrides.sampleCount ?? 1;
	const snapshot = await registry.dispatch("configuration", {
		prior: analysis,
		seeds: [{
			descriptor: WEBGPU_FRAME_CONFIGURATION_REQUEST_MESSAGE,
			value: {
				context,
				capabilities: {
					maxColorAttachments: overrides.maxColorAttachments ?? 8,
					maxColorAttachmentBytesPerSample:
						overrides.maxColorAttachmentBytesPerSample ?? 64,
					maxStorageTexturesPerShaderStage:
						overrides.maxStorageTexturesPerShaderStage ?? 4,
				},
				options: {
					enableEarlyZPrepass: true,
					enableDeferredLighting: true,
					samplePlan: {
						requestedSampleCount: sampleCount,
						sampleCount,
						selectionSignature: "test",
						runtimeFallbackActive: false,
					},
					supportsInFrameTextureCopy: true,
				},
			},
		}],
	});
	return snapshot.get(WEBGPU_FRAME_CONFIGURATION_MESSAGE);
}

const emptyContext = createContext();
const emptyAnalysis = await analyze(emptyContext, packets(), []);
const empty = await configure(emptyAnalysis, emptyContext);
assert.equal(empty.sceneTargetMode, "single");
assert.equal(empty.targetRequirements, null);

const colorOnlyAnalysis = await analyze(emptyContext, packets(), [{
	id: "bloom",
	declaration: { color: { access: "read", output: "new-version" } },
}]);
const colorOnly = await configure(colorOnlyAnalysis, emptyContext);
assert.equal(colorOnly.sceneTargetMode, "color");
assert.equal(colorOnly.targetRequirements.sceneTargetMode, "color");

const oitContext = createContext();
oitContext.features.enableOIT = true;
const oitPackets = packets([], [{ material: {} }]);
const oitAnalysis = await analyze(oitContext, oitPackets, []);
const oit = await configure(oitAnalysis, oitContext);
assert.equal(oit.oitActive, true);
assert.equal(oit.targetRequirements.needsOITTargets, true);
const oitMsaa = await configure(oitAnalysis, oitContext, { sampleCount: 4 });
assert.equal(oitMsaa.oitActive, false);
assert.ok(oitMsaa.diagnostics.some(({ code }) => code === "webgpu-oit-disabled-msaa"));

const deferredContext = createContext();
const deferredPackets = packets([{ material: new PBRMaterial() }]);
const deferredAnalysis = await analyze(deferredContext, deferredPackets, []);
const deferred = await configure(deferredAnalysis, deferredContext);
assert.equal(deferred.deferredActive, true);
assert.equal(deferred.sceneTargetMode, "gbuffer");
assert.equal(deferred.deferredGBufferLayout, "base");

const unsupported = await configure(deferredAnalysis, deferredContext, {
	maxColorAttachments: 1,
	maxColorAttachmentBytesPerSample: 16,
	maxStorageTexturesPerShaderStage: 0,
});
assert.equal(unsupported.deferredActive, false);
assert.equal(unsupported.sceneTargetMode, "single");
for (const code of [
	"webgpu-mrt-disabled-attachments",
	"webgpu-deferred-disabled-attachments",
	"webgpu-deferred-disabled-bytes",
	"webgpu-deferred-disabled-storage-textures",
]) assert.ok(unsupported.diagnostics.some((diagnostic) => diagnostic.code === code));

let transmissionReads = 0;
const retryContext = createContext();
const retryPackets = packets([], [{
	material: {
		get transmissionFactor() {
			transmissionReads++;
			return 1;
		},
	},
}]);
const retryAnalysis = await analyze(retryContext, retryPackets, []);
await configure(retryAnalysis, retryContext);
await configure(retryAnalysis, retryContext);
assert.equal(transmissionReads, 1);

const conflictRegistry = new WebGPUFrameMessageRegistry();
for (const [moduleId, id] of [
	["a", WEBGPU_FRAME_LOGICAL_RESOURCES.postProcessTargets],
	["b", WEBGPU_FRAME_LOGICAL_RESOURCES.hiZTarget],
]) {
	conflictRegistry.register({
		id: "demand",
		moduleId,
		phase: "configuration",
		inputs: [{ descriptor: WEBGPU_FRAME_CONFIGURATION_REQUEST_MESSAGE }],
		outputs: [WEBGPU_FRAME_CONFIGURATION_DEMAND_MESSAGE],
		run(_messages, publisher) {
			publisher.publish(WEBGPU_FRAME_CONFIGURATION_DEMAND_MESSAGE, {
				source: moduleId,
				resources: [{ id, exclusiveGroup: "test-target" }],
			});
		},
	});
}
for (const handler of new WebGPUFrameConfigurationModule().messageHandlers) {
	conflictRegistry.register(handler);
}
conflictRegistry.seal();
await assert.rejects(
	conflictRegistry.dispatch("configuration", {
		seeds: [{
			descriptor: WEBGPU_FRAME_CONFIGURATION_REQUEST_MESSAGE,
			value: {
				context: emptyContext,
				capabilities: {
					maxColorAttachments: 8,
					maxColorAttachmentBytesPerSample: 64,
					maxStorageTexturesPerShaderStage: 4,
				},
				options: {
					enableEarlyZPrepass: true,
					enableDeferredLighting: true,
					samplePlan: {
						requestedSampleCount: 1,
						sampleCount: 1,
						selectionSignature: "conflict",
						runtimeFallbackActive: false,
					},
					supportsInFrameTextureCopy: true,
				},
			},
		}],
	}),
	(error) => /conflicting exclusive resource/.test(error.cause?.message),
);

console.log("WebGPU frame configuration message tests passed");
