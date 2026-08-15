import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

const orchestratorPath = resolve(
	process.cwd(),
	"src/backends/webgpu/rendergraph/WebGPUFrameOrchestrator.ts",
);
const backendPath = resolve(
	process.cwd(),
	"src/backends/webgpu/WebGPUBackend.ts",
);
const sessionPath = resolve(
	process.cwd(),
	"src/backends/webgpu/rendergraph/WebGPUFrameSession.ts",
);
const compositionPath = resolve(
	process.cwd(),
	"src/backends/webgpu/rendergraph/WebGPUFrameRuntimeComposition.ts",
);
const source = readFileSync(orchestratorPath, "utf8");
const backendSource = readFileSync(backendPath, "utf8");
const sessionSource = readFileSync(sessionPath, "utf8");
const compositionSource = readFileSync(compositionPath, "utf8");

const forbiddenFeatureDependencies = [
	"WebGPUDeferredLightingPass",
	"WebGPUDeferredDecalPass",
	"WebGPUPlanarReflectionPass",
	"WebGPUHiZBuilder",
	"WebGPUOcclusionCullingRuntime",
	"WebGPUTransparencyRuntime",
	"WebGPUScenePassRecorder",
	"WebGPUPostProcessBridge",
	"WebGPUPresentationRuntime",
	"WebGPUCustomRenderTargetRuntime",
];

for (const dependency of forbiddenFeatureDependencies) {
	assert.equal(
		source.includes(dependency),
		false,
		`WebGPUFrameOrchestrator must not depend on ${dependency}`,
	);
}

for (const method of [
	"_recordDeferred",
	"_recordPlanarReflection",
	"_recordHiZ",
	"_recordOcclusion",
	"_createNodeRuntimes",
	"exclusiveModuleId",
	"hasPass(",
	"WebGPUFrameFeatureAnalyzer",
	"WebGPUFrameConfigurationResolver",
	"WebGPUFrameConfigurationBuilder",
	"WebGPUFrameGraphPlanningUtils",
	"WebGPUFrameModuleStateStore",
]) {
	assert.equal(
		source.includes(method),
		false,
		`WebGPUFrameOrchestrator must not define feature method ${method}`,
	);
}

for (const forbidden of [
	"WebGPUFrameGraphRecordingContext",
	"runtimeCapabilities",
	"getPreparedFrameResources",
	"getSceneTargetModeForFrame",
	"commitGraphAnalysis",
	"abortGraphAnalysis",
	"_frameContext",
	"_frameResources",
	"_mrtEnabled",
	"_deferredEnabled",
	"_oitActive",
]) {
	assert.equal(source.includes(forbidden), false);
}
assert.match(source, /frameModules: WebGPUFrameGraphModuleRegistry/);
assert.match(source, /this\._frameModules\.execute\(node, session\)/);
assert.doesNotMatch(source, /from "\.\/WebGPUFrameGraphPlanner"/);
assert.match(backendSource, /createWebGPUFrameRuntimeComposition/);
assert.match(backendSource, /this\._frameRuntime = createWebGPUFrameRuntimeComposition/);
for (const forbidden of [
	"motionHistoryWriteTarget",
	"presented",
	"deferredOpaqueFrameState",
	"hiZStatus",
	"hiZBuildCount",
	"transparencyMode",
]) {
	assert.equal(sessionSource.includes(forbidden), false);
}
for (const forbidden of [
	"getSession",
	"requireSession",
	"requireFrameResources",
	"recording:",
]) {
	assert.equal(compositionSource.includes(forbidden), false);
}

console.log("WebGPU frame orchestrator module boundary tests passed");
