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
const source = readFileSync(orchestratorPath, "utf8");
const backendSource = readFileSync(backendPath, "utf8");

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
]) {
	assert.equal(
		source.includes(method),
		false,
		`WebGPUFrameOrchestrator must not define feature method ${method}`,
	);
}

assert.match(source, /WebGPUFrameRuntimeCompositionFactory/);
assert.match(source, /this\._frameModules\.execute\(node, session\)/);
assert.doesNotMatch(source, /from "\.\/WebGPUFrameGraphPlanner"/);
assert.match(backendSource, /createWebGPUFrameRuntimeCompositionFactory/);

console.log("WebGPU frame orchestrator module boundary tests passed");
