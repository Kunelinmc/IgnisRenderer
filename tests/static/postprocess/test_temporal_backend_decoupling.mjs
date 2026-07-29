import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";

const ROOT = join(import.meta.dir, "..", "..", "..");
const backendFiles = [
	"src/backends/software/SoftwareBackend.ts",
	"src/backends/webgl/WebGLFrameServiceOwner.ts",
	"src/backends/webgpu/WebGPUFrameBindingCache.ts",
];

for (const relativePath of backendFiles) {
	const source = readFileSync(join(ROOT, relativePath), "utf8");
	for (const forbidden of [
		'getOptions<TAAOptions>',
		'isEnabled("taa")',
		"DEFAULT_TAA_OPTIONS",
		"TEMPORAL_ANTI_ALIASING_PASS_ID",
		"PostProcessFrameRequirements",
	]) {
		assert.equal(
			source.includes(forbidden),
			false,
			`${relativePath} must not depend on ${forbidden}`,
		);
	}
}

const frameRequirementsSource = readFileSync(
	join(ROOT, "src/pipeline/FrameRequirements.ts"),
	"utf8",
);
for (const required of [
	"interface CameraJitterRequirement",
	"interface FramePreparationRequirements",
]) {
	assert.equal(
		frameRequirementsSource.includes(required),
		true,
		`Pipeline frame requirements must define ${required}`,
	);
}

const webgpuResourceContractsSource = readFileSync(
	join(ROOT, "src/backends/webgpu/WebGPUResourceContracts.ts"),
	"utf8",
);
const webgpuFrameServiceSource = readFileSync(
	join(ROOT, "src/backends/webgpu/WebGPUFrameServiceOwner.ts"),
	"utf8",
);
assert.equal(
	webgpuResourceContractsSource.includes(
		"interface WebGPUFrameScopePrepareOptions",
	),
	true,
	"WebGPU scope preparation must expose scope-owned options",
);
assert.equal(
	webgpuFrameServiceSource.includes(
		"interface WebGPUFrameServicePrepareOptions",
	),
	true,
	"WebGPU frame services must use service-owned preparation options",
);
for (const source of [
	webgpuResourceContractsSource,
	webgpuFrameServiceSource,
]) {
	assert.equal(
		source.includes("interface WebGPUPrepareFrameOptions"),
		false,
		"Distinct WebGPU preparation option shapes must not share one name",
	);
}

const webgpuFrameBindingSource = readFileSync(
	join(ROOT, "src/backends/webgpu/WebGPUFrameBindingCache.ts"),
	"utf8",
);
assert.equal(
	webgpuFrameBindingSource.includes("TemporalFrameState"),
	true,
	"WebGPU frame bindings must reuse the cross-backend temporal transaction",
);
for (const forbidden of [
	"TemporalJitterState",
	"TemporalJitterCheckpoint",
	"_temporalCheckpoint",
	"_pendingViewProjection",
	"_prevViewProjection",
	"_temporalTransactionManaged",
	"beginTemporalFrameTransaction",
]) {
	assert.equal(
		webgpuFrameBindingSource.includes(forbidden),
		false,
		`WebGPU frame bindings must not duplicate ${forbidden}`,
	);
}

for (const relativePath of [
	"src/backends/webgl/WebGLPostProcessBridge.ts",
	"src/backends/webgl/WebGLPostProcessServices.ts",
]) {
	const source = readFileSync(join(ROOT, relativePath), "utf8");
	for (const forbidden of [
		"markTAAHistoryValid",
		"applyPipelineHistories",
		"_taaHistoryValid",
		"histories.taa",
	]) {
		assert.equal(
			source.includes(forbidden),
			false,
			`${relativePath} must not retain TAA-specific history state`,
		);
	}
}

console.log("Temporal backend decoupling tests passed");
