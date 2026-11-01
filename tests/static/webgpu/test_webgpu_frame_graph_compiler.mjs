import assert from "node:assert/strict";
import { WebGPUFrameGraphCompiler } from "../../../src/backends/webgpu/rendergraph/WebGPUFrameGraphCompiler.ts";

function createPass(stage = "main-opaque") {
	return { stage, executor: "backend", enabled: true, dependsOn: [] };
}

function compile(nodes, initialResources = []) {
	const compiler = new WebGPUFrameGraphCompiler();
	compiler.beginFrame(initialResources);
	return {
		compiler,
		stage: compiler.compileStage({
			pass: createPass(),
			nodes,
		}),
	};
}

function testReadBeforeCreateDiagnostic() {
	const { stage } = compile([{
		id: "test:read",
		stage: "main-opaque",
		kind: "opaque-scene",
		label: "ReadInactive",
		reads: [{ id: "frame:scene-color-main", usage: "texture-binding" }],
	}]);

	assert.equal(stage.diagnostics.length, 1);
	assert.equal(stage.diagnostics[0].code, "read-before-create");
	assert.equal(stage.diagnostics[0].resource, "frame:scene-color-main");
}

function testDuplicateCreateDiagnosticMatchesLegacyShape() {
	const { stage } = compile([{
		id: "test:create",
		stage: "main-opaque",
		kind: "opaque-scene",
		label: "CreateActive",
		creates: [{ id: "frame:depth" }],
	}], ["frame:depth"]);

	assert.equal(stage.diagnostics[0].code, "duplicate-create");
	assert.equal(
		stage.diagnostics[0].message,
		'Frame graph node "test:create" creates already active resource "frame:depth".',
	);
}

function testOptionalReadDoesNotCreateResource() {
	const { compiler } = compile([
		{
			id: "test:optional-read",
			stage: "main-transparent",
			kind: "transparent-scene",
			label: "OptionalRead",
			reads: [{
				id: "shadow-atlas",
				usage: "texture-binding",
				optional: true,
			}],
		},
		{
			id: "test:required-read",
			stage: "main-opaque",
			kind: "opaque-scene",
			label: "RequiredRead",
			reads: [{
				id: "shadow-atlas",
				usage: "texture-binding",
			}],
		},
	]);

	const state = compiler
		.getResourceDebugState()
		.find((resource) => resource.id === "shadow-atlas");
	assert.ok(state);
	assert.equal(state.initialized, false);
	assert.equal(state.lastAccess, "read");
	assert.equal(compiler.getDiagnostics().length, 1);
	assert.equal(compiler.getDiagnostics()[0].code, "read-before-create");
}

function testOptionalReadDoesNotEmitBarrierForLaterWrite() {
	const { stage } = compile([
		{
			id: "test:optional-read",
			stage: "main-transparent",
			kind: "transparent-scene",
			label: "OptionalRead",
			reads: [{
				id: "planar-reflection:capture",
				usage: "texture-binding",
				optional: true,
			}],
		},
		{
			id: "test:write",
			stage: "reflection",
			kind: "planar-reflection-capture",
			label: "WriteReflection",
			writes: [{
				id: "planar-reflection:capture",
				usage: "render-attachment",
			}],
		},
	]);

	assert.equal(stage.diagnostics.length, 0);
	assert.equal(stage.barriers.length, 0);
}

function testUsageTransitionsEmitBarriers() {
	const { compiler, stage } = compile([
		{
			id: "test:write",
			stage: "main-opaque",
			kind: "opaque-scene",
			label: "WriteScene",
			writes: [{
				id: "frame:scene-color-main",
				usage: "render-attachment",
			}],
		},
		{
			id: "test:read",
			stage: "postprocess",
			kind: "particles",
			label: "ReadScene",
			reads: [{
				id: "frame:scene-color-main",
				usage: "texture-binding",
			}],
		},
	]);

	assert.equal(stage.diagnostics.length, 0);
	assert.equal(stage.barriers.length, 1);
	assert.equal(stage.barriers[0].reason, "read-after-write");
	assert.equal(stage.barriers[0].fromUsage, "render-attachment");
	assert.equal(stage.barriers[0].toUsage, "texture-binding");
	assert.equal(compiler.getBarriers().length, 1);
}

function testResourceDebugStateTracksLastAccess() {
	const { compiler } = compile([{
		id: "test:write-depth",
		stage: "main-opaque",
		kind: "opaque-scene",
		label: "WriteDepth",
		writes: [{ id: "frame:depth", usage: "depth-attachment" }],
	}]);

	const state = compiler
		.getResourceDebugState()
		.find((resource) => resource.id === "frame:depth");
	assert.ok(state);
	assert.equal(state.initialized, true);
	assert.equal(state.lastNodeId, "test:write-depth");
	assert.equal(state.lastAccess, "write");
	assert.equal(state.lastUsage, "depth-attachment");
}

function testGroupedAnalysisSeparatesShadowDiagnostics() {
	const { compiler, stage } = compile([{
		id: "test:read-imported",
		stage: "main-opaque",
		kind: "opaque-scene",
		label: "ReadImported",
		reads: [{ id: "frame:depth", usage: "texture-binding" }],
	}], ["frame:depth"]);

	assert.equal(stage.diagnostics.length, 0);
	assert.equal(compiler.getDiagnostics().length, 0);
	assert.ok(
		compiler.getGraphAnalysis().current.shadowDiagnostics.some(
			(diagnostic) => diagnostic.code === "read-content-unknown"
		)
	);
	compiler.recordOpaqueStage(
		"particle-sim",
		"Particle simulation is backend-private.",
	);
	compiler.seal();
	compiler.commit();
	const analysis = compiler.getGraphAnalysis();
	assert.equal(analysis.state, "committed");
	assert.equal(analysis.lastSuccessful.completeness, "opaque");
	assert.equal(analysis.lastSuccessful.diagnostics.length, 0);
}

function testForwardMRTGBufferTransitionsToPostProcessReads() {
	const resources = ["gbuffer:albedo-alpha", "gbuffer:normal-rough-metal",
		"gbuffer:emissive-occlusion", "gbuffer:motion-depth"];
	const { compiler } = compile([{
		id: "main-opaque:scene",
		stage: "main-opaque",
		kind: "opaque-scene",
		label: "WebGPUMainMRT",
		writes: resources.map((id) => ({ id, usage: "render-attachment" })),
	}]);
	const stage = compiler.compileStage({
		pass: createPass("postprocess"),
		nodes: [{
			id: "postprocess:effect",
			stage: "postprocess",
			kind: "post-process-pass",
			label: "PostProcessGBufferConsumer",
			reads: resources.map((id) => ({ id, usage: "texture-binding" })),
		}],
	});
	assert.equal(stage.diagnostics.length, 0);
	assert.deepEqual(stage.barriers.map((barrier) => ({
		resource: barrier.resource,
		reason: barrier.reason,
		fromUsage: barrier.fromUsage,
		toUsage: barrier.toUsage,
	})), resources.map((resource) => ({
		resource,
		reason: "read-after-write",
		fromUsage: "render-attachment",
		toUsage: "texture-binding",
	})));
	for (const id of resources) {
		const state = compiler.getResourceDebugState().find((entry) => entry.id === id);
		assert.equal(state.lastNodeId, "postprocess:effect");
		assert.equal(state.lastAccess, "read");
		assert.equal(state.lastUsage, "texture-binding");
	}
}

function run() {
	testReadBeforeCreateDiagnostic();
	testDuplicateCreateDiagnosticMatchesLegacyShape();
	testOptionalReadDoesNotCreateResource();
	testOptionalReadDoesNotEmitBarrierForLaterWrite();
	testUsageTransitionsEmitBarriers();
	testResourceDebugStateTracksLastAccess();
	testGroupedAnalysisSeparatesShadowDiagnostics();
	testForwardMRTGBufferTransitionsToPostProcessReads();
	console.log("test_webgpu_frame_graph_compiler: ok");
}

run();
