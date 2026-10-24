import assert from "node:assert/strict";
import { WebGLFrameGraphCompiler } from "../../../src/backends/webgl/rendergraph/WebGLFrameGraphCompiler.ts";

function createPass(stage = "main-opaque") {
	return {
		stage,
		executor: "backend",
		enabled: true,
		dependsOn: [],
	};
}

function compile(nodes, initialResources = []) {
	const compiler = new WebGLFrameGraphCompiler();
	compiler.beginFrame(initialResources);
	const compiled = compiler.compileStage({
		pass: createPass(),
		nodes,
	});
	return {
		compiler,
		compiled,
	};
}

function testReadBeforeCreateDiagnostic() {
	const { compiled } = compile([
		{
			id: "node:read",
			stage: "main-opaque",
			kind: "opaque-scene",
			label: "ReadMissing",
			reads: [{ id: "frame:scene-color", usage: "texture-sampling" }],
		},
	]);

	assert.deepEqual(compiled.diagnostics.map((d) => d.code), [
		"read-before-create",
	]);
}

function testDuplicateCreateDiagnostic() {
	const { compiled } = compile(
		[
			{
				id: "node:create",
				stage: "main-opaque",
				kind: "opaque-scene",
				label: "CreateDuplicate",
				creates: [{ id: "frame:scene-color" }],
			},
		],
		["frame:scene-color"]
	);

	assert.deepEqual(compiled.diagnostics.map((d) => d.code), [
		"duplicate-create",
	]);
	assert.equal(
		compiled.diagnostics[0].message,
		'WebGL frame graph node "node:create" creates already active ' +
			'resource "frame:scene-color".',
	);
}

function testMissingRequiredResourceDiagnostic() {
	const { compiled } = compile([
		{
			id: "node:oit",
			stage: "main-transparent",
			kind: "oit-clear",
			label: "MissingOIT",
			requires: [{ id: "oit:accum" }],
		},
	]);

	assert.deepEqual(compiled.diagnostics.map((d) => d.code), [
		"missing-resource",
	]);
	assert.equal(
		compiled.diagnostics[0].message,
		'WebGL frame graph node "node:oit" requires missing resource "oit:accum".',
	);
}

function testTextureFeedbackLoopDiagnostic() {
	const { compiled } = compile(
		[
			{
				id: "node:feedback",
				stage: "postprocess",
				kind: "post-process-pass",
				label: "Feedback",
				reads: [{ id: "post:color", usage: "texture-sampling" }],
				writes: [{ id: "post:color", usage: "framebuffer-color" }],
			},
		],
		["post:color"]
	);

	assert.deepEqual(compiled.diagnostics.map((d) => d.code), [
		"texture-feedback-loop",
	]);
}

function testUnsupportedUsageDiagnostic() {
	const { compiled } = compile([
		{
			id: "node:bad-usage",
			stage: "main-opaque",
			kind: "opaque-scene",
			label: "BadUsage",
			reads: [{ id: "frame:scene-color", usage: "storage-binding" }],
		},
	]);

	assert.ok(
		compiled.diagnostics.some(
			(diagnostic) => diagnostic.code === "unsupported-node-resource"
		)
	);
}

function testNodeOrderAndBarriersArePreserved() {
	const { compiler, compiled } = compile(
		[
			{
				id: "node:write",
				stage: "main-opaque",
				kind: "opaque-scene",
				label: "Write",
				writes: [{ id: "frame:scene-color", usage: "framebuffer-color" }],
			},
			{
				id: "node:read",
				stage: "postprocess",
				kind: "post-process-pass",
				label: "Read",
				reads: [{ id: "frame:scene-color", usage: "texture-sampling" }],
			},
		],
		["frame:scene-color"]
	);

	assert.deepEqual(compiled.nodes.map((node) => node.id), [
		"node:write",
		"node:read",
	]);
	assert.deepEqual(compiler.getBarriers().map((barrier) => barrier.reason), [
		"read-after-write",
	]);
}

function testGroupedAnalysisCommitBoundary() {
	const { compiler, compiled } = compile(
		[{
			id: "node:read-imported",
			stage: "main-opaque",
			kind: "opaque-scene",
			label: "ReadImported",
			reads: [{ id: "frame:scene-color", usage: "texture-sampling" }],
		}],
		["frame:scene-color"],
	);
	assert.equal(compiled.diagnostics.length, 0);
	assert.ok(
		compiler.getGraphAnalysis().current.shadowDiagnostics.some(
			(diagnostic) => diagnostic.code === "read-content-unknown"
		)
	);
	compiler.seal();
	assert.equal(compiler.getGraphAnalysis().lastSuccessful, null);
	compiler.commit();
	assert.equal(compiler.getGraphAnalysis().lastSuccessful.state, "committed");
}

function testWholeFrameCacheInvalidatesCompletePlanAndResetsAttempts() {
	const compiler = new WebGLFrameGraphCompiler();
	const plan = {
		resources: [{
			id: "color", origin: "imported", kind: "texture",
			initialContent: "valid", width: 16, height: 16, format: "rgba8unorm",
		}],
		bindings: [{ resourceId: "color", physicalId: "physical:color", kind: "texture" }],
		stages: [{
			pass: createPass(),
			nodes: [{
				id: "draw", stage: "main-opaque", kind: "opaque-scene", label: "Draw",
				reads: [{ id: "color", usage: "texture-sampling" }],
			}],
		}],
	};
	const first = compiler.compileFrame(plan);
	compiler.recordSkippedNode("draw", "color", "color");
	compiler.seal();
	compiler.commit();
	const successful = compiler.getGraphAnalysis().lastSuccessful;
	const reused = compiler.compileFrame(structuredClone(plan));
	assert.strictEqual(reused, first, "equal metadata must reuse compilation");
	assert.notStrictEqual(compiler.getGraphAnalysis().current, successful);
	assert.equal(compiler.getGraphAnalysis().current.state, "active");
	assert.deepEqual(compiler.getGraphAnalysis().current.executionOverlay, {
		skippedNodeIds: [], resourceAliases: [],
	});
	compiler.abort(new Error("cached frame failed"));
	assert.strictEqual(compiler.getGraphAnalysis().lastSuccessful, successful);
	assert.equal(compiler.getGraphAnalysis().lastAttempt.state, "aborted");

	for (const mutate of [
		(next) => { next.resources[0].width = 32; },
		(next) => { next.resources[0].format = "rgba16float"; },
		(next) => { next.bindings[0].generation = 2; },
		(next) => { next.bindings[0].physicalId = "physical:replacement"; },
		(next) => { next.stages[0].pass.dependsOn = ["shadow"]; },
		(next) => { next.stages[0].nodes[0].packetIndex = 1; },
		(next) => { next.stages[0].nodes[0].reads[0].optional = true; },
		(next) => { next.exports = [{ name: "color", resource: "color" }]; },
		(next) => { next.completeness = "opaque"; },
	]) {
		const next = structuredClone(plan);
		mutate(next);
		const changed = compiler.compileFrame(next);
		assert.notStrictEqual(changed, first);
		assert.strictEqual(compiler.compileFrame(structuredClone(next)), changed);
		assert.notStrictEqual(compiler.compileFrame(plan), changed);
	}
}

function run() {
	testReadBeforeCreateDiagnostic();
	testDuplicateCreateDiagnostic();
	testMissingRequiredResourceDiagnostic();
	testTextureFeedbackLoopDiagnostic();
	testUnsupportedUsageDiagnostic();
	testNodeOrderAndBarriersArePreserved();
	testGroupedAnalysisCommitBoundary();
	testWholeFrameCacheInvalidatesCompletePlanAndResetsAttempts();
	console.log("WebGL frame graph compiler tests passed");
}

run();
