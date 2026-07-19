import assert from "node:assert/strict";
import { WebGLFrameGraphCompiler } from "../../../src/renderers/webgl/rendergraph/WebGLFrameGraphCompiler.ts";

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
				kind: "postprocess",
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
				kind: "postprocess",
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

function run() {
	testReadBeforeCreateDiagnostic();
	testDuplicateCreateDiagnostic();
	testMissingRequiredResourceDiagnostic();
	testTextureFeedbackLoopDiagnostic();
	testUnsupportedUsageDiagnostic();
	testNodeOrderAndBarriersArePreserved();
	testGroupedAnalysisCommitBoundary();
	console.log("WebGL frame graph compiler tests passed");
}

run();
