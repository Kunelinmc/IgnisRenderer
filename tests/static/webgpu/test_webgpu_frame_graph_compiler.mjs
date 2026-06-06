import assert from "node:assert/strict";
import { WebGPUFrameGraphCompiler } from "../../../src/renderers/webgpu/rendergraph/WebGPUFrameGraphCompiler.ts";

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

function testOptionalReadDoesNotDiagnose() {
	const { stage } = compile([{
		id: "test:optional-read",
		stage: "main-transparent",
		kind: "transparent-scene",
		label: "OptionalRead",
		reads: [{
			id: "frame:depth",
			usage: "depth-attachment",
			optional: true,
		}],
	}]);

	assert.equal(stage.diagnostics.length, 0);
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

function testDeferredDecalNodeRecordsGBufferTransitions() {
	const { compiler, stage } = compile([
		{
			id: "main-opaque:opaque-scene",
			stage: "main-opaque",
			kind: "opaque-scene",
			label: "WebGPUGBuffer",
			writes: [
				{
					id: "gbuffer:albedo-alpha",
					usage: "render-attachment",
				},
				{
					id: "gbuffer:material-ext0",
					usage: "storage-binding",
				},
			],
		},
		{
			id: "main-opaque:deferred-decal",
			stage: "main-opaque",
			kind: "deferred-decal",
			label: "WebGPUDeferredDecal",
			reads: [
				{
					id: "gbuffer:albedo-alpha",
					usage: "copy-src",
				},
				{
					id: "gbuffer:material-ext0",
					usage: "copy-src",
				},
			],
			writes: [
				{
					id: "gbuffer:albedo-alpha",
					usage: "render-attachment",
				},
				{
					id: "gbuffer:material-ext0",
					usage: "storage-binding",
				},
			],
		},
		{
			id: "main-opaque:deferred-lighting",
			stage: "main-opaque",
			kind: "deferred-lighting",
			label: "WebGPUDeferredLighting",
			reads: [
				{
					id: "gbuffer:albedo-alpha",
					usage: "texture-binding",
				},
				{
					id: "gbuffer:material-ext0",
					usage: "texture-binding",
				},
			],
			writes: [{
				id: "frame:scene-color-main",
				usage: "render-attachment",
			}],
		},
	]);

	const albedoBarriers = stage.barriers.filter(
		(barrier) => barrier.resource === "gbuffer:albedo-alpha"
	);
	assert.deepEqual(
		albedoBarriers.map((barrier) => ({
			beforeNodeId: barrier.beforeNodeId,
			nodeId: barrier.nodeId,
			reason: barrier.reason,
			fromUsage: barrier.fromUsage,
			toUsage: barrier.toUsage,
		})),
		[
			{
				beforeNodeId: "main-opaque:opaque-scene",
				nodeId: "main-opaque:deferred-decal",
				reason: "read-after-write",
				fromUsage: "render-attachment",
				toUsage: "copy-src",
			},
			{
				beforeNodeId: "main-opaque:deferred-decal",
				nodeId: "main-opaque:deferred-decal",
				reason: "write-after-read",
				fromUsage: "copy-src",
				toUsage: "render-attachment",
			},
			{
				beforeNodeId: "main-opaque:deferred-decal",
				nodeId: "main-opaque:deferred-lighting",
				reason: "read-after-write",
				fromUsage: "render-attachment",
				toUsage: "texture-binding",
			},
		]
	);
	const materialExtBarriers = stage.barriers.filter(
		(barrier) => barrier.resource === "gbuffer:material-ext0"
	);
	assert.deepEqual(
		materialExtBarriers.map((barrier) => ({
			beforeNodeId: barrier.beforeNodeId,
			nodeId: barrier.nodeId,
			reason: barrier.reason,
			fromUsage: barrier.fromUsage,
			toUsage: barrier.toUsage,
		})),
		[
			{
				beforeNodeId: "main-opaque:opaque-scene",
				nodeId: "main-opaque:deferred-decal",
				reason: "read-after-write",
				fromUsage: "storage-binding",
				toUsage: "copy-src",
			},
			{
				beforeNodeId: "main-opaque:deferred-decal",
				nodeId: "main-opaque:deferred-decal",
				reason: "write-after-read",
				fromUsage: "copy-src",
				toUsage: "storage-binding",
			},
			{
				beforeNodeId: "main-opaque:deferred-decal",
				nodeId: "main-opaque:deferred-lighting",
				reason: "read-after-write",
				fromUsage: "storage-binding",
				toUsage: "texture-binding",
			},
		]
	);
	const albedoState = compiler
		.getResourceDebugState()
		.find((resource) => resource.id === "gbuffer:albedo-alpha");
	assert.ok(albedoState);
	assert.equal(albedoState.lastNodeId, "main-opaque:deferred-lighting");
	assert.equal(albedoState.lastAccess, "read");
	assert.equal(albedoState.lastUsage, "texture-binding");
}

function run() {
	testReadBeforeCreateDiagnostic();
	testOptionalReadDoesNotDiagnose();
	testOptionalReadDoesNotCreateResource();
	testOptionalReadDoesNotEmitBarrierForLaterWrite();
	testUsageTransitionsEmitBarriers();
	testResourceDebugStateTracksLastAccess();
	testDeferredDecalNodeRecordsGBufferTransitions();
	console.log("test_webgpu_frame_graph_compiler: ok");
}

run();
