import assert from "node:assert/strict";
import { WebGPUFrameGraphPlanner } from "../../../src/renderers/webgpu/rendergraph/WebGPUFrameGraphPlanner.ts";

function createPass(stage) {
	return {
		stage,
		executor: "backend",
		enabled: true,
		dependsOn: [],
	};
}

function createState(overrides = {}) {
	return {
		deferredActive: false,
		oitActive: false,
		sceneTargetMode: "mrt",
		hasFrameTargets: true,
		hasMSAATargets: false,
		needsPlanarReflectionMask: false,
		...overrides,
	};
}

function run() {
	const planner = new WebGPUFrameGraphPlanner();
	const context = { scene: { decalPackets: [] } };
	const contextWithDecals = { scene: { decalPackets: [{}] } };
	const pagedShadowContext = {
		backendProfile: {
			shadow: {
				supportsPagedShadows: true,
			},
		},
		shadowMaps: new Map([
			[
				{ id: "light-0" },
				{
					storageMode: "paged",
				},
			],
		]),
		scene: { decalPackets: [] },
	};

	const shadow = planner.planStage(
		createPass("shadow"),
		context,
		createState()
	);
	assert.deepEqual(
		shadow.nodes.map((node) => node.kind),
		["shadow"]
	);

	const pagedShadow = planner.planStage(
		createPass("shadow"),
		pagedShadowContext,
		createState()
	);
	assert.deepEqual(
		pagedShadow.nodes.map((node) => node.kind),
		[
			"shadow",
			"paged-shadow-page-mark",
			"paged-shadow-page-allocate",
			"paged-shadow-depth",
		]
	);
	assert.equal(
		pagedShadow.nodes[1].writes[0].id,
		"paged-shadow:page-requests"
	);
	assert.equal(
		pagedShadow.nodes[3].writes[0].id,
		"paged-shadow:physical-depth"
	);

	const opaque = planner.planStage(
		createPass("main-opaque"),
		contextWithDecals,
		createState({ deferredActive: true, sceneTargetMode: "gbuffer" })
	);
	assert.deepEqual(
		opaque.nodes.map((node) => node.kind),
		["opaque-scene", "deferred-decal", "deferred-lighting"]
	);
	assert.equal(opaque.nodes[0].label, "WebGPUGBuffer");
	assert.equal(opaque.nodes[1].label, "WebGPUDeferredDecal");
	assert.equal(opaque.nodes[1].reads.length, 11);
	assert.equal(opaque.nodes[1].writes.length, 11);
	assert.equal(
		opaque.nodes[1].writes.find(
			(write) => write.id === "gbuffer:albedo-alpha"
		)?.usage,
		"render-attachment"
	);
	assert.equal(
		opaque.nodes[1].writes.find(
			(write) => write.id === "gbuffer:material-ext0"
		)?.usage,
		"storage-binding"
	);
	assert.equal(opaque.nodes[2].label, "WebGPUDeferredLighting");
	assert.ok(
		opaque.nodes[2].reads.some(
			(read) => read.id === "gbuffer:albedo-alpha"
		)
	);

	const opaqueWithoutDecals = planner.planStage(
		createPass("main-opaque"),
		context,
		createState({ deferredActive: true, sceneTargetMode: "gbuffer" })
	);
	assert.deepEqual(
		opaqueWithoutDecals.nodes.map((node) => node.kind),
		["opaque-scene", "deferred-lighting"]
	);

	const opaqueWithOcclusion = planner.planStage(
		createPass("main-opaque"),
		contextWithDecals,
		createState({
			deferredActive: true,
			sceneTargetMode: "gbuffer",
			needsOcclusionTest: true,
		})
	);
	assert.deepEqual(
		opaqueWithOcclusion.nodes.map((node) => node.kind),
		[
			"opaque-scene",
			"deferred-decal",
			"deferred-lighting",
			"occlusion-test",
		]
	);
	assert.equal(
		opaqueWithOcclusion.nodes.at(-1).reads[0].id,
		"gbuffer:motion-depth"
	);

	const forwardWithOcclusion = planner.planStage(
		createPass("main-opaque"),
		context,
		createState({
			sceneTargetMode: "mrt",
			needsOcclusionTest: true,
		})
	);
	assert.deepEqual(
		forwardWithOcclusion.nodes.map((node) => node.kind),
		["opaque-scene", "occlusion-test"]
	);

	const transparent = planner.planStage(
		createPass("main-transparent"),
		context,
		createState({ oitActive: true })
	);
	assert.deepEqual(
		transparent.nodes.map((node) => node.kind),
		["oit-transparent"]
	);

	const unknown = planner.planStage(
		createPass("custom-pass"),
		context,
		createState()
	);
	assert.deepEqual(unknown.nodes, []);

	console.log("test_webgpu_frame_graph_planner: ok");
}

run();
