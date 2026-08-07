import assert from "node:assert/strict";
import { WebGPUDeferredFrameModule } from "../../../src/backends/webgpu/rendergraph/WebGPUDeferredFrameModule.ts";
import { WebGPUReflectionFrameModule } from "../../../src/backends/webgpu/rendergraph/WebGPUReflectionFrameModule.ts";
import { WebGPUSceneFrameModule } from "../../../src/backends/webgpu/rendergraph/WebGPUSceneFrameModule.ts";
import { WebGPUShadowFrameModule } from "../../../src/backends/webgpu/rendergraph/WebGPUShadowFrameModule.ts";
import { WebGPUTransparencyRuntime } from "../../../src/backends/webgpu/rendergraph/WebGPUTransparencyRuntime.ts";
import { WebGPUVisibilityFrameModule } from "../../../src/backends/webgpu/rendergraph/WebGPUVisibilityFrameModule.ts";

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

function createModulePlanner() {
	const unused = {};
	const modules = [
		new WebGPUSceneFrameModule(unused, unused),
		new WebGPUDeferredFrameModule(unused, unused, unused, unused),
		new WebGPUReflectionFrameModule(unused, unused, unused),
		new WebGPUVisibilityFrameModule(unused, unused, unused),
		new WebGPUShadowFrameModule(unused, unused),
		new WebGPUTransparencyRuntime(unused, unused, unused, unused, unused, unused),
	];
	return {
		planStage(pass, context, state) {
			const contributions = modules
				.flatMap((module) => module.planStage?.({
					pass,
					context,
					state,
					moduleState: {},
				}) ?? [])
				.sort((a, b) => a.order - b.order);
			return {
				pass,
				nodes: contributions.flatMap((contribution) => contribution.nodes ?? []),
			};
		},
	};
}

function run() {
	const planner = createModulePlanner();
	const context = { scene: { decalPackets: [] } };
	const contextWithDecals = { scene: { decalPackets: [{}] } };
	const pagedShadowContext = {
		backendProfile: {
			shadow: {
				supportsPagedShadows: true,
				supportsPagedShadowRendering: true,
			},
		},
		shadowMaps: new Map([
			[
				{ id: "light-0" },
				{
					storageMode: "paged",
					layout: {
						paged: {
							feedbackMode: "screen-feedback",
						},
					},
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
			"paged-shadow-page-table-copy",
			"paged-shadow-depth",
		]
	);
	assert.equal(
		pagedShadow.nodes[1].writes[0].id,
		"paged-shadow:page-request-flags"
	);
	assert.ok(
		pagedShadow.nodes[1].writes.some(
			(write) => write.id === "paged-shadow:page-requests"
		)
	);
	assert.ok(
		pagedShadow.nodes[2].writes.some(
			(write) => write.id === "paged-shadow:dirty-physical-pages"
		)
	);
	assert.ok(
		pagedShadow.nodes[4].writes.some(
			(write) => write.id === "paged-shadow:draw-instances"
		)
	);
	assert.ok(
		pagedShadow.nodes[4].writes.some(
			(write) => write.id === "paged-shadow:draw-indirect-args"
		)
	);
	assert.ok(
		pagedShadow.nodes[4].writes.some(
			(write) => write.id === "paged-shadow:clear-draw-indirect-args"
		)
	);
	assert.ok(
		pagedShadow.nodes[4].writes.some(
			(write) => write.id === "paged-shadow:physical-depth"
		)
	);
	assert.equal(
		pagedShadow.nodes[4].writes.some(
			(write) => write.id === "paged-shadow:physical-transmittance"
		),
		false
	);

	const pagedOpaque = planner.planStage(
		createPass("main-opaque"),
		pagedShadowContext,
		createState()
	);
	assert.ok(
		pagedOpaque.nodes[0].reads.some(
			(read) => read.id === "paged-shadow:page-table-texture"
		)
	);
	assert.ok(
		pagedOpaque.nodes[0].reads.some(
			(read) => read.id === "paged-shadow:physical-depth"
		)
	);
	assert.equal(
		pagedOpaque.nodes[pagedOpaque.nodes.length - 1].kind,
		"paged-shadow-feedback"
	);
	assert.ok(
		pagedOpaque.nodes[pagedOpaque.nodes.length - 1].writes.some(
			(write) => write.id === "paged-shadow:next-feedback-flags"
		)
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
	assert.equal(opaque.nodes[1].reads.length, 9);
	assert.equal(opaque.nodes[1].writes.length, 9);
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
	const baseOpaque = planner.planStage(
		createPass("main-opaque"),
		context,
		createState({
			deferredActive: true,
			sceneTargetMode: "gbuffer",
			deferredGBufferLayout: "base",
		})
	);
	assert.equal(
		baseOpaque.nodes[0].writes.some(
			(write) => write.id === "gbuffer:specular"
		),
		false
	);
	assert.equal(
		baseOpaque.nodes[1].reads.some(
			(read) => read.id === "gbuffer:specular"
		),
		false
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
			needsHiZBuild: true,
		})
	);
	assert.deepEqual(
		opaqueWithOcclusion.nodes.map((node) => node.kind),
		[
			"opaque-scene",
			"deferred-decal",
			"deferred-lighting",
			"hiz-build",
			"occlusion-test",
		]
	);
	assert.equal(
		opaqueWithOcclusion.nodes.at(-1).reads[0].id,
		"frame:hiz"
	);

	const forwardWithOcclusion = planner.planStage(
		createPass("main-opaque"),
		context,
		createState({
			sceneTargetMode: "mrt",
			needsOcclusionTest: true,
			needsHiZBuild: true,
		})
	);
	assert.deepEqual(
		forwardWithOcclusion.nodes.map((node) => node.kind),
		["opaque-scene", "hiz-build", "occlusion-test"]
	);

	const transparent = planner.planStage(
		createPass("main-transparent"),
		context,
		createState({ oitActive: true })
	);
	assert.deepEqual(
		transparent.nodes.map((node) => node.kind),
		[
			"oit-prepare",
			"oit-clear",
			"oit-mesh-accumulate",
			"oit-resolve",
			"transmission",
		]
	);

	const transmission = planner.planStage(
		createPass("main-transparent"),
		context,
		createState({
			hasOITMeshContributors: false,
			hasTransmissionPackets: true,
		})
	);
	assert.deepEqual(
		transmission.nodes.map((node) => node.kind),
		["transmission"]
	);
	assert.equal(
		transmission.nodes[0].reads.find(
			(read) => read.id === "frame:scene-color-main"
		)?.usage,
		"render-attachment"
	);

	const transmissionCapture = planner.planStage(
		createPass("main-transparent"),
		context,
		createState({
			hasOITMeshContributors: false,
			hasTransmissionPackets: true,
			needsTransmissionTargets: true,
		})
	);
	assert.deepEqual(
		transmissionCapture.nodes.map((node) => node.kind),
		["transmission"]
	);
	assert.equal(
		transmissionCapture.nodes[0].reads.find(
			(read) => read.id === "frame:scene-color-main"
		)?.usage,
		"copy-src"
	);
	assert.equal(
		transmissionCapture.nodes[0].writes.some(
			(write) =>
				write.id === "frame:scene-color-main" ||
				write.id === "frame:depth"
		),
		false
	);
	assert.ok(
		transmissionCapture.nodes[0].writes.some(
			(write) => write.id === "transmission:scene-color-copy"
		)
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
