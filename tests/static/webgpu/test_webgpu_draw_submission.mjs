import assert from "node:assert/strict";
import { createTestDrawPacket } from "../helpers/drawPacket.mjs";

const drawPacket = (id, extras = {}) =>
	Object.assign(createTestDrawPacket({ id }), extras);
import {
	getDefaultWebGPUDrawBindings,
	submitWebGPUDraws,
} from "../../../src/backends/webgpu/WebGPUDrawSubmission.ts";
import {
	resolveWebGPUScenePassDescriptor,
} from "../../../src/backends/webgpu/WebGPUScenePassDescriptors.ts";
import { resolveWebGPUPlanarReflectionPassDescriptor } from "../../../src/backends/webgpu/WebGPUPlanarReflectionPassDescriptor.ts";
import { WebGPUDirtyRectResolver } from "../../../src/backends/webgpu/rendergraph/WebGPUDirtyRectResolver.ts";
import { PreparedSceneTileSpatialIndex } from "../../../src/pipeline/PreparedSceneSpatialIndex.ts";

function createDrawResource(id) {
	return {
		pipeline: { id: `pipeline-${id}` },
		bindings: [
			{ slot: 0, group: { id: `frame-${id}` } },
			{ slot: 1, group: { id: `model-${id}` } },
			{ slot: 2, group: { id: `clustered-${id}` } },
		],
		vertexBuffer: { id: `vertex-${id}` },
		indexBuffer: { id: `index-${id}` },
		indexCount: 3,
	};
}

function createEncoder() {
	const calls = [];
	return {
		calls,
		setPipeline(pipeline) {
			calls.push(["setPipeline", pipeline.id]);
		},
		setBindingGroup(slot, group, dynamicOffsets) {
			calls.push(dynamicOffsets
				? ["setBindingGroup", slot, group.id, [...dynamicOffsets]]
				: ["setBindingGroup", slot, group.id]);
		},
		setVertexBuffer(slot, buffer) {
			calls.push(["setVertexBuffer", slot, buffer.id]);
		},
		setIndexBuffer(buffer, format) {
			calls.push(["setIndexBuffer", buffer.id, format]);
		},
		drawIndexed(indexCount, instanceCount, firstIndex, baseVertex, firstInstance) {
			calls.push([
				"drawIndexed",
				indexCount,
				instanceCount,
				firstIndex,
				baseVertex,
				firstInstance,
			]);
		},
		setScissorRect(x, y, width, height) {
			calls.push(["setScissorRect", x, y, width, height]);
		},
	};
}

async function testStaticDrawsMergeContiguousInstanceRanges() {
	const encoder = createEncoder();
	const packets = [drawPacket("instance-a"), drawPacket("instance-b")];
	const shared = createDrawResource("shared");
	const prepared = new Map(packets.map((packet, index) => [packet, [{
		...shared,
		staticBatchKey: "batch:shared",
		firstInstance: index,
	}]]));
	const result = await submitWebGPUDraws({
		encoder,
		resources: {
			getDrawResources() {
				throw new Error("Static batch test must use prepared resources.");
			},
		},
		frameResources: createFrameResources(),
		packets,
		preparedResources: prepared,
		resolveDrawOptions() {
			throw new Error("Static batch test must not resolve draw options.");
		},
	});
	assert.equal(result.drawCount, 1);
	assert.deepEqual([...result.submittedPacketIds], ["instance-a", "instance-b"]);
	assert.deepEqual(
		encoder.calls.find((call) => call[0] === "drawIndexed"),
		["drawIndexed", 3, 2, 0, 0, 0],
	);
}

async function testFiveThousandStaticDrawsCollapseToEightBatches() {
	const encoder = createEncoder();
	const packets = Array.from({ length: 5_000 }, (_, index) =>
		drawPacket(`mesh:${index}`, {
			materialIndex: Math.floor(index / 625),
		}));
	const resources = Array.from({ length: 8 }, (_, index) =>
		createDrawResource(`material:${index}`));
	const prepared = new Map(packets.map((packet, index) => [packet, [{
		...resources[packet.materialIndex],
		staticBatchKey: `batch:${packet.materialIndex}`,
		firstInstance: index,
	}]]));
	const result = await submitWebGPUDraws({
		encoder,
		resources: {
			getDrawResources() {
				throw new Error("Static stress test must use prepared resources.");
			},
		},
		frameResources: createFrameResources(),
		packets,
		preparedResources: prepared,
		resolveDrawOptions() {
			throw new Error("Static stress test must not resolve draw options.");
		},
	});
	assert.equal(result.drawCount, 8);
	assert.equal(result.submittedPacketIds.size, 5_000);
}

async function submitStaticFixture(draws, options = {}) {
	const encoder = createEncoder();
	const packets = draws.map((_, index) => drawPacket(`instance:${index}`));
	const shared = new Map();
	const prepared = new Map(packets.map((packet, index) => {
		const entry = draws[index];
		if (!entry) return [packet, null];
		const [key, firstInstance] = entry;
		if (!shared.has(key)) shared.set(key, createDrawResource(key ?? "dynamic"));
		return [packet, [{ ...shared.get(key), staticBatchKey: key, firstInstance }]];
	}));
	const result = await submitWebGPUDraws({
		encoder,
		resources: {},
		frameResources: createFrameResources(),
		packets,
		preparedResources: prepared,
		resolveDrawOptions() {
			throw new Error("Prepared fixture must not resolve draw options.");
		},
		...options,
	});
	return { result, calls: encoder.calls };
}

async function testStaticBatchSplitReasonsPreserveDrawCommands() {
	const cases = [
		{ name: "empty", draws: [], splits: [0, 0], commands: [] },
		{
			name: "contiguous range",
			draws: [["A", 0], ["A", 1]], splits: [0, 0], commands: [[2, 0]],
		},
		{
			name: "gap after merged range",
			draws: [["A", 0], ["A", 1], ["A", 3], ["A", 4]],
			splits: [0, 1], commands: [[2, 0], [2, 3]],
		},
		{
			name: "duplicate index",
			draws: [["A", 0], ["A", 0]], splits: [0, 1], commands: [[1, 0], [1, 0]],
		},
		{
			name: "descending index",
			draws: [["A", 2], ["A", 0]], splits: [0, 1], commands: [[1, 2], [1, 0]],
		},
		{
			name: "first batch of another key is not a split",
			draws: [["A", 0], ["B", 1]], splits: [0, 0], commands: [[1, 0], [1, 1]],
		},
		{
			name: "non-adjacent key with contiguous indices",
			draws: [["A", 0], ["B", 8], ["A", 1]],
			splits: [1, 0], commands: [[1, 0], [1, 8], [1, 1]],
		},
		{
			name: "non-adjacency takes precedence over an index gap",
			draws: [["A", 0], ["B", 1], ["A", 2], ["A", 3]],
			splits: [1, 0], commands: [[1, 0], [1, 1], [2, 2]],
		},
		{
			name: "each later run counts once",
			draws: [["A", 0], ["A", 2], ["B", 3], ["A", 4], ["B", 5]],
			splits: [2, 1], commands: [[1, 0], [1, 2], [1, 3], [1, 4], [1, 5]],
		},
		{
			name: "dynamic draw separates static batches",
			draws: [["A", 0], [undefined, 0], ["A", 1]],
			splits: [1, 0], commands: [[1, 0], [1, 0], [1, 1]],
		},
		{
			name: "skipped draw does not separate commands",
			draws: [["A", 0], null, ["A", 1]], splits: [0, 0], commands: [[2, 0]],
		},
	];
	for (const { name, draws, splits, commands } of cases) {
		const { result, calls } = await submitStaticFixture(draws);
		assert.deepEqual(result.staticBatching, {
			nonAdjacentKeySplits: splits[0],
			nonContiguousInstanceSplits: splits[1],
		}, name);
		assert.equal(result.drawCount, commands.length, name);
		assert.deepEqual(
			calls.filter((call) => call[0] === "drawIndexed")
				.map((call) => [call[2], call[5]]),
			commands,
			name,
		);
	}
}

async function testStaticBatchSplitsAreLocalToEachDirtyRectangle() {
	const { result, calls } = await submitStaticFixture(
		[["A", 0], ["B", 1], ["A", 2], ["A", 3]],
		{
			dirtyRects: [
				{ x: 0, y: 0, width: 8, height: 8 },
				{ x: 8, y: 0, width: 8, height: 8 },
				{ x: 16, y: 0, width: 8, height: 8 },
			],
			selectPacketsForRect: (packets, rect) => rect.x === 0
				? packets : rect.x === 8 ? [packets[0], packets[2], packets[3]] : [],
		},
	);
	assert.deepEqual(result.staticBatching, {
		nonAdjacentKeySplits: 1,
		nonContiguousInstanceSplits: 1,
	});
	assert.equal(result.drawCount, 5);
	assert.deepEqual(
		calls.filter((call) => call[0] === "drawIndexed").map((call) => [call[2], call[5]]),
		[[1, 0], [1, 1], [2, 2], [1, 0], [2, 2]],
	);
}

function createFrameResources() {
	return {
		scopeKey: "test",
		sceneTargetMode: "mrt",
		frameBinding: { id: "frame" },
		decalFrameBinding: { id: "decal-frame" },
		environmentBinding: { id: "environment" },
		clusteredSceneBinding: { id: "clustered" },
	};
}

function createSpatialContext(opaquePackets, transparentPackets, packetRects) {
	return { scene: {
		opaquePackets, transparentPackets,
		spatialIndex: new PreparedSceneTileSpatialIndex({
			viewportWidth: 100, viewportHeight: 100, tileSize: 16,
			opaquePackets, transparentPackets, packetRects,
		}),
	} };
}

function testDirtyRectSelectionPreservesCallerSubsetAndOrder() {
	const inside = drawPacket("inside");
	const outside = drawPacket("outside");
	const fallback = drawPacket("unknown-bounds");
	const transparentInside = drawPacket("transparent-inside");
	const transparentOutside = drawPacket("transparent-outside");
	const transient = drawPacket("transient-not-indexed");
	const rect = { x: 0, y: 0, width: 20, height: 20 };
	const away = { x: 70, y: 70, width: 10, height: 10 };
	const opaque = [inside, outside, fallback];
	const transparent = [transparentInside, transparentOutside];
	const context = createSpatialContext(opaque, transparent, new Map([
		[inside.submission.id, rect], [outside.submission.id, away],
		[transparentInside.submission.id, rect], [transparentOutside.submission.id, away],
	]));
	const cases = [
		{ packets: opaque, expected: [inside, fallback] },
		{ packets: opaque.slice(), expected: [inside, fallback] },
		{ packets: transparent, expected: [transparentInside] },
		{ packets: transparent.slice(), expected: [transparentInside] },
		{ packets: [fallback, outside], expected: [fallback] },
		{ packets: [outside], expected: [] },
		{ packets: [], expected: [] },
		{
			packets: [transparentOutside, fallback, transparentInside, inside, transient, outside, inside],
			expected: [fallback, transparentInside, inside, transient, inside],
		},
	];
	for (const { packets, expected } of cases) {
		const before = packets.slice();
		assert.deepEqual(WebGPUDirtyRectResolver.selectPacketsForRect(context, packets, rect), expected);
		assert.deepEqual(packets, before, "Selection must not mutate the caller's candidates.");
	}
	assert.deepEqual(WebGPUDirtyRectResolver.selectPacketsForRect(
		{ scene: { ...context.scene, spatialIndex: null } }, [outside, transient], rect,
	), [outside, transient]);
}

async function testCopiedOpaqueDirtyRectsDoNotBatchExcludedInstances() {
	const packets = [drawPacket("left-a"), drawPacket("right"), drawPacket("left-b")];
	const rect = { x: 0, y: 0, width: 20, height: 20 };
	const context = createSpatialContext(packets, [], new Map([
		["left-a", rect], ["right", { x: 70, y: 70, width: 10, height: 10 }], ["left-b", rect],
	]));
	const shared = createDrawResource("shared");
	const encoder = createEncoder();
	const result = await submitWebGPUDraws({
		encoder, resources: {}, frameResources: createFrameResources(), packets: packets.slice(),
		preparedResources: new Map(packets.map((packet, firstInstance) => [packet, [{
			...shared, staticBatchKey: "shared", firstInstance,
		}]])),
		dirtyRects: [rect],
		selectPacketsForRect: (candidates, dirtyRect) =>
			WebGPUDirtyRectResolver.selectPacketsForRect(context, candidates, dirtyRect),
		resolveDrawOptions() { throw new Error("Prepared draws must be reused."); },
	});
	assert.deepEqual([...result.submittedPacketIds], ["left-a", "left-b"]);
	assert.deepEqual(result.staticBatching, {
		nonAdjacentKeySplits: 0, nonContiguousInstanceSplits: 1,
	});
	assert.deepEqual(encoder.calls.filter((call) => call[0] === "drawIndexed"), [
		["drawIndexed", 3, 1, 0, 0, 0], ["drawIndexed", 3, 1, 0, 0, 2],
	]);
}

async function testDefaultSubmissionFiltersDirtyRectsAndTracksPackets() {
	const encoder = createEncoder();
	const options = [];
	const resources = {
		async getDrawResources(packet, _frameResources, drawOptions) {
			options.push([packet.submission.id, drawOptions]);
			if (packet.submission.id === "skip") {
				return null;
			}
			return [createDrawResource(packet.submission.id)];
		},
	};
	const packets = [
		drawPacket("a"),
		drawPacket("skip"),
		drawPacket("b"),
	];

	const result = await submitWebGPUDraws({
		encoder,
		resources,
		frameResources: createFrameResources(),
		packets,
		dirtyRects: [
			{ x: 0, y: 0, width: 8, height: 8 },
			{ x: 8, y: 0, width: 8, height: 8 },
		],
		selectPacketsForRect: (candidatePackets, rect) =>
			rect.x === 0 ?
				candidatePackets.filter((packet) => packet.submission.id !== "b")
			:	candidatePackets.filter((packet) => packet.submission.id === "b"),
		resolveDrawOptions: (packet, rect) => ({
			sceneTargetMode: packet.submission.id === "a" ? "gbuffer" : "mrt",
			drawMode: rect?.x === 0 ? "early-z-prepass" : "default",
			sampleCount: 1,
		}),
	});

	assert.equal(result.drawCount, 2);
	assert.deepEqual([...result.submittedPacketIds].sort(), ["a", "b"]);
	assert.deepEqual(
		options.map(([id, drawOptions]) => [
			id,
			drawOptions.sceneTargetMode,
			drawOptions.drawMode,
		]),
		[
			["a", "gbuffer", "early-z-prepass"],
			["skip", "mrt", "early-z-prepass"],
			["b", "mrt", "default"],
		]
	);
	assert.deepEqual(
		encoder.calls.filter((call) => call[0] === "setBindingGroup").slice(0, 3),
		[
			["setBindingGroup", 0, "frame-a"],
			["setBindingGroup", 1, "model-a"],
			["setBindingGroup", 2, "clustered-a"],
		]
	);
}

async function testSubmissionSupportsExtraAndReplacementBindings() {
	const encoder = createEncoder();
	const resources = {
		async getDrawResources(packet) {
			return [createDrawResource(packet.submission.id)];
		},
	};

	await submitWebGPUDraws({
		encoder,
		resources,
		frameResources: createFrameResources(),
		packets: [drawPacket("gbuffer")],
		resolveDrawOptions: () => ({ sampleCount: 1 }),
		resolveBindings: (draw) => [
			...getDefaultWebGPUDrawBindings(draw),
			{ slot: 3, group: { id: "gbuffer-write" } },
		],
	});
	await submitWebGPUDraws({
		encoder,
		resources,
		frameResources: createFrameResources(),
		packets: [drawPacket("planar")],
		resolveDrawOptions: () => ({ sampleCount: 1 }),
		resolveBindings: (draw) => [
			...draw.bindings.filter((binding) => binding.slot < 2),
			{ slot: 2, group: { id: "reflection-texture" } },
		],
	});

	assert.deepEqual(
		encoder.calls.filter((call) => call[0] === "setBindingGroup"),
		[
			["setBindingGroup", 0, "frame-gbuffer"],
			["setBindingGroup", 1, "model-gbuffer"],
			["setBindingGroup", 2, "clustered-gbuffer"],
			["setBindingGroup", 3, "gbuffer-write"],
			["setBindingGroup", 0, "frame-planar"],
			["setBindingGroup", 1, "model-planar"],
			["setBindingGroup", 2, "reflection-texture"],
		]
	);
}

async function testSubmissionReusesPreparedResourcesAcrossDirtyRects() {
	const encoder = createEncoder();
	const packet = drawPacket("prepared");
	const preparedDraw = createDrawResource(packet.submission.id);
	const resources = {
		async getDrawResources() {
			throw new Error(
				"Prepared submission must not resolve draw resources again."
			);
		},
	};

	const result = await submitWebGPUDraws({
		encoder,
		resources,
		frameResources: createFrameResources(),
		packets: [packet],
		preparedResources: new Map([[packet, [preparedDraw]]]),
		dirtyRects: [
			{ x: 0, y: 0, width: 8, height: 8 },
			{ x: 8, y: 0, width: 8, height: 8 },
		],
		resolveDrawOptions: () => {
			throw new Error("Prepared submission must not resolve draw options again.");
		},
	});

	assert.equal(result.drawCount, 2);
	assert.deepEqual([...result.submittedPacketIds], [packet.submission.id]);
	assert.equal(
		encoder.calls.filter((call) => call[0] === "setPipeline").length,
		1
	);
	assert.equal(
		encoder.calls.filter((call) => call[0] === "setBindingGroup" && call[1] === 1).length,
		1,
		"repeated dirty rects reuse the same model group and offset"
	);
}

async function testEmptyDirtyRectsUseFullFramePreparation() {
	const encoder = createEncoder();
	const packet = drawPacket("empty-rects");
	let preparations = 0;
	const result = await submitWebGPUDraws({
		encoder,
		resources: {
			async getDrawResources() {
				preparations++;
				return [createDrawResource(packet.submission.id)];
			},
		},
		frameResources: createFrameResources(),
		packets: [packet],
		dirtyRects: [],
		resolveDrawOptions: (_packet, rect) => ({
			sampleCount: 1,
			drawMode: rect === null ? "default" : "early-z-prepass",
		}),
	});

	assert.equal(preparations, 1);
	assert.equal(result.drawCount, 1);
}

async function testSharedModelGroupRebindsPerDynamicOffset() {
	const encoder = createEncoder();
	const frame = { id: "shared-frame" };
	const model = { id: "shared-model" };
	const clustered = { id: "shared-clustered" };
	const packets = [drawPacket("slot-a"), drawPacket("slot-b")];
	const resources = {
		async getDrawResources(packet) {
			return [{
				pipeline: { id: "shared-pipeline" },
				bindings: [
					{ slot: 0, group: frame },
					{
						slot: 1,
						group: model,
						dynamicOffsets: [packet.submission.id === "slot-a" ? 0 : 256],
					},
					{ slot: 2, group: clustered },
				],
				vertexBuffer: { id: "shared-vertex" },
				indexBuffer: { id: "shared-index" },
				indexCount: 3,
			}];
		},
	};
	await submitWebGPUDraws({
		encoder,
		resources,
		frameResources: createFrameResources(),
		packets,
		resolveDrawOptions: () => ({ sampleCount: 1 }),
	});
	assert.deepEqual(
		encoder.calls.filter((call) => call[0] === "setBindingGroup" && call[1] === 1),
		[
			["setBindingGroup", 1, "shared-model", [0]],
			["setBindingGroup", 1, "shared-model", [256]],
		],
		"shared model groups must rebind when the object-arena offset changes",
	);
	assert.equal(
		encoder.calls.filter((call) =>
			call[0] === "setBindingGroup" && call[2] === "shared-frame").length,
		1,
		"unchanged groups without dynamic offsets stay deduplicated",
	);
}

function testScenePassDescriptorsExposePipelineStateKeyParts() {
	const gbuffer = resolveWebGPUScenePassDescriptor(
		"gbuffer",
		"default",
		"early-z-color"
	);
	assert.equal(gbuffer.pipelineLayoutKind, "scene");
	assert.equal(gbuffer.fragmentTargetKind, "gbuffer");
	assert.equal(gbuffer.shaderEntryMode, "gbuffer");
	assert.equal(gbuffer.depthStateMode, "early-z-color");
	assert.equal(gbuffer.sampleCountMode, "single-sample");
	assert.equal(gbuffer.depthFormatMode, "depth32float");
	assert.ok(gbuffer.pipelineKeyPart.includes("gbuffer|default|early-z-color"));

	const oit = resolveWebGPUScenePassDescriptor("mrt", "oit", "default");
	assert.equal(oit.fragmentTargetKind, "oit");
	assert.equal(oit.shaderEntryMode, "oit");
	assert.equal(oit.sampleCountMode, "mrt-msaa");

	const transmissionCapture = resolveWebGPUScenePassDescriptor(
		"mrt",
		"transmission-capture",
		"default"
	);
	assert.equal(transmissionCapture.fragmentTargetKind, "transmission-capture");
	assert.equal(transmissionCapture.shaderEntryMode, "transmission-capture");
	assert.ok(
		transmissionCapture.pipelineKeyPart.includes(
			"targets:transmission-capture"
		)
	);

	const planar = resolveWebGPUPlanarReflectionPassDescriptor("mrt");
	assert.equal(planar.pipelineLayoutKind, "planar-reflection");
	assert.equal(planar.fragmentTargetKind, "planar-reflection");
	assert.equal(planar.depthStateMode, "planar-reflection");
}

await testDefaultSubmissionFiltersDirtyRectsAndTracksPackets();
await testSubmissionSupportsExtraAndReplacementBindings();
await testSubmissionReusesPreparedResourcesAcrossDirtyRects();
await testEmptyDirtyRectsUseFullFramePreparation();
await testStaticDrawsMergeContiguousInstanceRanges();
await testFiveThousandStaticDrawsCollapseToEightBatches();
await testStaticBatchSplitReasonsPreserveDrawCommands();
await testStaticBatchSplitsAreLocalToEachDirtyRectangle();
testDirtyRectSelectionPreservesCallerSubsetAndOrder();
await testCopiedOpaqueDirtyRectsDoNotBatchExcludedInstances();
await testSharedModelGroupRebindsPerDynamicOffset();
testScenePassDescriptorsExposePipelineStateKeyParts();
console.log("WebGPU draw submission tests passed");
