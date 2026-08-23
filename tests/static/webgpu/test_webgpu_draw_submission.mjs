import assert from "node:assert/strict";
import {
	getDefaultWebGPUDrawBindings,
	submitWebGPUDraws,
} from "../../../src/backends/webgpu/WebGPUDrawSubmission.ts";
import {
	resolveWebGPUScenePassDescriptor,
} from "../../../src/backends/webgpu/WebGPUScenePassDescriptors.ts";
import { resolveWebGPUPlanarReflectionPassDescriptor } from "../../../src/backends/webgpu/WebGPUPlanarReflectionPassDescriptor.ts";

function createDrawResource(id) {
	return {
		pipeline: { id: `pipeline-${id}` },
		frameBinding: { id: `frame-${id}` },
		modelBinding: { id: `model-${id}` },
		clusteredBinding: { id: `clustered-${id}` },
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
		setBindingGroup(slot, group) {
			calls.push(["setBindingGroup", slot, group.id]);
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
	const packets = [{ id: "instance-a" }, { id: "instance-b" }];
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
	const packets = Array.from({ length: 5_000 }, (_, index) => ({
		id: `mesh:${index}`,
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

async function testDefaultSubmissionFiltersDirtyRectsAndTracksPackets() {
	const encoder = createEncoder();
	const options = [];
	const resources = {
		async getDrawResources(packet, _frameResources, drawOptions) {
			options.push([packet.id, drawOptions]);
			if (packet.id === "skip") {
				return null;
			}
			return [createDrawResource(packet.id)];
		},
	};
	const packets = [
		{ id: "a" },
		{ id: "skip" },
		{ id: "b" },
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
				candidatePackets.filter((packet) => packet.id !== "b")
			:	candidatePackets.filter((packet) => packet.id === "b"),
		resolveDrawOptions: (packet, rect) => ({
			sceneTargetMode: packet.id === "a" ? "gbuffer" : "mrt",
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
			return [createDrawResource(packet.id)];
		},
	};

	await submitWebGPUDraws({
		encoder,
		resources,
		frameResources: createFrameResources(),
		packets: [{ id: "gbuffer" }],
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
		packets: [{ id: "planar" }],
		resolveDrawOptions: () => ({ sampleCount: 1 }),
		resolveBindings: (draw) => [
			{ slot: 0, group: draw.frameBinding },
			{ slot: 1, group: draw.modelBinding },
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
	const packet = { id: "prepared" };
	const preparedDraw = createDrawResource(packet.id);
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
	assert.deepEqual([...result.submittedPacketIds], [packet.id]);
	assert.equal(
		encoder.calls.filter((call) => call[0] === "setPipeline").length,
		1
	);
}

async function testEmptyDirtyRectsUseFullFramePreparation() {
	const encoder = createEncoder();
	const packet = { id: "empty-rects" };
	let preparations = 0;
	const result = await submitWebGPUDraws({
		encoder,
		resources: {
			async getDrawResources() {
				preparations++;
				return [createDrawResource(packet.id)];
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

function testScenePassDescriptorsExposePipelineStateKeyParts() {
	const gbuffer = resolveWebGPUScenePassDescriptor(
		"gbuffer",
		"default",
		"early-z-color"
	);
	assert.equal(gbuffer.pipelineLayoutKind, "scene-gbuffer");
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
testScenePassDescriptorsExposePipelineStateKeyParts();
console.log("WebGPU draw submission tests passed");
