import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import { LightType } from "../../../src/lights/index.ts";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import { BufferUsage } from "../../../src/renderers/types.ts";
import {
	WEBGPU_PAGED_SHADOW_NON_RESIDENT,
	WebGPUPagedShadowRuntime,
	collectWebGPUPagedShadowPageRequests,
} from "../../../src/renderers/webgpu/WebGPUPagedShadowRuntime.ts";

function createMockBackend() {
	const buffers = [];
	const textures = [];
	const writes = [];
	const bindingGroups = [];
	const pipelines = [];
	return {
		buffers,
		textures,
		writes,
		bindingGroups,
		pipelines,
		createBuffer(desc) {
			const buffer = {
				size: desc.size,
				usage: desc.usage,
				label: desc.label,
				data: null,
				destroyed: false,
				destroy() {
					this.destroyed = true;
				},
			};
			buffers.push(buffer);
			return buffer;
		},
		createTexture(desc) {
			const texture = {
				width: desc.width,
				height: desc.height,
				label: desc.label,
				destroyed: false,
				destroy() {
					this.destroyed = true;
				},
			};
			textures.push(texture);
			return texture;
		},
		writeBuffer(buffer, data) {
			buffer.data = new Uint32Array(data.buffer ?? data);
			writes.push([buffer.label, buffer.data]);
		},
		async createShaderModule(desc) {
			return { label: desc.label };
		},
		async createComputePipeline(desc) {
			const pipeline = { label: desc.label };
			pipelines.push(pipeline);
			return pipeline;
		},
		createBindingGroup(desc) {
			const group = { label: desc.label, entries: desc.entries };
			bindingGroups.push(group);
			return group;
		},
	};
}

function createEncoder() {
	const calls = [];
	return {
		calls,
		beginComputePass(desc) {
			calls.push(["beginComputePass", desc.label]);
		},
		setComputePipeline(pipeline) {
			calls.push(["setComputePipeline", pipeline.label]);
		},
		setBindingGroup(index, group) {
			calls.push(["setBindingGroup", index, group.label]);
		},
		dispatchWorkgroups(x, y = 1, z = 1) {
			calls.push(["dispatchWorkgroups", x, y, z]);
		},
		endComputePass() {
			calls.push(["endComputePass"]);
		},
	};
}

function createRenderSet(overrides = {}) {
	const paged = {
		virtualResolution: 4096,
		pageGridSize: 4,
		pageSize: 64,
		physicalPageCount: 2,
		maxPagesPerFrame: 1,
		cacheFrames: 0,
		feedbackMode: "screen-feedback",
		...overrides.paged,
	};
	return {
		storageMode: "paged",
		configSignature: overrides.configSignature ?? "paged-test",
		layout: {
			storageMode: "paged",
			regions: [],
			paged,
		},
		slices: [
			{
				splitNear: 0,
				splitFar: 1,
				shadowMap: {
					viewProjectionMatrix: Matrix4.identity(),
				},
			},
		],
	};
}

function createRequest(renderSet, packets, encoder = null) {
	const light = {
		id: "sun",
		type: LightType.Directional,
	};
	return {
		context: {
			features: { enableShadows: true },
			camera: {
				viewProjectionMatrix: Matrix4.identity(),
			},
			attachments: { width: 64, height: 64 },
			scene: {
				shadowCasterPackets: packets,
				shadowTransmitterPackets: [],
			},
		},
		encoder,
		renderSets: new Map([[light, renderSet]]),
		shadowCasterPackets: packets,
		shadowTransmitterPackets: [],
		feedbackDepthTexture: {
			width: 64,
			height: 64,
			label: "FeedbackDepth",
			destroy() {},
		},
		feedbackMotionDepthTexture: null,
	};
}

function createPacket(id, x, y, indexCount = 36) {
	return {
		id,
		worldMatrix: Matrix4.identity(),
		worldBounds: {
			center: { x, y, z: 0 },
			radius: 0.01,
		},
		geometry: {
			indices: new Uint32Array(indexCount),
		},
	};
}

function createLayout(renderSet) {
	return {
		renderSet,
		metadata: renderSet.layout.paged,
		pageTableBase: 0,
		pageTableCascadeStride:
			renderSet.layout.paged.pageGridSize * renderSet.layout.paged.pageGridSize,
		cascadeCount: 1,
	};
}

function getBuffer(backend, label) {
	return backend.buffers.find((buffer) => buffer.label === label);
}

function getLatestBuffer(backend, label) {
	return backend.buffers.findLast((buffer) => buffer.label === label);
}

function getLatestTexture(backend, label) {
	return backend.textures.findLast((texture) => texture.label === label);
}

function createPackets(count) {
	return Array.from({ length: count }, (_value, index) =>
		createPacket(`caster-${index}`, -0.75 + index * 0.01, 0.75)
	);
}

function testCpuRequesterRemainsAvailableForDiagnostics() {
	const renderSet = createRenderSet();
	const packets = [
		createPacket("left", -0.75, 0.75),
		createPacket("right", 0.75, 0.75),
	];
	const requests = collectWebGPUPagedShadowPageRequests(
		createRequest(renderSet, packets),
		[createLayout(renderSet)]
	);

	assert.equal(requests.length, 2);
	assert.deepEqual(
		requests.map((request) => [
			request.key,
			request.pageX,
			request.pageY,
			request.pageTableIndex,
		]),
		[
			["sun:paged-test:0:0:0", 0, 0, 0],
			["sun:paged-test:0:3:0", 3, 0, 3],
		]
	);
}

async function testRuntimeCreatesGpuAuthoritativeBuffers() {
	const backend = createMockBackend();
	const shadowPass = { renderPagedDepthIndirect() {} };
	const runtime = new WebGPUPagedShadowRuntime(backend, shadowPass);
	const renderSet = createRenderSet({
		paged: {
			physicalPageCount: 4,
			maxPagesPerFrame: 2,
		},
	});
	const request = createRequest(renderSet, [
		createPacket("left", -0.75, 0.75),
		createPacket("right", 0.75, 0.75),
	]);

	runtime.prepareFrame(request);

	const table = getBuffer(backend, "WebGPUPagedShadowPageTable");
	const indirect = getBuffer(backend, "WebGPUPagedShadowDrawIndirectArgs");
	assert.ok(table);
	assert.ok(indirect);
	assert.ok(indirect.usage & BufferUsage.Indirect);
	assert.equal(table.data[0], WEBGPU_PAGED_SHADOW_NON_RESIDENT);
	assert.equal(table.data[3], WEBGPU_PAGED_SHADOW_NON_RESIDENT);
	assert.equal(runtime.getDebugState().gpuAuthoritative, true);
	assert.equal(runtime.getDebugState().drawCandidateCount, 2);
	assert.equal(runtime.getDebugState().drawInstanceCapacity, 32);
}

async function testCasterGrowthDoesNotRecreatePhysicalAtlas() {
	const backend = createMockBackend();
	const shadowPass = { renderPagedDepthIndirect() {} };
	const runtime = new WebGPUPagedShadowRuntime(backend, shadowPass);
	const renderSet = createRenderSet({
		paged: {
			physicalPageCount: 64,
			maxPagesPerFrame: 4,
		},
	});

	runtime.prepareFrame(createRequest(renderSet, createPackets(2)));
	const firstAtlas = getLatestTexture(
		backend,
		"WebGPUPagedShadowPhysicalDepthAtlas"
	);
	assert.ok(firstAtlas);
	assert.equal(runtime.getDebugState().drawInstanceCapacity, 32);

	runtime.prepareFrame(createRequest(renderSet, createPackets(40)));

	const latestAtlas = getLatestTexture(
		backend,
		"WebGPUPagedShadowPhysicalDepthAtlas"
	);
	assert.equal(latestAtlas, firstAtlas);
	assert.equal(firstAtlas.destroyed, false);
	assert.equal(
		backend.textures.filter(
			(texture) => texture.label === "WebGPUPagedShadowPhysicalDepthAtlas"
		).length,
		1
	);
	assert.equal(runtime.getDebugState().casterCapacity, 40);
	assert.equal(runtime.getDebugState().drawInstanceCapacity, 160);
}

async function testPhysicalShapeChangesRecreateAtlasAndResetResidency() {
	const backend = createMockBackend();
	const shadowPass = { renderPagedDepthIndirect() {} };
	const runtime = new WebGPUPagedShadowRuntime(backend, shadowPass);
	const packets = createPackets(2);
	const renderSet = createRenderSet({
		paged: {
			pageSize: 64,
			physicalPageCount: 2,
		},
	});

	runtime.prepareFrame(createRequest(renderSet, packets));
	const firstAtlas = getLatestTexture(
		backend,
		"WebGPUPagedShadowPhysicalDepthAtlas"
	);
	const firstResidency = getLatestBuffer(
		backend,
		"WebGPUPagedShadowResidencyState"
	);
	assert.ok(firstAtlas);
	assert.ok(firstResidency);

	renderSet.layout.paged.physicalPageCount = 4;
	runtime.prepareFrame(createRequest(renderSet, packets));
	const secondAtlas = getLatestTexture(
		backend,
		"WebGPUPagedShadowPhysicalDepthAtlas"
	);
	const secondResidency = getLatestBuffer(
		backend,
		"WebGPUPagedShadowResidencyState"
	);

	assert.notEqual(secondAtlas, firstAtlas);
	assert.equal(firstAtlas.destroyed, true);
	assert.notEqual(secondResidency, firstResidency);
	assert.equal(secondResidency.data[0], WEBGPU_PAGED_SHADOW_NON_RESIDENT);
	assert.equal(runtime.getDebugState().physicalPageCount, 4);

	renderSet.layout.paged.pageSize = 128;
	runtime.prepareFrame(createRequest(renderSet, packets));
	const thirdAtlas = getLatestTexture(
		backend,
		"WebGPUPagedShadowPhysicalDepthAtlas"
	);
	assert.notEqual(thirdAtlas, secondAtlas);
	assert.equal(secondAtlas.destroyed, true);
	assert.equal(runtime.getDebugState().pageSize, 128);
}

async function testDrawInstanceCapacityGrowsByOnePointFive() {
	const backend = createMockBackend();
	const shadowPass = { renderPagedDepthIndirect() {} };
	const runtime = new WebGPUPagedShadowRuntime(backend, shadowPass);
	const renderSet = createRenderSet({
		paged: {
			physicalPageCount: 512,
			maxPagesPerFrame: 4,
		},
	});

	runtime.prepareFrame(createRequest(renderSet, createPackets(10)));
	assert.equal(runtime.getDebugState().drawInstanceCapacity, 40);

	runtime.prepareFrame(createRequest(renderSet, createPackets(11)));
	assert.equal(runtime.getDebugState().drawInstanceCapacity, 60);
}

async function testGpuPassesDispatchWithoutCpuPageTableAllocation() {
	const backend = createMockBackend();
	const encoder = createEncoder();
	const shadowPass = { renderPagedDepthIndirect() {} };
	const runtime = new WebGPUPagedShadowRuntime(backend, shadowPass);
	const renderSet = createRenderSet({
		paged: {
			physicalPageCount: 2,
			maxPagesPerFrame: 1,
		},
	});
	const request = createRequest(renderSet, [
		createPacket("left", -0.75, 0.75),
		createPacket("right", 0.75, 0.75),
	], encoder);

	runtime.prepareFrame(request);
	await runtime.recordPageMarkPass(request);
	await runtime.recordPageAllocationPass(request);

	const table = getBuffer(backend, "WebGPUPagedShadowPageTable");
	assert.equal(table.data[0], WEBGPU_PAGED_SHADOW_NON_RESIDENT);
	assert.ok(
		encoder.calls.some(
			(call) => call[0] === "beginComputePass" &&
				call[1] === "WebGPUPagedShadowRequestMark"
		)
	);
	assert.ok(
		encoder.calls.some(
			(call) => call[0] === "beginComputePass" &&
				call[1] === "WebGPUPagedShadowResidencyAllocate"
		)
	);
	assert.ok(
		encoder.calls.some(
			(call) => call[0] === "beginComputePass" &&
				call[1] === "WebGPUPagedShadowDirtyCompact"
		)
	);
}

async function testDepthPassBuildsGpuDrawsAndUsesIndirectRenderer() {
	const backend = createMockBackend();
	const encoder = createEncoder();
	const rendered = [];
	const shadowPass = {
		async renderPagedDepthIndirect(_context, resources, _encoder, packets) {
			rendered.push({
				candidates: resources.drawCandidateCount,
				indirect: resources.drawIndirectArgsBuffer.label,
				packetCount: packets.length,
			});
		},
	};
	const runtime = new WebGPUPagedShadowRuntime(backend, shadowPass);
	const renderSet = createRenderSet({
		paged: {
			physicalPageCount: 2,
			maxPagesPerFrame: 2,
		},
	});
	const request = createRequest(renderSet, [
		createPacket("left", -0.75, 0.75),
		createPacket("right", 0.75, 0.75),
	], encoder);

	runtime.prepareFrame(request);
	await runtime.recordDepthPass(request);

	assert.deepEqual(rendered, [
		{
			candidates: 2,
			indirect: "WebGPUPagedShadowDrawIndirectArgs",
			packetCount: 2,
		},
	]);
	assert.ok(
		encoder.calls.some(
			(call) => call[0] === "beginComputePass" &&
				call[1] === "WebGPUPagedShadowDrawBuild"
		)
	);
}

async function testFeedbackPassUsesScreenDepthTexture() {
	const backend = createMockBackend();
	const encoder = createEncoder();
	const shadowPass = { renderPagedDepthIndirect() {} };
	const runtime = new WebGPUPagedShadowRuntime(backend, shadowPass);
	const renderSet = createRenderSet();
	const request = createRequest(renderSet, [createPacket("left", -0.75, 0.75)], encoder);

	runtime.prepareFrame(request);
	await runtime.recordFeedbackPass(request);

	assert.ok(
		encoder.calls.some(
			(call) => call[0] === "beginComputePass" &&
				call[1] === "WebGPUPagedShadowFeedback"
		)
	);
	const feedbackBinding = backend.bindingGroups.find(
		(group) => group.label === "WebGPUPagedShadowFeedbackBindGroup"
	);
	assert.ok(feedbackBinding);
	assert.equal(feedbackBinding.entries[5].resource.label, "FeedbackDepth");
	assert.equal(
		feedbackBinding.entries[6].resource.label,
		"WebGPUPagedShadowFeedbackCamera"
	);
}

function testResidencyShaderRefreshesCachedPages() {
	const source = readFileSync(
		new URL(
			"../../../src/shaders/webgpu/shadow/pagedShadowResidencyAllocate.wgsl",
			import.meta.url
		),
		"utf8"
	);
	assert.match(source, /residencyState\[base \+ 3u\] = dirty;/);
	assert.match(source, /writePhysicalPageMetadata\(physicalPage, requestBase, tableIndex, dirty\);/);
}

async function testCascadeProjectionChangesSetForceDirty() {
	const backend = createMockBackend();
	const shadowPass = { renderPagedDepthIndirect() {} };
	const runtime = new WebGPUPagedShadowRuntime(backend, shadowPass);
	const packets = createPackets(2);
	const renderSet = createRenderSet({
		paged: {
			pageSize: 64,
			physicalPageCount: 4,
		},
	});

	// Helper to get the latest forceDirty flag from written allocation params
	const getLatestForceDirty = () => {
		const allocParamsWrites = backend.writes.filter(([label]) => label === "WebGPUPagedShadowAllocationParams");
		const latestWrite = allocParamsWrites[allocParamsWrites.length - 1];
		return latestWrite ? latestWrite[1][6] : undefined;
	};

	// Frame 1: Initial call, should have forceDirty = 1
	runtime.prepareFrame(createRequest(renderSet, packets));
	assert.equal(getLatestForceDirty(), 1);

	// Frame 2: Identical projections, should have forceDirty = 0
	runtime.prepareFrame(createRequest(renderSet, packets));
	assert.equal(getLatestForceDirty(), 0);

	// Frame 3: Modified projections, should have forceDirty = 1 again
	renderSet.slices[0].shadowMap.viewProjectionMatrix.elements[0][0] = 2;
	runtime.prepareFrame(createRequest(renderSet, packets));
	assert.equal(getLatestForceDirty(), 1);
}

async function run() {
	testCpuRequesterRemainsAvailableForDiagnostics();
	await testRuntimeCreatesGpuAuthoritativeBuffers();
	await testCasterGrowthDoesNotRecreatePhysicalAtlas();
	await testPhysicalShapeChangesRecreateAtlasAndResetResidency();
	await testDrawInstanceCapacityGrowsByOnePointFive();
	await testGpuPassesDispatchWithoutCpuPageTableAllocation();
	await testDepthPassBuildsGpuDrawsAndUsesIndirectRenderer();
	await testFeedbackPassUsesScreenDepthTexture();
	testResidencyShaderRefreshesCachedPages();
	await testCascadeProjectionChangesSetForceDirty();
	console.log("test_webgpu_paged_shadow_runtime: ok");
}

await run();
