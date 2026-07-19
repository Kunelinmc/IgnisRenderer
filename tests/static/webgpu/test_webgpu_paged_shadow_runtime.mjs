import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import { LightType } from "../../../src/lights/index.ts";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import { BufferUsage, TextureUsage } from "../../../src/backends/types.ts";
import {
	WEBGPU_PAGED_SHADOW_NON_RESIDENT,
	WebGPUPagedShadowRuntime,
	collectWebGPUPagedShadowPageRequests,
} from "../../../src/backends/webgpu/WebGPUPagedShadowRuntime.ts";
import { WebGPUCommandEncoder } from "../../../src/backends/webgpu/WebGPUCommandEncoder.ts";
import { WebGPUShadowPass } from "../../../src/backends/webgpu/WebGPUShadowPass.ts";

function createMockBackend() {
	const buffers = [];
	const textures = [];
	const writes = [];
	const textureWrites = [];
	const bindingGroups = [];
	const pipelines = [];
	const nativeBuffers = [];
	return {
		buffers,
		textures,
		writes,
		textureWrites,
		bindingGroups,
		pipelines,
		nativeBuffers,
		device: {
			createBuffer(desc) {
				const mappedData = new ArrayBuffer(desc.size);
				const buffer = {
					size: desc.size,
					usage: desc.usage,
					label: desc.label,
					mappedData,
					destroyed: false,
					mapAsync() {
						return Promise.resolve();
					},
					getMappedRange() {
						return mappedData;
					},
					unmap() {},
					destroy() {
						this.destroyed = true;
					},
				};
				nativeBuffers.push(buffer);
				return buffer;
			},
		},
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
			buffer._gpuResource = buffer;
			buffers.push(buffer);
			return buffer;
		},
		createTexture(desc) {
			const texture = {
				width: desc.width,
				height: desc.height,
				usage: desc.usage,
				label: desc.label,
				destroyed: false,
				destroy() {
					this.destroyed = true;
				},
			};
			textures.push(texture);
			return texture;
		},
		writeBuffer(buffer, data, offset = 0) {
			const source =
				ArrayBuffer.isView(data) ?
					new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
				:	new Uint8Array(data);
			const copy = new Uint8Array(source);
			buffer.data = new Uint32Array(copy.buffer);
			buffer.byteLength = copy.byteLength;
			buffer.writeOffset = offset;
			writes.push([buffer.label, buffer.data, offset, copy.byteLength]);
		},
		writeTexture(texture, data, desc, size) {
			const source =
				ArrayBuffer.isView(data) ?
					new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
				:	new Uint8Array(data);
			const copy = new Uint8Array(source);
			textureWrites.push({
				label: texture.label,
				data: new Uint32Array(copy.buffer),
				desc,
				size,
			});
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

function createWebGPUEncoder(copies = []) {
	const nativeEncoder = {
		copyBufferToBuffer(source, sourceOffset, destination, destinationOffset, size) {
			copies.push({ source, sourceOffset, destination, destinationOffset, size });
		},
	};
	const host = {
		createPassTimestampWrites() {
			return undefined;
		},
		getCurrentColorView() {
			return null;
		},
		getCurrentDepthView() {
			return null;
		},
		getCanvasColorTexture() {
			return { width: 1, height: 1 };
		},
	};
	return new WebGPUCommandEncoder(nativeEncoder, host, {});
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
			viewCamera: {
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

function testSamplingFallbackDoesNotCreateRenderBuffers() {
	const backend = createMockBackend();
	const shadowPass = { renderPagedDepthIndirect() {} };
	const runtime = new WebGPUPagedShadowRuntime(backend, shadowPass);

	const resources = runtime.getSamplingResources();
	const pageTableTexture = getLatestTexture(
		backend,
		"WebGPUPagedShadowFallbackPageTableTexture"
	);
	const depthAtlas = getLatestTexture(
		backend,
		"WebGPUPagedShadowFallbackDepthAtlas"
	);

	assert.equal(resources.pageTableTexture, pageTableTexture);
	assert.equal(resources.physicalDepthAtlas, depthAtlas);
	assert.ok(pageTableTexture.usage & TextureUsage.TextureBinding);
	assert.ok(pageTableTexture.usage & TextureUsage.CopyDst);
	assert.equal(
		backend.buffers.some((buffer) =>
			buffer.label.startsWith("WebGPUPagedShadowFallback")
		),
		false
	);
	assert.equal(runtime.getIndirectRenderResources(), null);
}

function testSamplingFallbackWritesNonResidentPageTable() {
	const backend = createMockBackend();
	const shadowPass = { renderPagedDepthIndirect() {} };
	const runtime = new WebGPUPagedShadowRuntime(backend, shadowPass);

	runtime.getSamplingResources();

	assert.deepEqual(backend.textureWrites, [
		{
			label: "WebGPUPagedShadowFallbackPageTableTexture",
			data: new Uint32Array([WEBGPU_PAGED_SHADOW_NON_RESIDENT]),
			desc: {},
			size: { width: 1, height: 1 },
		},
	]);
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
	const clearIndirect = getBuffer(
		backend,
		"WebGPUPagedShadowClearDrawIndirectArgs"
	);
	assert.ok(table);
	assert.ok(indirect);
	assert.ok(clearIndirect);
	assert.ok(indirect.usage & BufferUsage.Indirect);
	assert.ok(clearIndirect.usage & BufferUsage.Indirect);
	assert.deepEqual(Array.from(clearIndirect.data), [6, 0, 0, 0]);
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

async function testRequestMarkDispatchUsesActualCasterWorkCount() {
	const backend = createMockBackend();
	const encoder = createEncoder();
	const shadowPass = { renderPagedDepthIndirect() {} };
	const runtime = new WebGPUPagedShadowRuntime(backend, shadowPass);
	const renderSet = createRenderSet({
		paged: {
			pageGridSize: 1,
			physicalPageCount: 4,
			maxPagesPerFrame: 1,
		},
	});

	runtime.prepareFrame(createRequest(renderSet, createPackets(80)));
	runtime._previousCasterBounds.clear();
	runtime.prepareFrame(createRequest(renderSet, [createPacket("single", 0, 0)], encoder));
	await runtime.recordPageMarkPass(createRequest(renderSet, [createPacket("single", 0, 0)], encoder));

	const markBegin = encoder.calls.findIndex(
		(call) => call[0] === "beginComputePass" &&
			call[1] === "WebGPUPagedShadowRequestMark"
	);
	const markDispatch = encoder.calls.slice(markBegin).find(
		(call) => call[0] === "dispatchWorkgroups"
	);
	assert.deepEqual(markDispatch, ["dispatchWorkgroups", 1, 1, 1]);
}

async function testFrameInputsUploadOnlyActiveSpans() {
	const backend = createMockBackend();
	const shadowPass = { renderPagedDepthIndirect() {} };
	const runtime = new WebGPUPagedShadowRuntime(backend, shadowPass);
	const renderSet = createRenderSet({
		paged: {
			physicalPageCount: 64,
			maxPagesPerFrame: 4,
		},
	});

	runtime.prepareFrame(createRequest(renderSet, createPackets(80)));
	runtime.prepareFrame(createRequest(renderSet, createPackets(2)));

	const matrixWrites = backend.writes.filter(
		([label]) => label === "WebGPUPagedShadowDrawWorldMatrices"
	);
	const indirectWrites = backend.writes.filter(
		([label]) => label === "WebGPUPagedShadowDrawIndirectArgs"
	);
	const latestMatrixWrite = matrixWrites[matrixWrites.length - 1];
	const latestIndirectWrite = indirectWrites[indirectWrites.length - 1];
	assert.equal(latestMatrixWrite[3], 2 * 16 * 4);
	assert.equal(latestIndirectWrite[3], 2 * 5 * 4);
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

	const compactParamsWrites = backend.writes.filter(([label]) => label === "WebGPUPagedShadowCompactParams");
	const compactParams = compactParamsWrites[compactParamsWrites.length - 1]?.[1];
	assert.equal(compactParams?.[2], 1);
	const compactBinding = backend.bindingGroups.find(
		(group) => group.label === "WebGPUPagedShadowRequestCompactBindGroup"
	);
	assert.ok(compactBinding);
	assert.equal(
		compactBinding.entries[4].resource.label,
		"WebGPUPagedShadowPageAddresses"
	);
	const pageAddress = getBuffer(backend, "WebGPUPagedShadowPageAddresses");
	assert.equal(pageAddress.data[0], 0);
	assert.equal(pageAddress.data[1], 0);
	assert.equal(pageAddress.data[2], 0);
	assert.equal(pageAddress.data[3], 4);
	assert.equal(pageAddress.data[4], 5);
	assert.equal(pageAddress.data[5], 1);
	assert.equal(pageAddress.data[8], 0);
	assert.equal(pageAddress.data[9], 1);
	assert.equal(pageAddress.data[10], 0);

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
				clearIndirect: resources.clearDrawIndirectArgsBuffer.label,
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
			clearIndirect: "WebGPUPagedShadowClearDrawIndirectArgs",
			packetCount: 2,
		},
	]);
	const dirtyGridIndex = encoder.calls.findIndex(
		(call) => call[0] === "beginComputePass" &&
			call[1] === "WebGPUPagedShadowDirtyGridBuild"
	);
	const drawBuildIndex = encoder.calls.findIndex(
		(call) => call[0] === "beginComputePass" &&
			call[1] === "WebGPUPagedShadowDrawBuild"
	);
	assert.ok(dirtyGridIndex >= 0);
	assert.ok(drawBuildIndex > dirtyGridIndex);
	const dirtyGridBinding = backend.bindingGroups.find(
		(group) => group.label === "WebGPUPagedShadowDirtyGridBuildBindGroup"
	);
	const drawBuildBinding = backend.bindingGroups.find(
		(group) => group.label === "WebGPUPagedShadowDrawBuildBindGroup"
	);
	assert.equal(
		dirtyGridBinding.entries[6].resource.label,
		"WebGPUPagedShadowDirtyPageUvRanges"
	);
	assert.equal(
		dirtyGridBinding.entries[7].resource.label,
		"WebGPUPagedShadowClearDrawIndirectArgs"
	);
	assert.equal(
		drawBuildBinding.entries[13].resource.label,
		"WebGPUPagedShadowDirtyPageUvRanges"
	);
}

async function testEmptyCasterSetStillClearsDirtyPagedDepthPages() {
	const calls = [];
	const nativeBuffer = { destroy() {} };
	const nativePass = {
		setViewport() {},
		setScissorRect() {},
		setPipeline() {},
		setBindGroup() {},
		drawIndirect() {
			calls.push("drawIndirect");
		},
		end() {
			calls.push("end");
		},
	};
	const nativeEncoder = {
		beginRenderPass() {
			calls.push("beginRenderPass");
			return nativePass;
		},
		finish() {
			return {};
		},
	};
	const backend = {
		device: {
			createCommandEncoder() {
				return nativeEncoder;
			},
			createBindGroup(desc) {
				calls.push(desc.label);
				return {};
			},
		},
		queue: {
			writeBuffer() {},
			submit() {
				calls.push("submit");
			},
		},
	};
	const shadowPass = new WebGPUShadowPass(backend, {}, {});
	const renderPipeline = { _gpuResource: {} };
	Object.assign(shadowPass, {
		_pipeline: renderPipeline,
		_transmittancePipeline: renderPipeline,
		_bindGroupLayout: {},
		_animationBindGroupLayout: {},
		_fallbackStorageBuffer: nativeBuffer,
		_pagedClearPipeline: renderPipeline,
		_pagedClearBindGroupLayout: {},
		_pagedClearParamsBuffer: nativeBuffer,
	});
	const physicalDepthAtlas = {
		_gpuResource: {
			createView() {
				return {};
			},
		},
	};
	const resources = {
		physicalDepthAtlas,
		dirtyPhysicalPages: { _gpuResource: nativeBuffer },
		drawMvpBuffer: { _gpuResource: nativeBuffer },
		drawInstanceMetaBuffer: { _gpuResource: nativeBuffer },
		drawTransmittanceBuffer: { _gpuResource: nativeBuffer },
		drawIndirectArgsBuffer: { _gpuResource: nativeBuffer },
		clearDrawIndirectArgsBuffer: { _gpuResource: nativeBuffer },
		pageSize: 64,
		physicalGridSize: 2,
		physicalAtlasSize: 128,
		drawCandidateCount: 0,
		drawInstanceCapacity: 1,
		physicalPageCount: 4,
	};
	const context = {
		features: { enableShadows: true },
		scene: { shadowCasterPackets: [] },
	};

	await shadowPass.renderPagedDepthIndirect(context, resources);

	assert.ok(calls.includes("beginRenderPass"));
	assert.ok(calls.includes("drawIndirect"));
	assert.equal(calls.includes("WebGPUPagedShadowDepthIndirectBindGroup"), false);
	assert.ok(calls.includes("submit"));
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

async function testDrawCounterReadbackUsesFixedRing() {
	const previousUsage = globalThis.GPUBufferUsage;
	globalThis.GPUBufferUsage = { COPY_DST: 1, MAP_READ: 2 };
	try {
		const backend = createMockBackend();
		const copies = [];
		const encoder = createWebGPUEncoder(copies);
		const shadowPass = { renderPagedDepthIndirect() {} };
		const runtime = new WebGPUPagedShadowRuntime(backend, shadowPass);
		const renderSet = createRenderSet();
		const request = createRequest(
			renderSet,
			[createPacket("left", -0.75, 0.75)],
			encoder
		);

		runtime.prepareFrame(request);
		for (let i = 0; i < 6; i++) {
			runtime._queueDrawCounterReadback(request);
		}

		assert.equal(backend.nativeBuffers.length, 4);
		assert.equal(copies.length, 4);
		runtime.destroy();
		assert.ok(backend.nativeBuffers.every((buffer) => buffer.destroyed));
	} finally {
		globalThis.GPUBufferUsage = previousUsage;
	}
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
	const getLatestResidencyScanLimit = () => {
		const allocParamsWrites = backend.writes.filter(([label]) => label === "WebGPUPagedShadowAllocationParams");
		const latestWrite = allocParamsWrites[allocParamsWrites.length - 1];
		return latestWrite ? latestWrite[1][7] : undefined;
	};

	// Frame 1: Initial call, should have forceDirty = 1
	runtime.prepareFrame(createRequest(renderSet, packets));
	assert.equal(getLatestForceDirty(), 1);
	assert.equal(getLatestResidencyScanLimit(), 4);

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
	testSamplingFallbackDoesNotCreateRenderBuffers();
	testSamplingFallbackWritesNonResidentPageTable();
	await testRuntimeCreatesGpuAuthoritativeBuffers();
	await testCasterGrowthDoesNotRecreatePhysicalAtlas();
	await testPhysicalShapeChangesRecreateAtlasAndResetResidency();
	await testDrawInstanceCapacityGrowsByOnePointFive();
	await testRequestMarkDispatchUsesActualCasterWorkCount();
	await testFrameInputsUploadOnlyActiveSpans();
	await testGpuPassesDispatchWithoutCpuPageTableAllocation();
	await testDepthPassBuildsGpuDrawsAndUsesIndirectRenderer();
	await testEmptyCasterSetStillClearsDirtyPagedDepthPages();
	await testFeedbackPassUsesScreenDepthTexture();
	await testDrawCounterReadbackUsesFixedRing();
	testResidencyShaderRefreshesCachedPages();
	await testCascadeProjectionChangesSetForceDirty();
	console.log("test_webgpu_paged_shadow_runtime: ok");
}

await run();
