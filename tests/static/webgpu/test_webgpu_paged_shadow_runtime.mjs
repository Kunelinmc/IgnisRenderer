import assert from "node:assert/strict";

import { LightType } from "../../../src/lights/index.ts";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import {
	WEBGPU_PAGED_SHADOW_NON_RESIDENT,
	WebGPUPagedShadowRuntime,
	collectWebGPUPagedShadowPageRequests,
} from "../../../src/renderers/webgpu/WebGPUPagedShadowRuntime.ts";

function createMockBackend() {
	const buffers = [];
	const textures = [];
	return {
		buffers,
		textures,
		createBuffer(desc) {
			const buffer = {
				size: desc.size,
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
			buffer.data = new Uint32Array(data);
		},
	};
}

function createRenderSet(overrides = {}) {
	const paged = {
		virtualResolution: 4096,
		pageGridSize: 4,
		pageSize: 64,
		physicalPageCount: 1,
		maxPagesPerFrame: 1,
		cacheFrames: 0,
		feedbackMode: "conservative",
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

function createRequest(renderSet, packets) {
	const light = {
		id: "sun",
		type: LightType.Directional,
	};
	return {
		context: {
			features: { enableShadows: true },
			scene: {
				shadowCasterPackets: packets,
				shadowTransmitterPackets: [],
			},
		},
		encoder: null,
		renderSets: new Map([[light, renderSet]]),
		shadowCasterPackets: packets,
		shadowTransmitterPackets: [],
	};
}

function createPacket(id, x, y) {
	return {
		id,
		worldBounds: {
			center: { x, y, z: 0 },
			radius: 0.01,
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

function getPageTableBuffer(backend) {
	return backend.buffers.find(
		(buffer) => buffer.label === "WebGPUPagedShadowPageTable"
	);
}

function countResidentEntries(table) {
	return Array.from(table).filter(
		(entry) => entry !== WEBGPU_PAGED_SHADOW_NON_RESIDENT
	).length;
}

function testCpuRequesterProducesStablePageKeys() {
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

function testAllocationHonorsMaxPagesAndPacksNonResidentEntries() {
	const backend = createMockBackend();
	const shadowPass = { renderPagedDepthPages() {} };
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
	]);

	runtime.prepareFrame(request);
	runtime.recordPageAllocationPass(request);

	const table = getPageTableBuffer(backend).data;
	assert.equal(table.length, 16);
	assert.equal(countResidentEntries(table), 1);
	assert.equal(table[0], 0);
	assert.equal(table[3], WEBGPU_PAGED_SHADOW_NON_RESIDENT);
	assert.equal(runtime.getDebugState().residentCount, 1);
}

function testLruDoesNotEvictCurrentFrameRequests() {
	const backend = createMockBackend();
	const shadowPass = { renderPagedDepthPages() {} };
	const runtime = new WebGPUPagedShadowRuntime(backend, shadowPass);
	const renderSet = createRenderSet({
		paged: {
			physicalPageCount: 1,
			maxPagesPerFrame: 1,
			cacheFrames: 0,
		},
	});
	const firstRequest = createRequest(renderSet, [
		createPacket("left", -0.75, 0.75),
	]);
	runtime.prepareFrame(firstRequest);
	runtime.recordPageAllocationPass(firstRequest);

	const secondRequest = createRequest(renderSet, [
		createPacket("left", -0.75, 0.75),
		createPacket("right", 0.75, 0.75),
	]);
	runtime.prepareFrame(secondRequest);
	runtime.recordPageAllocationPass(secondRequest);

	const table = getPageTableBuffer(backend).data;
	assert.equal(countResidentEntries(table), 1);
	assert.equal(table[0], 0);
	assert.equal(table[3], WEBGPU_PAGED_SHADOW_NON_RESIDENT);
	assert.equal(runtime.getDebugState().residentCount, 1);
}

function run() {
	testCpuRequesterProducesStablePageKeys();
	testAllocationHonorsMaxPagesAndPacksNonResidentEntries();
	testLruDoesNotEvictCurrentFrameRequests();
	console.log("test_webgpu_paged_shadow_runtime: ok");
}

run();
