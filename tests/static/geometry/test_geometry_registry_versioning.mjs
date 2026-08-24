import assert from "node:assert/strict";
import { Material } from "../../../src/materials/Material.ts";
import { WebGLGeometryRegistry } from "../../../src/backends/webgl/WebGLGeometryRegistry.ts";
import { WebGPUGeometryRegistry } from "../../../src/backends/webgpu/WebGPUGeometryRegistry.ts";
import { createTestDrawPacket } from "../helpers/drawPacket.mjs";

function createGeometryBinding(primitive) {
	return createTestDrawPacket({ primitive }).submission.geometry;
}

function createPrimitive() {
	return {
		id: "primitive-versioned",
		geometryVersion: 0,
		topology: "triangle-list",
		material: new Material(),
		geometry: {
			positions: new Float32Array([
				0,
				0,
				0,
				1,
				0,
				0,
				0,
				1,
				0,
			]),
			normals: new Float32Array([
				0,
				0,
				1,
				0,
				0,
				1,
				0,
				0,
				1,
			]),
			uv0: new Float32Array([0, 0, 1, 0, 0, 1]),
			indices: new Uint32Array([0, 1, 2]),
		},
		boundingSphere: {
			center: { x: 0.5, y: 0.5, z: 0 },
			radius: 1,
		},
		boundingBox: {
			min: { x: 0, y: 0, z: 0 },
			max: { x: 1, y: 1, z: 0 },
		},
		visible: true,
		castShadows: true,
		receiveShadows: true,
	};
}

function testWebGLGeometryVersionInvalidation() {
	const calls = {
		createBuffer: 0,
		deleteBuffer: 0,
	};
	const gl = {
		TRIANGLES: 0x0004,
		LINES: 0x0001,
		POINTS: 0x0000,
		UNSIGNED_SHORT: 0x1403,
		UNSIGNED_INT: 0x1405,
		ARRAY_BUFFER: 0x8892,
		ELEMENT_ARRAY_BUFFER: 0x8893,
		STATIC_DRAW: 0x88e4,
		FLOAT: 0x1406,
		createVertexArray() {
			return {};
		},
		createBuffer() {
			calls.createBuffer++;
			return {};
		},
		deleteVertexArray() {},
		deleteBuffer() {
			calls.deleteBuffer++;
		},
		bindVertexArray() {},
		bindBuffer() {},
		bufferData() {},
		enableVertexAttribArray() {},
		vertexAttribPointer() {},
	};
	const registry = new WebGLGeometryRegistry(gl, () => {});
	const primitive = createPrimitive();
	let packet = createTestDrawPacket({
		id: "packet-versioned",
		primitive,
	});

	registry.getGeometry(packet);
	const firstCreateCount = calls.createBuffer;
	registry.getGeometry(packet);
	assert.equal(calls.createBuffer, firstCreateCount);

	primitive.geometryVersion = 1;
	packet = createTestDrawPacket({ id: "packet-versioned", primitive });
	registry.getGeometry(packet);
	assert.ok(calls.createBuffer > firstCreateCount);
	assert.ok(calls.deleteBuffer >= 2);
}

function testWebGPUGeometryVersionInvalidation() {
	const calls = {
		createBuffer: 0,
		destroyBuffer: 0,
	};
	const backend = {
		createBuffer() {
			calls.createBuffer++;
			return {
				destroy() {
					calls.destroyBuffer++;
				},
			};
		},
		writeBuffer() {},
	};
	const registry = new WebGPUGeometryRegistry(backend);
	const primitive = createPrimitive();

	registry.getGeometry(createGeometryBinding(primitive));
	const firstCreateCount = calls.createBuffer;
	registry.getGeometry(createGeometryBinding(primitive));
	assert.equal(calls.createBuffer, firstCreateCount);

	primitive.geometryVersion = 1;
	registry.getGeometry(createGeometryBinding(primitive));
	assert.ok(calls.createBuffer > firstCreateCount);
	assert.ok(calls.destroyBuffer >= 3);
}

function createWebGPUBackendHarness() {
	const buffers = [];
	const writes = [];
	return {
		buffers,
		writes,
		backend: {
			createBuffer(desc) {
				const buffer = {
					desc,
					destroyed: false,
					destroy() {
						this.destroyed = true;
					},
				};
				buffers.push(buffer);
				return buffer;
			},
			writeBuffer(buffer, data) {
				writes.push({ buffer, data: new Uint8Array(
					data.buffer,
					data.byteOffset,
					data.byteLength,
				).slice() });
			},
		},
	};
}

function testWebGPUSemanticPackingAndLazyWireframe() {
	const harness = createWebGPUBackendHarness();
	const registry = new WebGPUGeometryRegistry(harness.backend);
	const primitive = createPrimitive();
	const handle = registry.getGeometry(createGeometryBinding(primitive));

	assert.equal(handle.vertexByteLength, 3 * 28);
	assert.equal(handle.indexFormat, "uint16");
	assert.equal(handle.indexByteLength, 6);
	assert.equal(handle.surfaceBuffer.desc.size, 48);
	assert.deepEqual(handle.vertexBindings.map((binding) => binding.slot), [0, 1, 2, 3]);
	assert.equal(handle.sceneVertexLayouts[1].arrayStride, 16);
	assert.equal(handle.wireframeIndexBuffer, null);
	assert.equal(
		harness.buffers.some((buffer) => buffer.desc.label.startsWith("Wireframe")),
		false,
	);

	const wireframe = registry.getWireframeGeometry(createGeometryBinding(primitive));
	assert.equal(wireframe.wireframeIndexFormat, "uint16");
	assert.equal(wireframe.wireframeIndexCount, 6);
	assert.equal(wireframe.wireframeIndexByteLength, 12);
	const bufferCount = harness.buffers.length;
	registry.getWireframeGeometry(createGeometryBinding(primitive));
	assert.equal(harness.buffers.length, bufferCount);
}

function testWebGPUWireframeDeduplicatesSharedEdges() {
	const harness = createWebGPUBackendHarness();
	const registry = new WebGPUGeometryRegistry(harness.backend);
	const primitive = createPrimitive();
	primitive.geometry.positions = new Float32Array(12);
	primitive.geometry.normals = new Float32Array(12);
	primitive.geometry.uv0 = new Float32Array(8);
	primitive.geometry.indices = new Uint32Array([0, 1, 2, 2, 1, 3]);

	const handle = registry.getWireframeGeometry(createGeometryBinding(primitive));
	assert.equal(handle.wireframeIndexCount, 10);
}

function testWebGPUMorphPackingAllocatesOnlyPresentSemantics() {
	const harness = createWebGPUBackendHarness();
	const registry = new WebGPUGeometryRegistry(harness.backend);
	const primitive = createPrimitive();
	primitive.geometry.morphTargets = [{
		positions: new Float32Array(9).fill(0.25),
	}];

	const handle = registry.getGeometry(createGeometryBinding(primitive));
	assert.equal(handle.morphTargetCount, 1);
	assert.equal(handle.morphSemanticMask, 1);
	assert.equal(handle.morphPositionBuffer.desc.size, 36);
	assert.equal(handle.morphNormalBuffer, null);
	assert.equal(handle.morphByteLength, 36);
}

function testWebGPUIndexFormatBoundary() {
	const harness = createWebGPUBackendHarness();
	const registry = new WebGPUGeometryRegistry(harness.backend);
	const primitive = createPrimitive();
	primitive.geometry.indices = new Uint32Array([0, 65536, 1]);

	const handle = registry.getGeometry(createGeometryBinding(primitive));
	assert.equal(handle.indexFormat, "uint32");
	assert.equal(handle.indexByteLength, 12);
}

function run() {
	testWebGLGeometryVersionInvalidation();
	testWebGPUGeometryVersionInvalidation();
	testWebGPUSemanticPackingAndLazyWireframe();
	testWebGPUWireframeDeduplicatesSharedEdges();
	testWebGPUMorphPackingAllocatesOnlyPresentSemantics();
	testWebGPUIndexFormatBoundary();
	console.log("Geometry registry versioning tests passed");
}

run();
