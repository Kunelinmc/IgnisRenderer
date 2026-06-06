import assert from "node:assert/strict";
import { Material } from "../../../src/materials/Material.ts";
import { WebGLGeometryRegistry } from "../../../src/renderers/webgl/WebGLGeometryRegistry.ts";
import { WebGPUGeometryRegistry } from "../../../src/renderers/webgpu/WebGPUGeometryRegistry.ts";

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
	const packet = {
		id: "packet-versioned",
		primitive,
	};

	registry.getGeometry(packet);
	const firstCreateCount = calls.createBuffer;
	registry.getGeometry(packet);
	assert.equal(calls.createBuffer, firstCreateCount);

	primitive.geometryVersion = 1;
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

	registry.getGeometry(primitive);
	const firstCreateCount = calls.createBuffer;
	registry.getGeometry(primitive);
	assert.equal(calls.createBuffer, firstCreateCount);

	primitive.geometryVersion = 1;
	registry.getGeometry(primitive);
	assert.ok(calls.createBuffer > firstCreateCount);
	assert.ok(calls.destroyBuffer >= 3);
}

function run() {
	testWebGLGeometryVersionInvalidation();
	testWebGPUGeometryVersionInvalidation();
	console.log("Geometry registry versioning tests passed");
}

run();
