import assert from "node:assert/strict";
import {
	arrayOf,
	scalar,
	structOf,
	StructuredBufferLayout,
	vec,
} from "../../../src/backends/webgpu/StructuredBufferLayout.ts";
import {
	packWebGPUVertexGeometry,
} from "../../../src/backends/webgpu/WebGPUGeometryPacking.ts";

function testVertexAddressSpaceUsesPackedVectorAlignment() {
	const vertexLayout = new StructuredBufferLayout(
		structOf([
			{ name: "position", type: vec(3, "f32") },
			{ name: "uv", type: vec(2, "f32") },
			{ name: "normal", type: vec(3, "f32") },
		]),
		"vertex"
	);
	const storageLayout = new StructuredBufferLayout(
		structOf([
			{ name: "position", type: vec(3, "f32") },
			{ name: "uv", type: vec(2, "f32") },
			{ name: "normal", type: vec(3, "f32") },
		]),
		"storage"
	);

	assert.equal(vertexLayout.byteOffsetOf("position"), 0);
	assert.equal(vertexLayout.byteOffsetOf("uv"), 12);
	assert.equal(vertexLayout.byteOffsetOf("normal"), 20);
	assert.equal(vertexLayout.byteSize, 32);

	assert.equal(storageLayout.byteOffsetOf("position"), 0);
	assert.equal(storageLayout.byteOffsetOf("uv"), 16);
	assert.equal(storageLayout.byteOffsetOf("normal"), 32);
	assert.equal(storageLayout.byteSize, 48);
}

function testCreateVertexBufferLayoutInfersFormatsAndValidatesStride() {
	const layout = new StructuredBufferLayout(
		structOf([
			{ name: "position", type: vec(3, "f32") },
			{ name: "uv", type: vec(2, "f32") },
			{ name: "normal", type: vec(3, "f32") },
		]),
		"vertex"
	);
	const vertexBufferLayout = layout.createVertexBufferLayout({
		stepMode: "vertex",
		attributes: [
			{ path: "position", shaderLocation: 0 },
			{ path: "uv", shaderLocation: 1 },
			{ path: "normal", shaderLocation: 2 },
		],
	});

	assert.equal(vertexBufferLayout.arrayStride, 32);
	assert.equal(vertexBufferLayout.stepMode, "vertex");
	assert.deepEqual(vertexBufferLayout.attributes, [
		{ shaderLocation: 0, offset: 0, format: "float32x3" },
		{ shaderLocation: 1, offset: 12, format: "float32x2" },
		{ shaderLocation: 2, offset: 20, format: "float32x3" },
	]);

	assert.throws(
		() =>
			layout.createVertexBufferLayout({
				arrayStride: 16,
				attributes: [{ path: "normal", shaderLocation: 0 }],
			}),
		/arrayStride 16 is smaller than required 32/
	);
}

function testCreateVertexBufferLayoutSupportsExplicitPackedFormat() {
	const packedColorLayout = new StructuredBufferLayout(
		structOf([{ name: "packedColor", type: scalar("u32") }]),
		"vertex"
	);
	const vertexBufferLayout = packedColorLayout.createVertexBufferLayout({
		attributes: [
			{
				path: "packedColor",
				shaderLocation: 4,
				format: "unorm8x4",
			},
		],
	});

	assert.equal(vertexBufferLayout.arrayStride, 4);
	assert.deepEqual(vertexBufferLayout.attributes, [
		{
			shaderLocation: 4,
			offset: 0,
			format: "unorm8x4",
		},
	]);
}

function testCreateVertexBufferLayoutSupportsZeroStrideAndPackedFormats() {
	const layout = new StructuredBufferLayout(
		structOf([
			{ name: "uv", type: vec(2, "f32") },
			{ name: "tangent", type: vec(4, "f32") },
			{ name: "weights", type: vec(4, "f32") },
		]),
		"vertex"
	);
	const vertexBufferLayout = layout.createVertexBufferLayout({
		arrayStride: 0,
		attributes: [
			{ path: "uv", shaderLocation: 1, format: "float16x2" },
			{ path: "tangent", shaderLocation: 3, format: "snorm16x4" },
			{ path: "weights", shaderLocation: 6, format: "unorm16x4" },
		],
	});

	assert.equal(vertexBufferLayout.arrayStride, 0);
	assert.deepEqual(vertexBufferLayout.attributes, [
		{ shaderLocation: 1, offset: 0, format: "float16x2" },
		{ shaderLocation: 3, offset: 8, format: "snorm16x4" },
		{ shaderLocation: 6, offset: 24, format: "unorm16x4" },
	]);
}

function testCreateVertexBufferLayoutRequiresVertexAddressSpace() {
	const layout = new StructuredBufferLayout(
		structOf([{ name: "value", type: scalar("f32") }]),
		"uniform"
	);

	assert.throws(
		() =>
			layout.createVertexBufferLayout({
				attributes: [{ path: "value", shaderLocation: 0 }],
			}),
		/requires addressSpace "vertex"/
	);
}

function testPathAccessSupportsNestedArrayPaths() {
	const layout = new StructuredBufferLayout(
		structOf([
			{
				name: "entries",
				type: arrayOf(
					structOf([
						{ name: "value", type: vec(2, "f32") },
						{ name: "weight", type: scalar("f32") },
					]),
					3
				),
			},
		]),
		"storage"
	);

	assert.equal(layout.byteOffsetOf(["entries", 0, "value"]), 0);
	assert.equal(layout.byteOffsetOf(["entries", 1, "value"]), 16);
	assert.equal(layout.byteOffsetOf(["entries", 2, "weight"]), 40);
	assert.equal(layout.byteSizeOf(["entries", 2, "value"]), 8);
	assert.equal(layout.byteSizeOf(["entries", 2, "weight"]), 4);
}

function testSceneGeometryUsesSemanticStreams() {
	const geometry = {
		positions: new Float32Array(9),
		normals: new Float32Array(9),
		uv0: new Float32Array([0, 0, 1, 0, 0, 1]),
		indices: new Uint32Array([0, 1, 2]),
	};
	const packed = packWebGPUVertexGeometry(geometry, 3);

	assert.equal(packed.vertexByteLength, 3 * 28);
	assert.equal(packed.position.layout.arrayStride, 12);
	assert.equal(packed.surface.layout.arrayStride, 16);
	assert.deepEqual(packed.surface.layout.attributes, [
		{ shaderLocation: 2, offset: 0, format: "float32x3" },
		{ shaderLocation: 1, offset: 12, format: "float16x2" },
	]);
	assert.equal(packed.sceneLayouts[3].arrayStride, 0);
	assert.equal(packed.shadowLayouts[0].arrayStride, 12);
	assert.equal(packed.shadowLayouts[1].attributes.length, 0);
}

function testGeometryPackingFallsBackForHighErrorUv() {
	const geometry = {
		positions: new Float32Array(9),
		uv0: new Float32Array([10.1, 0, 10.1, 0, 10.1, 0]),
		indices: new Uint32Array([0, 1, 2]),
	};
	const packed = packWebGPUVertexGeometry(geometry, 3);

	assert.equal(packed.surface.layout.arrayStride, 8);
	assert.equal(packed.surface.layout.attributes[0].format, "float32x2");
	assert.equal(packed.vertexByteLength, 3 * 20);
}

function testFullGeometryUsesNinetySixByteBalancedLayout() {
	const uv = new Float32Array([0, 0, 1, 0, 0, 1]);
	const geometry = {
		positions: new Float32Array(9),
		normals: new Float32Array(9),
		tangents: new Float32Array([
			1, 0, 0, 1,
			1, 0, 0, 1,
			1, 0, 0, 1,
		]),
		uv0: uv,
		uv1: uv,
		uv2: uv,
		uv3: uv,
		joints0: new Uint16Array(12),
		weights0: new Float32Array([
			1, 0, 0, 0,
			1, 0, 0, 0,
			1, 0, 0, 0,
		]),
		joints1: new Uint16Array(12),
		weights1: new Float32Array(12),
		indices: new Uint32Array([0, 1, 2]),
	};
	const packed = packWebGPUVertexGeometry(geometry, 3);

	assert.equal(packed.skinProfile, "skin8");
	assert.equal(packed.surface.layout.arrayStride, 36);
	assert.equal(packed.skin.layout.arrayStride, 48);
	assert.equal(packed.vertexByteLength, 3 * 96);
	assert.equal(packed.shadowLayouts[1].attributes.length, 0);
	assert.equal(packed.shadowLayouts[2].arrayStride, 48);
}

testVertexAddressSpaceUsesPackedVectorAlignment();
testCreateVertexBufferLayoutInfersFormatsAndValidatesStride();
testCreateVertexBufferLayoutSupportsExplicitPackedFormat();
testCreateVertexBufferLayoutSupportsZeroStrideAndPackedFormats();
testCreateVertexBufferLayoutRequiresVertexAddressSpace();
testPathAccessSupportsNestedArrayPaths();
testSceneGeometryUsesSemanticStreams();
testGeometryPackingFallsBackForHighErrorUv();
testFullGeometryUsesNinetySixByteBalancedLayout();
console.log("WebGPU structured buffer layout tests passed");
