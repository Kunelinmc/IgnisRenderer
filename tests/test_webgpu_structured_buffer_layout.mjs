import assert from "node:assert/strict";
import {
	arrayOf,
	scalar,
	structOf,
	StructuredBufferLayout,
	vec,
} from "../src/renderers/webgpu/StructuredBufferLayout.ts";

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

testVertexAddressSpaceUsesPackedVectorAlignment();
testCreateVertexBufferLayoutInfersFormatsAndValidatesStride();
testCreateVertexBufferLayoutSupportsExplicitPackedFormat();
testCreateVertexBufferLayoutRequiresVertexAddressSpace();
testPathAccessSupportsNestedArrayPaths();
console.log("WebGPU structured buffer layout tests passed");
