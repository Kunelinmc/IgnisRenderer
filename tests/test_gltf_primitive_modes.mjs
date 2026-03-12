import assert from "node:assert/strict";
import { GLTFLoader } from "../src/loaders/GLTFLoader.ts";

const COMPONENT_TYPE_FLOAT = 5126;
const COMPONENT_TYPE_UNSIGNED_SHORT = 5123;
const TYPE_SCALAR = "SCALAR";
const TYPE_VEC3 = "VEC3";

function buildPrimitiveSource(indices) {
	const positions = new Float32Array([
		0, 0, 0,
		1, 0, 0,
		1, 1, 0,
		0, 1, 0,
	]);
	const indexArray = new Uint16Array(indices);
	const combined = new Uint8Array(positions.byteLength + indexArray.byteLength);
	combined.set(new Uint8Array(positions.buffer), 0);
	combined.set(new Uint8Array(indexArray.buffer), positions.byteLength);

	return {
		json: {
			accessors: [
				{
					bufferView: 0,
					byteOffset: 0,
					componentType: COMPONENT_TYPE_FLOAT,
					count: 4,
					type: TYPE_VEC3,
				},
				{
					bufferView: 1,
					byteOffset: 0,
					componentType: COMPONENT_TYPE_UNSIGNED_SHORT,
					count: indexArray.length,
					type: TYPE_SCALAR,
				},
			],
			bufferViews: [
				{
					buffer: 0,
					byteOffset: 0,
					byteLength: positions.byteLength,
				},
				{
					buffer: 0,
					byteOffset: positions.byteLength,
					byteLength: indexArray.byteLength,
				},
			],
		},
		buffers: [combined],
		primitive: {
			attributes: {
				POSITION: 0,
			},
			indices: 1,
		},
	};
}

function parseMode(mode, indices) {
	const loader = new GLTFLoader();
	const source = buildPrimitiveSource(indices);
	const parsed = loader.parsePrimitive(
		source.json,
		{
			...source.primitive,
			mode,
		},
		source.buffers,
		[]
	);
	assert.ok(parsed, `Expected primitive for mode ${mode}`);
	return parsed;
}

function testPoints() {
	const primitive = parseMode(0, [0, 1, 2, 3]);
	assert.equal(primitive.topology, "point-list");
	assert.deepEqual(Array.from(primitive.geometry.indices), [0, 1, 2, 3]);
}

function testLines() {
	const primitive = parseMode(1, [0, 1, 2, 3]);
	assert.equal(primitive.topology, "line-list");
	assert.deepEqual(Array.from(primitive.geometry.indices), [0, 1, 2, 3]);
}

function testLineLoop() {
	const primitive = parseMode(2, [0, 1, 2, 3]);
	assert.equal(primitive.topology, "line-list");
	assert.deepEqual(Array.from(primitive.geometry.indices), [
		0, 1, 1, 2, 2, 3, 3, 0,
	]);
}

function testLineStrip() {
	const primitive = parseMode(3, [0, 1, 2, 3]);
	assert.equal(primitive.topology, "line-list");
	assert.deepEqual(Array.from(primitive.geometry.indices), [0, 1, 1, 2, 2, 3]);
}

function testTriangles() {
	const primitive = parseMode(4, [0, 1, 2, 2, 3, 0]);
	assert.equal(primitive.topology, "triangle-list");
	assert.deepEqual(Array.from(primitive.geometry.indices), [0, 1, 2, 2, 3, 0]);
}

function testTriangleStrip() {
	const primitive = parseMode(5, [0, 1, 2, 3]);
	assert.equal(primitive.topology, "triangle-list");
	assert.deepEqual(Array.from(primitive.geometry.indices), [0, 1, 2, 2, 1, 3]);
}

function testTriangleFan() {
	const primitive = parseMode(6, [0, 1, 2, 3]);
	assert.equal(primitive.topology, "triangle-list");
	assert.deepEqual(Array.from(primitive.geometry.indices), [0, 1, 2, 0, 2, 3]);
}

function testDefaultModeIsTriangles() {
	const loader = new GLTFLoader();
	const source = buildPrimitiveSource([0, 1, 2, 2, 3, 0]);
	const parsed = loader.parsePrimitive(source.json, source.primitive, source.buffers, []);
	assert.ok(parsed);
	assert.equal(parsed.topology, "triangle-list");
	assert.deepEqual(Array.from(parsed.geometry.indices), [0, 1, 2, 2, 3, 0]);
}

function testUnsupportedModeThrows() {
	const loader = new GLTFLoader();
	const source = buildPrimitiveSource([0, 1, 2]);
	assert.throws(() =>
		loader.parsePrimitive(
			source.json,
			{
				...source.primitive,
				mode: 9,
			},
			source.buffers,
			[]
		)
	);
}

function run() {
	try {
		console.log("Starting glTF primitive mode tests...");
		testPoints();
		testLines();
		testLineLoop();
		testLineStrip();
		testTriangles();
		testTriangleStrip();
		testTriangleFan();
		testDefaultModeIsTriangles();
		testUnsupportedModeThrows();
		console.log("glTF primitive mode tests passed");
	} catch (error) {
		console.error("glTF primitive mode tests failed");
		console.error(error);
		process.exit(1);
	}
}

run();
