import assert from "node:assert/strict";
import {
	arrayOf,
	mat4x4f32,
	scalar,
	structOf,
	StructuredBufferLayout,
	vec,
} from "../../../src/renderers/webgpu/StructuredBufferLayout.ts";
import {
	arrayStruct,
	arrayVec4,
	boolF32,
	createStructuredBufferPacker,
	custom,
	f32,
	i32,
	mat4,
	u32,
	vec2,
	vec3,
	vec4,
} from "../../../src/renderers/webgpu/StructuredBufferPacker.ts";

function readF32(layout, data, path) {
	return data[layout.byteOffsetOf(path) >> 2];
}

function readU32(layout, data, path) {
	return new Uint32Array(data.buffer)[layout.byteOffsetOf(path) >> 2];
}

function readI32(layout, data, path) {
	return new Int32Array(data.buffer)[layout.byteOffsetOf(path) >> 2];
}

function readVec(layout, data, path, length) {
	const offset = layout.byteOffsetOf(path) >> 2;
	return Array.from(data.slice(offset, offset + length));
}

function testHelpersPackScalarsVectorsMatrixArraysAndStructs() {
	const layout = new StructuredBufferLayout(
		structOf([
			{ name: "time", type: scalar("f32") },
			{ name: "count", type: scalar("u32") },
			{ name: "signed", type: scalar("i32") },
			{ name: "enabled", type: scalar("f32") },
			{ name: "uv", type: vec(2, "f32") },
			{ name: "normal", type: vec(3, "f32") },
			{ name: "color", type: vec(4, "f32") },
			{ name: "matrix", type: mat4x4f32() },
			{ name: "items", type: arrayOf(vec(4, "f32"), 2) },
			{
				name: "entries",
				type: arrayOf(
					structOf([
						{ name: "data", type: vec(4, "f32") },
						{ name: "weight", type: scalar("f32") },
					]),
					2
				),
			},
			{ name: "customValue", type: scalar("f32") },
			{ name: "optional", type: vec(4, "f32") },
		]),
		"uniform"
	);
	const packer = createStructuredBufferPacker({
		label: "TestUniforms",
		layout,
		output: "float32Array",
		fields: [
			f32("time", (input) => input.time),
			u32("count", (input) => input.count),
			i32("signed", (input) => input.signed),
			boolF32("enabled", (input) => input.enabled),
			vec2("uv", (input) => input.uv),
			vec3("normal", (input) => input.normal),
			vec4("color", (input) => input.color),
			mat4("matrix", (input) => input.matrix),
			arrayVec4("items", 2, (input, index) => input.items[index]),
			arrayStruct("entries", 2, (input, index) => input.entries[index], [
				vec4("data", (entry) => entry.data),
				f32("weight", (entry) => entry.weight),
			]),
			custom("customValue", (writer, input) => {
				writer.writeF32("customValue", input.customValue);
			}),
			vec4("optional", (input) => input.optional),
		],
	});
	const input = {
		time: 1.25,
		count: 7,
		signed: -3,
		enabled: true,
		uv: [2, 3],
		normal: [4, 5, 6],
		color: [7, 8, 9, 10],
		matrix: [
			[1, 2, 3, 4],
			[5, 6, 7, 8],
			[9, 10, 11, 12],
			[13, 14, 15, 16],
		],
		items: [
			[17, 18, 19, 20],
			[21, 22, 23, 24],
		],
		entries: [
			{ data: [25, 26, 27, 28], weight: 29 },
			{ data: [30, 31, 32, 33], weight: 34 },
		],
		customValue: 35,
		optional: [36, 37, 38, 39],
	};
	const data = packer.pack(input);

	assert.ok(data instanceof Float32Array);
	assert.equal(readF32(layout, data, "time"), 1.25);
	assert.equal(readU32(layout, data, "count"), 7);
	assert.equal(readI32(layout, data, "signed"), -3);
	assert.equal(readF32(layout, data, "enabled"), 1);
	assert.deepEqual(readVec(layout, data, "uv", 2), [2, 3]);
	assert.deepEqual(readVec(layout, data, "normal", 3), [4, 5, 6]);
	assert.deepEqual(readVec(layout, data, "color", 4), [7, 8, 9, 10]);
	assert.deepEqual(readVec(layout, data, "matrix", 16), [
		1, 5, 9, 13,
		2, 6, 10, 14,
		3, 7, 11, 15,
		4, 8, 12, 16,
	]);
	assert.deepEqual(readVec(layout, data, ["items", 1], 4), [21, 22, 23, 24]);
	assert.deepEqual(readVec(layout, data, ["entries", 1, "data"], 4), [
		30, 31, 32, 33,
	]);
	assert.equal(readF32(layout, data, ["entries", 1, "weight"]), 34);
	assert.equal(readF32(layout, data, "customValue"), 35);
}

function testSkippedValuesAndWriterReuseClearBytes() {
	const layout = new StructuredBufferLayout(
		structOf([{ name: "value", type: vec(4, "f32") }]),
		"uniform"
	);
	const packer = createStructuredBufferPacker({
		label: "ClearTest",
		layout,
		output: "float32Array",
		fields: [vec4("value", (input) => input.value)],
	});
	const writer = packer.createWriter();
	const first = packer.packInto(writer, { value: [1, 2, 3, 4] });
	assert.deepEqual(readVec(layout, first, "value", 4), [1, 2, 3, 4]);

	const second = packer.packInto(writer, { value: undefined });
	assert.deepEqual(readVec(layout, second, "value", 4), [0, 0, 0, 0]);
}

function testArrayBufferOutputModeIsDefault() {
	const layout = new StructuredBufferLayout(
		structOf([{ name: "value", type: scalar("f32") }]),
		"uniform"
	);
	const packer = createStructuredBufferPacker({
		label: "ArrayBufferTest",
		layout,
		fields: [f32("value", (input) => input.value)],
	});
	const buffer = packer.pack({ value: 42 });
	const data = new Float32Array(buffer);

	assert.ok(buffer instanceof ArrayBuffer);
	assert.equal(data[layout.byteOffsetOf("value") >> 2], 42);
}

function testCreationValidatesFieldPaths() {
	const layout = new StructuredBufferLayout(
		structOf([{ name: "value", type: scalar("f32") }]),
		"uniform"
	);

	assert.throws(
		() =>
			createStructuredBufferPacker({
				label: "InvalidPath",
				layout,
				fields: [f32("missing", (input) => input.value)],
			}),
		/InvalidPath packer field "missing" validation failed/
	);
}

testHelpersPackScalarsVectorsMatrixArraysAndStructs();
testSkippedValuesAndWriterReuseClearBytes();
testArrayBufferOutputModeIsDefault();
testCreationValidatesFieldPaths();
console.log("WebGPU structured buffer packer tests passed");
