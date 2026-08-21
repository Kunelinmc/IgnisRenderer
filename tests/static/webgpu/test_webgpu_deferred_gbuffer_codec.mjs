import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import { ShaderSource } from "../../../src/shaders/ShaderSource.ts";
import {
	WEBGPU_DEFERRED_MATERIAL_ANISOTROPY_BIT,
	WEBGPU_DEFERRED_MATERIAL_CLEARCOAT_BIT,
	WEBGPU_DEFERRED_MATERIAL_IRIDESCENCE_BIT,
	WEBGPU_DEFERRED_MATERIAL_MODEL_MASK,
	WEBGPU_DEFERRED_MATERIAL_RECEIVE_SHADOWS_BIT,
	WEBGPU_DEFERRED_MATERIAL_SHEEN_BIT,
	WEBGPU_DEFERRED_MATERIAL_SPECULAR_BIT,
	WEBGPU_DEFERRED_RENDER_LAYER_MASK,
} from "../../../src/backends/webgpu/constants.ts";

function roundTripFloat16(value) {
	const array = new Float16Array(1);
	array[0] = value;
	return array[0];
}

function encodeOctahedralNormal(normal) {
	const length = Math.abs(normal[0]) + Math.abs(normal[1]) + Math.abs(normal[2]);
	let x = normal[0] / length;
	let y = normal[1] / length;
	const z = normal[2] / length;
	if (z < 0) {
		const oldX = x;
		x = (1 - Math.abs(y)) * Math.sign(oldX || 1);
		y = (1 - Math.abs(oldX)) * Math.sign(y || 1);
	}
	return [x * 0.5 + 0.5, y * 0.5 + 0.5];
}

function decodeOctahedralNormal(encoded) {
	let x = encoded[0] * 2 - 1;
	let y = encoded[1] * 2 - 1;
	let z = 1 - Math.abs(x) - Math.abs(y);
	if (z < 0) {
		const oldX = x;
		x = (1 - Math.abs(y)) * Math.sign(oldX || 1);
		y = (1 - Math.abs(oldX)) * Math.sign(y || 1);
	}
	const length = Math.hypot(x, y, z);
	return [x / length, y / length, z / length];
}

const featureBits = [
	WEBGPU_DEFERRED_MATERIAL_CLEARCOAT_BIT,
	WEBGPU_DEFERRED_MATERIAL_SHEEN_BIT,
	WEBGPU_DEFERRED_MATERIAL_IRIDESCENCE_BIT,
	WEBGPU_DEFERRED_MATERIAL_ANISOTROPY_BIT,
	WEBGPU_DEFERRED_MATERIAL_SPECULAR_BIT,
	WEBGPU_DEFERRED_MATERIAL_RECEIVE_SHADOWS_BIT,
].reduce((word, bit) => word | bit, 0);
for (const shadingModel of [0, 1, 2, 3]) {
	const word = (shadingModel & WEBGPU_DEFERRED_MATERIAL_MODEL_MASK) | featureBits;
	assert.equal(roundTripFloat16(word), word);
	assert.equal(word & WEBGPU_DEFERRED_MATERIAL_MODEL_MASK, shadingModel);
}
for (const shininess of [0, 1, 32, 128]) {
	assert.equal(roundTripFloat16(shininess), shininess);
}

const renderLayer = 0xffff & WEBGPU_DEFERRED_RENDER_LAYER_MASK;
assert.equal(renderLayer, 0x7ff);
for (const strength of [0, 0.25, 0.5, 1]) {
	const packed = Math.round(strength * 65535);
	assert.ok(Math.abs(packed / 65535 - strength) <= 1 / 65535);
}
for (const tangent of [[1, 0, 0], [0, 1, 0], [0.5, 0.5, Math.SQRT1_2]]) {
	const length = Math.hypot(...tangent);
	const normalized = tangent.map((value) => value / length);
	const encoded = encodeOctahedralNormal(normalized);
	const quantized = encoded.map((value) => Math.round(value * 65535) / 65535);
	const decoded = decodeOctahedralNormal(quantized);
	const alignment = decoded.reduce(
		(total, value, index) => total + value * normalized[index],
		0
	);
	assert.ok(alignment > 0.99999);
}

const codec = readFileSync(
	new URL(
		"../../../src/shaders/webgpu/common/deferredGBufferCodec.wgsl",
		import.meta.url
	),
	"utf8"
);
assert.match(codec, /fn encodeDeferredMaterialWord\(/);
assert.match(codec, /fn decodeDeferredMaterialWord\(/);
assert.match(codec, /fn packDeferredExt3\(/);
assert.match(codec, /fn decodeDeferredExt3RenderLayers\(/);
assert.match(codec, /DEFERRED_RENDER_LAYER_MASK: u32 = 0x7ffu/);

const scene = (await ShaderSource.load("webgpu.scene")).source.code;
const deferred = (await ShaderSource.load("webgpu.deferredLighting")).source.code;
const decal = (await ShaderSource.load("webgpu.utility.decal")).source.code;
assert.match(
	scene,
	/resolvedMaterialWord[\s\S]*nodeRenderLayers\.y[\s\S]*DEFERRED_MATERIAL_RECEIVE_SHADOWS_BIT[\s\S]*f32\(resolvedMaterialWord\)/
);
for (const source of [scene, deferred, decal]) {
	assert.equal((source.match(/fn encodeDeferredMaterialWord\(/g) ?? []).length, 1);
	assert.equal((source.match(/fn packDeferredExt3\(/g) ?? []).length, 1);
}

console.log("WebGPU deferred G-buffer codec tests passed");
