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
const renderLayer = 0xffff & WEBGPU_DEFERRED_RENDER_LAYER_MASK;
assert.equal(renderLayer, 0x7ff);
const codec = readFileSync(
	new URL(
		"../../../src/shaders/webgpu/common/deferredGBufferCodec.wgsl",
		import.meta.url
	),
	"utf8"
);
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
