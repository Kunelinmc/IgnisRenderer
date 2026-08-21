import assert from "node:assert/strict";

import { PBRMaterial } from "../../../src/materials/PBRMaterial.ts";
import { WebGPUMaterialSnapshotCache } from "../../../src/backends/webgpu/WebGPUMaterialSnapshotCache.ts";
import {
	WEBGPU_TEXTURE_SLOT,
	WEBGPU_TEXTURE_SLOT_COUNT,
} from "../../../src/backends/webgpu/constants.ts";

const counters = { textures: 0, samplers: 0 };
const textures = {
	async getTextureForSlotAsync(_texture, slot) {
		counters.textures++;
		return { id: `texture:${slot}` };
	},
	getSamplerForTexture() {
		counters.samplers++;
		return { id: "sampler" };
	},
};
const cache = new WebGPUMaterialSnapshotCache(textures);
const material = new PBRMaterial();

cache.beginFrame();
const first = await cache.resolve(material, false);
const second = await cache.resolve(material, false);
assert.equal(first, second);
assert.equal(counters.textures, WEBGPU_TEXTURE_SLOT_COUNT);
assert.equal(counters.samplers, WEBGPU_TEXTURE_SLOT_COUNT);
assert.equal(first.textures[WEBGPU_TEXTURE_SLOT.ANISOTROPY].id, "texture:16");
assert.equal("anisotropyTexture" in first, false);

material.roughness = 0.2;
cache.beginFrame();
const third = await cache.resolve(material, false);
assert.notEqual(third, first);
assert.equal(counters.textures, WEBGPU_TEXTURE_SLOT_COUNT * 2);
assert.equal(counters.samplers, WEBGPU_TEXTURE_SLOT_COUNT * 2);

console.log("WebGPU material snapshot cache tests passed");
