import assert from "node:assert/strict";

import { PBRMaterial } from "../../../src/materials/PBRMaterial.ts";
import { WebGPUMaterialSnapshotCache } from "../../../src/backends/webgpu/WebGPUMaterialSnapshotCache.ts";

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
assert.equal(counters.textures, 17);
assert.equal(counters.samplers, 16);

material.roughness = 0.2;
cache.beginFrame();
const third = await cache.resolve(material, false);
assert.notEqual(third, first);
assert.equal(counters.textures, 34);
assert.equal(counters.samplers, 32);

console.log("WebGPU material snapshot cache tests passed");
