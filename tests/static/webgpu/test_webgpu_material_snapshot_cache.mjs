import assert from "node:assert/strict";

import { PBRMaterial } from "../../../src/materials/PBRMaterial.ts";
import { ShaderMaterial } from "../../../src/materials/ShaderMaterial.ts";
import { WebGPUDrawResourceCache } from "../../../src/backends/webgpu/WebGPUDrawResourceCache.ts";
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
const cache = new WebGPUDrawResourceCache({}, {}, {}, textures);
const material = new PBRMaterial();

cache.beginFrame();
const first = await cache.resolveMaterial(material, false);
const second = await cache.resolveMaterial(material, false);
assert.equal(first, second);
assert.equal(counters.textures, WEBGPU_TEXTURE_SLOT_COUNT);
assert.equal(counters.samplers, WEBGPU_TEXTURE_SLOT_COUNT);
assert.equal(first.textures[WEBGPU_TEXTURE_SLOT.ANISOTROPY].id, "texture:16");
assert.equal("anisotropyTexture" in first, false);

material.roughness = 0.2;
cache.beginFrame();
const third = await cache.resolveMaterial(material, false);
assert.notEqual(third, first);
assert.equal(counters.textures, WEBGPU_TEXTURE_SLOT_COUNT * 2);
assert.equal(counters.samplers, WEBGPU_TEXTURE_SLOT_COUNT * 2);

const solid = cache.resolveMaterial(material, false);
const wire = cache.resolveMaterial(material, true);
assert.notEqual(solid, wire, "wireframe snapshots retain separate identities");
assert.equal(cache.resolveMaterial(material, true), wire, "concurrent lookups share one promise");
await wire;
cache.beginFrame();
assert.deepEqual(cache.getDebugStats().materialSnapshots, { frameHits: 0, frameResolves: 0 });
assert.equal(cache.resolveMaterial(material, false), solid, "frame changes retain valid snapshots");
cache.destroy();
assert.throws(() => cache.resolveMaterial(material, false), /destroyed/);

let failTextures = true;
const retryCache = new WebGPUDrawResourceCache({}, {}, {}, {
	async getTextureForSlotAsync() {
		if (failTextures) throw new Error("texture unavailable");
		return {};
	},
	getSamplerForTexture() { return {}; },
});
await assert.rejects(retryCache.resolveMaterial(material, false), /texture unavailable/);
failTextures = false;
assert.ok(await retryCache.resolveMaterial(material, false), "failed snapshots remain retryable");
retryCache.destroy();

let finishTexture;
let samplerCalls = 0;
const pendingTexture = new Promise((resolve) => { finishTexture = resolve; });
const pendingCache = new WebGPUDrawResourceCache({}, {}, {}, {
	getTextureForSlotAsync() { return pendingTexture; },
	getSamplerForTexture() { samplerCalls++; return {}; },
});
const pending = pendingCache.resolveMaterial(material, false);
pendingCache.destroy();
const rejected = assert.rejects(pending, /destroyed/);
finishTexture({});
await rejected;
assert.equal(samplerCalls, 0, "pending work cannot resolve resources after cache teardown");

const refreshCache = new WebGPUDrawResourceCache({}, {}, {}, textures);
const refreshed = new PBRMaterial();
const refreshRevision = refreshed.refreshRevision.bind(refreshed);
let refreshCalls = 0;
refreshed.refreshRevision = () => { refreshCalls++; return refreshRevision(); };
refreshCache.beginFrame();
refreshCache.refreshMaterialRevision(refreshed);
const refreshSolid = await refreshCache.resolveMaterial(refreshed, false);
const refreshWire = await refreshCache.resolveMaterial(refreshed, true);
assert.equal(refreshSolid.wireframe, false);
assert.equal(refreshWire.wireframe, true);
assert.equal(refreshCalls, 1, "both snapshots share their material's refresh state");
refreshCache.beginFrame();
refreshCache.refreshMaterialRevision(refreshed);
await refreshCache.resolveMaterial(refreshed, true);
assert.equal(refreshCalls, 2);
const custom = new ShaderMaterial({ uniformBindings: [{ name: "time", type: "f32", value: 1 }] });
const customFirst = await refreshCache.resolveMaterial(custom, false);
custom.setUniform("time", 2);
assert.notEqual(await refreshCache.resolveMaterial(custom, false), customFirst,
	"custom uniforms still observe edits within the same frame");
refreshCache.destroy();

console.log("WebGPU material snapshot cache tests passed");
