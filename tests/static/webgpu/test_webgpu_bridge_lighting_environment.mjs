import assert from "node:assert/strict";
import { DirectionalLight } from "../../../src/lights/DirectionalLight.ts";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import { collectWebGPULightingCatalog } from "../../../src/backends/webgpu/lights.ts";

const light = new DirectionalLight();
const prepared = { light, lightId: light.id, definition: { bias: {}, sampling: {}, projection: {} }, requestedTechnique: "single", effectiveTechnique: "single", requestedCascadeCount: 1, effectiveCascadeCount: 1, requestedResolution: 512, effectiveResolution: 512, sampling: { pcfRadius: 1, radius: 0, strength: 1, samples: 16, searchSamples: 16 }, filterMode: "pcf", storage: "atlas", priority: 0, cost: 1, score: 1, slices: [{ index: 0, resolution: 512, view: Matrix4.identity(), projection: Matrix4.identity(), viewProjection: Matrix4.identity(), lightDirection: { x: 0, y: -1, z: 0 }, splitNear: 0, splitFar: 1 }] };
const catalog = collectWebGPULightingCatalog([light], true, false, true, { revision: 1, lights: [prepared], jobs: [], diagnostics: [], hasRasterWork: false, hasTransmissionWork: false, hasPagedWork: false });
assert.equal(catalog.lights[0].shadow.enabled, true);
console.log("WebGPU lighting bridge reads prepared shadow data");
