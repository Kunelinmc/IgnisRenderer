import assert from "node:assert/strict";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import { createParticleShadowVolumeGrid, injectParticleBatchIntoShadowVolume } from "../../../src/pipeline/ParticleShadowVolume.ts";
import { ParticleBlendMode } from "../../../src/particles/types.ts";

const grid = createParticleShadowVolumeGrid({ width: 4, height: 4, depth: 4 });
injectParticleBatchIntoShadowVolume(grid, Matrix4.identity(), { castShadows: true, blendMode: ParticleBlendMode.Alpha, shadowDensity: 1, shadowSoftness: 1, particles: [{ position: { x: 0, y: 0, z: 0 }, size: 1, color: { a: 1 } }] });
assert.equal(grid.active, true);
console.log("WebGPU particle fixtures use matrices without legacy shadow maps");
