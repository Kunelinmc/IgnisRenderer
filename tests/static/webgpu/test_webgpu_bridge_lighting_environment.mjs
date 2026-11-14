import assert from "node:assert/strict";
import { DirectionalLight } from "../../../src/lights/DirectionalLight.ts";
import { AmbientLight } from "../../../src/lights/AmbientLight.ts";
import { LightProbe } from "../../../src/lights/LightProbe.ts";
import { SH } from "../../../src/maths/SH.ts";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import { collectWebGPULightingCatalog } from "../../../src/backends/webgpu/lights.ts";
import { collectWebGPUEnvironment } from "../../../src/backends/webgpu/environment.ts";

const light = new DirectionalLight();
const prepared = {
	light,
	lightId: light.id,
	definition: {
		bias: {},
		filterMode: "pcf",
		sampling: { quality: "medium" },
		strength: 1,
		projection: {},
	},
	requestedTechnique: "single",
	effectiveTechnique: "single",
	requestedCascadeCount: 1,
	effectiveCascadeCount: 1,
	requestedResolution: 512,
	effectiveResolution: 512,
	sampling: { quality: "medium" },
	requestedFilterMode: "pcf",
	effectiveFilterMode: "pcf",
	priority: 0,
	cost: 1,
	score: 1,
	slices: [{
		index: 0,
		resolution: 512,
		view: Matrix4.identity(),
		projection: Matrix4.identity(),
		viewProjection: Matrix4.identity(),
		lightDirection: { x: 0, y: -1, z: 0 },
		splitNear: 0,
		splitFar: 1,
	}],
};
const plan = {
	revision: 1,
	lights: [prepared],
	diagnostics: [],
	hasRasterWork: true,
	hasTransmissionWork: false,
};
const catalog = collectWebGPULightingCatalog([light], false, true, plan);
assert.equal(catalog.lights[0].shadow.enabled, true);
console.log("WebGPU lighting bridge reads prepared shadow data");

const ambient = new AmbientLight({ intensity: 2 });
const globalSH = SH.empty();
globalSH[0] = { r: 30, g: 20, b: 10 };
globalSH[5] = { r: -4, g: 2, b: 3 };
const globalProbe = new LightProbe({ sh: globalSH });
const localProbe = new LightProbe({ sh: globalSH, shape: "sphere" });
for (const probes of [[], [globalProbe], [localProbe]]) {
	const lights = [ambient, light, ...probes];
	const scene = {
		lights,
		environment: {
			backgroundEnabled: false, lightingEnabled: false,
			backgroundTexture: null, iblTexture: null,
			backgroundStrength: 1, backgroundTintLinear: { r: 1, g: 1, b: 1 },
			backgroundExposure: 0,
		},
	};
	const environment = collectWebGPUEnvironment(scene, true, null);
	assert.deepEqual(environment.shAmbientCoeffs, probes[0] === globalProbe ? globalSH : SH.empty());
	assert.equal(environment.hasSHAmbient, probes.length > 0);
	assert.equal(environment.localLightProbeCount, probes[0] === localProbe ? 1 : 0);
	const lighting = collectWebGPULightingCatalog(lights, true, false);
	assert.deepEqual(lighting.ambientColor, [2, 2, 2]);
}
console.log("WebGPU environment keeps analytical ambient separate from probe SH");
