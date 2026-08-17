import assert from "node:assert/strict";

import {
	createWebGLSceneSamplerLayout,
} from "../../../src/backends/webgl/WebGLSceneSamplerLayout.ts";
import {
	WEBGL_FULL_SCENE_VARIANT,
} from "../../../src/shaders/webgl/sceneVariants.ts";

function testFullLayoutIsDenseAndCollisionFree() {
	const layout = createWebGLSceneSamplerLayout(
		64,
		WEBGL_FULL_SCENE_VARIANT,
	);
	const units = Object.values(layout.units);
	assert.equal(units.length, layout.required);
	assert.equal(new Set(units).size, units.length);
	assert.deepEqual(units, units.map((_, index) => index));
	assert.ok(layout.activeSamplerNames.includes("uAnisotropyMap"));
	assert.ok(layout.activeSamplerNames.includes("uIridescenceThicknessMap"));
	assert.notEqual(
		layout.units.uAnisotropyMap,
		layout.units.uIridescenceThicknessMap,
	);
}

function testOverflowHasStableDiagnostic() {
	const required = createWebGLSceneSamplerLayout(
		64,
		WEBGL_FULL_SCENE_VARIANT,
	).required;
	assert.throws(
		() => createWebGLSceneSamplerLayout(
			required - 1,
			WEBGL_FULL_SCENE_VARIANT,
		),
		(error) => {
			assert.equal(error?.code, "material-texture-unit-overflow");
			assert.match(error.message, new RegExp(`required=${required}`));
			assert.match(error.message, new RegExp(`available=${required - 1}`));
			assert.match(error.message, /uTransmissionBackgroundMap/);
			assert.match(error.message, /uAnisotropyMap/);
			return true;
		},
	);
}

function testCustomLayoutDeduplicatesActiveNames() {
	const layout = createWebGLSceneSamplerLayout(
		2,
		undefined,
		["uCustomMap", "uCustomMap", "uNoiseMap"],
	);
	assert.deepEqual(layout.activeSamplerNames, ["uCustomMap", "uNoiseMap"]);
	assert.deepEqual(layout.units, { uCustomMap: 0, uNoiseMap: 1 });
}

function testSparseShadowAndGridLayoutFitsBelowLegacyThreshold() {
	const variant = {
		...WEBGL_FULL_SCENE_VARIANT,
		output: "single",
		oit: false,
		scene: {
			shadows: true,
			shadowTransmittance: true,
			clusteredLighting: false,
			sh: true,
			localLightProbes: true,
			irradianceProbeGrid: true,
			reflectionProbes: false,
			environmentSpecular: false,
		},
		material: {
			...WEBGL_FULL_SCENE_VARIANT.material,
			model: "unlit",
			baseMap: false,
			metallicRoughnessMap: false,
			specularMap: false,
			specularColorMap: false,
			normalMap: false,
			emissiveMap: false,
			occlusionMap: false,
			clearcoat: false,
			clearcoatMap: false,
			clearcoatRoughnessMap: false,
			clearcoatNormalMap: false,
			sheen: false,
			sheenColorMap: false,
			sheenRoughnessMap: false,
			iridescence: false,
			iridescenceMap: false,
			iridescenceThicknessMap: false,
			anisotropy: false,
			anisotropyMap: false,
			transmission: false,
			transmissionMap: false,
			thicknessMap: false,
			alphaMask: false,
		},
	};
	const layout = createWebGLSceneSamplerLayout(5, variant);
	assert.equal(layout.required, 5);
	assert.ok(layout.activeSamplerNames.includes("uShadowTransmittanceAtlas"));
	assert.ok(layout.activeSamplerNames.includes("uIrradianceProbeGridCoeffs"));
}

testFullLayoutIsDenseAndCollisionFree();
testOverflowHasStableDiagnostic();
testCustomLayoutDeduplicatesActiveNames();
testSparseShadowAndGridLayoutFitsBelowLegacyThreshold();
console.log("WebGL scene sampler layout tests passed");
