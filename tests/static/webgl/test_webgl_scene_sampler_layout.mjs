import assert from "node:assert/strict";

import {
	createWebGLSceneSamplerLayout,
} from "../../../src/backends/webgl/WebGLSceneSamplerLayout.ts";
import {
	WEBGL_FULL_SCENE_VARIANT,
} from "../../../src/shaders/webgl/sceneVariants.ts";
import {
	resolveWebGLBuiltinSceneVariant,
} from "../../../src/backends/webgl/WebGLSceneProgramVariants.ts";
import { PBRMaterial } from "../../../src/materials/PBRMaterial.ts";

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

function testInactiveExtensionMapsArePruned() {
	const clearcoatMap = {};
	const specularMap = {};
	const material = new PBRMaterial({
		clearcoatMap,
		specularFactor: 0,
		specularMap,
	});
	const environment = {
		lightState: null,
		enableShadowTransmittance: false,
		enableIrradianceProbeGrid: false,
	};
	const context = { features: {} };
	const inactive = resolveWebGLBuiltinSceneVariant(
		context,
		material,
		"single",
		0,
		environment,
	);
	assert.ok(inactive);
	assert.equal(inactive.material.clearcoat, false);
	assert.equal(inactive.material.clearcoatMap, false);
	assert.equal(inactive.material.specularMap, false);
	const inactiveLayout = createWebGLSceneSamplerLayout(64, inactive);
	assert.ok(!inactiveLayout.activeSamplerNames.includes("uClearcoatMap"));
	assert.ok(!inactiveLayout.activeSamplerNames.includes("uSpecularMap"));

	material.clearcoat = 1;
	material.specularFactor = 1;
	const active = resolveWebGLBuiltinSceneVariant(
		context,
		material,
		"single",
		0,
		environment,
	);
	assert.ok(active);
	assert.equal(active.material.clearcoat, true);
	assert.equal(active.material.clearcoatMap, true);
	assert.equal(active.material.specularMap, true);
	const activeLayout = createWebGLSceneSamplerLayout(64, active);
	assert.ok(activeLayout.activeSamplerNames.includes("uClearcoatMap"));
	assert.ok(activeLayout.activeSamplerNames.includes("uSpecularMap"));

	const specularColorMap = {};
	const blackSpecular = new PBRMaterial({
		specularColor: { r: 0, g: 0, b: 0 },
		specularColorMap,
	});
	const blackSpecularVariant = resolveWebGLBuiltinSceneVariant(
		context,
		blackSpecular,
		"single",
		0,
		environment,
	);
	assert.ok(blackSpecularVariant);
	assert.equal(blackSpecularVariant.material.specularColorMap, false);
	blackSpecular.specularColor = { r: 255, g: 255, b: 255 };
	const whiteSpecularVariant = resolveWebGLBuiltinSceneVariant(
		context,
		blackSpecular,
		"single",
		0,
		environment,
	);
	assert.ok(whiteSpecularVariant);
	assert.equal(whiteSpecularVariant.material.specularColorMap, true);
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
testInactiveExtensionMapsArePruned();
testCustomLayoutDeduplicatesActiveNames();
testSparseShadowAndGridLayoutFitsBelowLegacyThreshold();
console.log("WebGL scene sampler layout tests passed");
