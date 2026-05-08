import assert from "node:assert/strict";
import { Material, AlphaMode } from "../src/materials/Material.ts";
import { PBRMaterial } from "../src/materials/PBRMaterial.ts";
import { resolveMaterialShadowTransmittance } from "../src/materials/transparency.ts";

function assertClose(actual, expected, tolerance = 1e-6) {
	assert.ok(
		Math.abs(actual - expected) <= tolerance,
		`expected ${actual} to be within ${tolerance} of ${expected}`
	);
}

function testPBRTransmissionIncludesFresnelAndVolumeAbsorption() {
	const material = new PBRMaterial({
		albedo: { r: 255, g: 128, b: 64 },
		transmissionFactor: 0.75,
		ior: 1.5,
		thicknessFactor: 0.2,
		attenuationDistance: 0.4,
		attenuationColor: { r: 255, g: 64, b: 16 },
	});
	const transmittance = resolveMaterialShadowTransmittance(material);
	const fresnelTransmittance = 1 - Math.pow((1.5 - 1) / (1.5 + 1), 2);

	assertClose(transmittance.r, 0.75 * fresnelTransmittance);
	assertClose(
		transmittance.g,
		0.75 * fresnelTransmittance * (128 / 255) * Math.sqrt(64 / 255)
	);
	assertClose(
		transmittance.b,
		0.75 * fresnelTransmittance * (64 / 255) * Math.sqrt(16 / 255)
	);
}

function testAlphaBlendFallbackKeepsOpacityWeightedColorFilter() {
	const material = new Material({
		alphaMode: AlphaMode.Blend,
		opacity: 0.25,
	});
	material.color = { r: 255, g: 0, b: 0 };

	const transmittance = resolveMaterialShadowTransmittance(material);
	assertClose(transmittance.r, 1);
	assertClose(transmittance.g, 0.75);
	assertClose(transmittance.b, 0.75);
}

function testOpaqueNonTransmissiveMaterialBlocksShadowLight() {
	const material = new Material();
	const transmittance = resolveMaterialShadowTransmittance(material);

	assert.deepEqual(transmittance, { r: 0, g: 0, b: 0 });
}

testPBRTransmissionIncludesFresnelAndVolumeAbsorption();
testAlphaBlendFallbackKeepsOpacityWeightedColorFilter();
testOpaqueNonTransmissiveMaterialBlocksShadowLight();

console.log("transparent shadow transmission tests passed");
