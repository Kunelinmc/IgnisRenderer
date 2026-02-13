import assert from "node:assert/strict";
import { SH } from "../src/maths/SH.ts";

function testSHReconstruction() {
	console.log("Testing SH reconstruction...");

	// Case 1: Constant Ambient
	// Reconstruct irradiance from a constant SH DC coefficient
	// For constant radiance L, irradiance should be L * PI
	const radiance = 100;
	const sh = SH.empty();
	// Y00 = 0.282095. To get constant radiance radiance, coeff[0] = radiance / Y00
	const Y00 = 0.282095;
	sh[0] = { r: radiance / Y00, g: radiance / Y00, b: radiance / Y00 };

	const normal = { x: 0, y: 1, z: 0 };
	const irradiance = SH.calculateIrradiance(normal, sh);

	// Expected irradiance = radiance * PI
	const expected = radiance * Math.PI;
	const actual = irradiance.r;
	const diff = Math.abs(actual - expected);

	assert.ok(
		diff < 1.0,
		`Irradiance mismatch: got ${actual}, expected ${expected}`
	);
	console.log("✅ Basic SH Irradiance reconstruction passed");
}

function testSHDirectionality() {
	console.log("Testing SH directionality...");

	// Project a light from +Y
	const lightDir = { x: 0, y: 1, z: 0 };
	const color = { r: 100, g: 0, b: 0 };
	const sh = SH.projectDirectionalLight(lightDir, color);

	// Normal facing light (+Y)
	const irradianceHit = SH.calculateIrradiance({ x: 0, y: 1, z: 0 }, sh);
	// Normal facing away (-Y)
	const irradianceMiss = SH.calculateIrradiance({ x: 0, y: -1, z: 0 }, sh);

	assert.ok(
		irradianceHit.r > irradianceMiss.r,
		"Normal facing light should receive more irradiance"
	);
	assert.ok(irradianceMiss.r >= 0, "Irradiance should not be negative");

	console.log("✅ SH Directional projection passed");
}

function run() {
	try {
		testSHReconstruction();
		testSHDirectionality();
	} catch (e) {
		console.error("❌ SH test failed");
		console.error(e);
		process.exit(1);
	}
}

run();
