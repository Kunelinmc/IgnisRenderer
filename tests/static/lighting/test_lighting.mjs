import assert from "node:assert/strict";
import { AmbientLight } from "../../../src/lights/AmbientLight.ts";
import { AreaLight } from "../../../src/lights/AreaLight.ts";
import { DirectionalLight } from "../../../src/lights/DirectionalLight.ts";
import { PointLight } from "../../../src/lights/PointLight.ts";
import { SpotLight } from "../../../src/lights/SpotLight.ts";
import { LightProbe } from "../../../src/lights/LightProbe.ts";
import { Scene } from "../../../src/core/Scene.ts";
import { evaluateLightContribution } from "../../../src/shaders/software/LightEvaluator.ts";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import { SH } from "../../../src/maths/SH.ts";
import { isShadowCastingLight } from "../../../src/lights/index.ts";

function assertColorClose(actual, expected, tolerance = 1.0) {
	const dr = Math.abs(actual.r - expected.r);
	const dg = Math.abs(actual.g - expected.g);
	const db = Math.abs(actual.b - expected.b);
	assert.ok(
		dr < tolerance && dg < tolerance && db < tolerance,
		`Color mismatch: got {${actual.r}, ${actual.g}, ${actual.b}}, expected {${expected.r}, ${expected.g}, ${expected.b}}`
	);
}

function testAmbient() {
	console.log("Testing AmbientLight...");
	const light = new AmbientLight({
		color: { r: 100, g: 100, b: 100 },
		intensity: 0.5,
	});
	const contribution = evaluateLightContribution(light, {
		position: { x: 0, y: 0, z: 0 },
	});
	assert.equal(contribution.type, "ambient");
	assertColorClose(contribution.color, { r: 100, g: 100, b: 100 });
	assert.ok(Math.abs((contribution.intensity ?? 0) - 0.5) < 1e-6);
}

function testDirectional() {
	console.log("Testing DirectionalLight...");
	const light = new DirectionalLight({
		color: { r: 255, g: 255, b: 255 },
		direction: { x: 0, y: -1, z: 0 },
		intensity: 1,
	});

	// Base contribution
	const contribution = evaluateLightContribution(light, {
		position: { x: 0, y: 0, z: 0 },
	});
	assert.equal(contribution.type, "direct");
	// L vector points TOWARDS light source (opposite of light direction)
	assert.ok(contribution.direction.y > 0.999);
	assertColorClose(contribution.color, { r: 255, g: 255, b: 255 });
	assert.ok(Math.abs((contribution.intensity ?? 0) - 1) < 1e-6);

	// With world rotation
	const rotation = Matrix4.rotationFromEuler(Math.PI / 2, 0, 0); // Rotate 90 deg around X. Y becomes Z.
	light.updateWorldMatrix(rotation);
	const contributionRotated = evaluateLightContribution(light, {
		position: { x: 0, y: 0, z: 0 },
	});
	// Original dir (0, -1, 0) rotated by 90 around X becomes (0, 0, -1)
	// L should be (0, 0, 1)
	assert.ok(contributionRotated.direction.z > 0.999);

	// Shadow config is now scene-managed, not light-owned.
	const scene = new Scene();
	scene.add(light);
	const shadowMap = scene.shadows.createSingle({ size: 1024 });
	scene.shadows.bind(light, shadowMap);
	const boundShadow = scene.shadows.getBoundShadowMap(light);
	assert.equal(boundShadow, shadowMap);
	assert.equal(shadowMap.snapshot().resolution, 1024);
}

function testPoint() {
	console.log("Testing PointLight...");
	const light = new PointLight({
		color: { r: 10, g: 10, b: 10 },
		position: { x: 0, y: 10, z: 0 },
		range: 100,
		intensity: 1,
	});

	// Directly under
	const contribution = evaluateLightContribution(light, {
		position: { x: 0, y: 0, z: 0 },
	});
	assert.notEqual(contribution, null);
	assert.ok(contribution.direction.y > 0.999);

	// Fade with distance
	const atSource = evaluateLightContribution(light, {
		position: { x: 0, y: 10, z: 0 },
	});
	assert.notEqual(
		atSource,
		null,
		"PointLight at distance 0 should not be null"
	);
	assertColorClose(atSource.color, { r: 10, g: 10, b: 10 });
	assert.ok(Math.abs((atSource.intensity ?? 0) - 1) < 1e-6);

	const closer = evaluateLightContribution(light, {
		position: { x: 0, y: 5, z: 0 },
	});
	const further = evaluateLightContribution(light, {
		position: { x: 0, y: 0, z: 0 },
	});
	assert.ok(
		(closer.intensity ?? 0) > (further.intensity ?? 0),
		"Closer point should have higher intensity"
	);

	// Out of range
	const outRange = evaluateLightContribution(light, {
		position: { x: 0, y: 200, z: 0 },
	});
	assert.equal(outRange, null);
}

function testSpot() {
	console.log("Testing SpotLight...");
	const light = new SpotLight({
		color: { r: 20, g: 20, b: 20 },
		position: { x: 0, y: 10, z: 0 },
		direction: { x: 0, y: -1, z: 0 },
		outerAngle: Math.PI / 4, // 45 deg
		range: 100,
	});

	// In center of cone
	const center = evaluateLightContribution(light, {
		position: { x: 0, y: 0, z: 0 },
	});
	assert.notEqual(center, null);

	// Outside cone
	const outside = evaluateLightContribution(light, {
		position: { x: 20, y: 0, z: 0 },
	}); // dist 10 down, x=20 is far outside 45 deg cone
	assert.equal(outside, null);
}

function testArea() {
	console.log("Testing AreaLight...");
	const light = new AreaLight({
		color: { r: 30, g: 30, b: 30 },
		width: 20,
		height: 10,
		range: 1_000_000,
		intensity: 1,
	});
	assert.equal(isShadowCastingLight(light), true);

	const contribution = evaluateLightContribution(light, {
		position: { x: 0, y: 10, z: 0 },
	});
	assert.notEqual(contribution, null);
	assert.ok(contribution.direction.y < -0.999);

	const halfWidth = 10;
	const halfHeight = 5;
	const distance = 10;
	const expectedSolidAngle =
		4 *
		Math.atan(
			(halfWidth * halfHeight) /
				(distance *
					Math.sqrt(
						distance * distance +
							halfWidth * halfWidth +
							halfHeight * halfHeight
					))
		);
	assert.ok(
		Math.abs((contribution.intensity ?? 0) - expectedSolidAngle) < 1e-5,
		"AreaLight should use projected solid angle instead of arbitrary scale"
	);

	const behind = evaluateLightContribution(light, {
		position: { x: 0, y: -10, z: 0 },
	});
	assert.equal(behind, null);
}

function testLightProbe() {
	console.log("Testing LightProbe...");
	const sh = SH.empty();
	// Set DC component.
	// To get a specific linear irradiance E, we set DC = E / (PI * Y00)
	// Let's target E = 127.5 (half max linear)
	const targetLinearIrr = 127.5;
	const Y00 = 0.282095;
	const dcVal = targetLinearIrr / (Math.PI * Y00);
	sh[0] = { r: dcVal, g: dcVal, b: dcVal };

	const probe = new LightProbe({ sh });
	const contribution = evaluateLightContribution(probe, {
		position: { x: 0, y: 0, z: 0 },
	});

	assert.ok(contribution, "LightProbe contribution should not be null");
	assert.equal(contribution.type, "irradiance");

	// Verification:
	// Linear Irradiance target was 127.5.
	// 127.5 / 255 = 0.5 linear.
	// sRGB(0.5) approx 0.735. 0.735 * 255 approx 187.
	assertColorClose(contribution.color, { r: 187, g: 187, b: 187 }, 5.0);
	assert.ok(Math.abs((contribution.intensity ?? 0) - 1.0) < 1e-6);
}

function testSHBasisBufferReuse() {
	console.log("Testing SH basis buffer reuse...");
	const invLen = 1 / Math.sqrt(14);
	const direction = { x: invLen, y: 2 * invLen, z: 3 * invLen };
	const expected = SH.evalBasis(direction);
	const buffer = new Float32Array(16);
	const actual = SH.evalBasis(direction, buffer);

	assert.equal(actual, buffer);
	for (let i = 0; i < expected.length; i++) {
		assert.ok(
			Math.abs(buffer[i] - expected[i]) < 1e-6,
			`SH basis mismatch at index ${i}: got ${buffer[i]}, expected ${expected[i]}`
		);
	}
}

function run() {
	try {
		console.log("Starting Comprehensive Lighting Tests...");
		testAmbient();
		testDirectional();
		testPoint();
		testSpot();
		testArea();
		testLightProbe();
		testSHBasisBufferReuse();
		console.log("✅ All lighting tests passed!");
	} catch (e) {
		console.error("❌ Test Failed:");
		console.error(e);
		process.exit(1);
	}
}

run();
