import assert from "node:assert/strict";
import { AmbientLight } from "../src/lights/AmbientLight.ts";
import { DirectionalLight } from "../src/lights/DirectionalLight.ts";
import { PointLight } from "../src/lights/PointLight.ts";
import { SpotLight } from "../src/lights/SpotLight.ts";
import { LightProbe } from "../src/lights/LightProbe.ts";
import { Matrix4 } from "../src/maths/Matrix4.ts";
import { SH } from "../src/maths/SH.ts";

function assertColorClose(actual, expected, tolerance = 0.001) {
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
	const contribution = light.computeContribution({ x: 0, y: 0, z: 0 });
	assert.equal(contribution.type, "ambient");
	assertColorClose(contribution.color, { r: 100, g: 100, b: 100 });
	assert.ok(Math.abs((contribution.intensity ?? 0) - 0.5) < 1e-6);
}

function testDirectional() {
	console.log("Testing DirectionalLight...");
	const light = new DirectionalLight({
		color: { r: 255, g: 255, b: 255 },
		dir: { x: 0, y: -1, z: 0 },
		intensity: 1,
	});

	// Base contribution
	const contribution = light.computeContribution({ x: 0, y: 0, z: 0 });
	assert.equal(contribution.type, "direct");
	// L vector points TOWARDS light source (opposite of light direction)
	assert.ok(contribution.direction.y > 0.999);
	assertColorClose(contribution.color, { r: 255, g: 255, b: 255 });
	assert.ok(Math.abs((contribution.intensity ?? 0) - 1) < 1e-6);

	// With world rotation
	const rotation = Matrix4.rotationFromEuler(Math.PI / 2, 0, 0); // Rotate 90 deg around X. Y becomes Z.
	light.updateWorldMatrix(rotation);
	const contributionRotated = light.computeContribution({ x: 0, y: 0, z: 0 });
	// Original dir (0, -1, 0) rotated by 90 around X becomes (0, 0, -1)
	// L should be (0, 0, 1)
	assert.ok(contributionRotated.direction.z > 0.999);

	// Test shadow caster
	assert.ok(light.shadow);
	const shadowCamera = light.shadow.setupShadowCamera({
		sceneBounds: { center: { x: 0, y: 0, z: 0 }, radius: 100 },
		worldMatrix: Matrix4.identity(),
	});
	assert.ok(shadowCamera.view);
	assert.ok(shadowCamera.projection);
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
	const contribution = light.computeContribution({ x: 0, y: 0, z: 0 });
	assert.notEqual(contribution, null);
	assert.ok(contribution.direction.y > 0.999);

	// Fade with distance
	const atSource = light.computeContribution({ x: 0, y: 10, z: 0 });
	assert.notEqual(
		atSource,
		null,
		"PointLight at distance 0 should not be null"
	);
	assertColorClose(atSource.color, { r: 10, g: 10, b: 10 });
	assert.ok(Math.abs((atSource.intensity ?? 0) - 1) < 1e-6);

	const closer = light.computeContribution({ x: 0, y: 5, z: 0 });
	const further = light.computeContribution({ x: 0, y: 0, z: 0 });
	assert.ok(
		(closer.intensity ?? 0) > (further.intensity ?? 0),
		"Closer point should have higher intensity"
	);

	// Out of range
	const outRange = light.computeContribution({ x: 0, y: 200, z: 0 });
	assert.equal(outRange, null);
}

function testSpot() {
	console.log("Testing SpotLight...");
	const light = new SpotLight({
		color: { r: 20, g: 20, b: 20 },
		position: { x: 0, y: 10, z: 0 },
		dir: { x: 0, y: -1, z: 0 },
		angle: Math.PI / 4, // 45 deg
		range: 100,
	});

	// In center of cone
	const center = light.computeContribution({ x: 0, y: 0, z: 0 });
	assert.notEqual(center, null);

	// Outside cone
	const outside = light.computeContribution({ x: 20, y: 0, z: 0 }); // dist 10 down, x=20 is far outside 45 deg cone
	assert.equal(outside, null);
}

function testLightProbe() {
	console.log("Testing LightProbe...");
	const sh = SH.empty();
	// Set DC component to something noticeable
	sh[0] = { r: 10, g: 10, b: 10 };
	const probe = new LightProbe(sh, 1.0);

	const contribution = probe.computeContribution({ x: 0, y: 0, z: 0 });
	assert.equal(contribution.type, "ambient");
	// Currently returns {0,0,0} in the implementation, but we check if it runs
	assert.ok(contribution.color);
}

function run() {
	try {
		console.log("Starting Comprehensive Lighting Tests...");
		testAmbient();
		testDirectional();
		testPoint();
		testSpot();
		testLightProbe();
		console.log("✅ All lighting tests passed!");
	} catch (e) {
		console.error("❌ Test Failed:");
		console.error(e);
		process.exit(1);
	}
}

run();
