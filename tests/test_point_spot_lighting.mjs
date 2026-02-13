import assert from "node:assert/strict";
import { PointLight } from "../src/lights/PointLight.ts";
import { SpotLight } from "../src/lights/SpotLight.ts";
import { Matrix4 } from "../src/maths/Matrix4.ts";

function assertColorClose(actual, expected, tolerance = 0.001) {
	const dr = Math.abs(actual.r - expected.r);
	const dg = Math.abs(actual.g - expected.g);
	const db = Math.abs(actual.b - expected.b);
	assert.ok(
		dr < tolerance && dg < tolerance && db < tolerance,
		`Color mismatch: got {${actual.r}, ${actual.g}, ${actual.b}}, expected {${expected.r}, ${expected.g}, ${expected.b}}`
	);
}

function testPointLightEdgeCases() {
	console.log("Testing PointLight edge cases...");
	const light = new PointLight({
		color: { r: 100, g: 100, b: 100 },
		position: { x: 10, y: 10, z: 10 },
		range: 50,
		intensity: 1.0,
	});

	// Case 1: Exactly at the light position (Distance 0)
	const atSource = light.computeContribution({ x: 10, y: 10, z: 10 });
	assert.notEqual(
		atSource,
		null,
		"Contribution at distance 0 should not be null"
	);
	assertColorClose(atSource.color, { r: 100, g: 100, b: 100 });
	assert.ok(atSource.direction, "Should have a direction even at distance 0");

	// Case 2: Just inside range
	const inside = light.computeContribution({ x: 10 + 49.9, y: 10, z: 10 });
	assert.notEqual(inside, null);
	assert.ok(inside.color.r > 0);

	// Case 3: Just outside range
	const outside = light.computeContribution({ x: 10 + 50.1, y: 10, z: 10 });
	assert.equal(outside, null, "Should be null outside range");

	// Case 4: With world transformation
	const transform = Matrix4.fromTranslation([-10, -10, -10]); // Move light to origin
	light.updateWorldMatrix(transform);
	const atNewSource = light.computeContribution({ x: 0, y: 0, z: 0 });
	assert.notEqual(atNewSource, null);
	assertColorClose(atNewSource.color, { r: 100, g: 100, b: 100 });
}

function testSpotLightEdgeCases() {
	console.log("Testing SpotLight edge cases...");
	const light = new SpotLight({
		color: { r: 100, g: 100, b: 100 },
		position: { x: 0, y: 10, z: 0 },
		dir: { x: 0, y: -1, z: 0 },
		angle: Math.PI / 4, // 45 deg
		range: 100,
		penumbra: 0.5, // inner angle is 22.5 deg
	});

	// Case 1: At source (distance 0)
	const atSource = light.computeContribution({ x: 0, y: 10, z: 0 });
	assert.notEqual(atSource, null);
	assertColorClose(atSource.color, { r: 100, g: 100, b: 100 });

	// Case 2: On the axis
	const onAxis = light.computeContribution({ x: 0, y: 0, z: 0 });
	assert.notEqual(onAxis, null);

	// Case 3: In penumbra (between inner and outer)
	// Outer is 45 deg, inner is 22.5 deg. At y=0 (dist 10), x=7 is ~35 deg
	const inPenumbra = light.computeContribution({ x: 7, y: 0, z: 0 });
	assert.notEqual(inPenumbra, null);
	assert.ok(
		inPenumbra.color.r < onAxis.color.r,
		"Penumbra should be dimmer than axis"
	);

	// Case 4: Rotation
	const rotation = Matrix4.rotationFromEuler(0, 0, Math.PI / 2); // Rotate 90 deg around Z. Dir -Y becomes +X
	light.updateWorldMatrix(rotation);
	const hit = light.computeContribution({ x: 10, y: 10, z: 0 }); // 10 units in +X from position (0,10,0)
	assert.notEqual(
		hit,
		null,
		"Should hit point in new direction after rotation"
	);
}

function run() {
	try {
		testPointLightEdgeCases();
		testSpotLightEdgeCases();
		console.log("✅ Point and Spot light edge case tests passed!");
	} catch (e) {
		console.error("❌ Test Failed:");
		console.error(e);
		process.exit(1);
	}
}

run();
