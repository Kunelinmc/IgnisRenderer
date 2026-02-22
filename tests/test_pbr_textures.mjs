import assert from "node:assert/strict";
import { PBREvaluator } from "../src/shaders/PBREvaluator.ts";
import { PBRMaterial } from "../src/materials/PBRMaterial.ts";
import { Texture } from "../src/core/Texture.ts";
import { Vector3 } from "../src/maths/Vector3.ts";

function createMockFace() {
	return {
		vertices: [],
		projected: [],
		center: { x: 0, y: 0, z: 0 },
		depthInfo: { min: 0, max: 0, avg: 0 },
	};
}

function createMockInput() {
	return {
		zCam: 1,
		world: { x: 0, y: 0, z: 0 },
		normal: { x: 0, y: 0, z: 1 },
		tangent: { x: 1, y: 0, z: 0, w: 1 },
		u: 0.5,
		v: 0.5,
	};
}

function create1x1Texture(r, g, b, a = 255) {
	const data = new Uint8ClampedArray([r, g, b, a]);
	return new Texture(data, 1, 1);
}

function assertColorClose(actual, expected, tolerance = 0.001) {
	const dr = Math.abs(actual.r - expected.r);
	const dg = Math.abs(actual.g - expected.g);
	const db = Math.abs(actual.b - expected.b);
	assert.ok(
		dr <= tolerance && dg <= tolerance && db <= tolerance,
		`Color mismatch: got {${actual.r}, ${actual.g}, ${actual.b}}, expected {${expected.r}, ${expected.g}, ${expected.b}}`
	);
}

function assertVectorClose(actual, expected, tolerance = 0.001) {
	const dx = Math.abs(actual.x - expected.x);
	const dy = Math.abs(actual.y - expected.y);
	const dz = Math.abs(actual.z - expected.z);
	assert.ok(
		dx <= tolerance && dy <= tolerance && dz <= tolerance,
		`Vector mismatch: got {${actual.x}, ${actual.y}, ${actual.z}}, expected {${expected.x}, ${expected.y}, ${expected.z}}`
	);
}

function testAlbedoMap() {
	console.log("Testing Albedo Map...");
	const material = new PBRMaterial({
		albedo: { r: 255, g: 255, b: 255 },
	});
	// Red texture
	material.map = create1x1Texture(255, 0, 0, 255);

	const evaluator = new PBREvaluator(material);
	const face = createMockFace();
	const input = createMockInput();

	const surface = evaluator.evaluate(input, face);
	assert.ok(surface);
	assertColorClose(surface.albedo, { r: 255, g: 0, b: 0 });
	assert.equal(surface.opacity, 1);
}

function testMetallicRoughnessMap() {
	console.log("Testing Metallic-Roughness Map...");
	const material = new PBRMaterial({
		roughness: 1.0,
		metalness: 1.0,
	});
	// In glTF, B channel is metalness, G channel is roughness
	// Let's set Roughness to 0.5 (127 or 128) and Metallic to 0.25 (64)
	material.metallicRoughnessMap = create1x1Texture(0, 128, 64, 255);

	const evaluator = new PBREvaluator(material);
	const face = createMockFace();
	const input = createMockInput();

	const surface = evaluator.evaluate(input, face);
	assert.ok(surface);
	// Roughness: 1.0 * (128 / 255) = 0.5019...
	assert.ok(Math.abs(surface.roughness - 128 / 255) < 0.001);
	// Metalness: 1.0 * (64 / 255) = 0.2509...
	assert.ok(Math.abs(surface.metalness - 64 / 255) < 0.001);
}

function testEmissiveMap() {
	console.log("Testing Emissive Map...");
	const material = new PBRMaterial({
		emissive: { r: 255, g: 255, b: 255 },
	});
	// Green emissive texture
	material.emissiveMap = create1x1Texture(0, 255, 0, 255);

	const evaluator = new PBREvaluator(material);
	const face = createMockFace();
	const input = createMockInput();

	const surface = evaluator.evaluate(input, face);
	assert.ok(surface);
	assertColorClose(surface.emissive, { r: 0, g: 255, b: 0 });
}

function testOcclusionMap() {
	console.log("Testing Occlusion Map...");
	const material = new PBRMaterial();
	// R channel is occlusion
	material.occlusionMap = create1x1Texture(128, 0, 0, 255);

	const evaluator = new PBREvaluator(material);
	const face = createMockFace();
	const input = createMockInput();

	const surface = evaluator.evaluate(input, face);
	assert.ok(surface);
	// Occlusion: 128 / 255 = 0.5019...
	assert.ok(Math.abs(surface.occlusion - 128 / 255) < 0.001);
}

function testNormalMap() {
	console.log("Testing Normal Map...");
	const material = new PBRMaterial();
	// Normal vector pointing in +X direction in tangent space:
	// mapped to RGB: (1, 0, 0) -> (255, 128, 128)
	material.normalMap = create1x1Texture(255, 128, 128, 255);

	const evaluator = new PBREvaluator(material);
	const face = createMockFace();
	const input = createMockInput();
	// N is (0, 0, 1), T is (1, 0, 0), B should be (0, 1, 0)
	input.normal = { x: 0, y: 0, z: 1 };
	input.tangent = { x: 1, y: 0, z: 0, w: 1 };

	const surface = evaluator.evaluate(input, face);
	assert.ok(surface);

	// The sampled Normal map is (255, 128, 128)
	// Extracted vector:
	// X: (255/255)*2 - 1 = 1.0
	// Y: (128/255)*2 - 1 = 0.00392 (roughly 0)
	// Z: (128/255)*2 - 1 = 0.00392 (roughly 0)
	// Let's precisely calculate expected Y and Z
	const expectedY = (128 / 255) * 2 - 1;
	const expectedZ = (128 / 255) * 2 - 1;

	// Because N=(0,0,1) and T=(1,0,0) and w=1
	// B = Cross(N, T) * w = (0,1,0)
	// NewNormal = T*x + B*y + N*z = (1*1, 1*expectedY, 1*expectedZ) -> Normalized
	const expectedVec = {
		x: 1,
		y: expectedY,
		z: expectedZ,
	};
	Vector3.normalizeInPlace(expectedVec);

	assertVectorClose(surface.normal, expectedVec);
}

function testNormalMapHandedness() {
	console.log("Testing Normal Map Handedness (w = -1)...");
	const material = new PBRMaterial();
	// Normal vector pointing in +Y direction in tangent space:
	// mapped to RGB: (0, 1, 0) -> (128, 255, 128)
	material.normalMap = create1x1Texture(128, 255, 128, 255);

	const evaluator = new PBREvaluator(material);
	const face = createMockFace();
	const input = createMockInput();

	input.normal = { x: 0, y: 0, z: 1 };
	// Set handedness w = -1
	input.tangent = { x: 1, y: 0, z: 0, w: -1 };

	const surface = evaluator.evaluate(input, face);
	assert.ok(surface);

	const expectedX = (128 / 255) * 2 - 1;
	const expectedY = (255 / 255) * 2 - 1;
	const expectedZ = (128 / 255) * 2 - 1;

	// B = Cross(N, T) * w = Cross((0,0,1), (1,0,0)) * -1 = (0,1,0) * -1 = (0,-1,0)
	// NewNormal = T*expectedX + B*expectedY + N*expectedZ
	// T=(1,0,0), B=(0,-1,0), N=(0,0,1)
	const expectedVec = {
		x: 1 * expectedX,
		y: -1 * expectedY,
		z: 1 * expectedZ,
	};
	Vector3.normalizeInPlace(expectedVec);

	assertVectorClose(surface.normal, expectedVec);
}

function run() {
	try {
		console.log("Starting PBR Texture Maps Tests...");
		testAlbedoMap();
		testMetallicRoughnessMap();
		testEmissiveMap();
		testOcclusionMap();
		testNormalMap();
		testNormalMapHandedness();
		console.log("✅ All PBR texture tests passed!");
	} catch (e) {
		console.error("❌ Test Failed:");
		console.error(e);
		process.exit(1);
	}
}

run();
