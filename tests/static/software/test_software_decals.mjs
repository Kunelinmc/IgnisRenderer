import assert from "node:assert/strict";
import { Decal } from "../../../src/decals/Decal.ts";
import { PBRMaterial } from "../../../src/materials/PBRMaterial.ts";
import { PhongMaterial } from "../../../src/materials/PhongMaterial.ts";
import { linearToSRGB, sRGBToLinear } from "../../../src/maths/Common.ts";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import { SoftwareDecalSurfaceModifier } from
	"../../../src/backends/software/SoftwareDecalSurfaceModifier.ts";

function createSurface() {
	return {
		type: "pbr",
		albedo: { r: 0, g: 0, b: 0 },
		opacity: 1,
		normal: { x: 0, y: 0, z: 1 },
		emissive: { r: 0, g: 0, b: 0 },
		emissiveIntensity: 1,
		roughness: 1,
		metalness: 0,
		reflectance: 0.5,
		specularFactor: 0,
		specularColor: { r: 0, g: 0, b: 0 },
		occlusion: 1,
		clearcoat: 0,
		clearcoatRoughness: 1,
		clearcoatNormal: { x: 0, y: 0, z: 1 },
		sheenColor: { r: 0, g: 0, b: 0 },
		sheenRoughness: 0,
		transmission: 0,
		ior: 1.5,
		iridescence: 0,
		iridescenceIor: 1.3,
		iridescenceThickness: 100,
		anisotropyStrength: 0,
		anisotropyTangent: { x: 1, y: 0, z: 0 },
		anisotropyBitangent: { x: 0, y: 1, z: 0 },
		thickness: 0,
		attenuationDistance: Infinity,
		attenuationColor: { r: 255, g: 255, b: 255 },
	};
}

function createInput(x = 0, y = 0, z = 0) {
	return {
		zCam: 1,
		world: { x, y, z },
		normal: { x: 0, y: 0, z: 1 },
		tangent: { x: 1, y: 0, z: 0, w: 1 },
		u: 0,
		v: 0,
		u2: 0,
		v2: 0,
		u3: 0,
		v3: 0,
		u4: 0,
		v4: 0,
	};
}

function createPhongSurface() {
	return {
		type: "phong",
		albedo: { r: 0, g: 0, b: 0 },
		opacity: 1,
		normal: { x: 0, y: 0, z: 1 },
		emissive: { r: 0, g: 0, b: 0 },
		emissiveIntensity: 1,
		ambient: { r: 0, g: 0, b: 0 },
		specular: { r: 0, g: 0, b: 0 },
		shininess: 32,
	};
}

function createPacket(material, options = {}) {
	const decal = new Decal({
		material,
		opacity: options.opacity ?? 1,
		edgeFade: options.edgeFade ?? 0,
	});
	return {
		id: "software-decal",
		decal,
		material,
		worldMatrix: Matrix4.identity(),
		inverseWorldMatrix: Matrix4.identity(),
		normalMatrix: Matrix4.identity(),
		worldBounds: { center: { x: 0, y: 0, z: 0 }, radius: 1 },
		receiverLayerMask: 1,
		priority: 0,
		opacity: decal.opacity,
		edgeFade: decal.edgeFade,
		channelBlendModes: decal.channelBlendModes,
		sceneOrder: 0,
	};
}

function testAppliesAllPbrScalarAndColorChannels() {
	const material = new PBRMaterial({
		albedo: { r: 255, g: 64, b: 32 },
		roughness: 0.25,
		metalness: 0.75,
		emissive: { r: 10, g: 20, b: 30 },
		specularFactor: 0.8,
		specularColor: { r: 200, g: 180, b: 160 },
		clearcoat: 0.7,
		clearcoatRoughness: 0.2,
		sheenColorFactor: { r: 40, g: 50, b: 60 },
		sheenRoughnessFactor: 0.3,
		transmissionFactor: 0.4,
		thicknessFactor: 2,
		iridescenceFactor: 0.6,
		iridescenceThicknessMaximum: 500,
		anisotropyStrength: 0.9,
	});
	const modifier = new SoftwareDecalSurfaceModifier();
	modifier.prepare([createPacket(material)]);
	const surface = createSurface();

	modifier.apply(createInput(), surface);

	assert.deepEqual(surface.albedo, material.albedo);
	assert.equal(surface.roughness, 0.25);
	assert.equal(surface.metalness, 0.75);
	assert.deepEqual(surface.emissive, material.emissive);
	assert.equal(surface.specularFactor, 0.8);
	assert.deepEqual(surface.specularColor, material.specularColor);
	assert.equal(surface.clearcoat, 0.7);
	assert.ok(Math.abs(surface.clearcoatRoughness - 0.2) < 1e-8);
	assert.deepEqual(surface.sheenColor, material.sheenColorFactor);
	assert.equal(surface.sheenRoughness, 0.3);
	assert.equal(surface.transmission, 0.4);
	assert.equal(surface.thickness, 2);
	assert.equal(surface.iridescence, 0.6);
	assert.equal(surface.iridescenceThickness, 500);
	assert.equal(surface.anisotropyStrength, 0.9);
}

function testCoverageAndProjectorBounds() {
	const material = new PBRMaterial({
		albedo: { r: 200, g: 100, b: 50 },
	});
	const modifier = new SoftwareDecalSurfaceModifier();
	modifier.prepare([createPacket(material, { opacity: 0.5 })]);
	const covered = createSurface();
	modifier.apply(createInput(), covered);
	assert.deepEqual(covered.albedo, { r: 100, g: 50, b: 25 });

	const outside = createSurface();
	modifier.apply(createInput(0.51, 0, 0), outside);
	assert.deepEqual(outside.albedo, { r: 0, g: 0, b: 0 });
}

function testNormalizesColorAcrossMaterialModels() {
	const phongSource = new PhongMaterial({
		diffuse: { r: 128, g: 64, b: 32 },
	});
	const pbrReceiver = createSurface();
	const phongModifier = new SoftwareDecalSurfaceModifier();
	phongModifier.prepare([createPacket(phongSource)]);
	phongModifier.apply(createInput(), pbrReceiver);
	assert.ok(
		Math.abs(pbrReceiver.albedo.r - sRGBToLinear(128 / 255) * 255) < 1e-8
	);
	assert.ok(
		Math.abs(pbrReceiver.albedo.g - sRGBToLinear(64 / 255) * 255) < 1e-8
	);
	assert.ok(
		Math.abs(pbrReceiver.albedo.b - sRGBToLinear(32 / 255) * 255) < 1e-8
	);

	const pbrSource = new PBRMaterial({
		albedo: { r: 64, g: 32, b: 16 },
	});
	const phongReceiver = createPhongSurface();
	const pbrModifier = new SoftwareDecalSurfaceModifier();
	pbrModifier.prepare([createPacket(pbrSource)]);
	pbrModifier.apply(createInput(), phongReceiver);
	assert.ok(
		Math.abs(phongReceiver.albedo.r - linearToSRGB(64 / 255) * 255) < 1e-8
	);
	assert.ok(
		Math.abs(phongReceiver.albedo.g - linearToSRGB(32 / 255) * 255) < 1e-8
	);
	assert.ok(
		Math.abs(phongReceiver.albedo.b - linearToSRGB(16 / 255) * 255) < 1e-8
	);
}

function run() {
	testAppliesAllPbrScalarAndColorChannels();
	testCoverageAndProjectorBounds();
	testNormalizesColorAcrossMaterialModels();
	console.log("Software decal tests passed");
}

run();
