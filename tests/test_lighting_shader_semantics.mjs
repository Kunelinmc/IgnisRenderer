import assert from "node:assert/strict";
import { AmbientLight } from "../src/lights/AmbientLight.ts";
import { LightProbe } from "../src/lights/LightProbe.ts";
import { SH } from "../src/maths/SH.ts";
import { BlinnPhongStrategy } from "../src/shaders/BlinnPhongStrategy.ts";
import { PBRStrategy } from "../src/shaders/PBRStrategy.ts";
import { PBREvaluator } from "../src/shaders/PBREvaluator.ts";
import { PhongEvaluator } from "../src/shaders/PhongEvaluator.ts";
import { PBRMaterial } from "../src/materials/PBRMaterial.ts";
import { PhongMaterial } from "../src/materials/PhongMaterial.ts";
import { Material } from "../src/materials/Material.ts";
import { Texture } from "../src/core/Texture.ts";
import { ShadowMap } from "../src/utils/ShadowMapping.ts";
import { Rasterizer } from "../src/core/Rasterizer.ts";

function createBaseContext(enableSH = true) {
	return {
		renderer: { shadowMaps: new Map() },
		cameraPos: { x: 0, y: 0, z: 1 },
		lights: [
			new AmbientLight({
				color: { r: 255, g: 255, b: 255 },
				intensity: 1.0,
			}),
		],
		worldMatrix: undefined,
		shAmbientCoeffs: SH.empty(),
		enableShadows: false,
		enableSH,
		enableGamma: false,
		enableLighting: true,
		gamma: 2.2,
	};
}

function testSHAmbientGateForBlinnPhong() {
	const strategy = new BlinnPhongStrategy();
	const context = createBaseContext(true);
	const surface = {
		type: "phong",
		albedo: { r: 255, g: 255, b: 255 },
		opacity: 1,
		normal: { x: 0, y: 1, z: 0 },
		emissive: { r: 0, g: 0, b: 0 },
		emissiveIntensity: 1,
		ambient: { r: 255, g: 255, b: 255 },
		specular: { r: 0, g: 0, b: 0 },
		shininess: 32,
	};

	const color = strategy.calculate(
		{ x: 0, y: 0, z: 0 },
		{ x: 0, y: 1, z: 0 },
		{ x: 0, y: 1, z: 0 },
		surface,
		context
	);

	assert.ok(
		color.r > 1 && color.g > 1 && color.b > 1,
		"Blinn-Phong should still receive ambient when SH is enabled but empty"
	);
}

function testSHAmbientGateForPBR() {
	const strategy = new PBRStrategy();
	const context = createBaseContext(true);
	const surface = {
		type: "pbr",
		albedo: { r: 255, g: 255, b: 255 },
		opacity: 1,
		normal: { x: 0, y: 1, z: 0 },
		emissive: { r: 0, g: 0, b: 0 },
		emissiveIntensity: 1,
		roughness: 1,
		metalness: 0,
		reflectance: 0.5,
		occlusion: 1,
		clearcoat: 0,
		clearcoatRoughness: 0,
	};

	const color = strategy.calculate(
		{ x: 0, y: 0, z: 0 },
		{ x: 0, y: 1, z: 0 },
		{ x: 0, y: 1, z: 0 },
		surface,
		context
	);

	assert.ok(
		color.r > 1 && color.g > 1 && color.b > 1,
		"PBR should still receive ambient when SH is enabled but empty"
	);
}

function testPBRNormalMapFallbackWithoutTangent() {
	const material = new PBRMaterial();
	material.normalMap = new Texture(
		new Uint8ClampedArray([255, 128, 128, 255]),
		1,
		1
	);

	const evaluator = new PBREvaluator(material);
	const face = {
		vertices: [],
		projected: [],
		center: { x: 0, y: 0, z: 0 },
		depthInfo: { min: 0, max: 0, avg: 0 },
	};
	const input = {
		zCam: 1,
		world: { x: 0, y: 0, z: 0 },
		normal: { x: 0, y: 0, z: 1 },
		tangent: { x: 0, y: 0, z: 0, w: 1 },
		u: 0,
		v: 0,
	};

	const surface = evaluator.evaluate(input, face);
	assert.ok(surface, "PBR evaluator should return surface properties");
	assert.ok(
		surface.normal.z > 0.9,
		"Missing tangents should fall back to geometric normal when normalMap is present"
	);
}

function testEvaluatorMaterialApiCompatibility() {
	const evaluator = new PBREvaluator(
		new PBRMaterial({
			albedo: { r: 255, g: 0, b: 0 },
		})
	);
	const face = {
		vertices: [],
		projected: [],
		center: { x: 0, y: 0, z: 0 },
		depthInfo: { min: 0, max: 0, avg: 0 },
	};
	const input = {
		zCam: 1,
		world: { x: 0, y: 0, z: 0 },
		normal: { x: 0, y: 0, z: 1 },
		tangent: { x: 1, y: 0, z: 0, w: 1 },
		u: 0,
		v: 0,
	};

	const legacyMaterial = new PBRMaterial({
		albedo: { r: 0, g: 255, b: 0 },
	});
	const compiledMaterial = new PBRMaterial({
		albedo: { r: 0, g: 0, b: 255 },
	});

	let surface = evaluator.evaluate(input, face);
	assert.ok(surface);
	assert.equal(surface.albedo.r, 255);
	assert.equal(surface.albedo.g, 0);
	assert.equal(surface.albedo.b, 0);

	evaluator.setMaterial(legacyMaterial);
	surface = evaluator.evaluate(input, face);
	assert.ok(surface);
	assert.equal(surface.albedo.r, 0);
	assert.equal(surface.albedo.g, 255);
	assert.equal(surface.albedo.b, 0);

	evaluator.compile(compiledMaterial);
	surface = evaluator.evaluate(input, face);
	assert.ok(surface);
	assert.equal(surface.albedo.r, 0);
	assert.equal(surface.albedo.g, 0);
	assert.equal(surface.albedo.b, 255);
}

function testPhongEvaluatorDirectEvaluate() {
	const evaluator = new PhongEvaluator(
		new PhongMaterial({
			diffuse: { r: 32, g: 64, b: 96 },
		})
	);
	const face = {
		vertices: [],
		projected: [],
		center: { x: 0, y: 0, z: 0 },
		depthInfo: { min: 0, max: 0, avg: 0 },
	};
	const input = {
		zCam: 1,
		world: { x: 0, y: 0, z: 0 },
		normal: { x: 0, y: 0, z: 1 },
		tangent: { x: 1, y: 0, z: 0, w: 1 },
		u: 0,
		v: 0,
	};

	const surface = evaluator.evaluate(input, face);
	assert.ok(surface, "Phong evaluator should return surface properties");
	assert.equal(surface.albedo.r, 32);
	assert.equal(surface.albedo.g, 64);
	assert.equal(surface.albedo.b, 96);
}

function testLightProbeFallbackContributionFromDC() {
	const sh = SH.empty();
	sh[0] = { r: 120, g: 60, b: 0 };

	const probe = new LightProbe(sh, 0.75);
	const contribution = probe.computeContribution({ x: 0, y: 0, z: 0 });

	assert.ok(
		contribution,
		"LightProbe should provide ambient fallback from SH DC term"
	);
	assert.equal(contribution.type, "ambient");
	assert.ok(Math.abs((contribution.intensity ?? 0) - 0.75) < 1e-6);
	assert.ok(contribution.color.r > 0 || contribution.color.g > 0);
}

function testMaskShadowDepthWriteUsesAlphaCutoff() {
	const rasterizer = new Rasterizer({});
	const shadowMap = new ShadowMap(8);

	const makeTri = () => [
		{
			x: 1,
			y: 1,
			z: 0.2,
			w: 1,
			u: 0,
			v: 0,
			world: { x: 0, y: 0, z: 0 },
		},
		{
			x: 6,
			y: 1,
			z: 0.2,
			w: 1,
			u: 0,
			v: 0,
			world: { x: 0, y: 0, z: 0 },
		},
		{
			x: 1,
			y: 6,
			z: 0.2,
			w: 1,
			u: 0,
			v: 0,
			world: { x: 0, y: 0, z: 0 },
		},
	];

	const maskMaterial = new Material({
		alphaMode: "MASK",
		alphaCutoff: 0.5,
		opacity: 1,
		map: new Texture(new Uint8ClampedArray([255, 255, 255, 0]), 1, 1),
	});

	shadowMap.clear();
	rasterizer.drawDepthTriangle(makeTri(), shadowMap, maskMaterial);
	const wroteTransparent = shadowMap.buffer.some((d) => Number.isFinite(d));
	assert.equal(
		wroteTransparent,
		false,
		"Fully transparent mask texels should not write to shadow depth"
	);

	maskMaterial.map = new Texture(
		new Uint8ClampedArray([255, 255, 255, 255]),
		1,
		1
	);
	shadowMap.clear();
	rasterizer.drawDepthTriangle(makeTri(), shadowMap, maskMaterial);
	const wroteOpaque = shadowMap.buffer.some((d) => Number.isFinite(d));
	assert.equal(
		wroteOpaque,
		true,
		"Opaque mask texels should write to shadow depth"
	);
}

function run() {
	try {
		testSHAmbientGateForBlinnPhong();
		testSHAmbientGateForPBR();
		testPBRNormalMapFallbackWithoutTangent();
		testEvaluatorMaterialApiCompatibility();
		testPhongEvaluatorDirectEvaluate();
		testLightProbeFallbackContributionFromDC();
		testMaskShadowDepthWriteUsesAlphaCutoff();
		console.log("✅ Shader semantics tests passed");
	} catch (error) {
		console.error("❌ Shader semantics test failed");
		console.error(error);
		process.exit(1);
	}
}

run();
