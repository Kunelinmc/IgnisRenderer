import assert from "node:assert/strict";
import { GLTFLoader } from "../src/loaders/GLTFLoader.ts";
import { Texture } from "../src/core/Texture.ts";
import { PBRMaterial } from "../src/materials/PBRMaterial.ts";
import { PBRStrategy } from "../src/shaders/PBRStrategy.ts";

function approx(actual, expected, epsilon = 1e-6) {
	assert.ok(
		Math.abs(actual - expected) <= epsilon,
		`Expected ${expected}, got ${actual}`
	);
}

function reflectanceFromIor(ior) {
	const f0 = Math.pow((ior - 1) / (ior + 1), 2);
	return Math.sqrt(f0 / 0.16);
}

function testUnlitExtensionParsing() {
	const loader = new GLTFLoader();
	const baseTex = new Texture(new Uint8ClampedArray([255, 255, 255, 255]), 1, 1);

	const materials = loader.parseMaterials(
		{
			materials: [
				{
					pbrMetallicRoughness: {
						baseColorFactor: [0.2, 0.4, 0.6, 0.8],
						baseColorTexture: { index: 0 },
					},
					doubleSided: true,
					alphaMode: "MASK",
					alphaCutoff: 0.25,
					extensions: { KHR_materials_unlit: {} },
				},
			],
		},
		[baseTex]
	);

	assert.equal(materials.length, 1);
	const mat = materials[0];
	assert.equal(mat.type, "Unlit");
	assert.equal(mat.shading, "Unlit");
	approx(mat.diffuse.r, 51);
	approx(mat.diffuse.g, 102);
	approx(mat.diffuse.b, 153);
	approx(mat.opacity, 0.8);
	assert.equal(mat.doubleSided, true);
	assert.equal(mat.alphaMode, "MASK");
	approx(mat.alphaCutoff, 0.25);
	assert.equal(mat.map, baseTex);
}

function testIorExtensionUpdatesReflectance() {
	const loader = new GLTFLoader();
	const ior = 2.0;
	const materials = loader.parseMaterials({
		materials: [
			{
				pbrMetallicRoughness: {},
				extensions: { KHR_materials_ior: { ior } },
			},
		],
	});

	assert.equal(materials.length, 1);
	const mat = materials[0];
	assert.equal(mat.type, "PBR");
	approx(mat.ior, ior);
	approx(mat.reflectance, reflectanceFromIor(ior));
}

function testPBRMaterialIorSetterSyncsReflectance() {
	const mat = new PBRMaterial({ reflectance: 0.1 });
	const ior = 1.8;
	mat.ior = ior;

	approx(mat.ior, ior);
	approx(mat.reflectance, reflectanceFromIor(ior));
}

function testSpecularExtensionParsing() {
	const loader = new GLTFLoader();
	const specTex = new Texture(new Uint8ClampedArray([0, 0, 0, 128]), 1, 1);
	const specColorTex = new Texture(new Uint8ClampedArray([255, 255, 255, 255]), 1, 1);

	const materials = loader.parseMaterials(
		{
			materials: [
				{
					pbrMetallicRoughness: {},
					extensions: {
						KHR_materials_specular: {
							specularFactor: 0.7,
							specularColorFactor: [0.5, 0.25, 1.0],
							specularTexture: { index: 0 },
							specularColorTexture: { index: 1 },
						},
					},
				},
			],
		},
		[specTex, specColorTex]
	);

	assert.equal(materials.length, 1);
	const mat = materials[0];
	assert.equal(mat.type, "PBR");
	approx(mat.specularFactor, 0.7);
	approx(mat.specularColor.r, 127.5);
	approx(mat.specularColor.g, 63.75);
	approx(mat.specularColor.b, 255);
	assert.equal(mat.specularMap, specTex);
	assert.equal(mat.specularColorMap, specColorTex);
}

function testSpecularColorUsesLinearSemanticsInPBRStrategy() {
	const strategy = new PBRStrategy();
	const context = {
		renderer: { shadowMaps: new Map() },
		cameraPos: { x: 0, y: 0, z: 1 },
		lights: [
			{
				computeContribution: () => ({
					type: "ambient",
					color: { r: 255, g: 255, b: 255 },
					intensity: 10.0,
				}),
			},
		],
		worldMatrix: undefined,
		shAmbientCoeffs: null,
		enableShadows: false,
		enableSH: false,
		enableGamma: false,
		enableLighting: true,
		gamma: 2.2,
	};

	const baseSurface = {
		type: "pbr",
		albedo: { r: 0, g: 0, b: 0 },
		opacity: 1,
		normal: { x: 0, y: 0, z: 1 },
		emissive: { r: 0, g: 0, b: 0 },
		emissiveIntensity: 1,
		roughness: 1,
		metalness: 0,
		reflectance: 0.5,
		specularFactor: 1,
		occlusion: 1,
		clearcoat: 0,
		clearcoatRoughness: 0,
	};

	const full = strategy.calculate(
		{ x: 0, y: 0, z: 0 },
		{ x: 0, y: 0, z: 1 },
		{ x: 0, y: 0, z: 1 },
		{
			...baseSurface,
			specularColor: { r: 255, g: 255, b: 255 },
		},
		context
	);

	const half = strategy.calculate(
		{ x: 0, y: 0, z: 0 },
		{ x: 0, y: 0, z: 1 },
		{ x: 0, y: 0, z: 1 },
		{
			...baseSurface,
			specularColor: { r: 128, g: 128, b: 128 },
		},
		context
	);

	assert.ok(full.r > half.r, "Expected full specularColor to be brighter than half");
	const ratio = half.r / full.r;
	assert.ok(
		ratio > 0.3 && ratio < 0.6,
		`Expected half/full ratio in [0.3, 0.6], got ${ratio}`
	);
}

function run() {
	try {
		console.log("Starting glTF material extensions tests...");
		testUnlitExtensionParsing();
		testIorExtensionUpdatesReflectance();
		testPBRMaterialIorSetterSyncsReflectance();
		testSpecularExtensionParsing();
		testSpecularColorUsesLinearSemanticsInPBRStrategy();
		console.log("✅ glTF material extensions tests passed");
	} catch (error) {
		console.error("❌ glTF material extensions test failed");
		console.error(error);
		process.exit(1);
	}
}

run();
