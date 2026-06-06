import assert from "node:assert/strict";
import { GLTFLoader } from "../../../src/loaders/GLTFLoader.ts";
import { Texture } from "../../../src/core/Texture.ts";
import { AmbientLight } from "../../../src/lights/AmbientLight.ts";
import { PBRMaterial, UVChannel } from "../../../src/materials/PBRMaterial.ts";
import { PBRStrategy } from "../../../src/shaders/software/PBRStrategy.ts";
import { PBREvaluator } from "../../../src/shaders/software/PBREvaluator.ts";

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
	const baseTex = new Texture(
		new Uint8ClampedArray([255, 255, 255, 255]),
		1,
		1
	);

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
	assert.equal(mat.map.data, baseTex.data);
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
	const specColorTex = new Texture(
		new Uint8ClampedArray([255, 255, 255, 255]),
		1,
		1
	);

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
	assert.ok(mat.specularMap !== specTex);
	assert.equal(mat.specularMap.data, specTex.data);
	assert.equal(mat.specularColorMap, specColorTex);
	assert.equal(mat.specularColorMap.data, specColorTex.data);
}

function testIridescenceExtensionParsingAndEvaluation() {
	const loader = new GLTFLoader();
	const iridescenceTex = new Texture(
		new Uint8ClampedArray([128, 0, 0, 255]),
		1,
		1
	);
	const thicknessTex = new Texture(
		new Uint8ClampedArray([0, 64, 0, 255]),
		1,
		1
	);

	const [mat] = loader.parseMaterials(
		{
			materials: [
				{
					pbrMetallicRoughness: {},
					extensions: {
						KHR_materials_iridescence: {
							iridescenceFactor: 0.8,
							iridescenceTexture: { index: 0, texCoord: 2 },
							iridescenceIor: 1.45,
							iridescenceThicknessMinimum: 200,
							iridescenceThicknessMaximum: 600,
							iridescenceThicknessTexture: { index: 1, texCoord: 3 },
						},
					},
				},
			],
		},
		[iridescenceTex, thicknessTex]
	);

	assert.equal(mat.type, "PBR");
	approx(mat.iridescenceFactor, 0.8);
	approx(mat.iridescenceIor, 1.45);
	approx(mat.iridescenceThicknessMinimum, 200);
	approx(mat.iridescenceThicknessMaximum, 600);
	assert.ok(mat.iridescenceMap !== iridescenceTex);
	assert.equal(mat.iridescenceMap.data, iridescenceTex.data);
	assert.equal(mat.iridescenceMap.colorSpace, "Linear");
	assert.equal(mat.iridescenceMapUV, UVChannel.UV2);
	assert.ok(mat.iridescenceThicknessMap !== thicknessTex);
	assert.equal(mat.iridescenceThicknessMap.data, thicknessTex.data);
	assert.equal(mat.iridescenceThicknessMap.colorSpace, "Linear");
	assert.equal(mat.iridescenceThicknessMapUV, UVChannel.UV3);

	const evaluator = new PBREvaluator(mat);
	const surface = evaluator.evaluate(
		{
			zCam: 1,
			world: { x: 0, y: 0, z: 0 },
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
		},
		{
			vertices: [],
			projected: [],
			center: { x: 0, y: 0, z: 0 },
			depthInfo: { min: 0, max: 0, avg: 0 },
		}
	);
	assert.ok(surface);
	approx(surface.iridescence, 0.8 * (128 / 255));
	approx(surface.iridescenceIor, 1.45);
	approx(surface.iridescenceThickness, 200 + (600 - 200) * (64 / 255));
}

function testAnisotropyExtensionParsingAndEvaluation() {
	const loader = new GLTFLoader();
	const anisotropyTex = new Texture(
		new Uint8ClampedArray([255, 128, 64, 255]),
		1,
		1
	);

	const [mat] = loader.parseMaterials(
		{
			materials: [
				{
					pbrMetallicRoughness: {},
					extensions: {
						KHR_materials_anisotropy: {
							anisotropyStrength: 0.75,
							anisotropyRotation: Math.PI / 2,
							anisotropyTexture: { index: 0, texCoord: 2 },
						},
					},
				},
			],
		},
		[anisotropyTex]
	);

	assert.equal(mat.type, "PBR");
	approx(mat.anisotropyStrength, 0.75);
	approx(mat.anisotropyRotation, Math.PI / 2);
	assert.ok(mat.anisotropyMap !== anisotropyTex);
	assert.equal(mat.anisotropyMap.data, anisotropyTex.data);
	assert.equal(mat.anisotropyMap.colorSpace, "Linear");
	assert.equal(mat.anisotropyMapUV, UVChannel.UV2);

	const evaluator = new PBREvaluator(mat);
	const surface = evaluator.evaluate(
		{
			zCam: 1,
			world: { x: 0, y: 0, z: 0 },
			normal: { x: 0, y: 0, z: 1 },
			tangent: { x: 1, y: 0, z: 0, w: 1 },
			u: 0,
			v: 0,
			u2: 0,
			v2: 0,
		},
		{
			vertices: [],
			projected: [],
			center: { x: 0, y: 0, z: 0 },
			depthInfo: { min: 0, max: 0, avg: 0 },
		}
	);
	assert.ok(surface);
	approx(surface.anisotropyStrength, 0.75 * (64 / 255));
	assert.ok(
		Math.abs(surface.anisotropyTangent.x) < 0.01 &&
			surface.anisotropyTangent.y > 0.99,
		"KHR_materials_anisotropy direction should rotate in tangent space"
	);
}

function testSharedBaseColorTextureDoesNotCloneWithoutOverrides() {
	const loader = new GLTFLoader();
	const baseTex = new Texture(
		new Uint8ClampedArray([255, 255, 255, 255]),
		1,
		1
	);

	const materials = loader.parseMaterials(
		{
			materials: [
				{
					pbrMetallicRoughness: {
						baseColorTexture: { index: 0 },
					},
				},
				{
					pbrMetallicRoughness: {
						baseColorTexture: { index: 0 },
					},
				},
			],
		},
		[baseTex]
	);

	assert.equal(materials.length, 2);
	assert.equal(materials[0].map, baseTex);
	assert.equal(materials[1].map, baseTex);
}

function testTextureTransformStillCreatesDistinctTextureInstance() {
	const loader = new GLTFLoader();
	const baseTex = new Texture(
		new Uint8ClampedArray([255, 255, 255, 255]),
		1,
		1
	);

	const [material] = loader.parseMaterials(
		{
			materials: [
				{
					pbrMetallicRoughness: {
						baseColorTexture: {
							index: 0,
							extensions: {
								KHR_texture_transform: {
									offset: [0.25, 0.5],
									scale: [2, 3],
									rotation: 0.2,
								},
							},
						},
					},
				},
			],
		},
		[baseTex]
	);

	assert.ok(material.map !== baseTex);
	approx(material.map.offset.x, 0.25);
	approx(material.map.offset.y, 0.5);
	approx(material.map.repeat.x, 2);
	approx(material.map.repeat.y, 3);
	approx(material.map.rotation, 0.2);
}

function testSpecularColorUsesLinearSemanticsInPBRStrategy() {
	const strategy = new PBRStrategy();
	const context = {
		renderer: { shadowMaps: new Map() },
		cameraPos: { x: 0, y: 0, z: 1 },
		lights: [
			new AmbientLight({
				color: { r: 255, g: 255, b: 255 },
				intensity: 10.0,
			}),
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

	assert.ok(
		full.r > half.r,
		"Expected full specularColor to be brighter than half"
	);
	const ratio = half.r / full.r;
	assert.ok(
		ratio > 0.3 && ratio < 0.6,
		`Expected half/full ratio in [0.3, 0.6], got ${ratio}`
	);
}

function testLinearFactorsStayLinearAcrossLoaderAndEvaluator() {
	const loader = new GLTFLoader();
	const [mat] = loader.parseMaterials({
		materials: [
			{
				pbrMetallicRoughness: {
					baseColorFactor: [0.5, 0.25, 1.0, 1.0],
				},
				emissiveFactor: [0.5, 0.25, 1.0],
				extensions: {
					KHR_materials_sheen: {
						sheenColorFactor: [0.5, 0.25, 1.0],
					},
					KHR_materials_volume: {
						attenuationColor: [0.5, 0.25, 1.0],
						attenuationDistance: 2.0,
					},
				},
			},
		],
	});

	const evaluator = new PBREvaluator(mat);
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
		u2: 0,
		v2: 0,
		u3: 0,
		v3: 0,
		u4: 0,
		v4: 0,
	};

	const surface = evaluator.evaluate(input, face);
	assert.ok(surface);
	approx(surface.albedo.r, 127.5);
	approx(surface.albedo.g, 63.75);
	approx(surface.albedo.b, 255);
	approx(surface.emissive.r, 127.5);
	approx(surface.emissive.g, 63.75);
	approx(surface.emissive.b, 255);
	approx(surface.sheenColor.r, 127.5);
	approx(surface.sheenColor.g, 63.75);
	approx(surface.sheenColor.b, 255);
	approx(surface.attenuationColor.r, 127.5);
	approx(surface.attenuationColor.g, 63.75);
	approx(surface.attenuationColor.b, 255);
}

function testGLTFMaterialTexturesUseExpectedColorSpaces() {
	const loader = new GLTFLoader();
	const makeTexture = () =>
		new Texture(new Uint8ClampedArray([255, 255, 255, 255]), 1, 1);

	const textures = Array.from({ length: 13 }, makeTexture);
	const [mat] = loader.parseMaterials(
		{
			materials: [
				{
					pbrMetallicRoughness: {
						baseColorTexture: { index: 0 },
						metallicRoughnessTexture: { index: 1 },
					},
					normalTexture: { index: 2 },
					emissiveTexture: { index: 3 },
					occlusionTexture: { index: 4 },
					extensions: {
						KHR_materials_specular: {
							specularTexture: { index: 5 },
							specularColorTexture: { index: 6 },
						},
						KHR_materials_sheen: {
							sheenColorTexture: { index: 7 },
							sheenRoughnessTexture: { index: 8 },
						},
						KHR_materials_transmission: {
							transmissionTexture: { index: 9 },
						},
						KHR_materials_iridescence: {
							iridescenceTexture: { index: 10 },
							iridescenceThicknessTexture: { index: 11 },
						},
						KHR_materials_anisotropy: {
							anisotropyTexture: { index: 12 },
						},
					},
				},
			],
		},
		textures
	);

	assert.equal(mat.map?.colorSpace, "sRGB");
	assert.equal(mat.metallicRoughnessMap?.colorSpace, "Linear");
	assert.equal(mat.normalMap?.colorSpace, "Linear");
	assert.equal(mat.emissiveMap?.colorSpace, "sRGB");
	assert.equal(mat.occlusionMap?.colorSpace, "Linear");
	assert.equal(mat.specularMap?.colorSpace, "Linear");
	assert.equal(mat.specularColorMap?.colorSpace, "sRGB");
	assert.equal(mat.sheenColorMap?.colorSpace, "sRGB");
	assert.equal(mat.sheenRoughnessMap?.colorSpace, "Linear");
	assert.equal(mat.transmissionMap?.colorSpace, "Linear");
	assert.equal(mat.iridescenceMap?.colorSpace, "Linear");
	assert.equal(mat.iridescenceThicknessMap?.colorSpace, "Linear");
	assert.equal(mat.anisotropyMap?.colorSpace, "Linear");
}

function testTexCoordAboveOnePreservesUVSet() {
	const loader = new GLTFLoader();
	const texture = new Texture(
		new Uint8ClampedArray([255, 255, 255, 255]),
		1,
		1
	);

	const [baseTexCoordMaterial] = loader.parseMaterials(
		{
			materials: [
				{
					pbrMetallicRoughness: {
						baseColorTexture: {
							index: 0,
							texCoord: 2,
						},
					},
				},
			],
		},
		[texture]
	);
	assert.equal(baseTexCoordMaterial.albedoMapUV, UVChannel.UV2);

	const [khrTransformMaterial] = loader.parseMaterials(
		{
			materials: [
				{
					pbrMetallicRoughness: {
						baseColorTexture: {
							index: 0,
							texCoord: 0,
							extensions: {
								KHR_texture_transform: {
									texCoord: 3,
								},
							},
						},
					},
				},
			],
		},
		[texture]
	);
	assert.equal(khrTransformMaterial.albedoMapUV, UVChannel.UV3);
}

function run() {
	try {
		console.log("Starting glTF material extensions tests...");
		testUnlitExtensionParsing();
		testIorExtensionUpdatesReflectance();
		testPBRMaterialIorSetterSyncsReflectance();
		testSpecularExtensionParsing();
		testIridescenceExtensionParsingAndEvaluation();
		testAnisotropyExtensionParsingAndEvaluation();
		testSharedBaseColorTextureDoesNotCloneWithoutOverrides();
		testTextureTransformStillCreatesDistinctTextureInstance();
		testSpecularColorUsesLinearSemanticsInPBRStrategy();
		testLinearFactorsStayLinearAcrossLoaderAndEvaluator();
		testGLTFMaterialTexturesUseExpectedColorSpaces();
		testTexCoordAboveOnePreservesUVSet();
		console.log("✅ glTF material extensions tests passed");
	} catch (error) {
		console.error("❌ glTF material extensions test failed");
		console.error(error);
		process.exit(1);
	}
}

run();
