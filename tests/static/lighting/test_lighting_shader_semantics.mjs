import assert from "node:assert/strict";
import { AmbientLight } from "../../../src/lights/AmbientLight.ts";
import { DirectionalLight } from "../../../src/lights/DirectionalLight.ts";
import { LightProbe } from "../../../src/lights/LightProbe.ts";
import { ReflectionProbe } from "../../../src/lights/ReflectionProbe.ts";
import { evaluateLightContribution } from "../../../src/renderers/software/LightEvaluator.ts";
import { SH } from "../../../src/maths/SH.ts";
import { BlinnPhongStrategy } from "../../../src/shaders/software/BlinnPhongStrategy.ts";
import { PBRStrategy } from "../../../src/shaders/software/PBRStrategy.ts";
import { PBREvaluator } from "../../../src/shaders/software/PBREvaluator.ts";
import { PhongEvaluator } from "../../../src/shaders/software/PhongEvaluator.ts";
import { PBRMaterial } from "../../../src/materials/PBRMaterial.ts";
import { PhongMaterial } from "../../../src/materials/PhongMaterial.ts";
import { Material } from "../../../src/materials/Material.ts";
import { Texture } from "../../../src/core/Texture.ts";
import { Renderer } from "../../../src/renderers/Renderer.ts";
import { Rasterizer } from "../../../src/renderers/software/Rasterizer.ts";

function createContext(overrides = {}) {
	return {
		renderer: { shadowMaps: new Map() },
		cameraPos: { x: 0, y: 0, z: 1 },
		lights: [],
		worldMatrix: undefined,
		shAmbientCoeffs: null,
		enableShadows: false,
		enableSH: true,
		enableGamma: false,
		enableLighting: true,
		...overrides,
	};
}

function createBaseContext(enableSH = true) {
	return createContext({
		lights: [
			new AmbientLight({
				color: { r: 255, g: 255, b: 255 },
				intensity: 1.0,
			}),
		],
		shAmbientCoeffs: SH.empty(),
		enableSH,
	});
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

function testPBRFallbackAmbientUsesLinearBrightness() {
	const strategy = new PBRStrategy();
	const fallbackLinear = 0.05;
	const sh = SH.empty();
	const dc = (fallbackLinear * 255) / (Math.PI * 0.282095);
	sh[0] = { r: dc, g: dc, b: dc };

	const surface = {
		type: "pbr",
		albedo: { r: 255, g: 255, b: 255 },
		opacity: 1,
		normal: { x: 0, y: 0, z: 1 },
		emissive: { r: 0, g: 0, b: 0 },
		emissiveIntensity: 1,
		roughness: 1,
		metalness: 0,
		reflectance: 0.5,
		occlusion: 1,
		clearcoat: 0,
		clearcoatRoughness: 0,
	};

	const fallbackColor = strategy.calculate(
		{ x: 0, y: 0, z: 0 },
		{ x: 0, y: 0, z: 1 },
		{ x: 0, y: 0, z: 1 },
		surface,
		createContext({ enableSH: false })
	);

	const shColor = strategy.calculate(
		{ x: 0, y: 0, z: 0 },
		{ x: 0, y: 0, z: 1 },
		{ x: 0, y: 0, z: 1 },
		surface,
		createContext({ enableSH: true, shAmbientCoeffs: sh })
	);

	assert.ok(
		Math.abs(fallbackColor.r - shColor.r) < 0.5 &&
			Math.abs(fallbackColor.g - shColor.g) < 0.5 &&
			Math.abs(fallbackColor.b - shColor.b) < 0.5,
		"PBR fallback ambient should use the same linear brightness as an equivalent SH fallback"
	);
}

function testPBRSHAmbientSpecularTracksReflectionDirection() {
	const strategy = new PBRStrategy();
	const sh = SH.projectDirectionalLight(
		{ x: 1, y: 0, z: 0 },
		{ r: 255, g: 255, b: 255 }
	);
	const context = createContext({
		enableSH: true,
		shAmbientCoeffs: sh,
	});
	const surface = {
		type: "pbr",
		albedo: { r: 0, g: 0, b: 0 },
		opacity: 1,
		normal: { x: 0, y: 0, z: 1 },
		emissive: { r: 0, g: 0, b: 0 },
		emissiveIntensity: 1,
		roughness: 0.04,
		metalness: 0,
		reflectance: 1,
		specularFactor: 1,
		specularColor: { r: 255, g: 255, b: 255 },
		occlusion: 1,
		clearcoat: 0,
		clearcoatRoughness: 0,
	};

	const aligned = strategy.calculate(
		{ x: 0, y: 0, z: 0 },
		{ x: 0, y: 0, z: 1 },
		{ x: -1, y: 0, z: 0 },
		surface,
		context
	);

	const misaligned = strategy.calculate(
		{ x: 0, y: 0, z: 0 },
		{ x: 0, y: 0, z: 1 },
		{ x: 1, y: 0, z: 0 },
		surface,
		context
	);

	assert.ok(
		aligned.r > misaligned.r * 1.2,
		"SH ambient specular fallback should respond to the reflection direction, not only the surface normal"
	);
}

function testClearcoatAttenuatesAmbientSheen() {
	const strategy = new PBRStrategy();
	const context = createContext({
		enableSH: false,
		lights: [
			new AmbientLight({
				color: { r: 255, g: 255, b: 255 },
				intensity: 0.25,
			}),
		],
	});
	const baseSurface = {
		type: "pbr",
		albedo: { r: 0, g: 0, b: 0 },
		opacity: 1,
		normal: { x: 0, y: 0, z: 1 },
		emissive: { r: 0, g: 0, b: 0 },
		emissiveIntensity: 1,
		roughness: 1,
		metalness: 0,
		reflectance: 0,
		occlusion: 1,
		clearcoatRoughness: 0.04,
		sheenRoughness: 0.04,
	};

	const noSheen = strategy.calculate(
		{ x: 0, y: 0, z: 0 },
		{ x: 0, y: 0, z: 1 },
		{ x: 1, y: 0, z: 0 },
		{
			...baseSurface,
			clearcoat: 0,
			sheenColor: { r: 0, g: 0, b: 0 },
		},
		context
	);

	const openSheen = strategy.calculate(
		{ x: 0, y: 0, z: 0 },
		{ x: 0, y: 0, z: 1 },
		{ x: 1, y: 0, z: 0 },
		{
			...baseSurface,
			clearcoat: 0,
			sheenColor: { r: 255, g: 0, b: 0 },
		},
		context
	);

	const coatedNoSheen = strategy.calculate(
		{ x: 0, y: 0, z: 0 },
		{ x: 0, y: 0, z: 1 },
		{ x: 1, y: 0, z: 0 },
		{
			...baseSurface,
			clearcoat: 1,
			sheenColor: { r: 0, g: 0, b: 0 },
		},
		context
	);

	const coatedSheen = strategy.calculate(
		{ x: 0, y: 0, z: 0 },
		{ x: 0, y: 0, z: 1 },
		{ x: 1, y: 0, z: 0 },
		{
			...baseSurface,
			clearcoat: 1,
			sheenColor: { r: 255, g: 0, b: 0 },
		},
		context
	);

	const openBoost = openSheen.r - noSheen.r;
	const coatedBoost = coatedSheen.r - coatedNoSheen.r;

	assert.ok(
		coatedBoost < openBoost * 0.5,
		"Clearcoat should attenuate the ambient sheen contribution from the layer below it"
	);
}

function testTransmissionVolumeAttenuationColorsAmbientLight() {
	const strategy = new PBRStrategy();
	const context = createContext({
		enableSH: false,
		lights: [
			new AmbientLight({
				color: { r: 255, g: 255, b: 255 },
				intensity: 1.0,
			}),
		],
	});
	const color = strategy.calculate(
		{ x: 0, y: 0, z: 0 },
		{ x: 0, y: 0, z: 1 },
		{ x: 0, y: 0, z: 1 },
		{
			type: "pbr",
			albedo: { r: 255, g: 255, b: 255 },
			opacity: 1,
			normal: { x: 0, y: 0, z: 1 },
			emissive: { r: 0, g: 0, b: 0 },
			emissiveIntensity: 1,
			roughness: 1,
			metalness: 0,
			reflectance: 0,
			occlusion: 1,
			clearcoat: 0,
			clearcoatRoughness: 0,
			transmission: 1,
			thickness: 1,
			attenuationDistance: 1,
			attenuationColor: { r: 0, g: 0, b: 255 },
		},
		context
	);

	assert.ok(
		color.b > 1 && color.r < color.b * 0.25 && color.g < color.b * 0.25,
		"Transmission volume attenuation should tint ambient light with the attenuation color"
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

function testEvaluatorCompileSwapsMaterial() {
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

	const updatedMaterial = new PBRMaterial({
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

	evaluator.compile(updatedMaterial);
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

	const probe = new LightProbe({ sh });
	const contribution = evaluateLightContribution(probe, {
		position: { x: 0, y: 0, z: 0 },
	});

	assert.ok(
		contribution,
		"LightProbe should provide ambient fallback from SH DC term"
	);
	assert.equal(contribution.type, "irradiance");
	assert.ok(Math.abs((contribution.intensity ?? 0) - 1) < 1e-6);
	assert.ok(contribution.color.r > 0 || contribution.color.g > 0);
}

function testMaskShadowDepthWriteUsesAlphaCutoff() {
	const rasterizer = new Rasterizer({});
	const shadowTarget = {
		size: 8,
		depthBuffer: new Float32Array(64),
		transmissionBuffer: new Float32Array(64 * 3),
	};

	const clearShadowTarget = () => {
		shadowTarget.depthBuffer.fill(Infinity);
		shadowTarget.transmissionBuffer.fill(1);
	};

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

	clearShadowTarget();
	rasterizer.drawDepthTriangle(makeTri(), shadowTarget, maskMaterial);
	const wroteTransparent = shadowTarget.depthBuffer.some((d) =>
		Number.isFinite(d)
	);
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
	clearShadowTarget();
	rasterizer.drawDepthTriangle(makeTri(), shadowTarget, maskMaterial);
	const wroteOpaque = shadowTarget.depthBuffer.some((d) => Number.isFinite(d));
	assert.equal(
		wroteOpaque,
		true,
		"Opaque mask texels should write to shadow depth"
	);
}

function testTransmissionOnlyRespondsToBackLighting() {
	const strategy = new PBRStrategy();
	const surface = {
		type: "pbr",
		albedo: { r: 255, g: 255, b: 255 },
		opacity: 1,
		normal: { x: 0, y: 0, z: 1 },
		emissive: { r: 0, g: 0, b: 0 },
		emissiveIntensity: 1,
		roughness: 1,
		metalness: 0,
		reflectance: 0,
		specularFactor: 0,
		specularColor: { r: 255, g: 255, b: 255 },
		occlusion: 1,
		clearcoat: 0,
		clearcoatRoughness: 0,
		sheenColor: { r: 0, g: 0, b: 0 },
		sheenRoughness: 0,
		transmission: 1,
		thickness: 0,
		attenuationDistance: Infinity,
		attenuationColor: { r: 255, g: 255, b: 255 },
	};

	const noLight = strategy.calculate(
		{ x: 0, y: 0, z: 0 },
		{ x: 0, y: 0, z: 1 },
		{ x: 0, y: 0, z: 1 },
		surface,
		createContext({ enableSH: false, lights: [] })
	);

	const frontLit = strategy.calculate(
		{ x: 0, y: 0, z: 0 },
		{ x: 0, y: 0, z: 1 },
		{ x: 0, y: 0, z: 1 },
		surface,
		createContext({
			enableSH: false,
			lights: [
				new DirectionalLight({
					color: { r: 255, g: 255, b: 255 },
					intensity: 1,
					direction: { x: 0, y: 0, z: -1 },
				}),
			],
		})
	);

	const backLit = strategy.calculate(
		{ x: 0, y: 0, z: 0 },
		{ x: 0, y: 0, z: 1 },
		{ x: 0, y: 0, z: 1 },
		surface,
		createContext({
			enableSH: false,
			lights: [
				new DirectionalLight({
					color: { r: 255, g: 255, b: 255 },
					intensity: 1,
					direction: { x: 0, y: 0, z: 1 },
				}),
			],
		})
	);

	assert.ok(
		Math.abs(frontLit.r - noLight.r) < 0.5,
		"Front lighting should not create a fake transmission lobe"
	);
	assert.ok(
		backLit.r > frontLit.r + 50,
		"Back lighting should drive the transmission term"
	);
}

function testMetalnessSuppressesTransmission() {
	const strategy = new PBRStrategy();
	const context = createContext({
		enableSH: false,
		lights: [
			new DirectionalLight({
				color: { r: 255, g: 255, b: 255 },
				intensity: 1,
				direction: { x: 0, y: 0, z: 1 },
			}),
		],
	});
	const baseSurface = {
		type: "pbr",
		albedo: { r: 255, g: 255, b: 255 },
		opacity: 1,
		normal: { x: 0, y: 0, z: 1 },
		emissive: { r: 0, g: 0, b: 0 },
		emissiveIntensity: 1,
		roughness: 1,
		reflectance: 0,
		specularFactor: 0,
		specularColor: { r: 255, g: 255, b: 255 },
		occlusion: 1,
		clearcoat: 0,
		clearcoatRoughness: 0,
		sheenColor: { r: 0, g: 0, b: 0 },
		sheenRoughness: 0,
		transmission: 1,
		thickness: 0,
		attenuationDistance: Infinity,
		attenuationColor: { r: 255, g: 255, b: 255 },
	};

	const dielectric = strategy.calculate(
		{ x: 0, y: 0, z: 0 },
		{ x: 0, y: 0, z: 1 },
		{ x: 0, y: 0, z: 1 },
		{ ...baseSurface, metalness: 0 },
		context
	);
	const metal = strategy.calculate(
		{ x: 0, y: 0, z: 0 },
		{ x: 0, y: 0, z: 1 },
		{ x: 0, y: 0, z: 1 },
		{ ...baseSurface, metalness: 1 },
		context
	);

	assert.ok(
		dielectric.r > metal.r + 50,
		"Metalness should suppress the transmission energy budget"
	);
}

function testTransmissionVolumeAttenuationUsesLinear255Color() {
	const strategy = new PBRStrategy();
	const color = strategy.calculate(
		{ x: 0, y: 0, z: 0 },
		{ x: 0, y: 0, z: 1 },
		{ x: 0, y: 0, z: 1 },
		{
			type: "pbr",
			albedo: { r: 255, g: 255, b: 255 },
			opacity: 1,
			normal: { x: 0, y: 0, z: 1 },
			emissive: { r: 0, g: 0, b: 0 },
			emissiveIntensity: 1,
			roughness: 1,
			metalness: 0,
			reflectance: 0,
			specularFactor: 0,
			specularColor: { r: 255, g: 255, b: 255 },
			occlusion: 1,
			clearcoat: 0,
			clearcoatRoughness: 0,
			sheenColor: { r: 0, g: 0, b: 0 },
			sheenRoughness: 0,
			transmission: 1,
			thickness: 1,
			attenuationDistance: 1,
			attenuationColor: { r: 128, g: 128, b: 128 },
		},
		createContext({
			enableSH: false,
			lights: [
				new AmbientLight({
					color: { r: 255, g: 255, b: 255 },
					intensity: 1.0,
				}),
			],
		})
	);

	assert.ok(
		Math.abs(color.r - 128) < 1.5,
		`Expected mid-gray attenuation near 128, got ${color.r}`
	);
}

function testIridescenceChangesPBRSpecularHue() {
	const strategy = new PBRStrategy();
	const context = createContext({
		enableSH: false,
		lights: [
			new DirectionalLight({
				color: { r: 255, g: 255, b: 255 },
				intensity: 4,
				direction: { x: 0, y: 0, z: -1 },
			}),
		],
	});
	const baseSurface = {
		type: "pbr",
		albedo: { r: 0, g: 0, b: 0 },
		opacity: 1,
		normal: { x: 0, y: 0, z: 1 },
		emissive: { r: 0, g: 0, b: 0 },
		emissiveIntensity: 1,
		roughness: 0.35,
		metalness: 0,
		reflectance: 0.5,
		specularFactor: 1,
		specularColor: { r: 255, g: 255, b: 255 },
		occlusion: 1,
		clearcoat: 0,
		clearcoatRoughness: 0,
		sheenColor: { r: 0, g: 0, b: 0 },
		sheenRoughness: 0,
		transmission: 0,
		ior: 1.5,
		thickness: 0,
		attenuationDistance: Infinity,
		attenuationColor: { r: 255, g: 255, b: 255 },
	};

	const plain = strategy.calculate(
		{ x: 0, y: 0, z: 0 },
		{ x: 0, y: 0, z: 1 },
		{ x: 0, y: 0, z: 1 },
		{
			...baseSurface,
			iridescence: 0,
			iridescenceIor: 1.3,
			iridescenceThickness: 400,
		},
		context
	);
	const iridescent = strategy.calculate(
		{ x: 0, y: 0, z: 0 },
		{ x: 0, y: 0, z: 1 },
		{ x: 0, y: 0, z: 1 },
		{
			...baseSurface,
			iridescence: 1,
			iridescenceIor: 1.3,
			iridescenceThickness: 450,
		},
		context
	);

	const delta = Math.max(
		Math.abs(iridescent.r - plain.r),
		Math.abs(iridescent.g - plain.g),
		Math.abs(iridescent.b - plain.b)
	);
	const iridescentSpread =
		Math.max(iridescent.r, iridescent.g, iridescent.b) -
		Math.min(iridescent.r, iridescent.g, iridescent.b);

	assert.ok(delta > 0.5, "Iridescence should change the base specular response");
	assert.ok(
		iridescentSpread > 0.5,
		"Iridescence should introduce wavelength-dependent specular color"
	);
}

function testAnisotropyChangesPBRSpecularLobe() {
	const strategy = new PBRStrategy();
	const context = createContext({
		enableSH: false,
		lights: [
			new DirectionalLight({
				color: { r: 255, g: 255, b: 255 },
				intensity: 4,
				direction: { x: 0, y: 0, z: -1 },
			}),
		],
	});
	const baseSurface = {
		type: "pbr",
		albedo: { r: 0, g: 0, b: 0 },
		opacity: 1,
		normal: { x: 0, y: 0, z: 1 },
		emissive: { r: 0, g: 0, b: 0 },
		emissiveIntensity: 1,
		roughness: 0.35,
		metalness: 0,
		reflectance: 1,
		specularFactor: 1,
		specularColor: { r: 255, g: 255, b: 255 },
		occlusion: 1,
		clearcoat: 0,
		clearcoatRoughness: 0,
		sheenColor: { r: 0, g: 0, b: 0 },
		sheenRoughness: 0,
		transmission: 0,
		ior: 1.5,
		thickness: 0,
		attenuationDistance: Infinity,
		attenuationColor: { r: 255, g: 255, b: 255 },
		anisotropyTangent: { x: 1, y: 0, z: 0 },
		anisotropyBitangent: { x: 0, y: 1, z: 0 },
	};

	const isotropic = strategy.calculate(
		{ x: 0, y: 0, z: 0 },
		{ x: 0, y: 0, z: 1 },
		{ x: 0, y: 0, z: 1 },
		{
			...baseSurface,
			anisotropyStrength: 0,
		},
		context
	);
	const anisotropic = strategy.calculate(
		{ x: 0, y: 0, z: 0 },
		{ x: 0, y: 0, z: 1 },
		{ x: 0, y: 0, z: 1 },
		{
			...baseSurface,
			anisotropyStrength: 1,
		},
		context
	);

	assert.ok(
		Math.abs(anisotropic.r - isotropic.r) > 0.5,
		"Anisotropy should change the base PBR specular lobe"
	);
}

function testRendererUpdateSHPreservesHigherOrderProbeCoeffs() {
	const probe = new LightProbe({ sh: SH.empty() });
	probe.sh[0] = { r: 10, g: 0, b: 0 };
	probe.sh[15] = { r: 7, g: 3, b: 1 };

	const fakeRenderer = {
		params: { worldMatrix: undefined },
		scene: { getLights: () => [probe] },
		_shAmbientCoeffs: SH.empty(),
		_shCoeffs: SH.empty(),
	};

	Renderer.prototype.updateSH.call(fakeRenderer);

	assert.equal(fakeRenderer._shAmbientCoeffs[15].r, 7);
	assert.equal(fakeRenderer._shAmbientCoeffs[15].g, 3);
	assert.equal(fakeRenderer._shAmbientCoeffs[15].b, 1);
}

function testRendererUpdateSHIgnoresReflectionProbeSpecularMap() {
	const probe = new LightProbe({ sh: SH.empty() });
	probe.sh[0] = { r: 10, g: 0, b: 0 };
	probe.sh[15] = { r: 7, g: 3, b: 1 };
	const reflectionProbe = new ReflectionProbe({
		prefilteredMap: new Texture(new Float32Array([1, 1, 1, 1]), 1, 1, "HDR"),
	});

	const fakeRenderer = {
		params: { worldMatrix: undefined },
		scene: { getLights: () => [probe, reflectionProbe] },
		_shAmbientCoeffs: SH.empty(),
		_shCoeffs: SH.empty(),
	};

	Renderer.prototype.updateSH.call(fakeRenderer);

	assert.ok(Math.abs(fakeRenderer._shAmbientCoeffs[15].r - 7) < 1e-6);
	assert.ok(Math.abs(fakeRenderer._shAmbientCoeffs[15].g - 3) < 1e-6);
	assert.ok(Math.abs(fakeRenderer._shAmbientCoeffs[15].b - 1) < 1e-6);
}

function testRendererUpdateSHTreatsLocalizedProbeAsGlobalWithoutBackend() {
	const probe = new LightProbe({ sh: SH.empty() });
	probe.shape = "sphere";
	probe.sh[15] = { r: 7, g: 3, b: 1 };

	const fakeRenderer = {
		params: { worldMatrix: undefined },
		scene: { getLights: () => [probe] },
		_shAmbientCoeffs: SH.empty(),
		_shCoeffs: SH.empty(),
	};

	Renderer.prototype.updateSH.call(fakeRenderer);

	assert.equal(fakeRenderer._shAmbientCoeffs[15].r, 7);
	assert.equal(fakeRenderer._shAmbientCoeffs[15].g, 3);
	assert.equal(fakeRenderer._shAmbientCoeffs[15].b, 1);
}

function testRendererUpdateSHSkipsLocalizedProbeForGPUBackends() {
	const globalProbe = new LightProbe({ sh: SH.empty() });
	globalProbe.sh[15] = { r: 2, g: 1, b: 0.5 };

	const localizedProbe = new LightProbe({ sh: SH.empty() });
	localizedProbe.shape = "box";
	localizedProbe.sh[15] = { r: 7, g: 3, b: 1 };

	for (const backendType of ["webgl", "webgpu"]) {
		const fakeRenderer = {
			backend: { type: backendType },
			params: { worldMatrix: undefined },
			scene: { getLights: () => [globalProbe, localizedProbe] },
			_shAmbientCoeffs: SH.empty(),
			_shCoeffs: SH.empty(),
		};

		Renderer.prototype.updateSH.call(fakeRenderer);

		assert.equal(fakeRenderer._shAmbientCoeffs[15].r, 2);
		assert.equal(fakeRenderer._shAmbientCoeffs[15].g, 1);
		assert.equal(fakeRenderer._shAmbientCoeffs[15].b, 0.5);
	}
}

function run() {
	try {
		testSHAmbientGateForBlinnPhong();
		testSHAmbientGateForPBR();
		testPBRFallbackAmbientUsesLinearBrightness();
		testPBRSHAmbientSpecularTracksReflectionDirection();
		testClearcoatAttenuatesAmbientSheen();
		testTransmissionVolumeAttenuationColorsAmbientLight();
		testPBRNormalMapFallbackWithoutTangent();
		testEvaluatorCompileSwapsMaterial();
		testPhongEvaluatorDirectEvaluate();
		testLightProbeFallbackContributionFromDC();
		testMaskShadowDepthWriteUsesAlphaCutoff();
		testTransmissionOnlyRespondsToBackLighting();
		testMetalnessSuppressesTransmission();
		testTransmissionVolumeAttenuationUsesLinear255Color();
		testIridescenceChangesPBRSpecularHue();
		testAnisotropyChangesPBRSpecularLobe();
		testRendererUpdateSHPreservesHigherOrderProbeCoeffs();
		testRendererUpdateSHIgnoresReflectionProbeSpecularMap();
		testRendererUpdateSHTreatsLocalizedProbeAsGlobalWithoutBackend();
		testRendererUpdateSHSkipsLocalizedProbeForGPUBackends();
		console.log("✅ Shader semantics tests passed");
	} catch (error) {
		console.error("❌ Shader semantics test failed");
		console.error(error);
		process.exit(1);
	}
}

run();
