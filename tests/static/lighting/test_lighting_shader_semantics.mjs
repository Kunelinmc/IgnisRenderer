import assert from "node:assert/strict";
import { AmbientLight } from "../../../src/lights/AmbientLight.ts";
import { DirectionalLight } from "../../../src/lights/DirectionalLight.ts";
import { LightProbe } from "../../../src/lights/LightProbe.ts";
import { ReflectionProbe } from "../../../src/lights/ReflectionProbe.ts";
import { evaluateLightContribution } from "../../../src/shaders/software/LightEvaluator.ts";
import { SH } from "../../../src/maths/SH.ts";
import { BlinnPhongStrategy } from "../../../src/shaders/software/BlinnPhongStrategy.ts";
import { LitShader } from "../../../src/shaders/software/LitShader.ts";
import { PBRStrategy } from "../../../src/shaders/software/PBRStrategy.ts";
import { PBREvaluator } from "../../../src/shaders/software/PBREvaluator.ts";
import { PhongEvaluator } from "../../../src/shaders/software/PhongEvaluator.ts";
import { PBRMaterial } from "../../../src/materials/PBRMaterial.ts";
import { PhongMaterial } from "../../../src/materials/PhongMaterial.ts";
import { Material } from "../../../src/materials/Material.ts";
import { Texture } from "../../../src/core/Texture.ts";
import { Renderer } from "../../../src/rendering/Renderer.ts";
import { Rasterizer } from "../../../src/backends/software/Rasterizer.ts";

function createContext(overrides = {}) {
	return {
		renderer: { shadowMaps: new Map() },
		cameraPos: { x: 0, y: 0, z: 1 },
		lights: [],
		shAmbientCoeffs: null,
		enableShadows: false,
		enableSH: true,
		enableGamma: false,
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

function createConstantWhiteSH(linearRadiance = 1) {
	const sh = SH.empty();
	const dc = (linearRadiance * 255) / 0.282095;
	sh[0] = { r: dc, g: dc, b: dc };
	return sh;
}

function testDoubleSidedNormalUsesRasterizerFacing() {
	let observedNormal = null;
	const shader = new LitShader(
		{
			calculate(_world, normal) {
				observedNormal = { ...normal };
				return { r: 0, g: 0, b: 0 };
			},
		},
		{
			evaluate(input) {
				return {
					opacity: 1,
					normal: { ...input.normal },
				};
			},
		}
	);
	const face = {
		material: new Material({ doubleSided: true }),
		center: { x: 0, y: 0, z: 0 },
		depthInfo: { min: 0, max: 0, avg: 0 },
	};
	shader.initialize(
		{
			...face,
			frontFacing: true,
		},
		createContext({ cameraPos: { x: 0, y: 0, z: 1 } })
	);
	shader.shade({
		world: { x: 0, y: 0, z: 0 },
		normal: { x: 0, y: 0, z: -1 },
	});

	assert.deepEqual(observedNormal, { x: 0, y: 0, z: -1 });

	shader.initialize(
		{ ...face, frontFacing: false },
		createContext({ cameraPos: { x: 0, y: 0, z: 1 } })
	);
	shader.shade({
		world: { x: 0, y: 0, z: 0 },
		normal: { x: 0, y: 0, z: -1 },
	});

	assert.ok(Math.abs(observedNormal.x) <= 1e-6);
	assert.ok(Math.abs(observedNormal.y) <= 1e-6);
	assert.equal(observedNormal.z, 1);
	const surfaceNormal = shader.getSurfaceNormal();
	assert.ok(surfaceNormal);
	assert.ok(Math.abs(surfaceNormal.x) <= 1e-6);
	assert.ok(Math.abs(surfaceNormal.y) <= 1e-6);
	assert.equal(surfaceNormal.z, 1);
}

function testDoubleSidedNormalMapUsesOrientedTBN() {
	const material = new PBRMaterial({ doubleSided: true });
	material.normalMap = new Texture({
		data: new Uint8ClampedArray([255, 128, 128, 255]),
		width: 1,
		height: 1,
	});
	let observedNormal = null;
	const shader = new LitShader(
		{
			calculate(_world, normal) {
				observedNormal = { ...normal };
				return { r: 0, g: 0, b: 0 };
			},
		},
		new PBREvaluator(material)
	);
	shader.initialize(
		{
			material,
			frontFacing: false,
			center: { x: 0, y: 0, z: 0 },
			depthInfo: { min: 0, max: 0, avg: 0 },
		},
		createContext({ cameraPos: { x: 0, y: 0, z: 1 } })
	);
	shader.shade({
		zCam: 1,
		world: { x: 0, y: 0, z: 0 },
		normal: { x: 0, y: 0, z: -1 },
		tangent: { x: 1, y: 0, z: 0, w: 1 },
		u: 0,
		v: 0,
		u2: 0,
		v2: 0,
		u3: 0,
		v3: 0,
		u4: 0,
		v4: 0,
	});

	assert.ok(observedNormal);
	assert.ok(observedNormal.x > 0.9);
	assert.ok(Math.abs(observedNormal.z) < 0.1);
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
		color.r > 0 && color.g > 0 && color.b > 0,
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
		color.r > 0 && color.g > 0 && color.b > 0,
		"PBR should still receive ambient when SH is enabled but empty"
	);
}

function testPBRAmbientIrradianceIsViewIndependent() {
	const strategy = new PBRStrategy();
	const surface = {
		type: "pbr",
		albedo: { r: 94, g: 141, b: 206 },
		opacity: 1,
		normal: { x: 0, y: 0, z: 1 },
		emissive: { r: 0, g: 0, b: 0 },
		emissiveIntensity: 1,
		roughness: 0.6,
		metalness: 0,
		reflectance: 0.5,
		occlusion: 1,
		clearcoat: 0,
		clearcoatRoughness: 0,
	};
	const context = createBaseContext(false);

	const facing = strategy.calculate(
		{ x: 0, y: 0, z: 0 },
		{ x: 0, y: 0, z: 1 },
		{ x: 0, y: 0, z: 1 },
		surface,
		context
	);
	const grazing = strategy.calculate(
		{ x: 0, y: 0, z: 0 },
		{ x: 0, y: 0, z: 1 },
		{ x: 0.9999995, y: 0, z: 0.001 },
		surface,
		context
	);

	assert.deepEqual(grazing, facing);
}

function testAnalyticalAmbientAddsIndependentlyOfProbeSH() {
	const sh = createConstantWhiteSH(0.25);
	const probe = new LightProbe({ sh });
	const ambient = new AmbientLight({ intensity: 1 });
	const position = { x: 0, y: 0, z: 0 };
	const normal = { x: 0, y: 0, z: 1 };
	const baseSurface = {
		albedo: { r: 255, g: 255, b: 255 }, opacity: 1, normal,
		emissive: { r: 0, g: 0, b: 0 }, emissiveIntensity: 1,
		roughness: 0.6, metalness: 0, reflectance: 0.5, occlusion: 1,
		clearcoat: 0, clearcoatRoughness: 0.1,
		ambient: { r: 255, g: 255, b: 255 },
		specular: { r: 0, g: 0, b: 0 }, shininess: 32,
	};
	for (const [strategy, surface, expectedAmbient] of [
		[new BlinnPhongStrategy(), { ...baseSurface, type: "phong" }, 1 / Math.PI],
		[new PBRStrategy(), { ...baseSurface, type: "pbr" }, 1 / Math.PI],
		[new PBRStrategy(), { ...baseSurface, type: "pbr", metalness: 1 }, 0],
		[new PBRStrategy(), { ...baseSurface, type: "pbr", clearcoat: 1 }, 0.96 / Math.PI],
	]) {
		for (const view of [normal, { x: 0.9999995, y: 0, z: 0.001 }]) {
			for (const enableSH of [false, true]) {
				const context = createContext({ enableSH, shAmbientCoeffs: sh, lights: [probe] });
				const probeOnly = strategy.calculate(position, normal, view, surface, context);
				const combined = strategy.calculate(position, normal, view, surface, {
					...context, lights: [probe, ambient],
				});
				for (const channel of ["r", "g", "b"]) {
					assert.ok(Math.abs(combined[channel] - probeOnly[channel] - expectedAmbient) < 1e-6,
						`${surface.type}: ambient diffuse must remain additive with SH=${enableSH}`);
				}
			}
		}
	}
}

function testZeroProbeAggregateRemainsAuthoritative() {
	const positive = createConstantWhiteSH(0.25);
	const negative = SH.empty();
	negative[0] = { r: -positive[0].r, g: -positive[0].g, b: -positive[0].b };
	const lights = [new AmbientLight({ intensity: 1 }),
		new LightProbe({ sh: positive }), new LightProbe({ sh: negative })];
	const normal = { x: 0, y: 0, z: 1 };
	const surface = {
		albedo: { r: 255, g: 255, b: 255 }, opacity: 1, normal,
		emissive: { r: 0, g: 0, b: 0 }, emissiveIntensity: 1,
		roughness: 1, metalness: 0, reflectance: 0, occlusion: 1,
		clearcoat: 0, clearcoatRoughness: 0,
		ambient: { r: 255, g: 255, b: 255 },
		specular: { r: 0, g: 0, b: 0 }, shininess: 32,
	};
	for (const strategy of [new BlinnPhongStrategy(), new PBRStrategy()]) {
		for (const [enableSH, coefficients, expected] of [
			[true, SH.empty(), 1 / Math.PI],
			[true, null, 1 / Math.PI + 0.25],
			[false, SH.empty(), 1 / Math.PI + 0.25],
		]) {
			const result = strategy.calculate({ x: 0, y: 0, z: 0 }, normal, normal,
				surface, createContext({ lights, enableSH, shAmbientCoeffs: coefficients }));
			assert.ok(Math.abs(result.r - expected) < 1e-6,
				"A present zero SH aggregate must not fall back to clamped individual probes");
		}
	}
}

function testPBREnvironmentTexturesWorkWithoutDiffuseSH() {
	const map = new Texture({ width: 1, height: 1, colorSpace: "HDR",
		data: new Float32Array([1.75, 0.5, 2.5, 1]) });
	const radiance = { r: 1.75, g: 0.5, b: 2.5 };
	const brdfLUT = new Texture({ width: 1, height: 1, colorSpace: "HDR",
		data: new Float32Array([1, 0, 0, 1]) });
	const probe = new ReflectionProbe({ prefilteredMap: map, radius: 10, parallaxMode: "off" });
	const strategy = new PBRStrategy();
	const normal = { x: 0, y: 0, z: 1 };
	const baseSurface = {
		type: "pbr", albedo: { r: 255, g: 255, b: 255 }, opacity: 1, normal,
		emissive: { r: 0, g: 0, b: 0 }, emissiveIntensity: 1,
		roughness: 0.5, metalness: 0, reflectance: 0, occlusion: 1,
		clearcoat: 0, clearcoatRoughness: 0.5,
	};
	// Reuse a strategy with populated SH scratch data to catch stale reads.
	strategy.calculate({ x: 0, y: 0, z: 0 }, normal, normal, baseSurface,
		createContext({ shAmbientCoeffs: createConstantWhiteSH(0.25) }));
	for (const [surface, lutResponse, fallbackResponse, ambientContribution] of [
		[{ ...baseSurface, metalness: 1 }, 1, 0.25, 0],
		[{ ...baseSurface, transmission: 1 }, 1, 1, 0],
		[{ ...baseSurface, albedo: { r: 0, g: 0, b: 0 }, clearcoat: 1 }, 0.04, 0.01, 0],
		[baseSurface, 0, 0, 1 / Math.PI],
	]) {
		for (const enableSH of [true, false]) {
			for (const ambient of [[], [new AmbientLight({ intensity: 1 })]]) {
				for (const useReflectionProbe of [false, true]) {
					for (const useLUT of [true, false]) {
						const result = strategy.calculate({ x: 0, y: 0, z: 0 }, normal, normal,
							surface, createContext({ enableSH, lights: ambient,
								brdfLUT: useLUT ? brdfLUT : null,
								shAmbientCoeffs: enableSH ? SH.empty() : null,
								reflectionProbes: useReflectionProbe ? [probe] : [],
								reflectionProbeFallbackMap: useReflectionProbe ? null : map,
							}));
						for (const channel of ["r", "g", "b"]) {
							const response = useLUT ? lutResponse : fallbackResponse;
							const expected = radiance[channel] * response +
								(ambient.length > 0 ? ambientContribution : 0);
							assert.ok(Math.abs(result[channel] - expected) < 1e-6,
								`Environment lighting must survive missing SH/LUT: ${result[channel]}`);
						}
					}
				}
			}
		}
	}
}

function testPBRMissingLUTPrefersTextureOverSH() {
	const map = new Texture({ width: 1, height: 1, colorSpace: "HDR",
		data: new Float32Array([0, 0.5, 1.75, 1]) });
	const normal = { x: 0, y: 0, z: 1 };
	const base = {
		type: "pbr", albedo: { r: 255, g: 255, b: 255 }, opacity: 1, normal,
		emissive: { r: 0, g: 0, b: 0 }, emissiveIntensity: 1,
		roughness: 0.5, metalness: 0, reflectance: 0, occlusion: 1,
		clearcoat: 0, clearcoatRoughness: 0.5,
	};
	for (const [surface, expected] of [
		[{ ...base, metalness: 1 }, { r: 0, g: 0.125, b: 0.4375 }],
		[{ ...base, albedo: { r: 0, g: 0, b: 0 }, clearcoat: 1 },
			{ r: 0, g: 0.005, b: 0.0175 }],
	]) {
		const result = new PBRStrategy().calculate({ x: 0, y: 0, z: 0 }, normal, normal,
			surface, createContext({ brdfLUT: null,
				shAmbientCoeffs: createConstantWhiteSH(0.25),
				reflectionProbeFallbackMap: map,
			}));
		for (const channel of ["r", "g", "b"]) {
			assert.ok(Math.abs(result[channel] - expected[channel]) < 1e-6,
				"Valid texture radiance, including zero channels, must take precedence over SH");
		}
	}
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
		enableSH: true,
		shAmbientCoeffs: createConstantWhiteSH(0.25),
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

function testTransmissionVolumeAttenuationColorsSHAmbient() {
	const strategy = new PBRStrategy();
	const context = createContext({
		enableSH: true,
		shAmbientCoeffs: createConstantWhiteSH(),
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
		color.b > 0 && color.r < color.b * 0.25 && color.g < color.b * 0.25,
		"Transmission volume attenuation should tint SH ambient radiance"
	);
}

function testPBRNormalMapFallbackWithoutTangent() {
	const material = new PBRMaterial();
	material.normalMap = new Texture({
		data: new Uint8ClampedArray([255, 128, 128, 255]),
		width: 1,
		height: 1,
	});

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
		map: new Texture({ data: new Uint8ClampedArray([255, 255, 255, 0]), width: 1, height: 1 }),
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

	maskMaterial.map = new Texture({
		data: new Uint8ClampedArray([255, 255, 255, 255]),
		width: 1,
		height: 1,
	});
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
		backLit.r > frontLit.r + 50 / 255,
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
		dielectric.r > metal.r + 50 / 255,
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
			enableSH: true,
			shAmbientCoeffs: createConstantWhiteSH(),
		})
	);

	assert.ok(
		Math.abs(color.r - 128 / 255) < 1.5 / 255,
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

function testRendererUpdateSHRequiresCoordinator() {
	assert.throws(
		() => Renderer.prototype.updateSH.call({}),
		/Renderer\.updateSH\(\) requires the renderer-owned FrameCoordinator\./
	);
}

function testRendererUpdateSHDelegatesToCoordinator() {
	const fakeRenderer = {};
	let receivedDelegate = null;
	fakeRenderer._coordinator = {
		updateSH(delegate) {
			receivedDelegate = delegate;
		},
	};

	Renderer.prototype.updateSH.call(fakeRenderer);

	assert.equal(receivedDelegate, fakeRenderer);
}

function run() {
	try {
		testDoubleSidedNormalUsesRasterizerFacing();
		testDoubleSidedNormalMapUsesOrientedTBN();
		testSHAmbientGateForBlinnPhong();
		testSHAmbientGateForPBR();
		testPBRAmbientIrradianceIsViewIndependent();
		testAnalyticalAmbientAddsIndependentlyOfProbeSH();
		testZeroProbeAggregateRemainsAuthoritative();
		testPBREnvironmentTexturesWorkWithoutDiffuseSH();
		testPBRMissingLUTPrefersTextureOverSH();
		testPBRSHAmbientSpecularTracksReflectionDirection();
		testClearcoatAttenuatesAmbientSheen();
		testTransmissionVolumeAttenuationColorsSHAmbient();
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
		testRendererUpdateSHRequiresCoordinator();
		testRendererUpdateSHDelegatesToCoordinator();
		console.log("✅ Shader semantics tests passed");
	} catch (error) {
		console.error("❌ Shader semantics test failed");
		console.error(error);
		process.exit(1);
	}
}

run();
