import assert from "node:assert/strict";
import { Texture } from "../../../src/core/Texture.ts";
import { AlphaMode, BasicMaterial, PBRMaterial } from "../../../src/materials/index.ts";
import { SoftwareMaterialRuntime } from "../../../src/renderers/software/SoftwareMaterialRuntime.ts";
import { sampleSoftwareTextureMap } from "../../../src/shaders/software/textureSampling.ts";

function createFace(material) {
	return {
		material,
		vertices: [],
		projected: [],
		center: { x: 0, y: 0, z: 0 },
		normal: { x: 0, y: 0, z: 1 },
		depthInfo: { min: 1, max: 1, avg: 1 },
	};
}

function createContext(overrides = {}) {
	return {
		camera: {
			position: { x: 0, y: 0, z: 4 },
		},
		lights: [],
		sampleShadow: undefined,
		shAmbientCoeffs: null,
		environmentSpecularTexture: null,
		enableLighting: false,
		enableSH: false,
		enableShadows: false,
		...overrides,
	};
}

function createInput(overrides = {}) {
	return {
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
		...overrides,
	};
}

function testLightingDisabledUsesUnlitProgram() {
	const runtime = new SoftwareMaterialRuntime();
	const material = new BasicMaterial({
		diffuse: { r: 32, g: 64, b: 96 },
	});

	const program = runtime.prepareFragmentProgram(
		createFace(material),
		createContext({ enableLighting: false, lights: [] }),
		false
	);
	const output = program.shade(createInput());

	assert.ok(output, "Software fragment program should shade visible fragments");
	assert.equal(output.color.r, 32);
	assert.equal(output.color.g, 64);
	assert.equal(output.color.b, 96);
}

function testPBRMaterialKeepsPBREvaluatorWhenUnlit() {
	const runtime = new SoftwareMaterialRuntime();
	const material = new PBRMaterial({
		albedo: { r: 12, g: 34, b: 56 },
	});

	const program = runtime.prepareFragmentProgram(
		createFace(material),
		createContext({ enableLighting: false }),
		false
	);
	const output = program.shade(createInput());

	assert.ok(output, "PBR material should shade through the unlit path");
	assert.equal(output.color.r, 12);
	assert.equal(output.color.g, 34);
	assert.equal(output.color.b, 56);
}

function testDepthWriteReflectsTransparentPassAndMaterialState() {
	const runtime = new SoftwareMaterialRuntime();
	const material = new BasicMaterial({
		diffuse: { r: 255, g: 255, b: 255 },
		depthWrite: true,
	});

	const opaqueProgram = runtime.prepareFragmentProgram(
		createFace(material),
		createContext(),
		false
	);
	const transparentProgram = runtime.prepareFragmentProgram(
		createFace(material),
		createContext(),
		true
	);
	material.depthWrite = false;
	const depthReadProgram = runtime.prepareFragmentProgram(
		createFace(material),
		createContext(),
		false
	);

	assert.equal(opaqueProgram.shouldWriteDepth, true);
	assert.equal(transparentProgram.shouldWriteDepth, false);
	assert.equal(depthReadProgram.shouldWriteDepth, false);
}

function testAlphaMaskUsesSharedTextureSampling() {
	const runtime = new SoftwareMaterialRuntime();
	const texture = new Texture(
		new Uint8ClampedArray([
			255, 255, 255, 64,
			255, 255, 255, 192,
		]),
		2,
		1
	);
	texture.wrapS = "Clamp";
	texture.wrapT = "Clamp";
	texture.offset.x = 0.25;
	const material = new BasicMaterial({
		alphaMode: AlphaMode.Mask,
		opacity: 0.5,
		map: texture,
	});

	const sample = sampleSoftwareTextureMap(texture, 0.75, 0);
	assert.ok(sample, "Shared sampler should return the expected texture sample");
	assert.equal(runtime.sampleAlphaMask(material, 0.75, 0), sample.a * 0.5);
}

function run() {
	testLightingDisabledUsesUnlitProgram();
	testPBRMaterialKeepsPBREvaluatorWhenUnlit();
	testDepthWriteReflectsTransparentPassAndMaterialState();
	testAlphaMaskUsesSharedTextureSampling();
	console.log("Software material runtime tests passed");
}

run();
