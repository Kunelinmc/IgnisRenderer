import assert from "node:assert/strict";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import {
	arrayOf,
	mat4x4f32,
	structOf,
	StructuredBufferLayout,
	vec,
} from "../../../src/renderers/webgpu/StructuredBufferLayout.ts";
import {
	WEBGPU_FRAME_UNIFORM_BYTE_SIZE,
	WEBGPU_MODEL_UNIFORM_BYTE_SIZE,
	packFrameUniformData,
	packModelUniformData,
} from "../../../src/renderers/webgpu/packing.ts";
import {
	WEBGPU_MAX_AREA_LIGHTS,
	WEBGPU_MAX_DIRECTIONAL_LIGHTS,
	WEBGPU_MAX_LOCAL_LIGHT_PROBES,
	WEBGPU_MAX_POINT_LIGHTS,
	WEBGPU_MAX_REFLECTION_PROBES,
	WEBGPU_MAX_SPOT_LIGHTS,
	WEBGPU_SH_COEFFICIENT_COUNT,
	WEBGPU_TEXTURE_SLOT_COUNT,
} from "../../../src/renderers/webgpu/constants.ts";

const VEC4_F32 = vec(4, "f32");
const MAT4X4_F32 = mat4x4f32();

function createModelLayout() {
	return new StructuredBufferLayout(
		structOf([
			{ name: "modelMatrix", type: MAT4X4_F32 },
			{ name: "prevModelMatrix", type: MAT4X4_F32 },
			{ name: "normalMatrix", type: MAT4X4_F32 },
			{ name: "baseColorFactor", type: VEC4_F32 },
			{ name: "emissiveFactor", type: VEC4_F32 },
			{ name: "surfaceParams0", type: VEC4_F32 },
			{ name: "surfaceParams1", type: VEC4_F32 },
			{ name: "surfaceParams2", type: VEC4_F32 },
			{ name: "surfaceParams3", type: VEC4_F32 },
			{ name: "specularColorFactor", type: VEC4_F32 },
			{ name: "phongAmbientShininess", type: VEC4_F32 },
			{ name: "phongSpecularShading", type: VEC4_F32 },
			{ name: "sheenColorClearcoatNormalScale", type: VEC4_F32 },
			{ name: "attenuationColor", type: VEC4_F32 },
			{ name: "anisotropyParams", type: VEC4_F32 },
			{ name: "anisotropyTextureTransformA", type: VEC4_F32 },
			{ name: "anisotropyTextureTransformB", type: VEC4_F32 },
			{ name: "materialFlags", type: VEC4_F32 },
			{ name: "nodeRenderLayers", type: VEC4_F32 },
			{
				name: "textureTransformA",
				type: arrayOf(VEC4_F32, WEBGPU_TEXTURE_SLOT_COUNT),
			},
			{
				name: "textureTransformB",
				type: arrayOf(VEC4_F32, WEBGPU_TEXTURE_SLOT_COUNT),
			},
		]),
		"uniform"
	);
}

function createFrameLayout() {
	const directionalLightSchema = structOf([
		{ name: "direction", type: VEC4_F32 },
		{ name: "color", type: VEC4_F32 },
	]);
	const pointLightSchema = structOf([
		{ name: "positionRange", type: VEC4_F32 },
		{ name: "color", type: VEC4_F32 },
	]);
	const spotLightSchema = structOf([
		{ name: "positionRange", type: VEC4_F32 },
		{ name: "directionOuter", type: VEC4_F32 },
		{ name: "colorInner", type: VEC4_F32 },
	]);
	const areaLightSchema = structOf([
		{ name: "positionRange", type: VEC4_F32 },
		{ name: "rightWidth", type: VEC4_F32 },
		{ name: "upHeight", type: VEC4_F32 },
		{ name: "normalAreaScale", type: VEC4_F32 },
		{ name: "color", type: VEC4_F32 },
	]);
	const shadowDataSchema = structOf([
		{ name: "viewProjection", type: MAT4X4_F32 },
		{ name: "cascadeViewProjections", type: arrayOf(MAT4X4_F32, 4) },
		{ name: "cascadeSplits", type: arrayOf(VEC4_F32, 4) },
		{ name: "paramsA", type: VEC4_F32 },
		{ name: "paramsB", type: VEC4_F32 },
		{ name: "paramsC", type: VEC4_F32 },
		{ name: "paramsD", type: VEC4_F32 },
	]);
	const reflectionProbeSchema = structOf([
		{ name: "worldToProbeRow0", type: VEC4_F32 },
		{ name: "worldToProbeRow1", type: VEC4_F32 },
		{ name: "worldToProbeRow2", type: VEC4_F32 },
		{ name: "probeToWorldRow0", type: VEC4_F32 },
		{ name: "probeToWorldRow1", type: VEC4_F32 },
		{ name: "probeToWorldRow2", type: VEC4_F32 },
		{ name: "dataA", type: VEC4_F32 },
		{ name: "dataB", type: VEC4_F32 },
		{ name: "dataC", type: VEC4_F32 },
	]);

	return new StructuredBufferLayout(
		structOf([
			{ name: "viewProjection", type: MAT4X4_F32 },
			{ name: "prevViewProjection", type: MAT4X4_F32 },
			{ name: "cameraPosition", type: VEC4_F32 },
			{ name: "environmentBasisRight", type: VEC4_F32 },
			{ name: "environmentBasisUp", type: VEC4_F32 },
			{ name: "environmentBasisBackward", type: VEC4_F32 },
			{ name: "ambientColor", type: VEC4_F32 },
			{ name: "lightCounts", type: VEC4_F32 },
			{ name: "options", type: VEC4_F32 },
			{ name: "environmentOptionsA", type: VEC4_F32 },
			{ name: "environmentOptionsB", type: VEC4_F32 },
			{ name: "taaJitterCurrentPrev", type: VEC4_F32 },
			{
				name: "directionalLights",
				type: arrayOf(directionalLightSchema, WEBGPU_MAX_DIRECTIONAL_LIGHTS),
			},
			{ name: "pointLights", type: arrayOf(pointLightSchema, WEBGPU_MAX_POINT_LIGHTS) },
			{ name: "spotLights", type: arrayOf(spotLightSchema, WEBGPU_MAX_SPOT_LIGHTS) },
			{
				name: "directionalShadows",
				type: arrayOf(shadowDataSchema, WEBGPU_MAX_DIRECTIONAL_LIGHTS),
			},
			{
				name: "spotShadows",
				type: arrayOf(shadowDataSchema, WEBGPU_MAX_SPOT_LIGHTS),
			},
			{ name: "shAmbientCoeffs", type: arrayOf(VEC4_F32, WEBGPU_SH_COEFFICIENT_COUNT) },
			{
				name: "reflectionProbes",
				type: arrayOf(reflectionProbeSchema, WEBGPU_MAX_REFLECTION_PROBES),
			},
			{ name: "localLightProbeCounts", type: VEC4_F32 },
			{
				name: "localLightProbeWorldToProbeRow0",
				type: arrayOf(VEC4_F32, WEBGPU_MAX_LOCAL_LIGHT_PROBES),
			},
			{
				name: "localLightProbeWorldToProbeRow1",
				type: arrayOf(VEC4_F32, WEBGPU_MAX_LOCAL_LIGHT_PROBES),
			},
			{
				name: "localLightProbeWorldToProbeRow2",
				type: arrayOf(VEC4_F32, WEBGPU_MAX_LOCAL_LIGHT_PROBES),
			},
			{
				name: "localLightProbeDataA",
				type: arrayOf(VEC4_F32, WEBGPU_MAX_LOCAL_LIGHT_PROBES),
			},
			{
				name: "localLightProbeDataB",
				type: arrayOf(VEC4_F32, WEBGPU_MAX_LOCAL_LIGHT_PROBES),
			},
			{
				name: "localLightProbeSHAmbientCoeffs",
				type: arrayOf(
					VEC4_F32,
					WEBGPU_MAX_LOCAL_LIGHT_PROBES * WEBGPU_SH_COEFFICIENT_COUNT
				),
			},
			{ name: "irradianceProbeGridWorldToGridRow0", type: VEC4_F32 },
			{ name: "irradianceProbeGridWorldToGridRow1", type: VEC4_F32 },
			{ name: "irradianceProbeGridWorldToGridRow2", type: VEC4_F32 },
			{ name: "irradianceProbeGridDataA", type: VEC4_F32 },
			{ name: "irradianceProbeGridDataB", type: VEC4_F32 },
			{ name: "irradianceProbeGridDataC", type: VEC4_F32 },
			{ name: "areaLightCounts", type: VEC4_F32 },
			{ name: "areaLights", type: arrayOf(areaLightSchema, WEBGPU_MAX_AREA_LIGHTS) },
		]),
		"uniform"
	);
}

function matrix(base) {
	return new Matrix4([
		[base + 0, base + 1, base + 2, base + 3],
		[base + 4, base + 5, base + 6, base + 7],
		[base + 8, base + 9, base + 10, base + 11],
		[base + 12, base + 13, base + 14, base + 15],
	]);
}

function readVec(layout, data, path, length) {
	const offset = layout.byteOffsetOf(path) >> 2;
	return Array.from(data.slice(offset, offset + length));
}

function createTextureSlot(transformA, transformB) {
	return {
		map: null,
		transformA,
		transformB,
	};
}

function createMaterialData() {
	const textureSlots = Array.from({ length: WEBGPU_TEXTURE_SLOT_COUNT }, () => null);
	textureSlots[0] = createTextureSlot([101, 102, 103, 104], [201, 202, 203, 204]);
	textureSlots[1] = createTextureSlot([105, 106, 107, 108], [205, 206, 207, 208]);

	return {
		baseColorFactor: [1, 2, 3, 4],
		emissiveFactor: [5, 6, 7, 8],
		surfaceParams0: [9, 10, 11, 12],
		surfaceParams1: [13, 14, 15, 16],
		surfaceParams2: [17, 18, 19, 20],
		surfaceParams3: [21, 22, 23, 24],
		specularColorFactor: [25, 26, 27, 28],
		phongAmbientShininess: [29, 30, 31, 32],
		phongSpecularShading: [33, 34, 35, 36],
		sheenColorClearcoatNormalScale: [37, 38, 39, 40],
		attenuationColor: [41, 42, 43, 44],
		anisotropyParams: [45, 46, 47, 48],
		anisotropyTexture: createTextureSlot([49, 50, 51, 52], [53, 54, 55, 56]),
		materialFlags: [57, 58, 59, 60],
		textureSlots,
		shaderUniforms: {
			cacheKey: "none",
			byteLength: 16,
			valueRevision: 0,
			data: null,
		},
		pipelineKey: "test",
		warnings: [],
	};
}

function testModelUniformPacking() {
	const layout = createModelLayout();
	assert.equal(layout.byteSize, WEBGPU_MODEL_UNIFORM_BYTE_SIZE);

	const data = packModelUniformData(
		matrix(1),
		[
			[301, 302, 303],
			[304, 305, 306],
			[307, 308, 309],
		],
		createMaterialData(),
		matrix(401),
		7
	);

	assert.equal(data.length * 4, WEBGPU_MODEL_UNIFORM_BYTE_SIZE);
	assert.deepEqual(readVec(layout, data, "modelMatrix", 16), [
		1, 5, 9, 13,
		2, 6, 10, 14,
		3, 7, 11, 15,
		4, 8, 12, 16,
	]);
	assert.deepEqual(readVec(layout, data, "normalMatrix", 16), [
		301, 304, 307, 0,
		302, 305, 308, 0,
		303, 306, 309, 0,
		0, 0, 0, 1,
	]);
	assert.deepEqual(readVec(layout, data, "baseColorFactor", 4), [1, 2, 3, 4]);
	assert.deepEqual(readVec(layout, data, "anisotropyTextureTransformA", 4), [
		49, 50, 51, 52,
	]);
	assert.deepEqual(readVec(layout, data, "nodeRenderLayers", 4), [7, 0, 0, 0]);
	assert.deepEqual(readVec(layout, data, ["textureTransformA", 1], 4), [
		105, 106, 107, 108,
	]);
	assert.deepEqual(readVec(layout, data, ["textureTransformB", 0], 4), [
		201, 202, 203, 204,
	]);
}

function createReflectionProbe(index) {
	return {
		id: `probe-${index}`,
		worldToProbeMatrix: matrix(1000 + index * 100),
		probeToWorldMatrix: matrix(2000 + index * 100),
		invHalfExtents: [1 + index, 2 + index, 3 + index],
		radiusInv: 4 + index,
		captureWorldPosition: [5 + index, 6 + index, 7 + index],
		shape: index % 2,
		parallaxMode: index,
		blendDistance: 8 + index,
		blendExponent: 9 + index,
		layer: 10 + index,
	};
}

function createFrameInput() {
	return {
		viewProjectionMatrix: matrix(1),
		prevViewProjectionMatrix: matrix(101),
		cameraPosition: { x: 1, y: 2, z: 3 },
		environmentRight: [4, 5, 6],
		environmentUp: [8, 9, 10],
		environmentBackward: [12, 13, 14],
		environmentTanHalfFov: 7,
		environmentAspect: 11,
		environmentIsOrthographic: true,
		ambientColor: [15, 16, 17],
		shAmbientCoeffs: null,
		localLightProbeCount: 0,
		localLightProbes: [],
		irradianceProbeGrid: {
			id: "grid-0",
			worldToGridMatrix: matrix(3000),
			dimensions: [3, 4, 5],
			invHalfExtents: [0.25, 0.5, 0.75],
			blendDistance: 12,
			cellCount: 60,
			textureRevision: 1,
			sh: [],
			validMask: new Uint8Array(60),
		},
		directionalLights: [
			{
				direction: [18, 19, 20],
				color: [21, 22, 23],
			},
		],
		directionalShadows: [],
		pointLights: [
			{
				position: [24, 25, 26],
				range: 27,
				color: [28, 29, 30],
			},
		],
		spotLights: [
			{
				position: [31, 32, 33],
				range: 34,
				direction: [35, 36, 37],
				outerCos: 38,
				color: [39, 40, 41],
				innerCos: 42,
			},
		],
		spotShadows: [],
		areaLights: [
			{
				position: [47, 48, 49],
				range: 50,
				right: [51, 52, 53],
				width: 54,
				up: [55, 56, 57],
				height: 58,
				normal: [59, 60, 61],
				areaScale: 62,
				color: [63, 64, 65],
			},
		],
		reflectionProbeCount: 2,
		reflectionProbes: [
			createReflectionProbe(0),
			createReflectionProbe(1),
		],
		enableLighting: true,
		enableGamma: false,
		enableShadows: true,
		enableSH: true,
		enableClusteredLighting: true,
		encodeGammaInShader: true,
		hasSHAmbient: false,
		hasEnvironment: true,
		environmentIsLinear: false,
		hasEnvSpecular: true,
		hasEnvSpecularFallback: true,
		hasBRDFLUT: true,
		envSpecularMaxMipLevel: 5,
		envSpecularFallbackMaxMipLevel: 6,
		taaJitterCurrentPrev: [43, 44, 45, 46],
	};
}

function testFrameUniformPacking() {
	const layout = createFrameLayout();
	assert.equal(layout.byteSize, WEBGPU_FRAME_UNIFORM_BYTE_SIZE);

	const data = packFrameUniformData(createFrameInput());
	assert.equal(data.length * 4, WEBGPU_FRAME_UNIFORM_BYTE_SIZE);
	assert.deepEqual(readVec(layout, data, "cameraPosition", 4), [1, 2, 3, 1]);
	assert.deepEqual(readVec(layout, data, "environmentBasisRight", 4), [
		4, 5, 6, 7,
	]);
	assert.deepEqual(readVec(layout, data, "lightCounts", 4), [1, 1, 1, 2]);
	assert.deepEqual(readVec(layout, data, "options", 4), [1, 0, 1, 1]);
	assert.deepEqual(readVec(layout, data, "environmentOptionsA", 4), [
		1, 0, 7, 1,
	]);
	assert.deepEqual(readVec(layout, data, "environmentOptionsB", 4), [
		1, 5, 0, 1,
	]);
	assert.deepEqual(readVec(layout, data, ["directionalLights", 0, "direction"], 4), [
		18, 19, 20, 0,
	]);
	assert.deepEqual(readVec(layout, data, ["pointLights", 0, "positionRange"], 4), [
		24, 25, 26, 27,
	]);
	assert.deepEqual(readVec(layout, data, ["spotLights", 0, "colorInner"], 4), [
		39, 40, 41, 42,
	]);
	assert.deepEqual(readVec(layout, data, "areaLightCounts", 4), [1, 0, 0, 0]);
	assert.deepEqual(readVec(layout, data, ["areaLights", 0, "positionRange"], 4), [
		47, 48, 49, 50,
	]);
	assert.deepEqual(readVec(layout, data, ["areaLights", 0, "normalAreaScale"], 4), [
		59, 60, 61, 62,
	]);
	assert.deepEqual(
		readVec(layout, data, ["reflectionProbes", 0, "worldToProbeRow0"], 4),
		[1000, 1001, 1002, 1003]
	);
	assert.deepEqual(
		readVec(layout, data, ["reflectionProbes", 1, "probeToWorldRow2"], 4),
		[2108, 2109, 2110, 2111]
	);
	assert.deepEqual(readVec(layout, data, ["reflectionProbes", 1, "dataC"], 4), [
		1, 9, 10, 11,
	]);
	assert.deepEqual(readVec(layout, data, "irradianceProbeGridWorldToGridRow1", 4), [
		3004, 3005, 3006, 3007,
	]);
	assert.deepEqual(readVec(layout, data, "irradianceProbeGridDataA", 4), [
		3, 4, 5, 60,
	]);
	assert.deepEqual(readVec(layout, data, "irradianceProbeGridDataB", 4), [
		0.25, 0.5, 0.75, 12,
	]);
	assert.deepEqual(readVec(layout, data, "irradianceProbeGridDataC", 4), [
		1, WEBGPU_SH_COEFFICIENT_COUNT, 60, 0,
	]);
}

testModelUniformPacking();
testFrameUniformPacking();
console.log("WebGPU packing tests passed");
