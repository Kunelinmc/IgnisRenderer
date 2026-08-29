import assert from "node:assert/strict";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import {
	WEBGPU_FRAME_CAMERA_UNIFORM_BYTE_SIZE,
	WEBGPU_FRAME_ENVIRONMENT_UNIFORM_BYTE_SIZE,
	WEBGPU_FRAME_LIGHT_UNIFORM_BYTE_SIZE,
	WEBGPU_FRAME_SHADOW_UNIFORM_BYTE_SIZE,
	WEBGPU_FLAT_MATERIAL_UNIFORM_BYTE_SIZE,
	WEBGPU_MATERIAL_COMMON_UNIFORM_BYTE_SIZE,
	WEBGPU_OBJECT_UNIFORM_BYTE_SIZE,
	WEBGPU_PBR_MATERIAL_UNIFORM_BYTE_SIZE,
	WEBGPU_PHONG_MATERIAL_UNIFORM_BYTE_SIZE,
	packFrameCameraUniformData,
	packFrameEnvironmentUniformData,
	packFrameLightUniformData,
	packFrameShadowUniformData,
	packFlatMaterialUniformData,
	packMaterialCommonUniformData,
	packObjectUniformData,
	packPBRMaterialUniformData,
	packPhongMaterialUniformData,
} from "../../../src/backends/webgpu/packing.ts";
import {
	MAX_AREA_LIGHTS,
	MAX_DIRECTIONAL_LIGHTS,
	MAX_LOCAL_LIGHT_PROBES,
	MAX_POINT_LIGHTS,
	MAX_REFLECTION_PROBES,
	MAX_SPOT_LIGHTS,
} from "../../../src/backends/constants.ts";
import {
	WEBGPU_SH_COEFFICIENT_COUNT,
	WEBGPU_TEXTURE_SLOT_COUNT,
} from "../../../src/backends/webgpu/constants.ts";
import {
	WEBGPU_FRAME_CAMERA_UNIFORM_LAYOUT,
	WEBGPU_FRAME_ENVIRONMENT_UNIFORM_LAYOUT,
	WEBGPU_FRAME_LIGHT_UNIFORM_LAYOUT,
	WEBGPU_FRAME_SHADOW_UNIFORM_LAYOUT,
	WEBGPU_FLAT_MATERIAL_UNIFORM_LAYOUT,
	WEBGPU_MATERIAL_COMMON_UNIFORM_LAYOUT,
	WEBGPU_OBJECT_UNIFORM_LAYOUT,
	WEBGPU_PBR_MATERIAL_UNIFORM_LAYOUT,
	WEBGPU_PHONG_MATERIAL_UNIFORM_LAYOUT,
} from "../../../src/backends/webgpu/bufferLayouts.ts";

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

function readU32Vec(layout, data, path, length) {
	const offset = layout.byteOffsetOf(path) >> 2;
	return Array.from(
		new Uint32Array(data.buffer, data.byteOffset, data.length).slice(
			offset,
			offset + length
		)
	);
}

function createTextureSlot(transformA, transformB) {
	return {
		map: null,
		transformA,
		transformB,
	};
}

function createCommonMaterialData() {
	const textureSlots = Array.from({ length: WEBGPU_TEXTURE_SLOT_COUNT }, () => null);
	textureSlots[0] = createTextureSlot([101, 102, 103, 104], [201, 202, 203, 204]);
	textureSlots[1] = createTextureSlot([105, 106, 107, 108], [205, 206, 207, 208]);
	textureSlots[16] = createTextureSlot([49, 50, 51, 52], [53, 54, 55, 56]);

	return {
		baseColorFactor: [1, 2, 3, 4],
		emissiveFactor: [5, 6, 7, 8],
		materialParams: [57, 58, 59, 60],
		renderParams: [63, 0, 0, 0],
		textureSlots,
	};
}

function createPBRMaterialData() {
	return {
		surfaceParams0: [9, 10, 11, 12],
		surfaceParams1: [13, 14, 15, 16],
		surfaceParams2: [17, 18, 19, 20],
		surfaceParams3: [21, 22, 23, 24],
		specularColorFactor: [25, 26, 27, 28],
		sheenColorClearcoatNormalScale: [37, 38, 39, 40],
		attenuationColor: [41, 42, 43, 44],
		anisotropyParams: [45, 46, 47, 48],
		pbrMasks: [61, 62, 0, 0],
	};
}

function testSeparatedMaterialUniformPacking() {
	const objectData = packObjectUniformData(
		matrix(1),
		[
			[301, 302, 303],
			[304, 305, 306],
			[307, 308, 309],
		],
		matrix(401),
		7,
		true,
		false,
	);
	assert.equal(objectData.length * 4, WEBGPU_OBJECT_UNIFORM_BYTE_SIZE);
	assert.deepEqual(readVec(WEBGPU_OBJECT_UNIFORM_LAYOUT, objectData, "modelMatrix", 16), [
		1, 5, 9, 13,
		2, 6, 10, 14,
		3, 7, 11, 15,
		4, 8, 12, 16,
	]);
	assert.deepEqual(readVec(WEBGPU_OBJECT_UNIFORM_LAYOUT, objectData, "normalMatrix", 16), [
		301, 304, 307, 0,
		302, 305, 308, 0,
		303, 306, 309, 0,
		0, 0, 0, 1,
	]);
	assert.deepEqual(
		readVec(WEBGPU_OBJECT_UNIFORM_LAYOUT, objectData, "instanceData", 4),
		[7, 1, 0, 0],
	);

	const commonData = packMaterialCommonUniformData(createCommonMaterialData());
	assert.equal(commonData.length * 4, WEBGPU_MATERIAL_COMMON_UNIFORM_BYTE_SIZE);
	assert.deepEqual(
		readVec(WEBGPU_MATERIAL_COMMON_UNIFORM_LAYOUT, commonData, "baseColorFactor", 4),
		[1, 2, 3, 4],
	);
	assert.deepEqual(readVec(WEBGPU_MATERIAL_COMMON_UNIFORM_LAYOUT, commonData, ["textureTransformA", 16], 4), [
		49, 50, 51, 52,
	]);
	assert.deepEqual(readVec(WEBGPU_MATERIAL_COMMON_UNIFORM_LAYOUT, commonData, ["textureTransformA", 1], 4), [
		105, 106, 107, 108,
	]);
	assert.deepEqual(readVec(WEBGPU_MATERIAL_COMMON_UNIFORM_LAYOUT, commonData, ["textureTransformB", 0], 4), [
		201, 202, 203, 204,
	]);

	const pbrData = packPBRMaterialUniformData(createPBRMaterialData());
	assert.equal(pbrData.length * 4, WEBGPU_PBR_MATERIAL_UNIFORM_BYTE_SIZE);
	assert.deepEqual(
		readU32Vec(WEBGPU_PBR_MATERIAL_UNIFORM_LAYOUT, pbrData, "pbrMasks", 4),
		[61, 62, 0, 0],
	);

	const legacy = { ambientShininess: [29, 30, 31, 32], specular: [33, 34, 35, 0] };
	const phongData = packPhongMaterialUniformData(legacy);
	const flatData = packFlatMaterialUniformData(legacy);
	assert.equal(phongData.length * 4, WEBGPU_PHONG_MATERIAL_UNIFORM_BYTE_SIZE);
	assert.equal(flatData.length * 4, WEBGPU_FLAT_MATERIAL_UNIFORM_BYTE_SIZE);
	assert.deepEqual(
		readVec(WEBGPU_PHONG_MATERIAL_UNIFORM_LAYOUT, phongData, "ambientShininess", 4),
		legacy.ambientShininess,
	);
	assert.deepEqual(
		readVec(WEBGPU_FLAT_MATERIAL_UNIFORM_LAYOUT, flatData, "specular", 4),
		legacy.specular,
	);
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
		enableShadows: true,
		enableSH: true,
		enableClusteredLighting: true,
		hasSHAmbient: false,
		environmentIsLinear: false,
		hasEnvSpecular: true,
		hasBRDFLUT: true,
		envSpecularMaxMipLevel: 5,
		taaJitterCurrentPrev: [43, 44, 45, 46],
	};
}

function testFrameUniformPacking() {
	const input = createFrameInput();
	input.directionalShadows = [{
		enabled: true,
		strategyType: "single-map",
		cascadeCount: 1,
		cascadeBlendRatio: 0,
		cascadeViewProjectionMatrices: [null, null, null, null],
		cascadeSplits: [[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
		depthProjectionParams: [[-1, -2, -1, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]],
		viewProjectionMatrix: matrix(500),
		depthBias: 0.01,
		slopeBias: 0.02,
		normalBias: 0.03,
		normalBiasMin: 0.04,
		filterMode: "pcss",
		samplingQuality: "high",
		shadowStrength: 0.9,
		shadowMapBaseSize: 1024,
		shadowMapSize: 1024,
		atlasTileSize: 1024,
	}];
	const cameraData = packFrameCameraUniformData(input);
	const lightData = packFrameLightUniformData(input);
	const shadowData = packFrameShadowUniformData(input);
	const environmentData = packFrameEnvironmentUniformData(input);
	assert.equal(
		WEBGPU_FRAME_CAMERA_UNIFORM_LAYOUT.byteSize,
		WEBGPU_FRAME_CAMERA_UNIFORM_BYTE_SIZE
	);
	assert.equal(
		WEBGPU_FRAME_LIGHT_UNIFORM_LAYOUT.byteSize,
		WEBGPU_FRAME_LIGHT_UNIFORM_BYTE_SIZE
	);
	assert.equal(
		WEBGPU_FRAME_SHADOW_UNIFORM_LAYOUT.byteSize,
		WEBGPU_FRAME_SHADOW_UNIFORM_BYTE_SIZE
	);
	assert.equal(
		WEBGPU_FRAME_ENVIRONMENT_UNIFORM_LAYOUT.byteSize,
		WEBGPU_FRAME_ENVIRONMENT_UNIFORM_BYTE_SIZE
	);
	assert.equal(
		cameraData.byteLength + lightData.byteLength + shadowData.byteLength + environmentData.byteLength,
		12688
	);
	assert.deepEqual(
		readVec(
			WEBGPU_FRAME_SHADOW_UNIFORM_LAYOUT,
			shadowData,
			["directionalShadows", 0, "paramsD"],
			4,
		),
		[1, 2, 0, 0],
	);
	assert.deepEqual(
		readVec(
			WEBGPU_FRAME_SHADOW_UNIFORM_LAYOUT,
			shadowData,
			["directionalShadows", 0, "depthProjectionParams", 0],
			4,
		),
		[-1, -2, -1, 0],
	);
	assert.deepEqual(
		readVec(WEBGPU_FRAME_CAMERA_UNIFORM_LAYOUT, cameraData, "cameraPosition", 4),
		[1, 2, 3, 1]
	);
	assert.deepEqual(readVec(WEBGPU_FRAME_CAMERA_UNIFORM_LAYOUT, cameraData, "environmentBasisRight", 4), [
		4, 5, 6, 7,
	]);
	assert.deepEqual(readVec(WEBGPU_FRAME_CAMERA_UNIFORM_LAYOUT, cameraData, "lightCounts", 4), [1, 1, 1, 1]);
	assert.deepEqual(readVec(WEBGPU_FRAME_CAMERA_UNIFORM_LAYOUT, cameraData, "options", 4), [1, 0, 1, 0]);
	assert.deepEqual(readVec(WEBGPU_FRAME_CAMERA_UNIFORM_LAYOUT, cameraData, "environmentOptionsA", 4), [
		1, 0, 0, 1,
	]);
	assert.deepEqual(readVec(WEBGPU_FRAME_CAMERA_UNIFORM_LAYOUT, cameraData, "environmentOptionsB", 4), [
		1, 5, 0, 1,
	]);
	assert.deepEqual(readVec(WEBGPU_FRAME_LIGHT_UNIFORM_LAYOUT, lightData, ["directionalLights", 0, "direction"], 4), [
		18, 19, 20, 0,
	]);
	assert.deepEqual(readVec(WEBGPU_FRAME_LIGHT_UNIFORM_LAYOUT, lightData, ["pointLights", 0, "positionRange"], 4), [
		24, 25, 26, 27,
	]);
	assert.deepEqual(readVec(WEBGPU_FRAME_LIGHT_UNIFORM_LAYOUT, lightData, ["spotLights", 0, "colorInner"], 4), [
		39, 40, 41, 42,
	]);
	assert.deepEqual(
		readVec(WEBGPU_FRAME_ENVIRONMENT_UNIFORM_LAYOUT, environmentData, "localLightProbeCounts", 4),
		[0, 2, 0, 0]
	);
	assert.deepEqual(readVec(WEBGPU_FRAME_LIGHT_UNIFORM_LAYOUT, lightData, ["areaLights", 0, "positionRange"], 4), [
		47, 48, 49, 50,
	]);
	assert.deepEqual(readVec(WEBGPU_FRAME_LIGHT_UNIFORM_LAYOUT, lightData, ["areaLights", 0, "normalAreaScale"], 4), [
		59, 60, 61, 62,
	]);
	assert.deepEqual(
		readVec(WEBGPU_FRAME_ENVIRONMENT_UNIFORM_LAYOUT, environmentData, ["reflectionProbes", 0, "worldToProbeRow0"], 4),
		[1000, 1001, 1002, 1003]
	);
	assert.deepEqual(
		readVec(WEBGPU_FRAME_ENVIRONMENT_UNIFORM_LAYOUT, environmentData, ["reflectionProbes", 1, "probeToWorldRow2"], 4),
		[2108, 2109, 2110, 2111]
	);
	assert.deepEqual(readVec(WEBGPU_FRAME_ENVIRONMENT_UNIFORM_LAYOUT, environmentData, ["reflectionProbes", 1, "dataC"], 4), [
		1, 9, 10, 11,
	]);
	assert.deepEqual(readVec(WEBGPU_FRAME_ENVIRONMENT_UNIFORM_LAYOUT, environmentData, "irradianceProbeGridWorldToGridRow1", 4), [
		3004, 3005, 3006, 3007,
	]);
	assert.deepEqual(readVec(WEBGPU_FRAME_ENVIRONMENT_UNIFORM_LAYOUT, environmentData, "irradianceProbeGridDataA", 4), [
		3, 4, 5, 60,
	]);
	assert.deepEqual(readVec(WEBGPU_FRAME_ENVIRONMENT_UNIFORM_LAYOUT, environmentData, "irradianceProbeGridDataB", 4), [
		0.25, 0.5, 0.75, 12,
	]);
	assert.deepEqual(readVec(WEBGPU_FRAME_ENVIRONMENT_UNIFORM_LAYOUT, environmentData, "irradianceProbeGridDataC", 4), [
		1, WEBGPU_SH_COEFFICIENT_COUNT, 60, 0,
	]);
	assert.equal(
		readVec(WEBGPU_FRAME_SHADOW_UNIFORM_LAYOUT, shadowData, ["directionalShadows", 0, "paramsA"], 4)[0],
		1
	);
}

testSeparatedMaterialUniformPacking();
testFrameUniformPacking();
console.log("WebGPU packing tests passed");
