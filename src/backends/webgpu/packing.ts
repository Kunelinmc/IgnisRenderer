import type { Vec4Tuple } from "../../maths/Vector4";
import { Matrix4 } from "../../maths/Matrix4";
import type { Matrix3Arr } from "../../maths/types";
import {
	SHADOW_FILTER_MODE_CODE,
	SHADOW_QUALITY_CODE,
} from "../../lights/shadows/shadowSampling";
import {
	MAX_AREA_LIGHTS,
	MAX_DIRECTIONAL_LIGHTS,
	MAX_LOCAL_LIGHT_PROBES,
	MAX_POINT_LIGHTS,
	MAX_REFLECTION_PROBES,
	MAX_SPOT_LIGHTS,
} from "../constants";

import {
	WEBGPU_SH_COEFFICIENT_COUNT,
	WEBGPU_TEXTURE_SLOT_COUNT,
} from "./constants";
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
} from "./bufferLayouts";
import type { StructuredBufferLayout } from "./StructuredBufferLayout";
import {
	arrayStruct as packArrayStruct,
	arrayVec4 as packArrayVec4,
	createStructuredBufferPacker,
	custom as packCustom,
	mat4 as packMat4,
	vec4 as packVec4,
} from "./StructuredBufferPacker";
import type {
	WebGPUAreaLightUniform,
	WebGPUDirectionalLightUniform,
	WebGPUFrameUniformInput,
	WebGPUFlatMaterialUniformData,
	WebGPUMaterialCommonUniformData,
	WebGPUPBRMaterialUniformData,
	WebGPUPhongMaterialUniformData,
	WebGPUPointLightUniform,
	WebGPUReflectionProbeUniform,
	WebGPUSpotLightUniform,
} from "./types";

export {
	WEBGPU_FRAME_CAMERA_UNIFORM_BYTE_SIZE,
	WEBGPU_FRAME_ENVIRONMENT_UNIFORM_BYTE_SIZE,
	WEBGPU_FRAME_LIGHT_UNIFORM_BYTE_SIZE,
	WEBGPU_FRAME_SHADOW_UNIFORM_BYTE_SIZE,
	WEBGPU_FLAT_MATERIAL_UNIFORM_BYTE_SIZE,
	WEBGPU_MATERIAL_COMMON_UNIFORM_BYTE_SIZE,
	WEBGPU_OBJECT_UNIFORM_BYTE_SIZE,
	WEBGPU_PBR_MATERIAL_UNIFORM_BYTE_SIZE,
	WEBGPU_PHONG_MATERIAL_UNIFORM_BYTE_SIZE,
} from "./constants";

interface WebGPUObjectUniformInput {
	modelMatrix: Matrix4 | number[][];
	normalMatrix: Matrix3Arr | Matrix4;
	prevModelMatrix: Matrix4 | number[][];
	renderLayers: number;
	receiveShadows: boolean;
	staticBatch: boolean;
}

const FRAME_CAMERA_UNIFORM_PACKER = createStructuredBufferPacker<
	WebGPUFrameUniformInput,
	"float32Array"
>({
	label: "FrameCameraUniforms",
	layout: WEBGPU_FRAME_CAMERA_UNIFORM_LAYOUT,
	output: "float32Array",
	fields: [
		packMat4("viewProjection", (input) => input.viewProjectionMatrix),
		packMat4("prevViewProjection", (input) => input.prevViewProjectionMatrix),
		packVec4("cameraPosition", (input) => [
			input.cameraPosition.x,
			input.cameraPosition.y,
			input.cameraPosition.z,
			1,
		]),
		packVec4("environmentBasisRight", (input) => [
			input.environmentRight[0],
			input.environmentRight[1],
			input.environmentRight[2],
			input.environmentTanHalfFov,
		]),
		packVec4("environmentBasisUp", (input) => [
			input.environmentUp[0],
			input.environmentUp[1],
			input.environmentUp[2],
			input.environmentAspect,
		]),
		packVec4("environmentBasisBackward", (input) => [
			input.environmentBackward[0],
			input.environmentBackward[1],
			input.environmentBackward[2],
			input.environmentIsOrthographic ? 1 : 0,
		]),
		packVec4("ambientColor", (input) => [
			input.ambientColor[0],
			input.ambientColor[1],
			input.ambientColor[2],
			1,
		]),
		packVec4("lightCounts", (input) => [
			input.directionalLights.length,
			input.pointLights.length,
			input.spotLights.length,
			input.areaLights.length,
		]),
		packVec4("options", (input) => [
			input.enableLighting ? 1 : 0,
			0, // Reserved to preserve the published frame uniform layout.
			input.enableShadows ? 1 : 0,
			0, // Reserved to keep post-process state out of scene uniforms.
		]),
		packVec4("environmentOptionsA", (input) => [
			input.enableSH ? 1 : 0,
			input.hasSHAmbient ? 1 : 0,
			0, // Reserved to preserve the published frame uniform layout.
			input.hasEnvSpecular ? 1 : 0,
		]),
		packVec4("environmentOptionsB", (input) => [
			input.hasBRDFLUT ? 1 : 0,
			Math.max(0, input.envSpecularMaxMipLevel),
			input.environmentIsLinear ? 1 : 0,
			input.enableClusteredLighting ? 1 : 0,
		]),
		packVec4("taaJitterCurrentPrev", (input) => input.taaJitterCurrentPrev),
	],
});

const FRAME_LIGHT_UNIFORM_PACKER = createStructuredBufferPacker<
	WebGPUFrameUniformInput,
	"float32Array"
>({
	label: "FrameLightUniforms",
	layout: WEBGPU_FRAME_LIGHT_UNIFORM_LAYOUT,
	output: "float32Array",
	fields: [
		packArrayStruct<WebGPUFrameUniformInput, WebGPUDirectionalLightUniform>(
			"directionalLights",
			MAX_DIRECTIONAL_LIGHTS,
			(input, index) => input.directionalLights[index],
			[
				packVec4("direction", (light) => [
					light.direction[0],
					light.direction[1],
					light.direction[2],
					0,
				]),
				packVec4("color", (light) => [
					light.color[0],
					light.color[1],
					light.color[2],
					0,
				]),
			]
		),
		packArrayStruct<WebGPUFrameUniformInput, WebGPUPointLightUniform>(
			"pointLights",
			MAX_POINT_LIGHTS,
			(input, index) => input.pointLights[index],
			[
				packVec4("positionRange", (light) => [
					light.position[0],
					light.position[1],
					light.position[2],
					light.range,
				]),
				packVec4("color", (light) => [
					light.color[0],
					light.color[1],
					light.color[2],
					0,
				]),
			]
		),
		packArrayStruct<WebGPUFrameUniformInput, WebGPUSpotLightUniform>(
			"spotLights",
			MAX_SPOT_LIGHTS,
			(input, index) => input.spotLights[index],
			[
				packVec4("positionRange", (light) => [
					light.position[0],
					light.position[1],
					light.position[2],
					light.range,
				]),
				packVec4("directionOuter", (light) => [
					light.direction[0],
					light.direction[1],
					light.direction[2],
					light.outerCos,
				]),
				packVec4("colorInner", (light) => [
					light.color[0],
					light.color[1],
					light.color[2],
					light.innerCos,
				]),
			]
		),
		packArrayStruct<WebGPUFrameUniformInput, WebGPUAreaLightUniform>(
			"areaLights",
			MAX_AREA_LIGHTS,
			(input, index) => input.areaLights[index],
			[
				packVec4("positionRange", (light) => [
					light.position[0],
					light.position[1],
					light.position[2],
					light.range,
				]),
				packVec4("rightWidth", (light) => [
					light.right[0],
					light.right[1],
					light.right[2],
					light.width,
				]),
				packVec4("upHeight", (light) => [
					light.up[0],
					light.up[1],
					light.up[2],
					light.height,
				]),
				packVec4("normalAreaScale", (light) => [
					light.normal[0],
					light.normal[1],
					light.normal[2],
					light.areaScale,
				]),
				packVec4("color", (light) => [
					light.color[0],
					light.color[1],
					light.color[2],
					0,
				]),
			]
		),
	],
});

const FRAME_SHADOW_UNIFORM_PACKER = createStructuredBufferPacker<
	WebGPUFrameUniformInput,
	"float32Array"
>({
	label: "FrameShadowUniforms",
	layout: WEBGPU_FRAME_SHADOW_UNIFORM_LAYOUT,
	output: "float32Array",
	fields: [
		packCustom("directionalShadows", (writer, input) => {
			for (let i = 0; i < MAX_DIRECTIONAL_LIGHTS; i++) {
				writeShadowData(writer, "directionalShadows", i, input.directionalShadows[i]);
			}
		}),
		packCustom("spotShadows", (writer, input) => {
			for (let i = 0; i < MAX_SPOT_LIGHTS; i++) {
				writeShadowData(writer, "spotShadows", i, input.spotShadows[i]);
			}
		}),
	],
});

const FRAME_ENVIRONMENT_UNIFORM_PACKER = createStructuredBufferPacker<
	WebGPUFrameUniformInput,
	"float32Array"
>({
	label: "FrameEnvironmentUniforms",
	layout: WEBGPU_FRAME_ENVIRONMENT_UNIFORM_LAYOUT,
	output: "float32Array",
	fields: [
		packArrayVec4("shAmbientCoeffs", WEBGPU_SH_COEFFICIENT_COUNT, (input, i) => {
			const coefficient = input.shAmbientCoeffs?.[i];
			return coefficient ? [coefficient.r, coefficient.g, coefficient.b, 0] : null;
		}),
		packArrayStruct<WebGPUFrameUniformInput, WebGPUReflectionProbeUniform>(
			"reflectionProbes",
			MAX_REFLECTION_PROBES,
			(input, index) => input.reflectionProbes[index],
			[
				packVec4("worldToProbeRow0", (probe) =>
					matrixRow(probe.worldToProbeMatrix, 0)
				),
				packVec4("worldToProbeRow1", (probe) =>
					matrixRow(probe.worldToProbeMatrix, 1)
				),
				packVec4("worldToProbeRow2", (probe) =>
					matrixRow(probe.worldToProbeMatrix, 2)
				),
				packVec4("probeToWorldRow0", (probe) =>
					matrixRow(probe.probeToWorldMatrix, 0)
				),
				packVec4("probeToWorldRow1", (probe) =>
					matrixRow(probe.probeToWorldMatrix, 1)
				),
				packVec4("probeToWorldRow2", (probe) =>
					matrixRow(probe.probeToWorldMatrix, 2)
				),
				packVec4("dataA", (probe) => [
					probe.invHalfExtents[0],
					probe.invHalfExtents[1],
					probe.invHalfExtents[2],
					probe.radiusInv,
				]),
				packVec4("dataB", (probe) => [
					probe.captureWorldPosition[0],
					probe.captureWorldPosition[1],
					probe.captureWorldPosition[2],
					probe.shape,
				]),
				packVec4("dataC", (probe) => [
					probe.parallaxMode,
					probe.blendDistance,
					probe.blendExponent,
					probe.layer,
				]),
			]
		),
		packVec4("localLightProbeCounts", (input) => [
			Math.max(0, input.localLightProbeCount),
			Math.max(0, input.reflectionProbeCount),
			0,
			0,
		]),
		packArrayVec4(
			"localLightProbeWorldToProbeRow0",
			MAX_LOCAL_LIGHT_PROBES,
			(input, i) => {
				const probe = input.localLightProbes[i];
				return probe ? matrixRow(probe.worldToProbeMatrix, 0) : null;
			}
		),
		packArrayVec4(
			"localLightProbeWorldToProbeRow1",
			MAX_LOCAL_LIGHT_PROBES,
			(input, i) => {
				const probe = input.localLightProbes[i];
				return probe ? matrixRow(probe.worldToProbeMatrix, 1) : null;
			}
		),
		packArrayVec4(
			"localLightProbeWorldToProbeRow2",
			MAX_LOCAL_LIGHT_PROBES,
			(input, i) => {
				const probe = input.localLightProbes[i];
				return probe ? matrixRow(probe.worldToProbeMatrix, 2) : null;
			}
		),
		packArrayVec4(
			"localLightProbeDataA",
			MAX_LOCAL_LIGHT_PROBES,
			(input, i) => {
				const probe = input.localLightProbes[i];
				return probe ?
						[
							probe.invHalfExtents[0],
							probe.invHalfExtents[1],
							probe.invHalfExtents[2],
							probe.radiusInv,
						]
					:	null;
			}
		),
		packArrayVec4(
			"localLightProbeDataB",
			MAX_LOCAL_LIGHT_PROBES,
			(input, i) => {
				const probe = input.localLightProbes[i];
				return probe ?
						[
							probe.blendDistance,
							probe.priority,
							probe.shape,
							0,
						]
					:	null;
			}
		),
		packArrayVec4(
			"localLightProbeSHAmbientCoeffs",
			MAX_LOCAL_LIGHT_PROBES * WEBGPU_SH_COEFFICIENT_COUNT,
			(input, i) => {
				const probeIndex = Math.floor(i / WEBGPU_SH_COEFFICIENT_COUNT);
				const coefficientIndex = i % WEBGPU_SH_COEFFICIENT_COUNT;
				const coefficient =
					input.localLightProbes[probeIndex]?.sh[coefficientIndex];
				return coefficient ? [coefficient.r, coefficient.g, coefficient.b, 0] : null;
			}
		),
		packVec4("irradianceProbeGridWorldToGridRow0", (input) =>
			input.irradianceProbeGrid ?
				matrixRow(input.irradianceProbeGrid.worldToGridMatrix, 0)
			:	[1, 0, 0, 0]
		),
		packVec4("irradianceProbeGridWorldToGridRow1", (input) =>
			input.irradianceProbeGrid ?
				matrixRow(input.irradianceProbeGrid.worldToGridMatrix, 1)
			:	[0, 1, 0, 0]
		),
		packVec4("irradianceProbeGridWorldToGridRow2", (input) =>
			input.irradianceProbeGrid ?
				matrixRow(input.irradianceProbeGrid.worldToGridMatrix, 2)
			:	[0, 0, 1, 0]
		),
		packVec4("irradianceProbeGridDataA", (input) => {
			const grid = input.irradianceProbeGrid;
			return grid ?
					[
						grid.dimensions[0],
						grid.dimensions[1],
						grid.dimensions[2],
						grid.cellCount,
					]
				:	[1, 1, 1, 0];
		}),
		packVec4("irradianceProbeGridDataB", (input) => {
			const grid = input.irradianceProbeGrid;
			return grid ?
					[
						grid.invHalfExtents[0],
						grid.invHalfExtents[1],
						grid.invHalfExtents[2],
						grid.blendDistance,
					]
				:	[1, 1, 1, 0.01];
		}),
		packVec4("irradianceProbeGridDataC", (input) => {
			const grid = input.irradianceProbeGrid;
			return grid ? [1, WEBGPU_SH_COEFFICIENT_COUNT, grid.cellCount, 0] : [0, 16, 1, 0];
		}),
	],
});

const OBJECT_UNIFORM_PACKER = createStructuredBufferPacker<
	WebGPUObjectUniformInput,
	"float32Array"
>({
	label: "ObjectUniforms",
	layout: WEBGPU_OBJECT_UNIFORM_LAYOUT,
	output: "float32Array",
	fields: [
		packMat4("modelMatrix", (input) => input.modelMatrix),
		packMat4("prevModelMatrix", (input) => input.prevModelMatrix),
		packMat4("normalMatrix", (input) =>
			createNormalMatrixRows(input.normalMatrix)
		),
		packVec4("instanceData", (input) => [
			Math.max(0, Math.floor(input.renderLayers)) >>> 0,
			input.receiveShadows ? 1 : 0,
			input.staticBatch ? 1 : 0,
			0,
		]),
	],
});

const MATERIAL_COMMON_UNIFORM_PACKER = createStructuredBufferPacker<
	WebGPUMaterialCommonUniformData,
	"float32Array"
>({
	label: "MaterialCommonUniforms",
	layout: WEBGPU_MATERIAL_COMMON_UNIFORM_LAYOUT,
	output: "float32Array",
	fields: [
		packVec4("baseColorFactor", (input) => input.baseColorFactor),
		packVec4("emissiveFactor", (input) => input.emissiveFactor),
		packVec4("materialParams", (input) => input.materialParams),
		packVec4("renderParams", (input) => input.renderParams),
		packArrayVec4("textureTransformA", WEBGPU_TEXTURE_SLOT_COUNT, (input, i) =>
			input.textureSlots[i]?.transformA
		),
		packArrayVec4("textureTransformB", WEBGPU_TEXTURE_SLOT_COUNT, (input, i) =>
			input.textureSlots[i]?.transformB
		),
	],
});

const PBR_MATERIAL_UNIFORM_PACKER = createStructuredBufferPacker<
	WebGPUPBRMaterialUniformData,
	"float32Array"
>({
	label: "PBRMaterialUniforms",
	layout: WEBGPU_PBR_MATERIAL_UNIFORM_LAYOUT,
	output: "float32Array",
	fields: [
		packVec4("surfaceParams0", (input) => input.surfaceParams0),
		packVec4("surfaceParams1", (input) => input.surfaceParams1),
		packVec4("surfaceParams2", (input) => input.surfaceParams2),
		packVec4("surfaceParams3", (input) => input.surfaceParams3),
		packVec4("specularColorFactor", (input) => input.specularColorFactor),
		packVec4(
			"sheenColorClearcoatNormalScale",
			(input) => input.sheenColorClearcoatNormalScale
		),
		packVec4("attenuationColor", (input) => input.attenuationColor),
		packVec4("anisotropyParams", (input) => input.anisotropyParams),
		packVec4("pbrMasks", (input) => input.pbrMasks),
	],
});

const PHONG_MATERIAL_UNIFORM_PACKER = createStructuredBufferPacker<
	WebGPUPhongMaterialUniformData,
	"float32Array"
>({
	label: "PhongMaterialUniforms",
	layout: WEBGPU_PHONG_MATERIAL_UNIFORM_LAYOUT,
	output: "float32Array",
	fields: [
		packVec4("ambientShininess", (input) => input.ambientShininess),
		packVec4("specular", (input) => input.specular),
	],
});

const FLAT_MATERIAL_UNIFORM_PACKER = createStructuredBufferPacker<
	WebGPUFlatMaterialUniformData,
	"float32Array"
>({
	label: "FlatMaterialUniforms",
	layout: WEBGPU_FLAT_MATERIAL_UNIFORM_LAYOUT,
	output: "float32Array",
	fields: [
		packVec4("ambientShininess", (input) => input.ambientShininess),
		packVec4("specular", (input) => input.specular),
	],
});

export function packMatrix4ForWGSL(matrix: Matrix4 | number[][]): Float32Array {
	return Matrix4.toColumnMajorArray(matrix);
}

export function packNormalMatrix4ForWGSL(
	normalMatrix: Matrix3Arr | Matrix4
): Float32Array {
	return packMatrix4ForWGSL(createNormalMatrixRows(normalMatrix));
}

export function packFrameCameraUniformData(
	input: WebGPUFrameUniformInput
): Float32Array {
	return FRAME_CAMERA_UNIFORM_PACKER.pack(input);
}

export function packFrameLightUniformData(
	input: WebGPUFrameUniformInput
): Float32Array {
	return FRAME_LIGHT_UNIFORM_PACKER.pack(input);
}

export function packFrameShadowUniformData(
	input: WebGPUFrameUniformInput
): Float32Array {
	return FRAME_SHADOW_UNIFORM_PACKER.pack(input);
}

export function packFrameEnvironmentUniformData(
	input: WebGPUFrameUniformInput
): Float32Array {
	return FRAME_ENVIRONMENT_UNIFORM_PACKER.pack(input);
}

export function remapClipSpaceDepth(clipZ: number, clipW: number): number {
	return clipZ * 0.5 + clipW * 0.5;
}

export function packObjectUniformData(
	modelMatrix: Matrix4 | number[][],
	normalMatrix: Matrix3Arr | Matrix4,
	prevModelMatrix: Matrix4 | number[][],
	renderLayers = 1,
	receiveShadows = true,
	staticBatch = false,
): Float32Array<ArrayBuffer> {
	return OBJECT_UNIFORM_PACKER.pack({
		modelMatrix,
		normalMatrix,
		prevModelMatrix,
		renderLayers,
		receiveShadows,
		staticBatch,
	});
}

export type WebGPUObjectUniformWriter = ReturnType<
	StructuredBufferLayout["createWriter"]
>;

/**
 * Creates a reusable writer for `ObjectUniforms` data.
 *
 * @returns A zero-initialized structured buffer writer matching the WebGPU
 * model uniform layout.
 */
export function createObjectUniformWriter(): WebGPUObjectUniformWriter {
	return OBJECT_UNIFORM_PACKER.createWriter();
}

/**
 * Writes `ObjectUniforms` into an existing writer and returns its typed view.
 *
 * @param writer - Reusable writer created by `createObjectUniformWriter`.
 * @param modelMatrix - Current model transform in engine row-major layout.
 * @param normalMatrix - Current normal matrix; only its upper-left 3x3 is used.
 * @param prevModelMatrix - Previous-frame model transform for motion vectors.
 * @param renderLayers - Unsigned render-layer mask for this draw packet.
 * @param receiveShadows - Whether this object samples shadow visibility.
 * @returns The writer-owned `Float32Array`; callers must consume it before
 * reusing the same writer.
 */
export function writeObjectUniformData(
	writer: WebGPUObjectUniformWriter,
	modelMatrix: Matrix4 | number[][],
	normalMatrix: Matrix3Arr | Matrix4,
	prevModelMatrix: Matrix4 | number[][],
	renderLayers = 1,
	receiveShadows = true,
	staticBatch = false,
): Float32Array<ArrayBuffer> {
	return OBJECT_UNIFORM_PACKER.packInto(writer, {
		modelMatrix,
		normalMatrix,
		prevModelMatrix,
		renderLayers,
		receiveShadows,
		staticBatch,
	});
}

export function packMaterialCommonUniformData(
	input: WebGPUMaterialCommonUniformData,
): Float32Array<ArrayBuffer> {
	return MATERIAL_COMMON_UNIFORM_PACKER.pack(input);
}

export function packPBRMaterialUniformData(
	input: WebGPUPBRMaterialUniformData,
): Float32Array<ArrayBuffer> {
	return PBR_MATERIAL_UNIFORM_PACKER.pack(input);
}

export function packPhongMaterialUniformData(
	input: WebGPUPhongMaterialUniformData,
): Float32Array<ArrayBuffer> {
	return PHONG_MATERIAL_UNIFORM_PACKER.pack(input);
}

export function packFlatMaterialUniformData(
	input: WebGPUFlatMaterialUniformData,
): Float32Array<ArrayBuffer> {
	return FLAT_MATERIAL_UNIFORM_PACKER.pack(input);
}

function resolveMatrixRows(matrix: Matrix4 | number[][]): number[][] {
	return matrix instanceof Matrix4 ? matrix.elements : matrix;
}

function createNormalMatrixRows(normalMatrix: Matrix3Arr | Matrix4): number[][] {
	const rows =
		normalMatrix instanceof Matrix4 ? normalMatrix.elements : normalMatrix;

	return [
		[rows[0][0], rows[0][1], rows[0][2], 0],
		[rows[1][0], rows[1][1], rows[1][2], 0],
		[rows[2][0], rows[2][1], rows[2][2], 0],
		[0, 0, 0, 1],
	];
}

function matrixRow(
	matrix: Matrix4,
	row: 0 | 1 | 2
): Vec4Tuple {
	const elements = matrix.elements;
	return [
		elements[row][0],
		elements[row][1],
		elements[row][2],
		elements[row][3],
	];
}

function writeShadowData(
	writer: ReturnType<StructuredBufferLayout["createWriter"]>,
	arrayField: "directionalShadows" | "spotShadows",
	index: number,
	shadow: WebGPUFrameUniformInput["directionalShadows"][number] | undefined
): void {
	if (shadow?.enabled && shadow.viewProjectionMatrix) {
		writer.writeMat4([arrayField, index, "viewProjection"], shadow.viewProjectionMatrix);
	}

	for (let cascadeIndex = 0; cascadeIndex < 4; cascadeIndex++) {
		const cascadeMatrix =
			shadow?.enabled ? shadow.cascadeViewProjectionMatrices[cascadeIndex] : null;
		if (cascadeMatrix) {
			writer.writeMat4(
				[arrayField, index, "cascadeViewProjections", cascadeIndex],
				cascadeMatrix
			);
		}
	}

	for (let cascadeIndex = 0; cascadeIndex < 4; cascadeIndex++) {
		const split = shadow?.cascadeSplits?.[cascadeIndex];
		if (split) {
			writer.writeVec([arrayField, index, "cascadeSplits", cascadeIndex], split);
		}
	}

	for (let cascadeIndex = 0; cascadeIndex < 4; cascadeIndex++) {
		const params = shadow?.depthProjectionParams?.[cascadeIndex];
		if (params) {
			writer.writeVec(
				[arrayField, index, "depthProjectionParams", cascadeIndex],
				params,
			);
		}
	}

	const isCSM =
		shadow?.enabled && shadow.strategyType === "csm" && shadow.cascadeCount > 1;
	const cascadeCount =
		isCSM ? Math.max(1, Math.min(4, shadow?.cascadeCount ?? 1)) : 1;
	const cascadeBlendRatio =
		isCSM ? Math.max(0, Math.min(1, shadow?.cascadeBlendRatio ?? 0)) : 0;

	writer.writeVec([arrayField, index, "paramsA"], [
		shadow?.enabled ? 1 : 0,
		shadow?.depthBias ?? 0,
		shadow?.normalBias ?? 0,
		shadow?.normalBiasMin ?? 0,
	]);
	writer.writeVec([arrayField, index, "paramsB"], [
		0,
		shadow?.shadowStrength ?? 0,
		shadow?.shadowMapSize ?? 0,
		shadow?.atlasTileSize ?? 0,
	]);
	writer.writeVec([arrayField, index, "paramsC"], [
		shadow?.slopeBias ?? 0,
		isCSM ? 1 : 0,
		cascadeCount,
		cascadeBlendRatio,
	]);
	writer.writeVec([arrayField, index, "paramsD"], [
		shadow ? SHADOW_FILTER_MODE_CODE[shadow.filterMode] : 0,
		shadow ? SHADOW_QUALITY_CODE[shadow.samplingQuality] : 1,
		0,
		0,
	]);
	writer.writeVec([arrayField, index, "paramsE"], [0, 0, 0, 0]);
	writer.writeVec([arrayField, index, "paramsF"], [0, 0, 0, 0]);
}
