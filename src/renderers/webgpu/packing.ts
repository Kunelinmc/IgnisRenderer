import { Matrix4 } from "../../maths/Matrix4";
import type { Matrix3Arr } from "../../maths/types";

import {
	WEBGPU_MAX_AREA_LIGHTS,
	WEBGPU_MAX_DIRECTIONAL_LIGHTS,
	WEBGPU_MAX_LOCAL_LIGHT_PROBES,
	WEBGPU_MAX_POINT_LIGHTS,
	WEBGPU_MAX_REFLECTION_PROBES,
	WEBGPU_MAX_SPOT_LIGHTS,
	WEBGPU_SH_COEFFICIENT_COUNT,
	WEBGPU_TEXTURE_SLOT_COUNT,
} from "./constants";
import {
	WEBGPU_FRAME_UNIFORM_LAYOUT as FRAME_UNIFORM_LAYOUT,
	WEBGPU_MODEL_UNIFORM_LAYOUT as MODEL_UNIFORM_LAYOUT,
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
	WebGPUMaterialUniformData,
	WebGPUPointLightUniform,
	WebGPUReflectionProbeUniform,
	WebGPUSpotLightUniform,
} from "./types";

export {
	WEBGPU_FRAME_UNIFORM_BYTE_SIZE,
	WEBGPU_MODEL_UNIFORM_BYTE_SIZE,
} from "./constants";

interface WebGPUModelUniformInput {
	modelMatrix: Matrix4 | number[][];
	normalMatrix: Matrix3Arr | Matrix4;
	materialData: WebGPUMaterialUniformData;
	prevModelMatrix: Matrix4 | number[][];
	renderLayers: number;
}

const FRAME_UNIFORM_PACKER = createStructuredBufferPacker<
	WebGPUFrameUniformInput,
	"float32Array"
>({
	label: "FrameUniforms",
	layout: FRAME_UNIFORM_LAYOUT,
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
			input.reflectionProbeCount,
		]),
		packVec4("options", (input) => [
			input.enableLighting ? 1 : 0,
			input.enableGamma ? 1 : 0,
			input.enableShadows ? 1 : 0,
			input.encodeGammaInShader ? 1 : 0,
		]),
		packVec4("environmentOptionsA", (input) => [
			input.enableSH ? 1 : 0,
			input.hasSHAmbient ? 1 : 0,
			input.hasEnvSpecularFallback ?
				Math.max(0, input.envSpecularFallbackMaxMipLevel) + 1
			:	0,
			input.hasEnvSpecular ? 1 : 0,
		]),
		packVec4("environmentOptionsB", (input) => [
			input.hasBRDFLUT ? 1 : 0,
			Math.max(0, input.envSpecularMaxMipLevel),
			input.environmentIsLinear ? 1 : 0,
			input.enableClusteredLighting ? 1 : 0,
		]),
		packVec4("taaJitterCurrentPrev", (input) => input.taaJitterCurrentPrev),
		packArrayStruct<WebGPUFrameUniformInput, WebGPUDirectionalLightUniform>(
			"directionalLights",
			WEBGPU_MAX_DIRECTIONAL_LIGHTS,
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
			WEBGPU_MAX_POINT_LIGHTS,
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
			WEBGPU_MAX_SPOT_LIGHTS,
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
		packCustom("directionalShadows", (writer, input) => {
			for (let i = 0; i < WEBGPU_MAX_DIRECTIONAL_LIGHTS; i++) {
				writeShadowData(writer, "directionalShadows", i, input.directionalShadows[i]);
			}
		}),
		packCustom("spotShadows", (writer, input) => {
			for (let i = 0; i < WEBGPU_MAX_SPOT_LIGHTS; i++) {
				writeShadowData(writer, "spotShadows", i, input.spotShadows[i]);
			}
		}),
		packArrayVec4("shAmbientCoeffs", WEBGPU_SH_COEFFICIENT_COUNT, (input, i) => {
			const coefficient = input.shAmbientCoeffs?.[i];
			return coefficient ? [coefficient.r, coefficient.g, coefficient.b, 0] : null;
		}),
		packArrayStruct<WebGPUFrameUniformInput, WebGPUReflectionProbeUniform>(
			"reflectionProbes",
			WEBGPU_MAX_REFLECTION_PROBES,
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
			0,
			0,
			0,
		]),
		packArrayVec4(
			"localLightProbeWorldToProbeRow0",
			WEBGPU_MAX_LOCAL_LIGHT_PROBES,
			(input, i) => {
				const probe = input.localLightProbes[i];
				return probe ? matrixRow(probe.worldToProbeMatrix, 0) : null;
			}
		),
		packArrayVec4(
			"localLightProbeWorldToProbeRow1",
			WEBGPU_MAX_LOCAL_LIGHT_PROBES,
			(input, i) => {
				const probe = input.localLightProbes[i];
				return probe ? matrixRow(probe.worldToProbeMatrix, 1) : null;
			}
		),
		packArrayVec4(
			"localLightProbeWorldToProbeRow2",
			WEBGPU_MAX_LOCAL_LIGHT_PROBES,
			(input, i) => {
				const probe = input.localLightProbes[i];
				return probe ? matrixRow(probe.worldToProbeMatrix, 2) : null;
			}
		),
		packArrayVec4(
			"localLightProbeDataA",
			WEBGPU_MAX_LOCAL_LIGHT_PROBES,
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
			WEBGPU_MAX_LOCAL_LIGHT_PROBES,
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
			WEBGPU_MAX_LOCAL_LIGHT_PROBES * WEBGPU_SH_COEFFICIENT_COUNT,
			(input, i) => {
				const probeIndex = Math.floor(i / WEBGPU_SH_COEFFICIENT_COUNT);
				const coefficientIndex = i % WEBGPU_SH_COEFFICIENT_COUNT;
				const coefficient =
					input.localLightProbes[probeIndex]?.sh[coefficientIndex];
				return coefficient ? [coefficient.r, coefficient.g, coefficient.b, 0] : null;
			}
		),
		packVec4("areaLightCounts", (input) => [
			Math.min(input.areaLights.length, WEBGPU_MAX_AREA_LIGHTS),
			0,
			0,
			0,
		]),
		packArrayStruct<WebGPUFrameUniformInput, WebGPUAreaLightUniform>(
			"areaLights",
			WEBGPU_MAX_AREA_LIGHTS,
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

const MODEL_UNIFORM_PACKER = createStructuredBufferPacker<
	WebGPUModelUniformInput,
	"float32Array"
>({
	label: "ModelUniforms",
	layout: MODEL_UNIFORM_LAYOUT,
	output: "float32Array",
	fields: [
		packMat4("modelMatrix", (input) => input.modelMatrix),
		packMat4("prevModelMatrix", (input) => input.prevModelMatrix),
		packMat4("normalMatrix", (input) =>
			createNormalMatrixRows(input.normalMatrix)
		),
		packVec4("baseColorFactor", (input) => input.materialData.baseColorFactor),
		packVec4("emissiveFactor", (input) => input.materialData.emissiveFactor),
		packVec4("surfaceParams0", (input) => input.materialData.surfaceParams0),
		packVec4("surfaceParams1", (input) => input.materialData.surfaceParams1),
		packVec4("surfaceParams2", (input) => input.materialData.surfaceParams2),
		packVec4("surfaceParams3", (input) => input.materialData.surfaceParams3),
		packVec4(
			"specularColorFactor",
			(input) => input.materialData.specularColorFactor
		),
		packVec4(
			"phongAmbientShininess",
			(input) => input.materialData.phongAmbientShininess
		),
		packVec4(
			"phongSpecularShading",
			(input) => input.materialData.phongSpecularShading
		),
		packVec4(
			"sheenColorClearcoatNormalScale",
			(input) => input.materialData.sheenColorClearcoatNormalScale
		),
		packVec4("attenuationColor", (input) => input.materialData.attenuationColor),
		packVec4("anisotropyParams", (input) => input.materialData.anisotropyParams),
		packVec4(
			"anisotropyTextureTransformA",
			(input) => input.materialData.anisotropyTexture.transformA
		),
		packVec4(
			"anisotropyTextureTransformB",
			(input) => input.materialData.anisotropyTexture.transformB
		),
		packVec4("materialFlags", (input) => input.materialData.materialFlags),
		packVec4("nodeRenderLayers", (input) => [
			Math.max(0, Math.floor(input.renderLayers)) >>> 0,
			0,
			0,
			0,
		]),
		packArrayVec4("textureTransformA", WEBGPU_TEXTURE_SLOT_COUNT, (input, i) =>
			input.materialData.textureSlots[i]?.transformA
		),
		packArrayVec4("textureTransformB", WEBGPU_TEXTURE_SLOT_COUNT, (input, i) =>
			input.materialData.textureSlots[i]?.transformB
		),
	],
});

export function packMatrix4ForWGSL(matrix: Matrix4 | number[][]): Float32Array {
	const elements = resolveMatrixRows(matrix);

	return new Float32Array([
		elements[0][0],
		elements[1][0],
		elements[2][0],
		elements[3][0],
		elements[0][1],
		elements[1][1],
		elements[2][1],
		elements[3][1],
		elements[0][2],
		elements[1][2],
		elements[2][2],
		elements[3][2],
		elements[0][3],
		elements[1][3],
		elements[2][3],
		elements[3][3],
	]);
}

export function packNormalMatrix4ForWGSL(
	normalMatrix: Matrix3Arr | Matrix4
): Float32Array {
	return packMatrix4ForWGSL(createNormalMatrixRows(normalMatrix));
}

export function packFrameUniformData(
	input: WebGPUFrameUniformInput
): Float32Array {
	return FRAME_UNIFORM_PACKER.pack(input);
}

export function remapClipSpaceDepth(clipZ: number, clipW: number): number {
	return clipZ * 0.5 + clipW * 0.5;
}

export function packModelUniformData(
	modelMatrix: Matrix4 | number[][],
	normalMatrix: Matrix3Arr | Matrix4,
	materialData: WebGPUMaterialUniformData,
	prevModelMatrix: Matrix4 | number[][],
	renderLayers = 1
): Float32Array<ArrayBuffer> {
	return MODEL_UNIFORM_PACKER.pack({
		modelMatrix,
		normalMatrix,
		materialData,
		prevModelMatrix,
		renderLayers,
	});
}

export type WebGPUModelUniformWriter = ReturnType<
	StructuredBufferLayout["createWriter"]
>;

/**
 * Creates a reusable writer for `ModelUniforms` data.
 *
 * @returns A zero-initialized structured buffer writer matching the WebGPU
 * model uniform layout.
 */
export function createModelUniformWriter(): WebGPUModelUniformWriter {
	return MODEL_UNIFORM_PACKER.createWriter();
}

/**
 * Writes `ModelUniforms` into an existing writer and returns its typed view.
 *
 * @param writer - Reusable writer created by `createModelUniformWriter`.
 * @param modelMatrix - Current model transform in engine row-major layout.
 * @param normalMatrix - Current normal matrix; only its upper-left 3x3 is used.
 * @param materialData - Packed material scalar and texture transform data.
 * @param prevModelMatrix - Previous-frame model transform for motion vectors.
 * @param renderLayers - Unsigned render-layer mask for this draw packet.
 * @returns The writer-owned `Float32Array`; callers must consume it before
 * reusing the same writer.
 */
export function writeModelUniformData(
	writer: WebGPUModelUniformWriter,
	modelMatrix: Matrix4 | number[][],
	normalMatrix: Matrix3Arr | Matrix4,
	materialData: WebGPUMaterialUniformData,
	prevModelMatrix: Matrix4 | number[][],
	renderLayers = 1
): Float32Array<ArrayBuffer> {
	return MODEL_UNIFORM_PACKER.packInto(writer, {
		modelMatrix,
		normalMatrix,
		materialData,
		prevModelMatrix,
		renderLayers,
	});
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
): [number, number, number, number] {
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
		shadow?.pcfRadius ?? 0,
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
		shadow?.pcssEnabled ? 1 : 0,
		shadow?.pcssRadius ?? 0,
		shadow?.shadowSamples ?? 0,
		shadow?.shadowSearchSamples ?? 0,
	]);
}
