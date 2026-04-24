import { Matrix4 } from "../../maths/Matrix4";
import type { Matrix3Arr } from "../../maths/types";

import {
	WEBGPU_FRAME_UNIFORM_FLOATS,
	WEBGPU_MAX_DIRECTIONAL_LIGHTS,
	WEBGPU_MAX_POINT_LIGHTS,
	WEBGPU_MAX_REFLECTION_PROBES,
	WEBGPU_MAX_SPOT_LIGHTS,
	WEBGPU_MODEL_UNIFORM_FLOATS,
	WEBGPU_SH_COEFFICIENT_COUNT,
	WEBGPU_TEXTURE_SLOT_COUNT,
} from "./constants";
import {
	StructuredBufferLayout,
	arrayOf,
	mat4x4f32,
	structOf,
	vec,
} from "./StructuredBufferLayout";
import type {
	WebGPUFrameUniformInput,
	WebGPUMaterialUniformData,
} from "./types";

const VEC4_F32 = vec(4, "f32");
const MAT4X4_F32 = mat4x4f32();

const DIRECTIONAL_LIGHT_SCHEMA = structOf([
	{ name: "direction", type: VEC4_F32 },
	{ name: "color", type: VEC4_F32 },
]);

const POINT_LIGHT_SCHEMA = structOf([
	{ name: "positionRange", type: VEC4_F32 },
	{ name: "color", type: VEC4_F32 },
]);

const SPOT_LIGHT_SCHEMA = structOf([
	{ name: "positionRange", type: VEC4_F32 },
	{ name: "directionOuter", type: VEC4_F32 },
	{ name: "colorInner", type: VEC4_F32 },
]);

const SHADOW_DATA_SCHEMA = structOf([
	{ name: "viewProjection", type: MAT4X4_F32 },
	{ name: "cascadeViewProjections", type: arrayOf(MAT4X4_F32, 4) },
	{ name: "cascadeSplits", type: arrayOf(VEC4_F32, 4) },
	{ name: "paramsA", type: VEC4_F32 },
	{ name: "paramsB", type: VEC4_F32 },
	{ name: "paramsC", type: VEC4_F32 },
]);

const FRAME_UNIFORM_LAYOUT = new StructuredBufferLayout(
	structOf([
		{ name: "viewProjection", type: MAT4X4_F32 },
		{ name: "prevViewProjection", type: MAT4X4_F32 },
		{ name: "cameraPosition", type: VEC4_F32 },
		{ name: "skyboxBasisRight", type: VEC4_F32 },
		{ name: "skyboxBasisUp", type: VEC4_F32 },
		{ name: "skyboxBasisBackward", type: VEC4_F32 },
		{ name: "ambientColor", type: VEC4_F32 },
		{ name: "lightCounts", type: VEC4_F32 },
		{ name: "options", type: VEC4_F32 },
		{ name: "environmentOptionsA", type: VEC4_F32 },
		{ name: "environmentOptionsB", type: VEC4_F32 },
		{ name: "taaJitterCurrentPrev", type: VEC4_F32 },
		{
			name: "directionalLights",
			type: arrayOf(DIRECTIONAL_LIGHT_SCHEMA, WEBGPU_MAX_DIRECTIONAL_LIGHTS),
		},
		{
			name: "pointLights",
			type: arrayOf(POINT_LIGHT_SCHEMA, WEBGPU_MAX_POINT_LIGHTS),
		},
		{
			name: "spotLights",
			type: arrayOf(SPOT_LIGHT_SCHEMA, WEBGPU_MAX_SPOT_LIGHTS),
		},
		{
			name: "directionalShadows",
			type: arrayOf(SHADOW_DATA_SCHEMA, WEBGPU_MAX_DIRECTIONAL_LIGHTS),
		},
		{
			name: "spotShadows",
			type: arrayOf(SHADOW_DATA_SCHEMA, WEBGPU_MAX_SPOT_LIGHTS),
		},
		{
			name: "shAmbientCoeffs",
			type: arrayOf(VEC4_F32, WEBGPU_SH_COEFFICIENT_COUNT),
		},
		{
			name: "reflectionProbeWorldToProbeRow0",
			type: arrayOf(VEC4_F32, WEBGPU_MAX_REFLECTION_PROBES),
		},
		{
			name: "reflectionProbeWorldToProbeRow1",
			type: arrayOf(VEC4_F32, WEBGPU_MAX_REFLECTION_PROBES),
		},
		{
			name: "reflectionProbeWorldToProbeRow2",
			type: arrayOf(VEC4_F32, WEBGPU_MAX_REFLECTION_PROBES),
		},
		{
			name: "reflectionProbeProbeToWorldRow0",
			type: arrayOf(VEC4_F32, WEBGPU_MAX_REFLECTION_PROBES),
		},
		{
			name: "reflectionProbeProbeToWorldRow1",
			type: arrayOf(VEC4_F32, WEBGPU_MAX_REFLECTION_PROBES),
		},
		{
			name: "reflectionProbeProbeToWorldRow2",
			type: arrayOf(VEC4_F32, WEBGPU_MAX_REFLECTION_PROBES),
		},
		{
			name: "reflectionProbeDataA",
			type: arrayOf(VEC4_F32, WEBGPU_MAX_REFLECTION_PROBES),
		},
		{
			name: "reflectionProbeDataB",
			type: arrayOf(VEC4_F32, WEBGPU_MAX_REFLECTION_PROBES),
		},
		{
			name: "reflectionProbeDataC",
			type: arrayOf(VEC4_F32, WEBGPU_MAX_REFLECTION_PROBES),
		},
	]),
	"uniform"
);

const MODEL_UNIFORM_LAYOUT = new StructuredBufferLayout(
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
		{ name: "materialFlags", type: VEC4_F32 },
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

FRAME_UNIFORM_LAYOUT.assertByteSize(
	WEBGPU_FRAME_UNIFORM_FLOATS * 4,
	"FrameUniforms"
);
MODEL_UNIFORM_LAYOUT.assertByteSize(
	WEBGPU_MODEL_UNIFORM_FLOATS * 4,
	"ModelUniforms"
);

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
	const writer = FRAME_UNIFORM_LAYOUT.createWriter();
	writer.expectByteLength(WEBGPU_FRAME_UNIFORM_FLOATS * 4, "FrameUniforms");

	writer.writeMat4("viewProjection", input.viewProjectionMatrix);
	writer.writeMat4("prevViewProjection", input.prevViewProjectionMatrix);
	writer.writeVec("cameraPosition", [
		input.cameraPosition.x,
		input.cameraPosition.y,
		input.cameraPosition.z,
		1,
	]);
	writer.writeVec("skyboxBasisRight", [
		input.skyboxRight[0],
		input.skyboxRight[1],
		input.skyboxRight[2],
		input.skyboxTanHalfFov,
	]);
	writer.writeVec("skyboxBasisUp", [
		input.skyboxUp[0],
		input.skyboxUp[1],
		input.skyboxUp[2],
		input.skyboxAspect,
	]);
	writer.writeVec("skyboxBasisBackward", [
		input.skyboxBackward[0],
		input.skyboxBackward[1],
		input.skyboxBackward[2],
		input.skyboxIsOrthographic ? 1 : 0,
	]);
	writer.writeVec("ambientColor", [
		input.ambientColor[0],
		input.ambientColor[1],
		input.ambientColor[2],
		1,
	]);
	writer.writeVec("lightCounts", [
		input.directionalLights.length,
		input.pointLights.length,
		input.spotLights.length,
		input.reflectionProbeCount,
	]);
	writer.writeVec("options", [
		input.enableLighting ? 1 : 0,
		input.enableGamma ? 1 : 0,
		input.enableShadows ? 1 : 0,
		input.encodeGammaInShader ? 1 : 0,
	]);
	writer.writeVec("environmentOptionsA", [
		input.enableSH ? 1 : 0,
		input.hasSHAmbient ? 1 : 0,
		input.hasSkybox ? 1 : 0,
		input.hasEnvSpecular ? 1 : 0,
	]);
	writer.writeVec("environmentOptionsB", [
		input.hasBRDFLUT ? 1 : 0,
		Math.max(0, input.envSpecularMaxMipLevel),
		input.skyboxIsLinear ? 1 : 0,
		input.enableClusteredLighting ? 1 : 0,
	]);
	writer.writeVec("taaJitterCurrentPrev", input.taaJitterCurrentPrev);

	for (let i = 0; i < WEBGPU_MAX_DIRECTIONAL_LIGHTS; i++) {
		const light = input.directionalLights[i];
		if (light) {
			writer.writeVec(["directionalLights", i, "direction"], [
				light.direction[0],
				light.direction[1],
				light.direction[2],
				0,
			]);
			writer.writeVec(["directionalLights", i, "color"], [
				light.color[0],
				light.color[1],
				light.color[2],
				0,
			]);
		}
	}

	for (let i = 0; i < WEBGPU_MAX_POINT_LIGHTS; i++) {
		const light = input.pointLights[i];
		if (light) {
			writer.writeVec(["pointLights", i, "positionRange"], [
				light.position[0],
				light.position[1],
				light.position[2],
				light.range,
			]);
			writer.writeVec(["pointLights", i, "color"], [
				light.color[0],
				light.color[1],
				light.color[2],
				0,
			]);
		}
	}

	for (let i = 0; i < WEBGPU_MAX_SPOT_LIGHTS; i++) {
		const light = input.spotLights[i];
		if (light) {
			writer.writeVec(["spotLights", i, "positionRange"], [
				light.position[0],
				light.position[1],
				light.position[2],
				light.range,
			]);
			writer.writeVec(["spotLights", i, "directionOuter"], [
				light.direction[0],
				light.direction[1],
				light.direction[2],
				light.outerCos,
			]);
			writer.writeVec(["spotLights", i, "colorInner"], [
				light.color[0],
				light.color[1],
				light.color[2],
				light.innerCos,
			]);
		}
	}

	for (let i = 0; i < WEBGPU_MAX_DIRECTIONAL_LIGHTS; i++) {
		writeShadowData(writer, "directionalShadows", i, input.directionalShadows[i]);
	}

	for (let i = 0; i < WEBGPU_MAX_SPOT_LIGHTS; i++) {
		writeShadowData(writer, "spotShadows", i, input.spotShadows[i]);
	}

	for (let i = 0; i < WEBGPU_SH_COEFFICIENT_COUNT; i++) {
		const coefficient = input.shAmbientCoeffs?.[i];
		if (coefficient) {
			writer.writeVec(["shAmbientCoeffs", i], [
				coefficient.r,
				coefficient.g,
				coefficient.b,
				0,
			]);
		}
	}

	for (let i = 0; i < WEBGPU_MAX_REFLECTION_PROBES; i++) {
		const probe = input.reflectionProbes[i];
		if (probe) {
			const worldToProbe = probe.worldToProbeMatrix.elements;
			writer.writeVec(["reflectionProbeWorldToProbeRow0", i], [
				worldToProbe[0][0],
				worldToProbe[0][1],
				worldToProbe[0][2],
				worldToProbe[0][3],
			]);
		}
	}
	for (let i = 0; i < WEBGPU_MAX_REFLECTION_PROBES; i++) {
		const probe = input.reflectionProbes[i];
		if (probe) {
			const worldToProbe = probe.worldToProbeMatrix.elements;
			writer.writeVec(["reflectionProbeWorldToProbeRow1", i], [
				worldToProbe[1][0],
				worldToProbe[1][1],
				worldToProbe[1][2],
				worldToProbe[1][3],
			]);
		}
	}
	for (let i = 0; i < WEBGPU_MAX_REFLECTION_PROBES; i++) {
		const probe = input.reflectionProbes[i];
		if (probe) {
			const worldToProbe = probe.worldToProbeMatrix.elements;
			writer.writeVec(["reflectionProbeWorldToProbeRow2", i], [
				worldToProbe[2][0],
				worldToProbe[2][1],
				worldToProbe[2][2],
				worldToProbe[2][3],
			]);
		}
	}

	for (let i = 0; i < WEBGPU_MAX_REFLECTION_PROBES; i++) {
		const probe = input.reflectionProbes[i];
		if (probe) {
			const probeToWorld = probe.probeToWorldMatrix.elements;
			writer.writeVec(["reflectionProbeProbeToWorldRow0", i], [
				probeToWorld[0][0],
				probeToWorld[0][1],
				probeToWorld[0][2],
				probeToWorld[0][3],
			]);
		}
	}
	for (let i = 0; i < WEBGPU_MAX_REFLECTION_PROBES; i++) {
		const probe = input.reflectionProbes[i];
		if (probe) {
			const probeToWorld = probe.probeToWorldMatrix.elements;
			writer.writeVec(["reflectionProbeProbeToWorldRow1", i], [
				probeToWorld[1][0],
				probeToWorld[1][1],
				probeToWorld[1][2],
				probeToWorld[1][3],
			]);
		}
	}
	for (let i = 0; i < WEBGPU_MAX_REFLECTION_PROBES; i++) {
		const probe = input.reflectionProbes[i];
		if (probe) {
			const probeToWorld = probe.probeToWorldMatrix.elements;
			writer.writeVec(["reflectionProbeProbeToWorldRow2", i], [
				probeToWorld[2][0],
				probeToWorld[2][1],
				probeToWorld[2][2],
				probeToWorld[2][3],
			]);
		}
	}

	for (let i = 0; i < WEBGPU_MAX_REFLECTION_PROBES; i++) {
		const probe = input.reflectionProbes[i];
		if (probe) {
			writer.writeVec(["reflectionProbeDataA", i], [
				probe.invHalfExtents[0],
				probe.invHalfExtents[1],
				probe.invHalfExtents[2],
				probe.radiusInv,
			]);
		}
	}

	for (let i = 0; i < WEBGPU_MAX_REFLECTION_PROBES; i++) {
		const probe = input.reflectionProbes[i];
		if (probe) {
			writer.writeVec(["reflectionProbeDataB", i], [
				probe.probeWorldPosition[0],
				probe.probeWorldPosition[1],
				probe.probeWorldPosition[2],
				probe.shape,
			]);
		}
	}

	for (let i = 0; i < WEBGPU_MAX_REFLECTION_PROBES; i++) {
		const probe = input.reflectionProbes[i];
		if (probe) {
			writer.writeVec(["reflectionProbeDataC", i], [
				probe.parallaxMode,
				probe.blendDistance,
				probe.blendExponent,
				probe.layer,
			]);
		}
	}

	return writer.toFloat32Array();
}

export function remapClipSpaceDepth(clipZ: number, clipW: number): number {
	return clipZ * 0.5 + clipW * 0.5;
}

export function packModelUniformData(
	modelMatrix: Matrix4 | number[][],
	normalMatrix: Matrix3Arr | Matrix4,
	materialData: WebGPUMaterialUniformData,
	prevModelMatrix: Matrix4 | number[][]
): Float32Array {
	const writer = MODEL_UNIFORM_LAYOUT.createWriter();
	writer.expectByteLength(WEBGPU_MODEL_UNIFORM_FLOATS * 4, "ModelUniforms");

	writer.writeMat4("modelMatrix", modelMatrix);
	writer.writeMat4("prevModelMatrix", prevModelMatrix);
	writer.writeMat4("normalMatrix", createNormalMatrixRows(normalMatrix));
	writer.writeVec("baseColorFactor", materialData.baseColorFactor);
	writer.writeVec("emissiveFactor", materialData.emissiveFactor);
	writer.writeVec("surfaceParams0", materialData.surfaceParams0);
	writer.writeVec("surfaceParams1", materialData.surfaceParams1);
	writer.writeVec("surfaceParams2", materialData.surfaceParams2);
	writer.writeVec("surfaceParams3", materialData.surfaceParams3);
	writer.writeVec("specularColorFactor", materialData.specularColorFactor);
	writer.writeVec("phongAmbientShininess", materialData.phongAmbientShininess);
	writer.writeVec("phongSpecularShading", materialData.phongSpecularShading);
	writer.writeVec(
		"sheenColorClearcoatNormalScale",
		materialData.sheenColorClearcoatNormalScale
	);
	writer.writeVec("attenuationColor", materialData.attenuationColor);
	writer.writeVec("materialFlags", materialData.materialFlags);

	for (let i = 0; i < WEBGPU_TEXTURE_SLOT_COUNT; i++) {
		const slot = materialData.textureSlots[i];
		if (!slot) {
			continue;
		}
		writer.writeVec(["textureTransformA", i], slot.transformA);
		writer.writeVec(["textureTransformB", i], slot.transformB);
	}

	return writer.toFloat32Array();
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
}
