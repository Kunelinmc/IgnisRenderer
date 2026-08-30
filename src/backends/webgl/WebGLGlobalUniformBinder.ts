import type { Vec3Tuple } from "../../maths/Vector3";
import type { Vec4Tuple } from "../../maths/Vector4";
import { clamp } from "../../maths/Common";
import { Matrix4 } from "../../maths/Matrix4";
import { finiteOr } from "../../maths/Misc";
import type { SHCoefficients } from "../../maths/types";
import type { FrameContext } from "../../pipeline/types";
import type { FogOptions } from "../../postprocess/passes/FogPass";
import { IBLBRDF } from "../../lights/ibl/IBLBRDF";
import {
	SHADOW_FILTER_MODE_CODE,
	SHADOW_QUALITY_CODE,
} from "../../lights/shadows/shadowSampling";
import { Logger } from "../../foundation/Logger";
import {
	MAX_DIRECTIONAL_LIGHTS,
	MAX_LOCAL_LIGHT_PROBES,
	MAX_POINT_LIGHTS,
	MAX_REFLECTION_PROBES,
	MAX_SPOT_LIGHTS,
} from "../constants";
import type {
	WebGLLightState,
	WebGLClusteredLight,
	WebGLLocalLightProbeUniform,
	WebGLReflectionProbeUniform,
	WebGLShadowData,
} from "./WebGLLightCollector";
import type { WebGLSceneProgram } from "./WebGLSceneProgram";
import type { WebGLShadowSamplingState } from "./WebGLShadowRuntime";
import { getWebGLSceneSamplerUnit } from "./WebGLSceneSamplerLayout";

const SH_COEFFICIENT_COUNT = 16;
const SH_AMBIENT_UNIFORM_VALUES = new Float32Array(SH_COEFFICIENT_COUNT * 3);
const IRRADIANCE_PROBE_GRID_ROW = new Float32Array(4);
const IRRADIANCE_PROBE_GRID_DATA_A = new Float32Array(4);
const IRRADIANCE_PROBE_GRID_DATA_B = new Float32Array(4);

const IDENTITY_MATRIX4_COLUMN_MAJOR = new Float32Array([
	1, 0, 0, 0,
	0, 1, 0, 0,
	0, 0, 1, 0,
	0, 0, 0, 1,
]);

function flattenVec4<T>(
	values: T[],
	mapper: (value: T) => Vec4Tuple,
	maxCount: number
): Float32Array {
	const resolvedMaxCount =
		typeof maxCount === "number" && Number.isFinite(maxCount) ?
			Math.max(0, Math.floor(maxCount))
		:	0;
	const packed = new Float32Array(resolvedMaxCount * 4);
	const count = Math.min(resolvedMaxCount, values.length);
	for (let i = 0; i < count; i++) {
		const value = mapper(values[i]);
		const offset = i * 4;
		packed[offset] = value[0];
		packed[offset + 1] = value[1];
		packed[offset + 2] = value[2];
		packed[offset + 3] = value[3];
	}
	return packed;
}

function flattenShadowViewProjection(
	values: WebGLShadowData[],
	maxCount: number
): Float32Array {
	const packed = new Float32Array(maxCount * 16);
	const count = Math.min(maxCount, values.length);
	for (let i = 0; i < count; i++) {
		const matrix = values[i]?.viewProjectionMatrix;
		if (!matrix) {
			continue;
		}
		packed.set(Matrix4.toColumnMajorArray(matrix), i * 16);
	}
	return packed;
}

function flattenShadowCascadeViewProjection(
	values: WebGLShadowData[],
	maxCount: number
): Float32Array {
	const cascadesPerLight = 4;
	const packed = new Float32Array(maxCount * cascadesPerLight * 16);
	const count = Math.min(maxCount, values.length);
	for (let lightIndex = 0; lightIndex < count; lightIndex++) {
		const shadow = values[lightIndex];
		if (!shadow?.enabled) {
			continue;
		}
		const cascades = shadow.cascadeViewProjectionMatrices ?? [];
		for (
			let cascadeIndex = 0;
			cascadeIndex < Math.min(cascadesPerLight, cascades.length);
			cascadeIndex++
		) {
			const matrix = cascades[cascadeIndex];
			if (!matrix) {
				continue;
			}
			const offset = (lightIndex * cascadesPerLight + cascadeIndex) * 16;
			packed.set(Matrix4.toColumnMajorArray(matrix), offset);
		}
	}
	return packed;
}

function flattenShadowCascadeSplits(
	values: WebGLShadowData[],
	maxCount: number
): Float32Array {
	const cascadesPerLight = 4;
	const packed = new Float32Array(maxCount * cascadesPerLight * 4);
	const count = Math.min(maxCount, values.length);
	for (let lightIndex = 0; lightIndex < count; lightIndex++) {
		const shadow = values[lightIndex];
		const splits = shadow.cascadeSplits ?? [];
		for (
			let cascadeIndex = 0;
			cascadeIndex < Math.min(cascadesPerLight, splits.length);
			cascadeIndex++
		) {
			const split = splits[cascadeIndex];
			const offset = (lightIndex * cascadesPerLight + cascadeIndex) * 4;
			packed[offset] = finiteOr(split[0], 0);
			packed[offset + 1] = finiteOr(split[1], 0);
			packed[offset + 2] = finiteOr(split[2], 0);
			packed[offset + 3] = finiteOr(split[3], 0);
		}
	}
	return packed;
}

function flattenShadowParamsA(
	values: WebGLShadowData[],
	maxCount: number
): Float32Array {
	const packed = new Float32Array(maxCount * 4);
	const count = Math.min(maxCount, values.length);
	for (let i = 0; i < count; i++) {
		const shadow = values[i];
		const offset = i * 4;
		packed[offset] = shadow.enabled ? 1 : 0;
		packed[offset + 1] = finiteOr(shadow.depthBias, 0);
		packed[offset + 2] = finiteOr(shadow.normalBias, 0);
		packed[offset + 3] = finiteOr(shadow.normalBiasMin, 0);
	}
	return packed;
}

function flattenShadowDepthProjectionParams(
	values: WebGLShadowData[],
	maxCount: number,
): Float32Array {
	const cascadesPerLight = 4;
	const packed = new Float32Array(maxCount * cascadesPerLight * 4);
	const count = Math.min(maxCount, values.length);
	for (let lightIndex = 0; lightIndex < count; lightIndex++) {
		const params = values[lightIndex].depthProjectionParams;
		for (let cascadeIndex = 0; cascadeIndex < cascadesPerLight; cascadeIndex++) {
			const source = params[cascadeIndex] ?? [0, 0, 0, 1];
			const offset = (lightIndex * cascadesPerLight + cascadeIndex) * 4;
			packed[offset] = finiteOr(source[0], 0);
			packed[offset + 1] = finiteOr(source[1], 0);
			packed[offset + 2] = finiteOr(source[2], 0);
			packed[offset + 3] = finiteOr(source[3], 1);
		}
	}
	return packed;
}

function flattenShadowParamsB(
	values: WebGLShadowData[],
	maxCount: number
): Float32Array {
	const packed = new Float32Array(maxCount * 4);
	const count = Math.min(maxCount, values.length);
	for (let i = 0; i < count; i++) {
		const shadow = values[i];
		const offset = i * 4;
		packed[offset] = 0;
		packed[offset + 1] = finiteOr(shadow.shadowStrength, 0);
		packed[offset + 2] = finiteOr(shadow.shadowMapSize, 0);
		packed[offset + 3] = finiteOr(shadow.atlasTileSize, 0);
	}
	return packed;
}

function flattenShadowParamsC(
	values: WebGLShadowData[],
	maxCount: number
): Float32Array {
	const packed = new Float32Array(maxCount * 4);
	const count = Math.min(maxCount, values.length);
	for (let i = 0; i < count; i++) {
		const shadow = values[i];
		const offset = i * 4;
		const isCSM =
			shadow.enabled &&
			shadow.strategyType === "csm" &&
			shadow.cascadeCount > 1;
		const cascadeCount =
			isCSM ? Math.max(1, Math.min(4, shadow.cascadeCount | 0)) : 1;
		packed[offset] = finiteOr(shadow.slopeBias, 0);
		packed[offset + 1] = isCSM ? 1 : 0;
		packed[offset + 2] = cascadeCount;
		packed[offset + 3] =
			isCSM ? clamp(finiteOr(shadow.cascadeBlendRatio, 0), 0, 1) : 0;
	}
	return packed;
}

function flattenShadowParamsD(
	values: WebGLShadowData[],
	maxCount: number
): Float32Array {
	const packed = new Float32Array(maxCount * 4);
	const count = Math.min(maxCount, values.length);
	for (let i = 0; i < count; i++) {
		const shadow = values[i];
		const offset = i * 4;
		packed[offset] = SHADOW_FILTER_MODE_CODE[shadow.filterMode];
		packed[offset + 1] = SHADOW_QUALITY_CODE[shadow.samplingQuality];
		packed[offset + 2] = 0;
		packed[offset + 3] = 0;
	}
	return packed;
}

function flattenReflectionProbeRows(
	values: WebGLReflectionProbeUniform[],
	matrixKey: "worldToProbeMatrix" | "probeToWorldMatrix",
	row: 0 | 1 | 2
): Float32Array {
	const packed = new Float32Array(MAX_REFLECTION_PROBES * 4);
	const count = Math.min(MAX_REFLECTION_PROBES, values.length);
	for (let i = 0; i < count; i++) {
		const matrix = values[i][matrixKey].elements;
		const offset = i * 4;
		packed[offset] = finiteOr(matrix[row][0], 0);
		packed[offset + 1] = finiteOr(matrix[row][1], 0);
		packed[offset + 2] = finiteOr(matrix[row][2], 0);
		packed[offset + 3] = finiteOr(matrix[row][3], 0);
	}
	return packed;
}

function flattenReflectionProbeVec4(
	values: WebGLReflectionProbeUniform[],
	mapper: (probe: WebGLReflectionProbeUniform) => Vec4Tuple
): Float32Array {
	const packed = new Float32Array(MAX_REFLECTION_PROBES * 4);
	const count = Math.min(MAX_REFLECTION_PROBES, values.length);
	for (let i = 0; i < count; i++) {
		const mapped = mapper(values[i]);
		const offset = i * 4;
		packed[offset] = finiteOr(mapped[0], 0);
		packed[offset + 1] = finiteOr(mapped[1], 0);
		packed[offset + 2] = finiteOr(mapped[2], 0);
		packed[offset + 3] = finiteOr(mapped[3], 0);
	}
	return packed;
}

function flattenLocalLightProbeRows(
	values: WebGLLocalLightProbeUniform[],
	row: 0 | 1 | 2
): Float32Array {
	const packed = new Float32Array(MAX_LOCAL_LIGHT_PROBES * 4);
	const count = Math.min(MAX_LOCAL_LIGHT_PROBES, values.length);
	for (let i = 0; i < count; i++) {
		const matrix = values[i].worldToProbeMatrix.elements;
		const offset = i * 4;
		packed[offset] = finiteOr(matrix[row][0], 0);
		packed[offset + 1] = finiteOr(matrix[row][1], 0);
		packed[offset + 2] = finiteOr(matrix[row][2], 0);
		packed[offset + 3] = finiteOr(matrix[row][3], 0);
	}
	return packed;
}

function flattenLocalLightProbeVec4(
	values: WebGLLocalLightProbeUniform[],
	mapper: (probe: WebGLLocalLightProbeUniform) => Vec4Tuple
): Float32Array {
	const packed = new Float32Array(MAX_LOCAL_LIGHT_PROBES * 4);
	const count = Math.min(MAX_LOCAL_LIGHT_PROBES, values.length);
	for (let i = 0; i < count; i++) {
		const mapped = mapper(values[i]);
		const offset = i * 4;
		packed[offset] = finiteOr(mapped[0], 0);
		packed[offset + 1] = finiteOr(mapped[1], 0);
		packed[offset + 2] = finiteOr(mapped[2], 0);
		packed[offset + 3] = finiteOr(mapped[3], 0);
	}
	return packed;
}

function toFiniteColumnMajorMat4(
	matrix: Matrix4 | number[][]
): Float32Array | null {
	const values = Matrix4.toColumnMajorArray(matrix);
	for (let i = 0; i < values.length; i++) {
		if (!Number.isFinite(values[i])) {
			return null;
		}
	}
	return values;
}

function sanitizeFloat32Array(
	values: Float32Array,
	fallback: number
): {
	values: Float32Array;
	hadInvalid: boolean;
} {
	let hadInvalid = false;
	for (let i = 0; i < values.length; i++) {
		if (!Number.isFinite(values[i])) {
			values[i] = fallback;
			hadInvalid = true;
		}
	}
	return { values, hadInvalid };
}

function logWebGLGlobalUniformWarning(key: string, message: string): void {
	Logger.warn(`[${key}] ${message}`, {
		scope: "WebGLGlobalUniformBinder",
		onceKey: key,
	});
}

function packSHAmbientUniformValues(
	coeffs: SHCoefficients | null | undefined
): Float32Array {
	for (let index = 0; index < SH_COEFFICIENT_COUNT; index++) {
		const coeff = coeffs?.[index];
		const base = index * 3;
		SH_AMBIENT_UNIFORM_VALUES[base] = finiteOr(coeff?.r, 0);
		SH_AMBIENT_UNIFORM_VALUES[base + 1] = finiteOr(coeff?.g, 0);
		SH_AMBIENT_UNIFORM_VALUES[base + 2] = finiteOr(coeff?.b, 0);
	}
	return SH_AMBIENT_UNIFORM_VALUES;
}

function packIrradianceProbeGridMatrixRow(
	grid: WebGLLightState["irradianceProbeGrid"],
	row: 0 | 1 | 2
): Float32Array {
	const matrix = grid?.worldToGridMatrix.elements;
	IRRADIANCE_PROBE_GRID_ROW[0] = finiteOr(matrix?.[row]?.[0], row === 0 ? 1 : 0);
	IRRADIANCE_PROBE_GRID_ROW[1] = finiteOr(matrix?.[row]?.[1], row === 1 ? 1 : 0);
	IRRADIANCE_PROBE_GRID_ROW[2] = finiteOr(matrix?.[row]?.[2], row === 2 ? 1 : 0);
	IRRADIANCE_PROBE_GRID_ROW[3] = finiteOr(matrix?.[row]?.[3], 0);
	return IRRADIANCE_PROBE_GRID_ROW;
}

function packIrradianceProbeGridDataA(
	grid: WebGLLightState["irradianceProbeGrid"]
): Float32Array {
	IRRADIANCE_PROBE_GRID_DATA_A[0] = finiteOr(grid?.dimensions[0], 1);
	IRRADIANCE_PROBE_GRID_DATA_A[1] = finiteOr(grid?.dimensions[1], 1);
	IRRADIANCE_PROBE_GRID_DATA_A[2] = finiteOr(grid?.dimensions[2], 1);
	IRRADIANCE_PROBE_GRID_DATA_A[3] = finiteOr(grid?.cellCount, 0);
	return IRRADIANCE_PROBE_GRID_DATA_A;
}

function packIrradianceProbeGridDataB(
	grid: WebGLLightState["irradianceProbeGrid"]
): Float32Array {
	IRRADIANCE_PROBE_GRID_DATA_B[0] = finiteOr(grid?.invHalfExtents[0], 1);
	IRRADIANCE_PROBE_GRID_DATA_B[1] = finiteOr(grid?.invHalfExtents[1], 1);
	IRRADIANCE_PROBE_GRID_DATA_B[2] = finiteOr(grid?.invHalfExtents[2], 1);
	IRRADIANCE_PROBE_GRID_DATA_B[3] = finiteOr(grid?.blendDistance, 0.01);
	return IRRADIANCE_PROBE_GRID_DATA_B;
}

export interface WebGLGlobalUniformBinderHost {
	_gl: WebGL2RenderingContext;
	_lightState: WebGLLightState | null;
	_textures: {
		getEnvironmentSpecularTexture(texture: any | null): {
			texture: WebGLTexture | null;
			isLinear: boolean;
		};
		getBRDFLUTTexture(texture: any | null): {
			texture: WebGLTexture | null;
			isLinear: boolean;
		};
	};
	_clusteredLighting: {
		getState(): {
			enabled: boolean;
			screenWidth: number;
			screenHeight: number;
			tilesX: number;
			tilesY: number;
			zSlices: number;
			maxLightsPerCluster: number;
			logScale: number;
			logBias: number;
			headerTexture: WebGLTexture | null;
			headerTexWidth: number;
			headerTexHeight: number;
			indexTexture: WebGLTexture | null;
			indexTexWidth: number;
			indexTexHeight: number;
			lightTexture: WebGLTexture | null;
			lightTexWidth: number;
			lightTexHeight: number;
		};
	};
	getShadowSamplingState(): WebGLShadowSamplingState;
	_temporalJitterCurrentPrev: Float32Array;
	_previousViewProjection: Float32Array | null;
	/** Probe SH coefficient texture source consumed by scene uniform binding. */
	_probeSHTextures: WebGLProbeSHUniformSource;
	/** Packed scene-fog uniform values shared with the particle pass. */
	_fog: WebGLSceneFogUniformState;
}

/**
 * Narrow probe-texture surface required by global uniform binding. Satisfied
 * structurally by `WebGLProbeSHTextures`.
 */
export interface WebGLProbeSHUniformSource {
	localLightProbeSHTexture: WebGLTexture | null;
	localLightProbeSHTextureWidth: number;
	localLightProbeSHTextureHeight: number;
	irradianceProbeGridSHTexture: WebGLTexture | null;
	irradianceProbeGridSHTextureWidth: number;
	irradianceProbeGridSHTextureHeight: number;
	uploadLocalLightProbeCoefficients(
		probes: WebGLLightState["localLightProbes"]
	): boolean;
	uploadIrradianceProbeGridCoefficients(
		grid: WebGLLightState["irradianceProbeGrid"]
	): boolean;
}

/**
 * Narrow packed-fog surface required by global uniform binding. Satisfied
 * structurally by `WebGLFogState`.
 */
export interface WebGLSceneFogUniformState {
	params0: Float32Array;
	params1: Float32Array;
	update(options: FogOptions | undefined, enabled: boolean): void;
}

export function bindWebGLGlobalUniforms(
	host: WebGLGlobalUniformBinderHost,
	sceneProgram: WebGLSceneProgram,
	context: FrameContext
): void {
	const gl = host._gl;
	const uniforms = sceneProgram.uniforms;
	const samplerUnit = (name: string): number =>
		getWebGLSceneSamplerUnit(sceneProgram.samplerLayout, name);
	const lightState = host._lightState as Partial<WebGLLightState> | null;
	const ambientColorCandidate = lightState?.ambientColor;
	const ambientColor: Vec3Tuple =
		Array.isArray(ambientColorCandidate) && ambientColorCandidate.length >= 3 ?
			[
				ambientColorCandidate[0],
				ambientColorCandidate[1],
				ambientColorCandidate[2],
			]
		:	[0, 0, 0];
	const directionalLights = lightState?.directionalLights ?? [];
	const directionalShadows = lightState?.directionalShadows ?? [];
	const pointLights = lightState?.pointLights ?? [];
	const spotLights = lightState?.spotLights ?? [];
	const spotShadows = lightState?.spotShadows ?? [];
	const clusteredLights =
		lightState?.clusteredLights ?? ([] as WebGLClusteredLight[]);
	const localLightProbes = lightState?.localLightProbes ?? [];
	const localLightProbeCountSource =
		Number.isFinite(lightState?.localLightProbeCount) ?
			(lightState?.localLightProbeCount as number)
		:	localLightProbes.length;
	const reflectionProbes = lightState?.reflectionProbes ?? [];
	const reflectionProbeCountSource =
		Number.isFinite(lightState?.reflectionProbeCount) ?
			(lightState?.reflectionProbeCount as number)
		:	reflectionProbes.length;
	const lights = {
		ambientColor,
		directionalLights,
		directionalShadows,
		pointLights,
		spotLights,
		spotShadows,
		clusteredLights,
		envSpecularMap: lightState?.envSpecularMap ?? null,
		envSpecularFallbackMap: lightState?.envSpecularFallbackMap ?? null,
		localLightProbeCount: localLightProbeCountSource,
		localLightProbes,
		irradianceProbeGrid: lightState?.irradianceProbeGrid ?? null,
		reflectionProbeCount: reflectionProbeCountSource,
		reflectionProbes,
	};

	if (uniforms.viewProjection) {
		const viewProjection = toFiniteColumnMajorMat4(
			context.viewCamera.viewProjectionMatrix
		);
		if (!viewProjection) {
			logWebGLGlobalUniformWarning(
				"webgl-camera-view-projection-invalid",
				"WebGL camera view-projection matrix is non-finite; using identity matrix."
			);
		}
		gl.uniformMatrix4fv(
			uniforms.viewProjection,
			false,
			viewProjection ?? IDENTITY_MATRIX4_COLUMN_MAJOR
		);
	}
	if (uniforms.viewMatrix) {
		const viewMatrix = toFiniteColumnMajorMat4(context.viewCamera.viewMatrix);
		if (!viewMatrix) {
			logWebGLGlobalUniformWarning(
				"webgl-camera-view-matrix-invalid",
				"WebGL camera view matrix is non-finite; using identity matrix."
			);
		}
		gl.uniformMatrix4fv(
			uniforms.viewMatrix,
			false,
			viewMatrix ?? IDENTITY_MATRIX4_COLUMN_MAJOR
		);
	}
	if (uniforms.cameraPosition) {
		const cameraPosition = context.viewCamera.getWorldPosition();
		const cameraX = finiteOr(cameraPosition.x, 0);
		const cameraY = finiteOr(cameraPosition.y, 0);
		const cameraZ = finiteOr(cameraPosition.z, 0);
		if (
			cameraX !== cameraPosition.x ||
			cameraY !== cameraPosition.y ||
			cameraZ !== cameraPosition.z
		) {
			logWebGLGlobalUniformWarning(
				"webgl-camera-position-invalid",
				"WebGL camera position is non-finite; using origin fallback."
			);
		}
		gl.uniform3f(uniforms.cameraPosition, cameraX, cameraY, cameraZ);
	}
	const fogOptions = context.postProcess.getOptions<FogOptions>("fog") ?? {};
	const sceneFogEnabled =
		context.postProcess.isEnabled("fog") &&
		(fogOptions.application ?? "postprocess") === "scene";
	host._fog.update(fogOptions, sceneFogEnabled);
	if (uniforms.fogParams0) {
		gl.uniform4fv(uniforms.fogParams0, host._fog.params0);
	}
	if (uniforms.fogParams1) {
		gl.uniform4fv(uniforms.fogParams1, host._fog.params1);
	}
	if (uniforms.ambientColor) {
		const ambientR = finiteOr(lights.ambientColor[0], 0);
		const ambientG = finiteOr(lights.ambientColor[1], 0);
		const ambientB = finiteOr(lights.ambientColor[2], 0);
		if (
			ambientR !== lights.ambientColor[0] ||
			ambientG !== lights.ambientColor[1] ||
			ambientB !== lights.ambientColor[2]
		) {
			logWebGLGlobalUniformWarning(
				"webgl-ambient-color-invalid",
				"WebGL ambient light color contains non-finite values; using black fallback."
			);
		}
		gl.uniform3f(uniforms.ambientColor, ambientR, ambientG, ambientB);
	}
	const hasSHAmbientCoefficients =
		Array.isArray(context.shAmbientCoeffs) && context.shAmbientCoeffs.length > 0;
	const localLightProbeTextureReady =
		host._probeSHTextures.uploadLocalLightProbeCoefficients(
			localLightProbes
		);
	const irradianceProbeGridTextureReady =
		host._probeSHTextures.uploadIrradianceProbeGridCoefficients(
			lights.irradianceProbeGrid
		);
	const resolvedIrradianceProbeGrid =
		irradianceProbeGridTextureReady ? lights.irradianceProbeGrid : null;
	const resolvedLocalLightProbeCount =
		localLightProbeTextureReady ?
			Math.max(0, Math.floor(localLightProbeCountSource))
		:	0;
	if (uniforms.shAmbientCoeffs) {
		gl.uniform3fv(
			uniforms.shAmbientCoeffs,
			packSHAmbientUniformValues(context.shAmbientCoeffs)
		);
	}
	if (uniforms.enableSH) {
		gl.uniform1i(
			uniforms.enableSH,
			context.features.enableSH &&
				(
					hasSHAmbientCoefficients ||
					resolvedLocalLightProbeCount > 0 ||
					!!resolvedIrradianceProbeGrid
				) ?
				1
			:	0
		);
	}
	if (uniforms.localLightProbeCount) {
		gl.uniform1i(uniforms.localLightProbeCount, resolvedLocalLightProbeCount);
	}
	if (uniforms.localLightProbeWorldToProbeRow0) {
		gl.uniform4fv(
			uniforms.localLightProbeWorldToProbeRow0,
			flattenLocalLightProbeRows(localLightProbes, 0)
		);
	}
	if (uniforms.localLightProbeWorldToProbeRow1) {
		gl.uniform4fv(
			uniforms.localLightProbeWorldToProbeRow1,
			flattenLocalLightProbeRows(localLightProbes, 1)
		);
	}
	if (uniforms.localLightProbeWorldToProbeRow2) {
		gl.uniform4fv(
			uniforms.localLightProbeWorldToProbeRow2,
			flattenLocalLightProbeRows(localLightProbes, 2)
		);
	}
	if (uniforms.localLightProbeDataA) {
		gl.uniform4fv(
			uniforms.localLightProbeDataA,
			flattenLocalLightProbeVec4(localLightProbes, (probe) => [
				probe.invHalfExtents[0],
				probe.invHalfExtents[1],
				probe.invHalfExtents[2],
				probe.radiusInv,
			])
		);
	}
	if (uniforms.localLightProbeDataB) {
		gl.uniform4fv(
			uniforms.localLightProbeDataB,
			flattenLocalLightProbeVec4(localLightProbes, (probe) => [
				probe.blendDistance,
				probe.priority,
				probe.shape,
				0,
			])
		);
	}
	if (uniforms.localLightProbeCoeffs) {
		const unit = samplerUnit("uLocalLightProbeCoeffs");
		gl.activeTexture(gl.TEXTURE0 + unit);
		gl.bindTexture(
			gl.TEXTURE_2D,
			host._probeSHTextures.localLightProbeSHTexture
		);
		gl.uniform1i(uniforms.localLightProbeCoeffs, unit);
	}
	if (uniforms.localLightProbeCoeffsSize) {
		gl.uniform2f(
			uniforms.localLightProbeCoeffsSize,
			host._probeSHTextures.localLightProbeSHTextureWidth,
			host._probeSHTextures.localLightProbeSHTextureHeight
		);
	}
	if (uniforms.irradianceProbeGridEnabled) {
		gl.uniform1i(
			uniforms.irradianceProbeGridEnabled,
			resolvedIrradianceProbeGrid ? 1 : 0
		);
	}
	if (uniforms.irradianceProbeGridWorldToGridRow0) {
		gl.uniform4fv(
			uniforms.irradianceProbeGridWorldToGridRow0,
			packIrradianceProbeGridMatrixRow(resolvedIrradianceProbeGrid, 0)
		);
	}
	if (uniforms.irradianceProbeGridWorldToGridRow1) {
		gl.uniform4fv(
			uniforms.irradianceProbeGridWorldToGridRow1,
			packIrradianceProbeGridMatrixRow(resolvedIrradianceProbeGrid, 1)
		);
	}
	if (uniforms.irradianceProbeGridWorldToGridRow2) {
		gl.uniform4fv(
			uniforms.irradianceProbeGridWorldToGridRow2,
			packIrradianceProbeGridMatrixRow(resolvedIrradianceProbeGrid, 2)
		);
	}
	if (uniforms.irradianceProbeGridDataA) {
		gl.uniform4fv(
			uniforms.irradianceProbeGridDataA,
			packIrradianceProbeGridDataA(resolvedIrradianceProbeGrid)
		);
	}
	if (uniforms.irradianceProbeGridDataB) {
		gl.uniform4fv(
			uniforms.irradianceProbeGridDataB,
			packIrradianceProbeGridDataB(resolvedIrradianceProbeGrid)
		);
	}
	if (uniforms.irradianceProbeGridCoeffs) {
		const unit = samplerUnit("uIrradianceProbeGridCoeffs");
		gl.activeTexture(gl.TEXTURE0 + unit);
		gl.bindTexture(
			gl.TEXTURE_2D,
			host._probeSHTextures.irradianceProbeGridSHTexture
		);
		gl.uniform1i(uniforms.irradianceProbeGridCoeffs, unit);
	}
	if (uniforms.irradianceProbeGridCoeffsSize) {
		gl.uniform2f(
			uniforms.irradianceProbeGridCoeffsSize,
			host._probeSHTextures.irradianceProbeGridSHTextureWidth,
			host._probeSHTextures.irradianceProbeGridSHTextureHeight
		);
	}

	const clusteredState = host._clusteredLighting.getState();
	const clusteredEnabled =
		context.features.enableClusteredLighting &&
		clusteredState.enabled &&
		!!clusteredState.headerTexture &&
		!!clusteredState.indexTexture &&
		!!clusteredState.lightTexture;
	if (uniforms.enableClusteredLighting) {
		gl.uniform1i(uniforms.enableClusteredLighting, clusteredEnabled ? 1 : 0);
	}
	if (uniforms.clusterParams0) {
		gl.uniform4f(
			uniforms.clusterParams0,
			clusteredState.screenWidth,
			clusteredState.screenHeight,
			clusteredState.tilesX,
			clusteredState.tilesY
		);
	}
	if (uniforms.clusterParams1) {
		gl.uniform4f(
			uniforms.clusterParams1,
			clusteredState.zSlices,
			clusteredState.maxLightsPerCluster,
			clusteredState.logScale,
			clusteredState.logBias
		);
	}
	if (uniforms.clusterHeaderTexSize) {
		gl.uniform2f(
			uniforms.clusterHeaderTexSize,
			clusteredState.headerTexWidth,
			clusteredState.headerTexHeight
		);
	}
	if (uniforms.clusterIndexTexSize) {
		gl.uniform2f(
			uniforms.clusterIndexTexSize,
			clusteredState.indexTexWidth,
			clusteredState.indexTexHeight
		);
	}
	if (uniforms.clusterLightTexSize) {
		gl.uniform2f(
			uniforms.clusterLightTexSize,
			clusteredState.lightTexWidth,
			clusteredState.lightTexHeight
		);
	}
	if (uniforms.clusterHeaderTexture) {
		const unit = samplerUnit("uClusterHeaderTexture");
		gl.activeTexture(gl.TEXTURE0 + unit);
		gl.bindTexture(gl.TEXTURE_2D, clusteredState.headerTexture);
		gl.uniform1i(uniforms.clusterHeaderTexture, unit);
	}
	if (uniforms.clusterIndexTexture) {
		const unit = samplerUnit("uClusterIndexTexture");
		gl.activeTexture(gl.TEXTURE0 + unit);
		gl.bindTexture(gl.TEXTURE_2D, clusteredState.indexTexture);
		gl.uniform1i(uniforms.clusterIndexTexture, unit);
	}
	if (uniforms.clusterLightTexture) {
		const unit = samplerUnit("uClusterLightTexture");
		gl.activeTexture(gl.TEXTURE0 + unit);
		gl.bindTexture(gl.TEXTURE_2D, clusteredState.lightTexture);
		gl.uniform1i(uniforms.clusterLightTexture, unit);
	}
	if (uniforms.enableLighting) {
		gl.uniform1i(uniforms.enableLighting, context.features.enableLighting ? 1 : 0);
	}
	const shadowSampling = host.getShadowSamplingState();
	const shadowsEnabled = context.features.enableShadows && shadowSampling.enabled;
	if (uniforms.enableShadows) {
		gl.uniform1i(uniforms.enableShadows, shadowsEnabled ? 1 : 0);
	}
	if (uniforms.shadowAtlas) {
		const unit = samplerUnit("uShadowAtlas");
		gl.activeTexture(gl.TEXTURE0 + unit);
		gl.bindTexture(gl.TEXTURE_2D, shadowSampling.atlasTexture);
		gl.uniform1i(uniforms.shadowAtlas, unit);
	}
	const hasShadowTransmittance = shadowSampling.transmittanceAvailable;
	if (uniforms.shadowTransmittanceAtlasAvailable) {
		gl.uniform1i(
			uniforms.shadowTransmittanceAtlasAvailable,
			hasShadowTransmittance ? 1 : 0
		);
	}
	if (uniforms.shadowTransmittanceAtlas && hasShadowTransmittance) {
		const unit = samplerUnit("uShadowTransmittanceAtlas");
		gl.activeTexture(gl.TEXTURE0 + unit);
		gl.bindTexture(gl.TEXTURE_2D, shadowSampling.transmittanceTexture);
		gl.uniform1i(
			uniforms.shadowTransmittanceAtlas,
			unit
		);
	}
	if (uniforms.particleShadowVolumeAtlas && shadowSampling.particleVolumeTexture) {
		const unit = samplerUnit("uParticleShadowVolumeAtlas");
		gl.activeTexture(gl.TEXTURE0 + unit);
		gl.bindTexture(gl.TEXTURE_2D, shadowSampling.particleVolumeTexture);
		gl.uniform1i(
			uniforms.particleShadowVolumeAtlas,
			unit
		);
	}
	if (uniforms.particleShadowVolumeAtlasSize) {
		gl.uniform2f(
			uniforms.particleShadowVolumeAtlasSize,
			shadowSampling.particleVolumeAtlasSize[0],
			shadowSampling.particleVolumeAtlasSize[1]
		);
	}
	if (uniforms.particleShadowVolumeGridSize) {
		gl.uniform4fv(
			uniforms.particleShadowVolumeGridSize,
			shadowSampling.particleVolumeGridSize as Float32Array
		);
	}
	if (uniforms.particleShadowVolumeSliceParams) {
		gl.uniform4fv(
			uniforms.particleShadowVolumeSliceParams,
			shadowSampling.particleVolumeSliceParams as Float32Array
		);
	}

	const usesEnvSpecularUniforms =
		!!uniforms.envSpecularMap ||
		!!uniforms.hasEnvSpecularMap ||
		!!uniforms.envSpecularMapIsLinear ||
		!!uniforms.envSpecularMaxMipLevel ||
		!!uniforms.envSpecularFallbackMap ||
		!!uniforms.hasEnvSpecularFallbackMap ||
		!!uniforms.envSpecularFallbackMapIsLinear ||
		!!uniforms.envSpecularFallbackMaxMipLevel ||
		!!uniforms.brdfLUT ||
		!!uniforms.reflectionProbeCount ||
		!!uniforms.reflectionProbeWorldToProbeRow0 ||
		!!uniforms.reflectionProbeWorldToProbeRow1 ||
		!!uniforms.reflectionProbeWorldToProbeRow2 ||
		!!uniforms.reflectionProbeProbeToWorldRow0 ||
		!!uniforms.reflectionProbeProbeToWorldRow1 ||
		!!uniforms.reflectionProbeProbeToWorldRow2 ||
		!!uniforms.reflectionProbeDataA ||
		!!uniforms.reflectionProbeDataB ||
		!!uniforms.reflectionProbeDataC;
	if (usesEnvSpecularUniforms) {
		const envSpecularMap = lights.envSpecularMap;
		const envSpecularFallbackMap = lights.envSpecularFallbackMap;
		const hasEnvSpecularMap = !!envSpecularMap;
		const hasEnvSpecularFallbackMap = !!envSpecularFallbackMap;
		const envSpecularMaxMipLevel =
			hasEnvSpecularMap && envSpecularMap ?
				Math.max(0, envSpecularMap.mipmaps.length - 1)
			:	0;
		const envSpecularFallbackMaxMipLevel =
			hasEnvSpecularFallbackMap && envSpecularFallbackMap ?
				Math.max(0, envSpecularFallbackMap.mipmaps.length - 1)
			:	0;
		const resolvedEnvSpecular =
			host._textures.getEnvironmentSpecularTexture(envSpecularMap ?? null);
		const resolvedEnvSpecularFallback =
			host._textures.getEnvironmentSpecularTexture(
				envSpecularFallbackMap ?? null
			);
		const resolvedBrdfLUT = host._textures.getBRDFLUTTexture(
			hasEnvSpecularMap ? IBLBRDF.getLUT() : null
		);

		if (uniforms.envSpecularMap) {
			const unit = samplerUnit("uEnvSpecularMap");
			gl.activeTexture(gl.TEXTURE0 + unit);
			gl.bindTexture(gl.TEXTURE_2D, resolvedEnvSpecular.texture);
			gl.uniform1i(uniforms.envSpecularMap, unit);
		}
		if (uniforms.envSpecularFallbackMap) {
			const unit = samplerUnit("uEnvSpecularFallbackMap");
			gl.activeTexture(gl.TEXTURE0 + unit);
			gl.bindTexture(
				gl.TEXTURE_2D,
				resolvedEnvSpecularFallback.texture
			);
			gl.uniform1i(
				uniforms.envSpecularFallbackMap,
				unit
			);
		}
		if (uniforms.brdfLUT) {
			const unit = samplerUnit("uBrdfLUT");
			gl.activeTexture(gl.TEXTURE0 + unit);
			gl.bindTexture(gl.TEXTURE_2D, resolvedBrdfLUT.texture);
			gl.uniform1i(uniforms.brdfLUT, unit);
		}
		if (uniforms.hasEnvSpecularMap) {
			gl.uniform1i(uniforms.hasEnvSpecularMap, hasEnvSpecularMap ? 1 : 0);
		}
		if (uniforms.envSpecularMapIsLinear) {
			gl.uniform1i(
				uniforms.envSpecularMapIsLinear,
				resolvedEnvSpecular.isLinear ? 1 : 0
			);
		}
		if (uniforms.envSpecularMaxMipLevel) {
			gl.uniform1f(uniforms.envSpecularMaxMipLevel, envSpecularMaxMipLevel);
		}
		if (uniforms.hasEnvSpecularFallbackMap) {
			gl.uniform1i(
				uniforms.hasEnvSpecularFallbackMap,
				hasEnvSpecularFallbackMap ? 1 : 0
			);
		}
		if (uniforms.envSpecularFallbackMapIsLinear) {
			gl.uniform1i(
				uniforms.envSpecularFallbackMapIsLinear,
				resolvedEnvSpecularFallback.isLinear ? 1 : 0
			);
		}
		if (uniforms.envSpecularFallbackMaxMipLevel) {
			gl.uniform1f(
				uniforms.envSpecularFallbackMaxMipLevel,
				envSpecularFallbackMaxMipLevel
			);
		}
		const reflectionProbeCount = Math.max(0, Math.floor(reflectionProbeCountSource));
		if (uniforms.reflectionProbeCount) {
			gl.uniform1i(uniforms.reflectionProbeCount, reflectionProbeCount);
		}
		if (uniforms.reflectionProbeWorldToProbeRow0) {
			gl.uniform4fv(
				uniforms.reflectionProbeWorldToProbeRow0,
				flattenReflectionProbeRows(
					reflectionProbes,
					"worldToProbeMatrix",
					0
				)
			);
		}
		if (uniforms.reflectionProbeWorldToProbeRow1) {
			gl.uniform4fv(
				uniforms.reflectionProbeWorldToProbeRow1,
				flattenReflectionProbeRows(
					reflectionProbes,
					"worldToProbeMatrix",
					1
				)
			);
		}
		if (uniforms.reflectionProbeWorldToProbeRow2) {
			gl.uniform4fv(
				uniforms.reflectionProbeWorldToProbeRow2,
				flattenReflectionProbeRows(
					reflectionProbes,
					"worldToProbeMatrix",
					2
				)
			);
		}
		if (uniforms.reflectionProbeProbeToWorldRow0) {
			gl.uniform4fv(
				uniforms.reflectionProbeProbeToWorldRow0,
				flattenReflectionProbeRows(
					reflectionProbes,
					"probeToWorldMatrix",
					0
				)
			);
		}
		if (uniforms.reflectionProbeProbeToWorldRow1) {
			gl.uniform4fv(
				uniforms.reflectionProbeProbeToWorldRow1,
				flattenReflectionProbeRows(
					reflectionProbes,
					"probeToWorldMatrix",
					1
				)
			);
		}
		if (uniforms.reflectionProbeProbeToWorldRow2) {
			gl.uniform4fv(
				uniforms.reflectionProbeProbeToWorldRow2,
				flattenReflectionProbeRows(
					reflectionProbes,
					"probeToWorldMatrix",
					2
				)
			);
		}
		if (uniforms.reflectionProbeDataA) {
			gl.uniform4fv(
				uniforms.reflectionProbeDataA,
				flattenReflectionProbeVec4(reflectionProbes, (probe) => [
					probe.invHalfExtents[0],
					probe.invHalfExtents[1],
					probe.invHalfExtents[2],
					probe.radiusInv,
				])
			);
		}
		if (uniforms.reflectionProbeDataB) {
			gl.uniform4fv(
				uniforms.reflectionProbeDataB,
				flattenReflectionProbeVec4(reflectionProbes, (probe) => [
					probe.captureWorldPosition[0],
					probe.captureWorldPosition[1],
					probe.captureWorldPosition[2],
					probe.shape,
				])
			);
		}
		if (uniforms.reflectionProbeDataC) {
			gl.uniform4fv(
				uniforms.reflectionProbeDataC,
				flattenReflectionProbeVec4(reflectionProbes, (probe) => [
					probe.parallaxMode,
					probe.blendDistance,
					probe.blendExponent,
					probe.layer,
				])
			);
		}
	}
	gl.activeTexture(gl.TEXTURE0);

	if (uniforms.taaJitter) {
		gl.uniform4fv(uniforms.taaJitter, host._temporalJitterCurrentPrev);
	}
	if (uniforms.prevViewProjection) {
		const prevViewProjection = sanitizeFloat32Array(
			host._previousViewProjection ??
				Matrix4.toColumnMajorArray(context.viewCamera.viewProjectionMatrix),
			0
		);
		if (prevViewProjection.hadInvalid) {
			logWebGLGlobalUniformWarning(
				"webgl-prev-view-projection-invalid",
				"WebGL previous view-projection matrix is non-finite; using sanitized values."
			);
		}
		gl.uniformMatrix4fv(
			uniforms.prevViewProjection,
			false,
			prevViewProjection.values
		);
	}

	if (uniforms.dirLightCount) {
		gl.uniform1i(uniforms.dirLightCount, lights.directionalLights.length);
	}
	if (uniforms.dirLightDirection) {
		const packedDirection = sanitizeFloat32Array(
			flattenVec4(lights.directionalLights, (light) => [
				light.direction[0],
				light.direction[1],
				light.direction[2],
				0,
			], MAX_DIRECTIONAL_LIGHTS),
			0
		);
		if (packedDirection.hadInvalid) {
			logWebGLGlobalUniformWarning(
				"webgl-dir-light-direction-invalid",
				"WebGL directional light direction contains non-finite values; using sanitized values."
			);
		}
		gl.uniform4fv(uniforms.dirLightDirection, packedDirection.values);
	}
	if (uniforms.dirLightColor) {
		const packedColor = sanitizeFloat32Array(
			flattenVec4(lights.directionalLights, (light) => [
				light.color[0],
				light.color[1],
				light.color[2],
				0,
			], MAX_DIRECTIONAL_LIGHTS),
			0
		);
		if (packedColor.hadInvalid) {
			logWebGLGlobalUniformWarning(
				"webgl-dir-light-color-invalid",
				"WebGL directional light color contains non-finite values; using sanitized values."
			);
		}
		gl.uniform4fv(uniforms.dirLightColor, packedColor.values);
	}
	if (uniforms.dirShadowViewProjection) {
		const packedShadowViewProjection = sanitizeFloat32Array(
			flattenShadowViewProjection(lights.directionalShadows, MAX_DIRECTIONAL_LIGHTS),
			0
		);
		if (packedShadowViewProjection.hadInvalid) {
			logWebGLGlobalUniformWarning(
				"webgl-dir-shadow-view-projection-invalid",
				"WebGL directional shadow matrix contains non-finite values; using sanitized values."
			);
		}
		gl.uniformMatrix4fv(
			uniforms.dirShadowViewProjection,
			false,
			packedShadowViewProjection.values
		);
	}
	if (uniforms.dirShadowCascadeViewProjection) {
		const packedCascadeViewProjection = sanitizeFloat32Array(
			flattenShadowCascadeViewProjection(
				lights.directionalShadows,
				MAX_DIRECTIONAL_LIGHTS
			),
			0
		);
		if (packedCascadeViewProjection.hadInvalid) {
			logWebGLGlobalUniformWarning(
				"webgl-dir-shadow-cascade-view-projection-invalid",
				"WebGL directional cascade shadow matrices contain non-finite values; using sanitized values."
			);
		}
		gl.uniformMatrix4fv(
			uniforms.dirShadowCascadeViewProjection,
			false,
			packedCascadeViewProjection.values
		);
	}
	if (uniforms.dirShadowCascadeSplits) {
		const packedCascadeSplits = sanitizeFloat32Array(
			flattenShadowCascadeSplits(
				lights.directionalShadows,
				MAX_DIRECTIONAL_LIGHTS
			),
			0
		);
		if (packedCascadeSplits.hadInvalid) {
			logWebGLGlobalUniformWarning(
				"webgl-dir-shadow-cascade-splits-invalid",
				"WebGL directional cascade split parameters contain non-finite values; using sanitized values."
			);
		}
		gl.uniform4fv(
			uniforms.dirShadowCascadeSplits,
			packedCascadeSplits.values
		);
	}
	if (uniforms.dirShadowDepthProjectionParams) {
		gl.uniform4fv(
			uniforms.dirShadowDepthProjectionParams,
			flattenShadowDepthProjectionParams(
				lights.directionalShadows,
				MAX_DIRECTIONAL_LIGHTS,
			),
		);
	}
	if (uniforms.dirShadowParamsA) {
		const packedDirShadowParamsA = sanitizeFloat32Array(
			flattenShadowParamsA(lights.directionalShadows, MAX_DIRECTIONAL_LIGHTS),
			0
		);
		if (packedDirShadowParamsA.hadInvalid) {
			logWebGLGlobalUniformWarning(
				"webgl-dir-shadow-params-a-invalid",
				"WebGL directional shadow parameters contain non-finite values; using sanitized values."
			);
		}
		gl.uniform4fv(uniforms.dirShadowParamsA, packedDirShadowParamsA.values);
	}
	if (uniforms.dirShadowParamsB) {
		const packedDirShadowParamsB = sanitizeFloat32Array(
			flattenShadowParamsB(lights.directionalShadows, MAX_DIRECTIONAL_LIGHTS),
			0
		);
		if (packedDirShadowParamsB.hadInvalid) {
			logWebGLGlobalUniformWarning(
				"webgl-dir-shadow-params-b-invalid",
				"WebGL directional shadow parameters contain non-finite values; using sanitized values."
			);
		}
		gl.uniform4fv(uniforms.dirShadowParamsB, packedDirShadowParamsB.values);
	}
	if (uniforms.dirShadowParamsC) {
		const packedDirShadowParamsC = sanitizeFloat32Array(
			flattenShadowParamsC(lights.directionalShadows, MAX_DIRECTIONAL_LIGHTS),
			0
		);
		if (packedDirShadowParamsC.hadInvalid) {
			logWebGLGlobalUniformWarning(
				"webgl-dir-shadow-params-c-invalid",
				"WebGL directional shadow slope parameters contain non-finite values; using sanitized values."
			);
		}
		gl.uniform4fv(uniforms.dirShadowParamsC, packedDirShadowParamsC.values);
	}
	if (uniforms.dirShadowParamsD) {
		const packedDirShadowParamsD = sanitizeFloat32Array(
			flattenShadowParamsD(lights.directionalShadows, MAX_DIRECTIONAL_LIGHTS),
			0
		);
		if (packedDirShadowParamsD.hadInvalid) {
			logWebGLGlobalUniformWarning(
				"webgl-dir-shadow-params-d-invalid",
				"WebGL directional shadow PCSS parameters contain non-finite values; using sanitized values."
			);
		}
		gl.uniform4fv(uniforms.dirShadowParamsD, packedDirShadowParamsD.values);
	}

	if (uniforms.pointLightCount) {
		gl.uniform1i(uniforms.pointLightCount, lights.pointLights.length);
	}
	if (uniforms.pointLightPositionRange) {
		const packedPointPositionRange = sanitizeFloat32Array(
			flattenVec4(lights.pointLights, (light) => [
				light.position[0],
				light.position[1],
				light.position[2],
				light.range,
			], MAX_POINT_LIGHTS),
			0
		);
		if (packedPointPositionRange.hadInvalid) {
			logWebGLGlobalUniformWarning(
				"webgl-point-light-position-invalid",
				"WebGL point light position/range contains non-finite values; using sanitized values."
			);
		}
		gl.uniform4fv(uniforms.pointLightPositionRange, packedPointPositionRange.values);
	}
	if (uniforms.pointLightColor) {
		const packedPointColor = sanitizeFloat32Array(
			flattenVec4(lights.pointLights, (light) => [
				light.color[0],
				light.color[1],
				light.color[2],
				0,
			], MAX_POINT_LIGHTS),
			0
		);
		if (packedPointColor.hadInvalid) {
			logWebGLGlobalUniformWarning(
				"webgl-point-light-color-invalid",
				"WebGL point light color contains non-finite values; using sanitized values."
			);
		}
		gl.uniform4fv(uniforms.pointLightColor, packedPointColor.values);
	}

	if (uniforms.spotLightCount) {
		gl.uniform1i(uniforms.spotLightCount, lights.spotLights.length);
	}
	if (uniforms.spotLightPositionRange) {
		const packedSpotPositionRange = sanitizeFloat32Array(
			flattenVec4(lights.spotLights, (light) => [
				light.position[0],
				light.position[1],
				light.position[2],
				light.range,
			], MAX_SPOT_LIGHTS),
			0
		);
		if (packedSpotPositionRange.hadInvalid) {
			logWebGLGlobalUniformWarning(
				"webgl-spot-light-position-invalid",
				"WebGL spot light position/range contains non-finite values; using sanitized values."
			);
		}
		gl.uniform4fv(uniforms.spotLightPositionRange, packedSpotPositionRange.values);
	}
	if (uniforms.spotLightDirectionOuter) {
		const packedSpotDirectionOuter = sanitizeFloat32Array(
			flattenVec4(lights.spotLights, (light) => [
				light.direction[0],
				light.direction[1],
				light.direction[2],
				light.outerCos,
			], MAX_SPOT_LIGHTS),
			0
		);
		if (packedSpotDirectionOuter.hadInvalid) {
			logWebGLGlobalUniformWarning(
				"webgl-spot-light-direction-invalid",
				"WebGL spot light direction/outer cone contains non-finite values; using sanitized values."
			);
		}
		gl.uniform4fv(uniforms.spotLightDirectionOuter, packedSpotDirectionOuter.values);
	}
	if (uniforms.spotLightColorInner) {
		const packedSpotColorInner = sanitizeFloat32Array(
			flattenVec4(lights.spotLights, (light) => [
				light.color[0],
				light.color[1],
				light.color[2],
				light.innerCos,
			], MAX_SPOT_LIGHTS),
			0
		);
		if (packedSpotColorInner.hadInvalid) {
			logWebGLGlobalUniformWarning(
				"webgl-spot-light-color-invalid",
				"WebGL spot light color/inner cone contains non-finite values; using sanitized values."
			);
		}
		gl.uniform4fv(uniforms.spotLightColorInner, packedSpotColorInner.values);
	}
	if (uniforms.spotShadowViewProjection) {
		const packedSpotShadowViewProjection = sanitizeFloat32Array(
			flattenShadowViewProjection(lights.spotShadows, MAX_SPOT_LIGHTS),
			0
		);
		if (packedSpotShadowViewProjection.hadInvalid) {
			logWebGLGlobalUniformWarning(
				"webgl-spot-shadow-view-projection-invalid",
				"WebGL spot shadow matrix contains non-finite values; using sanitized values."
			);
		}
		gl.uniformMatrix4fv(
			uniforms.spotShadowViewProjection,
			false,
			packedSpotShadowViewProjection.values
		);
	}
	if (uniforms.spotShadowDepthProjectionParams) {
		gl.uniform4fv(
			uniforms.spotShadowDepthProjectionParams,
			flattenShadowDepthProjectionParams(
				lights.spotShadows,
				MAX_SPOT_LIGHTS,
			),
		);
	}
	if (uniforms.spotShadowParamsA) {
		const packedSpotShadowParamsA = sanitizeFloat32Array(
			flattenShadowParamsA(lights.spotShadows, MAX_SPOT_LIGHTS),
			0
		);
		if (packedSpotShadowParamsA.hadInvalid) {
			logWebGLGlobalUniformWarning(
				"webgl-spot-shadow-params-a-invalid",
				"WebGL spot shadow parameters contain non-finite values; using sanitized values."
			);
		}
		gl.uniform4fv(uniforms.spotShadowParamsA, packedSpotShadowParamsA.values);
	}
	if (uniforms.spotShadowParamsB) {
		const packedSpotShadowParamsB = sanitizeFloat32Array(
			flattenShadowParamsB(lights.spotShadows, MAX_SPOT_LIGHTS),
			0
		);
		if (packedSpotShadowParamsB.hadInvalid) {
			logWebGLGlobalUniformWarning(
				"webgl-spot-shadow-params-b-invalid",
				"WebGL spot shadow parameters contain non-finite values; using sanitized values."
			);
		}
		gl.uniform4fv(uniforms.spotShadowParamsB, packedSpotShadowParamsB.values);
	}
	if (uniforms.spotShadowParamsC) {
		const packedSpotShadowParamsC = sanitizeFloat32Array(
			flattenShadowParamsC(lights.spotShadows, MAX_SPOT_LIGHTS),
			0
		);
		if (packedSpotShadowParamsC.hadInvalid) {
			logWebGLGlobalUniformWarning(
				"webgl-spot-shadow-params-c-invalid",
				"WebGL spot shadow slope parameters contain non-finite values; using sanitized values."
			);
		}
		gl.uniform4fv(uniforms.spotShadowParamsC, packedSpotShadowParamsC.values);
	}
	if (uniforms.spotShadowParamsD) {
		const packedSpotShadowParamsD = sanitizeFloat32Array(
			flattenShadowParamsD(lights.spotShadows, MAX_SPOT_LIGHTS),
			0
		);
		if (packedSpotShadowParamsD.hadInvalid) {
			logWebGLGlobalUniformWarning(
				"webgl-spot-shadow-params-d-invalid",
				"WebGL spot shadow PCSS parameters contain non-finite values; using sanitized values."
			);
		}
		gl.uniform4fv(uniforms.spotShadowParamsD, packedSpotShadowParamsD.values);
	}
}

