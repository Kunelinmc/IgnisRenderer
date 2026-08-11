import type { SHCoefficients } from "../../maths/types";
import type { FrameContext } from "../../pipeline/types";
import type { FogOptions } from "../../postprocess/passes/FogPass";
import { IBLBRDF } from "../../lights/ibl/IBLBRDF";
import { Logger } from "../../foundation/Logger";
import {
	MAX_DIRECTIONAL_LIGHTS,
	MAX_POINT_LIGHTS,
	MAX_SPOT_LIGHTS,
} from "../constants";
import {
	WEBGL_TEXTURE_UNIT_PARTICLE_SHADOW_VOLUME,
	WEBGL_TEXTURE_UNIT_SHADOW_TRANSMITTANCE,
} from "./constants";
import {
	finiteOr,
	flattenLocalLightProbeRows,
	flattenLocalLightProbeVec4,
	flattenReflectionProbeRows,
	flattenReflectionProbeVec4,
	flattenShadowParamsA,
	flattenShadowParamsB,
	flattenShadowParamsC,
	flattenShadowParamsD,
	flattenShadowCascadeSplits,
	flattenShadowCascadeViewProjection,
	flattenShadowViewProjection,
	flattenVec4,
	sanitizeFloat32Array,
	toColumnMajorMat4,
	toFiniteColumnMajorMat4,
} from "./WebGLFrameMath";
import type { WebGLLightState, WebGLClusteredLight } from "./WebGLLightCollector";
import type { WebGLSceneProgram } from "./WebGLProgramLibrary";
import type { WebGLShadowSamplingState } from "./WebGLShadowRuntime";
import { getWebGLSceneSamplerUnit } from "./WebGLSceneSamplerLayout";

const WEBGL_TEXTURE_UNIT_BASE_MAP = 0;
const WEBGL_TEXTURE_UNIT_SHADOW_ATLAS = 1;
const WEBGL_TEXTURE_UNIT_ENV_SPECULAR = 2;
const WEBGL_TEXTURE_UNIT_BRDF_LUT = 3;
const WEBGL_TEXTURE_UNIT_CLUSTER_HEADER = 5;
const WEBGL_TEXTURE_UNIT_CLUSTER_INDEX = 6;
const WEBGL_TEXTURE_UNIT_CLUSTER_LIGHT = 7;
const WEBGL_TEXTURE_UNIT_LOCAL_LIGHT_PROBE_SH = 4;
const WEBGL_TEXTURE_UNIT_ENV_SPECULAR_FALLBACK = 13;
const WEBGL_TEXTURE_UNIT_IRRADIANCE_PROBE_GRID_SH = 15;
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
	_maxTextureImageUnits: number;
	_irradianceProbeGridSamplingSupported: boolean;
	_temporalJitterCurrentPrev: Float32Array;
	_previousViewProjection: Float32Array | null;
	_shAmbientTexture: WebGLTexture | null;
	_shAmbientTextureWidth: number;
	_shAmbientTextureHeight: number;
	_localLightProbeSHTexture: WebGLTexture | null;
	_localLightProbeSHTextureWidth: number;
	_localLightProbeSHTextureHeight: number;
	_irradianceProbeGridSHTexture: WebGLTexture | null;
	_irradianceProbeGridSHTextureWidth: number;
	_irradianceProbeGridSHTextureHeight: number;
	_fogParams0: Float32Array;
	_fogParams1: Float32Array;
	_updateFogParams(options: FogOptions | undefined, enabled: boolean): void;
	_uploadSHAmbientCoefficients(
		coeffs: SHCoefficients | null | undefined
	): boolean;
	_uploadLocalLightProbeCoefficients(
		probes: WebGLLightState["localLightProbes"]
	): boolean;
	_uploadIrradianceProbeGridCoefficients(
		grid: WebGLLightState["irradianceProbeGrid"]
	): boolean;
}

export interface WebGLSHAmbientUploadHost {
	_gl: WebGL2RenderingContext;
	_shAmbientTexture: WebGLTexture | null;
	_shAmbientTextureWidth: number;
	_shAmbientTextureHeight: number;
}

export interface WebGLLocalLightProbeUploadHost {
	_gl: WebGL2RenderingContext;
	_localLightProbeSHTexture: WebGLTexture | null;
	_localLightProbeSHTextureWidth: number;
	_localLightProbeSHTextureHeight: number;
}

export interface WebGLIrradianceProbeGridUploadHost {
	_gl: WebGL2RenderingContext;
	_irradianceProbeGridSHTexture: WebGLTexture | null;
	_irradianceProbeGridSHTextureWidth: number;
	_irradianceProbeGridSHTextureHeight: number;
}

export function bindWebGLGlobalUniforms(
	host: WebGLGlobalUniformBinderHost,
	sceneProgram: WebGLSceneProgram,
	context: FrameContext
): void {
	const gl = host._gl;
	const uniforms = sceneProgram.uniforms;
	const samplerUnit = (name: string, fallback: number): number =>
		getWebGLSceneSamplerUnit(sceneProgram.samplerLayout, name, fallback);
	const lightState = host._lightState as Partial<WebGLLightState> | null;
	const ambientColorCandidate = lightState?.ambientColor;
	const ambientColor: [number, number, number] =
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
	host._updateFogParams(fogOptions, sceneFogEnabled);
	if (uniforms.fogParams0) {
		gl.uniform4fv(uniforms.fogParams0, host._fogParams0);
	}
	if (uniforms.fogParams1) {
		gl.uniform4fv(uniforms.fogParams1, host._fogParams1);
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
	const localLightProbeTextureReady = host._uploadLocalLightProbeCoefficients(
		localLightProbes
	);
	const hasIrradianceProbeGridTextureUnit =
		host._irradianceProbeGridSamplingSupported;
	const irradianceProbeGridTextureReady =
		hasIrradianceProbeGridTextureUnit &&
		host._uploadIrradianceProbeGridCoefficients(lights.irradianceProbeGrid);
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
		const unit = samplerUnit(
			"uLocalLightProbeCoeffs",
			WEBGL_TEXTURE_UNIT_LOCAL_LIGHT_PROBE_SH,
		);
		gl.activeTexture(gl.TEXTURE0 + unit);
		gl.bindTexture(gl.TEXTURE_2D, host._localLightProbeSHTexture);
		gl.uniform1i(uniforms.localLightProbeCoeffs, unit);
	}
	if (uniforms.localLightProbeCoeffsSize) {
		gl.uniform2f(
			uniforms.localLightProbeCoeffsSize,
			host._localLightProbeSHTextureWidth,
			host._localLightProbeSHTextureHeight
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
		const unit = samplerUnit(
			"uIrradianceProbeGridCoeffs",
			WEBGL_TEXTURE_UNIT_IRRADIANCE_PROBE_GRID_SH,
		);
		gl.activeTexture(gl.TEXTURE0 + unit);
		gl.bindTexture(gl.TEXTURE_2D, host._irradianceProbeGridSHTexture);
		gl.uniform1i(uniforms.irradianceProbeGridCoeffs, unit);
	}
	if (uniforms.irradianceProbeGridCoeffsSize) {
		gl.uniform2f(
			uniforms.irradianceProbeGridCoeffsSize,
			host._irradianceProbeGridSHTextureWidth,
			host._irradianceProbeGridSHTextureHeight
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
		const unit = samplerUnit("uClusterHeaderTexture", WEBGL_TEXTURE_UNIT_CLUSTER_HEADER);
		gl.activeTexture(gl.TEXTURE0 + unit);
		gl.bindTexture(gl.TEXTURE_2D, clusteredState.headerTexture);
		gl.uniform1i(uniforms.clusterHeaderTexture, unit);
	}
	if (uniforms.clusterIndexTexture) {
		const unit = samplerUnit("uClusterIndexTexture", WEBGL_TEXTURE_UNIT_CLUSTER_INDEX);
		gl.activeTexture(gl.TEXTURE0 + unit);
		gl.bindTexture(gl.TEXTURE_2D, clusteredState.indexTexture);
		gl.uniform1i(uniforms.clusterIndexTexture, unit);
	}
	if (uniforms.clusterLightTexture) {
		const unit = samplerUnit("uClusterLightTexture", WEBGL_TEXTURE_UNIT_CLUSTER_LIGHT);
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
		const unit = samplerUnit("uShadowAtlas", WEBGL_TEXTURE_UNIT_SHADOW_ATLAS);
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
		const unit = samplerUnit(
			"uShadowTransmittanceAtlas",
			WEBGL_TEXTURE_UNIT_SHADOW_TRANSMITTANCE,
		);
		gl.activeTexture(gl.TEXTURE0 + unit);
		gl.bindTexture(gl.TEXTURE_2D, shadowSampling.transmittanceTexture);
		gl.uniform1i(
			uniforms.shadowTransmittanceAtlas,
			unit
		);
	}
	if (uniforms.particleShadowVolumeAtlas && shadowSampling.particleVolumeTexture) {
		const unit = samplerUnit(
			"uParticleShadowVolumeAtlas",
			WEBGL_TEXTURE_UNIT_PARTICLE_SHADOW_VOLUME,
		);
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
			const unit = samplerUnit("uEnvSpecularMap", WEBGL_TEXTURE_UNIT_ENV_SPECULAR);
			gl.activeTexture(gl.TEXTURE0 + unit);
			gl.bindTexture(gl.TEXTURE_2D, resolvedEnvSpecular.texture);
			gl.uniform1i(uniforms.envSpecularMap, unit);
		}
		if (uniforms.envSpecularFallbackMap) {
			const unit = samplerUnit(
				"uEnvSpecularFallbackMap",
				WEBGL_TEXTURE_UNIT_ENV_SPECULAR_FALLBACK,
			);
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
			const unit = samplerUnit("uBrdfLUT", WEBGL_TEXTURE_UNIT_BRDF_LUT);
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
	gl.activeTexture(gl.TEXTURE0 + WEBGL_TEXTURE_UNIT_BASE_MAP);

	if (uniforms.taaJitter) {
		gl.uniform4fv(uniforms.taaJitter, host._temporalJitterCurrentPrev);
	}
	if (uniforms.prevViewProjection) {
		const prevViewProjection = sanitizeFloat32Array(
			host._previousViewProjection ??
				toColumnMajorMat4(context.viewCamera.viewProjectionMatrix),
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

export function uploadWebGLSHAmbientCoefficients(
	host: WebGLSHAmbientUploadHost,
	coeffs: SHCoefficients | null | undefined
): boolean {
	const gl = host._gl;
	const texelCount = SH_COEFFICIENT_COUNT;
	const data = new Float32Array(texelCount * 4);
	for (let i = 0; i < texelCount; i++) {
		const coeff = coeffs?.[i];
		const base = i * 4;
		data[base] = finiteOr(coeff?.r, 0);
		data[base + 1] = finiteOr(coeff?.g, 0);
		data[base + 2] = finiteOr(coeff?.b, 0);
		data[base + 3] = 0;
	}

	if (!host._shAmbientTexture) {
		if (typeof gl.createTexture !== "function") {
			logWebGLGlobalUniformWarning(
				"webgl-sh-ambient-texture-create-unsupported",
				"WebGL context does not expose createTexture(); disabling SH for this frame."
			);
			return false;
		}
		host._shAmbientTexture = gl.createTexture();
		if (!host._shAmbientTexture) {
			logWebGLGlobalUniformWarning(
				"webgl-sh-ambient-texture-create-failed",
				"Failed to create WebGL SH ambient texture; disabling SH for this frame."
			);
			return false;
		}
		gl.bindTexture(gl.TEXTURE_2D, host._shAmbientTexture);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
	}

	gl.bindTexture(gl.TEXTURE_2D, host._shAmbientTexture);
	try {
		const internalFormat =
			(
				gl as WebGL2RenderingContext & {
					RGBA32F?: number;
				}
			).RGBA32F ?? gl.RGBA;
		gl.texImage2D(
			gl.TEXTURE_2D,
			0,
			internalFormat,
			SH_COEFFICIENT_COUNT,
			1,
			0,
			gl.RGBA,
			gl.FLOAT,
			data
		);
		host._shAmbientTextureWidth = SH_COEFFICIENT_COUNT;
		host._shAmbientTextureHeight = 1;
		return true;
	} catch (error) {
		logWebGLGlobalUniformWarning(
			"webgl-sh-ambient-texture-upload-failed",
			`WebGL SH ambient texture upload failed; disabling SH for this frame (${String(error)})`
		);
		return false;
	}
}

export function uploadWebGLLocalLightProbeCoefficients(
	host: WebGLLocalLightProbeUploadHost,
	probes: WebGLLightState["localLightProbes"] | null | undefined
): boolean {
	const gl = host._gl;
	const resolvedProbes = Array.isArray(probes) ? probes : [];
	const width = SH_COEFFICIENT_COUNT;
	const height = Math.max(1, resolvedProbes.length);
	const data = new Float32Array(width * height * 4);

	for (let probeIndex = 0; probeIndex < resolvedProbes.length; probeIndex++) {
		const probe = resolvedProbes[probeIndex];
		for (let coeffIndex = 0; coeffIndex < width; coeffIndex++) {
			const coeff = probe.sh[coeffIndex];
			const base = (probeIndex * width + coeffIndex) * 4;
			data[base] = finiteOr(coeff?.r, 0);
			data[base + 1] = finiteOr(coeff?.g, 0);
			data[base + 2] = finiteOr(coeff?.b, 0);
			data[base + 3] = 0;
		}
	}

	if (!host._localLightProbeSHTexture) {
		if (typeof gl.createTexture !== "function") {
			logWebGLGlobalUniformWarning(
				"webgl-local-light-probe-texture-create-unsupported",
				"WebGL context does not expose createTexture(); disabling local light probe SH for this frame."
			);
			return false;
		}
		host._localLightProbeSHTexture = gl.createTexture();
		if (!host._localLightProbeSHTexture) {
			logWebGLGlobalUniformWarning(
				"webgl-local-light-probe-texture-create-failed",
				"Failed to create WebGL local light probe texture; disabling local light probe SH for this frame."
			);
			return false;
		}
		gl.bindTexture(gl.TEXTURE_2D, host._localLightProbeSHTexture);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
	}

	gl.bindTexture(gl.TEXTURE_2D, host._localLightProbeSHTexture);
	try {
		const internalFormat =
			(
				gl as WebGL2RenderingContext & {
					RGBA32F?: number;
				}
			).RGBA32F ?? gl.RGBA;
		gl.texImage2D(
			gl.TEXTURE_2D,
			0,
			internalFormat,
			width,
			height,
			0,
			gl.RGBA,
			gl.FLOAT,
			data
		);
		host._localLightProbeSHTextureWidth = width;
		host._localLightProbeSHTextureHeight = height;
		return true;
	} catch (error) {
		logWebGLGlobalUniformWarning(
			"webgl-local-light-probe-texture-upload-failed",
			`WebGL local light probe texture upload failed; disabling local light probe SH for this frame (${String(error)})`
		);
		return false;
	}
}

export function uploadWebGLIrradianceProbeGridCoefficients(
	host: WebGLIrradianceProbeGridUploadHost,
	grid: WebGLLightState["irradianceProbeGrid"] | null | undefined
): boolean {
	if (!grid || grid.cellCount <= 0) {
		return false;
	}
	const gl = host._gl;
	const width = SH_COEFFICIENT_COUNT;
	const height = Math.max(1, Math.floor(grid.cellCount));
	const data = new Float32Array(width * height * 4);

	for (let cellIndex = 0; cellIndex < height; cellIndex++) {
		const cellSH = grid.sh[cellIndex];
		const valid = grid.validMask[cellIndex] ? 1 : 0;
		for (let coeffIndex = 0; coeffIndex < width; coeffIndex++) {
			const coeff = cellSH?.[coeffIndex];
			const base = (cellIndex * width + coeffIndex) * 4;
			data[base] = finiteOr(coeff?.r, 0);
			data[base + 1] = finiteOr(coeff?.g, 0);
			data[base + 2] = finiteOr(coeff?.b, 0);
			data[base + 3] = valid;
		}
	}

	if (!host._irradianceProbeGridSHTexture) {
		if (typeof gl.createTexture !== "function") {
			logWebGLGlobalUniformWarning(
				"webgl-irradiance-probe-grid-texture-create-unsupported",
				"WebGL context does not expose createTexture(); disabling irradiance probe grid for this frame."
			);
			return false;
		}
		host._irradianceProbeGridSHTexture = gl.createTexture();
		if (!host._irradianceProbeGridSHTexture) {
			logWebGLGlobalUniformWarning(
				"webgl-irradiance-probe-grid-texture-create-failed",
				"Failed to create WebGL irradiance probe grid texture; disabling the grid for this frame."
			);
			return false;
		}
		gl.bindTexture(gl.TEXTURE_2D, host._irradianceProbeGridSHTexture);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
	}

	gl.bindTexture(gl.TEXTURE_2D, host._irradianceProbeGridSHTexture);
	try {
		const internalFormat =
			(
				gl as WebGL2RenderingContext & {
					RGBA32F?: number;
				}
			).RGBA32F ?? gl.RGBA;
		gl.texImage2D(
			gl.TEXTURE_2D,
			0,
			internalFormat,
			width,
			height,
			0,
			gl.RGBA,
			gl.FLOAT,
			data
		);
		host._irradianceProbeGridSHTextureWidth = width;
		host._irradianceProbeGridSHTextureHeight = height;
		return true;
	} catch (error) {
		logWebGLGlobalUniformWarning(
			"webgl-irradiance-probe-grid-texture-upload-failed",
			`WebGL irradiance probe grid texture upload failed; disabling the grid for this frame (${String(error)})`
		);
		return false;
	}
}
