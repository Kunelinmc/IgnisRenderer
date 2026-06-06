import type { Matrix4 } from "../../maths/Matrix4";
import type { IVector3, SHCoefficients } from "../../maths/types";
import type { getPrimaryShadowMap, ShadowStrategyType } from "../../lights/shadows/ShadowMapping";
import type { ClusteredLightingOptions } from "../../pipeline/types";
import type { BloomOptions } from "../../postprocess/passes/BloomPass";
import type { FogOptions } from "../../postprocess/passes/FogPass";
import type { TAAOptions } from "../../postprocess/passes/TemporalAntiAliasingPass";
import type { Texture } from "../../core/Texture";

export interface WebGPUWarning {
	key: string;
	message: string;
}

export type WebGPUVec3 = [number, number, number];

interface WebGPULightUniformBase {
	color: WebGPUVec3;
}

export interface WebGPUDirectionalLightUniform extends WebGPULightUniformBase {
	direction: WebGPUVec3;
}

export interface WebGPUPointLightUniform extends WebGPULightUniformBase {
	position: WebGPUVec3;
	range: number;
}

export interface WebGPUSpotLightUniform extends WebGPUPointLightUniform {
	direction: WebGPUVec3;
	outerCos: number;
	innerCos: number;
}

export interface WebGPUAreaLightUniform extends WebGPULightUniformBase {
	position: WebGPUVec3;
	range: number;
	right: WebGPUVec3;
	width: number;
	up: WebGPUVec3;
	height: number;
	normal: WebGPUVec3;
	areaScale: number;
}

export type WebGPUDirectionalLight = WebGPUDirectionalLightUniform;
export type WebGPUPointLight = WebGPUPointLightUniform;
export type WebGPUSpotLight = WebGPUSpotLightUniform;
export type WebGPUAreaLight = WebGPUAreaLightUniform;
export type WebGPUVolumetricLightType = 0 | 1 | 2;
export type WebGPUClusteredLightType = 0 | 1 | 2;

export interface WebGPUVolumetricLightUniform extends WebGPULightUniformBase {
	type: WebGPUVolumetricLightType;
	position: WebGPUVec3;
	range: number;
	direction: WebGPUVec3;
	outerCos: number;
	innerCos: number;
}

export interface WebGPUClusteredLightUniform extends WebGPULightUniformBase {
	type: WebGPUClusteredLightType;
	position: WebGPUVec3;
	range: number;
	direction: WebGPUVec3;
	outerCos: number;
	innerCos: number;
	right: WebGPUVec3;
	width: number;
	up: WebGPUVec3;
	height: number;
	normal: WebGPUVec3;
	areaScale: number;
	castsShadow: boolean;
	affectsVolumetric: boolean;
	shadowIndex: number;
}

export interface WebGPUShadowData {
	enabled: boolean;
	strategyType: ShadowStrategyType;
	cascadeCount: number;
	cascadeBlendRatio: number;
	cascadeViewProjectionMatrices: Array<Matrix4 | null>;
	cascadeSplits: Array<[number, number, number, number]>;
	viewProjectionMatrix: Matrix4 | null;
	depthBias: number;
	slopeBias: number;
	normalBias: number;
	normalBiasMin: number;
	pcfRadius: number;
	pcssEnabled: boolean;
	pcssRadius: number;
	shadowSamples: number;
	shadowSearchSamples: number;
	shadowStrength: number;
	shadowMapBaseSize: number;
	shadowMapSize: number;
	atlasTileSize: number;
	shadowMap: ReturnType<typeof getPrimaryShadowMap>;
}

export interface WebGPULightingState {
	ambientColor: WebGPUVec3;
	directionalLights: WebGPUDirectionalLightUniform[];
	directionalShadows: WebGPUShadowData[];
	pointLights: WebGPUPointLightUniform[];
	spotLights: WebGPUSpotLightUniform[];
	spotShadows: WebGPUShadowData[];
	areaLights: WebGPUAreaLightUniform[];
	clusteredLights: WebGPUClusteredLightUniform[];
	volumetricLights: WebGPUVolumetricLightUniform[];
	warnings: WebGPUWarning[];
}

export interface WebGPUFeatureState {
	enableLighting: boolean;
	enableGamma: boolean;
	enableToneMapping: boolean;
	enableSH: boolean;
	enableShadows: boolean;
	enableReflection: boolean;
	enableEnvironment: boolean;
	enableOIT: boolean;
	enableSSAO: boolean;
	enableSSGI: boolean;
	enableTAA: boolean;
	enableSSR: boolean;
	enableVolumetric: boolean;
	enableFog: boolean;
	enableBloom: boolean;
	enableClusteredLighting: boolean;
	taaOptions?: TAAOptions;
	fogOptions?: FogOptions;
	bloomOptions?: BloomOptions;
	clusteredLightingOptions?: ClusteredLightingOptions;
	warnings: WebGPUWarning[];
}

export interface WebGPUClusterGridParams {
	screenWidth: number;
	screenHeight: number;
	tilesX: number;
	tilesY: number;
	zSlices: number;
	clusterCount: number;
	near: number;
	far: number;
	logScale: number;
	logBias: number;
	lightCount: number;
	maxLightsPerCluster: number;
}

export interface WebGPUClusterLightRecord {
	positionRange: [number, number, number, number];
	directionOuter: [number, number, number, number];
	colorInner: [number, number, number, number];
	rightWidth: [number, number, number, number];
	upHeight: [number, number, number, number];
	normalAreaScale: [number, number, number, number];
	packedFlags: number;
	shadowIndex: number;
	reserved0: number;
	reserved1: number;
}

export interface WebGPUClusterHeader {
	offset: number;
	count: number;
	flags: number;
	reserved: number;
}

export interface WebGPUEnvironmentState {
	shAmbientCoeffs: SHCoefficients | null;
	enableSH: boolean;
	hasSHAmbient: boolean;
	environmentTexture: Texture | null;
	envSpecularTexture: Texture | null;
	envSpecularFallbackTexture: Texture | null;
	localLightProbeCount: number;
	localLightProbes: WebGPULocalLightProbeUniform[];
	reflectionProbeCount: number;
	reflectionProbes: WebGPUReflectionProbeUniform[];
	brdfLUTTexture: Texture | null;
	envSpecularMaxMipLevel: number;
	envSpecularFallbackMaxMipLevel: number;
	warnings: WebGPUWarning[];
}

export interface WebGPULocalLightProbeUniform {
	id: string;
	worldToProbeMatrix: Matrix4;
	invHalfExtents: [number, number, number];
	radiusInv: number;
	shape: 0 | 1;
	blendDistance: number;
	priority: number;
	sh: SHCoefficients;
}

export interface WebGPUReflectionProbeUniform {
	id: string;
	worldToProbeMatrix: Matrix4;
	probeToWorldMatrix: Matrix4;
	invHalfExtents: [number, number, number];
	radiusInv: number;
	captureWorldPosition: [number, number, number];
	shape: 0 | 1;
	parallaxMode: 0 | 1 | 2;
	blendDistance: number;
	blendExponent: number;
	layer: number;
}

export interface WebGPUTextureSlotData {
	map: Texture | null;
	transformA: [number, number, number, number];
	transformB: [number, number, number, number];
}

export interface WebGPUShaderUniformData {
	cacheKey: string;
	byteLength: number;
	valueRevision: number;
	data: Uint8Array<ArrayBuffer> | null;
}

export interface WebGPUMaterialUniformData {
	baseColorFactor: [number, number, number, number];
	emissiveFactor: [number, number, number, number];
	surfaceParams0: [number, number, number, number];
	surfaceParams1: [number, number, number, number];
	surfaceParams2: [number, number, number, number];
	surfaceParams3: [number, number, number, number];
	specularColorFactor: [number, number, number, number];
	phongAmbientShininess: [number, number, number, number];
	phongSpecularShading: [number, number, number, number];
	sheenColorClearcoatNormalScale: [number, number, number, number];
	attenuationColor: [number, number, number, number];
	anisotropyParams: [number, number, number, number];
	anisotropyTexture: WebGPUTextureSlotData;
	materialFlags: [number, number, number, number];
	textureSlots: WebGPUTextureSlotData[];
	shaderUniforms: WebGPUShaderUniformData;
	pipelineKey: string;
	warnings: WebGPUWarning[];
}

export interface WebGPUFrameUniformInput {
	viewProjectionMatrix: Matrix4 | number[][];
	prevViewProjectionMatrix: Matrix4 | number[][];
	cameraPosition: IVector3;
	environmentRight: WebGPUVec3;
	environmentUp: WebGPUVec3;
	environmentBackward: WebGPUVec3;
	environmentTanHalfFov: number;
	environmentAspect: number;
	environmentIsOrthographic: boolean;
	ambientColor: WebGPULightingState["ambientColor"];
	shAmbientCoeffs: SHCoefficients | null;
	localLightProbeCount: number;
	localLightProbes: WebGPULocalLightProbeUniform[];
	directionalLights: WebGPULightingState["directionalLights"];
	directionalShadows: WebGPULightingState["directionalShadows"];
	pointLights: WebGPULightingState["pointLights"];
	spotLights: WebGPULightingState["spotLights"];
	spotShadows: WebGPULightingState["spotShadows"];
	areaLights: WebGPULightingState["areaLights"];
	reflectionProbeCount: number;
	reflectionProbes: WebGPUReflectionProbeUniform[];
	enableLighting: boolean;
	enableGamma: boolean;
	enableShadows: boolean;
	enableSH: boolean;
	enableClusteredLighting: boolean;
	encodeGammaInShader: boolean;
	hasSHAmbient: boolean;
	hasEnvironment: boolean;
	environmentIsLinear: boolean;
	hasEnvSpecular: boolean;
	hasEnvSpecularFallback: boolean;
	hasBRDFLUT: boolean;
	envSpecularMaxMipLevel: number;
	envSpecularFallbackMaxMipLevel: number;
	taaJitterCurrentPrev: [number, number, number, number];
}
