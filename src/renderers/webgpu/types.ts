import type { Matrix4 } from "../../maths/Matrix4";
import type { IVector3, SHCoefficients } from "../../maths/types";
import type {
	BloomOptions,
	ClusteredLightingOptions,
	TAAOptions,
} from "../../pipeline/types";
import type { ShadowMap } from "../../lights/ShadowMapping";
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

export type WebGPUDirectionalLight = WebGPUDirectionalLightUniform;
export type WebGPUPointLight = WebGPUPointLightUniform;
export type WebGPUSpotLight = WebGPUSpotLightUniform;
export type WebGPUVolumetricLightType = 0 | 1 | 2;
export type WebGPUClusteredLightType = 0 | 1;

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
	castsShadow: boolean;
	affectsVolumetric: boolean;
	shadowIndex: number;
}

export interface WebGPUShadowData {
	enabled: boolean;
	viewProjectionMatrix: Matrix4 | null;
	depthBias: number;
	slopeBias: number;
	normalBias: number;
	normalBiasMin: number;
	pcfRadius: number;
	shadowStrength: number;
	shadowMapSize: number;
	atlasTileSize: number;
	shadowMap: ShadowMap | null;
}

export interface WebGPULightingState {
	ambientColor: WebGPUVec3;
	directionalLights: WebGPUDirectionalLightUniform[];
	directionalShadows: WebGPUShadowData[];
	pointLights: WebGPUPointLightUniform[];
	spotLights: WebGPUSpotLightUniform[];
	spotShadows: WebGPUShadowData[];
	clusteredLights: WebGPUClusteredLightUniform[];
	volumetricLights: WebGPUVolumetricLightUniform[];
	warnings: WebGPUWarning[];
}

export interface WebGPUFeatureState {
	enableLighting: boolean;
	enableGamma: boolean;
	enableSH: boolean;
	enableShadows: boolean;
	enableReflection: boolean;
	enableSkybox: boolean;
	enableSSAO: boolean;
	enableTAA: boolean;
	enableSSR: boolean;
	enableVolumetric: boolean;
	enableBloom: boolean;
	enableClusteredLighting: boolean;
	taaOptions?: TAAOptions;
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
}

export interface WebGPUClusterLightRecord {
	positionRange: [number, number, number, number];
	directionOuter: [number, number, number, number];
	colorInner: [number, number, number, number];
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
	skyboxTexture: Texture | null;
	envSpecularTexture: Texture | null;
	brdfLUTTexture: Texture | null;
	envSpecularMaxMipLevel: number;
	warnings: WebGPUWarning[];
}

export interface WebGPUTextureSlotData {
	map: Texture | null;
	transformA: [number, number, number, number];
	transformB: [number, number, number, number];
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
	materialFlags: [number, number, number, number];
	textureSlots: WebGPUTextureSlotData[];
	pipelineKey: string;
	warnings: WebGPUWarning[];
}

export interface WebGPUFrameUniformInput {
	viewProjectionMatrix: Matrix4 | number[][];
	prevViewProjectionMatrix: Matrix4 | number[][];
	cameraPosition: IVector3;
	skyboxRight: WebGPUVec3;
	skyboxUp: WebGPUVec3;
	skyboxBackward: WebGPUVec3;
	skyboxTanHalfFov: number;
	skyboxAspect: number;
	skyboxIsOrthographic: boolean;
	ambientColor: WebGPULightingState["ambientColor"];
	shAmbientCoeffs: SHCoefficients | null;
	directionalLights: WebGPULightingState["directionalLights"];
	directionalShadows: WebGPULightingState["directionalShadows"];
	pointLights: WebGPULightingState["pointLights"];
	spotLights: WebGPULightingState["spotLights"];
	spotShadows: WebGPULightingState["spotShadows"];
	enableLighting: boolean;
	enableGamma: boolean;
	enableShadows: boolean;
	enableSH: boolean;
	enableClusteredLighting: boolean;
	encodeGammaInShader: boolean;
	hasSHAmbient: boolean;
	hasSkybox: boolean;
	skyboxIsLinear: boolean;
	hasEnvSpecular: boolean;
	hasBRDFLUT: boolean;
	envSpecularMaxMipLevel: number;
	taaJitterCurrentPrev: [number, number, number, number];
}
