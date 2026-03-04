import type { Matrix4 } from "../../../maths/Matrix4";
import type { IVector3 } from "../../../maths/types";
import type { ShadowMap } from "../../../utils/ShadowMapping";
import type { Texture } from "../../Texture";

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

export interface WebGPUShadowData {
	enabled: boolean;
	viewProjectionMatrix: Matrix4 | null;
	depthBias: number;
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
	enableVolumetric: boolean;
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
	cameraPosition: IVector3;
	ambientColor: WebGPULightingState["ambientColor"];
	directionalLights: WebGPULightingState["directionalLights"];
	directionalShadows: WebGPULightingState["directionalShadows"];
	pointLights: WebGPULightingState["pointLights"];
	spotLights: WebGPULightingState["spotLights"];
	spotShadows: WebGPULightingState["spotShadows"];
	enableLighting: boolean;
	enableGamma: boolean;
	enableShadows: boolean;
}
