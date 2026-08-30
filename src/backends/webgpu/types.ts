import type { Vec3Tuple } from "../../maths/Vector3";
import type { Vec4Tuple } from "../../maths/Vector4";
import type { Matrix4 } from "../../maths/Matrix4";
import type { IVector3, SHCoefficients } from "../../maths/types";
import type { ResolvedShadowStrategy } from "../../lights/runtime/lightingRuntime";
import type { ClusteredLightingOptions } from "../../pipeline/types";
import { defineTransientKey } from "../../foundation/TransientStore";
import type { ResolvedPostProcessState } from "../../postprocess";
import type { Texture } from "../../core/Texture";
import type { IPrimitive } from "../../core/types";
import type { Material } from "../../materials/Material";
import type { MeshAsset } from "../../meshes/MeshAsset";
import { ParticleBlendMode } from "../../particles";
import type { IRenderBuffer } from "../types";

export interface WebGPUWarning {
	key: string;
	message: string;
}

interface WebGPULightUniformBase {
	color: Vec3Tuple;
}

export interface WebGPUDirectionalLightUniform extends WebGPULightUniformBase {
	direction: Vec3Tuple;
}

export interface WebGPUPointLightUniform extends WebGPULightUniformBase {
	position: Vec3Tuple;
	range: number;
}

export interface WebGPUSpotLightUniform extends WebGPUPointLightUniform {
	direction: Vec3Tuple;
	outerCos: number;
	innerCos: number;
}

export interface WebGPUAreaLightUniform extends WebGPULightUniformBase {
	position: Vec3Tuple;
	range: number;
	right: Vec3Tuple;
	width: number;
	up: Vec3Tuple;
	height: number;
	normal: Vec3Tuple;
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
	position: Vec3Tuple;
	range: number;
	direction: Vec3Tuple;
	outerCos: number;
	innerCos: number;
}

export interface WebGPUClusteredLightUniform extends WebGPULightUniformBase {
	type: WebGPUClusteredLightType;
	position: Vec3Tuple;
	range: number;
	direction: Vec3Tuple;
	outerCos: number;
	innerCos: number;
	right: Vec3Tuple;
	width: number;
	up: Vec3Tuple;
	height: number;
	normal: Vec3Tuple;
	areaScale: number;
	castsShadow: boolean;
	affectsVolumetric: boolean;
	shadowIndex: number;
}

export type WebGPULightingCatalogLightType =
	| "directional"
	| "point"
	| "spot"
	| "area";

export interface WebGPULightingCatalogLight extends WebGPULightUniformBase {
	readonly type: WebGPULightingCatalogLightType;
	readonly source: unknown;
	readonly position: Vec3Tuple;
	readonly range: number;
	readonly direction: Vec3Tuple;
	readonly outerCos: number;
	readonly innerCos: number;
	readonly right: Vec3Tuple;
	readonly width: number;
	readonly up: Vec3Tuple;
	readonly height: number;
	readonly normal: Vec3Tuple;
	readonly areaScale: number;
	readonly shadow: WebGPUShadowData | null;
	readonly shadowIndex: number;
}

export interface WebGPULightingCatalog {
	ambientColor: Vec3Tuple;
	lights: WebGPULightingCatalogLight[];
	warnings: WebGPUWarning[];
}

export interface WebGPUSurfaceLightingView {
	directionalLights: WebGPUDirectionalLightUniform[];
	pointLights: WebGPUPointLightUniform[];
	spotLights: WebGPUSpotLightUniform[];
	areaLights: WebGPUAreaLightUniform[];
	clusteredLights: WebGPUClusteredLightUniform[];
}

export interface WebGPUClusteredLightingData {
	lights: WebGPUClusteredLightUniform[];
	warnings: WebGPUWarning[];
}

export interface WebGPUVolumetricLightingData {
	lights: WebGPUVolumetricLightUniform[];
	warnings: WebGPUWarning[];
}

export interface WebGPUShadowData {
	enabled: boolean;
	strategyType: ResolvedShadowStrategy;
	cascadeCount: number;
	cascadeBlendRatio: number;
	cascadeViewProjectionMatrices: Array<Matrix4 | null>;
	cascadeSplits: Array<Vec4Tuple>;
	depthProjectionParams: Array<Readonly<Vec4Tuple>>;
	viewProjectionMatrix: Matrix4 | null;
	depthBias: number;
	slopeBias: number;
	normalBias: number;
	normalBiasMin: number;
	filterMode: "pcf" | "pcss";
	samplingQuality: "low" | "medium" | "high";
	shadowStrength: number;
	shadowMapBaseSize: number;
	shadowMapSize: number;
	atlasTileSize: number;
}

export interface WebGPULightingState {
	ambientColor: Vec3Tuple;
	directionalLights: WebGPUDirectionalLightUniform[];
	directionalShadows: WebGPUShadowData[];
	pointLights: WebGPUPointLightUniform[];
	spotLights: WebGPUSpotLightUniform[];
	spotShadows: WebGPUShadowData[];
	areaLights: WebGPUAreaLightUniform[];
	warnings: WebGPUWarning[];
}

export interface WebGPUFeatureState {
	enableLighting: boolean;
	enableSH: boolean;
	enableShadows: boolean;
	enableReflection: boolean;
	enableEnvironment: boolean;
	enableOIT: boolean;
	enableClusteredLighting: boolean;
	clusteredLightingOptions?: ClusteredLightingOptions;
	/** Per-frame resolved post-process registry snapshot. */
	postProcess: ResolvedPostProcessState;
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

export type WebGPUClusteredCullingMode = "gather" | "scatter";

export interface WebGPUClusteredAreaPayload {
	rightWidth: Vec4Tuple;
	upHeight: Vec4Tuple;
	normalAreaScale: Vec4Tuple;
}

export interface WebGPUClusterLightRecord {
	positionRange: Vec4Tuple;
	directionOuter: Vec4Tuple;
	colorInner: Vec4Tuple;
	rightWidth: Vec4Tuple;
	upHeight: Vec4Tuple;
	normalAreaScale: Vec4Tuple;
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
	localLightProbeCount: number;
	localLightProbes: WebGPULocalLightProbeUniform[];
	irradianceProbeGrid: WebGPUIrradianceProbeGridUniform | null;
	reflectionProbeCount: number;
	reflectionProbes: WebGPUReflectionProbeUniform[];
	brdfLUTTexture: Texture | null;
	envSpecularMaxMipLevel: number;
	warnings: WebGPUWarning[];
}

export interface WebGPULocalLightProbeUniform {
	id: string;
	worldToProbeMatrix: Matrix4;
	invHalfExtents: Vec3Tuple;
	radiusInv: number;
	shape: 0 | 1;
	blendDistance: number;
	priority: number;
	sh: SHCoefficients;
}

export interface WebGPUIrradianceProbeGridUniform {
	id: string;
	worldToGridMatrix: Matrix4;
	dimensions: Vec3Tuple;
	invHalfExtents: Vec3Tuple;
	blendDistance: number;
	cellCount: number;
	textureRevision: number;
	sh: SHCoefficients[];
	validMask: Uint8Array;
}

export interface WebGPUReflectionProbeUniform {
	id: string;
	worldToProbeMatrix: Matrix4;
	probeToWorldMatrix: Matrix4;
	invHalfExtents: Vec3Tuple;
	radiusInv: number;
	captureWorldPosition: Vec3Tuple;
	shape: 0 | 1;
	parallaxMode: 0 | 1 | 2;
	blendDistance: number;
	blendExponent: number;
	layer: number;
}

export interface WebGPUTextureSlotData {
	map: Texture | null;
	transformA: Vec4Tuple;
	transformB: Vec4Tuple;
}

export interface WebGPUShaderUniformData {
	cacheKey: string;
	byteLength: number;
	valueRevision: number;
	data: Uint8Array<ArrayBuffer> | null;
}

export type WebGPUShadingFamily = "pbr" | "phong" | "flat" | "unlit";

export interface WebGPUMaterialCommonUniformData {
	baseColorFactor: Vec4Tuple;
	emissiveFactor: Vec4Tuple;
	materialParams: Vec4Tuple;
	renderParams: Vec4Tuple;
	textureSlots: WebGPUTextureSlotData[];
}

export interface WebGPUPBRMaterialUniformData {
	surfaceParams0: Vec4Tuple;
	surfaceParams1: Vec4Tuple;
	surfaceParams2: Vec4Tuple;
	surfaceParams3: Vec4Tuple;
	specularColorFactor: Vec4Tuple;
	sheenColorClearcoatNormalScale: Vec4Tuple;
	attenuationColor: Vec4Tuple;
	anisotropyParams: Vec4Tuple;
	pbrMasks: Vec4Tuple;
}

export interface WebGPUPhongMaterialUniformData {
	ambientShininess: Vec4Tuple;
	specular: Vec4Tuple;
}

export type WebGPUFlatMaterialUniformData = WebGPUPhongMaterialUniformData;

interface WebGPUMaterialUniformDataBase {
	common: WebGPUMaterialCommonUniformData;
	shaderUniforms: WebGPUShaderUniformData;
	pipelineKey: string;
	warnings: WebGPUWarning[];
}

export type WebGPUMaterialUniformData =
	| (WebGPUMaterialUniformDataBase & {
			shadingFamily: "pbr";
			lighting: WebGPUPBRMaterialUniformData;
	  })
	| (WebGPUMaterialUniformDataBase & {
			shadingFamily: "phong";
			lighting: WebGPUPhongMaterialUniformData;
	  })
	| (WebGPUMaterialUniformDataBase & {
			shadingFamily: "flat";
			lighting: WebGPUFlatMaterialUniformData;
	  })
	| (WebGPUMaterialUniformDataBase & {
			shadingFamily: "unlit";
			lighting: null;
	  });

export interface WebGPUFrameUniformInput {
	viewProjectionMatrix: Matrix4 | number[][];
	prevViewProjectionMatrix: Matrix4 | number[][];
	cameraPosition: IVector3;
	environmentRight: Vec3Tuple;
	environmentUp: Vec3Tuple;
	environmentBackward: Vec3Tuple;
	environmentTanHalfFov: number;
	environmentAspect: number;
	environmentIsOrthographic: boolean;
	ambientColor: WebGPULightingState["ambientColor"];
	shAmbientCoeffs: SHCoefficients | null;
	localLightProbeCount: number;
	localLightProbes: WebGPULocalLightProbeUniform[];
	irradianceProbeGrid: WebGPUIrradianceProbeGridUniform | null;
	directionalLights: WebGPULightingState["directionalLights"];
	directionalShadows: WebGPULightingState["directionalShadows"];
	pointLights: WebGPULightingState["pointLights"];
	spotLights: WebGPULightingState["spotLights"];
	spotShadows: WebGPULightingState["spotShadows"];
	areaLights: WebGPULightingState["areaLights"];
	reflectionProbeCount: number;
	reflectionProbes: WebGPUReflectionProbeUniform[];
	enableLighting: boolean;
	enableShadows: boolean;
	enableSH: boolean;
	enableClusteredLighting: boolean;
	hasSHAmbient: boolean;
	environmentIsLinear: boolean;
	hasEnvSpecular: boolean;
	hasBRDFLUT: boolean;
	envSpecularMaxMipLevel: number;
	taaJitterCurrentPrev: Vec4Tuple;
}

export interface WebGPUParticleDrawBatch {
	systemId: string;
	templateIndex?: number;
	templateId?: string;
	blendMode: ParticleBlendMode;
	texture: Texture | null;
	receiveShadows: boolean;
	castShadows: boolean;
	shadowDensity: number;
	shadowSoftness: number;
	instanceBuffer: IRenderBuffer;
	instanceCount: number;
	indirectBuffer: IRenderBuffer;
	indirectOffset: number;
}

export interface WebGPUParticleMeshDrawBatch {
	systemId: string;
	templateIndex: number;
	templateId?: string;
	mesh: MeshAsset;
	primitive: IPrimitive;
	material: Material;
	receiveShadows: boolean;
	castShadows: boolean;
	shadowDensity: number;
	shadowSoftness: number;
	instanceBuffer: IRenderBuffer;
	instanceCount: number;
	indirectBuffer: IRenderBuffer;
	indirectOffset: number;
}

export const WEBGPU_PARTICLE_DRAW_BATCHES_KEY =
	defineTransientKey<WebGPUParticleDrawBatch[]>(
		"webgpu:particle-draw-batches"
	);

export const WEBGPU_PARTICLE_MESH_DRAW_BATCHES_KEY =
	defineTransientKey<WebGPUParticleMeshDrawBatch[]>(
		"webgpu:particle-mesh-draw-batches"
	);
