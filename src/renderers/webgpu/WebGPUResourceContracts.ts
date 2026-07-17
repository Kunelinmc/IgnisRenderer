import type { Texture } from "../../core/Texture";
import type {
	DrawPacket,
	FrameContext,
	ParticleMeshRenderBatch,
	ParticleMeshRenderItem,
} from "../../pipeline/types";
import type { ICommandEncoder } from "../ICommandEncoder";
import type {
	IBindingGroup,
	IComputePipeline,
	IRenderBuffer,
	IRenderPipeline,
	IRenderTexture,
	ISampler,
} from "../types";
import type { WebGPUFrameFeatureDataStore } from "./FrameFeatures";
import type {
	WebGPUEnvironmentState,
	WebGPUFeatureState,
	WebGPULightingState,
} from "./types";
import type {
	WebGPUScenePipelineDrawMode,
	WebGPUSceneTargetMode,
	WebGPUTransparentPipelineMode,
} from "./WebGPUPipelineLibrary";
import type { WebGPUTemporalStateMode } from "./WebGPUFrameBindingCache";
import type {
	JointMatrixMap,
	MorphWeightMap,
} from "../../simulation/animation/types";
import type {
	WebGPUPagedShadowFrameRequest,
	WebGPUPagedShadowSamplingResources,
} from "./WebGPUPagedShadowRuntime";
import type { ParticleBlendMode } from "../../particles";

/** @internal WebGPU scene draw resolution result. */
export interface WebGPUDrawResources {
	pipeline: IRenderPipeline;
	frameBinding: IBindingGroup;
	modelBinding: IBindingGroup;
	clusteredBinding: IBindingGroup;
	vertexBuffer: IRenderBuffer;
	indexBuffer: IRenderBuffer;
	indexCount: number;
}

/** @internal WebGPU environment draw resolution result. */
export interface WebGPUEnvironmentDrawResources {
	pipeline: IRenderPipeline;
	frameBinding: IBindingGroup;
}

/** @internal Render-pass attachments accepted by the billboard particle renderer. */
export interface WebGPUParticlePassTargets {
	colorAttachments: Array<{
		view: unknown;
		resolveTarget?: unknown;
		clearValue?: { r: number; g: number; b: number; a: number };
		loadOp: "clear" | "load";
		storeOp: "store" | "discard";
	}>;
	depth: unknown;
	label: string;
}

/** @internal Scene draw pipeline selection. */
export interface WebGPUDrawResourceOptions {
	transparentPipelineMode?: WebGPUTransparentPipelineMode;
	sceneTargetMode?: WebGPUSceneTargetMode;
	drawMode?: WebGPUScenePipelineDrawMode;
	sampleCountOverride?: number;
}

/** @internal Environment pipeline selection. */
export interface WebGPUEnvironmentResourceOptions {
	sampleCountOverride?: number;
}

/** @internal Billboard particle pipeline selection. */
export interface WebGPUParticleRenderOptions {
	includeBlendModes?: readonly ParticleBlendMode[];
	pipelineMode?: "legacy" | "oit";
	sampleCountOverride?: number;
}

/** @internal Mesh particle packet filters. */
export interface WebGPUParticleMeshPacketOptions {
	includeOpaque?: boolean;
	includeTransparent?: boolean;
	includeShadowCasters?: boolean;
	includeShadowTransmitters?: boolean;
	includeReflective?: boolean;
}

/** @internal Scoped frame preparation options. */
export interface WebGPUPrepareFrameOptions {
	readonly sceneTargetMode: WebGPUSceneTargetMode;
	readonly temporalStateMode?: WebGPUTemporalStateMode;
}

/** @internal Prepared data and bindings for one WebGPU frame scope. */
export interface WebGPUPreparedFrameResources {
	readonly sceneTargetMode: WebGPUSceneTargetMode;
	frameBinding: IBindingGroup;
	decalFrameBinding: IBindingGroup;
	environmentBinding: IBindingGroup;
	clusteredSceneBinding: IBindingGroup;
	readonly lightingState: WebGPULightingState;
	readonly featureData: WebGPUFrameFeatureDataStore;
	readonly featureState: WebGPUFeatureState;
	readonly environmentState: WebGPUEnvironmentState;
	readonly jointMatrixMap: JointMatrixMap | null;
	readonly morphWeightMap: MorphWeightMap | null;
}

/** @internal Owned WebGPU frame-binding and clustered-lighting scope. */
export interface WebGPUFrameResourceScope {
	prepare(
		context: FrameContext,
		options: WebGPUPrepareFrameOptions,
	): WebGPUPreparedFrameResources;
	updateParticleShadowVolumes(
		context: FrameContext,
	): void;
	destroy(): void;
}

/** @internal Frame preparation and scoped binding ownership. */
export interface WebGPUFrameResourceProvider {
	createFrameScope(): WebGPUFrameResourceScope;
}

/** @internal Scene pipeline, environment, and clustered-lighting capability. */
export interface WebGPUSceneResourceProvider {
	getDrawResources(
		packet: DrawPacket,
		frameResources: WebGPUPreparedFrameResources,
		options?: WebGPUDrawResourceOptions,
	): Promise<WebGPUDrawResources[] | null>;
	getEnvironmentResources(
		frameResources: WebGPUPreparedFrameResources,
		sceneTargetMode?: WebGPUSceneTargetMode,
		options?: WebGPUEnvironmentResourceOptions,
	): Promise<WebGPUEnvironmentDrawResources | null>;
	buildClusteredLighting(
		encoder: ICommandEncoder,
		frameResources: WebGPUPreparedFrameResources,
	): Promise<void>;
}

/** @internal Texture and sampler capability used by material/decal paths. */
export interface WebGPUTextureResourceProvider {
	getTextureForSlot(texture: Texture | null, slotIndex: number): IRenderTexture;
	getTextureForSlotAsync(
		texture: Texture | null,
		slotIndex: number,
	): Promise<IRenderTexture>;
	getSamplerForTexture(texture: Texture | null): ISampler;
	registerExternalTexture(
		texture: Texture,
		resource: IRenderTexture,
		uploadedVersion?: number,
		mipLevelCount?: number,
	): void;
	unregisterExternalTexture(texture: Texture): void;
}

/** @internal Billboard rendering and mesh-particle packet capability. */
export interface WebGPUParticleRenderProvider {
	buildParticleMeshDrawPackets(
		context: FrameContext,
		options?: WebGPUParticleMeshPacketOptions,
	): DrawPacket[];
	renderParticles(
		encoder: ICommandEncoder,
		context: FrameContext,
		targets: WebGPUParticlePassTargets,
		frameResources: WebGPUPreparedFrameResources,
		mode: WebGPUSceneTargetMode,
		options?: WebGPUParticleRenderOptions,
	): Promise<number>;
}

/** @internal Deferred G-buffer and decal pipeline capability. */
export interface WebGPUDeferredResourceProvider {
	getGBufferWriteLayout(): GPUBindGroupLayout;
	getGBufferReadLayout(): GPUBindGroupLayout;
	getDecalBindGroupLayout(): GPUBindGroupLayout;
	getDecalOutputBindGroupLayout(): GPUBindGroupLayout;
	getDecalBatchBindGroupLayout(): GPUBindGroupLayout;
	getDeferredUnusedBinding(): IBindingGroup;
	getDeferredLightingPipeline(): Promise<IRenderPipeline>;
	getDecalPipeline(): Promise<IRenderPipeline>;
	getDecalBatchPipeline(): Promise<IComputePipeline>;
}

/** @internal Planar-reflection composite layout capability. */
export interface WebGPUPlanarReflectionResourceProvider {
	getPlanarReflectionLayout(): GPUBindGroupLayout;
}

/** @internal Regular and paged-shadow recording capability. */
export interface WebGPUShadowRenderProvider {
	preparePagedShadowFrame(request: WebGPUPagedShadowFrameRequest): void;
	recordPagedShadowPageMarkPass(request: WebGPUPagedShadowFrameRequest): Promise<void>;
	recordPagedShadowPageAllocationPass(request: WebGPUPagedShadowFrameRequest): Promise<void>;
	recordPagedShadowPageTableCopyPass(request: WebGPUPagedShadowFrameRequest): Promise<void>;
	recordPagedShadowDepthPass(request: WebGPUPagedShadowFrameRequest): Promise<void>;
	recordPagedShadowFeedbackPass(request: WebGPUPagedShadowFrameRequest): Promise<void>;
	getPagedShadowSamplingResources(): WebGPUPagedShadowSamplingResources;
}

export type { ParticleMeshRenderBatch, ParticleMeshRenderItem };
