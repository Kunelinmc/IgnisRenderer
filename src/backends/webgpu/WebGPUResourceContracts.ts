import type { Texture } from "../../core/Texture";
import type { PrimitiveDrawTopology } from "../../core/types";
import type {
	DrawPacket,
	FrameContext,
} from "../../pipeline/types";
import type { PreparedFramePacketSet } from "../../pipeline/FramePackets";
import type { ICommandEncoder } from "../ICommandEncoder";
import type {
	IBindingGroup,
	IComputePipeline,
	IndexFormat,
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
	WebGPUMaterialUniformData,
} from "./types";
import type { WebGPUVertexBufferBinding } from "./WebGPUGeometryRegistry";
import type { WebGPUGeometryHandle } from "./WebGPUGeometryRegistry";
import type {
	WebGPUDrawPassDescriptor,
	WebGPUDrawPipelineMode,
	WebGPUSceneTargetMode,
	WebGPUTransparentPipelineMode,
} from "./WebGPUScenePassDescriptors";
import type { WebGPUTemporalStateMode } from "./WebGPUFrameBindingCache";
import type {
	JointMatrixMap,
	MorphWeightMap,
} from "../../simulation/animation/types";
import type { FramePreparationRequirements } from "../../pipeline/FrameRequirements";
import type { WebGPUPagedShadowFrameRequest } from "./WebGPUPagedShadowTechnique";
import type { WebGPUPagedShadowFrameState } from "./WebGPUPagedShadowExperiment";
import type { ParticleBlendMode } from "../../particles";
import type { WebGPUDeferredGBufferLayout } from "./constants";
import type { WebGPUMaterialPipelineState } from "./WebGPUMaterialPipelineResolver";

/** @internal Inputs retained with a resolved WebGPU scene draw. */
export interface WebGPUResolvedDrawInputs {
	readonly materialData: WebGPUMaterialUniformData;
	readonly textures: readonly IRenderTexture[];
	readonly samplers: readonly ISampler[];
	readonly geometry: {
		readonly vertexBindings: readonly WebGPUVertexBufferBinding[];
		readonly indexBuffer: IRenderBuffer;
		readonly indexFormat: IndexFormat;
		readonly indexCount: number;
		readonly topology: PrimitiveDrawTopology;
		readonly wireframeIndexBuffer: IRenderBuffer | null;
		readonly wireframeIndexFormat: IndexFormat;
		readonly wireframeIndexCount: number;
		readonly vertexCount: number;
		readonly morphTargetCount: number;
		readonly morphSemanticMask: number;
		readonly morphPositionBuffer: IRenderBuffer | null;
		readonly morphNormalBuffer: IRenderBuffer | null;
	};
}

/** @internal WebGPU scene draw resolution result. */
export interface WebGPUDrawResources {
	pipeline: IRenderPipeline;
	frameBinding: IBindingGroup;
	modelBinding: IBindingGroup;
	clusteredBinding: IBindingGroup;
	vertexBindings: readonly WebGPUVertexBufferBinding[];
	indexBuffer: IRenderBuffer;
	indexFormat: IndexFormat;
	indexCount: number;
	/** @internal Key shared by compatible static instance draws. */
	staticBatchKey?: string;
	/** @internal Frame-arena instance record used by static batching. */
	firstInstance?: number;
	resolvedInputs: WebGPUResolvedDrawInputs;
}

/** @internal Immutable inputs accepted by a feature-owned draw pipeline provider. */
export interface WebGPUDrawPipelineRequest {
	readonly materialState: WebGPUMaterialPipelineState;
	readonly pass: WebGPUDrawPassDescriptor;
	readonly topology: PrimitiveDrawTopology;
	readonly geometryLayout: {
		readonly layoutKey: string;
		readonly sceneVertexLayouts: WebGPUGeometryHandle["sceneVertexLayouts"];
	};
	readonly sampleCount: number;
}

/** @internal Feature-owned pipeline selection used by shared draw preparation. */
export interface WebGPUDrawPipelineProvider {
	resolvePipeline(
		request: WebGPUDrawPipelineRequest,
	): Promise<IRenderPipeline | null>;
}

/** @internal WebGPU environment draw resolution result. */
export interface WebGPUEnvironmentDrawResources {
	pipeline: IRenderPipeline;
	frameBinding: IBindingGroup;
}

/** @internal Render-pass attachments accepted by the billboard particle renderer. */
export interface WebGPUParticlePassTargets {
	readonly sampleCount: number;
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
	drawMode?: WebGPUDrawPipelineMode;
	deferredGBufferLayout?: WebGPUDeferredGBufferLayout;
	sampleCount: number;
}

/** @internal Environment pipeline selection. */
export interface WebGPUEnvironmentResourceOptions {
	sampleCount: number;
}

/** @internal Billboard particle pipeline selection. */
export interface WebGPUParticleRenderOptions {
	includeBlendModes?: readonly ParticleBlendMode[];
	pipelineMode?: "legacy" | "oit";
}

/** @internal Scoped frame preparation options. */
export interface WebGPUFrameScopePrepareOptions {
	readonly sceneTargetMode: WebGPUSceneTargetMode;
	readonly framePackets: PreparedFramePacketSet;
	readonly temporalStateMode?: WebGPUTemporalStateMode;
	readonly frameRequirements?: FramePreparationRequirements;
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
		options: WebGPUFrameScopePrepareOptions,
	): WebGPUPreparedFrameResources;
	updateParticleShadowVolumes(
		context: FrameContext,
	): void;
	destroy(): void;
}

/** @internal Controls which scoped bindings may expose main-view experiments. */
export type WebGPUFrameScopeRole = "main" | "auxiliary";

/** @internal Frame preparation and scoped binding ownership. */
export interface WebGPUFrameResourceProvider {
	createFrameScope(role?: WebGPUFrameScopeRole): WebGPUFrameResourceScope;
}

/** @internal Scene pipeline, environment, and clustered-lighting capability. */
export interface WebGPUSceneResourceProvider {
	getDrawResources(
		packet: DrawPacket,
		frameResources: WebGPUPreparedFrameResources,
		options: WebGPUDrawResourceOptions,
	): Promise<WebGPUDrawResources[] | null>;
	getEnvironmentResources(
		frameResources: WebGPUPreparedFrameResources,
		sceneTargetMode: WebGPUSceneTargetMode,
		options: WebGPUEnvironmentResourceOptions,
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

/** @internal Billboard particle pass-recording capability. */
export interface WebGPUParticleBillboardRenderer {
	renderParticles(
		encoder: ICommandEncoder,
		context: FrameContext,
		targets: WebGPUParticlePassTargets,
		frameResources: WebGPUPreparedFrameResources,
		mode: WebGPUSceneTargetMode,
		options?: WebGPUParticleRenderOptions,
	): Promise<number>;
}

/** @internal Resolves the billboard renderer without widening its owner type. */
export interface WebGPUParticleBillboardRendererProvider {
	getParticleBillboardRenderer(): WebGPUParticleBillboardRenderer;
}

/** @internal Deferred G-buffer and decal pipeline capability. */
export interface WebGPUDeferredResourceProvider {
	getGBufferWriteLayout(): GPUBindGroupLayout;
	getGBufferReadLayout(): GPUBindGroupLayout;
	getDecalBindGroupLayout(): GPUBindGroupLayout;
	getDecalOutputBindGroupLayout(): GPUBindGroupLayout;
	getDecalBatchBindGroupLayout(): GPUBindGroupLayout;
	getDeferredUnusedBinding(): IBindingGroup;
	getDeferredPlaceholderTextures(): {
		readonly rgba16Float: IRenderTexture;
		readonly rgba8Unorm: IRenderTexture;
		readonly rgba16Uint: IRenderTexture;
	};
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
	resolvePagedShadowFrame(context: FrameContext): WebGPUPagedShadowFrameState | null;
	renderShadows(
		context: FrameContext,
		framePackets: PreparedFramePacketSet,
		encoder?: ICommandEncoder | null,
	): Promise<void>;
	preparePagedShadowFrame(request: WebGPUPagedShadowFrameRequest): void;
	recordPagedShadowPageMarkPass(request: WebGPUPagedShadowFrameRequest): void | Promise<void>;
	recordPagedShadowPageAllocationPass(
		request: WebGPUPagedShadowFrameRequest,
	): void | Promise<void>;
	recordPagedShadowPageTableCopyPass(
		request: WebGPUPagedShadowFrameRequest,
	): void | Promise<void>;
	recordPagedShadowDepthPass(request: WebGPUPagedShadowFrameRequest): Promise<void>;
	recordPagedShadowFeedbackPass(request: WebGPUPagedShadowFrameRequest): void | Promise<void>;
}
