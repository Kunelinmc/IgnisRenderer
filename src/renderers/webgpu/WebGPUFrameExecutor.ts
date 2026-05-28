import type {
	DrawPacket,
	FrameContext,
	FramePass,
} from "../../pipeline/types";
import type {
	LogicalGBufferBridge,
	PostProcessPass,
	PostProcessPassExecutionContextRequest,
	PostProcessPassImplementation,
	PostProcessPassRequest,
	PostProcessResourceDescriptor,
	PostProcessResourceHandle,
} from "../../postprocess";
import type { ICommandEncoder } from "../ICommandEncoder";
import {
	AddressMode,
	BufferUsage,
	FilterMode,
	TextureFormat,
	TextureUsage,
	type IBindingGroup,
	type IRenderBuffer,
	type IRenderPipeline,
	type IRenderTexture,
	type ISampler,
	type IShaderModule,
} from "../types";
import type { WebGPUBackend } from "../WebGPUBackend";
import type {
	WebGPUPreparedFrameResources,
	WebGPURenderResources,
} from "./WebGPURenderResources";
import { resolveWebGPUComputeFacade } from "./ComputeFacade";
import { createInlineCompositeShaderSource } from "../../shaders/runtime";
import {
	WEBGPU_DEFERRED_COLOR_BYTES_PER_SAMPLE,
	WEBGPU_DEFERRED_COLOR_TARGET_COUNT,
	WEBGPU_DEFERRED_STORAGE_TEXTURE_COUNT,
	WEBGPU_MRT_COLOR_BYTES_PER_SAMPLE,
	WEBGPU_MRT_COLOR_TARGET_COUNT,
} from "./constants";
import {
	isWebGPUPostProcessContextMetadata,
	type WebGPUFrameTargets,
	type WebGPUPostProcessContextMetadata,
} from "./WebGPUPostProcessContracts";
import {
	WebGPUPostProcessRuntime,
} from "./WebGPUPostProcessRuntime";
import {
	getDefaultWebGPUDrawBindings,
	submitWebGPUDraws,
} from "./WebGPUDrawSubmission";
import { TexturePool, type TexturePoolOptions } from "./TexturePool";
import type {
	WarmupPhaseCounters,
	WarmupPlan,
} from "../../pipeline/WarmupPlanner";
import {
	WARMUP_POST_PROCESS_DESCRIPTORS_TRANSIENT_KEY,
	toShaderCompileError,
} from "../../pipeline/WarmupPlanner";
import type { ShaderCompileError } from "../../shaders/runtime";
import {
	DEFAULT_GAMMA,
	MIN_GAMMA,
} from "../constants";
import { Logger } from "../../foundation/Logger";
import { materialUsesTransmission } from "../../materials/transparency";
import { ParticleBlendMode } from "../../particles";
import { getWebGPUTexture } from "./WebGPUResourceAccess";
import { materialSupportsWebGPUDeferredLighting } from "./material";
import {
	WebGPUPlanarReflectionPass,
	type WebGPUPlanarReflectionMSAATargets,
} from "./WebGPUPlanarReflectionPass";

type WebGPUFramePassHandler = (context: FrameContext) => Promise<void>;

const WEBGPU_PRESENT_SHADER = /* wgsl */ `
struct PresentParams {
	gamma: f32,
	applyGamma: f32,
	_pad0: f32,
	_pad1: f32,
}

struct PresentVSOut {
	@builtin(position) position: vec4<f32>,
	@location(0) uv: vec2<f32>,
}

@group(0) @binding(0) var srcTexture: texture_2d<f32>;
@group(0) @binding(1) var srcSampler: sampler;
@group(0) @binding(2) var<uniform> presentParams: PresentParams;

@vertex
fn vsMain(@builtin(vertex_index) vertexIndex: u32) -> PresentVSOut {
	var positions = array<vec2<f32>, 3>(
		vec2<f32>(-1.0, -1.0),
		vec2<f32>(3.0, -1.0),
		vec2<f32>(-1.0, 3.0)
	);

	let pos = positions[vertexIndex];
	var output: PresentVSOut;
	output.position = vec4<f32>(pos, 0.0, 1.0);
	// WebGPU texture V-axis is top-origin for sampling; flip Y from clip-space mapping.
	output.uv = vec2<f32>(pos.x * 0.5 + 0.5, 0.5 - pos.y * 0.5);
	return output;
}

@fragment
fn fsMain(input: PresentVSOut) -> @location(0) vec4<f32> {
	let sampled = textureSample(srcTexture, srcSampler, input.uv);
	let gamma = max(presentParams.gamma, ${MIN_GAMMA});
	let linearColor = max(sampled.rgb, vec3<f32>(0.0));
	let gammaColor = pow(linearColor, vec3<f32>(1.0 / gamma));
	let outputColor = select(linearColor, gammaColor, presentParams.applyGamma > 0.5);
	return vec4<f32>(clamp(outputColor, vec3<f32>(0.0), vec3<f32>(1.0)), sampled.a);
}
`;

const WEBGPU_DEPTH_DIRTY_CLEAR_VERTEX_SHADER = /* wgsl */ `
@vertex
fn vsMain(@builtin(vertex_index) vertexIndex: u32) -> @builtin(position) vec4<f32> {
	var positions = array<vec2<f32>, 3>(
		vec2<f32>(-1.0, -1.0),
		vec2<f32>(3.0, -1.0),
		vec2<f32>(-1.0, 3.0)
	);
	let position = positions[vertexIndex];
	return vec4<f32>(position, 1.0, 1.0);
}
`;

const WEBGPU_OIT_RESOLVE_SHADER = /* wgsl */ `
struct ResolveVSOut {
	@builtin(position) position: vec4<f32>,
	@location(0) uv: vec2<f32>,
}

@group(0) @binding(0) var sceneColorTexture: texture_2d<f32>;
@group(0) @binding(1) var oitAccumTexture: texture_2d<f32>;
@group(0) @binding(2) var oitRevealTexture: texture_2d<f32>;
@group(0) @binding(3) var linearSampler: sampler;

@vertex
fn vsMain(@builtin(vertex_index) vertexIndex: u32) -> ResolveVSOut {
	var positions = array<vec2<f32>, 3>(
		vec2<f32>(-1.0, -1.0),
		vec2<f32>(3.0, -1.0),
		vec2<f32>(-1.0, 3.0)
	);
	let pos = positions[vertexIndex];
	var output: ResolveVSOut;
	output.position = vec4<f32>(pos, 0.0, 1.0);
	output.uv = vec2<f32>(pos.x * 0.5 + 0.5, 0.5 - pos.y * 0.5);
	return output;
}

@fragment
fn fsMain(input: ResolveVSOut) -> @location(0) vec4<f32> {
	let base = textureSample(sceneColorTexture, linearSampler, input.uv);
	let accum = textureSample(oitAccumTexture, linearSampler, input.uv);
	let reveal = clamp(
		textureSample(oitRevealTexture, linearSampler, input.uv).r,
		0.0,
		1.0
	);
	let weightedColor = accum.rgb / max(accum.a, 1e-5);
	let alpha = clamp(1.0 - reveal, 0.0, 1.0);
	let color = weightedColor * alpha + base.rgb * reveal;
	return vec4<f32>(max(color, vec3<f32>(0.0)), base.a);
}
`;

const WEBGPU_OIT_DISABLED_MRT_KEY = "webgpu-oit-disabled-mrt-unavailable";
const WEBGPU_OIT_DISABLED_MSAA_KEY = "webgpu-oit-disabled-msaa";
const WEBGPU_OIT_DISABLED_RUNTIME_KEY = "webgpu-oit-disabled-runtime";

interface WebGPUFrameMSAATargets {
	sceneColorMain: IRenderTexture;
	gAlbedoAlpha: IRenderTexture;
	gNormalRoughMetal: IRenderTexture;
	gEmissiveOcclusion: IRenderTexture;
	gMotionDepth: IRenderTexture;
	planarReflectionMask: IRenderTexture;
	depth: IRenderTexture;
}

export class WebGPUFrameExecutor {
	private _backend: WebGPUBackend;
	private _resources: WebGPURenderResources;
	private _encoder: ICommandEncoder | null = null;
	private _frameContext: FrameContext | null = null;
	private _frameResources: WebGPUPreparedFrameResources | null = null;
	private _frameTargets: WebGPUFrameTargets | null = null;
	private _msaaTargets: WebGPUFrameMSAATargets | null = null;
	private _targetWidth = 0;
	private _targetHeight = 0;
	private _targetMSAASampleCount = 1;
	private _hasPresentedInFrame = false;
	private _mrtEnabled = true;
	private _mrtSupportChecked = false;
	private _deferredEnabled = false;
	private _targetDeferredEnabled = false;
	private _postRuntime: WebGPUPostProcessRuntime;
	private _presentShaderModule: IShaderModule | null = null;
	private _presentPipeline: IRenderPipeline | null = null;
	private _presentSampler: ISampler | null = null;
	private _presentParamsBuffer: IRenderBuffer | null = null;
	private _presentBinding: IBindingGroup | null = null;
	private _presentBindingSource: IRenderTexture | null = null;
	private _oitResolveShaderModule: IShaderModule | null = null;
	private _oitResolvePipeline: IRenderPipeline | null = null;
	private _oitResolveSampler: ISampler | null = null;
	private _oitResolveBinding: IBindingGroup | null = null;
	private _oitResolveBindingScene: IRenderTexture | null = null;
	private _oitResolveBindingAccum: IRenderTexture | null = null;
	private _oitResolveBindingReveal: IRenderTexture | null = null;
	private _gbufferWriteBinding: IBindingGroup | null = null;
	private _gbufferWriteBindingSources: IRenderTexture[] = [];
	private _gbufferReadBinding: IBindingGroup | null = null;
	private _gbufferReadBindingSources: IRenderTexture[] = [];
	private _oitActive = false;
	private _oitHasContributors = false;
	private _oitTransmissionPackets: DrawPacket[] = [];
	private _oitNeedsTransmissionAfterParticles = false;
	private _motionHistoryWriteTarget: IRenderTexture | null = null;
	private _depthDirtyClearShaderModule: IShaderModule | null = null;
	private _depthDirtyClearPipelines = new Map<string, IRenderPipeline>();
	private _texturePools = new Map<string, TexturePool>();
	private _texturePoolOwners = new Map<IRenderTexture, TexturePool>();
	private _enableEarlyZPrepass = true;
	private _enableDeferredLighting = true;
	private _planarReflectionPass: WebGPUPlanarReflectionPass;
	private readonly _passHandlers: Map<FramePass["stage"], WebGPUFramePassHandler>;

	constructor(backend: WebGPUBackend, resources: WebGPURenderResources) {
		this._backend = backend;
		this._resources = resources;
		const backendOptions = this._backend as {
			isEarlyZPrepassEnabled?: () => boolean;
			enableEarlyZPrepass?: boolean;
			isDeferredLightingEnabled?: () => boolean;
			enableDeferredLighting?: boolean;
		};
		const earlyZGetter = backendOptions.isEarlyZPrepassEnabled;
		this._enableEarlyZPrepass =
			typeof earlyZGetter === "function" ?
				earlyZGetter.call(this._backend)
			:	backendOptions.enableEarlyZPrepass !== false;
		const deferredLightingGetter = backendOptions.isDeferredLightingEnabled;
		this._enableDeferredLighting =
			typeof deferredLightingGetter === "function" ?
				deferredLightingGetter.call(this._backend)
			:	backendOptions.enableDeferredLighting !== false;
		const computeFacade = resolveWebGPUComputeFacade(backend);
		this._postRuntime = new WebGPUPostProcessRuntime(
			computeFacade,
			(key, message) =>
				Logger.warn(`[${key}] ${message}`, {
					scope: "WebGPUFrameExecutor",
					onceKey: key,
				}),
			resources.sceneFrameLayout
		);
		this._planarReflectionPass = new WebGPUPlanarReflectionPass(
			backend,
			resources
		);
		this._passHandlers = this._createPassHandlers();
	}

	public beginFrame(context: FrameContext): void {
		this._frameContext = context;
		this._hasPresentedInFrame = false;
		this._oitActive = false;
		this._oitHasContributors = false;
		this._oitTransmissionPackets = [];
		this._oitNeedsTransmissionAfterParticles = false;
		this._motionHistoryWriteTarget = null;
		const targetWidth = this._resolveAttachmentDimension(
			context.attachments.width
		);
		const targetHeight = this._resolveAttachmentDimension(
			context.attachments.height
		);

		if (targetWidth <= 0 || targetHeight <= 0) {
			this._destroyFrameTargets();
			this._frameContext = null;
			this._frameResources = null;
			this._encoder = null;
			return;
		}

		this._encoder = this._backend.createCommandEncoder();

		this._ensureMRTSupport();
		this._configureDeferredLightingSupport();
		if (this._mrtEnabled) {
			this._ensureFrameTargets(
				targetWidth,
				targetHeight,
				this._deferredEnabled
			);
		} else {
			this._destroyFrameTargets();
		}
		this._configureOIT(context);
		this.prepareFrameResources(context);
	}

	/**
	 * Prepares scoped WebGPU frame resources for the active frame.
	 *
	 * @param context Current frame context.
	 * @returns Prepared main-frame resources, or `null` when the frame was
	 * rejected before an encoder was created.
	 * @sideEffects Writes frame uniforms and clustered lighting buffers.
	 */
	public prepareFrameResources(
		context: FrameContext
	): WebGPUPreparedFrameResources | null {
		if (!this._frameContext || !this._encoder) {
			this._frameResources = null;
			return null;
		}
		this._frameResources = this._resources.prepareFrame(context, {
			scopeKey: "main",
			sceneTargetMode: this.getSceneTargetModeForFrame(),
		});
		return this._frameResources;
	}

	/**
	 * Returns the resources prepared for the active main frame.
	 *
	 * @returns Main-frame resources, or `null` before preparation/after submit.
	 * @sideEffects None.
	 */
	public getPreparedFrameResources(): WebGPUPreparedFrameResources | null {
		return this._frameResources;
	}

	public createPostProcessResource(
		desc: PostProcessResourceDescriptor
	): PostProcessResourceHandle {
		const texture = this._backend.createTexture({
			width: desc.width,
			height: desc.height,
			format:
				desc.format === "rgba8unorm" ?
					TextureFormat.RGBA8Unorm
				:	TextureFormat.RGBA16Float,
			mipLevelCount:
				desc.mipMode === "full-chain" ?
					Math.floor(Math.log2(Math.max(desc.width, desc.height))) + 1
				:	undefined,
			usage:
				TextureUsage.TextureBinding |
				TextureUsage.StorageBinding |
				TextureUsage.RenderAttachment |
				TextureUsage.CopyDst |
				TextureUsage.CopySrc,
			label: `WebGPUPostHistory_${desc.id}`,
		});
		return {
			id: desc.id,
			backend: "webgpu",
			width: desc.width,
			height: desc.height,
			format: desc.format,
			mipMode: desc.mipMode ?? "single",
			resource: texture,
		};
	}

	public destroyPostProcessResource(handle: PostProcessResourceHandle): void {
		(handle.resource as IRenderTexture | null)?.destroy?.();
	}

	public createGBufferBridge(context: FrameContext): LogicalGBufferBridge {
		const targets = this._frameTargets;
		const width = Math.max(1, context.attachments.width);
		const height = Math.max(1, context.attachments.height);
		return {
			width,
			height,
			normalSpace: "world",
			depthEncoding: "hardware",
			motionEncoding: "ndc-delta",
			channels: targets ?
				{
					color: {
						semantic: "color",
						handle: { backend: "webgpu", texture: targets.sceneColor },
						width,
						height,
						format: "rgba16float",
					},
					depth: {
						semantic: "depth",
						handle: { backend: "webgpu", texture: targets.gMotionDepth },
						width,
						height,
						format: "rgba16float",
						encoding: "motion-depth.z",
					},
					normal: {
						semantic: "normal",
						handle: {
							backend: "webgpu",
							texture: targets.gNormalRoughMetal,
						},
						width,
						height,
						format: "rgba16float",
						encoding: "encoded-world-normal",
					},
					albedo: {
						semantic: "albedo",
						handle: { backend: "webgpu", texture: targets.gAlbedoAlpha },
						width,
						height,
						format: "rgba16float",
						encoding: "linear-rgb-alpha",
					},
					motion: {
						semantic: "motion",
						handle: { backend: "webgpu", texture: targets.gMotionDepth },
						width,
						height,
						format: "rgba16float",
						encoding: "motion-depth.xy",
					},
				}
			:	{},
			worldPosition: {
				source: "derived",
				available: !!targets,
			},
		};
	}

	public getSceneTargetModeForFrame(): "gbuffer" | "mrt" | "single" {
		if (!this._mrtEnabled) {
			return "single";
		}
		return this._deferredEnabled ? "gbuffer" : "mrt";
	}

	private _requireFrameResources(): WebGPUPreparedFrameResources {
		if (!this._frameResources) {
			throw new Error(
				"WebGPUFrameExecutor requires prepared main-frame resources."
			);
		}
		return this._frameResources;
	}

	public getPassExecutionContext(
		request: PostProcessPassExecutionContextRequest
	): unknown {
		if (!this._encoder || !this._frameTargets) {
			return undefined;
		}
		const metadata = request.implementation.metadata?.context;
		if (!isWebGPUPostProcessContextMetadata(metadata)) {
			return undefined;
		}
		return this._createWebGPUPostProcessContext(metadata, request, "execute");
	}

	private _getPostProcessHistoryTexture(
		request: PostProcessPassRequest,
		id: string,
		side: "read" | "write"
	): IRenderTexture | null {
		const slot = request.histories[id]?.[side];
		return (slot?.resource as IRenderTexture | null) ?? null;
	}

	private _getPostProcessTransientTexture(
		request: PostProcessPassRequest,
		id: string
	): IRenderTexture | null {
		const slot = request.transients?.[id];
		return (slot?.handle.resource as IRenderTexture | null) ?? null;
	}

	private _createWebGPUPostProcessContext(
		metadata: WebGPUPostProcessContextMetadata,
		request: PostProcessPassRequest | null,
		mode: "execute" | "warmup"
	): Record<string, unknown> | undefined {
		if (mode === "execute" && (!this._encoder || !this._frameTargets)) {
			return undefined;
		}
		if (metadata.kind === "present") {
			return {
				targets: this._frameTargets ?? undefined,
				presentToCanvas: (source: IRenderTexture, applyGamma: boolean) =>
					this._presentToCanvas(source, applyGamma),
				warmupPresent: () => this._ensurePresentResources(),
			};
		}

		const context: Record<string, unknown> = {
			encoder: this._encoder ?? undefined,
			targets: this._frameTargets ?? undefined,
			shared: this._postRuntime.sharedContext,
		};
		if (metadata.publishColorTarget && mode === "execute") {
			context.publishColorTarget = (texture: IRenderTexture): void => {
				if (this._frameTargets) {
					this._frameTargets.sceneColor = texture;
				}
			};
		}
		if (metadata.frameBinding && mode === "execute") {
			context.frameBinding = this._requireFrameResources().frameBinding;
		}
		if (metadata.lightingState && mode === "execute") {
			context.lightingState = this._requireFrameResources().lightingState;
		}
		if (request && mode === "execute") {
			for (const binding of metadata.histories ?? []) {
				context[binding.property] = this._getPostProcessHistoryTexture(
					request,
					binding.historyId,
					binding.side
				);
			}
			for (const binding of metadata.transients ?? []) {
				context[binding.property] = this._getPostProcessTransientTexture(
					request,
					binding.transientId
				);
			}
			const motionCopy = metadata.motionHistoryCopy;
			if (motionCopy) {
				const method = motionCopy.method ?? "writeMotionHistoryFromCurrent";
				context[method] = (): void => {
					this._motionHistoryWriteTarget =
						(context[motionCopy.writeProperty] as IRenderTexture | null) ??
						null;
				};
			}
		}
		return context;
	}

	/**
	 * Force frame targets to be rebuilt on the next beginFrame().
	 * Call on canvas resize so the post-process pipeline picks up
	 * the new dimensions.
	 */
	public invalidateFrameTargets(): void {
		this._destroyFrameTargets();
		this._postRuntime.invalidateBindings();
		this._planarReflectionPass.destroy();
	}

	public invalidatePostProcessBindings(): void {
		this._postRuntime.invalidateBindings();
	}

	public onShaderRuntimeChanged(): void {
		this._destroyManagedResource(this._presentShaderModule);
		this._destroyManagedResource(this._presentPipeline);
		this._destroyManagedResource(this._presentSampler);
		this._presentShaderModule = null;
		this._presentPipeline = null;
		this._destroyBindingGroup(this._presentBinding);
		this._presentBinding = null;
		this._presentBindingSource = null;
		this._destroyManagedResource(this._oitResolveShaderModule);
		this._destroyManagedResource(this._oitResolvePipeline);
		this._destroyManagedResource(this._oitResolveSampler);
		this._oitResolveShaderModule = null;
		this._oitResolvePipeline = null;
		this._oitResolveSampler = null;
		this._destroyBindingGroup(this._oitResolveBinding);
		this._oitResolveBinding = null;
		this._oitResolveBindingScene = null;
		this._oitResolveBindingAccum = null;
		this._oitResolveBindingReveal = null;
		this._destroyDeferredBindings();
		this._destroyManagedResource(this._depthDirtyClearShaderModule);
		for (const pipeline of this._depthDirtyClearPipelines.values()) {
			this._destroyManagedResource(pipeline);
		}
		this._depthDirtyClearShaderModule = null;
		this._depthDirtyClearPipelines.clear();
		this._postRuntime.onShaderRuntimeChanged();
		this._planarReflectionPass.destroy();
	}

	public async warmup(
		context: FrameContext,
		plan: WarmupPlan
	): Promise<WarmupPhaseCounters> {
		let total = 1;
		let compiled = 0;
		let failed = 0;
		const errors: ShaderCompileError[] = [];
		this._ensureMRTSupport();
		this._configureDeferredLightingSupport();

		try {
			await this._ensurePresentResources();
			compiled++;
		} catch (error) {
			failed++;
			errors.push(toShaderCompileError(error, "webgpu", "WebGPUPresentWarmup"));
		}

		const descriptorById = this._getWarmupPostProcessDescriptorMap(context);
		const hints = new Set<string>();
		if (plan.includePostProcess) {
			for (const passId of plan.postProcessPasses) {
				const implementation = descriptorById
					.get(passId)
					?.getImplementation("webgpu");
				for (const hint of implementation?.metadata?.warmupHints ?? []) {
					hints.add(hint);
				}
			}
		}
		if (hints.size > 0) {
			total += hints.size;
			const postWarmup = await this._postRuntime.warmupHints(Array.from(hints));
			compiled += postWarmup.compiled;
			failed += postWarmup.failed;
			if (postWarmup.errors.length > 0) {
				errors.push(...postWarmup.errors);
			}
		}

		const warmedPassImplementations = new Set<string>();
		for (const passId of plan.postProcessPasses) {
			if (warmedPassImplementations.has(passId)) {
				continue;
			}
			const implementation = descriptorById
				.get(passId)
				?.getImplementation("webgpu");
			if (typeof implementation?.warmup !== "function") {
				continue;
			}
			warmedPassImplementations.add(passId);
			total++;
			try {
				const warmupContext =
					this._getPassWarmupExecutionContext(implementation);
				await implementation.warmup(warmupContext, {
					frameContext: context,
					postProcess: context.postProcess,
					backend: "webgpu",
					context: warmupContext,
					options:
						context.postProcess.getOptions(passId) ??
						descriptorById.get(passId)?.normalizeOptions({
							frameContext: context,
							postProcess: context.postProcess,
							backend: "webgpu",
						}),
				});
				compiled++;
			} catch (error) {
				failed++;
				errors.push(
					toShaderCompileError(
						error,
						"webgpu",
						`WebGPUPostWarmup:${passId}`
					)
				);
			}
		}

		return {
			phase: "webgpu-frame",
			total,
			compiled,
			skipped: Math.max(0, total - compiled - failed),
			failed,
			errors,
		};
	}

	private _getWarmupPostProcessDescriptorMap(
		context: FrameContext
	): Map<string, PostProcessPass> {
		const descriptors =
			context.transient?.get(WARMUP_POST_PROCESS_DESCRIPTORS_TRANSIENT_KEY) ??
			context.postProcess.getEnabledPasses().map((pass) => pass.pass);
		return new Map(descriptors.map((pass) => [pass.id, pass]));
	}

	private _getPassWarmupExecutionContext(
		implementation: PostProcessPassImplementation
	): unknown {
		const metadata = implementation.metadata?.context;
		if (!isWebGPUPostProcessContextMetadata(metadata)) {
			return undefined;
		}
		return this._createWebGPUPostProcessContext(metadata, null, "warmup");
	}

	/**
	 * Release all GPU resources held by this executor.
	 */
	public destroy(): void {
		this._destroyFrameTargets();
		this._destroyTexturePools();
		this._postRuntime.destroy();
		this._planarReflectionPass.destroy();
		this._destroyManagedResource(this._presentShaderModule);
		this._destroyManagedResource(this._presentPipeline);
		this._destroyManagedResource(this._presentSampler);
		this._presentShaderModule = null;
		this._presentPipeline = null;
		this._presentSampler = null;
		this._destroyManagedResource(this._presentParamsBuffer);
		this._presentParamsBuffer = null;
		this._destroyBindingGroup(this._presentBinding);
		this._presentBinding = null;
		this._presentBindingSource = null;
		this._destroyManagedResource(this._oitResolveShaderModule);
		this._destroyManagedResource(this._oitResolvePipeline);
		this._destroyManagedResource(this._oitResolveSampler);
		this._oitResolveShaderModule = null;
		this._oitResolvePipeline = null;
		this._oitResolveSampler = null;
		this._destroyBindingGroup(this._oitResolveBinding);
		this._oitResolveBinding = null;
		this._oitResolveBindingScene = null;
		this._oitResolveBindingAccum = null;
		this._oitResolveBindingReveal = null;
		this._destroyManagedResource(this._depthDirtyClearShaderModule);
		for (const pipeline of this._depthDirtyClearPipelines.values()) {
			this._destroyManagedResource(pipeline);
		}
		this._depthDirtyClearShaderModule = null;
		this._depthDirtyClearPipelines.clear();
		this._encoder = null;
		this._frameContext = null;
	}

	public async executePass(
		pass: FramePass,
		context: FrameContext
	): Promise<void> {
		if (!this._encoder) return;

		const handler = this._passHandlers.get(pass.stage);
		if (!handler) {
			return;
		}
		await handler(context);
	}

	public async endFrame(): Promise<void> {
		if (!this._encoder) {
			this._frameContext = null;
			this._motionHistoryWriteTarget = null;
			return;
		}

		if (this._mrtEnabled && this._frameTargets && !this._hasPresentedInFrame) {
			await this._presentToCanvas(
				this._frameTargets.sceneColor,
				this._frameContext?.postProcess.isEnabled("gamma") !== false
			);
		}

		const encoder = this._encoder;
		const width = this._targetWidth;
		const height = this._targetHeight;

		this._backend.submit([encoder.finish()]);
		this._encoder = null;
		this._frameContext = null;
		this._frameResources = null;

		const motionSource =
			this._mrtEnabled && this._motionHistoryWriteTarget ?
				this._frameTargets?.gMotionDepth
			:	null;
		const motionTarget =
			this._mrtEnabled ? this._motionHistoryWriteTarget : null;
		if (motionSource && motionTarget && width > 0 && height > 0) {
			this._backend.copyTextureToTexture(
				{ texture: motionSource },
				{ texture: motionTarget },
				{ width, height, depthOrArrayLayers: 1 }
			);
		}
		this._motionHistoryWriteTarget = null;
	}

	private _createPassHandlers(): Map<
		FramePass["stage"],
		WebGPUFramePassHandler
	> {
		const handlers = new Map<FramePass["stage"], WebGPUFramePassHandler>([
			[
				"shadow",
				async (context) => {
					await this._resources.renderShadows(
						context,
						this._encoder ?? undefined
					);
				},
			],
			[
				"reflection",
				async (context) => {
					await this._recordPlanarReflectionPass(context);
				},
			],
			[
				"main-opaque",
				async (context) => {
					await this._recordOpaquePass(context);
				},
			],
			[
				"main-transparent",
				async (context) => {
					if (this._oitActive) {
						await this._recordOITTransparentPass(context);
					} else {
						await this._recordMainPass(
							context,
							context.scene.transparentPackets,
							false,
							false
						);
					}
				},
			],
			[
				"particles",
				async (context) => {
					if (this._oitActive) {
						await this._recordOITParticlePass(context);
					} else {
						await this._recordParticlePass(context);
					}
				},
			],
		]);
		return handlers;
	}

	private _ensureMRTSupport(): void {
		if (this._mrtSupportChecked) return;
		this._mrtSupportChecked = true;

		const maxColorAttachments =
			this._backend.device?.limits?.maxColorAttachments ?? 8;
		const maxColorAttachmentBytesPerSample =
			this._backend.device?.limits?.maxColorAttachmentBytesPerSample ?? 32;

		if (
			maxColorAttachments >= WEBGPU_MRT_COLOR_TARGET_COUNT &&
			maxColorAttachmentBytesPerSample >= WEBGPU_MRT_COLOR_BYTES_PER_SAMPLE
		) {
			return;
		}

		this._mrtEnabled = false;
		if (maxColorAttachments < WEBGPU_MRT_COLOR_TARGET_COUNT) {
			const key = "webgpu-mrt-disabled-attachments";
			Logger.warn(
				`[${key}] WebGPU device maxColorAttachments is ${maxColorAttachments}, requires ${WEBGPU_MRT_COLOR_TARGET_COUNT}; disabling MRT/GBuffer post-process pipeline`,
				{ scope: "WebGPUFrameExecutor", onceKey: key }
			);
		}
		if (maxColorAttachmentBytesPerSample < WEBGPU_MRT_COLOR_BYTES_PER_SAMPLE) {
			const key = "webgpu-mrt-disabled-bytes";
			Logger.warn(
				`[${key}] WebGPU device maxColorAttachmentBytesPerSample is ${maxColorAttachmentBytesPerSample}, requires ${WEBGPU_MRT_COLOR_BYTES_PER_SAMPLE}; disabling MRT/GBuffer post-process pipeline`,
				{ scope: "WebGPUFrameExecutor", onceKey: key }
			);
		}
	}

	private _configureDeferredLightingSupport(): void {
		if (!this._enableDeferredLighting) {
			this._deferredEnabled = false;
			return;
		}
		if (!this._mrtEnabled) {
			this._deferredEnabled = false;
			const key = "webgpu-deferred-disabled-mrt";
			Logger.warn(
				`[${key}] WebGPU deferred lighting requires MRT scene targets; using the non-deferred fallback path.`,
				{ scope: "WebGPUFrameExecutor", onceKey: key }
			);
			return;
		}

		const sampleCount = this._resolveMSAASampleCount();
		const maxColorAttachments =
			this._backend.device?.limits?.maxColorAttachments ?? 8;
		const maxColorAttachmentBytesPerSample =
			this._backend.device?.limits?.maxColorAttachmentBytesPerSample ?? 32;
		const maxStorageTexturesPerShaderStage =
			this._backend.device?.limits?.maxStorageTexturesPerShaderStage ?? 4;

		const supportsDeferred =
			sampleCount === 1 &&
			maxColorAttachments >= WEBGPU_DEFERRED_COLOR_TARGET_COUNT &&
			maxColorAttachmentBytesPerSample >=
				WEBGPU_DEFERRED_COLOR_BYTES_PER_SAMPLE &&
			maxStorageTexturesPerShaderStage >=
				WEBGPU_DEFERRED_STORAGE_TEXTURE_COUNT;

		this._deferredEnabled = supportsDeferred;
		if (supportsDeferred) {
			return;
		}

		if (sampleCount !== 1) {
			const key = "webgpu-deferred-disabled-msaa";
			Logger.warn(
				`[${key}] WebGPU deferred lighting requires sampleCount=1; using legacy MRT forward path for ${sampleCount}x MSAA.`,
				{ scope: "WebGPUFrameExecutor", onceKey: key }
			);
		}
		if (maxColorAttachments < WEBGPU_DEFERRED_COLOR_TARGET_COUNT) {
			const key = "webgpu-deferred-disabled-attachments";
			Logger.warn(
				`[${key}] WebGPU device maxColorAttachments is ${maxColorAttachments}, requires ${WEBGPU_DEFERRED_COLOR_TARGET_COUNT}; using legacy MRT forward path.`,
				{ scope: "WebGPUFrameExecutor", onceKey: key }
			);
		}
		if (
			maxColorAttachmentBytesPerSample <
			WEBGPU_DEFERRED_COLOR_BYTES_PER_SAMPLE
		) {
			const key = "webgpu-deferred-disabled-bytes";
			Logger.warn(
				`[${key}] WebGPU device maxColorAttachmentBytesPerSample is ${maxColorAttachmentBytesPerSample}, requires ${WEBGPU_DEFERRED_COLOR_BYTES_PER_SAMPLE}; using legacy MRT forward path.`,
				{ scope: "WebGPUFrameExecutor", onceKey: key }
			);
		}
		if (
			maxStorageTexturesPerShaderStage <
			WEBGPU_DEFERRED_STORAGE_TEXTURE_COUNT
		) {
			const key = "webgpu-deferred-disabled-storage-textures";
			Logger.warn(
				`[${key}] WebGPU device maxStorageTexturesPerShaderStage is ${maxStorageTexturesPerShaderStage}, requires ${WEBGPU_DEFERRED_STORAGE_TEXTURE_COUNT}; using legacy MRT forward path.`,
				{ scope: "WebGPUFrameExecutor", onceKey: key }
			);
		}
	}

	private _configureOIT(context: FrameContext): void {
		if (context.features.enableOIT !== true) {
			this._oitActive = false;
			return;
		}
		if (!this._mrtEnabled || !this._frameTargets) {
			this._warnOITDisabled(
				WEBGPU_OIT_DISABLED_MRT_KEY,
				"WebGPU OIT requires MRT scene targets; falling back to legacy transparent rendering."
			);
			this._oitActive = false;
			return;
		}
		if (this._targetMSAASampleCount > 1) {
			this._warnOITDisabled(
				WEBGPU_OIT_DISABLED_MSAA_KEY,
				"WebGPU OIT v1 only supports sampleCount=1; falling back to legacy transparent rendering."
			);
			this._oitActive = false;
			return;
		}
		if (
			!this._frameTargets.oitAccum ||
			!this._frameTargets.oitReveal ||
			!this._frameTargets.oitSceneColorCopy
		) {
			this._warnOITDisabled(
				WEBGPU_OIT_DISABLED_RUNTIME_KEY,
				"WebGPU OIT runtime targets are unavailable; falling back to legacy transparent rendering."
			);
			this._oitActive = false;
			return;
		}
		if (typeof this._encoder?.getNativeWebGPUCommandEncoder !== "function") {
			this._warnOITDisabled(
				WEBGPU_OIT_DISABLED_RUNTIME_KEY,
				"WebGPU OIT requires native command-encoder access; falling back to legacy transparent rendering."
			);
			this._oitActive = false;
			return;
		}
		this._oitActive = true;
	}

	private _warnOITDisabled(key: string, message: string): void {
		Logger.warn(`[${key}] ${message}`, {
			scope: "WebGPUFrameExecutor",
			onceKey: key,
		});
	}

	private _ensureFrameTargets(
		width: number,
		height: number,
		enableDeferred: boolean
	): void {
		const msaaSampleCount = this._resolveMSAASampleCount();
		if (width <= 0 || height <= 0) {
			this._destroyFrameTargets();
			return;
		}

		if (
			this._frameTargets &&
			this._targetWidth === width &&
			this._targetHeight === height &&
			this._targetMSAASampleCount === msaaSampleCount &&
			this._targetDeferredEnabled === enableDeferred
		) {
			this._frameTargets.sceneColor = this._frameTargets.sceneColorMain;
			return;
		}

		const acquiredTextures: IRenderTexture[] = [];
		let committed = false;
		const acquireTexture = (
			poolId: string,
			options: TexturePoolOptions,
			textureWidth: number,
			textureHeight: number,
			format: TextureFormat
		): IRenderTexture => {
			const texture = this._acquirePooledTexture(
				poolId,
				options,
				textureWidth,
				textureHeight,
				format
			);
			acquiredTextures.push(texture);
			return texture;
		};

		try {
			this._destroyFrameTargets();
			this._targetWidth = width;
			this._targetHeight = height;
			this._targetMSAASampleCount = msaaSampleCount;
			this._targetDeferredEnabled = enableDeferred;

			const sceneColorMain = acquireTexture(
				"scene-color-main",
				{
					usage:
						TextureUsage.RenderAttachment |
						TextureUsage.TextureBinding |
						TextureUsage.CopySrc |
						TextureUsage.CopyDst,
					label: "WebGPUSceneColorMain",
				},
				width,
				height,
				TextureFormat.RGBA16Float
			);
			const rgba16StoragePool: TexturePoolOptions = {
				usage: TextureUsage.TextureBinding | TextureUsage.StorageBinding,
				label: "WebGPUStorageRGBA16",
			};
			const postPing = acquireTexture(
				"rgba16-storage",
				rgba16StoragePool,
				width,
				height,
				TextureFormat.RGBA16Float
			);
			const postPong = acquireTexture(
				"rgba16-storage",
				rgba16StoragePool,
				width,
				height,
				TextureFormat.RGBA16Float
			);
			const gAlbedoAlpha = acquireTexture(
				"gbuffer-albedo",
				{
					usage: TextureUsage.RenderAttachment | TextureUsage.TextureBinding,
					label: "WebGPUGBuffer_AlbedoAlpha",
				},
				width,
				height,
				TextureFormat.RGBA8Unorm
			);
			const gNormalRoughMetal = acquireTexture(
				"gbuffer-rgba16",
				{
					usage: TextureUsage.RenderAttachment | TextureUsage.TextureBinding,
					label: "WebGPUGBuffer_RGBA16",
				},
				width,
				height,
				TextureFormat.RGBA16Float
			);
			const gEmissiveOcclusion = acquireTexture(
				"gbuffer-rgba16",
				{
					usage: TextureUsage.RenderAttachment | TextureUsage.TextureBinding,
					label: "WebGPUGBuffer_RGBA16",
				},
				width,
				height,
				TextureFormat.RGBA16Float
			);
			const gMotionDepth = acquireTexture(
				"gbuffer-motion-depth",
				{
					usage:
						TextureUsage.RenderAttachment |
						TextureUsage.TextureBinding |
						TextureUsage.CopySrc,
					label: "WebGPUGBuffer_MotionDepth",
				},
				width,
				height,
				TextureFormat.RGBA16Float
			);
			const deferredColorPool: TexturePoolOptions = {
				usage: TextureUsage.RenderAttachment | TextureUsage.TextureBinding,
				label: "WebGPUGBufferDeferredRGBA16",
			};
			const deferredStoragePool: TexturePoolOptions = {
				usage: TextureUsage.TextureBinding | TextureUsage.StorageBinding,
				label: "WebGPUGBufferDeferredStorageRGBA16",
			};
			const gSpecular =
				enableDeferred ?
					acquireTexture(
						"gbuffer-deferred-color",
						deferredColorPool,
						width,
						height,
						TextureFormat.RGBA16Float
					)
				:	null;
			const gCoatSheen =
				enableDeferred ?
					acquireTexture(
						"gbuffer-deferred-color",
						deferredColorPool,
						width,
						height,
						TextureFormat.RGBA16Float
					)
				:	null;
			const gSheenReflectance =
				enableDeferred ?
					acquireTexture(
						"gbuffer-deferred-color",
						deferredColorPool,
						width,
						height,
						TextureFormat.RGBA16Float
					)
				:	null;
			const gMaterialExt0 =
				enableDeferred ?
					acquireTexture(
						"gbuffer-deferred-storage",
						deferredStoragePool,
						width,
						height,
						TextureFormat.RGBA16Float
					)
				:	null;
			const gMaterialExt1 =
				enableDeferred ?
					acquireTexture(
						"gbuffer-deferred-storage",
						deferredStoragePool,
						width,
						height,
						TextureFormat.RGBA16Float
					)
				:	null;
			const gMaterialExt2 =
				enableDeferred ?
					acquireTexture(
						"gbuffer-deferred-storage",
						deferredStoragePool,
						width,
						height,
						TextureFormat.RGBA16Float
					)
				:	null;
			const gMaterialExt3 =
				enableDeferred ?
					acquireTexture(
						"gbuffer-deferred-storage",
						deferredStoragePool,
						width,
						height,
						TextureFormat.RGBA16Float
					)
				:	null;
			const depth = acquireTexture(
				"depth-sampleable",
				{
					usage: TextureUsage.RenderAttachment | TextureUsage.TextureBinding,
					label: "WebGPUDepthSampleable",
				},
				width,
				height,
				TextureFormat.Depth32Float
			);
			const oitAccum = acquireTexture(
				"oit-accum",
				{
					usage: TextureUsage.RenderAttachment | TextureUsage.TextureBinding,
					label: "WebGPUOITAccum",
				},
				width,
				height,
				TextureFormat.RGBA16Float
			);
			const oitReveal = acquireTexture(
				"oit-reveal",
				{
					usage: TextureUsage.RenderAttachment | TextureUsage.TextureBinding,
					label: "WebGPUOITReveal",
				},
				width,
				height,
				TextureFormat.R8Unorm
			);
			const oitSceneColorCopy = acquireTexture(
				"oit-scene-copy",
				{
					usage: TextureUsage.TextureBinding | TextureUsage.CopyDst,
					label: "WebGPUOITSceneColorCopy",
				},
				width,
				height,
				TextureFormat.RGBA16Float
			);
			const planarReflectionMask = acquireTexture(
				"planar-reflection-mask",
				{
					usage: TextureUsage.RenderAttachment | TextureUsage.TextureBinding,
					label: "WebGPUPlanarReflectionMask",
				},
				width,
				height,
				TextureFormat.R8Unorm
			);
			const useMSAA = msaaSampleCount > 1;
			const msaaPoolKey = `msaa-${msaaSampleCount}`;
			const msaaPoolOptions: TexturePoolOptions = {
				usage: TextureUsage.RenderAttachment,
				sampleCount: msaaSampleCount,
				label: `WebGPUMSAA_${msaaSampleCount}x`,
			};
			const nextMSAATargets: WebGPUFrameMSAATargets | null =
				useMSAA ?
					{
						sceneColorMain: acquireTexture(
							msaaPoolKey,
							msaaPoolOptions,
							width,
							height,
							TextureFormat.RGBA16Float
						),
						gAlbedoAlpha: acquireTexture(
							msaaPoolKey,
							msaaPoolOptions,
							width,
							height,
							TextureFormat.RGBA8Unorm
						),
						gNormalRoughMetal: acquireTexture(
							msaaPoolKey,
							msaaPoolOptions,
							width,
							height,
							TextureFormat.RGBA16Float
						),
						gEmissiveOcclusion: acquireTexture(
							msaaPoolKey,
							msaaPoolOptions,
							width,
							height,
							TextureFormat.RGBA16Float
						),
						gMotionDepth: acquireTexture(
							msaaPoolKey,
							msaaPoolOptions,
							width,
							height,
							TextureFormat.RGBA16Float
						),
						planarReflectionMask: acquireTexture(
							msaaPoolKey,
							msaaPoolOptions,
							width,
							height,
							TextureFormat.R8Unorm
						),
						depth: acquireTexture(
							msaaPoolKey,
							msaaPoolOptions,
							width,
							height,
							TextureFormat.Depth32Float
						),
					}
				:	null;

			const nextFrameTargets: WebGPUFrameTargets = {
				sceneColor: sceneColorMain,
				sceneColorMain,
				postPing,
				postPong,
				gAlbedoAlpha,
				gNormalRoughMetal,
				gEmissiveOcclusion,
				gMotionDepth,
				gSpecular,
				gCoatSheen,
				gSheenReflectance,
				gMaterialExt0,
				gMaterialExt1,
				gMaterialExt2,
				gMaterialExt3,
				depth,
				oitAccum,
				oitReveal,
				oitSceneColorCopy,
				planarReflectionMask,
			};
			this._msaaTargets = nextMSAATargets;
			this._frameTargets = nextFrameTargets;
			committed = true;
		} catch (error) {
			if (!committed) {
				for (const texture of new Set(acquiredTextures)) {
					this._releasePooledTexture(texture);
				}
			}
			this._destroyFrameTargets();
			if (enableDeferred) {
				this._deferredEnabled = false;
				const key = "webgpu-deferred-runtime-fallback";
				Logger.warn(
					`[${key}] WebGPU deferred frame target allocation failed; retrying with legacy MRT forward path. ${String(error)}`,
					{ scope: "WebGPUFrameExecutor", onceKey: key }
				);
				this._ensureFrameTargets(
					width,
					height,
					false
				);
				return;
			}
			if (msaaSampleCount > 1) {
				const setter = (
					this._backend as {
						setMSAASampleCount?: (sampleCount: number) => void;
					}
				).setMSAASampleCount;
				if (typeof setter === "function") {
					setter.call(this._backend, 1);
				}
				const key = "webgpu-msaa-runtime-fallback-1x";
				Logger.warn(
					`[${key}] WebGPU ${msaaSampleCount}x MSAA target allocation failed; retrying at 1x.`,
					{ scope: "WebGPUFrameExecutor", onceKey: key }
				);
				this._configureDeferredLightingSupport();
				this._ensureFrameTargets(
					width,
					height,
					this._deferredEnabled
				);
				return;
			}
			throw error;
		}
	}

	private _resolveMSAASampleCount(): number {
		const getter = (this._backend as { getMSAASampleCount?: () => number })
			.getMSAASampleCount;
		if (typeof getter !== "function") {
			return 1;
		}
		const sampleCount = getter.call(this._backend);
		if (!Number.isFinite(sampleCount)) {
			return 1;
		}
		return Math.max(1, Math.floor(sampleCount));
	}

	private _destroyFrameTargets(): void {
		const textures = new Set<IRenderTexture>();
		if (this._frameTargets) {
			textures.add(this._frameTargets.sceneColorMain);
			textures.add(this._frameTargets.postPing);
			textures.add(this._frameTargets.postPong);
			textures.add(this._frameTargets.gAlbedoAlpha);
			textures.add(this._frameTargets.gNormalRoughMetal);
			textures.add(this._frameTargets.gEmissiveOcclusion);
			textures.add(this._frameTargets.gMotionDepth);
			if (this._frameTargets.gSpecular) {
				textures.add(this._frameTargets.gSpecular);
			}
			if (this._frameTargets.gCoatSheen) {
				textures.add(this._frameTargets.gCoatSheen);
			}
			if (this._frameTargets.gSheenReflectance) {
				textures.add(this._frameTargets.gSheenReflectance);
			}
			if (this._frameTargets.gMaterialExt0) {
				textures.add(this._frameTargets.gMaterialExt0);
			}
			if (this._frameTargets.gMaterialExt1) {
				textures.add(this._frameTargets.gMaterialExt1);
			}
			if (this._frameTargets.gMaterialExt2) {
				textures.add(this._frameTargets.gMaterialExt2);
			}
			if (this._frameTargets.gMaterialExt3) {
				textures.add(this._frameTargets.gMaterialExt3);
			}
			textures.add(this._frameTargets.depth);
			textures.add(this._frameTargets.oitAccum);
			textures.add(this._frameTargets.oitReveal);
			textures.add(this._frameTargets.oitSceneColorCopy);
			textures.add(this._frameTargets.planarReflectionMask);
		}
		if (this._msaaTargets) {
			textures.add(this._msaaTargets.sceneColorMain);
			textures.add(this._msaaTargets.gAlbedoAlpha);
			textures.add(this._msaaTargets.gNormalRoughMetal);
			textures.add(this._msaaTargets.gEmissiveOcclusion);
			textures.add(this._msaaTargets.gMotionDepth);
			textures.add(this._msaaTargets.planarReflectionMask);
			textures.add(this._msaaTargets.depth);
		}
		for (const texture of textures) {
			this._releasePooledTexture(texture);
		}
		this._frameTargets = null;
		this._msaaTargets = null;
		this._destroyBindingGroup(this._presentBinding);
		this._presentBinding = null;
		this._presentBindingSource = null;
		this._destroyBindingGroup(this._oitResolveBinding);
		this._oitResolveBinding = null;
		this._oitResolveBindingScene = null;
		this._oitResolveBindingAccum = null;
		this._oitResolveBindingReveal = null;
		this._destroyDeferredBindings();
		this._targetWidth = 0;
		this._targetHeight = 0;
		this._targetMSAASampleCount = 1;
		this._targetDeferredEnabled = false;
		this._oitActive = false;
		this._oitHasContributors = false;
		this._oitTransmissionPackets = [];
		this._oitNeedsTransmissionAfterParticles = false;
		this._motionHistoryWriteTarget = null;
	}

	private _resolveAttachmentDimension(value: number): number {
		if (!Number.isFinite(value)) {
			return 0;
		}
		return Math.max(0, Math.floor(value));
	}

	private _acquirePooledTexture(
		poolId: string,
		options: TexturePoolOptions,
		width: number,
		height: number,
		format: TextureFormat
	): IRenderTexture {
		let pool = this._texturePools.get(poolId);
		if (!pool) {
			pool = new TexturePool(this._backend, options);
			this._texturePools.set(poolId, pool);
		}
		const texture = pool.acquire(width, height, format);
		this._texturePoolOwners.set(texture, pool);
		return texture;
	}

	private _releasePooledTexture(texture: IRenderTexture): void {
		const owner = this._texturePoolOwners.get(texture);
		if (!owner) {
			texture.destroy();
			return;
		}
		this._texturePoolOwners.delete(texture);
		owner.release(texture);
	}

	private _destroyTexturePools(): void {
		this._texturePoolOwners.clear();
		for (const pool of this._texturePools.values()) {
			pool.destroy();
		}
		this._texturePools.clear();
	}

	private _destroyBindingGroup(group: IBindingGroup | null): void {
		const destroyFn = (group as { destroy?: () => void } | null)?.destroy;
		if (typeof destroyFn === "function") {
			destroyFn.call(group);
		}
	}

	private _destroyDeferredBindings(): void {
		this._destroyBindingGroup(this._gbufferWriteBinding);
		this._gbufferWriteBinding = null;
		this._gbufferWriteBindingSources = [];
		this._destroyBindingGroup(this._gbufferReadBinding);
		this._gbufferReadBinding = null;
		this._gbufferReadBindingSources = [];
	}

	private _destroyManagedResource(resource: unknown): void {
		const destroyFn = (resource as { destroy?: () => void } | null)?.destroy;
		if (typeof destroyFn === "function") {
			destroyFn.call(resource);
		}
	}

	private _isIncrementalPartial(context: FrameContext | null): boolean {
		if (!context?.incremental) {
			return false;
		}
		return (
			context.incremental.enabled &&
			!context.incremental.forceFullFrame &&
			context.incremental.dirtyRects.length > 0
		);
	}

	private _resolveDirtyRects(
		context: FrameContext | null,
		targetWidth: number,
		targetHeight: number
	): Array<{ x: number; y: number; width: number; height: number }> {
		const width = Math.max(1, Math.floor(targetWidth));
		const height = Math.max(1, Math.floor(targetHeight));
		if (!context) {
			return [{
				x: 0,
				y: 0,
				width,
				height,
			}];
		}
		if (!this._isIncrementalPartial(context)) {
			return [{
				x: 0,
				y: 0,
				width,
				height,
			}];
		}
		const sourceWidth = Math.max(1, Math.floor(context.attachments.width));
		const sourceHeight = Math.max(1, Math.floor(context.attachments.height));
		const scaleX = width / sourceWidth;
		const scaleY = height / sourceHeight;
		const resolved: Array<{
			x: number;
			y: number;
			width: number;
			height: number;
		}> = [];
		for (const rect of context.incremental.dirtyRects) {
			const minX = Math.max(0, Math.floor(rect.x * scaleX));
			const minY = Math.max(0, Math.floor(rect.y * scaleY));
			const maxX = Math.min(
				width,
				Math.ceil((rect.x + rect.width) * scaleX)
			);
			const maxY = Math.min(
				height,
				Math.ceil((rect.y + rect.height) * scaleY)
			);
			const rectWidth = maxX - minX;
			const rectHeight = maxY - minY;
			if (rectWidth <= 0 || rectHeight <= 0) {
				continue;
			}
			resolved.push({
				x: minX,
				y: minY,
				width: rectWidth,
				height: rectHeight,
			});
		}
		return resolved;
	}

	private _resolvePacketsForRect(
		context: FrameContext,
		packets: DrawPacket[],
		rect: { x: number; y: number; width: number; height: number }
	): DrawPacket[] {
		const spatialIndex = context.scene.spatialIndex;
		if (!spatialIndex) {
			return packets;
		}
		if (packets === context.scene.opaquePackets) {
			return spatialIndex.queryOpaquePackets(rect);
		}
		if (packets === context.scene.transparentPackets) {
			return spatialIndex.queryTransparentPackets(rect);
		}
		return packets;
	}

	private _resolveTransparentSubsetForRect(
		context: FrameContext,
		packets: DrawPacket[],
		rect: { x: number; y: number; width: number; height: number }
	): DrawPacket[] {
		const spatialIndex = context.scene.spatialIndex;
		if (!spatialIndex) {
			return packets;
		}
		const rectPackets = spatialIndex.queryTransparentPackets(rect);
		if (packets === context.scene.transparentPackets) {
			return rectPackets;
		}
		if (packets.length <= 0 || rectPackets.length <= 0) {
			return [];
		}
		const packetSet = new Set(packets);
		return rectPackets.filter((packet) => packetSet.has(packet));
	}

	private async _clearDepthForDirtyRects(
		depthAttachment: IRenderTexture,
		depthFormat: TextureFormat,
		sampleCount: number,
		dirtyRects: Array<{ x: number; y: number; width: number; height: number }>
	): Promise<boolean> {
		if (!this._encoder || dirtyRects.length === 0) {
			return false;
		}
		try {
			const pipeline = await this._getDepthDirtyClearPipeline(
				depthFormat,
				sampleCount
			);
			this._encoder.beginRenderPass({
				label: "WebGPUDepthDirtyClear",
				colorAttachments: [],
				depthStencilAttachment: {
					view: depthAttachment,
					depthLoadOp: "load",
					depthStoreOp: "store",
					depthClearValue: 1,
				},
			});
			this._encoder.setPipeline(pipeline);
			for (const rect of dirtyRects) {
				this._encoder.setScissorRect?.(rect.x, rect.y, rect.width, rect.height);
				this._encoder.draw(3);
			}
			this._encoder.endRenderPass();
			return true;
		} catch (error) {
			const key = "webgpu-depth-partial-reuse-fallback";
			Logger.warn(
				`[${key}] WebGPU partial depth reuse unavailable; falling back to full depth clear. ${String(error)}`,
				{ scope: "WebGPUFrameExecutor", onceKey: key }
			);
			return false;
		}
	}

	private async _getDepthDirtyClearPipeline(
		depthFormat: TextureFormat,
		sampleCount: number
	): Promise<IRenderPipeline> {
		const resolvedSampleCount = Math.max(1, Math.floor(sampleCount || 1));
		const cacheKey = `${depthFormat}|${resolvedSampleCount}`;
		const cached = this._depthDirtyClearPipelines.get(cacheKey);
		if (cached) {
			return cached;
		}

		if (!this._depthDirtyClearShaderModule) {
			const composite = createInlineCompositeShaderSource(
				WEBGPU_DEPTH_DIRTY_CLEAR_VERTEX_SHADER,
				"<webgpu-depth-dirty-clear>",
				"source"
			);
			this._depthDirtyClearShaderModule = await this._backend.createShaderModule({
				label: "WebGPUDepthDirtyClearShader",
				code: composite.code,
				sourceMap: composite.sourceMap,
				language: "wgsl",
				stage: "unknown",
				sourceKind: "postprocess",
			});
		}

		const pipeline = this._backend.createPipeline({
			label: `WebGPUDepthDirtyClearPipeline_${cacheKey}`,
			vertex: {
				module: this._depthDirtyClearShaderModule,
				entryPoint: "vsMain",
			},
			primitive: {
				topology: "triangle-list" as any,
				cullMode: "none",
				frontFace: "ccw",
			},
			depthStencil: {
				format: depthFormat,
				depthWriteEnabled: true,
				depthCompare: "always",
			},
			sampleCount: resolvedSampleCount,
		} as any);
		this._depthDirtyClearPipelines.set(cacheKey, pipeline);
		return pipeline;
	}

	private async _ensurePresentResources(): Promise<void> {
		if (!this._presentShaderModule) {
			const composite = createInlineCompositeShaderSource(
				WEBGPU_PRESENT_SHADER,
				"<webgpu-present-shader>",
				"source"
			);
			this._presentShaderModule = await this._backend.createShaderModule({
				label: "WebGPUPresentShader",
				code: composite.code,
				sourceMap: composite.sourceMap,
				language: "wgsl",
				stage: "unknown",
				sourceKind: "builtin-present",
			});
		}

		if (!this._presentPipeline) {
			this._presentPipeline = this._backend.createPipeline({
				label: "WebGPUPresentPipeline",
				vertex: {
					module: this._presentShaderModule,
					entryPoint: "vsMain",
				},
				fragment: {
					module: this._presentShaderModule,
					entryPoint: "fsMain",
					targets: [{ format: this._backend.canvasFormat as any }],
				},
				primitive: {
					topology: "triangle-list" as any,
					cullMode: "none",
					frontFace: "ccw",
				},
			} as any);
		}

		if (!this._presentSampler) {
			this._presentSampler = this._backend.createSampler({
				label: "WebGPUPresentSampler",
				magFilter: FilterMode.Linear,
				minFilter: FilterMode.Linear,
				mipmapFilter: FilterMode.Linear,
				addressModeU: AddressMode.ClampToEdge,
				addressModeV: AddressMode.ClampToEdge,
			});
		}

		if (!this._presentParamsBuffer) {
			this._presentParamsBuffer = this._backend.createBuffer({
				label: "WebGPUPresentParams",
				size: 16,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
		}
	}

	private async _presentToCanvas(
		source: IRenderTexture,
		applyGamma: boolean
	): Promise<void> {
		if (!this._encoder) return;
		await this._ensurePresentResources();
		if (
			!this._presentPipeline ||
			!this._presentSampler ||
			!this._presentParamsBuffer
		) {
			return;
		}

		this._backend.writeBuffer(
			this._presentParamsBuffer,
			new Float32Array([DEFAULT_GAMMA, applyGamma ? 1 : 0, 0, 0])
		);

		if (!this._presentBinding || this._presentBindingSource !== source) {
			this._destroyBindingGroup(this._presentBinding);
			this._presentBinding = this._backend.createBindingGroup({
				pipeline: this._presentPipeline,
				layoutIndex: 0,
				entries: [
					{ binding: 0, resource: source },
					{ binding: 1, resource: this._presentSampler },
					{ binding: 2, resource: this._presentParamsBuffer },
				],
				label: "WebGPUPresentBinding",
			});
			this._presentBindingSource = source;
		}

		this._encoder.beginRenderPass({
			label: "WebGPUPresentPass",
			colorAttachments: [
				{
					clearValue: { r: 0, g: 0, b: 0, a: 1 },
					loadOp: "clear",
					storeOp: "store",
				},
			],
		});
		this._encoder.setPipeline(this._presentPipeline);
		this._encoder.setBindingGroup(0, this._presentBinding);
		const canvasTarget = this._backend.getCanvasColorTexture();
		const dirtyRects = this._resolveDirtyRects(
			this._frameContext,
			canvasTarget.width,
			canvasTarget.height
		);
		for (const rect of dirtyRects) {
			this._encoder.setScissorRect?.(rect.x, rect.y, rect.width, rect.height);
			this._encoder.draw(3);
		}
		this._encoder.endRenderPass();
		this._hasPresentedInFrame = true;
	}

	private _partitionTransparentPackets(packets: DrawPacket[]): {
		oitPackets: DrawPacket[];
		transmissionPackets: DrawPacket[];
	} {
		const oitPackets: DrawPacket[] = [];
		const transmissionPackets: DrawPacket[] = [];
		for (const packet of packets) {
			if (materialUsesTransmission(packet.material)) {
				transmissionPackets.push(packet);
				continue;
			}
			oitPackets.push(packet);
		}
		return {
			oitPackets,
			transmissionPackets,
		};
	}

	private _clearOITTargets(): void {
		if (!this._encoder || !this._frameTargets) {
			return;
		}
		this._encoder.beginRenderPass({
			label: "WebGPUOITClear",
			colorAttachments: [
				{
					view: this._frameTargets.oitAccum,
					clearValue: { r: 0, g: 0, b: 0, a: 0 },
					loadOp: "clear",
					storeOp: "store",
				},
				{
					view: this._frameTargets.oitReveal,
					clearValue: { r: 1, g: 1, b: 1, a: 1 },
					loadOp: "clear",
					storeOp: "store",
				},
			],
		});
		this._encoder.endRenderPass();
	}

	private async _drawOITPackets(
		context: FrameContext,
		packets: DrawPacket[]
	): Promise<number> {
		if (!this._encoder || !this._frameTargets || packets.length <= 0) {
			return 0;
		}
		const frameResources = this._requireFrameResources();
		const depthAttachment = this._msaaTargets?.depth ?? this._frameTargets.depth;
		this._encoder.beginRenderPass({
			label: "WebGPUOITDraw",
			colorAttachments: [
				{
					view: this._frameTargets.oitAccum,
					loadOp: "load",
					storeOp: "store",
				},
				{
					view: this._frameTargets.oitReveal,
					loadOp: "load",
					storeOp: "store",
				},
			],
			depthStencilAttachment: {
				view: depthAttachment,
				depthLoadOp: "load",
				depthStoreOp: "store",
			},
		});
		const dirtyRects = this._resolveDirtyRects(
			context,
			this._frameTargets.sceneColorMain.width,
			this._frameTargets.sceneColorMain.height
		);
		const submission = await submitWebGPUDraws({
			encoder: this._encoder,
			resources: this._resources,
			frameResources,
			packets,
			dirtyRects,
			selectPacketsForRect: (candidatePackets, rect) =>
				this._resolveTransparentSubsetForRect(context, candidatePackets, rect),
			resolveDrawOptions: () => ({
				sceneTargetMode: "mrt",
				transparentPipelineMode: "oit",
			}),
		});
		this._encoder.endRenderPass();
		return submission.drawCount;
	}

	private async _drawTransmissionPackets(
		context: FrameContext,
		packets: DrawPacket[]
	): Promise<void> {
		if (!this._encoder || packets.length <= 0) {
			return;
		}
		const frameResources = this._requireFrameResources();
		if (!this._mrtEnabled || !this._frameTargets) {
			await this._recordLegacyMainPass(context, packets, false, false);
			return;
		}
		const msaaTargets = this._msaaTargets;
		const sceneColorAttachment =
			msaaTargets?.sceneColorMain ?? this._frameTargets.sceneColorMain;
		const gAlbedoAttachment =
			msaaTargets?.gAlbedoAlpha ?? this._frameTargets.gAlbedoAlpha;
		const gNormalAttachment =
			msaaTargets?.gNormalRoughMetal ?? this._frameTargets.gNormalRoughMetal;
		const gEmissiveAttachment =
			msaaTargets?.gEmissiveOcclusion ?? this._frameTargets.gEmissiveOcclusion;
		const gMotionAttachment =
			msaaTargets?.gMotionDepth ?? this._frameTargets.gMotionDepth;
		const depthAttachment = msaaTargets?.depth ?? this._frameTargets.depth;
		this._encoder.beginRenderPass({
			label: "WebGPUTransmissionMRT",
			colorAttachments: [
				{
					view: sceneColorAttachment,
					resolveTarget:
						msaaTargets ? this._frameTargets.sceneColorMain : undefined,
					loadOp: "load",
					storeOp: "store",
				},
				{
					view: gAlbedoAttachment,
					resolveTarget:
						msaaTargets ? this._frameTargets.gAlbedoAlpha : undefined,
					loadOp: "load",
					storeOp: "store",
				},
				{
					view: gNormalAttachment,
					resolveTarget:
						msaaTargets ? this._frameTargets.gNormalRoughMetal : undefined,
					loadOp: "load",
					storeOp: "store",
				},
				{
					view: gEmissiveAttachment,
					resolveTarget:
						msaaTargets ? this._frameTargets.gEmissiveOcclusion : undefined,
					loadOp: "load",
					storeOp: "store",
				},
				{
					view: gMotionAttachment,
					resolveTarget:
						msaaTargets ? this._frameTargets.gMotionDepth : undefined,
					loadOp: "load",
					storeOp: "store",
				},
			],
			depthStencilAttachment: {
				view: depthAttachment,
				depthLoadOp: "load",
				depthStoreOp: "store",
			},
		});
		const dirtyRects = this._resolveDirtyRects(
			context,
			this._frameTargets.oitAccum.width,
			this._frameTargets.oitAccum.height
		);
		await submitWebGPUDraws({
			encoder: this._encoder,
			resources: this._resources,
			frameResources,
			packets,
			dirtyRects,
			selectPacketsForRect: (candidatePackets, rect) =>
				this._resolveTransparentSubsetForRect(context, candidatePackets, rect),
			resolveDrawOptions: () => ({
				sceneTargetMode: "mrt",
				transparentPipelineMode: "transmission",
			}),
		});
		this._encoder.endRenderPass();
	}

	private async _ensureOITResolveResources(): Promise<void> {
		if (!this._oitResolveShaderModule) {
			const composite = createInlineCompositeShaderSource(
				WEBGPU_OIT_RESOLVE_SHADER,
				"<webgpu-oit-resolve-shader>",
				"source"
			);
			this._oitResolveShaderModule = await this._backend.createShaderModule({
				label: "WebGPUOITResolveShader",
				code: composite.code,
				sourceMap: composite.sourceMap,
				language: "wgsl",
				stage: "unknown",
				sourceKind: "postprocess",
			});
		}
		if (!this._oitResolvePipeline) {
			this._oitResolvePipeline = this._backend.createPipeline({
				label: "WebGPUOITResolvePipeline",
				vertex: {
					module: this._oitResolveShaderModule,
					entryPoint: "vsMain",
				},
				fragment: {
					module: this._oitResolveShaderModule,
					entryPoint: "fsMain",
					targets: [{ format: TextureFormat.RGBA16Float }],
				},
				primitive: {
					topology: "triangle-list" as any,
					cullMode: "none",
					frontFace: "ccw",
				},
			} as any);
		}
		if (!this._oitResolveSampler) {
			this._oitResolveSampler = this._backend.createSampler({
				label: "WebGPUOITResolveSampler",
				magFilter: FilterMode.Linear,
				minFilter: FilterMode.Linear,
				mipmapFilter: FilterMode.Linear,
				addressModeU: AddressMode.ClampToEdge,
				addressModeV: AddressMode.ClampToEdge,
			});
		}
	}

	private _copySceneColorForOITResolve(): boolean {
		if (!this._encoder || !this._frameTargets) {
			return false;
		}
		const nativeEncoder = this._encoder.getNativeWebGPUCommandEncoder?.();
		if (
			!nativeEncoder ||
			typeof (nativeEncoder as GPUCommandEncoder).copyTextureToTexture !==
				"function"
		) {
			this._warnOITDisabled(
				WEBGPU_OIT_DISABLED_RUNTIME_KEY,
				"WebGPU OIT requires native command-encoder texture copy support; falling back to legacy transparent rendering."
			);
			this._oitActive = false;
			return false;
		}
		try {
			const sourceTexture = getWebGPUTexture(this._frameTargets.sceneColorMain);
			const destinationTexture = getWebGPUTexture(
				this._frameTargets.oitSceneColorCopy
			);
			(nativeEncoder as GPUCommandEncoder).copyTextureToTexture(
				{
					texture: sourceTexture.texture,
				},
				{
					texture: destinationTexture.texture,
				},
				{
					width: Math.max(1, this._targetWidth),
					height: Math.max(1, this._targetHeight),
					depthOrArrayLayers: 1,
				}
			);
			return true;
		} catch (error) {
			const key = "webgpu-oit-copy-scene-color-failed";
			Logger.warn(
				`[${key}] WebGPU OIT scene-color copy failed; falling back to legacy transparent rendering. ${String(error)}`,
				{ scope: "WebGPUFrameExecutor", onceKey: key }
			);
			this._oitActive = false;
			return false;
		}
	}

	private async _resolveOITComposition(context: FrameContext): Promise<void> {
		if (!this._encoder || !this._frameTargets || !this._oitHasContributors) {
			return;
		}
		if (!this._copySceneColorForOITResolve()) {
			return;
		}
		await this._ensureOITResolveResources();
		if (
			!this._oitResolvePipeline ||
			!this._oitResolveSampler ||
			!this._frameTargets
		) {
			return;
		}
		if (
			!this._oitResolveBinding ||
			this._oitResolveBindingScene !== this._frameTargets.oitSceneColorCopy ||
			this._oitResolveBindingAccum !== this._frameTargets.oitAccum ||
			this._oitResolveBindingReveal !== this._frameTargets.oitReveal
		) {
			this._destroyBindingGroup(this._oitResolveBinding);
			this._oitResolveBinding = this._backend.createBindingGroup({
				pipeline: this._oitResolvePipeline,
				layoutIndex: 0,
				entries: [
					{
						binding: 0,
						resource: this._frameTargets.oitSceneColorCopy,
					},
					{
						binding: 1,
						resource: this._frameTargets.oitAccum,
					},
					{
						binding: 2,
						resource: this._frameTargets.oitReveal,
					},
					{
						binding: 3,
						resource: this._oitResolveSampler,
					},
				],
				label: "WebGPUOITResolveBinding",
			});
			this._oitResolveBindingScene = this._frameTargets.oitSceneColorCopy;
			this._oitResolveBindingAccum = this._frameTargets.oitAccum;
			this._oitResolveBindingReveal = this._frameTargets.oitReveal;
		}
		this._encoder.beginRenderPass({
			label: "WebGPUOITResolvePass",
			colorAttachments: [
				{
					view: this._frameTargets.sceneColorMain,
					loadOp: "load",
					storeOp: "store",
				},
			],
		});
		this._encoder.setPipeline(this._oitResolvePipeline);
		this._encoder.setBindingGroup(0, this._oitResolveBinding);
		const dirtyRects = this._resolveDirtyRects(
			context,
			this._frameTargets.sceneColorMain.width,
			this._frameTargets.sceneColorMain.height
		);
		for (const rect of dirtyRects) {
			this._encoder.setScissorRect?.(rect.x, rect.y, rect.width, rect.height);
			this._encoder.draw(3);
		}
		this._encoder.endRenderPass();
	}

	private async _recordOITTransparentPass(context: FrameContext): Promise<void> {
		if (!this._encoder) {
			return;
		}
		if (!this._mrtEnabled || !this._frameTargets) {
			await this._recordMainPass(
				context,
				context.scene.transparentPackets,
				false,
				false
			);
			return;
		}
		const frameResources = this._requireFrameResources();
		await this._resources.buildClusteredLighting(this._encoder, frameResources);
		const { oitPackets, transmissionPackets } =
			this._partitionTransparentPackets(context.scene.transparentPackets);
		this._oitTransmissionPackets = transmissionPackets;
		this._oitNeedsTransmissionAfterParticles =
			(context.scene.particleSystems?.length ?? 0) > 0;
		this._oitHasContributors = false;
		if (oitPackets.length > 0) {
			this._clearOITTargets();
			const draws = await this._drawOITPackets(context, oitPackets);
			this._oitHasContributors = draws > 0;
		}
		if (!this._oitNeedsTransmissionAfterParticles) {
			if (this._oitHasContributors) {
				await this._resolveOITComposition(context);
			}
			await this._drawTransmissionPackets(context, this._oitTransmissionPackets);
			this._oitTransmissionPackets = [];
			this._oitHasContributors = false;
		}
	}

	private async _recordOITParticlePass(context: FrameContext): Promise<void> {
		if (!this._encoder) {
			return;
		}
		if (!this._mrtEnabled || !this._frameTargets) {
			await this._recordParticlePass(context);
			return;
		}
		const msaaTargets = this._msaaTargets;
		const depthAttachment = msaaTargets?.depth ?? this._frameTargets.depth;
		if (!this._oitHasContributors) {
			this._clearOITTargets();
		}
		const alphaParticleCount = await this._resources.renderParticles(
			this._encoder,
			context,
			{
				label: "WebGPUParticlesOIT",
				colorAttachments: [
					{
						view: this._frameTargets.oitAccum,
						loadOp: "load",
						storeOp: "store",
					},
					{
						view: this._frameTargets.oitReveal,
						loadOp: "load",
						storeOp: "store",
					},
				],
				depth: depthAttachment,
			},
			this._requireFrameResources(),
			"mrt",
			{
				includeBlendModes: [ParticleBlendMode.Alpha],
				pipelineMode: "oit",
			}
		);
		if (alphaParticleCount > 0) {
			this._oitHasContributors = true;
		}
		if (this._oitHasContributors) {
			await this._resolveOITComposition(context);
		}
		if (this._oitTransmissionPackets.length > 0) {
			await this._drawTransmissionPackets(context, this._oitTransmissionPackets);
		}
		await this._resources.renderParticles(
			this._encoder,
			context,
			{
				label: "WebGPUParticlesMRT_Additive",
				colorAttachments: [
					{
						view:
							msaaTargets?.sceneColorMain ?? this._frameTargets.sceneColorMain,
						resolveTarget:
							msaaTargets ? this._frameTargets.sceneColorMain : undefined,
						loadOp: "load",
						storeOp: "store",
					},
				],
				depth: depthAttachment,
			},
			this._requireFrameResources(),
			"mrt",
			{
				includeBlendModes: [ParticleBlendMode.Additive],
				pipelineMode: "legacy",
			}
		);
		this._oitTransmissionPackets = [];
		this._oitHasContributors = false;
		this._oitNeedsTransmissionAfterParticles = false;
	}

	private _getGBufferWriteBinding(): IBindingGroup {
		if (
			!this._frameTargets?.gMaterialExt0 ||
			!this._frameTargets.gMaterialExt1 ||
			!this._frameTargets.gMaterialExt2 ||
			!this._frameTargets.gMaterialExt3
		) {
			throw new Error("WebGPU deferred G-buffer storage targets are unavailable.");
		}
		const sources = [
			this._frameTargets.gMaterialExt0,
			this._frameTargets.gMaterialExt1,
			this._frameTargets.gMaterialExt2,
			this._frameTargets.gMaterialExt3,
		];
		if (
			this._gbufferWriteBinding &&
			this._gbufferWriteBindingSources.length === sources.length &&
			this._gbufferWriteBindingSources.every(
				(source, index) => source === sources[index]
			)
		) {
			return this._gbufferWriteBinding;
		}
		this._destroyBindingGroup(this._gbufferWriteBinding);
		this._gbufferWriteBinding = this._backend.createBindingGroup({
			layout: this._resources.getGBufferWriteLayout(),
			entries: [
				{ binding: 0, resource: sources[0] },
				{ binding: 1, resource: sources[1] },
				{ binding: 2, resource: sources[2] },
				{ binding: 3, resource: sources[3] },
			],
			label: "WebGPUGBufferWriteBinding",
		});
		this._gbufferWriteBindingSources = sources;
		return this._gbufferWriteBinding;
	}

	private _getGBufferReadBinding(): IBindingGroup {
		if (
			!this._frameTargets?.gSpecular ||
			!this._frameTargets.gCoatSheen ||
			!this._frameTargets.gSheenReflectance ||
			!this._frameTargets.gMaterialExt0 ||
			!this._frameTargets.gMaterialExt1 ||
			!this._frameTargets.gMaterialExt2 ||
			!this._frameTargets.gMaterialExt3
		) {
			throw new Error("WebGPU deferred G-buffer read targets are unavailable.");
		}
		const sources = [
			this._frameTargets.gAlbedoAlpha,
			this._frameTargets.gNormalRoughMetal,
			this._frameTargets.gEmissiveOcclusion,
			this._frameTargets.gMotionDepth,
			this._frameTargets.gSpecular,
			this._frameTargets.gCoatSheen,
			this._frameTargets.gSheenReflectance,
			this._frameTargets.gMaterialExt0,
			this._frameTargets.gMaterialExt1,
			this._frameTargets.gMaterialExt2,
			this._frameTargets.gMaterialExt3,
		];
		if (
			this._gbufferReadBinding &&
			this._gbufferReadBindingSources.length === sources.length &&
			this._gbufferReadBindingSources.every(
				(source, index) => source === sources[index]
			)
		) {
			return this._gbufferReadBinding;
		}
		this._destroyBindingGroup(this._gbufferReadBinding);
		this._gbufferReadBinding = this._backend.createBindingGroup({
			layout: this._resources.getGBufferReadLayout(),
			entries: sources.map((resource, binding) => ({
				binding,
				resource,
			})),
			label: "WebGPUGBufferReadBinding",
		});
		this._gbufferReadBindingSources = sources;
		return this._gbufferReadBinding;
	}

	private async _recordPlanarReflectionPass(
		context: FrameContext
	): Promise<void> {
		if (!this._encoder) {
			return;
		}
		this._submitCurrentFrameEncoder();
		await this._planarReflectionPass.capture(context);
		this._encoder = this._backend.createCommandEncoder();
	}

	private async _recordPlanarReflectionComposite(
		context: FrameContext
	): Promise<void> {
		if (!this._encoder || !this._mrtEnabled || !this._frameTargets) {
			return;
		}
		await this._planarReflectionPass.composite({
			encoder: this._encoder,
			context,
			frameResources: this._requireFrameResources(),
			frameTargets: this._frameTargets,
			msaaTargets:
				this._msaaTargets as WebGPUPlanarReflectionMSAATargets | null,
		});
	}

	private _submitCurrentFrameEncoder(): void {
		if (!this._encoder) {
			return;
		}
		const encoder = this._encoder;
		this._backend.submit([encoder.finish()]);
		this._encoder = null;
	}

	private async _recordOpaquePass(context: FrameContext): Promise<void> {
		if (!this._deferredEnabled || !this._mrtEnabled || !this._frameTargets) {
			await this._recordMainPass(
				context,
				context.scene.opaquePackets,
				true,
				true
			);
			await this._recordPlanarReflectionComposite(context);
			return;
		}

		const deferredPackets: DrawPacket[] = [];
		const fallbackPackets: DrawPacket[] = [];
		for (const packet of context.scene.opaquePackets) {
			if (materialSupportsWebGPUDeferredLighting(packet.material)) {
				deferredPackets.push(packet);
			} else {
				fallbackPackets.push(packet);
			}
		}

		if (deferredPackets.length <= 0 && fallbackPackets.length > 0) {
			await this._recordMainPass(context, fallbackPackets, true, true);
			await this._recordPlanarReflectionComposite(context);
			return;
		}

		await this._recordDeferredOpaquePass(
			context,
			deferredPackets,
			true,
			true
		);
		if (fallbackPackets.length > 0) {
			await this._recordMainPass(context, fallbackPackets, false, false);
		}
		await this._recordPlanarReflectionComposite(context);
	}

	private async _recordDeferredOpaquePass(
		context: FrameContext,
		packets: DrawPacket[],
		clearAttachments: boolean,
		allowEarlyZPrepass: boolean
	): Promise<void> {
		if (!this._encoder || !this._frameTargets) {
			return;
		}
		if (
			!this._frameTargets.gSpecular ||
			!this._frameTargets.gCoatSheen ||
			!this._frameTargets.gSheenReflectance ||
			!this._frameTargets.gMaterialExt0 ||
			!this._frameTargets.gMaterialExt1 ||
			!this._frameTargets.gMaterialExt2 ||
			!this._frameTargets.gMaterialExt3
		) {
			await this._recordMainPass(context, packets, clearAttachments, true);
			return;
		}

		const frameResources = this._requireFrameResources();
		await this._resources.buildClusteredLighting(this._encoder, frameResources);
		const incrementalPartial = this._isIncrementalPartial(context);
		const sceneColorAttachment = this._frameTargets.sceneColorMain;
		const depthAttachment = this._frameTargets.depth;
		const dirtyRects = this._resolveDirtyRects(
			context,
			sceneColorAttachment.width,
			sceneColorAttachment.height
		);
		const shouldClearAttachments = clearAttachments && !incrementalPartial;
		let depthPartialReuseApplied = false;
		if (incrementalPartial && dirtyRects.length > 0) {
			depthPartialReuseApplied = await this._clearDepthForDirtyRects(
				depthAttachment,
				TextureFormat.Depth32Float,
				1,
				dirtyRects
			);
		}

		let environmentDrawn = false;
		if (shouldClearAttachments) {
			const environmentResources =
				await this._resources.getEnvironmentResources(
					frameResources,
					"gbuffer"
				);
			if (environmentResources) {
				this._encoder.beginRenderPass({
					label: "WebGPUEnvironmentDeferred",
					colorAttachments: [
						{
							view: sceneColorAttachment,
							clearValue: { r: 0, g: 0, b: 0, a: 1 },
							loadOp: "clear",
							storeOp: "store",
						},
					],
					depthStencilAttachment: {
						view: depthAttachment,
						depthClearValue: 1,
						depthLoadOp: "clear",
						depthStoreOp: "store",
					},
				});
				this._encoder.setPipeline(environmentResources.pipeline);
				this._encoder.setBindingGroup(0, environmentResources.frameBinding);
				this._encoder.draw(3);
				this._encoder.endRenderPass();
				environmentDrawn = true;
			}
		}

		const shouldRunEarlyZ =
			allowEarlyZPrepass &&
			this._enableEarlyZPrepass &&
			packets.length > 0;
		const earlyZPacketIds =
			shouldRunEarlyZ ?
				await this._recordEarlyZPrepass(
					context,
					packets,
					dirtyRects,
					"gbuffer",
					depthAttachment,
					this._resolveMRTMainDepthLoadOp(
						depthPartialReuseApplied,
						incrementalPartial,
						shouldClearAttachments,
						environmentDrawn,
						false
					)
				)
			:	new Set<string>();
		const earlyZExecuted = earlyZPacketIds.size > 0;
		const gbufferWriteBinding = this._getGBufferWriteBinding();

		this._encoder.beginRenderPass({
			label:
				shouldClearAttachments ?
					"WebGPUGBuffer_Clear"
				:	"WebGPUGBuffer_Load",
			colorAttachments: [
				{
					view: this._frameTargets.gAlbedoAlpha,
					clearValue: { r: 0, g: 0, b: 0, a: 0 },
					loadOp: shouldClearAttachments ? "clear" : "load",
					storeOp: "store",
				},
				{
					view: this._frameTargets.gNormalRoughMetal,
					clearValue: { r: 0.5, g: 0.5, b: 1, a: 0 },
					loadOp: shouldClearAttachments ? "clear" : "load",
					storeOp: "store",
				},
				{
					view: this._frameTargets.gEmissiveOcclusion,
					clearValue: { r: 0, g: 0, b: 0, a: 1 },
					loadOp: shouldClearAttachments ? "clear" : "load",
					storeOp: "store",
				},
				{
					view: this._frameTargets.gMotionDepth,
					clearValue: { r: 0, g: 0, b: 0, a: 0 },
					loadOp: shouldClearAttachments ? "clear" : "load",
					storeOp: "store",
				},
				{
					view: this._frameTargets.gSpecular,
					clearValue: { r: 0, g: 0, b: 0, a: 0 },
					loadOp: shouldClearAttachments ? "clear" : "load",
					storeOp: "store",
				},
				{
					view: this._frameTargets.gCoatSheen,
					clearValue: { r: 0, g: 0, b: 0, a: 0 },
					loadOp: shouldClearAttachments ? "clear" : "load",
					storeOp: "store",
				},
				{
					view: this._frameTargets.gSheenReflectance,
					clearValue: { r: 0, g: 0, b: 0, a: 0 },
					loadOp: shouldClearAttachments ? "clear" : "load",
					storeOp: "store",
				},
			],
			depthStencilAttachment: {
				view: depthAttachment,
				depthClearValue: 1,
				depthLoadOp: this._resolveMRTMainDepthLoadOp(
					depthPartialReuseApplied,
					incrementalPartial,
					shouldClearAttachments,
					environmentDrawn,
					earlyZExecuted
				),
				depthStoreOp: "store",
			},
		});

		await submitWebGPUDraws({
			encoder: this._encoder,
			resources: this._resources,
			frameResources,
			packets,
			dirtyRects,
			selectPacketsForRect: (candidatePackets, rect) =>
				this._resolvePacketsForRect(context, candidatePackets, rect),
			resolveDrawOptions: (packet) => ({
				sceneTargetMode: "gbuffer",
				drawMode:
					earlyZExecuted && earlyZPacketIds.has(packet.id) ?
						"early-z-color"
					:	"default",
			}),
			resolveBindings: (draw) => [
				...getDefaultWebGPUDrawBindings(draw),
				{ slot: 3, group: gbufferWriteBinding },
			],
		});

		this._encoder.endRenderPass();
		await this._recordDeferredLightingPass(
			context,
			shouldClearAttachments && !environmentDrawn
		);
	}

	private async _recordDeferredLightingPass(
		context: FrameContext,
		clearSceneColor: boolean
	): Promise<void> {
		if (!this._encoder || !this._frameTargets) {
			return;
		}
		const pipeline = await this._resources.getDeferredLightingPipeline();
		const gbufferReadBinding = this._getGBufferReadBinding();
		this._encoder.beginRenderPass({
			label: "WebGPUDeferredLighting",
			colorAttachments: [
				{
					view: this._frameTargets.sceneColorMain,
					clearValue: { r: 0, g: 0, b: 0, a: 1 },
					loadOp: clearSceneColor ? "clear" : "load",
					storeOp: "store",
				},
			],
		});
		this._encoder.setPipeline(pipeline);
		const frameResources = this._requireFrameResources();
		this._encoder.setBindingGroup(0, frameResources.frameBinding);
		this._encoder.setBindingGroup(
			1,
			this._resources.getDeferredUnusedBinding()
		);
		this._encoder.setBindingGroup(2, frameResources.clusteredSceneBinding);
		this._encoder.setBindingGroup(3, gbufferReadBinding);
		const dirtyRects = this._resolveDirtyRects(
			context,
			this._frameTargets.sceneColorMain.width,
			this._frameTargets.sceneColorMain.height
		);
		for (const rect of dirtyRects) {
			this._encoder.setScissorRect?.(rect.x, rect.y, rect.width, rect.height);
			this._encoder.draw(3);
		}
		this._encoder.endRenderPass();
	}

	private async _recordMainPass(
		context: FrameContext,
		packets: DrawPacket[],
		clearAttachments: boolean,
		allowEarlyZPrepass: boolean
	): Promise<void> {
		if (!this._encoder) return;
		const frameResources = this._requireFrameResources();
		await this._resources.buildClusteredLighting(this._encoder, frameResources);
		const incrementalPartial = this._isIncrementalPartial(context);
		if (!this._mrtEnabled || !this._frameTargets) {
			await this._recordLegacyMainPass(
				context,
				packets,
				clearAttachments,
				allowEarlyZPrepass
			);
			return;
		}
		const msaaTargets = this._msaaTargets;
		const sceneColorAttachment =
			msaaTargets?.sceneColorMain ?? this._frameTargets.sceneColorMain;
		const gAlbedoAttachment =
			msaaTargets?.gAlbedoAlpha ?? this._frameTargets.gAlbedoAlpha;
		const gNormalAttachment =
			msaaTargets?.gNormalRoughMetal ?? this._frameTargets.gNormalRoughMetal;
		const gEmissiveAttachment =
			msaaTargets?.gEmissiveOcclusion ?? this._frameTargets.gEmissiveOcclusion;
		const gMotionAttachment =
			msaaTargets?.gMotionDepth ?? this._frameTargets.gMotionDepth;
		const depthAttachment = msaaTargets?.depth ?? this._frameTargets.depth;
		const dirtyRects = this._resolveDirtyRects(
			context,
			sceneColorAttachment.width,
			sceneColorAttachment.height
		);
		const shouldClearAttachments = clearAttachments && !incrementalPartial;
		let depthPartialReuseApplied = false;
		if (incrementalPartial && dirtyRects.length > 0) {
			depthPartialReuseApplied = await this._clearDepthForDirtyRects(
				depthAttachment,
				TextureFormat.Depth32Float,
				msaaTargets ? this._targetMSAASampleCount : 1,
				dirtyRects
			);
		}

		let environmentDrawn = false;
		if (shouldClearAttachments) {
			const environmentResources =
				await this._resources.getEnvironmentResources(frameResources, "mrt");
			if (environmentResources) {
				this._encoder.beginRenderPass({
					label: "WebGPUEnvironmentMRT",
					colorAttachments: [
						{
							view: sceneColorAttachment,
							resolveTarget:
								msaaTargets ? this._frameTargets.sceneColorMain : undefined,
							clearValue: { r: 0, g: 0, b: 0, a: 1 },
							loadOp: "clear",
							storeOp: "store",
						},
					],
					depthStencilAttachment: {
						view: depthAttachment,
						depthClearValue: 1,
						depthLoadOp: "clear",
						depthStoreOp: "store",
					},
				});
				this._encoder.setPipeline(environmentResources.pipeline);
				this._encoder.setBindingGroup(0, environmentResources.frameBinding);
				this._encoder.draw(3);
				this._encoder.endRenderPass();
				environmentDrawn = true;
			}
		}
		const shouldRunEarlyZ =
			allowEarlyZPrepass &&
			this._enableEarlyZPrepass &&
			packets.length > 0;
		const earlyZPacketIds =
			shouldRunEarlyZ ?
				await this._recordEarlyZPrepass(
					context,
					packets,
					dirtyRects,
					"mrt",
					depthAttachment,
					this._resolveMRTMainDepthLoadOp(
						depthPartialReuseApplied,
						incrementalPartial,
						shouldClearAttachments,
						environmentDrawn,
						false
					)
				)
			:	new Set<string>();
		const earlyZExecuted = earlyZPacketIds.size > 0;

		this._encoder.beginRenderPass({
			label:
				shouldClearAttachments ? "WebGPUMainMRT_Clear" : "WebGPUMainMRT_Load",
			colorAttachments: [
				{
					view: sceneColorAttachment,
					resolveTarget:
						msaaTargets ? this._frameTargets.sceneColorMain : undefined,
					clearValue: { r: 0, g: 0, b: 0, a: 1 },
					loadOp: shouldClearAttachments && !environmentDrawn ? "clear" : "load",
					storeOp: "store",
				},
				{
					view: gAlbedoAttachment,
					resolveTarget:
						msaaTargets ? this._frameTargets.gAlbedoAlpha : undefined,
					clearValue: { r: 0, g: 0, b: 0, a: 1 },
					loadOp: shouldClearAttachments ? "clear" : "load",
					storeOp: "store",
				},
				{
					view: gNormalAttachment,
					resolveTarget:
						msaaTargets ? this._frameTargets.gNormalRoughMetal : undefined,
					clearValue: { r: 0.5, g: 0.5, b: 1, a: 0 },
					loadOp: shouldClearAttachments ? "clear" : "load",
					storeOp: "store",
				},
				{
					view: gEmissiveAttachment,
					resolveTarget:
						msaaTargets ? this._frameTargets.gEmissiveOcclusion : undefined,
					clearValue: { r: 0, g: 0, b: 0, a: 1 },
					loadOp: shouldClearAttachments ? "clear" : "load",
					storeOp: "store",
				},
				{
					view: gMotionAttachment,
					resolveTarget:
						msaaTargets ? this._frameTargets.gMotionDepth : undefined,
					clearValue: { r: 0, g: 0, b: 0, a: 0 },
					loadOp: shouldClearAttachments ? "clear" : "load",
					storeOp: "store",
				},
			],
			depthStencilAttachment: {
				view: depthAttachment,
				depthClearValue: 1,
				depthLoadOp: this._resolveMRTMainDepthLoadOp(
					depthPartialReuseApplied,
					incrementalPartial,
					shouldClearAttachments,
					environmentDrawn,
					earlyZExecuted
				),
				depthStoreOp: "store",
			},
		});

		await submitWebGPUDraws({
			encoder: this._encoder,
			resources: this._resources,
			frameResources,
			packets,
			dirtyRects,
			selectPacketsForRect: (candidatePackets, rect) =>
				this._resolvePacketsForRect(context, candidatePackets, rect),
			resolveDrawOptions: (packet) => ({
				sceneTargetMode: "mrt",
				drawMode:
					earlyZExecuted && earlyZPacketIds.has(packet.id) ?
						"early-z-color"
					:	"default",
			}),
		});

		this._encoder.endRenderPass();
	}

	private async _recordLegacyMainPass(
		context: FrameContext,
		packets: DrawPacket[],
		clearAttachments: boolean,
		allowEarlyZPrepass: boolean
	): Promise<void> {
		if (!this._encoder) return;
		const frameResources = this._requireFrameResources();
		await this._resources.buildClusteredLighting(this._encoder, frameResources);
		const incrementalPartial = this._isIncrementalPartial(context);
		const colorTexture = this._backend.getCanvasColorTexture();
		const depthTexture = this._backend.getCanvasDepthTexture();
		const shouldClearAttachments = clearAttachments && !incrementalPartial;
		const dirtyRects = this._resolveDirtyRects(
			context,
			colorTexture.width,
			colorTexture.height
		);
		let depthPartialReuseApplied = false;
		if (incrementalPartial && dirtyRects.length > 0) {
			depthPartialReuseApplied = await this._clearDepthForDirtyRects(
				depthTexture,
				this._backend.canvasDepthFormat,
				1,
				dirtyRects
			);
		}
		const shouldRunEarlyZ =
			allowEarlyZPrepass &&
			this._enableEarlyZPrepass &&
			packets.length > 0;
		const earlyZPacketIds =
			shouldRunEarlyZ ?
				await this._recordEarlyZPrepass(
					context,
					packets,
					dirtyRects,
					"single",
					depthTexture,
					this._resolveLegacyMainDepthLoadOp(
						depthPartialReuseApplied,
						incrementalPartial,
						shouldClearAttachments,
						false
					)
				)
			:	new Set<string>();
		const earlyZExecuted = earlyZPacketIds.size > 0;

		this._encoder.beginRenderPass({
			colorAttachments: [
				{
					view: colorTexture,
					clearValue: { r: 0, g: 0, b: 0, a: 1 },
					loadOp: shouldClearAttachments ? "clear" : "load",
					storeOp: "store",
				},
			],
			depthStencilAttachment: {
				view: depthTexture,
				depthClearValue: 1,
				depthLoadOp: this._resolveLegacyMainDepthLoadOp(
					depthPartialReuseApplied,
					incrementalPartial,
					shouldClearAttachments,
					earlyZExecuted
				),
				depthStoreOp: "store",
			},
		});

		if (shouldClearAttachments) {
			const environmentResources =
				await this._resources.getEnvironmentResources(frameResources, "single");
			if (environmentResources) {
				this._encoder.setPipeline(environmentResources.pipeline);
				this._encoder.setBindingGroup(0, environmentResources.frameBinding);
				this._encoder.draw(3);
			}
		}

		await submitWebGPUDraws({
			encoder: this._encoder,
			resources: this._resources,
			frameResources,
			packets,
			dirtyRects,
			selectPacketsForRect: (candidatePackets, rect) =>
				this._resolvePacketsForRect(context, candidatePackets, rect),
			resolveDrawOptions: (packet) => ({
				sceneTargetMode: "single",
				drawMode:
					earlyZExecuted && earlyZPacketIds.has(packet.id) ?
						"early-z-color"
					:	"default",
			}),
		});

		this._encoder.endRenderPass();
	}

	private async _recordEarlyZPrepass(
		context: FrameContext,
		packets: DrawPacket[],
		dirtyRects: Array<{ x: number; y: number; width: number; height: number }>,
		sceneTargetMode: "gbuffer" | "mrt" | "single",
		depthAttachment: IRenderTexture,
		depthLoadOp: "clear" | "load"
	): Promise<Set<string>> {
		const prepassedPacketIds = new Set<string>();
		if (!this._encoder || packets.length <= 0) {
			return prepassedPacketIds;
		}
		this._encoder.beginRenderPass({
			label:
				sceneTargetMode === "gbuffer" ? "WebGPUEarlyZPrepassGBuffer"
				: sceneTargetMode === "mrt" ?
					"WebGPUEarlyZPrepassMRT"
				:	"WebGPUEarlyZPrepassSingle",
			colorAttachments: [],
			depthStencilAttachment: {
				view: depthAttachment,
				depthClearValue: 1,
				depthLoadOp,
				depthStoreOp: "store",
			},
		});

		const submission = await submitWebGPUDraws({
			encoder: this._encoder,
			resources: this._resources,
			frameResources: this._requireFrameResources(),
			packets,
			dirtyRects,
			selectPacketsForRect: (candidatePackets, rect) =>
				this._resolvePacketsForRect(context, candidatePackets, rect),
			resolveDrawOptions: () => ({
				sceneTargetMode,
				drawMode: "early-z-prepass",
			}),
		});

		this._encoder.endRenderPass();
		return submission.submittedPacketIds;
	}

	private _resolveMRTMainDepthLoadOp(
		depthPartialReuseApplied: boolean,
		incrementalPartial: boolean,
		shouldClearAttachments: boolean,
		environmentDrawn: boolean,
		earlyZExecuted: boolean
	): "load" | "clear" {
		if (earlyZExecuted || depthPartialReuseApplied) {
			return "load";
		}
		return incrementalPartial || (shouldClearAttachments && !environmentDrawn) ?
				"clear"
			:	"load";
	}

	private _resolveLegacyMainDepthLoadOp(
		depthPartialReuseApplied: boolean,
		incrementalPartial: boolean,
		shouldClearAttachments: boolean,
		earlyZExecuted: boolean
	): "load" | "clear" {
		if (earlyZExecuted || depthPartialReuseApplied) {
			return "load";
		}
		return incrementalPartial || shouldClearAttachments ? "clear" : "load";
	}

	private async _recordParticlePass(context: FrameContext): Promise<void> {
		if (!this._encoder) return;
		const frameResources = this._requireFrameResources();

		if (this._mrtEnabled && this._frameTargets) {
			const msaaTargets = this._msaaTargets;
			await this._resources.renderParticles(
				this._encoder,
				context,
				{
					label: "WebGPUParticlesMRT",
					colorAttachments: [
						{
							view:
								msaaTargets?.sceneColorMain ??
								this._frameTargets.sceneColorMain,
							resolveTarget:
								msaaTargets ? this._frameTargets.sceneColorMain : undefined,
							clearValue: { r: 0, g: 0, b: 0, a: 1 },
							loadOp: "load",
							storeOp: "store",
						},
				],
					depth: msaaTargets?.depth ?? this._frameTargets.depth,
				},
				frameResources,
				"mrt",
				{
					pipelineMode: "legacy",
				}
			);
			return;
		}

		await this._resources.renderParticles(
			this._encoder,
			context,
			{
				label: "WebGPUParticlesSingle",
				colorAttachments: [
					{
						view: this._backend.getCanvasColorTexture(),
						clearValue: { r: 0, g: 0, b: 0, a: 1 },
						loadOp: "load",
						storeOp: "store",
					},
				],
				depth: this._backend.getCanvasDepthTexture(),
			},
			frameResources,
			"single",
			{
				pipelineMode: "legacy",
			}
		);
	}
}
