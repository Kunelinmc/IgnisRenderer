import type { DrawPacket, FrameContext, FramePass } from "../../pipeline/types";
import {
	DEFAULT_SSAO_OPTIONS,
	DEFAULT_SSR_OPTIONS,
	INTERACTION_TRANSIENT_STATE_KEY,
	isFogPostProcessEnabled,
	type InteractionTransientState,
} from "../../pipeline/types";
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
import type { WebGPURenderResources } from "./WebGPURenderResources";
import { resolveWebGPUComputeFacade } from "./ComputeFacade";
import { createInlineCompositeShaderSource } from "../../shaders/runtime";
import {
	WEBGPU_MRT_COLOR_BYTES_PER_SAMPLE,
	WEBGPU_MRT_COLOR_TARGET_COUNT,
} from "./constants";
import {
	WebGPUPostProcessGraph,
	type WebGPUFrameTargets,
	type WebGPUPostProcessPassContext,
	type WebGPUPostProcessPassPlugin,
} from "./WebGPUPostProcessGraph";
import { WebGPUPostProcessRuntime } from "./WebGPUPostProcessRuntime";
import { TexturePool, type TexturePoolOptions } from "./TexturePool";
import type {
	WarmupPhaseCounters,
	WarmupPlan,
} from "../../pipeline/WarmupPlanner";
import { toShaderCompileError } from "../../pipeline/WarmupPlanner";
import type { ShaderCompileError } from "../../shaders/runtime";
import {
	DEFAULT_GAMMA,
	MIN_GAMMA,
	POST_PROCESS_STAGES,
} from "../constants";
import { Logger } from "../../foundation/Logger";

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

interface WebGPUFrameMSAATargets {
	sceneColorMain: IRenderTexture;
	gAlbedoAlpha: IRenderTexture;
	gNormalRoughMetal: IRenderTexture;
	gEmissiveOcclusion: IRenderTexture;
	gMotionDepth: IRenderTexture;
	depth: IRenderTexture;
}

export class WebGPUFrameExecutor {
	private _backend: WebGPUBackend;
	private _resources: WebGPURenderResources;
	private _encoder: ICommandEncoder | null = null;
	private _frameContext: FrameContext | null = null;
	private _frameTargets: WebGPUFrameTargets | null = null;
	private _msaaTargets: WebGPUFrameMSAATargets | null = null;
	private _targetWidth = 0;
	private _targetHeight = 0;
	private _targetSSAODownsample = DEFAULT_SSAO_OPTIONS.downsample;
	private _targetSSRDownsample = DEFAULT_SSR_OPTIONS.downsample;
	private _targetMSAASampleCount = 1;
	private _taaHistoryA: IRenderTexture | null = null;
	private _taaHistoryB: IRenderTexture | null = null;
	private _ssrHistoryA: IRenderTexture | null = null;
	private _ssrHistoryB: IRenderTexture | null = null;
	private _volumetricHistoryA: IRenderTexture | null = null;
	private _volumetricHistoryB: IRenderTexture | null = null;
	private _volumetricReservoirHistoryA: IRenderTexture | null = null;
	private _volumetricReservoirHistoryB: IRenderTexture | null = null;
	private _motionHistoryA: IRenderTexture | null = null;
	private _motionHistoryB: IRenderTexture | null = null;
	private _postGraphExecuted = false;
	private _hasPresentedInFrame = false;
	private _taaHistoryValid = false;
	private _taaHistoryFlip = false;
	private _taaHistoryUpdated = false;
	private _ssrHistoryValid = false;
	private _ssrHistoryFlip = false;
	private _ssrHistoryUpdated = false;
	private _volumetricHistoryValid = false;
	private _volumetricHistoryFlip = false;
	private _volumetricHistoryUpdated = false;
	private _motionHistoryValid = false;
	private _motionHistoryFlip = false;
	private _mrtEnabled = true;
	private _mrtSupportChecked = false;
	private _featureHistoryKey = "";
	private _postGraph: WebGPUPostProcessGraph;
	private _postRuntime: WebGPUPostProcessRuntime;
	private _presentShaderModule: IShaderModule | null = null;
	private _presentPipeline: IRenderPipeline | null = null;
	private _presentSampler: ISampler | null = null;
	private _presentParamsBuffer: IRenderBuffer | null = null;
	private _presentBinding: IBindingGroup | null = null;
	private _presentBindingSource: IRenderTexture | null = null;
	private _depthDirtyClearShaderModule: IShaderModule | null = null;
	private _depthDirtyClearPipelines = new Map<string, IRenderPipeline>();
	private _texturePools = new Map<string, TexturePool>();
	private _texturePoolOwners = new Map<IRenderTexture, TexturePool>();
	private readonly _passHandlers: Map<FramePass["stage"], WebGPUFramePassHandler>;

	constructor(backend: WebGPUBackend, resources: WebGPURenderResources) {
		this._backend = backend;
		this._resources = resources;
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
		this._postGraph = new WebGPUPostProcessGraph(this._createDefaultPasses());
		this._passHandlers = this._createPassHandlers();
	}

	public beginFrame(context: FrameContext): void {
		this._frameContext = context;
		this._postGraphExecuted = false;
		this._hasPresentedInFrame = false;
		this._taaHistoryUpdated = false;
		this._ssrHistoryUpdated = false;
		this._volumetricHistoryUpdated = false;
		const targetWidth = this._resolveAttachmentDimension(
			context.attachments.width
		);
		const targetHeight = this._resolveAttachmentDimension(
			context.attachments.height
		);

		if (targetWidth <= 0 || targetHeight <= 0) {
			this._destroyFrameTargets();
			this._resources.setSceneTargetMode("single");
			this._frameContext = null;
			this._encoder = null;
			return;
		}

		this._encoder = this._backend.createCommandEncoder();

		this._ensureMRTSupport();
		this._handleFeatureHistoryTransitions(context);
		if (context.incremental?.temporalHistoryReset) {
			this._taaHistoryValid = false;
			this._ssrHistoryValid = false;
			this._volumetricHistoryValid = false;
			this._motionHistoryValid = false;
		}
		if (this._mrtEnabled) {
			const ssaoDownsample = clampDownsample(
				context.features.ssaoOptions?.downsample,
				DEFAULT_SSAO_OPTIONS.downsample
			);
			const ssrDownsample = clampDownsample(
				context.features.ssrOptions?.downsample,
				DEFAULT_SSR_OPTIONS.downsample
			);
			this._ensureFrameTargets(
				targetWidth,
				targetHeight,
				ssaoDownsample,
				ssrDownsample
			);
			this._resources.setSceneTargetMode("mrt");
		} else {
			this._destroyFrameTargets();
			this._resources.setSceneTargetMode("single");
		}
	}

	public registerPostProcessPass(pass: WebGPUPostProcessPassPlugin): void {
		this._postGraph.registerPass(pass);
	}

	public unregisterPostProcessPass(id: string): void {
		this._postGraph.unregisterPass(id);
	}

	/**
	 * Force frame targets to be rebuilt on the next beginFrame().
	 * Call on canvas resize so the post-process pipeline picks up
	 * the new dimensions.
	 */
	public invalidateFrameTargets(): void {
		this._destroyFrameTargets();
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
		this._destroyManagedResource(this._depthDirtyClearShaderModule);
		for (const pipeline of this._depthDirtyClearPipelines.values()) {
			this._destroyManagedResource(pipeline);
		}
		this._depthDirtyClearShaderModule = null;
		this._depthDirtyClearPipelines.clear();
		this._postRuntime.onShaderRuntimeChanged();
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
		const sceneMode =
			this._mrtEnabled && plan.sceneTargetMode === "mrt" ? "mrt" : "single";
		this._resources.setSceneTargetMode(sceneMode);

		try {
			await this._ensurePresentResources();
			compiled++;
		} catch (error) {
			failed++;
			errors.push(toShaderCompileError(error, "webgpu", "WebGPUPresentWarmup"));
		}

		const enabledPasses = this._postGraph.getExecutionOrder(
			context.features,
			() => {}
		);
		const allowedPassIds = new Set(plan.postProcessPasses);
		const hints = new Set<string>();
		for (const pass of enabledPasses) {
			if (!allowedPassIds.has(pass.id)) {
				continue;
			}
			for (const hint of pass.precompileHints ?? [`postprocess:${pass.id}`]) {
				hints.add(hint);
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

		return {
			phase: "webgpu-frame",
			total,
			compiled,
			skipped: Math.max(0, total - compiled - failed),
			failed,
			errors,
		};
	}

	/**
	 * Release all GPU resources held by this executor.
	 */
	public destroy(): void {
		this._destroyFrameTargets();
		this._destroyTexturePools();
		this._postRuntime.invalidateBindings();
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
			return;
		}

		if (this._mrtEnabled && this._frameTargets && !this._hasPresentedInFrame) {
			await this._presentToCanvas(
				this._frameTargets.sceneColor,
				this._frameContext?.features.enableGamma !== false
			);
		}

		const encoder = this._encoder;
		const width = this._targetWidth;
		const height = this._targetHeight;

		this._backend.submit([encoder.finish()]);
		this._encoder = null;
		this._frameContext = null;

		const motionSource =
			this._mrtEnabled ? this._frameTargets?.gMotionDepth : null;
		const motionTarget =
			this._mrtEnabled ? this._frameTargets?.motionHistoryWrite : null;
		if (motionSource && motionTarget && width > 0 && height > 0) {
			this._backend.copyTextureToTexture(
				{ texture: motionSource },
				{ texture: motionTarget },
				{ width, height, depthOrArrayLayers: 1 }
			);
			this._motionHistoryValid = true;
			this._motionHistoryFlip = !this._motionHistoryFlip;
			if (this._frameTargets) {
				this._applyMotionHistoryFlip(this._frameTargets);
			}
		}

		if (this._taaHistoryUpdated) {
			this._taaHistoryValid = true;
			this._taaHistoryFlip = !this._taaHistoryFlip;
			if (this._frameTargets) {
				this._applyTAAHistoryFlip(this._frameTargets);
			}
		}

		if (this._ssrHistoryUpdated) {
			this._ssrHistoryValid = true;
			this._ssrHistoryFlip = !this._ssrHistoryFlip;
			if (this._frameTargets) {
				this._applySSRHistoryFlip(this._frameTargets);
			}
		}

		if (this._volumetricHistoryUpdated) {
			this._volumetricHistoryValid = true;
			this._volumetricHistoryFlip = !this._volumetricHistoryFlip;
			if (this._frameTargets) {
				this._applyVolumetricHistoryFlip(this._frameTargets);
			}
		}
	}

	private _createPassHandlers(): Map<
		FramePass["stage"],
		WebGPUFramePassHandler
	> {
		const handlers = new Map<FramePass["stage"], WebGPUFramePassHandler>([
			[
				"shadow",
				async (context) => {
					await this._resources.renderShadows(context);
				},
			],
			[
				"main-opaque",
				async (context) => {
					await this._recordMainPass(
						context,
						context.scene.opaquePackets,
						true
					);
				},
			],
			[
				"main-transparent",
				async (context) => {
					await this._recordMainPass(
						context,
						context.scene.transparentPackets,
						false
					);
				},
			],
			[
				"particles",
				async (context) => {
					await this._recordParticlePass(context);
				},
			],
		]);
		const runPostProcess: WebGPUFramePassHandler = async (context) => {
			if (!this._postGraphExecuted) {
				await this._runPostGraph(context);
			}
		};
		for (const stage of POST_PROCESS_STAGES) {
			handlers.set(stage, runPostProcess);
		}
		return handlers;
	}

	private _createDefaultPasses(): WebGPUPostProcessPassPlugin[] {
		return [
			{
				id: "ssao",
				kind: "compute",
				dependsOn: [],
				precompileHints: ["postprocess:ssao"],
				isEnabled: (features) => features.enableSSAO,
				execute: async (ctx) => {
					await this._postRuntime.executePass({
						passId: "ssao",
						encoder: ctx.encoder,
						targets: ctx.targets,
						frameContext: ctx.frameContext,
					});
				},
			},
			{
				id: "ssgi",
				kind: "compute",
				dependsOn: ["ssao"],
				precompileHints: ["postprocess:ssgi"],
				isEnabled: (features) => features.enableSSGI,
				execute: async (ctx) => {
					await this._postRuntime.executePass({
						passId: "ssgi",
						encoder: ctx.encoder,
						targets: ctx.targets,
						frameContext: ctx.frameContext,
					});
				},
			},
			{
				id: "taa",
				kind: "compute",
				dependsOn: ["ssgi", "ssao"],
				precompileHints: ["postprocess:taa"],
				isEnabled: (features) => features.enableTAA,
				execute: async (ctx) => {
					const historyValid =
						this._taaHistoryValid && this._motionHistoryValid;
					const result = await this._postRuntime.executePass({
						passId: "taa",
						encoder: ctx.encoder,
						targets: ctx.targets,
						frameContext: ctx.frameContext,
						historyValid,
					});
					this._taaHistoryUpdated = result.historyUpdated === true;
				},
			},
			{
				id: "ssr",
				kind: "compute",
				dependsOn: ["taa"],
				precompileHints: ["postprocess:ssr", "postprocess:hiz"],
				isEnabled: (features) => features.enableSSR,
				execute: async (ctx) => {
					const historyValid =
						this._ssrHistoryValid && this._motionHistoryValid;
					const result = await this._postRuntime.executePass({
						passId: "ssr",
						encoder: ctx.encoder,
						targets: ctx.targets,
						frameContext: ctx.frameContext,
						historyValid,
						frameBinding: this._resources.getFrameBinding(),
					});
					this._ssrHistoryUpdated = result.historyUpdated === true;
				},
			},
			{
				id: "volumetric",
				kind: "compute",
				dependsOn: ["ssr"],
				precompileHints: ["postprocess:volumetric", "postprocess:hiz"],
				isEnabled: (features) => features.enableVolumetric,
				execute: async (ctx) => {
					const historyValid =
						this._volumetricHistoryValid && this._motionHistoryValid;
					const lightingState = this._resources.getLightingState();
					const result = await this._postRuntime.executePass({
						passId: "volumetric",
						encoder: ctx.encoder,
						targets: ctx.targets,
						frameContext: ctx.frameContext,
						historyValid,
						frameBinding: this._resources.getFrameBinding(),
						lightingState,
					});
					this._volumetricHistoryUpdated = result.historyUpdated === true;
				},
			},
			{
				id: "fog",
				kind: "compute",
				dependsOn: ["volumetric"],
				precompileHints: ["postprocess:fog"],
				isEnabled: (features) => isFogPostProcessEnabled(features),
				execute: async (ctx) => {
					await this._postRuntime.executePass({
						passId: "fog",
						encoder: ctx.encoder,
						targets: ctx.targets,
						frameContext: ctx.frameContext,
					});
				},
			},
			{
				id: "motion-blur",
				kind: "compute",
				dependsOn: ["fog"],
				precompileHints: ["postprocess:motion-blur"],
				isEnabled: (features) => features.enableMotionBlur,
				execute: async (ctx) => {
					await this._postRuntime.executePass({
						passId: "motion-blur",
						encoder: ctx.encoder,
						targets: ctx.targets,
						frameContext: ctx.frameContext,
					});
				},
			},
			{
				id: "dof",
				kind: "compute",
				dependsOn: ["motion-blur"],
				precompileHints: ["postprocess:dof"],
				isEnabled: (features) => features.enableDOF,
				execute: async (ctx) => {
					await this._postRuntime.executePass({
						passId: "dof",
						encoder: ctx.encoder,
						targets: ctx.targets,
						frameContext: ctx.frameContext,
					});
				},
			},
			{
				id: "bloom",
				kind: "compute",
				dependsOn: ["dof"],
				precompileHints: [
					"postprocess:bloom",
				],
				isEnabled: (features) => features.enableBloom,
				execute: async (ctx) => {
					await this._postRuntime.executePass({
						passId: "bloom",
						encoder: ctx.encoder,
						targets: ctx.targets,
						frameContext: ctx.frameContext,
					});
				},
			},
			{
				id: "fxaa",
				kind: "compute",
				dependsOn: ["bloom"],
				precompileHints: ["postprocess:fxaa"],
				isEnabled: (features) => features.enableFXAA,
				execute: async (ctx) => {
					await this._postRuntime.executePass({
						passId: "fxaa",
						encoder: ctx.encoder,
						targets: ctx.targets,
						frameContext: ctx.frameContext,
					});
				},
			},
			{
				id: "interaction-outline",
				kind: "compute",
				dependsOn: ["fxaa"],
				precompileHints: ["postprocess:interaction-outline"],
				isEnabled: () => true,
				execute: async (ctx) => {
					const interaction = ctx.frameContext.transient.get(
						INTERACTION_TRANSIENT_STATE_KEY
					) as InteractionTransientState | null | undefined;
					if ((interaction?.selectedEntityIds?.length ?? 0) === 0) {
						return;
					}
					await this._postRuntime.executePass({
						passId: "interaction-outline",
						encoder: ctx.encoder,
						targets: ctx.targets,
						frameContext: ctx.frameContext,
						state: interaction,
					});
				},
			},
			{
				id: "gamma",
				kind: "render",
				dependsOn: ["interaction-outline"],
				precompileHints: [],
				isEnabled: (features) => features.enableGamma,
				execute: async (ctx) => {
					await this._presentToCanvas(ctx.targets.sceneColor, true);
				},
			},
		];
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

	private _ensureFrameTargets(
		width: number,
		height: number,
		ssaoDownsample: number,
		ssrDownsample: number
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
			this._targetSSAODownsample === ssaoDownsample &&
			this._targetSSRDownsample === ssrDownsample &&
			this._targetMSAASampleCount === msaaSampleCount
		) {
			this._frameTargets.sceneColor = this._frameTargets.sceneColorMain;
			this._applyTAAHistoryFlip(this._frameTargets);
			this._applySSRHistoryFlip(this._frameTargets);
			this._applyVolumetricHistoryFlip(this._frameTargets);
			this._applyMotionHistoryFlip(this._frameTargets);
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
			this._targetSSAODownsample = ssaoDownsample;
			this._targetSSRDownsample = ssrDownsample;
			this._targetMSAASampleCount = msaaSampleCount;
			this._taaHistoryValid = false;
			this._ssrHistoryValid = false;
			this._volumetricHistoryValid = false;
			this._motionHistoryValid = false;
			this._taaHistoryFlip = false;
			this._ssrHistoryFlip = false;
			this._volumetricHistoryFlip = false;
			this._motionHistoryFlip = false;

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
			const historyA = acquireTexture(
				"rgba16-storage",
				rgba16StoragePool,
				width,
				height,
				TextureFormat.RGBA16Float
			);
			const historyB = acquireTexture(
				"rgba16-storage",
				rgba16StoragePool,
				width,
				height,
				TextureFormat.RGBA16Float
			);
			const ssrWidth = Math.max(1, Math.floor(width / ssrDownsample));
			const ssrHeight = Math.max(1, Math.floor(height / ssrDownsample));
			const ssrRaw = acquireTexture(
				"rgba16-storage",
				rgba16StoragePool,
				ssrWidth,
				ssrHeight,
				TextureFormat.RGBA16Float
			);
			const ssrHistoryA = acquireTexture(
				"rgba16-storage",
				rgba16StoragePool,
				ssrWidth,
				ssrHeight,
				TextureFormat.RGBA16Float
			);
			const ssrHistoryB = acquireTexture(
				"rgba16-storage",
				rgba16StoragePool,
				ssrWidth,
				ssrHeight,
				TextureFormat.RGBA16Float
			);
			const volumetricHistoryA = acquireTexture(
				"rgba16-storage",
				rgba16StoragePool,
				width,
				height,
				TextureFormat.RGBA16Float
			);
			const volumetricHistoryB = acquireTexture(
				"rgba16-storage",
				rgba16StoragePool,
				width,
				height,
				TextureFormat.RGBA16Float
			);
			const volumetricReservoirHistoryA = acquireTexture(
				"rgba16-storage",
				rgba16StoragePool,
				width,
				height,
				TextureFormat.RGBA16Float
			);
			const volumetricReservoirHistoryB = acquireTexture(
				"rgba16-storage",
				rgba16StoragePool,
				width,
				height,
				TextureFormat.RGBA16Float
			);
			const motionHistoryA = acquireTexture(
				"motion-history",
				{
					usage: TextureUsage.TextureBinding | TextureUsage.CopyDst,
					label: "WebGPUMotionHistory",
				},
				width,
				height,
				TextureFormat.RGBA16Float
			);
			const motionHistoryB = acquireTexture(
				"motion-history",
				{
					usage: TextureUsage.TextureBinding | TextureUsage.CopyDst,
					label: "WebGPUMotionHistory",
				},
				width,
				height,
				TextureFormat.RGBA16Float
			);
			const aoRaw = acquireTexture(
				"rgba16-storage",
				rgba16StoragePool,
				Math.max(1, Math.floor(width / ssaoDownsample)),
				Math.max(1, Math.floor(height / ssaoDownsample)),
				TextureFormat.RGBA16Float
			);
			const aoBlur = acquireTexture(
				"rgba16-storage",
				rgba16StoragePool,
				Math.max(1, Math.floor(width / ssaoDownsample)),
				Math.max(1, Math.floor(height / ssaoDownsample)),
				TextureFormat.RGBA16Float
			);
			const hiZ = acquireTexture(
				"hiz",
				{
					usage: TextureUsage.TextureBinding | TextureUsage.StorageBinding,
					label: "WebGPUHiZDepth",
					mipLevelCount: (poolWidth, poolHeight) =>
						Math.floor(Math.log2(Math.max(poolWidth, poolHeight))) + 1,
				},
				width,
				height,
				TextureFormat.RGBA16Float
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
				depth,
				aoRaw,
				aoBlur,
				ssrRaw,
				hiZ,
				historyRead: historyA,
				historyWrite: historyB,
				ssrHistoryRead: ssrHistoryA,
				ssrHistoryWrite: ssrHistoryB,
				volumetricHistoryRead: volumetricHistoryA,
				volumetricHistoryWrite: volumetricHistoryB,
				volumetricReservoirHistoryRead: volumetricReservoirHistoryA,
				volumetricReservoirHistoryWrite: volumetricReservoirHistoryB,
				motionHistoryRead: motionHistoryA,
				motionHistoryWrite: motionHistoryB,
			};
			this._msaaTargets = nextMSAATargets;
			this._frameTargets = nextFrameTargets;
			this._taaHistoryA = historyA;
			this._taaHistoryB = historyB;
			this._ssrHistoryA = ssrHistoryA;
			this._ssrHistoryB = ssrHistoryB;
			this._volumetricHistoryA = volumetricHistoryA;
			this._volumetricHistoryB = volumetricHistoryB;
			this._volumetricReservoirHistoryA = volumetricReservoirHistoryA;
			this._volumetricReservoirHistoryB = volumetricReservoirHistoryB;
			this._motionHistoryA = motionHistoryA;
			this._motionHistoryB = motionHistoryB;
			committed = true;
			this._applyTAAHistoryFlip(nextFrameTargets);
			this._applySSRHistoryFlip(nextFrameTargets);
			this._applyVolumetricHistoryFlip(nextFrameTargets);
			this._applyMotionHistoryFlip(nextFrameTargets);
		} catch (error) {
			if (!committed) {
				for (const texture of new Set(acquiredTextures)) {
					this._releasePooledTexture(texture);
				}
			}
			this._destroyFrameTargets();
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
				this._ensureFrameTargets(width, height, ssaoDownsample, ssrDownsample);
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

	private _applyTAAHistoryFlip(targets: WebGPUFrameTargets): void {
		if (!this._taaHistoryA || !this._taaHistoryB) return;
		targets.historyRead =
			this._taaHistoryFlip ? this._taaHistoryB : this._taaHistoryA;
		targets.historyWrite =
			this._taaHistoryFlip ? this._taaHistoryA : this._taaHistoryB;
	}

	private _applySSRHistoryFlip(targets: WebGPUFrameTargets): void {
		if (!this._ssrHistoryA || !this._ssrHistoryB) return;
		targets.ssrHistoryRead =
			this._ssrHistoryFlip ? this._ssrHistoryB : this._ssrHistoryA;
		targets.ssrHistoryWrite =
			this._ssrHistoryFlip ? this._ssrHistoryA : this._ssrHistoryB;
	}

	private _applyVolumetricHistoryFlip(targets: WebGPUFrameTargets): void {
		if (
			!this._volumetricHistoryA ||
			!this._volumetricHistoryB ||
			!this._volumetricReservoirHistoryA ||
			!this._volumetricReservoirHistoryB
		) {
			return;
		}
		targets.volumetricHistoryRead =
			this._volumetricHistoryFlip ?
				this._volumetricHistoryB
			:	this._volumetricHistoryA;
		targets.volumetricHistoryWrite =
			this._volumetricHistoryFlip ?
				this._volumetricHistoryA
			:	this._volumetricHistoryB;
		targets.volumetricReservoirHistoryRead =
			this._volumetricHistoryFlip ?
				this._volumetricReservoirHistoryB
			:	this._volumetricReservoirHistoryA;
		targets.volumetricReservoirHistoryWrite =
			this._volumetricHistoryFlip ?
				this._volumetricReservoirHistoryA
			:	this._volumetricReservoirHistoryB;
	}

	private _applyMotionHistoryFlip(targets: WebGPUFrameTargets): void {
		if (!this._motionHistoryA || !this._motionHistoryB) return;
		targets.motionHistoryRead =
			this._motionHistoryFlip ? this._motionHistoryB : this._motionHistoryA;
		targets.motionHistoryWrite =
			this._motionHistoryFlip ? this._motionHistoryA : this._motionHistoryB;
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
			textures.add(this._frameTargets.depth);
			textures.add(this._frameTargets.aoRaw);
			textures.add(this._frameTargets.aoBlur);
			textures.add(this._frameTargets.ssrRaw);
			textures.add(this._frameTargets.hiZ);
			textures.add(this._frameTargets.historyRead);
			textures.add(this._frameTargets.historyWrite);
			textures.add(this._frameTargets.ssrHistoryRead);
			textures.add(this._frameTargets.ssrHistoryWrite);
			textures.add(this._frameTargets.volumetricHistoryRead);
			textures.add(this._frameTargets.volumetricHistoryWrite);
			textures.add(this._frameTargets.volumetricReservoirHistoryRead);
			textures.add(this._frameTargets.volumetricReservoirHistoryWrite);
			textures.add(this._frameTargets.motionHistoryRead);
			textures.add(this._frameTargets.motionHistoryWrite);
		}
		if (this._msaaTargets) {
			textures.add(this._msaaTargets.sceneColorMain);
			textures.add(this._msaaTargets.gAlbedoAlpha);
			textures.add(this._msaaTargets.gNormalRoughMetal);
			textures.add(this._msaaTargets.gEmissiveOcclusion);
			textures.add(this._msaaTargets.gMotionDepth);
			textures.add(this._msaaTargets.depth);
		}
		for (const texture of textures) {
			this._releasePooledTexture(texture);
		}
		this._frameTargets = null;
		this._msaaTargets = null;
		this._taaHistoryA = null;
		this._taaHistoryB = null;
		this._ssrHistoryA = null;
		this._ssrHistoryB = null;
		this._volumetricHistoryA = null;
		this._volumetricHistoryB = null;
		this._volumetricReservoirHistoryA = null;
		this._volumetricReservoirHistoryB = null;
		this._motionHistoryA = null;
		this._motionHistoryB = null;
		this._destroyBindingGroup(this._presentBinding);
		this._presentBinding = null;
		this._presentBindingSource = null;
		this._targetWidth = 0;
		this._targetHeight = 0;
		this._targetSSAODownsample = DEFAULT_SSAO_OPTIONS.downsample;
		this._targetSSRDownsample = DEFAULT_SSR_OPTIONS.downsample;
		this._targetMSAASampleCount = 1;
		this._taaHistoryValid = false;
		this._ssrHistoryValid = false;
		this._volumetricHistoryValid = false;
		this._motionHistoryValid = false;
		this._taaHistoryFlip = false;
		this._ssrHistoryFlip = false;
		this._volumetricHistoryFlip = false;
		this._motionHistoryFlip = false;
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

	private _destroyManagedResource(resource: unknown): void {
		const destroyFn = (resource as { destroy?: () => void } | null)?.destroy;
		if (typeof destroyFn === "function") {
			destroyFn.call(resource);
		}
	}

	private _handleFeatureHistoryTransitions(context: FrameContext): void {
		const historyKey =
			`mrt:${this._mrtEnabled ? 1 : 0}` +
			`|ssao:${context.features.enableSSAO ? 1 : 0}` +
			`|ssgi:${context.features.enableSSGI ? 1 : 0}` +
			`|taa:${context.features.enableTAA ? 1 : 0}` +
			`|ssr:${context.features.enableSSR ? 1 : 0}` +
			`|vol:${context.features.enableVolumetric ? 1 : 0}` +
			`|fog:${isFogPostProcessEnabled(context.features) ? 1 : 0}` +
			`|mblur:${context.features.enableMotionBlur ? 1 : 0}` +
			`|dof:${context.features.enableDOF ? 1 : 0}` +
			`|bloom:${context.features.enableBloom ? 1 : 0}` +
			`|fxaa:${context.features.enableFXAA ? 1 : 0}`;

		if (this._featureHistoryKey && this._featureHistoryKey !== historyKey) {
			this._taaHistoryValid = false;
			this._ssrHistoryValid = false;
			this._volumetricHistoryValid = false;
			this._motionHistoryValid = false;
		}
		this._featureHistoryKey = historyKey;
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
		context: FrameContext | null
	): Array<{ x: number; y: number; width: number; height: number }> {
		if (!context) {
			return [{
				x: 0,
				y: 0,
				width: Math.max(1, this._targetWidth),
				height: Math.max(1, this._targetHeight),
			}];
		}
		if (!this._isIncrementalPartial(context)) {
			return [{
				x: 0,
				y: 0,
				width: Math.max(1, context.attachments.width),
				height: Math.max(1, context.attachments.height),
			}];
		}
		return context.incremental.dirtyRects.map((rect) => ({
			x: Math.max(0, Math.floor(rect.x)),
			y: Math.max(0, Math.floor(rect.y)),
			width: Math.max(1, Math.floor(rect.width)),
			height: Math.max(1, Math.floor(rect.height)),
		}));
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

	private async _runPostGraph(context: FrameContext): Promise<void> {
		this._postGraphExecuted = true;
		if (!this._mrtEnabled || !this._frameTargets || !this._encoder) {
			return;
		}

		this._frameTargets.sceneColor = this._frameTargets.sceneColorMain;
		context.transient.set(
			"webgpu-taa-history-valid",
			this._taaHistoryValid && this._motionHistoryValid
		);
		context.transient.set(
			"webgpu-ssr-history-valid",
			this._ssrHistoryValid && this._motionHistoryValid
		);
		const postContext: WebGPUPostProcessPassContext = {
			backend: this._backend,
			encoder: this._encoder,
			frameContext: context,
			targets: this._frameTargets,
		};
		const executed = await this._postGraph.execute(
			postContext,
			context.features,
			(key, message) =>
				Logger.warn(`[${key}] ${message}`, {
					scope: "WebGPUFrameExecutor",
					onceKey: key,
				})
		);
		context.transient.set("webgpu-post-order", executed);

		if (!executed.includes("gamma")) {
			await this._presentToCanvas(
				this._frameTargets.sceneColor,
				context.features.enableGamma !== false
			);
		}
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
		const dirtyRects = this._resolveDirtyRects(this._frameContext);
		for (const rect of dirtyRects) {
			this._encoder.setScissorRect?.(rect.x, rect.y, rect.width, rect.height);
			this._encoder.draw(3);
		}
		this._encoder.endRenderPass();
		this._hasPresentedInFrame = true;
	}

	private async _recordMainPass(
		context: FrameContext,
		packets: DrawPacket[],
		clearAttachments: boolean
	): Promise<void> {
		if (!this._encoder) return;
		await this._resources.buildClusteredLighting(this._encoder);
		const incrementalPartial = this._isIncrementalPartial(context);
		if (!this._mrtEnabled || !this._frameTargets) {
			this._resources.setSceneTargetMode("single");
			await this._recordLegacyMainPass(context, packets, clearAttachments);
			return;
		}
		this._resources.setSceneTargetMode("mrt");
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
		const dirtyRects = this._resolveDirtyRects(context);
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

		let skyboxDrawn = false;
		if (shouldClearAttachments) {
			const skyboxResources = await this._resources.getSkyboxResources();
			if (skyboxResources) {
				this._encoder.beginRenderPass({
					label: "WebGPUSkyboxMRT",
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
				this._encoder.setPipeline(skyboxResources.pipeline);
				this._encoder.setBindingGroup(0, skyboxResources.frameBinding);
				this._encoder.draw(3);
				this._encoder.endRenderPass();
				skyboxDrawn = true;
			}
		}

		this._encoder.beginRenderPass({
			label:
				shouldClearAttachments ? "WebGPUMainMRT_Clear" : "WebGPUMainMRT_Load",
			colorAttachments: [
				{
					view: sceneColorAttachment,
					resolveTarget:
						msaaTargets ? this._frameTargets.sceneColorMain : undefined,
					clearValue: { r: 0, g: 0, b: 0, a: 1 },
					loadOp: shouldClearAttachments && !skyboxDrawn ? "clear" : "load",
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
				depthLoadOp:
					depthPartialReuseApplied ?
						"load"
					: incrementalPartial || (shouldClearAttachments && !skyboxDrawn) ?
						"clear"
					:	"load",
				depthStoreOp: "store",
			},
		});

		for (const rect of dirtyRects) {
			const packetsInRect = this._resolvePacketsForRect(context, packets, rect);
			if (packetsInRect.length === 0) {
				continue;
			}
			this._encoder.setScissorRect?.(rect.x, rect.y, rect.width, rect.height);
			for (const packet of packetsInRect) {
				const resourcesList = await this._resources.getDrawResources(packet);
				if (!resourcesList) continue;

				for (const resources of resourcesList) {
					this._encoder.setPipeline(resources.pipeline);
					this._encoder.setBindingGroup(0, resources.frameBinding);
					this._encoder.setBindingGroup(1, resources.modelBinding);
					this._encoder.setBindingGroup(2, resources.clusteredBinding);
					this._encoder.setVertexBuffer(0, resources.vertexBuffer);
					this._encoder.setIndexBuffer(resources.indexBuffer, "uint32");
					this._encoder.drawIndexed(resources.indexCount);
				}
			}
		}

		this._encoder.endRenderPass();
	}

	private async _recordLegacyMainPass(
		context: FrameContext,
		packets: DrawPacket[],
		clearAttachments: boolean
	): Promise<void> {
		if (!this._encoder) return;
		await this._resources.buildClusteredLighting(this._encoder);
		const incrementalPartial = this._isIncrementalPartial(context);
		const colorTexture = this._backend.getCanvasColorTexture();
		const depthTexture = this._backend.getCanvasDepthTexture();
		const shouldClearAttachments = clearAttachments && !incrementalPartial;
		const dirtyRects = this._resolveDirtyRects(context);
		let depthPartialReuseApplied = false;
		if (incrementalPartial && dirtyRects.length > 0) {
			depthPartialReuseApplied = await this._clearDepthForDirtyRects(
				depthTexture,
				this._backend.canvasDepthFormat,
				1,
				dirtyRects
			);
		}

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
				depthLoadOp:
					depthPartialReuseApplied ?
						"load"
					: incrementalPartial || shouldClearAttachments ?
						"clear"
					:	"load",
				depthStoreOp: "store",
			},
		});

		if (shouldClearAttachments) {
			const skyboxResources = await this._resources.getSkyboxResources();
			if (skyboxResources) {
				this._encoder.setPipeline(skyboxResources.pipeline);
				this._encoder.setBindingGroup(0, skyboxResources.frameBinding);
				this._encoder.draw(3);
			}
		}

		for (const rect of dirtyRects) {
			const packetsInRect = this._resolvePacketsForRect(context, packets, rect);
			if (packetsInRect.length === 0) {
				continue;
			}
			this._encoder.setScissorRect?.(rect.x, rect.y, rect.width, rect.height);
			for (const packet of packetsInRect) {
				const resourcesList = await this._resources.getDrawResources(packet);
				if (!resourcesList) continue;

				for (const resources of resourcesList) {
					this._encoder.setPipeline(resources.pipeline);
					this._encoder.setBindingGroup(0, resources.frameBinding);
					this._encoder.setBindingGroup(1, resources.modelBinding);
					this._encoder.setBindingGroup(2, resources.clusteredBinding);
					this._encoder.setVertexBuffer(0, resources.vertexBuffer);
					this._encoder.setIndexBuffer(resources.indexBuffer, "uint32");
					this._encoder.drawIndexed(resources.indexCount);
				}
			}
		}

		this._encoder.endRenderPass();
	}

	private async _recordParticlePass(context: FrameContext): Promise<void> {
		if (!this._encoder) return;

		if (this._mrtEnabled && this._frameTargets) {
			const msaaTargets = this._msaaTargets;
			await this._resources.renderParticles(
				this._encoder,
				context,
				{
					color:
						msaaTargets?.sceneColorMain ?? this._frameTargets.sceneColorMain,
					colorResolve:
						msaaTargets ? this._frameTargets.sceneColorMain : undefined,
					depth: msaaTargets?.depth ?? this._frameTargets.depth,
				},
				"mrt"
			);
			return;
		}

		await this._resources.renderParticles(
			this._encoder,
			context,
			{
				color: this._backend.getCanvasColorTexture(),
				depth: this._backend.getCanvasDepthTexture(),
			},
			"single"
		);
	}
}

function clampDownsample(value: unknown, fallback: number): number {
	if (typeof value !== "number" || !Number.isFinite(value)) {
		return fallback;
	}
	return Math.min(8, Math.max(1, Math.floor(value)));
}
