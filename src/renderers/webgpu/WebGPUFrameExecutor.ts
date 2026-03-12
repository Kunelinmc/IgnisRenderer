import type { DrawPacket, FrameContext, FramePass } from "../../pipeline/types";
import {
	DEFAULT_SSAO_OPTIONS,
	DEFAULT_SSR_OPTIONS,
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

const POST_PROCESS_STAGES = new Set<FramePass["stage"]>([
	"ssao",
	"taa",
	"ssr",
	"volumetric",
	"fxaa",
	"gamma",
]);

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
	let gamma = max(presentParams.gamma, 0.01);
	let linearColor = max(sampled.rgb, vec3<f32>(0.0));
	let gammaColor = pow(linearColor, vec3<f32>(1.0 / gamma));
	let outputColor = select(linearColor, gammaColor, presentParams.applyGamma > 0.5);
	return vec4<f32>(clamp(outputColor, vec3<f32>(0.0), vec3<f32>(1.0)), sampled.a);
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
	private _warnedKeys = new Set<string>();
	private _postGraph: WebGPUPostProcessGraph;
	private _postRuntime: WebGPUPostProcessRuntime;
	private _presentShaderModule: IShaderModule | null = null;
	private _presentPipeline: IRenderPipeline | null = null;
	private _presentSampler: ISampler | null = null;
	private _presentParamsBuffer: IRenderBuffer | null = null;
	private _presentBinding: IBindingGroup | null = null;
	private _presentBindingSource: IRenderTexture | null = null;

	constructor(backend: WebGPUBackend, resources: WebGPURenderResources) {
		this._backend = backend;
		this._resources = resources;
		this._postRuntime = new WebGPUPostProcessRuntime(
			backend,
			(key, message) => this._warnOnce(key, message),
			resources.sceneFrameLayout
		);
		this._postGraph = new WebGPUPostProcessGraph(this._createDefaultPasses());
	}

	public beginFrame(context: FrameContext): void {
		this._frameContext = context;
		this._encoder = this._backend.createCommandEncoder();
		this._postGraphExecuted = false;
		this._hasPresentedInFrame = false;
		this._taaHistoryUpdated = false;
		this._ssrHistoryUpdated = false;
		this._volumetricHistoryUpdated = false;

		this._ensureMRTSupport();
		this._handleFeatureHistoryTransitions(context);
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
				context.attachments.width,
				context.attachments.height,
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

	/**
	 * Release all GPU resources held by this executor.
	 */
	public destroy(): void {
		this._destroyFrameTargets();
		this._postRuntime.invalidateBindings();
		this._presentShaderModule = null;
		this._presentPipeline = null;
		this._presentSampler = null;
		this._presentParamsBuffer?.destroy();
		this._presentParamsBuffer = null;
		this._presentBinding = null;
		this._presentBindingSource = null;
		this._encoder = null;
		this._frameContext = null;
	}

	public async executePass(
		pass: FramePass,
		context: FrameContext
	): Promise<void> {
		if (!this._encoder) return;

		switch (pass.stage) {
			case "shadow":
				this._resources.renderShadows(context);
				return;
			case "main-opaque":
				await this._recordMainPass(context.scene.opaquePackets, true);
				return;
			case "main-transparent":
				await this._recordMainPass(context.scene.transparentPackets, false);
				return;
			case "particles":
				await this._recordParticlePass(context);
				return;
		}

		if (POST_PROCESS_STAGES.has(pass.stage) && !this._postGraphExecuted) {
			await this._runPostGraph(context);
		}
	}

	public async endFrame(): Promise<void> {
		if (!this._encoder) return;

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

	private _createDefaultPasses(): WebGPUPostProcessPassPlugin[] {
		return [
			{
				id: "ssao",
				kind: "compute",
				dependsOn: [],
				isEnabled: (features) => features.enableSSAO,
				execute: async (ctx) => {
					await this._postRuntime.executeSSAO(
						ctx.encoder,
						ctx.targets,
						ctx.frameContext
					);
				},
			},
			{
				id: "taa",
				kind: "compute",
				dependsOn: ["ssao"],
				isEnabled: (features) => features.enableTAA,
				execute: async (ctx) => {
					const historyValid =
						this._taaHistoryValid && this._motionHistoryValid;
					this._taaHistoryUpdated = await this._postRuntime.executeTAA(
						ctx.encoder,
						ctx.targets,
						ctx.frameContext,
						historyValid
					);
				},
			},
			{
				id: "ssr",
				kind: "compute",
				dependsOn: ["taa"],
				isEnabled: (features) => features.enableSSR,
				execute: async (ctx) => {
					const historyValid =
						this._ssrHistoryValid && this._motionHistoryValid;
					this._ssrHistoryUpdated = await this._postRuntime.executeSSR(
						ctx.encoder,
						ctx.targets,
						ctx.frameContext,
						historyValid,
						this._resources.getFrameBinding()
					);
				},
			},
			{
				id: "volumetric",
				kind: "compute",
				dependsOn: ["ssr"],
				isEnabled: (features) => features.enableVolumetric,
				execute: async (ctx) => {
					const historyValid =
						this._volumetricHistoryValid && this._motionHistoryValid;
					const lightingState = this._resources.getLightingState();
					this._volumetricHistoryUpdated =
						await this._postRuntime.executeVolumetric(
							ctx.encoder,
							ctx.targets,
							ctx.frameContext,
							historyValid,
							this._resources.getFrameBinding(),
							lightingState
						);
				},
			},
			{
				id: "fxaa",
				kind: "compute",
				dependsOn: ["volumetric"],
				isEnabled: (features) => features.enableFXAA,
				execute: async (ctx) => {
					await this._postRuntime.executeFXAA(ctx.encoder, ctx.targets);
				},
			},
			{
				id: "gamma",
				kind: "render",
				dependsOn: ["fxaa"],
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
			this._warnOnce(
				"webgpu-mrt-disabled-attachments",
				`WebGPU device maxColorAttachments is ${maxColorAttachments}, requires ${WEBGPU_MRT_COLOR_TARGET_COUNT}; disabling MRT/GBuffer post-process pipeline`
			);
		}
		if (maxColorAttachmentBytesPerSample < WEBGPU_MRT_COLOR_BYTES_PER_SAMPLE) {
			this._warnOnce(
				"webgpu-mrt-disabled-bytes",
				`WebGPU device maxColorAttachmentBytesPerSample is ${maxColorAttachmentBytesPerSample}, requires ${WEBGPU_MRT_COLOR_BYTES_PER_SAMPLE}; disabling MRT/GBuffer post-process pipeline`
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

		const sceneColorMain = this._backend.createTexture({
			width,
			height,
			format: TextureFormat.RGBA16Float,
			usage:
				TextureUsage.RenderAttachment |
				TextureUsage.TextureBinding |
				TextureUsage.CopySrc |
				TextureUsage.CopyDst,
			label: "WebGPUSceneColorMain",
		});
		const postPing = this._backend.createTexture({
			width,
			height,
			format: TextureFormat.RGBA16Float,
			usage: TextureUsage.TextureBinding | TextureUsage.StorageBinding,
			label: "WebGPUPostPing",
		});
		const postPong = this._backend.createTexture({
			width,
			height,
			format: TextureFormat.RGBA16Float,
			usage: TextureUsage.TextureBinding | TextureUsage.StorageBinding,
			label: "WebGPUPostPong",
		});
		const gAlbedoAlpha = this._backend.createTexture({
			width,
			height,
			format: TextureFormat.RGBA8Unorm,
			usage: TextureUsage.RenderAttachment | TextureUsage.TextureBinding,
			label: "WebGPUGBuffer_AlbedoAlpha",
		});
		const gNormalRoughMetal = this._backend.createTexture({
			width,
			height,
			format: TextureFormat.RGBA16Float,
			usage: TextureUsage.RenderAttachment | TextureUsage.TextureBinding,
			label: "WebGPUGBuffer_NormalRoughMetal",
		});
		const gEmissiveOcclusion = this._backend.createTexture({
			width,
			height,
			format: TextureFormat.RGBA16Float,
			usage: TextureUsage.RenderAttachment | TextureUsage.TextureBinding,
			label: "WebGPUGBuffer_EmissiveOcclusion",
		});
		const gMotionDepth = this._backend.createTexture({
			width,
			height,
			format: TextureFormat.RGBA16Float,
			usage:
				TextureUsage.RenderAttachment |
				TextureUsage.TextureBinding |
				TextureUsage.CopySrc,
			label: "WebGPUGBuffer_MotionDepth",
		});
		const depth = this._backend.createTexture({
			width,
			height,
			format: TextureFormat.Depth32Float,
			usage: TextureUsage.RenderAttachment | TextureUsage.TextureBinding,
			label: "WebGPUDepthSampleable",
		});
		const historyA = this._backend.createTexture({
			width,
			height,
			format: TextureFormat.RGBA16Float,
			usage: TextureUsage.TextureBinding | TextureUsage.StorageBinding,
			label: "WebGPUTAAHistoryA",
		});
		const historyB = this._backend.createTexture({
			width,
			height,
			format: TextureFormat.RGBA16Float,
			usage: TextureUsage.TextureBinding | TextureUsage.StorageBinding,
			label: "WebGPUTAAHistoryB",
		});
		const ssrWidth = Math.max(1, Math.floor(width / ssrDownsample));
		const ssrHeight = Math.max(1, Math.floor(height / ssrDownsample));
		const ssrRaw = this._backend.createTexture({
			width: ssrWidth,
			height: ssrHeight,
			format: TextureFormat.RGBA16Float,
			usage: TextureUsage.TextureBinding | TextureUsage.StorageBinding,
			label: "WebGPUSSRRaw",
		});
		const ssrHistoryA = this._backend.createTexture({
			width: ssrWidth,
			height: ssrHeight,
			format: TextureFormat.RGBA16Float,
			usage: TextureUsage.TextureBinding | TextureUsage.StorageBinding,
			label: "WebGPUSSRHistoryA",
		});
		const ssrHistoryB = this._backend.createTexture({
			width: ssrWidth,
			height: ssrHeight,
			format: TextureFormat.RGBA16Float,
			usage: TextureUsage.TextureBinding | TextureUsage.StorageBinding,
			label: "WebGPUSSRHistoryB",
		});
		const volumetricHistoryA = this._backend.createTexture({
			width,
			height,
			format: TextureFormat.RGBA16Float,
			usage: TextureUsage.TextureBinding | TextureUsage.StorageBinding,
			label: "WebGPUVolumetricHistoryA",
		});
		const volumetricHistoryB = this._backend.createTexture({
			width,
			height,
			format: TextureFormat.RGBA16Float,
			usage: TextureUsage.TextureBinding | TextureUsage.StorageBinding,
			label: "WebGPUVolumetricHistoryB",
		});
		const volumetricReservoirHistoryA = this._backend.createTexture({
			width,
			height,
			format: TextureFormat.RGBA16Float,
			usage: TextureUsage.TextureBinding | TextureUsage.StorageBinding,
			label: "WebGPUVolumetricReservoirHistoryA",
		});
		const volumetricReservoirHistoryB = this._backend.createTexture({
			width,
			height,
			format: TextureFormat.RGBA16Float,
			usage: TextureUsage.TextureBinding | TextureUsage.StorageBinding,
			label: "WebGPUVolumetricReservoirHistoryB",
		});
		const motionHistoryA = this._backend.createTexture({
			width,
			height,
			format: TextureFormat.RGBA16Float,
			usage: TextureUsage.TextureBinding | TextureUsage.CopyDst,
			label: "WebGPUMotionHistoryA",
		});
		const motionHistoryB = this._backend.createTexture({
			width,
			height,
			format: TextureFormat.RGBA16Float,
			usage: TextureUsage.TextureBinding | TextureUsage.CopyDst,
			label: "WebGPUMotionHistoryB",
		});
		const aoRaw = this._backend.createTexture({
			width: Math.max(1, Math.floor(width / ssaoDownsample)),
			height: Math.max(1, Math.floor(height / ssaoDownsample)),
			format: TextureFormat.RGBA16Float,
			usage: TextureUsage.TextureBinding | TextureUsage.StorageBinding,
			label: "WebGPUSSAORaw",
		});
		const aoBlur = this._backend.createTexture({
			width: Math.max(1, Math.floor(width / ssaoDownsample)),
			height: Math.max(1, Math.floor(height / ssaoDownsample)),
			format: TextureFormat.RGBA16Float,
			usage: TextureUsage.TextureBinding | TextureUsage.StorageBinding,
			label: "WebGPUSSAOBlur",
		});
		const hiZ = this._backend.createTexture({
			width,
			height,
			format: TextureFormat.RGBA16Float,
			mipLevelCount: Math.floor(Math.log2(Math.max(width, height))) + 1,
			usage: TextureUsage.TextureBinding | TextureUsage.StorageBinding,
			label: "WebGPUHiZDepth",
		});
		const useMSAA = msaaSampleCount > 1;
		this._msaaTargets =
			useMSAA ?
				{
					sceneColorMain: this._backend.createTexture({
						width,
						height,
						sampleCount: msaaSampleCount,
						format: TextureFormat.RGBA16Float,
						usage: TextureUsage.RenderAttachment,
						label: "WebGPUSceneColorMainMSAA",
					}),
					gAlbedoAlpha: this._backend.createTexture({
						width,
						height,
						sampleCount: msaaSampleCount,
						format: TextureFormat.RGBA8Unorm,
						usage: TextureUsage.RenderAttachment,
						label: "WebGPUGBuffer_AlbedoAlphaMSAA",
					}),
					gNormalRoughMetal: this._backend.createTexture({
						width,
						height,
						sampleCount: msaaSampleCount,
						format: TextureFormat.RGBA16Float,
						usage: TextureUsage.RenderAttachment,
						label: "WebGPUGBuffer_NormalRoughMetalMSAA",
					}),
					gEmissiveOcclusion: this._backend.createTexture({
						width,
						height,
						sampleCount: msaaSampleCount,
						format: TextureFormat.RGBA16Float,
						usage: TextureUsage.RenderAttachment,
						label: "WebGPUGBuffer_EmissiveOcclusionMSAA",
					}),
					gMotionDepth: this._backend.createTexture({
						width,
						height,
						sampleCount: msaaSampleCount,
						format: TextureFormat.RGBA16Float,
						usage: TextureUsage.RenderAttachment,
						label: "WebGPUGBuffer_MotionDepthMSAA",
					}),
					depth: this._backend.createTexture({
						width,
						height,
						sampleCount: msaaSampleCount,
						format: TextureFormat.Depth32Float,
						usage: TextureUsage.RenderAttachment,
						label: "WebGPUDepthMSAA",
					}),
				}
			:	null;

		this._frameTargets = {
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
		this._applyTAAHistoryFlip(this._frameTargets);
		this._applySSRHistoryFlip(this._frameTargets);
		this._applyVolumetricHistoryFlip(this._frameTargets);
		this._applyMotionHistoryFlip(this._frameTargets);
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
		if (!this._frameTargets && !this._msaaTargets) return;
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
			texture.destroy();
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

	private _handleFeatureHistoryTransitions(context: FrameContext): void {
		const historyKey =
			`mrt:${this._mrtEnabled ? 1 : 0}` +
			`|ssao:${context.features.enableSSAO ? 1 : 0}` +
			`|taa:${context.features.enableTAA ? 1 : 0}` +
			`|ssr:${context.features.enableSSR ? 1 : 0}` +
			`|vol:${context.features.enableVolumetric ? 1 : 0}` +
			`|fxaa:${context.features.enableFXAA ? 1 : 0}`;

		if (this._featureHistoryKey && this._featureHistoryKey !== historyKey) {
			this._taaHistoryValid = false;
			this._ssrHistoryValid = false;
			this._volumetricHistoryValid = false;
			this._motionHistoryValid = false;
		}
		this._featureHistoryKey = historyKey;
	}

	private _warnOnce(key: string, message: string): void {
		if (this._warnedKeys.has(key)) return;
		this._warnedKeys.add(key);
		console.warn(message);
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
			(key, message) => this._warnOnce(key, message)
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
			this._presentShaderModule = await this._backend.createShaderModule({
				label: "WebGPUPresentShader",
				code: WEBGPU_PRESENT_SHADER,
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
			new Float32Array([2.2, applyGamma ? 1 : 0, 0, 0])
		);

		if (!this._presentBinding || this._presentBindingSource !== source) {
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
		this._encoder.draw(3);
		this._encoder.endRenderPass();
		this._hasPresentedInFrame = true;
	}

	private async _recordMainPass(
		packets: DrawPacket[],
		clearAttachments: boolean
	): Promise<void> {
		if (!this._encoder) return;
		if (!this._mrtEnabled || !this._frameTargets) {
			await this._recordLegacyMainPass(packets, clearAttachments);
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

		let skyboxDrawn = false;
		if (clearAttachments) {
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
			label: clearAttachments ? "WebGPUMainMRT_Clear" : "WebGPUMainMRT_Load",
			colorAttachments: [
				{
					view: sceneColorAttachment,
					resolveTarget:
						msaaTargets ? this._frameTargets.sceneColorMain : undefined,
					clearValue: { r: 0, g: 0, b: 0, a: 1 },
					loadOp: clearAttachments && !skyboxDrawn ? "clear" : "load",
					storeOp: "store",
				},
				{
					view: gAlbedoAttachment,
					resolveTarget:
						msaaTargets ? this._frameTargets.gAlbedoAlpha : undefined,
					clearValue: { r: 0, g: 0, b: 0, a: 1 },
					loadOp: clearAttachments ? "clear" : "load",
					storeOp: "store",
				},
				{
					view: gNormalAttachment,
					resolveTarget:
						msaaTargets ? this._frameTargets.gNormalRoughMetal : undefined,
					clearValue: { r: 0.5, g: 0.5, b: 1, a: 0 },
					loadOp: clearAttachments ? "clear" : "load",
					storeOp: "store",
				},
				{
					view: gEmissiveAttachment,
					resolveTarget:
						msaaTargets ? this._frameTargets.gEmissiveOcclusion : undefined,
					clearValue: { r: 0, g: 0, b: 0, a: 1 },
					loadOp: clearAttachments ? "clear" : "load",
					storeOp: "store",
				},
				{
					view: gMotionAttachment,
					resolveTarget:
						msaaTargets ? this._frameTargets.gMotionDepth : undefined,
					clearValue: { r: 0, g: 0, b: 0, a: 0 },
					loadOp: clearAttachments ? "clear" : "load",
					storeOp: "store",
				},
			],
			depthStencilAttachment: {
				view: depthAttachment,
				depthClearValue: 1,
				depthLoadOp: clearAttachments && !skyboxDrawn ? "clear" : "load",
				depthStoreOp: "store",
			},
		});

		for (const packet of packets) {
			const resourcesList = await this._resources.getDrawResources(packet);
			if (!resourcesList) continue;

			for (const resources of resourcesList) {
				this._encoder.setPipeline(resources.pipeline);
				this._encoder.setBindingGroup(0, resources.frameBinding);
				this._encoder.setBindingGroup(1, resources.modelBinding);
				this._encoder.setVertexBuffer(0, resources.vertexBuffer);
				this._encoder.setIndexBuffer(resources.indexBuffer, "uint32");
				this._encoder.drawIndexed(resources.indexCount);
			}
		}

		this._encoder.endRenderPass();
	}

	private async _recordLegacyMainPass(
		packets: DrawPacket[],
		clearAttachments: boolean
	): Promise<void> {
		if (!this._encoder) return;
		const colorTexture = this._backend.getCanvasColorTexture();
		const depthTexture = this._backend.getCanvasDepthTexture();

		this._encoder.beginRenderPass({
			colorAttachments: [
				{
					view: colorTexture,
					clearValue: { r: 0, g: 0, b: 0, a: 1 },
					loadOp: clearAttachments ? "clear" : "load",
					storeOp: "store",
				},
			],
			depthStencilAttachment: {
				view: depthTexture,
				depthClearValue: 1,
				depthLoadOp: clearAttachments ? "clear" : "load",
				depthStoreOp: "store",
			},
		});

		if (clearAttachments) {
			const skyboxResources = await this._resources.getSkyboxResources();
			if (skyboxResources) {
				this._encoder.setPipeline(skyboxResources.pipeline);
				this._encoder.setBindingGroup(0, skyboxResources.frameBinding);
				this._encoder.draw(3);
			}
		}

		for (const packet of packets) {
			const resourcesList = await this._resources.getDrawResources(packet);
			if (!resourcesList) continue;

			for (const resources of resourcesList) {
				this._encoder.setPipeline(resources.pipeline);
				this._encoder.setBindingGroup(0, resources.frameBinding);
				this._encoder.setBindingGroup(1, resources.modelBinding);
				this._encoder.setVertexBuffer(0, resources.vertexBuffer);
				this._encoder.setIndexBuffer(resources.indexBuffer, "uint32");
				this._encoder.drawIndexed(resources.indexCount);
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
