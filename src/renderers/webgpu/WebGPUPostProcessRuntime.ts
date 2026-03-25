import { CameraType } from "../../cameras/Camera";
import type { FrameContext } from "../../pipeline/types";
import {
	DEFAULT_SSAO_OPTIONS,
	DEFAULT_SSR_OPTIONS,
	DEFAULT_TAA_OPTIONS,
	DEFAULT_VOLUMETRIC_OPTIONS,
} from "../../pipeline/types";
import type { ICommandEncoder } from "../ICommandEncoder";
import {
	AddressMode,
	BufferUsage,
	FilterMode,
	type IComputePipeline,
	type IRenderBuffer,
	type IRenderTexture,
	type ISampler,
	type IShaderModule,
} from "../types";
import type { WebGPUBackend } from "../WebGPUBackend";
import type { WebGPUFrameTargets } from "./WebGPUPostProcessGraph";
import { loadPostProcessShaderPartComposite } from "../../shaders/webgpu/shaderSource";
import type { IBindingGroup } from "../types";
import type { WebGPULightingState } from "./types";
import { getWebGPUTexture } from "./WebGPUResourceAccess";
import { clamp } from "../../maths/Common";
import type { ShaderCompileError } from "../../shaders/runtime";
import { toShaderCompileError } from "../warmup/WarmupPlanner";

const WORKGROUP_SIZE = 8;

const DEFAULT_FXAA = {
	edgeThresholdMin: 0.03125,
	edgeThresholdMultiplier: 0.166,
	subpixQuality: 0.75,
};

const VOLUMETRIC_LIGHT_STRIDE_FLOATS = 12;
const MAX_VOLUMETRIC_LIGHTS = 65000;

interface CachedBindGroup {
	group: IBindingGroup;
	resources: readonly unknown[];
}

export class WebGPUPostProcessRuntime {
	private _backend: WebGPUBackend;
	private _warn: (key: string, message: string) => void;
	private _sampler: ISampler | null = null;
	private _ssaoModule: IShaderModule | null = null;
	private _ssaoRawPipeline: IComputePipeline | null = null;
	private _ssaoBlurPipeline: IComputePipeline | null = null;
	private _ssaoCombinePipeline: IComputePipeline | null = null;
	private _ssaoParams: IRenderBuffer | null = null;
	private _taaModule: IShaderModule | null = null;
	private _taaPipeline: IComputePipeline | null = null;
	private _taaParams: IRenderBuffer | null = null;
	private _hizModule: IShaderModule | null = null;
	private _hizInitPipeline: IComputePipeline | null = null;
	private _hizReducePipeline: IComputePipeline | null = null;
	private _ssrModule: IShaderModule | null = null;
	private _ssrTracePipeline: IComputePipeline | null = null;
	private _ssrComposePipeline: IComputePipeline | null = null;
	private _ssrTraceParams: IRenderBuffer | null = null;
	private _ssrComposeParams: IRenderBuffer | null = null;
	private _volumetricModule: IShaderModule | null = null;
	private _volumetricPipeline: IComputePipeline | null = null;
	private _volumetricParams: IRenderBuffer | null = null;
	private _volumetricLightBuffer: IRenderBuffer | null = null;
	private _volumetricLightCapacity = 0;
	private _volumetricFrameIndex = 0;
	private _fxaaModule: IShaderModule | null = null;
	private _fxaaPipeline: IComputePipeline | null = null;
	private _fxaaParams: IRenderBuffer | null = null;
	private _copyModule: IShaderModule | null = null;
	private _copyPipeline: IComputePipeline | null = null;
	private _hizViewCache = new WeakMap<object, GPUTextureView[]>();
	private _bindGroupCache = new Map<string, CachedBindGroup>();
	private _ssrTraceGroupLayout0: GPUBindGroupLayout | null = null;
	private _ssrTracePipelineLayout: GPUPipelineLayout | null = null;
	private _volumetricGroupLayout0: GPUBindGroupLayout | null = null;
	private _volumetricPipelineLayout: GPUPipelineLayout | null = null;
	private _frameBindGroupLayout: GPUBindGroupLayout | null = null;
	private _ssaoFrameIndex: number = 0;
	private _ssrFrameIndex: number = 0;

	constructor(
		backend: WebGPUBackend,
		warn: (key: string, message: string) => void,
		frameBindGroupLayout?: GPUBindGroupLayout
	) {
		this._backend = backend;
		this._warn = warn;
		this._frameBindGroupLayout = frameBindGroupLayout || null;
	}

	/**
	 * Invalidate all cached bind groups. Call when frame targets are
	 * destroyed/rebuilt (e.g. on resize) so stale texture references are
	 * not reused.
	 */
	public invalidateBindings(): void {
		this._destroyCachedBindGroups();
	}

	public onShaderRuntimeChanged(): void {
		this._destroyCachedBindGroups();
		this._ssaoModule = null;
		this._ssaoRawPipeline = null;
		this._ssaoBlurPipeline = null;
		this._ssaoCombinePipeline = null;
		this._taaModule = null;
		this._taaPipeline = null;
		this._hizModule = null;
		this._hizInitPipeline = null;
		this._hizReducePipeline = null;
		this._ssrModule = null;
		this._ssrTracePipeline = null;
		this._ssrComposePipeline = null;
		this._volumetricModule = null;
		this._volumetricPipeline = null;
		this._fxaaModule = null;
		this._fxaaPipeline = null;
		this._copyModule = null;
		this._copyPipeline = null;
	}

	public async warmupHints(
		hints: readonly string[]
	): Promise<{
		compiled: number;
		failed: number;
		errors: ShaderCompileError[];
	}> {
		let compiled = 0;
		let failed = 0;
		const errors: ShaderCompileError[] = [];
		const seen = new Set<string>();
		for (const hint of hints) {
			if (seen.has(hint)) {
				continue;
			}
			seen.add(hint);
			try {
				const warmed = await this._warmupHint(hint);
				if (warmed) {
					compiled++;
				}
			} catch (error) {
				failed++;
				errors.push(
					toShaderCompileError(error, "webgpu", `WebGPUPostWarmup:${hint}`)
				);
			}
		}
		return {
			compiled,
			failed,
			errors,
		};
	}

	private _getCachedBindGroup(
		key: string,
		pipeline: IComputePipeline,
		entries: Array<{ binding: number; resource: any }>,
		label: string
	): IBindingGroup {
		const resources = entries.map((e) => e.resource);
		const cached = this._bindGroupCache.get(key);
		if (cached && cached.resources.length === resources.length) {
			let match = true;
			for (let i = 0; i < resources.length; i++) {
				if (cached.resources[i] !== resources[i]) {
					match = false;
					break;
				}
			}
			if (match) return cached.group;
		}
		if (cached) {
			this._destroyBindingGroup(cached.group);
		}
		const group = this._backend.createBindingGroup({
			pipeline,
			layoutIndex: 0,
			entries,
			label,
		});
		this._bindGroupCache.set(key, { group, resources });
		return group;
	}

	private _destroyCachedBindGroups(): void {
		for (const cached of this._bindGroupCache.values()) {
			this._destroyBindingGroup(cached.group);
		}
		this._bindGroupCache.clear();
	}

	private _destroyBindingGroup(group: IBindingGroup | null): void {
		const destroyFn = (group as { destroy?: () => void } | null)?.destroy;
		if (typeof destroyFn === "function") {
			destroyFn.call(group);
		}
	}

	private async _warmupHint(hint: string): Promise<boolean> {
		switch (hint) {
			case "postprocess:ssao":
				await this._ensureSSAOResources();
				return true;
			case "postprocess:taa":
				await this._ensureTAAResources();
				return true;
			case "postprocess:hiz":
				await this._ensureHiZResources();
				return true;
			case "postprocess:ssr":
				await this._ensureSSRResources();
				return true;
			case "postprocess:volumetric":
				await this._ensureVolumetricResources();
				return true;
			case "postprocess:fxaa":
				await this._ensureFXAAResources();
				return true;
			case "postprocess:copy":
				await this._ensureCopyResources();
				return true;
			default:
				return false;
		}
	}
	public async executeSSAO(
		encoder: ICommandEncoder,
		targets: WebGPUFrameTargets,
		frameContext: FrameContext
	): Promise<void> {
		await this._ensureSSAOResources();
		if (
			!this._sampler ||
			!this._ssaoRawPipeline ||
			!this._ssaoBlurPipeline ||
			!this._ssaoCombinePipeline ||
			!this._ssaoParams
		) {
			return;
		}
		const options = frameContext.features.ssaoOptions ?? {};
		const ssaoParams = this._ssaoParams;
		const radius = Math.max(
			1,
			finiteOr(options.radius, DEFAULT_SSAO_OPTIONS.radius)
		);
		const bias = Math.max(
			1e-4,
			finiteOr(options.bias, DEFAULT_SSAO_OPTIONS.bias)
		);
		const intensity = Math.max(
			0,
			finiteOr(options.intensity, DEFAULT_SSAO_OPTIONS.intensity)
		);
		const blurRadius = clamp(
			finiteOr(options.blurRadius, DEFAULT_SSAO_OPTIONS.blurRadius),
			1,
			4
		);
		const blurSharpness = Math.max(
			1e-3,
			finiteOr(options.blurSharpness, DEFAULT_SSAO_OPTIONS.blurSharpness)
		);
		const samples = clamp(
			Math.round(finiteOr(options.samples, DEFAULT_SSAO_OPTIONS.samples)),
			4,
			48
		);
		const isOrthographic = frameContext.camera.type === CameraType.Orthographic;
		const tanHalfFov =
			isOrthographic ? 0 : Math.tan((frameContext.camera.fov * Math.PI) / 360);
		const aspect =
			frameContext.camera.aspectRatio ||
			Math.max(targets.sceneColor.width, 1) /
				Math.max(targets.sceneColor.height, 1);
		const fullInvW = 1 / Math.max(targets.sceneColor.width, 1);
		const fullInvH = 1 / Math.max(targets.sceneColor.height, 1);
		const aoInvW = 1 / Math.max(targets.aoRaw.width, 1);
		const aoInvH = 1 / Math.max(targets.aoRaw.height, 1);
		this._ssaoFrameIndex = (this._ssaoFrameIndex + 1) % 1024;
		const frameJitter = this._ssaoFrameIndex / 1024;

		const writeSSAOParams = (blurDirX: number, blurDirY: number): void => {
			this._backend.writeBuffer(
				ssaoParams,
				new Float32Array([
					fullInvW,
					fullInvH,
					aoInvW,
					aoInvH,
					radius,
					bias,
					intensity,
					samples,
					blurRadius,
					blurSharpness,
					tanHalfFov,
					aspect,
					blurDirX,
					blurDirY,
					isOrthographic ? 1 : 0,
					frameJitter,
				])
			);
		};
		writeSSAOParams(1, 0);
		let binding = this._getCachedBindGroup(
			"ssao-raw",
			this._ssaoRawPipeline,
			[
				{ binding: 0, resource: targets.gNormalRoughMetal },
				{ binding: 1, resource: targets.gMotionDepth },
				{ binding: 2, resource: this._sampler },
				{ binding: 3, resource: ssaoParams },
				{ binding: 4, resource: targets.aoRaw },
			],
			"WebGPUSSAO_RawBinding"
		);
		encoder.beginComputePass({ label: "WebGPUSSAO_Raw" });
		encoder.setComputePipeline(this._ssaoRawPipeline);
		encoder.setBindingGroup(0, binding);
		encoder.dispatchWorkgroups(
			ceilDiv(targets.aoRaw.width, WORKGROUP_SIZE),
			ceilDiv(targets.aoRaw.height, WORKGROUP_SIZE),
			1
		);
		encoder.endComputePass();

		binding = this._getCachedBindGroup(
			"ssao-blur-h",
			this._ssaoBlurPipeline,
			[
				{ binding: 0, resource: targets.aoRaw },
				{ binding: 1, resource: targets.gMotionDepth },
				{ binding: 2, resource: this._sampler },
				{ binding: 3, resource: ssaoParams },
				{ binding: 4, resource: targets.aoBlur },
			],
			"WebGPUSSAO_BlurBinding"
		);
		encoder.beginComputePass({ label: "WebGPUSSAO_Blur" });
		encoder.setComputePipeline(this._ssaoBlurPipeline);
		encoder.setBindingGroup(0, binding);
		encoder.dispatchWorkgroups(
			ceilDiv(targets.aoBlur.width, WORKGROUP_SIZE),
			ceilDiv(targets.aoBlur.height, WORKGROUP_SIZE),
			1
		);
		encoder.endComputePass();
		writeSSAOParams(0, 1);
		binding = this._getCachedBindGroup(
			"ssao-blur-v",
			this._ssaoBlurPipeline,
			[
				{ binding: 0, resource: targets.aoBlur },
				{ binding: 1, resource: targets.gMotionDepth },
				{ binding: 2, resource: this._sampler },
				{ binding: 3, resource: ssaoParams },
				{ binding: 4, resource: targets.aoRaw },
			],
			"WebGPUSSAO_BlurBindingVertical"
		);
		encoder.beginComputePass({ label: "WebGPUSSAO_BlurVertical" });
		encoder.setComputePipeline(this._ssaoBlurPipeline);
		encoder.setBindingGroup(0, binding);
		encoder.dispatchWorkgroups(
			ceilDiv(targets.aoRaw.width, WORKGROUP_SIZE),
			ceilDiv(targets.aoRaw.height, WORKGROUP_SIZE),
			1
		);
		encoder.endComputePass();

		const combineTarget =
			targets.sceneColor === targets.postPing ?
				targets.postPong
			:	targets.postPing;
		binding = this._getCachedBindGroup(
			`ssao-combine-${combineTarget === targets.postPing ? "ping" : "pong"}`,
			this._ssaoCombinePipeline,
			[
				{ binding: 0, resource: targets.sceneColor },
				{ binding: 1, resource: targets.aoRaw },
				{ binding: 2, resource: this._sampler },
				{ binding: 3, resource: ssaoParams },
				{ binding: 4, resource: combineTarget },
			],
			"WebGPUSSAO_CombineBinding"
		);
		encoder.beginComputePass({ label: "WebGPUSSAO_Combine" });
		encoder.setComputePipeline(this._ssaoCombinePipeline);
		encoder.setBindingGroup(0, binding);
		encoder.dispatchWorkgroups(
			ceilDiv(combineTarget.width, WORKGROUP_SIZE),
			ceilDiv(combineTarget.height, WORKGROUP_SIZE),
			1
		);
		encoder.endComputePass();
		targets.sceneColor = combineTarget;
	}

	public async executeTAA(
		encoder: ICommandEncoder,
		targets: WebGPUFrameTargets,
		frameContext: FrameContext,
		historyValid: boolean
	): Promise<boolean> {
		await this._ensureTAAResources();
		if (!this._sampler || !this._taaPipeline || !this._taaParams) return false;
		const options = frameContext.features.taaOptions ?? {};
		const taaTarget =
			targets.sceneColor === targets.postPong ?
				targets.postPing
			:	targets.postPong;
		const invW = 1 / Math.max(taaTarget.width, 1);
		const invH = 1 / Math.max(taaTarget.height, 1);
		this._backend.writeBuffer(
			this._taaParams,
			new Float32Array([
				invW,
				invH,
				finiteOr(options.historyWeight, DEFAULT_TAA_OPTIONS.historyWeight),
				finiteOr(
					options.disocclusionDepthThreshold,
					DEFAULT_TAA_OPTIONS.disocclusionDepthThreshold
				),
				finiteOr(options.motionFactor, DEFAULT_TAA_OPTIONS.motionFactor),
				finiteOr(
					options.varianceClampGamma,
					DEFAULT_TAA_OPTIONS.varianceClampGamma
				),
				finiteOr(options.sharpen, DEFAULT_TAA_OPTIONS.sharpen),
				historyValid ? 1 : 0,
			])
		);
		const binding = this._getCachedBindGroup(
			`taa-${taaTarget === targets.postPing ? "ping" : "pong"}`,
			this._taaPipeline,
			[
				{ binding: 0, resource: targets.sceneColor },
				{ binding: 1, resource: targets.historyRead },
				{ binding: 2, resource: targets.gMotionDepth },
				{ binding: 3, resource: targets.motionHistoryRead },
				{ binding: 4, resource: this._sampler },
				{ binding: 5, resource: this._taaParams },
				{ binding: 6, resource: taaTarget },
				{ binding: 7, resource: targets.historyWrite },
			],
			"WebGPUTAA_Binding"
		);
		encoder.beginComputePass({ label: "WebGPUTAA" });
		encoder.setComputePipeline(this._taaPipeline);
		encoder.setBindingGroup(0, binding);
		encoder.dispatchWorkgroups(
			ceilDiv(taaTarget.width, WORKGROUP_SIZE),
			ceilDiv(taaTarget.height, WORKGROUP_SIZE),
			1
		);
		encoder.endComputePass();
		targets.sceneColor = taaTarget;
		return true;
	}

	public async executeSSR(
		encoder: ICommandEncoder,
		targets: WebGPUFrameTargets,
		frameContext: FrameContext,
		historyValid: boolean,
		frameBinding: IBindingGroup
	): Promise<boolean> {
		if (frameContext.camera.type === CameraType.Orthographic) {
			this._warn(
				"webgpu-ssr-orthographic-disabled",
				"WebGPU SSR is disabled for OrthographicCamera in v1"
			);
			return false;
		}
		await this._ensureSSRResources();
		if (
			!this._sampler ||
			!this._hizInitPipeline ||
			!this._hizReducePipeline ||
			!this._ssrTracePipeline ||
			!this._ssrComposePipeline ||
			!this._ssrTraceParams ||
			!this._ssrComposeParams
		)
			return false;
		const options = frameContext.features.ssrOptions ?? {};
		const hiZMips = this._getHiZMipViews(targets.hiZ);
		if (!this._buildHiZ(encoder, targets, hiZMips)) return false;
		this._ssrFrameIndex = (this._ssrFrameIndex + 1) % 1024;
		this._backend.writeBuffer(
			this._ssrTraceParams,
			new Float32Array([
				1 / Math.max(targets.ssrRaw.width, 1),
				1 / Math.max(targets.ssrRaw.height, 1),
				finiteOr(options.maxDistance, DEFAULT_SSR_OPTIONS.maxDistance),
				finiteOr(options.thickness, DEFAULT_SSR_OPTIONS.thickness),
				finiteOr(options.stride, DEFAULT_SSR_OPTIONS.stride),
				finiteOr(options.intensity, DEFAULT_SSR_OPTIONS.intensity),
				finiteOr(options.maxRoughness, DEFAULT_SSR_OPTIONS.maxRoughness),
				finiteOr(options.edgeFade, DEFAULT_SSR_OPTIONS.edgeFade),
				finiteOr(options.maxSteps, DEFAULT_SSR_OPTIONS.maxSteps),
				finiteOr(
					options.binarySearchSteps,
					DEFAULT_SSR_OPTIONS.binarySearchSteps
				),
				hiZMips.length - 1,
				finiteOr(options.historyWeight, DEFAULT_SSR_OPTIONS.historyWeight),
				historyValid ? 1 : 0,
				0.02,
				this._ssrFrameIndex, // frameIndex for blue-noise jitter
				0,
			])
		);
		let binding = this._getCachedBindGroup(
			"ssr-trace",
			this._ssrTracePipeline,
			[
				{ binding: 0, resource: targets.sceneColor },
				{ binding: 1, resource: targets.gNormalRoughMetal },
				{ binding: 2, resource: targets.gMotionDepth },
				{ binding: 3, resource: targets.hiZ },
				{ binding: 4, resource: targets.ssrHistoryRead },
				{ binding: 5, resource: targets.motionHistoryRead },
				{ binding: 6, resource: this._sampler },
				{ binding: 7, resource: this._ssrTraceParams },
				{ binding: 8, resource: targets.ssrRaw },
			],
			"WebGPUSSR_TraceBinding"
		);
		encoder.beginComputePass({ label: "WebGPUSSR_TraceTemporal" });
		encoder.setComputePipeline(this._ssrTracePipeline);
		encoder.setBindingGroup(0, binding);
		encoder.setBindingGroup(1, frameBinding);
		encoder.dispatchWorkgroups(
			ceilDiv(targets.ssrRaw.width, WORKGROUP_SIZE),
			ceilDiv(targets.ssrRaw.height, WORKGROUP_SIZE),
			1
		);
		encoder.endComputePass();
		await this._copyTexture(encoder, targets.ssrRaw, targets.ssrHistoryWrite);
		const composeTarget =
			targets.sceneColor === targets.postPing ?
				targets.postPong
			:	targets.postPing;
		this._backend.writeBuffer(
			this._ssrComposeParams,
			new Float32Array([
				1 / Math.max(composeTarget.width, 1),
				1 / Math.max(composeTarget.height, 1),
				0,
				0,
			])
		);
		binding = this._getCachedBindGroup(
			`ssr-compose-${composeTarget === targets.postPing ? "ping" : "pong"}`,
			this._ssrComposePipeline,
			[
				{ binding: 0, resource: targets.sceneColor },
				{ binding: 1, resource: targets.ssrRaw },
				{ binding: 2, resource: targets.gMotionDepth },
				{ binding: 3, resource: this._sampler },
				{ binding: 4, resource: this._ssrComposeParams },
				{ binding: 5, resource: composeTarget },
			],
			"WebGPUSSR_ComposeBinding"
		);
		encoder.beginComputePass({ label: "WebGPUSSR_Compose" });
		encoder.setComputePipeline(this._ssrComposePipeline);
		encoder.setBindingGroup(0, binding);
		encoder.dispatchWorkgroups(
			ceilDiv(composeTarget.width, WORKGROUP_SIZE),
			ceilDiv(composeTarget.height, WORKGROUP_SIZE),
			1
		);
		encoder.endComputePass();
		targets.sceneColor = composeTarget;
		return true;
	}

	public async executeVolumetric(
		encoder: ICommandEncoder,
		targets: WebGPUFrameTargets,
		frameContext: FrameContext,
		historyValid: boolean,
		frameBinding: IBindingGroup,
		lightingState: WebGPULightingState | null
	): Promise<boolean> {
		if (frameContext.camera.type === CameraType.Orthographic) {
			this._warn(
				"webgpu-volumetric-orthographic-disabled",
				"WebGPU volumetric lighting is disabled for OrthographicCamera in v1"
			);
			return false;
		}
		await this._ensureVolumetricResources();
		const lightCount = this._updateVolumetricLightBuffer(lightingState);
		if (
			!this._sampler ||
			!this._volumetricPipeline ||
			!this._volumetricParams ||
			!this._volumetricLightBuffer
		) {
			return false;
		}
		const hiZMips = this._getHiZMipViews(targets.hiZ);
		if (!this._buildHiZ(encoder, targets, hiZMips)) return false;

		const options = frameContext.features.volumetricOptions ?? {};
		const samples = Math.max(
			1,
			Math.min(
				128,
				finiteOr(options.samples, DEFAULT_VOLUMETRIC_OPTIONS.samples)
			)
		);
		const weight = Math.max(
			0,
			finiteOr(options.weight, DEFAULT_VOLUMETRIC_OPTIONS.weight)
		);
		const exposure = Math.max(
			0,
			finiteOr(options.exposure, DEFAULT_VOLUMETRIC_OPTIONS.exposure)
		);
		const airDensity = Math.max(
			0.001,
			finiteOr(options.airDensity, DEFAULT_VOLUMETRIC_OPTIONS.airDensity)
		);
		const anisotropy = Math.max(
			-0.95,
			Math.min(
				0.95,
				finiteOr(options.anisotropy, DEFAULT_VOLUMETRIC_OPTIONS.anisotropy)
			)
		);
		const maxRayDistance = Math.max(
			0.1,
			finiteOr(
				options.maxRayDistance,
				DEFAULT_VOLUMETRIC_OPTIONS.maxRayDistance
			)
		);
		const scatteringAlbedo = Math.max(
			0,
			Math.min(
				1,
				finiteOr(
					options.scatteringAlbedo,
					DEFAULT_VOLUMETRIC_OPTIONS.scatteringAlbedo
				)
			)
		);
		const shadowSampleInterval = Math.max(
			1,
			Math.min(
				32,
				finiteOr(
					options.shadowSampleInterval,
					DEFAULT_VOLUMETRIC_OPTIONS.shadowSampleInterval
				)
			)
		);
		const adaptiveSteps = options.adaptiveSteps === false ? 0 : 1;
		const depthThickness = Math.max(
			0.01,
			finiteOr(
				options.bilateralDepthSigma,
				DEFAULT_VOLUMETRIC_OPTIONS.bilateralDepthSigma
			) * 8
		);
		const maxMip = Math.max(0, hiZMips.length - 1);
		const restirCandidates = Math.max(
			1,
			Math.min(
				64,
				finiteOr(
					options["restirCandidates"],
					DEFAULT_VOLUMETRIC_OPTIONS.restirCandidates
				)
			)
		);
		const restirTemporalWeight = Math.max(
			0,
			Math.min(
				1,
				finiteOr(
					options["restirTemporalWeight"],
					DEFAULT_VOLUMETRIC_OPTIONS.restirTemporalWeight
				)
			)
		);
		const restirScaleClamp = Math.max(
			1,
			finiteOr(
				options["restirScaleClamp"],
				DEFAULT_VOLUMETRIC_OPTIONS.restirScaleClamp
			)
		);
		this._volumetricFrameIndex = (this._volumetricFrameIndex + 1) % 4096;

		this._backend.writeBuffer(
			this._volumetricParams,
			new Float32Array([
				1 / Math.max(targets.sceneColor.width, 1),
				1 / Math.max(targets.sceneColor.height, 1),
				samples,
				weight,
				exposure,
				airDensity,
				anisotropy,
				maxRayDistance,
				scatteringAlbedo,
				shadowSampleInterval,
				adaptiveSteps,
				depthThickness,
				maxMip,
				0.75,
				historyValid ? 1 : 0,
				lightCount,
				restirCandidates,
				restirTemporalWeight,
				restirScaleClamp,
				this._volumetricFrameIndex,
			])
		);

		const target =
			targets.sceneColor === targets.postPong ?
				targets.postPing
			:	targets.postPong;
		const binding = this._getCachedBindGroup(
			`volumetric-${target === targets.postPing ? "ping" : "pong"}`,
			this._volumetricPipeline,
			[
				{ binding: 0, resource: targets.sceneColor },
				{ binding: 1, resource: targets.gMotionDepth },
				{ binding: 2, resource: targets.hiZ },
				{ binding: 3, resource: targets.volumetricHistoryRead },
				{ binding: 4, resource: targets.motionHistoryRead },
				{ binding: 5, resource: this._sampler },
				{ binding: 6, resource: this._volumetricParams },
				{ binding: 7, resource: target },
				{ binding: 8, resource: targets.volumetricHistoryWrite },
				{
					binding: 9,
					resource: targets.volumetricReservoirHistoryRead,
				},
				{
					binding: 10,
					resource: targets.volumetricReservoirHistoryWrite,
				},
				{ binding: 11, resource: this._volumetricLightBuffer },
			],
			"WebGPUVolumetric_Binding"
		);
		encoder.beginComputePass({ label: "WebGPUVolumetric" });
		encoder.setComputePipeline(this._volumetricPipeline);
		encoder.setBindingGroup(0, binding);
		encoder.setBindingGroup(1, frameBinding);
		encoder.dispatchWorkgroups(
			ceilDiv(target.width, WORKGROUP_SIZE),
			ceilDiv(target.height, WORKGROUP_SIZE),
			1
		);
		encoder.endComputePass();
		targets.sceneColor = target;
		return true;
	}

	public async executeFXAA(
		encoder: ICommandEncoder,
		targets: WebGPUFrameTargets
	): Promise<void> {
		await this._ensureFXAAResources();
		if (!this._sampler || !this._fxaaPipeline || !this._fxaaParams) return;
		const target =
			targets.sceneColor === targets.postPong ?
				targets.postPing
			:	targets.postPong;
		this._backend.writeBuffer(
			this._fxaaParams,
			new Float32Array([
				1 / Math.max(target.width, 1),
				1 / Math.max(target.height, 1),
				DEFAULT_FXAA.edgeThresholdMin,
				DEFAULT_FXAA.edgeThresholdMultiplier,
				DEFAULT_FXAA.subpixQuality,
				0,
			])
		);
		const binding = this._getCachedBindGroup(
			`fxaa-${target === targets.postPing ? "ping" : "pong"}`,
			this._fxaaPipeline,
			[
				{ binding: 0, resource: targets.sceneColor },
				{ binding: 1, resource: this._sampler },
				{ binding: 2, resource: this._fxaaParams },
				{ binding: 3, resource: target },
			],
			"WebGPUFXAA_Binding"
		);
		encoder.beginComputePass({ label: "WebGPUFXAA" });
		encoder.setComputePipeline(this._fxaaPipeline);
		encoder.setBindingGroup(0, binding);
		encoder.dispatchWorkgroups(
			ceilDiv(target.width, WORKGROUP_SIZE),
			ceilDiv(target.height, WORKGROUP_SIZE),
			1
		);
		encoder.endComputePass();
		targets.sceneColor = target;
	}

	private _updateVolumetricLightBuffer(
		lightingState: WebGPULightingState | null
	): number {
		const sourceLights = lightingState?.volumetricLights ?? [];
		const clampedLightCount = Math.min(
			sourceLights.length,
			MAX_VOLUMETRIC_LIGHTS
		);
		if (sourceLights.length > MAX_VOLUMETRIC_LIGHTS) {
			this._warn(
				"webgpu-volumetric-light-count-clamped",
				`WebGPU volumetric ReSTIR clamps light count to ${MAX_VOLUMETRIC_LIGHTS}; extra lights are skipped`
			);
		}

		this._ensureVolumetricLightBufferCapacity(clampedLightCount);
		if (!this._volumetricLightBuffer) return 0;

		const packedCount = Math.max(1, clampedLightCount);
		const packed = new Float32Array(
			packedCount * VOLUMETRIC_LIGHT_STRIDE_FLOATS
		);

		for (let i = 0; i < clampedLightCount; i++) {
			const light = sourceLights[i];
			const base = i * VOLUMETRIC_LIGHT_STRIDE_FLOATS;
			const isDirectional = light.type === 0;
			const isSpot = light.type === 2;

			packed[base] = light.position[0];
			packed[base + 1] = light.position[1];
			packed[base + 2] = light.position[2];
			packed[base + 3] = isDirectional ? -1 : Math.max(light.range, 0.001);
			packed[base + 4] = light.direction[0];
			packed[base + 5] = light.direction[1];
			packed[base + 6] = light.direction[2];
			packed[base + 7] = isSpot ? light.outerCos : -2;
			packed[base + 8] = light.color[0];
			packed[base + 9] = light.color[1];
			packed[base + 10] = light.color[2];
			packed[base + 11] = isSpot ? light.innerCos : -2;
		}

		if (clampedLightCount === 0) {
			packed[3] = -1;
		}

		this._backend.writeBuffer(this._volumetricLightBuffer, packed);
		return clampedLightCount;
	}

	private _ensureVolumetricLightBufferCapacity(lightCount: number): void {
		const required = Math.max(1, lightCount);
		if (
			this._volumetricLightBuffer &&
			this._volumetricLightCapacity >= required
		) {
			return;
		}

		let capacity = Math.max(1, this._volumetricLightCapacity);
		while (capacity < required) {
			capacity *= 2;
		}

		this._volumetricLightBuffer?.destroy();
		this._volumetricLightBuffer = this._backend.createBuffer({
			label: "WebGPUVolumetricLights",
			size: capacity * VOLUMETRIC_LIGHT_STRIDE_FLOATS * 4,
			usage: BufferUsage.Storage | BufferUsage.CopyDst,
		});
		this._volumetricLightCapacity = capacity;
	}

	private async _copyTexture(
		encoder: ICommandEncoder,
		src: IRenderTexture,
		dst: IRenderTexture
	): Promise<void> {
		if (src === dst) return;
		await this._ensureCopyResources();
		if (!this._copyPipeline) return;
		const binding = this._getCachedBindGroup(
			`copy-${src === dst ? "same" : "diff"}`,
			this._copyPipeline,
			[
				{ binding: 0, resource: src },
				{ binding: 1, resource: dst },
			],
			"WebGPUPost_CopyBinding"
		);
		encoder.beginComputePass({ label: "WebGPUPost_Copy" });
		encoder.setComputePipeline(this._copyPipeline);
		encoder.setBindingGroup(0, binding);
		encoder.dispatchWorkgroups(
			ceilDiv(dst.width, WORKGROUP_SIZE),
			ceilDiv(dst.height, WORKGROUP_SIZE),
			1
		);
		encoder.endComputePass();
	}

	private _buildHiZ(
		encoder: ICommandEncoder,
		targets: WebGPUFrameTargets,
		hiZMips: GPUTextureView[]
	): boolean {
		if (!this._hizInitPipeline || !this._hizReducePipeline) return false;
		if (hiZMips.length === 0) return false;

		let binding = this._getCachedBindGroup(
			"hiz-init",
			this._hizInitPipeline,
			[
				{ binding: 0, resource: targets.gMotionDepth },
				{ binding: 1, resource: hiZMips[0] },
			],
			"WebGPUSSR_HiZInitBinding"
		);
		encoder.beginComputePass({ label: "WebGPUSSR_HiZInit" });
		encoder.setComputePipeline(this._hizInitPipeline);
		encoder.setBindingGroup(0, binding);
		encoder.dispatchWorkgroups(
			ceilDiv(targets.hiZ.width, WORKGROUP_SIZE),
			ceilDiv(targets.hiZ.height, WORKGROUP_SIZE),
			1
		);
		encoder.endComputePass();

		let srcW = targets.hiZ.width;
		let srcH = targets.hiZ.height;
		for (let mip = 1; mip < hiZMips.length; mip++) {
			const dstW = Math.max(1, srcW >> 1);
			const dstH = Math.max(1, srcH >> 1);
			binding = this._getCachedBindGroup(
				`hiz-reduce-${mip}`,
				this._hizReducePipeline,
				[
					{ binding: 0, resource: hiZMips[mip - 1] },
					{ binding: 1, resource: hiZMips[mip] },
				],
				`WebGPUSSR_HiZReduceBinding_${mip}`
			);
			encoder.beginComputePass({ label: `WebGPUSSR_HiZReduce_${mip}` });
			encoder.setComputePipeline(this._hizReducePipeline);
			encoder.setBindingGroup(0, binding);
			encoder.dispatchWorkgroups(
				ceilDiv(dstW, WORKGROUP_SIZE),
				ceilDiv(dstH, WORKGROUP_SIZE),
				1
			);
			encoder.endComputePass();
			srcW = dstW;
			srcH = dstH;
		}
		return true;
	}

	private async _ensureCommonResources(): Promise<void> {
		if (this._sampler) return;
		this._sampler = this._backend.createSampler({
			label: "WebGPUPost_LinearSampler",
			magFilter: FilterMode.Linear,
			minFilter: FilterMode.Linear,
			mipmapFilter: FilterMode.Linear,
			addressModeU: AddressMode.ClampToEdge,
			addressModeV: AddressMode.ClampToEdge,
		});
	}

	private async _ensureSSAOResources(): Promise<void> {
		await this._ensureCommonResources();
		if (!this._ssaoModule) {
			const shader = await loadPostProcessShaderPartComposite("ssao");
			this._ssaoModule = await this._backend.createShaderModule({
				label: "WebGPUSSAOShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!this._ssaoRawPipeline)
			this._ssaoRawPipeline = this._backend.createComputePipeline({
				label: "WebGPUSSAORawPipeline",
				compute: { module: this._ssaoModule, entryPoint: "csRaw" },
			});
		if (!this._ssaoBlurPipeline)
			this._ssaoBlurPipeline = this._backend.createComputePipeline({
				label: "WebGPUSSAOBlurPipeline",
				compute: { module: this._ssaoModule, entryPoint: "csBlur" },
			});
		if (!this._ssaoCombinePipeline)
			this._ssaoCombinePipeline = this._backend.createComputePipeline({
				label: "WebGPUSSAOCombinePipeline",
				compute: { module: this._ssaoModule, entryPoint: "csCombine" },
			});
		if (!this._ssaoParams)
			this._ssaoParams = this._backend.createBuffer({
				label: "WebGPUSSAOParams",
				size: 16 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
	}

	private async _ensureTAAResources(): Promise<void> {
		await this._ensureCommonResources();
		if (!this._taaModule) {
			const shader = await loadPostProcessShaderPartComposite("taa");
			this._taaModule = await this._backend.createShaderModule({
				label: "WebGPUTAAShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!this._taaPipeline)
			this._taaPipeline = this._backend.createComputePipeline({
				label: "WebGPUTAAPipeline",
				compute: { module: this._taaModule, entryPoint: "csMain" },
			});
		if (!this._taaParams)
			this._taaParams = this._backend.createBuffer({
				label: "WebGPUTAAParams",
				size: 8 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
	}

	private async _ensureHiZResources(): Promise<void> {
		await this._ensureCommonResources();
		if (!this._hizModule) {
			const shader = await loadPostProcessShaderPartComposite("hiz");
			this._hizModule = await this._backend.createShaderModule({
				label: "WebGPUHiZShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!this._hizInitPipeline)
			this._hizInitPipeline = this._backend.createComputePipeline({
				label: "WebGPUHiZInitPipeline",
				compute: { module: this._hizModule, entryPoint: "csInit" },
			});
		if (!this._hizReducePipeline)
			this._hizReducePipeline = this._backend.createComputePipeline({
				label: "WebGPUHiZReducePipeline",
				compute: { module: this._hizModule, entryPoint: "csReduce" },
			});
	}

	private async _ensureSSRResources(): Promise<void> {
		await this._ensureHiZResources();
		if (!this._ssrModule) {
			const shader = await loadPostProcessShaderPartComposite("ssr");
			this._ssrModule = await this._backend.createShaderModule({
				label: "WebGPUSSRShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!this._ssrTracePipeline) {
			const device = this._backend.device;
			if (device && this._frameBindGroupLayout) {
				this._ssrTraceGroupLayout0 = device.createBindGroupLayout({
					label: "WebGPUSSRTrace_GroupLayout0",
					entries: [
						{ binding: 0, visibility: GPUShaderStage.COMPUTE, texture: {} },
						{
							binding: 1,
							visibility: GPUShaderStage.COMPUTE,
							texture: {},
						},
						{
							binding: 2,
							visibility: GPUShaderStage.COMPUTE,
							texture: {},
						},
						{ binding: 3, visibility: GPUShaderStage.COMPUTE, texture: {} },
						{
							binding: 4,
							visibility: GPUShaderStage.COMPUTE,
							texture: {},
						},
						{
							binding: 5,
							visibility: GPUShaderStage.COMPUTE,
							texture: {},
						},
						{
							binding: 6,
							visibility: GPUShaderStage.COMPUTE,
							sampler: {},
						},
						{
							binding: 7,
							visibility: GPUShaderStage.COMPUTE,
							buffer: { type: "uniform" },
						},
						{
							binding: 8,
							visibility: GPUShaderStage.COMPUTE,
							storageTexture: {
								format: "rgba16float",
								access: "write-only",
							},
						},
					],
				});
				this._ssrTracePipelineLayout = device.createPipelineLayout({
					label: "WebGPUSSRTrace_PipelineLayout",
					bindGroupLayouts: [
						this._ssrTraceGroupLayout0,
						this._frameBindGroupLayout,
					],
				});
				this._ssrTracePipeline = this._backend.createComputePipeline({
					label: "WebGPUSSRTracePipeline",
					layout: this._ssrTracePipelineLayout,
					compute: { module: this._ssrModule!, entryPoint: "csTrace" },
				});
			} else {
				this._ssrTracePipeline = this._backend.createComputePipeline({
					label: "WebGPUSSRTracePipeline",
					compute: { module: this._ssrModule!, entryPoint: "csTrace" },
				});
			}
		}
		if (!this._ssrComposePipeline)
			this._ssrComposePipeline = this._backend.createComputePipeline({
				label: "WebGPUSSRComposePipeline",
				compute: { module: this._ssrModule, entryPoint: "csCompose" },
			});
		if (!this._ssrTraceParams)
			this._ssrTraceParams = this._backend.createBuffer({
				label: "WebGPUSSRTraceParams",
				size: 16 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
		if (!this._ssrComposeParams)
			this._ssrComposeParams = this._backend.createBuffer({
				label: "WebGPUSSRComposeParams",
				size: 4 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
	}

	private async _ensureVolumetricResources(): Promise<void> {
		await this._ensureHiZResources();
		if (!this._volumetricModule) {
			const shader = await loadPostProcessShaderPartComposite("volumetric");
			this._volumetricModule = await this._backend.createShaderModule({
				label: "WebGPUVolumetricShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!this._volumetricPipeline) {
			const device = this._backend.device;
			if (device && this._frameBindGroupLayout) {
				this._volumetricGroupLayout0 = device.createBindGroupLayout({
					label: "WebGPUVolumetric_GroupLayout0",
					entries: [
						{ binding: 0, visibility: GPUShaderStage.COMPUTE, texture: {} },
						{ binding: 1, visibility: GPUShaderStage.COMPUTE, texture: {} },
						{ binding: 2, visibility: GPUShaderStage.COMPUTE, texture: {} },
						{ binding: 3, visibility: GPUShaderStage.COMPUTE, texture: {} },
						{ binding: 4, visibility: GPUShaderStage.COMPUTE, texture: {} },
						{
							binding: 5,
							visibility: GPUShaderStage.COMPUTE,
							sampler: {},
						},
						{
							binding: 6,
							visibility: GPUShaderStage.COMPUTE,
							buffer: { type: "uniform" },
						},
						{
							binding: 7,
							visibility: GPUShaderStage.COMPUTE,
							storageTexture: {
								format: "rgba16float",
								access: "write-only",
							},
						},
						{
							binding: 8,
							visibility: GPUShaderStage.COMPUTE,
							storageTexture: {
								format: "rgba16float",
								access: "write-only",
							},
						},
						{ binding: 9, visibility: GPUShaderStage.COMPUTE, texture: {} },
						{
							binding: 10,
							visibility: GPUShaderStage.COMPUTE,
							storageTexture: {
								format: "rgba16float",
								access: "write-only",
							},
						},
						{
							binding: 11,
							visibility: GPUShaderStage.COMPUTE,
							buffer: { type: "read-only-storage" },
						},
					],
				});
				this._volumetricPipelineLayout = device.createPipelineLayout({
					label: "WebGPUVolumetric_PipelineLayout",
					bindGroupLayouts: [
						this._volumetricGroupLayout0,
						this._frameBindGroupLayout,
					],
				});
				this._volumetricPipeline = this._backend.createComputePipeline({
					label: "WebGPUVolumetricPipeline",
					layout: this._volumetricPipelineLayout,
					compute: { module: this._volumetricModule, entryPoint: "csMain" },
				});
			} else {
				this._volumetricPipeline = this._backend.createComputePipeline({
					label: "WebGPUVolumetricPipeline",
					compute: { module: this._volumetricModule, entryPoint: "csMain" },
				});
			}
		}
		if (!this._volumetricParams)
			this._volumetricParams = this._backend.createBuffer({
				label: "WebGPUVolumetricParams",
				size: 20 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
	}

	private async _ensureFXAAResources(): Promise<void> {
		await this._ensureCommonResources();
		if (!this._fxaaModule) {
			const shader = await loadPostProcessShaderPartComposite("fxaa");
			this._fxaaModule = await this._backend.createShaderModule({
				label: "WebGPUFXAAShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!this._fxaaPipeline)
			this._fxaaPipeline = this._backend.createComputePipeline({
				label: "WebGPUFXAAPipeline",
				compute: { module: this._fxaaModule, entryPoint: "csMain" },
			});
		if (!this._fxaaParams)
			this._fxaaParams = this._backend.createBuffer({
				label: "WebGPUFXAAParams",
				size: 6 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
	}

	private async _ensureCopyResources(): Promise<void> {
		if (!this._copyModule) {
			const shader = await loadPostProcessShaderPartComposite("copy");
			this._copyModule = await this._backend.createShaderModule({
				label: "WebGPUCopyShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!this._copyPipeline)
			this._copyPipeline = this._backend.createComputePipeline({
				label: "WebGPUCopyPipeline",
				compute: { module: this._copyModule, entryPoint: "csMain" },
			});
	}

	private _getHiZMipViews(texture: IRenderTexture): GPUTextureView[] {
		const cached = this._hizViewCache.get(texture as object);
		if (cached) return cached;
		const gpuTexture = getWebGPUTexture(texture).texture;
		if (!gpuTexture?.createView) return [];
		const mipCount =
			Math.floor(Math.log2(Math.max(texture.width, texture.height))) + 1;
		const views: GPUTextureView[] = [];
		for (let i = 0; i < mipCount; i++) {
			views.push(gpuTexture.createView({ baseMipLevel: i, mipLevelCount: 1 }));
		}
		this._hizViewCache.set(texture as object, views);
		return views;
	}
}

function finiteOr(value: unknown, fallback: number): number {
	return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}


function ceilDiv(value: number, divisor: number): number {
	return Math.max(1, Math.ceil(value / Math.max(divisor, 1)));
}
