import { CameraType } from "../../cameras/Camera";
import type { FrameContext } from "../../pipeline/types";
import {
	DEFAULT_BLOOM_OPTIONS,
	DEFAULT_DOF_OPTIONS,
	INTERACTION_TRANSIENT_STATE_KEY,
	DEFAULT_MOTION_BLUR_OPTIONS,
	DEFAULT_SSAO_OPTIONS,
	DEFAULT_SSGI_OPTIONS,
	DEFAULT_SSR_OPTIONS,
	DEFAULT_TAA_OPTIONS,
	DEFAULT_VOLUMETRIC_OPTIONS,
	type InteractionTransientState,
} from "../../pipeline/types";
import type { ICommandEncoder } from "../ICommandEncoder";
import {
	AddressMode,
	BufferUsage,
	FilterMode,
	TextureFormat,
	TextureUsage,
	type IComputePipeline,
	type IRenderBuffer,
	type IRenderTexture,
	type ISampler,
	type IShaderModule,
} from "../types";
import type { IWebGPUComputeFacade } from "./computeFacade";
import type { WebGPUFrameTargets } from "./WebGPUPostProcessGraph";
import { loadPostProcessShaderPartComposite } from "../../shaders/webgpu/shaderSource";
import type { IBindingGroup } from "../types";
import type { WebGPULightingState } from "./types";
import { clamp, sRGBToLinear } from "../../maths/Common";
import type { ShaderCompileError } from "../../shaders/runtime";
import { toShaderCompileError } from "../../pipeline/WarmupPlanner";
import { destroyResource } from "./computeUtils";
import {
	MAX_INTERACTION_OUTLINE_CIRCLES,
	collectProjectedOutlineCircles,
} from "../../interaction/outlineProjection";
import { getInteractionOutlineShapeCode } from "../../interaction/outlineShape";
import {
	FXAA_EDGE_THRESHOLD_MIN,
	FXAA_EDGE_THRESHOLD_MULTIPLIER,
	FXAA_SUBPIX_QUALITY,
} from "../postProcessConstants";

const WORKGROUP_SIZE = 8;
const INTERACTION_OUTLINE_HEADER_FLOATS = 16;
const INTERACTION_OUTLINE_PARAM_FLOATS =
	INTERACTION_OUTLINE_HEADER_FLOATS + MAX_INTERACTION_OUTLINE_CIRCLES * 4;

const VOLUMETRIC_LIGHT_STRIDE_FLOATS = 12;
const MAX_VOLUMETRIC_LIGHTS = 65000;

interface CachedBindGroup {
	group: IBindingGroup;
	resources: readonly unknown[];
}

export class WebGPUPostProcessRuntime {
	private _compute: IWebGPUComputeFacade;
	private _warn: (key: string, message: string) => void;
	private _sampler: ISampler | null = null;
	private _ssaoModule: IShaderModule | null = null;
	private _ssaoRawPipeline: IComputePipeline | null = null;
	private _ssaoBlurPipeline: IComputePipeline | null = null;
	private _ssaoCombinePipeline: IComputePipeline | null = null;
	private _ssaoParams: IRenderBuffer | null = null;
	private _ssgiModule: IShaderModule | null = null;
	private _ssgiPipeline: IComputePipeline | null = null;
	private _ssgiParams: IRenderBuffer | null = null;
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
	private _motionBlurModule: IShaderModule | null = null;
	private _motionBlurPipeline: IComputePipeline | null = null;
	private _motionBlurParams: IRenderBuffer | null = null;
	private _motionBlurParamData = new Float32Array(8);
	private _motionBlurParamUploaded = false;
	private _dofModule: IShaderModule | null = null;
	private _dofPipeline: IComputePipeline | null = null;
	private _dofParams: IRenderBuffer | null = null;
	private _bloomDownsampleModule: IShaderModule | null = null;
	private _bloomBlurHModule: IShaderModule | null = null;
	private _bloomBlurVModule: IShaderModule | null = null;
	private _bloomUpsampleModule: IShaderModule | null = null;
	private _bloomCompositeModule: IShaderModule | null = null;
	private _bloomDownsamplePipeline: IComputePipeline | null = null;
	private _bloomBlurHPipeline: IComputePipeline | null = null;
	private _bloomBlurVPipeline: IComputePipeline | null = null;
	private _bloomUpsamplePipeline: IComputePipeline | null = null;
	private _bloomCompositePipeline: IComputePipeline | null = null;
	private _bloomDownsampleParams: IRenderBuffer | null = null;
	private _bloomBlurParams: IRenderBuffer | null = null;
	private _bloomUpsampleParams: IRenderBuffer | null = null;
	private _bloomCompositeParams: IRenderBuffer | null = null;
	/**
	 * Cached bloom mip textures. Index 0 is the smallest mip (deepest level),
	 * index N-1 is half-res of the original scene. Two textures per level for
	 * ping-pong during separable blur. Re-allocated on size change.
	 */
	private _bloomMipTextures: Array<[IRenderTexture, IRenderTexture]> = [];
	private _bloomMipWidth = 0;
	private _bloomMipHeight = 0;
	private _bloomMipCount = 0;
	private _fxaaModule: IShaderModule | null = null;
	private _fxaaPipeline: IComputePipeline | null = null;
	private _fxaaParams: IRenderBuffer | null = null;
	private _interactionOutlineModule: IShaderModule | null = null;
	private _interactionOutlinePipeline: IComputePipeline | null = null;
	private _interactionOutlineParams: IRenderBuffer | null = null;
	private _interactionOutlineParamData = new Float32Array(
		INTERACTION_OUTLINE_PARAM_FLOATS
	);
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
		computeFacade: IWebGPUComputeFacade,
		warn: (key: string, message: string) => void,
		frameBindGroupLayout?: GPUBindGroupLayout
	) {
		this._compute = computeFacade;
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
		this._destroyBloomMipTextures();
	}

	public onShaderRuntimeChanged(): void {
		this._destroyCachedBindGroups();
		this._ssaoModule = null;
		this._ssaoRawPipeline = null;
		this._ssaoBlurPipeline = null;
		this._ssaoCombinePipeline = null;
		this._ssaoParams?.destroy();
		this._ssaoParams = null;
		this._ssgiModule = null;
		this._ssgiPipeline = null;
		this._ssgiParams?.destroy();
		this._ssgiParams = null;
		this._taaModule = null;
		this._taaPipeline = null;
		this._taaParams?.destroy();
		this._taaParams = null;
		this._hizModule = null;
		this._hizInitPipeline = null;
		this._hizReducePipeline = null;
		this._ssrModule = null;
		this._ssrTracePipeline = null;
		this._ssrComposePipeline = null;
		this._ssrTraceParams?.destroy();
		this._ssrTraceParams = null;
		this._ssrComposeParams?.destroy();
		this._ssrComposeParams = null;
		this._volumetricModule = null;
		this._volumetricPipeline = null;
		this._volumetricParams?.destroy();
		this._volumetricParams = null;
		this._volumetricLightBuffer?.destroy();
		this._volumetricLightBuffer = null;
		this._volumetricLightCapacity = 0;
		this._volumetricFrameIndex = 0;
		this._motionBlurModule = null;
		this._motionBlurPipeline = null;
		this._motionBlurParams?.destroy();
		this._motionBlurParams = null;
		this._motionBlurParamUploaded = false;
		this._dofModule = null;
		this._dofPipeline = null;
		this._dofParams?.destroy();
		this._dofParams = null;
		this._bloomDownsampleModule = null;
		this._bloomBlurHModule = null;
		this._bloomBlurVModule = null;
		this._bloomUpsampleModule = null;
		this._bloomCompositeModule = null;
		this._bloomDownsamplePipeline = null;
		this._bloomBlurHPipeline = null;
		this._bloomBlurVPipeline = null;
		this._bloomUpsamplePipeline = null;
		this._bloomCompositePipeline = null;
		this._bloomDownsampleParams?.destroy();
		this._bloomDownsampleParams = null;
		this._bloomBlurParams?.destroy();
		this._bloomBlurParams = null;
		this._bloomUpsampleParams?.destroy();
		this._bloomUpsampleParams = null;
		this._bloomCompositeParams?.destroy();
		this._bloomCompositeParams = null;
		this._destroyBloomMipTextures();
		this._fxaaModule = null;
		this._fxaaPipeline = null;
		this._fxaaParams?.destroy();
		this._fxaaParams = null;
		this._interactionOutlineModule = null;
		this._interactionOutlinePipeline = null;
		this._interactionOutlineParams?.destroy();
		this._interactionOutlineParams = null;
		this._copyModule = null;
		this._copyPipeline = null;
	}

	public async warmupHints(hints: readonly string[]): Promise<{
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
		const group = this._compute.createBindingGroup({
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
		destroyResource(group);
	}

	private async _warmupHint(hint: string): Promise<boolean> {
		switch (hint) {
			case "postprocess:ssao":
				await this._ensureSSAOResources();
				return true;
			case "postprocess:ssgi":
				await this._ensureSSGIResources();
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
			case "postprocess:motion-blur":
				await this._ensureMotionBlurResources();
				return true;
			case "postprocess:dof":
				await this._ensureDOFResources();
				return true;
			case "postprocess:bloom":
				await this._ensureBloomResources();
				return true;
			case "postprocess:fxaa":
				await this._ensureFXAAResources();
				return true;
			case "postprocess:interaction-outline":
				await this._ensureInteractionOutlineResources();
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
			this._compute.writeBuffer(
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

	public async executeSSGI(
		encoder: ICommandEncoder,
		targets: WebGPUFrameTargets,
		frameContext: FrameContext
	): Promise<void> {
		await this._ensureSSGIResources();
		if (!this._sampler || !this._ssgiPipeline || !this._ssgiParams) return;
		const options = frameContext.features.ssgiOptions ?? {};
		const target =
			targets.sceneColor === targets.postPong ?
				targets.postPing
			:	targets.postPong;
		const radius = clamp(
			finiteOr(options.radius, DEFAULT_SSGI_OPTIONS.radius),
			1,
			6
		);
		const intensity = Math.max(
			0,
			finiteOr(options.intensity, DEFAULT_SSGI_OPTIONS.intensity)
		);
		const falloff = Math.max(
			0.1,
			finiteOr(options.falloff, DEFAULT_SSGI_OPTIONS.falloff)
		);
		const depthPhi = Math.max(
			0.01,
			finiteOr(options.depthPhi, DEFAULT_SSGI_OPTIONS.depthPhi)
		);
		const normalPhi = Math.max(
			0.1,
			finiteOr(options.normalPhi, DEFAULT_SSGI_OPTIONS.normalPhi)
		);
		const albedoBoost = Math.max(
			0,
			finiteOr(options.albedoBoost, DEFAULT_SSGI_OPTIONS.albedoBoost)
		);
		this._compute.writeBuffer(
			this._ssgiParams,
			new Float32Array([
				1 / Math.max(target.width, 1),
				1 / Math.max(target.height, 1),
				radius,
				intensity,
				falloff,
				depthPhi,
				normalPhi,
				albedoBoost,
			])
		);
		const binding = this._getCachedBindGroup(
			`ssgi-${target === targets.postPing ? "ping" : "pong"}`,
			this._ssgiPipeline,
			[
				{ binding: 0, resource: targets.sceneColor },
				{ binding: 1, resource: targets.gAlbedoAlpha },
				{ binding: 2, resource: targets.gNormalRoughMetal },
				{ binding: 3, resource: targets.gMotionDepth },
				{ binding: 4, resource: this._sampler },
				{ binding: 5, resource: this._ssgiParams },
				{ binding: 6, resource: target },
			],
			"WebGPUSSGI_Binding"
		);
		encoder.beginComputePass({ label: "WebGPUSSGI" });
		encoder.setComputePipeline(this._ssgiPipeline);
		encoder.setBindingGroup(0, binding);
		encoder.dispatchWorkgroups(
			ceilDiv(target.width, WORKGROUP_SIZE),
			ceilDiv(target.height, WORKGROUP_SIZE),
			1
		);
		encoder.endComputePass();
		targets.sceneColor = target;
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
		this._compute.writeBuffer(
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
		this._compute.writeBuffer(
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
		this._compute.writeBuffer(
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

		this._compute.writeBuffer(
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

	public async executeMotionBlur(
		encoder: ICommandEncoder,
		targets: WebGPUFrameTargets,
		frameContext: FrameContext
	): Promise<void> {
		await this._ensureMotionBlurResources();
		if (
			!this._sampler ||
			!this._motionBlurPipeline ||
			!this._motionBlurParams
		) {
			return;
		}
		const options = frameContext.features.motionBlurOptions ?? {};
		const target =
			targets.sceneColor === targets.postPong ?
				targets.postPing
			:	targets.postPong;
		const shutterScale = clamp(
			finiteOr(options.shutterScale, DEFAULT_MOTION_BLUR_OPTIONS.shutterScale),
			0,
			2
		);
		const maxSamples = clamp(
			Math.round(
				finiteOr(options.maxSamples, DEFAULT_MOTION_BLUR_OPTIONS.maxSamples)
			),
			4,
			64
		);
		const velocityClamp = clamp(
			finiteOr(
				options.velocityClamp,
				DEFAULT_MOTION_BLUR_OPTIONS.velocityClamp
			),
			0.005,
			0.25
		);
		const depthReject = clamp(
			finiteOr(options.depthReject, DEFAULT_MOTION_BLUR_OPTIONS.depthReject),
			0.0001,
			0.25
		);
		const centerWeight = clamp(
			finiteOr(options.centerWeight, DEFAULT_MOTION_BLUR_OPTIONS.centerWeight),
			0,
			4
		);
		this._uploadMotionBlurParams(
			target.width,
			target.height,
			shutterScale,
			maxSamples,
			velocityClamp,
			depthReject,
			centerWeight
		);
		const binding = this._getCachedBindGroup(
			`motion-blur-${target === targets.postPing ? "ping" : "pong"}`,
			this._motionBlurPipeline,
			[
				{ binding: 0, resource: targets.sceneColor },
				{ binding: 1, resource: targets.gMotionDepth },
				{ binding: 2, resource: this._sampler },
				{ binding: 3, resource: this._motionBlurParams },
				{ binding: 4, resource: target },
			],
			"WebGPUMotionBlur_Binding"
		);
		encoder.beginComputePass({ label: "WebGPUMotionBlur" });
		encoder.setComputePipeline(this._motionBlurPipeline);
		encoder.setBindingGroup(0, binding);
		encoder.dispatchWorkgroups(
			ceilDiv(target.width, WORKGROUP_SIZE),
			ceilDiv(target.height, WORKGROUP_SIZE),
			1
		);
		encoder.endComputePass();
		targets.sceneColor = target;
	}

	public async executeDOF(
		encoder: ICommandEncoder,
		targets: WebGPUFrameTargets,
		frameContext: FrameContext
	): Promise<void> {
		await this._ensureDOFResources();
		if (!this._sampler || !this._dofPipeline || !this._dofParams) return;
		const options = frameContext.features.dofOptions ?? {};
		const target =
			targets.sceneColor === targets.postPong ?
				targets.postPing
			:	targets.postPong;
		const focusDistance = Math.max(
			0.01,
			finiteOr(options.focusDistance, DEFAULT_DOF_OPTIONS.focusDistance)
		);
		const focusRange = Math.max(
			0.001,
			finiteOr(options.focusRange, DEFAULT_DOF_OPTIONS.focusRange)
		);
		const nearStrength = clamp(
			finiteOr(options.nearStrength, DEFAULT_DOF_OPTIONS.nearStrength),
			0,
			2
		);
		const farStrength = clamp(
			finiteOr(options.farStrength, DEFAULT_DOF_OPTIONS.farStrength),
			0,
			2
		);
		const maxBlurRadius = clamp(
			finiteOr(options.maxBlurRadius, DEFAULT_DOF_OPTIONS.maxBlurRadius),
			0,
			32
		);
		const depthCurve = clamp(
			finiteOr(options.depthCurve, DEFAULT_DOF_OPTIONS.depthCurve),
			0.25,
			4
		);
		const highlightThreshold = Math.max(
			0,
			finiteOr(
				options.highlightThreshold,
				DEFAULT_DOF_OPTIONS.highlightThreshold
			)
		);
		const highlightGain = clamp(
			finiteOr(options.highlightGain, DEFAULT_DOF_OPTIONS.highlightGain),
			0,
			3
		);
		const chromaticAberration = clamp(
			finiteOr(
				options.chromaticAberration,
				DEFAULT_DOF_OPTIONS.chromaticAberration
			),
			0,
			2
		);
		this._compute.writeBuffer(
			this._dofParams,
			new Float32Array([
				1 / Math.max(target.width, 1),
				1 / Math.max(target.height, 1),
				focusDistance,
				focusRange,
				nearStrength,
				farStrength,
				maxBlurRadius,
				depthCurve,
				highlightThreshold,
				highlightGain,
				chromaticAberration,
				0,
			])
		);
		const binding = this._getCachedBindGroup(
			`dof-${target === targets.postPing ? "ping" : "pong"}`,
			this._dofPipeline,
			[
				{ binding: 0, resource: targets.sceneColor },
				{ binding: 1, resource: targets.gMotionDepth },
				{ binding: 2, resource: this._sampler },
				{ binding: 3, resource: this._dofParams },
				{ binding: 4, resource: target },
			],
			"WebGPUDOF_Binding"
		);
		encoder.beginComputePass({ label: "WebGPUDOF" });
		encoder.setComputePipeline(this._dofPipeline);
		encoder.setBindingGroup(0, binding);
		encoder.dispatchWorkgroups(
			ceilDiv(target.width, WORKGROUP_SIZE),
			ceilDiv(target.height, WORKGROUP_SIZE),
			1
		);
		encoder.endComputePass();
		targets.sceneColor = target;
	}

	public async executeBloom(
		encoder: ICommandEncoder,
		targets: WebGPUFrameTargets,
		frameContext: FrameContext
	): Promise<void> {
		await this._ensureBloomResources();
		if (
			!this._sampler ||
			!this._bloomDownsamplePipeline ||
			!this._bloomBlurHPipeline ||
			!this._bloomBlurVPipeline ||
			!this._bloomUpsamplePipeline ||
			!this._bloomCompositePipeline ||
			!this._bloomDownsampleParams ||
			!this._bloomBlurParams ||
			!this._bloomUpsampleParams ||
			!this._bloomCompositeParams
		) {
			return;
		}

		const options = frameContext.features.bloomOptions ?? {};
		const threshold = Math.max(
			0,
			finiteOr(options.threshold, DEFAULT_BLOOM_OPTIONS.threshold)
		);
		const softKnee = Math.max(
			1e-4,
			finiteOr(options.softKnee, DEFAULT_BLOOM_OPTIONS.softKnee)
		);
		const intensity = Math.max(
			0,
			finiteOr(options.intensity, DEFAULT_BLOOM_OPTIONS.intensity)
		);
		const filterRadius = clamp(
			finiteOr(options.filterRadius, DEFAULT_BLOOM_OPTIONS.filterRadius),
			0.5,
			4
		);
		const requestedMips = clamp(
			Math.round(
				finiteOr(options.mipPasses, DEFAULT_BLOOM_OPTIONS.mipPasses)
			),
			1,
			8
		);

		// Ensure mip-chain textures are sized correctly
		const srcW = targets.sceneColor.width;
		const srcH = targets.sceneColor.height;
		this._ensureBloomMipTextures(srcW, srcH, requestedMips);
		const mipCount = this._bloomMipCount;
		if (mipCount === 0) return;

		const mips = this._bloomMipTextures;

		// ---- Pass 1: Downsample + threshold into mip[mipCount-1] (half-res) ----
		const dsMipIndex = mipCount - 1;
		const dsDst = mips[dsMipIndex][0];
		const srcInvW = 1 / Math.max(srcW, 1);
		const srcInvH = 1 / Math.max(srcH, 1);
		this._compute.writeBuffer(
			this._bloomDownsampleParams,
			new Float32Array([srcInvW, srcInvH, threshold, softKnee])
		);
		let binding = this._getCachedBindGroup(
			"bloom-ds-0",
			this._bloomDownsamplePipeline,
			[
				{ binding: 0, resource: targets.sceneColor },
				{ binding: 1, resource: this._sampler },
				{ binding: 2, resource: this._bloomDownsampleParams },
				{ binding: 3, resource: dsDst },
			],
			"WebGPUBloom_Downsample0"
		);
		encoder.beginComputePass({ label: "WebGPUBloom_Downsample0" });
		encoder.setComputePipeline(this._bloomDownsamplePipeline);
		encoder.setBindingGroup(0, binding);
		encoder.dispatchWorkgroups(
			ceilDiv(dsDst.width, WORKGROUP_SIZE),
			ceilDiv(dsDst.height, WORKGROUP_SIZE),
			1
		);
		encoder.endComputePass();

		// Progressive downsample for subsequent mip levels (from dsMipIndex-1 down to 0)
		for (let i = dsMipIndex - 1; i >= 0; i--) {
			const src = mips[i + 1][0];
			const dst = mips[i][0];
			const dsInvW = 1 / Math.max(src.width, 1);
			const dsInvH = 1 / Math.max(src.height, 1);
			// Re-use downsample params (threshold=-1.0 to skip re-extraction)
			this._compute.writeBuffer(
				this._bloomDownsampleParams,
				new Float32Array([dsInvW, dsInvH, -1.0, 1e-4])
			);
			binding = this._getCachedBindGroup(
				`bloom-ds-${dsMipIndex - i}`,
				this._bloomDownsamplePipeline,
				[
					{ binding: 0, resource: src },
					{ binding: 1, resource: this._sampler },
					{ binding: 2, resource: this._bloomDownsampleParams },
					{ binding: 3, resource: dst },
				],
				`WebGPUBloom_Downsample${dsMipIndex - i}`
			);
			encoder.beginComputePass({
				label: `WebGPUBloom_Downsample${dsMipIndex - i}`,
			});
			encoder.setComputePipeline(this._bloomDownsamplePipeline);
			encoder.setBindingGroup(0, binding);
			encoder.dispatchWorkgroups(
				ceilDiv(dst.width, WORKGROUP_SIZE),
				ceilDiv(dst.height, WORKGROUP_SIZE),
				1
			);
			encoder.endComputePass();
		}

		// ---- Pass 2: Separable Gaussian blur per mip level ----
		for (let i = 0; i < mipCount; i++) {
			const texA = mips[i][0]; // source / result after downsample
			const texB = mips[i][1]; // temp ping-pong target

			// Horizontal blur: texA -> texB
			const invW = 1 / Math.max(texA.width, 1);
			const invH = 1 / Math.max(texA.height, 1);
			this._compute.writeBuffer(
				this._bloomBlurParams,
				new Float32Array([invW, invH, 1, 0])
			);
			binding = this._getCachedBindGroup(
				`bloom-blurH-${i}`,
				this._bloomBlurHPipeline,
				[
					{ binding: 0, resource: texA },
					{ binding: 1, resource: this._sampler },
					{ binding: 2, resource: this._bloomBlurParams },
					{ binding: 3, resource: texB },
				],
				`WebGPUBloom_BlurH_${i}`
			);
			encoder.beginComputePass({ label: `WebGPUBloom_BlurH_${i}` });
			encoder.setComputePipeline(this._bloomBlurHPipeline);
			encoder.setBindingGroup(0, binding);
			encoder.dispatchWorkgroups(
				ceilDiv(texB.width, WORKGROUP_SIZE),
				ceilDiv(texB.height, WORKGROUP_SIZE),
				1
			);
			encoder.endComputePass();

			// Vertical blur: texB -> texA
			this._compute.writeBuffer(
				this._bloomBlurParams,
				new Float32Array([invW, invH, 0, 1])
			);
			binding = this._getCachedBindGroup(
				`bloom-blurV-${i}`,
				this._bloomBlurVPipeline,
				[
					{ binding: 0, resource: texB },
					{ binding: 1, resource: this._sampler },
					{ binding: 2, resource: this._bloomBlurParams },
					{ binding: 3, resource: texA },
				],
				`WebGPUBloom_BlurV_${i}`
			);
			encoder.beginComputePass({ label: `WebGPUBloom_BlurV_${i}` });
			encoder.setComputePipeline(this._bloomBlurVPipeline);
			encoder.setBindingGroup(0, binding);
			encoder.dispatchWorkgroups(
				ceilDiv(texA.width, WORKGROUP_SIZE),
				ceilDiv(texA.height, WORKGROUP_SIZE),
				1
			);
			encoder.endComputePass();
		}

		// ---- Pass 3: Progressive upsample from smallest mip to largest mip ----
		for (let i = 1; i < mipCount; i++) {
			const smallerMip = mips[i - 1][0]; // smaller, already blurred
			const currentMip = mips[i][0]; // larger, blend target
			const dstMip = mips[i][1]; // write into ping-pong target

			const invW = 1 / Math.max(smallerMip.width, 1);
			const invH = 1 / Math.max(smallerMip.height, 1);
			this._compute.writeBuffer(
				this._bloomUpsampleParams,
				new Float32Array([invW, invH, filterRadius, 0])
			);
			binding = this._getCachedBindGroup(
				`bloom-up-${i}`,
				this._bloomUpsamplePipeline,
				[
					{ binding: 0, resource: smallerMip },
					{ binding: 1, resource: currentMip },
					{ binding: 2, resource: this._sampler },
					{ binding: 3, resource: this._bloomUpsampleParams },
					{ binding: 4, resource: dstMip },
				],
				`WebGPUBloom_Upsample_${i}`
			);
			encoder.beginComputePass({ label: `WebGPUBloom_Upsample_${i}` });
			encoder.setComputePipeline(this._bloomUpsamplePipeline);
			encoder.setBindingGroup(0, binding);
			encoder.dispatchWorkgroups(
				ceilDiv(dstMip.width, WORKGROUP_SIZE),
				ceilDiv(dstMip.height, WORKGROUP_SIZE),
				1
			);
			encoder.endComputePass();

			// Swap so mips[i][0] contains the latest result for the next pass
			mips[i] = [dstMip, currentMip];
		}

		// ---- Pass 4: Composite bloom onto scene color ----
		const bloomResult = mips[mipCount - 1][0]; // largest mip, fully upsampled
		const target =
			targets.sceneColor === targets.postPong ?
				targets.postPing
			:	targets.postPong;
		this._compute.writeBuffer(
			this._bloomCompositeParams,
			new Float32Array([
				1 / Math.max(target.width, 1),
				1 / Math.max(target.height, 1),
				intensity,
				0,
			])
		);
		binding = this._getCachedBindGroup(
			`bloom-comp-${target === targets.postPing ? "ping" : "pong"}`,
			this._bloomCompositePipeline,
			[
				{ binding: 0, resource: targets.sceneColor },
				{ binding: 1, resource: bloomResult },
				{ binding: 2, resource: this._sampler },
				{ binding: 3, resource: this._bloomCompositeParams },
				{ binding: 4, resource: target },
			],
			"WebGPUBloom_Composite"
		);
		encoder.beginComputePass({ label: "WebGPUBloom_Composite" });
		encoder.setComputePipeline(this._bloomCompositePipeline);
		encoder.setBindingGroup(0, binding);
		encoder.dispatchWorkgroups(
			ceilDiv(target.width, WORKGROUP_SIZE),
			ceilDiv(target.height, WORKGROUP_SIZE),
			1
		);
		encoder.endComputePass();
		targets.sceneColor = target;
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
		this._compute.writeBuffer(
			this._fxaaParams,
			new Float32Array([
				1 / Math.max(target.width, 1),
				1 / Math.max(target.height, 1),
				FXAA_EDGE_THRESHOLD_MIN,
				FXAA_EDGE_THRESHOLD_MULTIPLIER,
				FXAA_SUBPIX_QUALITY,
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

	public async executeInteractionOutline(
		encoder: ICommandEncoder,
		targets: WebGPUFrameTargets,
		frameContext: FrameContext,
		state?: InteractionTransientState | null
	): Promise<void> {
		const interactionState =
			state ??
			(frameContext.transient.get(INTERACTION_TRANSIENT_STATE_KEY) as
				| InteractionTransientState
				| null
				| undefined) ??
			null;
		const selectedEntityIds = interactionState?.selectedEntityIds ?? [];
		if (selectedEntityIds.length === 0) {
			return;
		}

		const circles = collectProjectedOutlineCircles(
			frameContext,
			selectedEntityIds,
			MAX_INTERACTION_OUTLINE_CIRCLES
		);
		if (circles.length === 0) {
			return;
		}

		await this._ensureInteractionOutlineResources();
		if (!this._interactionOutlinePipeline || !this._interactionOutlineParams) {
			return;
		}

		const target =
			targets.sceneColor === targets.postPong ?
				targets.postPing
			:	targets.postPong;
		const outlineColor = interactionState?.outline?.color ?? {
			r: 255,
			g: 196,
			b: 64,
			a: 1,
		};
		const colorScale =
			Math.max(outlineColor.r, outlineColor.g, outlineColor.b) > 1 ? 255 : 1;
		const opacity = clamp(
			finiteOr(interactionState?.outline?.opacity, 0.9) *
				finiteOr(outlineColor.a, 1),
			0,
			1
		);
		const thickness = Math.max(
			1,
			finiteOr(interactionState?.outline?.thickness, 2)
		);
		const shapeCode = getInteractionOutlineShapeCode(
			interactionState?.outline?.shape
		);
		const params = this._interactionOutlineParamData;
		params[0] = 1 / Math.max(target.width, 1);
		params[1] = 1 / Math.max(target.height, 1);
		params[2] = opacity;
		params[3] = thickness;
		params[4] = sRGBToLinear(
			clamp(outlineColor.r / Math.max(1, colorScale), 0, 1)
		);
		params[5] = sRGBToLinear(
			clamp(outlineColor.g / Math.max(1, colorScale), 0, 1)
		);
		params[6] = sRGBToLinear(
			clamp(outlineColor.b / Math.max(1, colorScale), 0, 1)
		);
		params[7] = 1;
		params[8] = circles.length;
		params[9] = shapeCode;
		params.fill(0, 10, INTERACTION_OUTLINE_HEADER_FLOATS);
		let offset = INTERACTION_OUTLINE_HEADER_FLOATS;
		for (let index = 0; index < MAX_INTERACTION_OUTLINE_CIRCLES; index++) {
			if (index < circles.length) {
				const circle = circles[index];
				params[offset] = circle.centerX;
				params[offset + 1] = circle.centerY;
				params[offset + 2] = circle.radius;
				params[offset + 3] = 0;
			} else {
				params[offset] = 0;
				params[offset + 1] = 0;
				params[offset + 2] = 0;
				params[offset + 3] = 0;
			}
			offset += 4;
		}
		this._compute.writeBuffer(this._interactionOutlineParams, params);

		const binding = this._getCachedBindGroup(
			`interaction-outline-${target === targets.postPing ? "ping" : "pong"}`,
			this._interactionOutlinePipeline,
			[
				{ binding: 0, resource: targets.sceneColor },
				{ binding: 2, resource: this._interactionOutlineParams },
				{ binding: 3, resource: target },
			],
			"WebGPUInteractionOutline_Binding"
		);
		encoder.beginComputePass({ label: "WebGPUInteractionOutline" });
		encoder.setComputePipeline(this._interactionOutlinePipeline);
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

		this._compute.writeBuffer(this._volumetricLightBuffer, packed);
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
		this._volumetricLightBuffer = this._compute.createBuffer({
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
		this._sampler = this._compute.createSampler({
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
			this._ssaoModule = await this._compute.createShaderModule({
				label: "WebGPUSSAOShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!this._ssaoRawPipeline)
			this._ssaoRawPipeline = this._compute.createComputePipeline({
				label: "WebGPUSSAORawPipeline",
				compute: { module: this._ssaoModule, entryPoint: "csRaw" },
			});
		if (!this._ssaoBlurPipeline)
			this._ssaoBlurPipeline = this._compute.createComputePipeline({
				label: "WebGPUSSAOBlurPipeline",
				compute: { module: this._ssaoModule, entryPoint: "csBlur" },
			});
		if (!this._ssaoCombinePipeline)
			this._ssaoCombinePipeline = this._compute.createComputePipeline({
				label: "WebGPUSSAOCombinePipeline",
				compute: { module: this._ssaoModule, entryPoint: "csCombine" },
			});
		if (!this._ssaoParams)
			this._ssaoParams = this._compute.createBuffer({
				label: "WebGPUSSAOParams",
				size: 16 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
	}

	private async _ensureSSGIResources(): Promise<void> {
		await this._ensureCommonResources();
		if (!this._ssgiModule) {
			const shader = await loadPostProcessShaderPartComposite("ssgi");
			this._ssgiModule = await this._compute.createShaderModule({
				label: "WebGPUSSGIShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!this._ssgiPipeline)
			this._ssgiPipeline = this._compute.createComputePipeline({
				label: "WebGPUSSGIPipeline",
				compute: { module: this._ssgiModule, entryPoint: "csMain" },
			});
		if (!this._ssgiParams)
			this._ssgiParams = this._compute.createBuffer({
				label: "WebGPUSSGIParams",
				size: 8 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
	}

	private async _ensureTAAResources(): Promise<void> {
		await this._ensureCommonResources();
		if (!this._taaModule) {
			const shader = await loadPostProcessShaderPartComposite("taa");
			this._taaModule = await this._compute.createShaderModule({
				label: "WebGPUTAAShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!this._taaPipeline)
			this._taaPipeline = this._compute.createComputePipeline({
				label: "WebGPUTAAPipeline",
				compute: { module: this._taaModule, entryPoint: "csMain" },
			});
		if (!this._taaParams)
			this._taaParams = this._compute.createBuffer({
				label: "WebGPUTAAParams",
				size: 8 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
	}

	private async _ensureHiZResources(): Promise<void> {
		await this._ensureCommonResources();
		if (!this._hizModule) {
			const shader = await loadPostProcessShaderPartComposite("hiz");
			this._hizModule = await this._compute.createShaderModule({
				label: "WebGPUHiZShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!this._hizInitPipeline)
			this._hizInitPipeline = this._compute.createComputePipeline({
				label: "WebGPUHiZInitPipeline",
				compute: { module: this._hizModule, entryPoint: "csInit" },
			});
		if (!this._hizReducePipeline)
			this._hizReducePipeline = this._compute.createComputePipeline({
				label: "WebGPUHiZReducePipeline",
				compute: { module: this._hizModule, entryPoint: "csReduce" },
			});
	}

	private async _ensureSSRResources(): Promise<void> {
		await this._ensureHiZResources();
		if (!this._ssrModule) {
			const shader = await loadPostProcessShaderPartComposite("ssr");
			this._ssrModule = await this._compute.createShaderModule({
				label: "WebGPUSSRShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!this._ssrTracePipeline) {
			if (this._frameBindGroupLayout) {
				this._ssrTraceGroupLayout0 = this._compute.createBindGroupLayout({
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
				this._ssrTracePipelineLayout = this._compute.createPipelineLayout({
					label: "WebGPUSSRTrace_PipelineLayout",
					bindGroupLayouts: [
						this._ssrTraceGroupLayout0,
						this._frameBindGroupLayout,
					],
				});
				this._ssrTracePipeline = this._compute.createComputePipeline({
					label: "WebGPUSSRTracePipeline",
					layout: this._ssrTracePipelineLayout,
					compute: { module: this._ssrModule!, entryPoint: "csTrace" },
				});
			} else {
				this._ssrTracePipeline = this._compute.createComputePipeline({
					label: "WebGPUSSRTracePipeline",
					compute: { module: this._ssrModule!, entryPoint: "csTrace" },
				});
			}
		}
		if (!this._ssrComposePipeline)
			this._ssrComposePipeline = this._compute.createComputePipeline({
				label: "WebGPUSSRComposePipeline",
				compute: { module: this._ssrModule, entryPoint: "csCompose" },
			});
		if (!this._ssrTraceParams)
			this._ssrTraceParams = this._compute.createBuffer({
				label: "WebGPUSSRTraceParams",
				size: 16 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
		if (!this._ssrComposeParams)
			this._ssrComposeParams = this._compute.createBuffer({
				label: "WebGPUSSRComposeParams",
				size: 4 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
	}

	private async _ensureVolumetricResources(): Promise<void> {
		await this._ensureHiZResources();
		if (!this._volumetricModule) {
			const shader = await loadPostProcessShaderPartComposite("volumetric");
			this._volumetricModule = await this._compute.createShaderModule({
				label: "WebGPUVolumetricShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!this._volumetricPipeline) {
			if (this._frameBindGroupLayout) {
				this._volumetricGroupLayout0 = this._compute.createBindGroupLayout({
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
				this._volumetricPipelineLayout = this._compute.createPipelineLayout({
					label: "WebGPUVolumetric_PipelineLayout",
					bindGroupLayouts: [
						this._volumetricGroupLayout0,
						this._frameBindGroupLayout,
					],
				});
				this._volumetricPipeline = this._compute.createComputePipeline({
					label: "WebGPUVolumetricPipeline",
					layout: this._volumetricPipelineLayout,
					compute: { module: this._volumetricModule, entryPoint: "csMain" },
				});
			} else {
				this._volumetricPipeline = this._compute.createComputePipeline({
					label: "WebGPUVolumetricPipeline",
					compute: { module: this._volumetricModule, entryPoint: "csMain" },
				});
			}
		}
		if (!this._volumetricParams)
			this._volumetricParams = this._compute.createBuffer({
				label: "WebGPUVolumetricParams",
				size: 20 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
	}

	private async _ensureMotionBlurResources(): Promise<void> {
		await this._ensureCommonResources();
		if (!this._motionBlurModule) {
			const shader = await loadPostProcessShaderPartComposite("motionBlur");
			this._motionBlurModule = await this._compute.createShaderModule({
				label: "WebGPUMotionBlurShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!this._motionBlurPipeline)
			this._motionBlurPipeline = this._compute.createComputePipeline({
				label: "WebGPUMotionBlurPipeline",
				compute: { module: this._motionBlurModule, entryPoint: "csMain" },
			});
		if (!this._motionBlurParams)
			this._motionBlurParams = this._compute.createBuffer({
				label: "WebGPUMotionBlurParams",
				size: 8 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
	}

	private _uploadMotionBlurParams(
		width: number,
		height: number,
		shutterScale: number,
		maxSamples: number,
		velocityClamp: number,
		depthReject: number,
		centerWeight: number
	): void {
		if (!this._motionBlurParams) {
			return;
		}
		const data = this._motionBlurParamData;
		let changed = !this._motionBlurParamUploaded;
		changed =
			this._setParamIfChanged(data, 0, 1 / Math.max(width, 1)) || changed;
		changed =
			this._setParamIfChanged(data, 1, 1 / Math.max(height, 1)) || changed;
		changed = this._setParamIfChanged(data, 2, shutterScale) || changed;
		changed = this._setParamIfChanged(data, 3, maxSamples) || changed;
		changed = this._setParamIfChanged(data, 4, velocityClamp) || changed;
		changed = this._setParamIfChanged(data, 5, depthReject) || changed;
		changed = this._setParamIfChanged(data, 6, centerWeight) || changed;
		changed = this._setParamIfChanged(data, 7, 0) || changed;
		if (!changed) {
			return;
		}
		this._compute.writeBuffer(this._motionBlurParams, data);
		this._motionBlurParamUploaded = true;
	}

	private _setParamIfChanged(
		data: Float32Array,
		index: number,
		value: number
	): boolean {
		const nextValue = Math.fround(value);
		if (data[index] === nextValue) {
			return false;
		}
		data[index] = nextValue;
		return true;
	}

	private async _ensureDOFResources(): Promise<void> {
		await this._ensureCommonResources();
		if (!this._dofModule) {
			const shader = await loadPostProcessShaderPartComposite("dof");
			this._dofModule = await this._compute.createShaderModule({
				label: "WebGPUDOFShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!this._dofPipeline)
			this._dofPipeline = this._compute.createComputePipeline({
				label: "WebGPUDOFPipeline",
				compute: { module: this._dofModule, entryPoint: "csMain" },
			});
		if (!this._dofParams)
			this._dofParams = this._compute.createBuffer({
				label: "WebGPUDOFParams",
				size: 12 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
	}

	private async _ensureBloomResources(): Promise<void> {
		await this._ensureCommonResources();

		// Load separate shader modules for each bloom pass
		if (!this._bloomDownsampleModule) {
			const shader = await loadPostProcessShaderPartComposite("bloomDownsample");
			this._bloomDownsampleModule = await this._compute.createShaderModule({
				label: "WebGPUBloomDownsampleShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!this._bloomBlurHModule) {
			const shader = await loadPostProcessShaderPartComposite("bloomBlurH");
			this._bloomBlurHModule = await this._compute.createShaderModule({
				label: "WebGPUBloomBlurHShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!this._bloomBlurVModule) {
			const shader = await loadPostProcessShaderPartComposite("bloomBlurV");
			this._bloomBlurVModule = await this._compute.createShaderModule({
				label: "WebGPUBloomBlurVShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!this._bloomUpsampleModule) {
			const shader = await loadPostProcessShaderPartComposite("bloomUpsample");
			this._bloomUpsampleModule = await this._compute.createShaderModule({
				label: "WebGPUBloomUpsampleShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!this._bloomCompositeModule) {
			const shader = await loadPostProcessShaderPartComposite("bloomComposite");
			this._bloomCompositeModule = await this._compute.createShaderModule({
				label: "WebGPUBloomCompositeShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}

		// Create compute pipelines
		if (!this._bloomDownsamplePipeline)
			this._bloomDownsamplePipeline = this._compute.createComputePipeline({
				label: "WebGPUBloomDownsamplePipeline",
				compute: {
					module: this._bloomDownsampleModule,
					entryPoint: "csMain",
				},
			});
		if (!this._bloomBlurHPipeline)
			this._bloomBlurHPipeline = this._compute.createComputePipeline({
				label: "WebGPUBloomBlurHPipeline",
				compute: {
					module: this._bloomBlurHModule,
					entryPoint: "csMain",
				},
			});
		if (!this._bloomBlurVPipeline)
			this._bloomBlurVPipeline = this._compute.createComputePipeline({
				label: "WebGPUBloomBlurVPipeline",
				compute: {
					module: this._bloomBlurVModule,
					entryPoint: "csMain",
				},
			});
		if (!this._bloomUpsamplePipeline)
			this._bloomUpsamplePipeline = this._compute.createComputePipeline({
				label: "WebGPUBloomUpsamplePipeline",
				compute: {
					module: this._bloomUpsampleModule,
					entryPoint: "csMain",
				},
			});
		if (!this._bloomCompositePipeline)
			this._bloomCompositePipeline = this._compute.createComputePipeline({
				label: "WebGPUBloomCompositePipeline",
				compute: {
					module: this._bloomCompositeModule,
					entryPoint: "csMain",
				},
			});

		// Create uniform buffers
		if (!this._bloomDownsampleParams)
			this._bloomDownsampleParams = this._compute.createBuffer({
				label: "WebGPUBloomDownsampleParams",
				size: 4 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
		if (!this._bloomBlurParams)
			this._bloomBlurParams = this._compute.createBuffer({
				label: "WebGPUBloomBlurParams",
				size: 4 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
		if (!this._bloomUpsampleParams)
			this._bloomUpsampleParams = this._compute.createBuffer({
				label: "WebGPUBloomUpsampleParams",
				size: 4 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
		if (!this._bloomCompositeParams)
			this._bloomCompositeParams = this._compute.createBuffer({
				label: "WebGPUBloomCompositeParams",
				size: 4 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
	}

	/**
	 * Allocate or resize bloom mip textures. Each mip level has two
	 * textures (ping / pong) for separable blur. Mip 0 is the smallest
	 * (deepest) level, mip N-1 is half-resolution of the source.
	 */
	private _ensureBloomMipTextures(
		srcWidth: number,
		srcHeight: number,
		requestedMips: number
	): void {
		const halfW = Math.max(1, Math.floor(srcWidth / 2));
		const halfH = Math.max(1, Math.floor(srcHeight / 2));
		const maxPossibleMips =
			Math.floor(Math.log2(Math.max(halfW, halfH))) + 1;
		const mipCount = Math.min(requestedMips, maxPossibleMips);

		// Check cache validity
		if (
			this._bloomMipWidth === halfW &&
			this._bloomMipHeight === halfH &&
			this._bloomMipCount === mipCount &&
			this._bloomMipTextures.length === mipCount
		) {
			return;
		}

		// Destroy old textures
		this._destroyBloomMipTextures();

		this._bloomMipWidth = halfW;
		this._bloomMipHeight = halfH;
		this._bloomMipCount = mipCount;

		// Allocate from largest (half-res) to smallest.
		// Index mipCount-1 = half-res, index 0 = smallest.
		for (let i = 0; i < mipCount; i++) {
			const level = mipCount - 1 - i; // 0=biggest, mipCount-1=smallest
			const w = Math.max(1, halfW >> level);
			const h = Math.max(1, halfH >> level);
			const texA = this._compute.createTexture({
				width: w,
				height: h,
				format: TextureFormat.RGBA16Float,
				usage:
					TextureUsage.TextureBinding |
					TextureUsage.StorageBinding |
					TextureUsage.ComputeStorage,
				label: `WebGPUBloomMip${i}_A_${w}x${h}`,
			});
			const texB = this._compute.createTexture({
				width: w,
				height: h,
				format: TextureFormat.RGBA16Float,
				usage:
					TextureUsage.TextureBinding |
					TextureUsage.StorageBinding |
					TextureUsage.ComputeStorage,
				label: `WebGPUBloomMip${i}_B_${w}x${h}`,
			});
			this._bloomMipTextures.push([texA, texB]);
		}
	}

	/**
	 * Destroy all cached bloom mip textures and invalidate bind group cache
	 * entries that reference them.
	 */
	private _destroyBloomMipTextures(): void {
		for (const [texA, texB] of this._bloomMipTextures) {
			texA.destroy();
			texB.destroy();
		}
		this._bloomMipTextures = [];
		this._bloomMipWidth = 0;
		this._bloomMipHeight = 0;
		this._bloomMipCount = 0;
		// Invalidate bind groups that may reference destroyed mip textures
		for (const key of Array.from(this._bindGroupCache.keys())) {
			if (key.startsWith("bloom-")) {
				const cached = this._bindGroupCache.get(key);
				if (cached) {
					this._destroyBindingGroup(cached.group);
				}
				this._bindGroupCache.delete(key);
			}
		}
	}

	private async _ensureFXAAResources(): Promise<void> {
		await this._ensureCommonResources();
		if (!this._fxaaModule) {
			const shader = await loadPostProcessShaderPartComposite("fxaa");
			this._fxaaModule = await this._compute.createShaderModule({
				label: "WebGPUFXAAShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!this._fxaaPipeline)
			this._fxaaPipeline = this._compute.createComputePipeline({
				label: "WebGPUFXAAPipeline",
				compute: { module: this._fxaaModule, entryPoint: "csMain" },
			});
		if (!this._fxaaParams)
			this._fxaaParams = this._compute.createBuffer({
				label: "WebGPUFXAAParams",
				size: 6 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
	}

	private async _ensureInteractionOutlineResources(): Promise<void> {
		await this._ensureCommonResources();
		if (!this._interactionOutlineModule) {
			const shader =
				await loadPostProcessShaderPartComposite("interactionOutline");
			this._interactionOutlineModule = await this._compute.createShaderModule({
				label: "WebGPUInteractionOutlineShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!this._interactionOutlinePipeline) {
			this._interactionOutlinePipeline = this._compute.createComputePipeline({
				label: "WebGPUInteractionOutlinePipeline",
				compute: {
					module: this._interactionOutlineModule,
					entryPoint: "csMain",
				},
			});
		}
		if (!this._interactionOutlineParams) {
			this._interactionOutlineParams = this._compute.createBuffer({
				label: "WebGPUInteractionOutlineParams",
				size: this._interactionOutlineParamData.byteLength,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
		}
	}

	private async _ensureCopyResources(): Promise<void> {
		if (!this._copyModule) {
			const shader = await loadPostProcessShaderPartComposite("copy");
			this._copyModule = await this._compute.createShaderModule({
				label: "WebGPUCopyShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!this._copyPipeline)
			this._copyPipeline = this._compute.createComputePipeline({
				label: "WebGPUCopyPipeline",
				compute: { module: this._copyModule, entryPoint: "csMain" },
			});
	}

	private _getHiZMipViews(texture: IRenderTexture): GPUTextureView[] {
		const cached = this._hizViewCache.get(texture as object);
		if (cached) return cached;
		const mipCount =
			Math.floor(Math.log2(Math.max(texture.width, texture.height))) + 1;
		const views: GPUTextureView[] = [];
		for (let i = 0; i < mipCount; i++) {
			views.push(
				this._compute.createTextureView(texture, {
					baseMipLevel: i,
					mipLevelCount: 1,
				})
			);
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
