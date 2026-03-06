import { CameraType } from "../../../cameras/Camera";
import type {
	FrameContext,
	SSAOOptions,
	SSROptions,
	TAAOptions,
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

import SSAO_SHADER from "../../../shaders/webgpu/postprocess/ssao.wgsl?raw";
import TAA_SHADER from "../../../shaders/webgpu/postprocess/taa.wgsl?raw";
import HIZ_SHADER from "../../../shaders/webgpu/postprocess/hiz.wgsl?raw";
import SSR_SHADER from "../../../shaders/webgpu/postprocess/ssr.wgsl?raw";
import FXAA_SHADER from "../../../shaders/webgpu/postprocess/fxaa.wgsl?raw";
import COPY_SHADER from "../../../shaders/webgpu/postprocess/copy.wgsl?raw";

interface InternalTexture extends IRenderTexture {
	_gpuTexture?: any;
	_gpuResource?: any;
}

const WORKGROUP_SIZE = 8;

const DEFAULT_SSAO: Required<
	Pick<
		SSAOOptions,
		| "samples"
		| "radius"
		| "bias"
		| "intensity"
		| "downsample"
		| "blurRadius"
		| "blurSharpness"
	>
> = {
	samples: 16,
	radius: 8,
	bias: 0.1,
	intensity: 1,
	downsample: 2,
	blurRadius: 2,
	blurSharpness: 8,
};

const DEFAULT_TAA: Required<
	Pick<
		TAAOptions,
		| "historyWeight"
		| "disocclusionDepthThreshold"
		| "motionFactor"
		| "varianceClampGamma"
		| "sharpen"
	>
> = {
	historyWeight: 0.9,
	disocclusionDepthThreshold: 0.02,
	motionFactor: 80,
	varianceClampGamma: 1,
	sharpen: 0.1,
};

const DEFAULT_SSR: Required<
	Pick<
		SSROptions,
		| "downsample"
		| "maxSteps"
		| "binarySearchSteps"
		| "maxDistance"
		| "thickness"
		| "stride"
		| "intensity"
		| "historyWeight"
		| "edgeFade"
		| "maxRoughness"
	>
> = {
	downsample: 2,
	maxSteps: 64,
	binarySearchSteps: 6,
	maxDistance: 100,
	thickness: 0.2,
	stride: 1,
	intensity: 1,
	historyWeight: 0.85,
	edgeFade: 0.12,
	maxRoughness: 0.85,
};

const DEFAULT_FXAA = {
	edgeThresholdMin: 0.03125,
	edgeThresholdMultiplier: 0.166,
	subpixQuality: 0.75,
};

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
	private _fxaaModule: IShaderModule | null = null;
	private _fxaaPipeline: IComputePipeline | null = null;
	private _fxaaParams: IRenderBuffer | null = null;
	private _copyModule: IShaderModule | null = null;
	private _copyPipeline: IComputePipeline | null = null;
	private _hizViewCache = new WeakMap<object, any[]>();

	constructor(
		backend: WebGPUBackend,
		warn: (key: string, message: string) => void
	) {
		this._backend = backend;
		this._warn = warn;
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
		const radius = finiteOr(options.radius, DEFAULT_SSAO.radius);
		const bias = finiteOr(options.bias, DEFAULT_SSAO.bias);
		const intensity = finiteOr(options.intensity, DEFAULT_SSAO.intensity);
		const blurRadius = finiteOr(options.blurRadius, DEFAULT_SSAO.blurRadius);
		const blurSharpness = finiteOr(
			options.blurSharpness,
			DEFAULT_SSAO.blurSharpness
		);
		const fullInvW = 1 / Math.max(targets.sceneColor.width, 1);
		const fullInvH = 1 / Math.max(targets.sceneColor.height, 1);
		const aoInvW = 1 / Math.max(targets.aoRaw.width, 1);
		const aoInvH = 1 / Math.max(targets.aoRaw.height, 1);
		this._backend.writeBuffer(
			this._ssaoParams,
			new Float32Array([
				fullInvW,
				fullInvH,
				aoInvW,
				aoInvH,
				radius,
				bias,
				intensity,
				blurRadius,
				blurSharpness,
				0,
				0,
				0,
			])
		);
		let binding = this._backend.createBindingGroup({
			pipeline: this._ssaoRawPipeline,
			layoutIndex: 0,
			entries: [
				{ binding: 0, resource: targets.gNormalRoughMetal },
				{ binding: 1, resource: targets.gMotionDepth },
				{ binding: 2, resource: this._sampler },
				{ binding: 3, resource: this._ssaoParams },
				{ binding: 4, resource: targets.aoRaw },
			],
			label: "WebGPUSSAO_RawBinding",
		});
		encoder.beginComputePass({ label: "WebGPUSSAO_Raw" });
		encoder.setComputePipeline(this._ssaoRawPipeline);
		encoder.setBindingGroup(0, binding);
		encoder.dispatchWorkgroups(
			ceilDiv(targets.aoRaw.width, WORKGROUP_SIZE),
			ceilDiv(targets.aoRaw.height, WORKGROUP_SIZE),
			1
		);
		encoder.endComputePass();
		binding = this._backend.createBindingGroup({
			pipeline: this._ssaoBlurPipeline,
			layoutIndex: 0,
			entries: [
				{ binding: 0, resource: targets.aoRaw },
				{ binding: 1, resource: targets.gMotionDepth },
				{ binding: 2, resource: this._sampler },
				{ binding: 3, resource: this._ssaoParams },
				{ binding: 4, resource: targets.aoBlur },
			],
			label: "WebGPUSSAO_BlurBinding",
		});
		encoder.beginComputePass({ label: "WebGPUSSAO_Blur" });
		encoder.setComputePipeline(this._ssaoBlurPipeline);
		encoder.setBindingGroup(0, binding);
		encoder.dispatchWorkgroups(
			ceilDiv(targets.aoBlur.width, WORKGROUP_SIZE),
			ceilDiv(targets.aoBlur.height, WORKGROUP_SIZE),
			1
		);
		encoder.endComputePass();

		const combineTarget =
			targets.sceneColor === targets.postPing
				? targets.postPong
				: targets.postPing;
		binding = this._backend.createBindingGroup({
			pipeline: this._ssaoCombinePipeline,
			layoutIndex: 0,
			entries: [
				{ binding: 0, resource: targets.sceneColor },
				{ binding: 1, resource: targets.aoBlur },
				{ binding: 2, resource: this._sampler },
				{ binding: 3, resource: this._ssaoParams },
				{ binding: 4, resource: combineTarget },
			],
			label: "WebGPUSSAO_CombineBinding",
		});
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
			targets.sceneColor === targets.postPong
				? targets.postPing
				: targets.postPong;
		const invW = 1 / Math.max(taaTarget.width, 1);
		const invH = 1 / Math.max(taaTarget.height, 1);
		this._backend.writeBuffer(
			this._taaParams,
			new Float32Array([
				invW,
				invH,
				finiteOr(options.historyWeight, DEFAULT_TAA.historyWeight),
				finiteOr(
					options.disocclusionDepthThreshold,
					DEFAULT_TAA.disocclusionDepthThreshold
				),
				finiteOr(options.motionFactor, DEFAULT_TAA.motionFactor),
				finiteOr(options.varianceClampGamma, DEFAULT_TAA.varianceClampGamma),
				finiteOr(options.sharpen, DEFAULT_TAA.sharpen),
				historyValid ? 1 : 0,
			])
		);
		const binding = this._backend.createBindingGroup({
			pipeline: this._taaPipeline,
			layoutIndex: 0,
			entries: [
				{ binding: 0, resource: targets.sceneColor },
				{ binding: 1, resource: targets.historyRead },
				{ binding: 2, resource: targets.gMotionDepth },
				{ binding: 3, resource: targets.motionHistoryRead },
				{ binding: 4, resource: this._sampler },
				{ binding: 5, resource: this._taaParams },
				{ binding: 6, resource: taaTarget },
				{ binding: 7, resource: targets.historyWrite },
			],
			label: "WebGPUTAA_Binding",
		});
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
		historyValid: boolean
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
		if (hiZMips.length === 0) return false;
		let binding = this._backend.createBindingGroup({
			pipeline: this._hizInitPipeline,
			layoutIndex: 0,
			entries: [
				{ binding: 0, resource: targets.gMotionDepth },
				{ binding: 1, resource: hiZMips[0] },
			],
			label: "WebGPUSSR_HiZInitBinding",
		});
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
			binding = this._backend.createBindingGroup({
				pipeline: this._hizReducePipeline,
				layoutIndex: 1,
				entries: [
					{ binding: 0, resource: hiZMips[mip - 1] },
					{ binding: 1, resource: hiZMips[mip] },
				],
				label: `WebGPUSSR_HiZReduceBinding_${mip}`,
			});
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
		this._backend.writeBuffer(
			this._ssrTraceParams,
			new Float32Array([
				1 / Math.max(targets.ssrRaw.width, 1),
				1 / Math.max(targets.ssrRaw.height, 1),
				finiteOr(options.maxDistance, DEFAULT_SSR.maxDistance),
				finiteOr(options.thickness, DEFAULT_SSR.thickness),
				finiteOr(options.stride, DEFAULT_SSR.stride),
				finiteOr(options.intensity, DEFAULT_SSR.intensity),
				finiteOr(options.maxRoughness, DEFAULT_SSR.maxRoughness),
				finiteOr(options.edgeFade, DEFAULT_SSR.edgeFade),
				finiteOr(options.maxSteps, DEFAULT_SSR.maxSteps),
				finiteOr(options.binarySearchSteps, DEFAULT_SSR.binarySearchSteps),
				hiZMips.length - 1,
				finiteOr(options.historyWeight, DEFAULT_SSR.historyWeight),
				historyValid ? 1 : 0,
				0.02,
				0,
				0,
			])
		);
		binding = this._backend.createBindingGroup({
			pipeline: this._ssrTracePipeline,
			layoutIndex: 0,
			entries: [
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
			label: "WebGPUSSR_TraceBinding",
		});
		encoder.beginComputePass({ label: "WebGPUSSR_TraceTemporal" });
		encoder.setComputePipeline(this._ssrTracePipeline);
		encoder.setBindingGroup(0, binding);
		encoder.dispatchWorkgroups(
			ceilDiv(targets.ssrRaw.width, WORKGROUP_SIZE),
			ceilDiv(targets.ssrRaw.height, WORKGROUP_SIZE),
			1
		);
		encoder.endComputePass();
		await this._copyTexture(encoder, targets.ssrRaw, targets.ssrHistoryWrite);
		const composeTarget =
			targets.sceneColor === targets.postPing
				? targets.postPong
				: targets.postPing;
		this._backend.writeBuffer(
			this._ssrComposeParams,
			new Float32Array([
				1 / Math.max(composeTarget.width, 1),
				1 / Math.max(composeTarget.height, 1),
				0,
				0,
			])
		);
		binding = this._backend.createBindingGroup({
			pipeline: this._ssrComposePipeline,
			layoutIndex: 1,
			entries: [
				{ binding: 0, resource: targets.sceneColor },
				{ binding: 1, resource: targets.ssrRaw },
				{ binding: 2, resource: targets.gMotionDepth },
				{ binding: 3, resource: this._sampler },
				{ binding: 4, resource: this._ssrComposeParams },
				{ binding: 5, resource: composeTarget },
			],
			label: "WebGPUSSR_ComposeBinding",
		});
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

	public async executeFXAA(
		encoder: ICommandEncoder,
		targets: WebGPUFrameTargets
	): Promise<void> {
		await this._ensureFXAAResources();
		if (!this._sampler || !this._fxaaPipeline || !this._fxaaParams) return;
		const target =
			targets.sceneColor === targets.postPong
				? targets.postPing
				: targets.postPong;
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
		const binding = this._backend.createBindingGroup({
			pipeline: this._fxaaPipeline,
			layoutIndex: 0,
			entries: [
				{ binding: 0, resource: targets.sceneColor },
				{ binding: 1, resource: this._sampler },
				{ binding: 2, resource: this._fxaaParams },
				{ binding: 3, resource: target },
			],
			label: "WebGPUFXAA_Binding",
		});
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

	private async _copyTexture(
		encoder: ICommandEncoder,
		src: IRenderTexture,
		dst: IRenderTexture
	): Promise<void> {
		if (src === dst) return;
		await this._ensureCopyResources();
		if (!this._copyPipeline) return;
		const binding = this._backend.createBindingGroup({
			pipeline: this._copyPipeline,
			layoutIndex: 0,
			entries: [
				{ binding: 0, resource: src },
				{ binding: 1, resource: dst },
			],
			label: "WebGPUPost_CopyBinding",
		});
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
		if (!this._ssaoModule)
			this._ssaoModule = await this._backend.createShaderModule({
				label: "WebGPUSSAOShader",
				code: SSAO_SHADER,
			});
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
				size: 12 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
	}

	private async _ensureTAAResources(): Promise<void> {
		await this._ensureCommonResources();
		if (!this._taaModule)
			this._taaModule = await this._backend.createShaderModule({
				label: "WebGPUTAAShader",
				code: TAA_SHADER,
			});
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

	private async _ensureSSRResources(): Promise<void> {
		await this._ensureCommonResources();
		if (!this._hizModule)
			this._hizModule = await this._backend.createShaderModule({
				label: "WebGPUHiZShader",
				code: HIZ_SHADER,
			});
		if (!this._ssrModule)
			this._ssrModule = await this._backend.createShaderModule({
				label: "WebGPUSSRShader",
				code: SSR_SHADER,
			});
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
		if (!this._ssrTracePipeline)
			this._ssrTracePipeline = this._backend.createComputePipeline({
				label: "WebGPUSSRTracePipeline",
				compute: { module: this._ssrModule, entryPoint: "csTrace" },
			});
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

	private async _ensureFXAAResources(): Promise<void> {
		await this._ensureCommonResources();
		if (!this._fxaaModule)
			this._fxaaModule = await this._backend.createShaderModule({
				label: "WebGPUFXAAShader",
				code: FXAA_SHADER,
			});
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
		if (!this._copyModule)
			this._copyModule = await this._backend.createShaderModule({
				label: "WebGPUCopyShader",
				code: COPY_SHADER,
			});
		if (!this._copyPipeline)
			this._copyPipeline = this._backend.createComputePipeline({
				label: "WebGPUCopyPipeline",
				compute: { module: this._copyModule, entryPoint: "csMain" },
			});
	}

	private _getHiZMipViews(texture: IRenderTexture): any[] {
		const cached = this._hizViewCache.get(texture as object);
		if (cached) return cached;
		const gpuTexture =
			(texture as InternalTexture)._gpuTexture ??
			(texture as InternalTexture)._gpuResource;
		if (!gpuTexture?.createView) return [];
		const mipCount =
			Math.floor(Math.log2(Math.max(texture.width, texture.height))) + 1;
		const views: any[] = [];
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
