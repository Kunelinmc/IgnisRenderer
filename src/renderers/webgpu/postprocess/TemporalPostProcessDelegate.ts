import { CameraType } from "../../../cameras/Camera";
import type { FrameContext } from "../../../pipeline/types";
import {
	DEFAULT_SSR_OPTIONS,
	DEFAULT_TAA_OPTIONS,
	DEFAULT_VOLUMETRIC_OPTIONS,
} from "../../../pipeline/types";
import type { ICommandEncoder } from "../../ICommandEncoder";
import {
	BufferUsage,
	type IBindingGroup,
	type IComputePipeline,
	type IRenderBuffer,
	type IRenderTexture,
	type IShaderModule,
} from "../../types";
import { loadPostProcessShaderPartComposite } from "../../../shaders/webgpu/shaderSource";
import { ceilDiv, finiteOr } from "../postProcessMath";
import type { WebGPUFrameTargets } from "../WebGPUPostProcessGraph";
import type { WebGPULightingState } from "../types";
import { PostProcessSharedContext } from "./PostProcessSharedContext";
import type {
	WebGPUPostProcessExecuteRequest,
	WebGPUPostProcessExecuteResult,
	WebGPUPostProcessPassDelegate,
	WebGPUPostProcessPassId,
	WebGPUPostProcessSSRExecuteRequest,
	WebGPUPostProcessTAAExecuteRequest,
	WebGPUPostProcessVolumetricExecuteRequest,
} from "./types";

const WORKGROUP_SIZE = 8;
const VOLUMETRIC_LIGHT_STRIDE_FLOATS = 12;
const MAX_VOLUMETRIC_LIGHTS = 65000;

export class TemporalPostProcessDelegate implements WebGPUPostProcessPassDelegate {
	public readonly passIds: readonly WebGPUPostProcessPassId[] = [
		"taa",
		"ssr",
		"volumetric",
	];

	private _shared: PostProcessSharedContext;
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
	private _copyModule: IShaderModule | null = null;
	private _copyPipeline: IComputePipeline | null = null;
	private _hizViewCache = new WeakMap<object, GPUTextureView[]>();
	private _ssrTraceGroupLayout0: GPUBindGroupLayout | null = null;
	private _ssrTracePipelineLayout: GPUPipelineLayout | null = null;
	private _volumetricGroupLayout0: GPUBindGroupLayout | null = null;
	private _volumetricPipelineLayout: GPUPipelineLayout | null = null;
	private _ssrFrameIndex = 0;

	constructor(shared: PostProcessSharedContext) {
		this._shared = shared;
	}

	public invalidateBindings(): void {}

	public onShaderRuntimeChanged(): void {
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
		this._copyModule = null;
		this._copyPipeline = null;
		this._ssrTraceGroupLayout0 = null;
		this._ssrTracePipelineLayout = null;
		this._volumetricGroupLayout0 = null;
		this._volumetricPipelineLayout = null;
	}

	public async warmupHint(hint: string): Promise<boolean> {
		switch (hint) {
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
			case "postprocess:copy":
				await this._ensureCopyResources();
				return true;
			default:
				return false;
		}
	}

	public async execute(
		request: WebGPUPostProcessExecuteRequest
	): Promise<WebGPUPostProcessExecuteResult | null> {
		switch (request.passId) {
			case "taa": {
				const historyUpdated = await this._executeTAA(request);
				return { ran: historyUpdated, historyUpdated };
			}
			case "ssr": {
				const historyUpdated = await this._executeSSR(request);
				return { ran: historyUpdated, historyUpdated };
			}
			case "volumetric": {
				const historyUpdated = await this._executeVolumetric(request);
				return { ran: historyUpdated, historyUpdated };
			}
			default:
				return null;
		}
	}

	private async _executeTAA(
		request: WebGPUPostProcessTAAExecuteRequest
	): Promise<boolean> {
		await this._ensureTAAResources();
		if (!this._shared.sampler || !this._taaPipeline || !this._taaParams) {
			return false;
		}
		const options = request.frameContext.features.taaOptions ?? {};
		const taaTarget =
			request.targets.sceneColor === request.targets.postPong ?
				request.targets.postPing
			:	request.targets.postPong;
		const invW = 1 / Math.max(taaTarget.width, 1);
		const invH = 1 / Math.max(taaTarget.height, 1);
		this._shared.compute.writeBuffer(
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
				request.historyValid ? 1 : 0,
			])
		);
		const binding = this._shared.getCachedBindGroup(
			`taa-${taaTarget === request.targets.postPing ? "ping" : "pong"}`,
			this._taaPipeline,
			[
				{ binding: 0, resource: request.targets.sceneColor },
				{ binding: 1, resource: request.targets.historyRead },
				{ binding: 2, resource: request.targets.gMotionDepth },
				{ binding: 3, resource: request.targets.motionHistoryRead },
				{ binding: 4, resource: this._shared.sampler },
				{ binding: 5, resource: this._taaParams },
				{ binding: 6, resource: taaTarget },
				{ binding: 7, resource: request.targets.historyWrite },
			],
			"WebGPUTAA_Binding"
		);
		request.encoder.beginComputePass({ label: "WebGPUTAA" });
		request.encoder.setComputePipeline(this._taaPipeline);
		request.encoder.setBindingGroup(0, binding);
		request.encoder.dispatchWorkgroups(
			ceilDiv(taaTarget.width, WORKGROUP_SIZE),
			ceilDiv(taaTarget.height, WORKGROUP_SIZE),
			1
		);
		request.encoder.endComputePass();
		request.targets.sceneColor = taaTarget;
		return true;
	}

	private async _executeSSR(
		request: WebGPUPostProcessSSRExecuteRequest
	): Promise<boolean> {
		if (request.frameContext.camera.type === CameraType.Orthographic) {
			this._shared.warn(
				"webgpu-ssr-orthographic-disabled",
				"WebGPU SSR is disabled for OrthographicCamera in v1"
			);
			return false;
		}
		await this._ensureSSRResources();
		if (
			!this._shared.sampler ||
			!this._hizInitPipeline ||
			!this._hizReducePipeline ||
			!this._ssrTracePipeline ||
			!this._ssrComposePipeline ||
			!this._ssrTraceParams ||
			!this._ssrComposeParams
		) {
			return false;
		}
		const options = request.frameContext.features.ssrOptions ?? {};
		const hiZMips = this._getHiZMipViews(request.targets.hiZ);
		if (!this._buildHiZ(request.encoder, request.targets, hiZMips)) {
			return false;
		}
		this._ssrFrameIndex = (this._ssrFrameIndex + 1) % 1024;
		this._shared.compute.writeBuffer(
			this._ssrTraceParams,
			new Float32Array([
				1 / Math.max(request.targets.ssrRaw.width, 1),
				1 / Math.max(request.targets.ssrRaw.height, 1),
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
				request.historyValid ? 1 : 0,
				0.02,
				this._ssrFrameIndex,
				0,
			])
		);
		let binding = this._shared.getCachedBindGroup(
			"ssr-trace",
			this._ssrTracePipeline,
			[
				{ binding: 0, resource: request.targets.sceneColor },
				{ binding: 1, resource: request.targets.gNormalRoughMetal },
				{ binding: 2, resource: request.targets.gMotionDepth },
				{ binding: 3, resource: request.targets.hiZ },
				{ binding: 4, resource: request.targets.ssrHistoryRead },
				{ binding: 5, resource: request.targets.motionHistoryRead },
				{ binding: 6, resource: this._shared.sampler },
				{ binding: 7, resource: this._ssrTraceParams },
				{ binding: 8, resource: request.targets.ssrRaw },
			],
			"WebGPUSSR_TraceBinding"
		);
		request.encoder.beginComputePass({ label: "WebGPUSSR_TraceTemporal" });
		request.encoder.setComputePipeline(this._ssrTracePipeline);
		request.encoder.setBindingGroup(0, binding);
		request.encoder.setBindingGroup(1, request.frameBinding);
		request.encoder.dispatchWorkgroups(
			ceilDiv(request.targets.ssrRaw.width, WORKGROUP_SIZE),
			ceilDiv(request.targets.ssrRaw.height, WORKGROUP_SIZE),
			1
		);
		request.encoder.endComputePass();
		await this._copyTexture(
			request.encoder,
			request.targets.ssrRaw,
			request.targets.ssrHistoryWrite
		);
		const composeTarget =
			request.targets.sceneColor === request.targets.postPing ?
				request.targets.postPong
			:	request.targets.postPing;
		this._shared.compute.writeBuffer(
			this._ssrComposeParams,
			new Float32Array([
				1 / Math.max(composeTarget.width, 1),
				1 / Math.max(composeTarget.height, 1),
				0,
				0,
			])
		);
		binding = this._shared.getCachedBindGroup(
			`ssr-compose-${composeTarget === request.targets.postPing ? "ping" : "pong"}`,
			this._ssrComposePipeline,
			[
				{ binding: 0, resource: request.targets.sceneColor },
				{ binding: 1, resource: request.targets.ssrRaw },
				{ binding: 2, resource: request.targets.gMotionDepth },
				{ binding: 3, resource: this._shared.sampler },
				{ binding: 4, resource: this._ssrComposeParams },
				{ binding: 5, resource: composeTarget },
			],
			"WebGPUSSR_ComposeBinding"
		);
		request.encoder.beginComputePass({ label: "WebGPUSSR_Compose" });
		request.encoder.setComputePipeline(this._ssrComposePipeline);
		request.encoder.setBindingGroup(0, binding);
		request.encoder.dispatchWorkgroups(
			ceilDiv(composeTarget.width, WORKGROUP_SIZE),
			ceilDiv(composeTarget.height, WORKGROUP_SIZE),
			1
		);
		request.encoder.endComputePass();
		request.targets.sceneColor = composeTarget;
		return true;
	}

	private async _executeVolumetric(
		request: WebGPUPostProcessVolumetricExecuteRequest
	): Promise<boolean> {
		if (request.frameContext.camera.type === CameraType.Orthographic) {
			this._shared.warn(
				"webgpu-volumetric-orthographic-disabled",
				"WebGPU volumetric lighting is disabled for OrthographicCamera in v1"
			);
			return false;
		}
		await this._ensureVolumetricResources();
		const lightCount = this._updateVolumetricLightBuffer(request.lightingState);
		if (
			!this._shared.sampler ||
			!this._volumetricPipeline ||
			!this._volumetricParams ||
			!this._volumetricLightBuffer
		) {
			return false;
		}
		const hiZMips = this._getHiZMipViews(request.targets.hiZ);
		if (!this._buildHiZ(request.encoder, request.targets, hiZMips)) {
			return false;
		}

		const options = request.frameContext.features.volumetricOptions ?? {};
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

		this._shared.compute.writeBuffer(
			this._volumetricParams,
			new Float32Array([
				1 / Math.max(request.targets.sceneColor.width, 1),
				1 / Math.max(request.targets.sceneColor.height, 1),
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
				request.historyValid ? 1 : 0,
				lightCount,
				restirCandidates,
				restirTemporalWeight,
				restirScaleClamp,
				this._volumetricFrameIndex,
			])
		);

		const target =
			request.targets.sceneColor === request.targets.postPong ?
				request.targets.postPing
			:	request.targets.postPong;
		const binding = this._shared.getCachedBindGroup(
			`volumetric-${target === request.targets.postPing ? "ping" : "pong"}`,
			this._volumetricPipeline,
			[
				{ binding: 0, resource: request.targets.sceneColor },
				{ binding: 1, resource: request.targets.gMotionDepth },
				{ binding: 2, resource: request.targets.hiZ },
				{ binding: 3, resource: request.targets.volumetricHistoryRead },
				{ binding: 4, resource: request.targets.motionHistoryRead },
				{ binding: 5, resource: this._shared.sampler },
				{ binding: 6, resource: this._volumetricParams },
				{ binding: 7, resource: target },
				{ binding: 8, resource: request.targets.volumetricHistoryWrite },
				{
					binding: 9,
					resource: request.targets.volumetricReservoirHistoryRead,
				},
				{
					binding: 10,
					resource: request.targets.volumetricReservoirHistoryWrite,
				},
				{ binding: 11, resource: this._volumetricLightBuffer },
			],
			"WebGPUVolumetric_Binding"
		);
		request.encoder.beginComputePass({ label: "WebGPUVolumetric" });
		request.encoder.setComputePipeline(this._volumetricPipeline);
		request.encoder.setBindingGroup(0, binding);
		request.encoder.setBindingGroup(1, request.frameBinding);
		request.encoder.dispatchWorkgroups(
			ceilDiv(target.width, WORKGROUP_SIZE),
			ceilDiv(target.height, WORKGROUP_SIZE),
			1
		);
		request.encoder.endComputePass();
		request.targets.sceneColor = target;
		return true;
	}

	private _updateVolumetricLightBuffer(
		lightingState: WebGPULightingState | null
	): number {
		const sourceLights = lightingState?.volumetricLights ?? [];
		const clampedLightCount = Math.min(sourceLights.length, MAX_VOLUMETRIC_LIGHTS);
		if (sourceLights.length > MAX_VOLUMETRIC_LIGHTS) {
			this._shared.warn(
				"webgpu-volumetric-light-count-clamped",
				`WebGPU volumetric ReSTIR clamps light count to ${MAX_VOLUMETRIC_LIGHTS}; extra lights are skipped`
			);
		}

		this._ensureVolumetricLightBufferCapacity(clampedLightCount);
		if (!this._volumetricLightBuffer) {
			return 0;
		}

		const packedCount = Math.max(1, clampedLightCount);
		const packed = new Float32Array(packedCount * VOLUMETRIC_LIGHT_STRIDE_FLOATS);

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

		this._shared.compute.writeBuffer(this._volumetricLightBuffer, packed);
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
		this._volumetricLightBuffer = this._shared.compute.createBuffer({
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
		if (src === dst) {
			return;
		}
		await this._ensureCopyResources();
		if (!this._copyPipeline) {
			return;
		}
		const binding = this._shared.getCachedBindGroup(
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
		if (!this._hizInitPipeline || !this._hizReducePipeline) {
			return false;
		}
		if (hiZMips.length === 0) {
			return false;
		}

		let binding = this._shared.getCachedBindGroup(
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
			binding = this._shared.getCachedBindGroup(
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

	private async _ensureTAAResources(): Promise<void> {
		await this._shared.ensureCommonResources();
		if (!this._taaModule) {
			const shader = await loadPostProcessShaderPartComposite("taa");
			this._taaModule = await this._shared.compute.createShaderModule({
				label: "WebGPUTAAShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!this._taaPipeline) {
			this._taaPipeline = this._shared.compute.createComputePipeline({
				label: "WebGPUTAAPipeline",
				compute: { module: this._taaModule, entryPoint: "csMain" },
			});
		}
		if (!this._taaParams) {
			this._taaParams = this._shared.compute.createBuffer({
				label: "WebGPUTAAParams",
				size: 8 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
		}
	}

	private async _ensureHiZResources(): Promise<void> {
		await this._shared.ensureCommonResources();
		if (!this._hizModule) {
			const shader = await loadPostProcessShaderPartComposite("hiz");
			this._hizModule = await this._shared.compute.createShaderModule({
				label: "WebGPUHiZShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!this._hizInitPipeline) {
			this._hizInitPipeline = this._shared.compute.createComputePipeline({
				label: "WebGPUHiZInitPipeline",
				compute: { module: this._hizModule, entryPoint: "csInit" },
			});
		}
		if (!this._hizReducePipeline) {
			this._hizReducePipeline = this._shared.compute.createComputePipeline({
				label: "WebGPUHiZReducePipeline",
				compute: { module: this._hizModule, entryPoint: "csReduce" },
			});
		}
	}

	private async _ensureSSRResources(): Promise<void> {
		await this._ensureHiZResources();
		if (!this._ssrModule) {
			const shader = await loadPostProcessShaderPartComposite("ssr");
			this._ssrModule = await this._shared.compute.createShaderModule({
				label: "WebGPUSSRShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!this._ssrTracePipeline) {
			if (this._shared.frameBindGroupLayout) {
				this._ssrTraceGroupLayout0 = this._shared.compute.createBindGroupLayout({
					label: "WebGPUSSRTrace_GroupLayout0",
					entries: [
						{ binding: 0, visibility: GPUShaderStage.COMPUTE, texture: {} },
						{ binding: 1, visibility: GPUShaderStage.COMPUTE, texture: {} },
						{ binding: 2, visibility: GPUShaderStage.COMPUTE, texture: {} },
						{ binding: 3, visibility: GPUShaderStage.COMPUTE, texture: {} },
						{ binding: 4, visibility: GPUShaderStage.COMPUTE, texture: {} },
						{ binding: 5, visibility: GPUShaderStage.COMPUTE, texture: {} },
						{ binding: 6, visibility: GPUShaderStage.COMPUTE, sampler: {} },
						{
							binding: 7,
							visibility: GPUShaderStage.COMPUTE,
							buffer: { type: "uniform" },
						},
						{
							binding: 8,
							visibility: GPUShaderStage.COMPUTE,
							storageTexture: { format: "rgba16float", access: "write-only" },
						},
					],
				});
				this._ssrTracePipelineLayout = this._shared.compute.createPipelineLayout({
					label: "WebGPUSSRTrace_PipelineLayout",
					bindGroupLayouts: [
						this._ssrTraceGroupLayout0,
						this._shared.frameBindGroupLayout,
					],
				});
				this._ssrTracePipeline = this._shared.compute.createComputePipeline({
					label: "WebGPUSSRTracePipeline",
					layout: this._ssrTracePipelineLayout,
					compute: { module: this._ssrModule, entryPoint: "csTrace" },
				});
			} else {
				this._ssrTracePipeline = this._shared.compute.createComputePipeline({
					label: "WebGPUSSRTracePipeline",
					compute: { module: this._ssrModule, entryPoint: "csTrace" },
				});
			}
		}
		if (!this._ssrComposePipeline) {
			this._ssrComposePipeline = this._shared.compute.createComputePipeline({
				label: "WebGPUSSRComposePipeline",
				compute: { module: this._ssrModule, entryPoint: "csCompose" },
			});
		}
		if (!this._ssrTraceParams) {
			this._ssrTraceParams = this._shared.compute.createBuffer({
				label: "WebGPUSSRTraceParams",
				size: 16 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
		}
		if (!this._ssrComposeParams) {
			this._ssrComposeParams = this._shared.compute.createBuffer({
				label: "WebGPUSSRComposeParams",
				size: 4 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
		}
	}

	private async _ensureVolumetricResources(): Promise<void> {
		await this._ensureHiZResources();
		if (!this._volumetricModule) {
			const shader = await loadPostProcessShaderPartComposite("volumetric");
			this._volumetricModule = await this._shared.compute.createShaderModule({
				label: "WebGPUVolumetricShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!this._volumetricPipeline) {
			if (this._shared.frameBindGroupLayout) {
				this._volumetricGroupLayout0 = this._shared.compute.createBindGroupLayout({
					label: "WebGPUVolumetric_GroupLayout0",
					entries: [
						{ binding: 0, visibility: GPUShaderStage.COMPUTE, texture: {} },
						{ binding: 1, visibility: GPUShaderStage.COMPUTE, texture: {} },
						{ binding: 2, visibility: GPUShaderStage.COMPUTE, texture: {} },
						{ binding: 3, visibility: GPUShaderStage.COMPUTE, texture: {} },
						{ binding: 4, visibility: GPUShaderStage.COMPUTE, texture: {} },
						{ binding: 5, visibility: GPUShaderStage.COMPUTE, sampler: {} },
						{
							binding: 6,
							visibility: GPUShaderStage.COMPUTE,
							buffer: { type: "uniform" },
						},
						{
							binding: 7,
							visibility: GPUShaderStage.COMPUTE,
							storageTexture: { format: "rgba16float", access: "write-only" },
						},
						{
							binding: 8,
							visibility: GPUShaderStage.COMPUTE,
							storageTexture: { format: "rgba16float", access: "write-only" },
						},
						{ binding: 9, visibility: GPUShaderStage.COMPUTE, texture: {} },
						{
							binding: 10,
							visibility: GPUShaderStage.COMPUTE,
							storageTexture: { format: "rgba16float", access: "write-only" },
						},
						{
							binding: 11,
							visibility: GPUShaderStage.COMPUTE,
							buffer: { type: "read-only-storage" },
						},
					],
				});
				this._volumetricPipelineLayout = this._shared.compute.createPipelineLayout(
					{
						label: "WebGPUVolumetric_PipelineLayout",
						bindGroupLayouts: [
							this._volumetricGroupLayout0,
							this._shared.frameBindGroupLayout,
						],
					}
				);
				this._volumetricPipeline = this._shared.compute.createComputePipeline({
					label: "WebGPUVolumetricPipeline",
					layout: this._volumetricPipelineLayout,
					compute: { module: this._volumetricModule, entryPoint: "csMain" },
				});
			} else {
				this._volumetricPipeline = this._shared.compute.createComputePipeline({
					label: "WebGPUVolumetricPipeline",
					compute: { module: this._volumetricModule, entryPoint: "csMain" },
				});
			}
		}
		if (!this._volumetricParams) {
			this._volumetricParams = this._shared.compute.createBuffer({
				label: "WebGPUVolumetricParams",
				size: 20 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
		}
	}

	private async _ensureCopyResources(): Promise<void> {
		if (!this._copyModule) {
			const shader = await loadPostProcessShaderPartComposite("copy");
			this._copyModule = await this._shared.compute.createShaderModule({
				label: "WebGPUCopyShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!this._copyPipeline) {
			this._copyPipeline = this._shared.compute.createComputePipeline({
				label: "WebGPUCopyPipeline",
				compute: { module: this._copyModule, entryPoint: "csMain" },
			});
		}
	}

	private _getHiZMipViews(texture: IRenderTexture): GPUTextureView[] {
		const cached = this._hizViewCache.get(texture as object);
		if (cached) {
			return cached;
		}
		const mipCount =
			Math.floor(Math.log2(Math.max(texture.width, texture.height))) + 1;
		const views: GPUTextureView[] = [];
		for (let i = 0; i < mipCount; i++) {
			views.push(
				this._shared.compute.createTextureView(texture, {
					baseMipLevel: i,
					mipLevelCount: 1,
				})
			);
		}
		this._hizViewCache.set(texture as object, views);
		return views;
	}
}
