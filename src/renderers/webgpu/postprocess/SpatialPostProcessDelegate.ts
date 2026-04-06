import { CameraType } from "../../../cameras/Camera";
import type { FrameContext } from "../../../pipeline/types";
import {
	DEFAULT_SSAO_OPTIONS,
	DEFAULT_SSGI_OPTIONS,
} from "../../../pipeline/types";
import type { ICommandEncoder } from "../../ICommandEncoder";
import {
	BufferUsage,
	type IComputePipeline,
	type IRenderBuffer,
	type IRenderTexture,
	type IShaderModule,
} from "../../types";
import { clamp } from "../../../maths/Common";
import { loadPostProcessShaderPartComposite } from "../../../shaders/webgpu/shaderSource";
import { ceilDiv, finiteOr } from "../postProcessMath";
import type { WebGPUFrameTargets } from "../WebGPUPostProcessGraph";
import { PostProcessSharedContext } from "./PostProcessSharedContext";
import type {
	WebGPUPostProcessExecuteRequest,
	WebGPUPostProcessExecuteResult,
	WebGPUPostProcessPassDelegate,
	WebGPUPostProcessPassId,
} from "./types";

const WORKGROUP_SIZE = 8;

export class SpatialPostProcessDelegate implements WebGPUPostProcessPassDelegate {
	public readonly passIds: readonly WebGPUPostProcessPassId[] = [
		"ssao",
		"ssgi",
	];

	private _shared: PostProcessSharedContext;
	private _ssaoModule: IShaderModule | null = null;
	private _ssaoRawPipeline: IComputePipeline | null = null;
	private _ssaoBlurPipeline: IComputePipeline | null = null;
	private _ssaoCombinePipeline: IComputePipeline | null = null;
	private _ssaoParams: IRenderBuffer | null = null;
	private _ssgiModule: IShaderModule | null = null;
	private _ssgiPipeline: IComputePipeline | null = null;
	private _ssgiParams: IRenderBuffer | null = null;
	private _ssaoFrameIndex = 0;

	constructor(shared: PostProcessSharedContext) {
		this._shared = shared;
	}

	public invalidateBindings(): void {}

	public onShaderRuntimeChanged(): void {
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
	}

	public async warmupHint(hint: string): Promise<boolean> {
		switch (hint) {
			case "postprocess:ssao":
				await this._ensureSSAOResources();
				return true;
			case "postprocess:ssgi":
				await this._ensureSSGIResources();
				return true;
			default:
				return false;
		}
	}

	public async execute(
		request: WebGPUPostProcessExecuteRequest
	): Promise<WebGPUPostProcessExecuteResult | null> {
		switch (request.passId) {
			case "ssao":
				await this._executeSSAO(
					request.encoder,
					request.targets,
					request.frameContext
				);
				return { ran: true };
			case "ssgi":
				await this._executeSSGI(
					request.encoder,
					request.targets,
					request.frameContext
				);
				return { ran: true };
			default:
				return null;
		}
	}

	private async _executeSSAO(
		encoder: ICommandEncoder,
		targets: WebGPUFrameTargets,
		frameContext: FrameContext
	): Promise<void> {
		await this._ensureSSAOResources();
		if (
			!this._shared.sampler ||
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
		const bias = Math.max(1e-4, finiteOr(options.bias, DEFAULT_SSAO_OPTIONS.bias));
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
			this._shared.compute.writeBuffer(
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
		let binding = this._shared.getCachedBindGroup(
			"ssao-raw",
			this._ssaoRawPipeline,
			[
				{ binding: 0, resource: targets.gNormalRoughMetal },
				{ binding: 1, resource: targets.gMotionDepth },
				{ binding: 2, resource: this._shared.sampler },
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

		binding = this._shared.getCachedBindGroup(
			"ssao-blur-h",
			this._ssaoBlurPipeline,
			[
				{ binding: 0, resource: targets.aoRaw },
				{ binding: 1, resource: targets.gMotionDepth },
				{ binding: 2, resource: this._shared.sampler },
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
		binding = this._shared.getCachedBindGroup(
			"ssao-blur-v",
			this._ssaoBlurPipeline,
			[
				{ binding: 0, resource: targets.aoBlur },
				{ binding: 1, resource: targets.gMotionDepth },
				{ binding: 2, resource: this._shared.sampler },
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
		binding = this._shared.getCachedBindGroup(
			`ssao-combine-${combineTarget === targets.postPing ? "ping" : "pong"}`,
			this._ssaoCombinePipeline,
			[
				{ binding: 0, resource: targets.sceneColor },
				{ binding: 1, resource: targets.aoRaw },
				{ binding: 2, resource: this._shared.sampler },
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

	private async _executeSSGI(
		encoder: ICommandEncoder,
		targets: WebGPUFrameTargets,
		frameContext: FrameContext
	): Promise<void> {
		await this._ensureSSGIResources();
		if (!this._shared.sampler || !this._ssgiPipeline || !this._ssgiParams) {
			return;
		}
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
		this._shared.compute.writeBuffer(
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
		const binding = this._shared.getCachedBindGroup(
			`ssgi-${target === targets.postPing ? "ping" : "pong"}`,
			this._ssgiPipeline,
			[
				{ binding: 0, resource: targets.sceneColor },
				{ binding: 1, resource: targets.gAlbedoAlpha },
				{ binding: 2, resource: targets.gNormalRoughMetal },
				{ binding: 3, resource: targets.gMotionDepth },
				{ binding: 4, resource: this._shared.sampler },
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

	private async _ensureSSAOResources(): Promise<void> {
		await this._shared.ensureCommonResources();
		if (!this._ssaoModule) {
			const shader = await loadPostProcessShaderPartComposite("ssao");
			this._ssaoModule = await this._shared.compute.createShaderModule({
				label: "WebGPUSSAOShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!this._ssaoRawPipeline) {
			this._ssaoRawPipeline = this._shared.compute.createComputePipeline({
				label: "WebGPUSSAORawPipeline",
				compute: { module: this._ssaoModule, entryPoint: "csRaw" },
			});
		}
		if (!this._ssaoBlurPipeline) {
			this._ssaoBlurPipeline = this._shared.compute.createComputePipeline({
				label: "WebGPUSSAOBlurPipeline",
				compute: { module: this._ssaoModule, entryPoint: "csBlur" },
			});
		}
		if (!this._ssaoCombinePipeline) {
			this._ssaoCombinePipeline = this._shared.compute.createComputePipeline({
				label: "WebGPUSSAOCombinePipeline",
				compute: { module: this._ssaoModule, entryPoint: "csCombine" },
			});
		}
		if (!this._ssaoParams) {
			this._ssaoParams = this._shared.compute.createBuffer({
				label: "WebGPUSSAOParams",
				size: 16 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
		}
	}

	private async _ensureSSGIResources(): Promise<void> {
		await this._shared.ensureCommonResources();
		if (!this._ssgiModule) {
			const shader = await loadPostProcessShaderPartComposite("ssgi");
			this._ssgiModule = await this._shared.compute.createShaderModule({
				label: "WebGPUSSGIShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!this._ssgiPipeline) {
			this._ssgiPipeline = this._shared.compute.createComputePipeline({
				label: "WebGPUSSGIPipeline",
				compute: { module: this._ssgiModule, entryPoint: "csMain" },
			});
		}
		if (!this._ssgiParams) {
			this._ssgiParams = this._shared.compute.createBuffer({
				label: "WebGPUSSGIParams",
				size: 8 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
		}
	}
}
