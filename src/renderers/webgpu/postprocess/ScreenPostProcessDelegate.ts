import type { FrameContext } from "../../../pipeline/types";
import {
	DEFAULT_BLOOM_OPTIONS,
	DEFAULT_DOF_OPTIONS,
	DEFAULT_FOG_OPTIONS,
	INTERACTION_TRANSIENT_STATE_KEY,
	DEFAULT_MOTION_BLUR_OPTIONS,
	type FogOptions,
	type InteractionTransientState,
} from "../../../pipeline/types";
import type { ICommandEncoder } from "../../ICommandEncoder";
import {
	BufferUsage,
	TextureFormat,
	TextureUsage,
	type IComputePipeline,
	type IRenderBuffer,
	type IRenderTexture,
	type IShaderModule,
} from "../../types";
import { loadPostProcessShaderPartComposite } from "../../../shaders/webgpu/shaderSource";
import { clamp, sRGBToLinear } from "../../../maths/Common";
import {
	MAX_INTERACTION_OUTLINE_CIRCLES,
	collectProjectedOutlineCircles,
} from "../../../interaction/outlineProjection";
import { getInteractionOutlineShapeCode } from "../../../interaction/outlineShape";
import {
	FXAA_EDGE_THRESHOLD_MIN,
	FXAA_EDGE_THRESHOLD_MULTIPLIER,
	FXAA_SUBPIX_QUALITY,
} from "../../postProcessConstants";
import { ceilDiv, finiteOr } from "../postProcessMath";
import type { WebGPUFrameTargets } from "../WebGPUPostProcessGraph";
import { PostProcessSharedContext } from "./PostProcessSharedContext";
import type {
	WebGPUPostProcessExecuteRequest,
	WebGPUPostProcessExecuteResult,
	WebGPUPostProcessInteractionOutlineExecuteRequest,
	WebGPUPostProcessPassDelegate,
	WebGPUPostProcessPassId,
} from "./types";

const WORKGROUP_SIZE = 8;
const INTERACTION_OUTLINE_HEADER_FLOATS = 16;
const INTERACTION_OUTLINE_PARAM_FLOATS =
	INTERACTION_OUTLINE_HEADER_FLOATS + MAX_INTERACTION_OUTLINE_CIRCLES * 4;

export class ScreenPostProcessDelegate implements WebGPUPostProcessPassDelegate {
	public readonly passIds: readonly WebGPUPostProcessPassId[] = [
		"fog",
		"motion-blur",
		"dof",
		"bloom",
		"fxaa",
		"interaction-outline",
	];

	private _shared: PostProcessSharedContext;
	private _fogModule: IShaderModule | null = null;
	private _fogPipeline: IComputePipeline | null = null;
	private _fogParams: IRenderBuffer | null = null;
	private _fogParamData = new Float32Array(8);
	private _fogParamUploaded = false;
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

	constructor(shared: PostProcessSharedContext) {
		this._shared = shared;
	}

	public invalidateBindings(): void {
		this._destroyBloomMipTextures();
	}

	public onShaderRuntimeChanged(): void {
		this._fogModule = null;
		this._fogPipeline = null;
		this._fogParams?.destroy();
		this._fogParams = null;
		this._fogParamUploaded = false;
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
	}

	public async warmupHint(hint: string): Promise<boolean> {
		switch (hint) {
			case "postprocess:fog":
				await this._ensureFogResources();
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
			default:
				return false;
		}
	}

	public async execute(
		request: WebGPUPostProcessExecuteRequest
	): Promise<WebGPUPostProcessExecuteResult | null> {
		switch (request.passId) {
			case "fog":
				await this._executeFog(
					request.encoder,
					request.targets,
					request.frameContext
				);
				return { ran: true };
			case "motion-blur":
				await this._executeMotionBlur(
					request.encoder,
					request.targets,
					request.frameContext
				);
				return { ran: true };
			case "dof":
				await this._executeDOF(
					request.encoder,
					request.targets,
					request.frameContext
				);
				return { ran: true };
			case "bloom":
				await this._executeBloom(
					request.encoder,
					request.targets,
					request.frameContext
				);
				return { ran: true };
			case "fxaa":
				await this._executeFXAA(request.encoder, request.targets);
				return { ran: true };
			case "interaction-outline":
				await this._executeInteractionOutline(
					request as WebGPUPostProcessInteractionOutlineExecuteRequest
				);
				return { ran: true };
			default:
				return null;
		}
	}

	private async _executeFog(
		encoder: ICommandEncoder,
		targets: WebGPUFrameTargets,
		frameContext: FrameContext
	): Promise<void> {
		await this._ensureFogResources();
		if (!this._shared.sampler || !this._fogPipeline || !this._fogParams) {
			return;
		}
		const target =
			targets.sceneColor === targets.postPong ? targets.postPing : targets.postPong;
		this._uploadFogParams(frameContext.features.fogOptions);
		const binding = this._shared.getCachedBindGroup(
			`fog-${target === targets.postPing ? "ping" : "pong"}`,
			this._fogPipeline,
			[
				{ binding: 0, resource: targets.sceneColor },
				{ binding: 1, resource: targets.gMotionDepth },
				{ binding: 2, resource: this._shared.sampler },
				{ binding: 3, resource: this._fogParams },
				{ binding: 4, resource: target },
			],
			"WebGPUFog_Binding"
		);
		encoder.beginComputePass({ label: "WebGPUFog" });
		encoder.setComputePipeline(this._fogPipeline);
		encoder.setBindingGroup(0, binding);
		encoder.dispatchWorkgroups(
			ceilDiv(target.width, WORKGROUP_SIZE),
			ceilDiv(target.height, WORKGROUP_SIZE),
			1
		);
		encoder.endComputePass();
		targets.sceneColor = target;
	}

	private async _executeMotionBlur(
		encoder: ICommandEncoder,
		targets: WebGPUFrameTargets,
		frameContext: FrameContext
	): Promise<void> {
		await this._ensureMotionBlurResources();
		if (
			!this._shared.sampler ||
			!this._motionBlurPipeline ||
			!this._motionBlurParams
		) {
			return;
		}
		const options = frameContext.features.motionBlurOptions ?? {};
		const target =
			targets.sceneColor === targets.postPong ? targets.postPing : targets.postPong;
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
		const binding = this._shared.getCachedBindGroup(
			`motion-blur-${target === targets.postPing ? "ping" : "pong"}`,
			this._motionBlurPipeline,
			[
				{ binding: 0, resource: targets.sceneColor },
				{ binding: 1, resource: targets.gMotionDepth },
				{ binding: 2, resource: this._shared.sampler },
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

	private async _executeDOF(
		encoder: ICommandEncoder,
		targets: WebGPUFrameTargets,
		frameContext: FrameContext
	): Promise<void> {
		await this._ensureDOFResources();
		if (!this._shared.sampler || !this._dofPipeline || !this._dofParams) {
			return;
		}
		const options = frameContext.features.dofOptions ?? {};
		const target =
			targets.sceneColor === targets.postPong ? targets.postPing : targets.postPong;
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
			finiteOr(options.highlightThreshold, DEFAULT_DOF_OPTIONS.highlightThreshold)
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
		this._shared.compute.writeBuffer(
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
		const binding = this._shared.getCachedBindGroup(
			`dof-${target === targets.postPing ? "ping" : "pong"}`,
			this._dofPipeline,
			[
				{ binding: 0, resource: targets.sceneColor },
				{ binding: 1, resource: targets.gMotionDepth },
				{ binding: 2, resource: this._shared.sampler },
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

	private async _executeBloom(
		encoder: ICommandEncoder,
		targets: WebGPUFrameTargets,
		frameContext: FrameContext
	): Promise<void> {
		await this._ensureBloomResources();
		if (
			!this._shared.sampler ||
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
			Math.round(finiteOr(options.mipPasses, DEFAULT_BLOOM_OPTIONS.mipPasses)),
			1,
			8
		);

		const srcW = targets.sceneColor.width;
		const srcH = targets.sceneColor.height;
		this._ensureBloomMipTextures(srcW, srcH, requestedMips);
		const mipCount = this._bloomMipCount;
		if (mipCount === 0) {
			return;
		}
		const mips = this._bloomMipTextures;
		const dsMipIndex = mipCount - 1;
		const dsDst = mips[dsMipIndex][0];
		const srcInvW = 1 / Math.max(srcW, 1);
		const srcInvH = 1 / Math.max(srcH, 1);
		this._shared.compute.writeBuffer(
			this._bloomDownsampleParams,
			new Float32Array([srcInvW, srcInvH, threshold, softKnee])
		);
		let binding = this._shared.getCachedBindGroup(
			"bloom-ds-0",
			this._bloomDownsamplePipeline,
			[
				{ binding: 0, resource: targets.sceneColor },
				{ binding: 1, resource: this._shared.sampler },
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

		for (let i = dsMipIndex - 1; i >= 0; i--) {
			const src = mips[i + 1][0];
			const dst = mips[i][0];
			const dsInvW = 1 / Math.max(src.width, 1);
			const dsInvH = 1 / Math.max(src.height, 1);
			this._shared.compute.writeBuffer(
				this._bloomDownsampleParams,
				new Float32Array([dsInvW, dsInvH, -1.0, 1e-4])
			);
			binding = this._shared.getCachedBindGroup(
				`bloom-ds-${dsMipIndex - i}`,
				this._bloomDownsamplePipeline,
				[
					{ binding: 0, resource: src },
					{ binding: 1, resource: this._shared.sampler },
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

		for (let i = 0; i < mipCount; i++) {
			const texA = mips[i][0];
			const texB = mips[i][1];
			const invW = 1 / Math.max(texA.width, 1);
			const invH = 1 / Math.max(texA.height, 1);
			this._shared.compute.writeBuffer(
				this._bloomBlurParams,
				new Float32Array([invW, invH, 1, 0])
			);
			binding = this._shared.getCachedBindGroup(
				`bloom-blurH-${i}`,
				this._bloomBlurHPipeline,
				[
					{ binding: 0, resource: texA },
					{ binding: 1, resource: this._shared.sampler },
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
			this._shared.compute.writeBuffer(
				this._bloomBlurParams,
				new Float32Array([invW, invH, 0, 1])
			);
			binding = this._shared.getCachedBindGroup(
				`bloom-blurV-${i}`,
				this._bloomBlurVPipeline,
				[
					{ binding: 0, resource: texB },
					{ binding: 1, resource: this._shared.sampler },
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

		for (let i = 1; i < mipCount; i++) {
			const smallerMip = mips[i - 1][0];
			const currentMip = mips[i][0];
			const dstMip = mips[i][1];
			const invW = 1 / Math.max(smallerMip.width, 1);
			const invH = 1 / Math.max(smallerMip.height, 1);
			this._shared.compute.writeBuffer(
				this._bloomUpsampleParams,
				new Float32Array([invW, invH, filterRadius, 0])
			);
			binding = this._shared.getCachedBindGroup(
				`bloom-up-${i}`,
				this._bloomUpsamplePipeline,
				[
					{ binding: 0, resource: smallerMip },
					{ binding: 1, resource: currentMip },
					{ binding: 2, resource: this._shared.sampler },
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
			mips[i] = [dstMip, currentMip];
		}

		const bloomResult = mips[mipCount - 1][0];
		const target =
			targets.sceneColor === targets.postPong ? targets.postPing : targets.postPong;
		this._shared.compute.writeBuffer(
			this._bloomCompositeParams,
			new Float32Array([
				1 / Math.max(target.width, 1),
				1 / Math.max(target.height, 1),
				intensity,
				0,
			])
		);
		binding = this._shared.getCachedBindGroup(
			`bloom-comp-${target === targets.postPing ? "ping" : "pong"}`,
			this._bloomCompositePipeline,
			[
				{ binding: 0, resource: targets.sceneColor },
				{ binding: 1, resource: bloomResult },
				{ binding: 2, resource: this._shared.sampler },
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

	private async _executeFXAA(
		encoder: ICommandEncoder,
		targets: WebGPUFrameTargets
	): Promise<void> {
		await this._ensureFXAAResources();
		if (!this._shared.sampler || !this._fxaaPipeline || !this._fxaaParams) {
			return;
		}
		const target =
			targets.sceneColor === targets.postPong ? targets.postPing : targets.postPong;
		this._shared.compute.writeBuffer(
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
		const binding = this._shared.getCachedBindGroup(
			`fxaa-${target === targets.postPing ? "ping" : "pong"}`,
			this._fxaaPipeline,
			[
				{ binding: 0, resource: targets.sceneColor },
				{ binding: 1, resource: this._shared.sampler },
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

	private async _executeInteractionOutline(
		request: WebGPUPostProcessInteractionOutlineExecuteRequest
	): Promise<void> {
		const interactionState =
			request.state ??
			(request.frameContext.transient.get(INTERACTION_TRANSIENT_STATE_KEY) as
				| InteractionTransientState
				| null
				| undefined) ??
			null;
		const selectedEntityIds = interactionState?.selectedEntityIds ?? [];
		if (selectedEntityIds.length === 0) {
			return;
		}

		const circles = collectProjectedOutlineCircles(
			request.frameContext,
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
			request.targets.sceneColor === request.targets.postPong ?
				request.targets.postPing
			:	request.targets.postPong;
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
		this._shared.compute.writeBuffer(this._interactionOutlineParams, params);

		const binding = this._shared.getCachedBindGroup(
			`interaction-outline-${target === request.targets.postPing ? "ping" : "pong"}`,
			this._interactionOutlinePipeline,
			[
				{ binding: 0, resource: request.targets.sceneColor },
				{ binding: 2, resource: this._interactionOutlineParams },
				{ binding: 3, resource: target },
			],
			"WebGPUInteractionOutline_Binding"
		);
		request.encoder.beginComputePass({ label: "WebGPUInteractionOutline" });
		request.encoder.setComputePipeline(this._interactionOutlinePipeline);
		request.encoder.setBindingGroup(0, binding);
		request.encoder.dispatchWorkgroups(
			ceilDiv(target.width, WORKGROUP_SIZE),
			ceilDiv(target.height, WORKGROUP_SIZE),
			1
		);
		request.encoder.endComputePass();
		request.targets.sceneColor = target;
	}

	private async _ensureFogResources(): Promise<void> {
		await this._shared.ensureCommonResources();
		if (!this._fogModule) {
			const shader = await loadPostProcessShaderPartComposite("fog");
			this._fogModule = await this._shared.compute.createShaderModule({
				label: "WebGPUFogShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!this._fogPipeline) {
			this._fogPipeline = this._shared.compute.createComputePipeline({
				label: "WebGPUFogPipeline",
				compute: { module: this._fogModule, entryPoint: "csMain" },
			});
		}
		if (!this._fogParams) {
			this._fogParams = this._shared.compute.createBuffer({
				label: "WebGPUFogParams",
				size: 8 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
		}
	}

	private _uploadFogParams(options: FogOptions | undefined): void {
		if (!this._fogParams) {
			return;
		}
		const source = options ?? DEFAULT_FOG_OPTIONS;
		const color = source.color ?? DEFAULT_FOG_OPTIONS.color;
		const start = Math.max(
			0,
			finiteOr(source.start, DEFAULT_FOG_OPTIONS.start)
		);
		const end = Math.max(
			start + 1e-4,
			finiteOr(source.end, DEFAULT_FOG_OPTIONS.end)
		);
		const density = Math.max(
			0,
			finiteOr(source.density, DEFAULT_FOG_OPTIONS.density)
		);
		const strength = Math.max(
			0,
			finiteOr(source.strength, DEFAULT_FOG_OPTIONS.strength)
		);

		const data = this._fogParamData;
		let changed = !this._fogParamUploaded;
		changed =
			this._setParamIfChanged(data, 0, this._resolveFogMode(source.mode)) ||
			changed;
		changed = this._setParamIfChanged(data, 1, start) || changed;
		changed = this._setParamIfChanged(data, 2, end) || changed;
		changed = this._setParamIfChanged(data, 3, density) || changed;
		changed = this._setParamIfChanged(data, 4, clamp(finiteOr(color[0], DEFAULT_FOG_OPTIONS.color[0]), 0, 1)) || changed;
		changed = this._setParamIfChanged(data, 5, clamp(finiteOr(color[1], DEFAULT_FOG_OPTIONS.color[1]), 0, 1)) || changed;
		changed = this._setParamIfChanged(data, 6, clamp(finiteOr(color[2], DEFAULT_FOG_OPTIONS.color[2]), 0, 1)) || changed;
		changed = this._setParamIfChanged(data, 7, strength) || changed;
		if (!changed) {
			return;
		}
		this._shared.compute.writeBuffer(this._fogParams, data);
		this._fogParamUploaded = true;
	}

	private _resolveFogMode(mode: FogOptions["mode"] | undefined): number {
		switch (mode) {
			case "exp":
				return 1;
			case "exp2":
				return 2;
			default:
				return 0;
		}
	}

	private async _ensureMotionBlurResources(): Promise<void> {
		await this._shared.ensureCommonResources();
		if (!this._motionBlurModule) {
			const shader = await loadPostProcessShaderPartComposite("motionBlur");
			this._motionBlurModule = await this._shared.compute.createShaderModule({
				label: "WebGPUMotionBlurShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!this._motionBlurPipeline) {
			this._motionBlurPipeline = this._shared.compute.createComputePipeline({
				label: "WebGPUMotionBlurPipeline",
				compute: { module: this._motionBlurModule, entryPoint: "csMain" },
			});
		}
		if (!this._motionBlurParams) {
			this._motionBlurParams = this._shared.compute.createBuffer({
				label: "WebGPUMotionBlurParams",
				size: 8 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
		}
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
		changed = this._setParamIfChanged(data, 0, 1 / Math.max(width, 1)) || changed;
		changed = this._setParamIfChanged(data, 1, 1 / Math.max(height, 1)) || changed;
		changed = this._setParamIfChanged(data, 2, shutterScale) || changed;
		changed = this._setParamIfChanged(data, 3, maxSamples) || changed;
		changed = this._setParamIfChanged(data, 4, velocityClamp) || changed;
		changed = this._setParamIfChanged(data, 5, depthReject) || changed;
		changed = this._setParamIfChanged(data, 6, centerWeight) || changed;
		changed = this._setParamIfChanged(data, 7, 0) || changed;
		if (!changed) {
			return;
		}
		this._shared.compute.writeBuffer(this._motionBlurParams, data);
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
		await this._shared.ensureCommonResources();
		if (!this._dofModule) {
			const shader = await loadPostProcessShaderPartComposite("dof");
			this._dofModule = await this._shared.compute.createShaderModule({
				label: "WebGPUDOFShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!this._dofPipeline) {
			this._dofPipeline = this._shared.compute.createComputePipeline({
				label: "WebGPUDOFPipeline",
				compute: { module: this._dofModule, entryPoint: "csMain" },
			});
		}
		if (!this._dofParams) {
			this._dofParams = this._shared.compute.createBuffer({
				label: "WebGPUDOFParams",
				size: 12 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
		}
	}

	private async _ensureBloomResources(): Promise<void> {
		await this._shared.ensureCommonResources();
		if (!this._bloomDownsampleModule) {
			const shader = await loadPostProcessShaderPartComposite("bloomDownsample");
			this._bloomDownsampleModule = await this._shared.compute.createShaderModule({
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
			this._bloomBlurHModule = await this._shared.compute.createShaderModule({
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
			this._bloomBlurVModule = await this._shared.compute.createShaderModule({
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
			this._bloomUpsampleModule = await this._shared.compute.createShaderModule({
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
			this._bloomCompositeModule = await this._shared.compute.createShaderModule({
				label: "WebGPUBloomCompositeShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!this._bloomDownsamplePipeline) {
			this._bloomDownsamplePipeline = this._shared.compute.createComputePipeline({
				label: "WebGPUBloomDownsamplePipeline",
				compute: { module: this._bloomDownsampleModule, entryPoint: "csMain" },
			});
		}
		if (!this._bloomBlurHPipeline) {
			this._bloomBlurHPipeline = this._shared.compute.createComputePipeline({
				label: "WebGPUBloomBlurHPipeline",
				compute: { module: this._bloomBlurHModule, entryPoint: "csMain" },
			});
		}
		if (!this._bloomBlurVPipeline) {
			this._bloomBlurVPipeline = this._shared.compute.createComputePipeline({
				label: "WebGPUBloomBlurVPipeline",
				compute: { module: this._bloomBlurVModule, entryPoint: "csMain" },
			});
		}
		if (!this._bloomUpsamplePipeline) {
			this._bloomUpsamplePipeline = this._shared.compute.createComputePipeline({
				label: "WebGPUBloomUpsamplePipeline",
				compute: { module: this._bloomUpsampleModule, entryPoint: "csMain" },
			});
		}
		if (!this._bloomCompositePipeline) {
			this._bloomCompositePipeline = this._shared.compute.createComputePipeline({
				label: "WebGPUBloomCompositePipeline",
				compute: { module: this._bloomCompositeModule, entryPoint: "csMain" },
			});
		}
		if (!this._bloomDownsampleParams) {
			this._bloomDownsampleParams = this._shared.compute.createBuffer({
				label: "WebGPUBloomDownsampleParams",
				size: 4 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
		}
		if (!this._bloomBlurParams) {
			this._bloomBlurParams = this._shared.compute.createBuffer({
				label: "WebGPUBloomBlurParams",
				size: 4 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
		}
		if (!this._bloomUpsampleParams) {
			this._bloomUpsampleParams = this._shared.compute.createBuffer({
				label: "WebGPUBloomUpsampleParams",
				size: 4 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
		}
		if (!this._bloomCompositeParams) {
			this._bloomCompositeParams = this._shared.compute.createBuffer({
				label: "WebGPUBloomCompositeParams",
				size: 4 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
		}
	}

	private _ensureBloomMipTextures(
		srcWidth: number,
		srcHeight: number,
		requestedMips: number
	): void {
		const halfW = Math.max(1, Math.floor(srcWidth / 2));
		const halfH = Math.max(1, Math.floor(srcHeight / 2));
		const maxPossibleMips = Math.floor(Math.log2(Math.max(halfW, halfH))) + 1;
		const mipCount = Math.min(requestedMips, maxPossibleMips);
		if (
			this._bloomMipWidth === halfW &&
			this._bloomMipHeight === halfH &&
			this._bloomMipCount === mipCount &&
			this._bloomMipTextures.length === mipCount
		) {
			return;
		}
		this._destroyBloomMipTextures();
		this._bloomMipWidth = halfW;
		this._bloomMipHeight = halfH;
		this._bloomMipCount = mipCount;
		for (let i = 0; i < mipCount; i++) {
			const level = mipCount - 1 - i;
			const w = Math.max(1, halfW >> level);
			const h = Math.max(1, halfH >> level);
			const texA = this._shared.compute.createTexture({
				width: w,
				height: h,
				format: TextureFormat.RGBA16Float,
				usage:
					TextureUsage.TextureBinding |
					TextureUsage.StorageBinding |
					TextureUsage.ComputeStorage,
				label: `WebGPUBloomMip${i}_A_${w}x${h}`,
			});
			const texB = this._shared.compute.createTexture({
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

	private _destroyBloomMipTextures(): void {
		for (const [texA, texB] of this._bloomMipTextures) {
			texA.destroy();
			texB.destroy();
		}
		this._bloomMipTextures = [];
		this._bloomMipWidth = 0;
		this._bloomMipHeight = 0;
		this._bloomMipCount = 0;
		this._shared.invalidateBindingsByPrefix("bloom-");
	}

	private async _ensureFXAAResources(): Promise<void> {
		await this._shared.ensureCommonResources();
		if (!this._fxaaModule) {
			const shader = await loadPostProcessShaderPartComposite("fxaa");
			this._fxaaModule = await this._shared.compute.createShaderModule({
				label: "WebGPUFXAAShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!this._fxaaPipeline) {
			this._fxaaPipeline = this._shared.compute.createComputePipeline({
				label: "WebGPUFXAAPipeline",
				compute: { module: this._fxaaModule, entryPoint: "csMain" },
			});
		}
		if (!this._fxaaParams) {
			this._fxaaParams = this._shared.compute.createBuffer({
				label: "WebGPUFXAAParams",
				size: 6 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
		}
	}

	private async _ensureInteractionOutlineResources(): Promise<void> {
		await this._shared.ensureCommonResources();
		if (!this._interactionOutlineModule) {
			const shader = await loadPostProcessShaderPartComposite("interactionOutline");
			this._interactionOutlineModule =
				await this._shared.compute.createShaderModule({
					label: "WebGPUInteractionOutlineShader",
					code: shader.code,
					sourceMap: shader.sourceMap,
					language: "wgsl",
					stage: "compute",
					sourceKind: "postprocess",
				});
		}
		if (!this._interactionOutlinePipeline) {
			this._interactionOutlinePipeline =
				this._shared.compute.createComputePipeline({
					label: "WebGPUInteractionOutlinePipeline",
					compute: {
						module: this._interactionOutlineModule,
						entryPoint: "csMain",
					},
				});
		}
		if (!this._interactionOutlineParams) {
			this._interactionOutlineParams = this._shared.compute.createBuffer({
				label: "WebGPUInteractionOutlineParams",
				size: this._interactionOutlineParamData.byteLength,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
		}
	}
}
