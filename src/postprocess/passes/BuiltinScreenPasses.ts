import {
	INTERACTION_TRANSIENT_STATE_KEY,
} from "../../pipeline/types";
import { clamp, sRGBToLinear } from "../../maths/Common";
import { ceilDiv, finiteOr } from "../../maths/Misc";
import {
	MAX_INTERACTION_OUTLINE_CIRCLES,
	collectProjectedOutlineCircles,
} from "../../interaction/outlineProjection";
import { getInteractionOutlineShapeCode } from "../../interaction/outlineShape";
import type { ICommandEncoder } from "../../renderers/ICommandEncoder";
import {
	BufferUsage,
	type IComputePipeline,
	type IRenderBuffer,
	type IRenderTexture,
	type IShaderModule,
} from "../../renderers/types";
import {
	WEBGPU_INTERACTION_OUTLINE_LAYOUT as INTERACTION_OUTLINE_LAYOUT,
} from "../../renderers/webgpu/bufferLayouts";
import {
	WEBGPU_2D_COMPUTE_WORKGROUP_SIZE as WEBGPU_WORKGROUP_SIZE,
} from "../../renderers/webgpu/constants";
import {
	WEBGPU_PRESENT_POST_PROCESS_CONTEXT_METADATA,
	WEBGPU_SCREEN_POST_PROCESS_CONTEXT_METADATA,
	type WebGPUPostProcessFrameTargets,
} from "../../renderers/webgpu/WebGPUPostProcessContracts";
import type { PostProcessSharedContext } from "../../renderers/webgpu/postprocess/PostProcessSharedContext";
import type { WebGLProgramLibrary } from "../../renderers/webgl/WebGLProgramLibrary";
import {
	DOF_CHROMATIC_ABERRATION_RANGE,
	DOF_DEPTH_CURVE_RANGE,
	DOF_HIGHLIGHT_GAIN_RANGE,
	DOF_MAX_BLUR_RADIUS_RANGE,
	DOF_NEAR_FAR_STRENGTH_RANGE,
	MOTION_BLUR_CENTER_WEIGHT_RANGE,
	MOTION_BLUR_DEPTH_REJECT_RANGE,
	MOTION_BLUR_MAX_SAMPLES_RANGE,
	MOTION_BLUR_SHUTTER_SCALE_RANGE,
	MOTION_BLUR_VELOCITY_CLAMP_RANGE,
} from "../../renderers/webgl/constants";
import { sanitizeFiniteClamped } from "../../renderers/webgl/WebGLFrameMath";
import { loadPostProcessShaderPartComposite } from "../../shaders/webgpu/shaderSource";
import {
	PostProcessPass,
	type PostProcessPassConfig,
	type PostProcessPassResolveRequest,
} from "../PostProcessPass";
import type {
	PostProcessPassImplementation,
	PostProcessPassRequest,
	PostProcessPassRequirements,
	PostProcessPassResult,
} from "../types";
import { SoftwareScreenPassRuntime } from "./SoftwareScreenPassRuntime";

type EmptyOptions = Record<string, never>;

export const MOTION_BLUR_PASS_ID = "motion-blur";
export const DEPTH_OF_FIELD_PASS_ID = "dof";
export const TONE_MAPPING_PASS_ID = "tonemap";
export const COLOR_FILTER_PASS_ID = "color-filter";
export const INTERACTION_OUTLINE_PASS_ID = "interaction-outline";
export const GAMMA_PASS_ID = "gamma";

export interface MotionBlurOptions {
	/** Virtual shutter duration multiplier. Higher values lengthen blur trails. */
	shutterScale?: number;
	/** Maximum samples per pixel along the motion vector. */
	maxSamples?: number;
	/** Maximum normalized screen velocity used for blur length. */
	velocityClamp?: number;
	/** Depth-difference threshold for rejecting samples across silhouettes. */
	depthReject?: number;
	/** Weight of the current pixel in the blur accumulation. */
	centerWeight?: number;
	/** Allows backend-specific experimental motion blur options. */
	[key: string]: unknown;
}

export interface DOFOptions {
	/** Focus plane distance in the depth units consumed by the backend. */
	focusDistance?: number;
	/** Depth range around the focus plane that remains sharp. */
	focusRange?: number;
	/** Blur strength for pixels closer than the focus plane. */
	nearStrength?: number;
	/** Blur strength for pixels farther than the focus plane. */
	farStrength?: number;
	/** Maximum circle-of-confusion blur radius in pixels. */
	maxBlurRadius?: number;
	/** Curve exponent for mapping depth error to blur amount. */
	depthCurve?: number;
	/** Luminance threshold for bokeh highlight boost. */
	highlightThreshold?: number;
	/** Intensity of boosted highlights inside blurred regions. */
	highlightGain?: number;
	/** Color-channel separation amount applied to out-of-focus samples. */
	chromaticAberration?: number;
	/** Allows backend-specific experimental depth-of-field options. */
	[key: string]: unknown;
}

export interface ColorFilterOptions {
	/** Additive brightness shift in normalized color space. */
	brightness?: number;
	/** Saturation multiplier. `0` is grayscale, `1` preserves source color. */
	saturation?: number;
	/** Contrast multiplier around mid-gray. */
	contrast?: number;
	/** Warm/cool color balance shift; positive values warm the image. */
	temperature?: number;
	/** Green/magenta tint shift; positive values bias toward magenta. */
	tint?: number;
	/** Allows backend-specific experimental color-filter options. */
	[key: string]: unknown;
}

export const DEFAULT_MOTION_BLUR_OPTIONS: Required<
	Pick<
		MotionBlurOptions,
		| "shutterScale"
		| "maxSamples"
		| "velocityClamp"
		| "depthReject"
		| "centerWeight"
	>
> = {
	shutterScale: 1,
	maxSamples: 16,
	velocityClamp: 0.06,
	depthReject: 0.025,
	centerWeight: 1,
};

export const DEFAULT_DOF_OPTIONS: Required<
	Pick<
		DOFOptions,
		| "focusDistance"
		| "focusRange"
		| "nearStrength"
		| "farStrength"
		| "maxBlurRadius"
		| "depthCurve"
		| "highlightThreshold"
		| "highlightGain"
		| "chromaticAberration"
	>
> = {
	focusDistance: 8,
	focusRange: 3,
	nearStrength: 0.85,
	farStrength: 1,
	maxBlurRadius: 12,
	depthCurve: 1.25,
	highlightThreshold: 1.2,
	highlightGain: 0.35,
	chromaticAberration: 0.2,
};

export const DEFAULT_COLOR_FILTER_OPTIONS: Required<
	Pick<
		ColorFilterOptions,
		"brightness" | "saturation" | "contrast" | "temperature" | "tint"
	>
> = {
	brightness: 0,
	saturation: 1,
	contrast: 1,
	temperature: 0,
	tint: 0,
};

export interface SoftwareBuiltinPostProcessContext {
	readonly canvasContext: CanvasRenderingContext2D | null;
}

export interface WebGPUScreenPostProcessContext {
	readonly encoder?: ICommandEncoder;
	readonly targets?: WebGPUPostProcessFrameTargets;
	readonly shared: PostProcessSharedContext;
	publishColorTarget?(texture: IRenderTexture): void;
}

export type WebGPURuntimePostProcessContext = WebGPUScreenPostProcessContext;

export interface WebGPUGammaContext {
	readonly targets?: WebGPUPostProcessFrameTargets;
	presentToCanvas?(
		source: IRenderTexture,
		applyGamma: boolean
	): void | Promise<void>;
	warmupPresent?(): void | Promise<void>;
}

export interface WebGLScreenPostProcessContext {
	readonly gl: WebGL2RenderingContext;
	readonly programs: WebGLProgramLibrary;
	readonly fullscreenVao: WebGLVertexArrayObject | null;
	readonly postFramebuffer: WebGLFramebuffer | null;
	readonly sceneColorTexture: WebGLTexture | null;
	readonly sceneMotionTexture?: WebGLTexture | null;
	readonly width: number;
	readonly height: number;
	getSourceTexture(): WebGLTexture | null;
	resolveTargetTexture(sourceTexture: WebGLTexture): WebGLTexture | null;
	bindColorTarget(texture: WebGLTexture): void;
	drawFullscreen(): void;
	publishColorTexture(texture: WebGLTexture): void;
}

export interface WebGLGammaContext {
	readonly gl: WebGL2RenderingContext;
	readonly programs: WebGLProgramLibrary;
	readonly fullscreenVao: WebGLVertexArrayObject | null;
	readonly width: number;
	readonly height: number;
	getSourceTexture(): WebGLTexture | null;
	drawFullscreen(): void;
	markPresented(): void;
}

export type WebGPUMotionBlurContext = WebGPURuntimePostProcessContext;
export type WebGPUDepthOfFieldContext = WebGPURuntimePostProcessContext;
export type WebGPUToneMappingContext = WebGPURuntimePostProcessContext;
export type WebGPUColorFilterContext = WebGPURuntimePostProcessContext;
export type WebGPUInteractionOutlineContext = WebGPURuntimePostProcessContext;

export type WebGLMotionBlurContext = WebGLScreenPostProcessContext;
export type WebGLDepthOfFieldContext = WebGLScreenPostProcessContext;
export type WebGLToneMappingContext = WebGLScreenPostProcessContext;
export type WebGLColorFilterContext = WebGLScreenPostProcessContext;
export type WebGLInteractionOutlineContext = WebGLScreenPostProcessContext;

export class SoftwareToneMappingImplementation
	implements PostProcessPassImplementation<SoftwareBuiltinPostProcessContext>
{
	public readonly id = "tonemap:software";
	private readonly _runtime = new SoftwareScreenPassRuntime();

	public execute(
		request: PostProcessPassRequest,
		_context: SoftwareBuiltinPostProcessContext | undefined
	): PostProcessPassResult {
		if (!request.frameContext.attachments.pixels) {
			return { ran: false };
		}
		this._runtime.applyToneMapping(request.frameContext);
		return { ran: true };
	}
}

export class SoftwareColorFilterImplementation
	implements PostProcessPassImplementation<SoftwareBuiltinPostProcessContext>
{
	public readonly id = "color-filter:software";
	private readonly _runtime = new SoftwareScreenPassRuntime();

	public execute(
		request: PostProcessPassRequest,
		_context: SoftwareBuiltinPostProcessContext | undefined
	): PostProcessPassResult {
		if (!request.frameContext.attachments.pixels) {
			return { ran: false };
		}
		this._runtime.applyColorFilter(request.frameContext, {
			...DEFAULT_COLOR_FILTER_OPTIONS,
			...((request.options as ColorFilterOptions | undefined) ?? {}),
		});
		return { ran: true };
	}
}

export class SoftwareInteractionOutlineImplementation
	implements PostProcessPassImplementation<SoftwareBuiltinPostProcessContext>
{
	public readonly id = "interaction-outline:software";
	private readonly _runtime = new SoftwareScreenPassRuntime();

	public execute(
		request: PostProcessPassRequest,
		_context: SoftwareBuiltinPostProcessContext | undefined
	): PostProcessPassResult {
		if (!request.frameContext.attachments.pixels) {
			return { ran: false };
		}
		this._runtime.applyInteractionOutline(request.frameContext);
		return { ran: true };
	}
}

export class SoftwareGammaImplementation
	implements PostProcessPassImplementation<SoftwareBuiltinPostProcessContext>
{
	public readonly id = "gamma:software";
	private readonly _runtime = new SoftwareScreenPassRuntime();

	public execute(
		request: PostProcessPassRequest,
		context: SoftwareBuiltinPostProcessContext | undefined
	): PostProcessPassResult {
		const canvasContext = context?.canvasContext ?? null;
		if (!request.frameContext.attachments.pixels && !canvasContext) {
			return { ran: false };
		}
		this._runtime.applyGamma(
			request.frameContext,
			canvasContext
		);
		return { ran: true };
	}
}

interface WebGPUMotionBlurResources {
	shared: PostProcessSharedContext;
	module: IShaderModule | null;
	pipeline: IComputePipeline | null;
	params: IRenderBuffer | null;
	paramData: Float32Array<ArrayBuffer>;
	paramUploaded: boolean;
}

interface WebGPUDepthOfFieldResources {
	shared: PostProcessSharedContext;
	module: IShaderModule | null;
	pipeline: IComputePipeline | null;
	params: IRenderBuffer | null;
	paramData: Float32Array<ArrayBuffer>;
}

interface WebGPUToneMappingResources {
	shared: PostProcessSharedContext;
	module: IShaderModule | null;
	pipeline: IComputePipeline | null;
}

interface WebGPUColorFilterResources {
	shared: PostProcessSharedContext;
	module: IShaderModule | null;
	pipeline: IComputePipeline | null;
	params: IRenderBuffer | null;
	paramData: Float32Array<ArrayBuffer>;
}

interface WebGPUInteractionOutlineResources {
	shared: PostProcessSharedContext;
	module: IShaderModule | null;
	pipeline: IComputePipeline | null;
	params: IRenderBuffer | null;
	paramWriter: ReturnType<typeof INTERACTION_OUTLINE_LAYOUT.createWriter>;
}

export class WebGPUMotionBlurImplementation
	implements PostProcessPassImplementation<WebGPUMotionBlurContext, MotionBlurOptions>
{
	public readonly id = "motion-blur:webgpu";
	public readonly metadata = {
		context: WEBGPU_SCREEN_POST_PROCESS_CONTEXT_METADATA,
	};
	private _resources =
		new WeakMap<PostProcessSharedContext, WebGPUMotionBlurResources>();
	private _resourceSet = new Set<WebGPUMotionBlurResources>();

	public async warmup(
		context: WebGPUMotionBlurContext | undefined
	): Promise<void> {
		if (context) {
			await this._ensureResources(context.shared);
		}
	}

	public async execute(
		request: PostProcessPassRequest<MotionBlurOptions>,
		context: WebGPUMotionBlurContext | undefined
	): Promise<PostProcessPassResult> {
		if (!context?.encoder || !context.targets) {
			return { ran: false };
		}
		const ran = await this._runMotionBlurKernel(request, context);
		return ran ? { ran: true } : { ran: false };
	}

	public invalidate(): void {
		for (const resources of this._resourceSet) {
			resources.shared.invalidateBindingsByPrefix("motion-blur-");
		}
	}

	public destroy(): void {
		for (const resources of this._resourceSet) {
			resources.shared.destroyManagedResource(
				resources.pipeline,
				"motion blur pipeline"
			);
			resources.shared.destroyManagedResource(
				resources.module,
				"motion blur shader module"
			);
			resources.shared.destroyManagedResource(
				resources.params,
				"motion blur params buffer"
			);
			resources.shared.invalidateBindingsByPrefix("motion-blur-");
			resources.module = null;
			resources.pipeline = null;
			resources.params = null;
			resources.paramUploaded = false;
		}
		this._resourceSet.clear();
		this._resources =
			new WeakMap<PostProcessSharedContext, WebGPUMotionBlurResources>();
	}

	private async _runMotionBlurKernel(
		request: PostProcessPassRequest<MotionBlurOptions>,
		context: WebGPUMotionBlurContext
	): Promise<boolean> {
		const resources = await this._ensureResources(context.shared);
		if (
			!context.encoder ||
			!context.targets ||
			!context.shared.sampler ||
			!resources.pipeline ||
			!resources.params
		) {
			return false;
		}
		const targets = context.targets;
		const target = resolveWebGPUTarget(targets);
		const options = request.options ?? DEFAULT_MOTION_BLUR_OPTIONS;
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
		uploadWebGPUMotionBlurParams(
			context.shared,
			resources,
			target.width,
			target.height,
			shutterScale,
			maxSamples,
			velocityClamp,
			depthReject,
			centerWeight
		);
		const binding = context.shared.getCachedBindGroup(
			`motion-blur-${target === targets.postPing ? "ping" : "pong"}`,
			resources.pipeline,
			[
				{ binding: 0, resource: targets.sceneColor },
				{ binding: 1, resource: targets.gMotionDepth },
				{ binding: 2, resource: context.shared.sampler },
				{ binding: 3, resource: resources.params },
				{ binding: 4, resource: target },
			],
			"WebGPUMotionBlur_Binding"
		);
		context.encoder.beginComputePass({ label: "WebGPUMotionBlur" });
		context.encoder.setComputePipeline(resources.pipeline);
		context.encoder.setBindingGroup(0, binding);
		context.encoder.dispatchWorkgroups(
			ceilDiv(target.width, WEBGPU_WORKGROUP_SIZE),
			ceilDiv(target.height, WEBGPU_WORKGROUP_SIZE),
			1
		);
		context.encoder.endComputePass();
		publishWebGPUColorTarget(context, target);
		return true;
	}

	private async _ensureResources(
		shared: PostProcessSharedContext
	): Promise<WebGPUMotionBlurResources> {
		let resources = this._resources.get(shared);
		if (!resources) {
			resources = {
				shared,
				module: null,
				pipeline: null,
				params: null,
				paramData: new Float32Array(8),
				paramUploaded: false,
			};
			this._resources.set(shared, resources);
			this._resourceSet.add(resources);
		}
		await shared.ensureCommonResources();
		if (!resources.module) {
			const shader = await loadPostProcessShaderPartComposite("motionBlur");
			resources.module = await shared.compute.createShaderModule({
				label: "WebGPUMotionBlurShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!resources.pipeline) {
			resources.pipeline = shared.compute.createComputePipeline({
				label: "WebGPUMotionBlurPipeline",
				compute: { module: resources.module, entryPoint: "csMain" },
			});
		}
		if (!resources.params) {
			resources.params = shared.compute.createBuffer({
				label: "WebGPUMotionBlurParams",
				size: 8 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
		}
		return resources;
	}
}

export class WebGPUDepthOfFieldImplementation
	implements PostProcessPassImplementation<WebGPUDepthOfFieldContext, DOFOptions>
{
	public readonly id = "dof:webgpu";
	public readonly metadata = {
		context: WEBGPU_SCREEN_POST_PROCESS_CONTEXT_METADATA,
	};
	private _resources =
		new WeakMap<PostProcessSharedContext, WebGPUDepthOfFieldResources>();
	private _resourceSet = new Set<WebGPUDepthOfFieldResources>();

	public async warmup(
		context: WebGPUDepthOfFieldContext | undefined
	): Promise<void> {
		if (context) {
			await this._ensureResources(context.shared);
		}
	}

	public async execute(
		request: PostProcessPassRequest<DOFOptions>,
		context: WebGPUDepthOfFieldContext | undefined
	): Promise<PostProcessPassResult> {
		if (!context?.encoder || !context.targets) {
			return { ran: false };
		}
		const ran = await this._runDOFKernel(request, context);
		return ran ? { ran: true } : { ran: false };
	}

	public invalidate(): void {
		for (const resources of this._resourceSet) {
			resources.shared.invalidateBindingsByPrefix("dof-");
		}
	}

	public destroy(): void {
		for (const resources of this._resourceSet) {
			resources.shared.destroyManagedResource(resources.pipeline, "DoF pipeline");
			resources.shared.destroyManagedResource(resources.module, "DoF shader module");
			resources.shared.destroyManagedResource(resources.params, "DoF params buffer");
			resources.shared.invalidateBindingsByPrefix("dof-");
			resources.module = null;
			resources.pipeline = null;
			resources.params = null;
		}
		this._resourceSet.clear();
		this._resources =
			new WeakMap<PostProcessSharedContext, WebGPUDepthOfFieldResources>();
	}

	private async _runDOFKernel(
		request: PostProcessPassRequest<DOFOptions>,
		context: WebGPUDepthOfFieldContext
	): Promise<boolean> {
		const resources = await this._ensureResources(context.shared);
		if (
			!context.encoder ||
			!context.targets ||
			!context.shared.sampler ||
			!resources.pipeline ||
			!resources.params
		) {
			return false;
		}
		const targets = context.targets;
		const target = resolveWebGPUTarget(targets);
		const options = request.options ?? DEFAULT_DOF_OPTIONS;
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
		const data = resources.paramData;
		data[0] = 1 / Math.max(target.width, 1);
		data[1] = 1 / Math.max(target.height, 1);
		data[2] = focusDistance;
		data[3] = focusRange;
		data[4] = nearStrength;
		data[5] = farStrength;
		data[6] = maxBlurRadius;
		data[7] = depthCurve;
		data[8] = highlightThreshold;
		data[9] = highlightGain;
		data[10] = chromaticAberration;
		data[11] = 0;
		context.shared.compute.writeBuffer(resources.params, data);
		const binding = context.shared.getCachedBindGroup(
			`dof-${target === targets.postPing ? "ping" : "pong"}`,
			resources.pipeline,
			[
				{ binding: 0, resource: targets.sceneColor },
				{ binding: 1, resource: targets.gMotionDepth },
				{ binding: 2, resource: context.shared.sampler },
				{ binding: 3, resource: resources.params },
				{ binding: 4, resource: target },
			],
			"WebGPUDOF_Binding"
		);
		context.encoder.beginComputePass({ label: "WebGPUDOF" });
		context.encoder.setComputePipeline(resources.pipeline);
		context.encoder.setBindingGroup(0, binding);
		context.encoder.dispatchWorkgroups(
			ceilDiv(target.width, WEBGPU_WORKGROUP_SIZE),
			ceilDiv(target.height, WEBGPU_WORKGROUP_SIZE),
			1
		);
		context.encoder.endComputePass();
		publishWebGPUColorTarget(context, target);
		return true;
	}

	private async _ensureResources(
		shared: PostProcessSharedContext
	): Promise<WebGPUDepthOfFieldResources> {
		let resources = this._resources.get(shared);
		if (!resources) {
			resources = {
				shared,
				module: null,
				pipeline: null,
				params: null,
				paramData: new Float32Array(12),
			};
			this._resources.set(shared, resources);
			this._resourceSet.add(resources);
		}
		await shared.ensureCommonResources();
		if (!resources.module) {
			const shader = await loadPostProcessShaderPartComposite("dof");
			resources.module = await shared.compute.createShaderModule({
				label: "WebGPUDOFShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!resources.pipeline) {
			resources.pipeline = shared.compute.createComputePipeline({
				label: "WebGPUDOFPipeline",
				compute: { module: resources.module, entryPoint: "csMain" },
			});
		}
		if (!resources.params) {
			resources.params = shared.compute.createBuffer({
				label: "WebGPUDOFParams",
				size: 12 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
		}
		return resources;
	}
}

export class WebGPUToneMappingImplementation
	implements PostProcessPassImplementation<WebGPUToneMappingContext, EmptyOptions>
{
	public readonly id = "tonemap:webgpu";
	public readonly metadata = {
		context: WEBGPU_SCREEN_POST_PROCESS_CONTEXT_METADATA,
	};
	private _resources =
		new WeakMap<PostProcessSharedContext, WebGPUToneMappingResources>();
	private _resourceSet = new Set<WebGPUToneMappingResources>();

	public async warmup(
		context: WebGPUToneMappingContext | undefined
	): Promise<void> {
		if (context) {
			await this._ensureResources(context.shared);
		}
	}

	public async execute(
		_request: PostProcessPassRequest<EmptyOptions>,
		context: WebGPUToneMappingContext | undefined
	): Promise<PostProcessPassResult> {
		if (!context?.encoder || !context.targets) {
			return { ran: false };
		}
		const ran = await this._runToneMappingKernel(context);
		return ran ? { ran: true } : { ran: false };
	}

	public invalidate(): void {
		for (const resources of this._resourceSet) {
			resources.shared.invalidateBindingsByPrefix("tonemap-");
		}
	}

	public destroy(): void {
		for (const resources of this._resourceSet) {
			resources.shared.destroyManagedResource(
				resources.pipeline,
				"tone mapping pipeline"
			);
			resources.shared.destroyManagedResource(
				resources.module,
				"tone mapping shader module"
			);
			resources.shared.invalidateBindingsByPrefix("tonemap-");
			resources.module = null;
			resources.pipeline = null;
		}
		this._resourceSet.clear();
		this._resources =
			new WeakMap<PostProcessSharedContext, WebGPUToneMappingResources>();
	}

	private async _runToneMappingKernel(
		context: WebGPUToneMappingContext
	): Promise<boolean> {
		const resources = await this._ensureResources(context.shared);
		if (!context.encoder || !context.targets || !resources.pipeline) {
			return false;
		}
		const targets = context.targets;
		const target = resolveWebGPUTarget(targets);
		const binding = context.shared.getCachedBindGroup(
			`tonemap-${target === targets.postPing ? "ping" : "pong"}`,
			resources.pipeline,
			[
				{ binding: 0, resource: targets.sceneColor },
				{ binding: 1, resource: target },
			],
			"WebGPUToneMapping_Binding"
		);
		context.encoder.beginComputePass({ label: "WebGPUToneMapping" });
		context.encoder.setComputePipeline(resources.pipeline);
		context.encoder.setBindingGroup(0, binding);
		context.encoder.dispatchWorkgroups(
			ceilDiv(target.width, WEBGPU_WORKGROUP_SIZE),
			ceilDiv(target.height, WEBGPU_WORKGROUP_SIZE),
			1
		);
		context.encoder.endComputePass();
		publishWebGPUColorTarget(context, target);
		return true;
	}

	private async _ensureResources(
		shared: PostProcessSharedContext
	): Promise<WebGPUToneMappingResources> {
		let resources = this._resources.get(shared);
		if (!resources) {
			resources = {
				shared,
				module: null,
				pipeline: null,
			};
			this._resources.set(shared, resources);
			this._resourceSet.add(resources);
		}
		if (!resources.module) {
			const shader = await loadPostProcessShaderPartComposite("toneMapping");
			resources.module = await shared.compute.createShaderModule({
				label: "WebGPUToneMappingShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!resources.pipeline) {
			resources.pipeline = shared.compute.createComputePipeline({
				label: "WebGPUToneMappingPipeline",
				compute: { module: resources.module, entryPoint: "csMain" },
			});
		}
		return resources;
	}
}

export class WebGPUColorFilterImplementation
	implements PostProcessPassImplementation<WebGPUColorFilterContext, ColorFilterOptions>
{
	public readonly id = "color-filter:webgpu";
	public readonly metadata = {
		context: WEBGPU_SCREEN_POST_PROCESS_CONTEXT_METADATA,
	};
	private _resources =
		new WeakMap<PostProcessSharedContext, WebGPUColorFilterResources>();
	private _resourceSet = new Set<WebGPUColorFilterResources>();

	public async warmup(
		context: WebGPUColorFilterContext | undefined
	): Promise<void> {
		if (context) {
			await this._ensureResources(context.shared);
		}
	}

	public async execute(
		request: PostProcessPassRequest<ColorFilterOptions>,
		context: WebGPUColorFilterContext | undefined
	): Promise<PostProcessPassResult> {
		if (!context?.encoder || !context.targets) {
			return { ran: false };
		}
		const ran = await this._runColorFilterKernel(request, context);
		return ran ? { ran: true } : { ran: false };
	}

	public invalidate(): void {
		for (const resources of this._resourceSet) {
			resources.shared.invalidateBindingsByPrefix("color-filter-");
		}
	}

	public destroy(): void {
		for (const resources of this._resourceSet) {
			resources.shared.destroyManagedResource(
				resources.pipeline,
				"color filter pipeline"
			);
			resources.shared.destroyManagedResource(
				resources.module,
				"color filter shader module"
			);
			resources.shared.destroyManagedResource(
				resources.params,
				"color filter params buffer"
			);
			resources.shared.invalidateBindingsByPrefix("color-filter-");
			resources.module = null;
			resources.pipeline = null;
			resources.params = null;
		}
		this._resourceSet.clear();
		this._resources =
			new WeakMap<PostProcessSharedContext, WebGPUColorFilterResources>();
	}

	private async _runColorFilterKernel(
		request: PostProcessPassRequest<ColorFilterOptions>,
		context: WebGPUColorFilterContext
	): Promise<boolean> {
		const resources = await this._ensureResources(context.shared);
		if (
			!context.encoder ||
			!context.targets ||
			!context.shared.sampler ||
			!resources.pipeline ||
			!resources.params
		) {
			return false;
		}
		const targets = context.targets;
		const target = resolveWebGPUTarget(targets);
		const options = request.options ?? DEFAULT_COLOR_FILTER_OPTIONS;
		const data = resources.paramData;
		data[0] = clamp(
			finiteOr(options.brightness, DEFAULT_COLOR_FILTER_OPTIONS.brightness),
			-1,
			1
		);
		data[1] = clamp(
			finiteOr(options.saturation, DEFAULT_COLOR_FILTER_OPTIONS.saturation),
			0,
			2
		);
		data[2] = clamp(
			finiteOr(options.contrast, DEFAULT_COLOR_FILTER_OPTIONS.contrast),
			0,
			2
		);
		data[3] = clamp(
			finiteOr(options.temperature, DEFAULT_COLOR_FILTER_OPTIONS.temperature),
			-1,
			1
		);
		data[4] = clamp(
			finiteOr(options.tint, DEFAULT_COLOR_FILTER_OPTIONS.tint),
			-1,
			1
		);
		data[5] = 0;
		data[6] = 0;
		data[7] = 0;
		context.shared.compute.writeBuffer(resources.params, data);
		const binding = context.shared.getCachedBindGroup(
			`color-filter-${target === targets.postPing ? "ping" : "pong"}`,
			resources.pipeline,
			[
				{ binding: 0, resource: targets.sceneColor },
				{ binding: 1, resource: context.shared.sampler },
				{ binding: 2, resource: resources.params },
				{ binding: 3, resource: target },
			],
			"WebGPUColorFilter_Binding"
		);
		context.encoder.beginComputePass({ label: "WebGPUColorFilter" });
		context.encoder.setComputePipeline(resources.pipeline);
		context.encoder.setBindingGroup(0, binding);
		context.encoder.dispatchWorkgroups(
			ceilDiv(target.width, WEBGPU_WORKGROUP_SIZE),
			ceilDiv(target.height, WEBGPU_WORKGROUP_SIZE),
			1
		);
		context.encoder.endComputePass();
		publishWebGPUColorTarget(context, target);
		return true;
	}

	private async _ensureResources(
		shared: PostProcessSharedContext
	): Promise<WebGPUColorFilterResources> {
		let resources = this._resources.get(shared);
		if (!resources) {
			resources = {
				shared,
				module: null,
				pipeline: null,
				params: null,
				paramData: new Float32Array(8),
			};
			this._resources.set(shared, resources);
			this._resourceSet.add(resources);
		}
		await shared.ensureCommonResources();
		if (!resources.module) {
			const shader = await loadPostProcessShaderPartComposite("colorFilter");
			resources.module = await shared.compute.createShaderModule({
				label: "WebGPUColorFilterShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!resources.pipeline) {
			resources.pipeline = shared.compute.createComputePipeline({
				label: "WebGPUColorFilterPipeline",
				compute: { module: resources.module, entryPoint: "csMain" },
			});
		}
		if (!resources.params) {
			resources.params = shared.compute.createBuffer({
				label: "WebGPUColorFilterParams",
				size: 8 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
		}
		return resources;
	}
}

export class WebGPUInteractionOutlineImplementation
	implements PostProcessPassImplementation<WebGPUInteractionOutlineContext, EmptyOptions>
{
	public readonly id = "interaction-outline:webgpu";
	public readonly metadata = {
		context: WEBGPU_SCREEN_POST_PROCESS_CONTEXT_METADATA,
	};
	private _resources =
		new WeakMap<PostProcessSharedContext, WebGPUInteractionOutlineResources>();
	private _resourceSet = new Set<WebGPUInteractionOutlineResources>();

	public async warmup(
		context: WebGPUInteractionOutlineContext | undefined
	): Promise<void> {
		if (context) {
			await this._ensureResources(context.shared);
		}
	}

	public async execute(
		request: PostProcessPassRequest<EmptyOptions>,
		context: WebGPUInteractionOutlineContext | undefined
	): Promise<PostProcessPassResult> {
		if (!context?.encoder || !context.targets) {
			return { ran: false };
		}
		const ran = await this._runInteractionOutlineKernel(request, context);
		return ran ? { ran: true } : { ran: false };
	}

	public invalidate(): void {
		for (const resources of this._resourceSet) {
			resources.shared.invalidateBindingsByPrefix("interaction-outline-");
		}
	}

	public destroy(): void {
		for (const resources of this._resourceSet) {
			resources.shared.destroyManagedResource(
				resources.pipeline,
				"interaction outline pipeline"
			);
			resources.shared.destroyManagedResource(
				resources.module,
				"interaction outline shader module"
			);
			resources.shared.destroyManagedResource(
				resources.params,
				"interaction outline params buffer"
			);
			resources.shared.invalidateBindingsByPrefix("interaction-outline-");
			resources.module = null;
			resources.pipeline = null;
			resources.params = null;
		}
		this._resourceSet.clear();
		this._resources =
			new WeakMap<PostProcessSharedContext, WebGPUInteractionOutlineResources>();
	}

	private async _runInteractionOutlineKernel(
		request: PostProcessPassRequest<EmptyOptions>,
		context: WebGPUInteractionOutlineContext
	): Promise<boolean> {
		const interactionState = request.frameContext.transient.get(
			INTERACTION_TRANSIENT_STATE_KEY
		);
		const selectedEntityIds = interactionState?.selectedEntityIds ?? [];
		if (selectedEntityIds.length === 0) {
			return false;
		}
		const circles = collectProjectedOutlineCircles(
			request.frameContext,
			selectedEntityIds,
			MAX_INTERACTION_OUTLINE_CIRCLES
		);
		if (circles.length === 0) {
			return false;
		}
		const resources = await this._ensureResources(context.shared);
		if (!context.encoder || !context.targets || !resources.pipeline || !resources.params) {
			return false;
		}
		const targets = context.targets;
		const target = resolveWebGPUTarget(targets);
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
		const params = resources.paramWriter;
		params.clear();
		params.writeVec("invSize", [
			1 / Math.max(target.width, 1),
			1 / Math.max(target.height, 1),
		]);
		params.writeF32("opacity", opacity);
		params.writeF32("thickness", thickness);
		params.writeVec("color", [
			sRGBToLinear(clamp(outlineColor.r / Math.max(1, colorScale), 0, 1)),
			sRGBToLinear(clamp(outlineColor.g / Math.max(1, colorScale), 0, 1)),
			sRGBToLinear(clamp(outlineColor.b / Math.max(1, colorScale), 0, 1)),
			1,
		]);
		params.writeF32("circleCount", circles.length);
		params.writeF32("shape", shapeCode);
		for (let index = 0; index < circles.length; index++) {
			const circle = circles[index];
			params.writeVec(["circles", index], [
				circle.centerX,
				circle.centerY,
				circle.radius,
				0,
			]);
		}
		context.shared.compute.writeBuffer(
			resources.params,
			params.toFloat32Array() as Float32Array<ArrayBuffer>
		);
		const binding = context.shared.getCachedBindGroup(
			`interaction-outline-${target === targets.postPing ? "ping" : "pong"}`,
			resources.pipeline,
			[
				{ binding: 0, resource: targets.sceneColor },
				{ binding: 2, resource: resources.params },
				{ binding: 3, resource: target },
			],
			"WebGPUInteractionOutline_Binding"
		);
		context.encoder.beginComputePass({ label: "WebGPUInteractionOutline" });
		context.encoder.setComputePipeline(resources.pipeline);
		context.encoder.setBindingGroup(0, binding);
		context.encoder.dispatchWorkgroups(
			ceilDiv(target.width, WEBGPU_WORKGROUP_SIZE),
			ceilDiv(target.height, WEBGPU_WORKGROUP_SIZE),
			1
		);
		context.encoder.endComputePass();
		publishWebGPUColorTarget(context, target);
		return true;
	}

	private async _ensureResources(
		shared: PostProcessSharedContext
	): Promise<WebGPUInteractionOutlineResources> {
		let resources = this._resources.get(shared);
		if (!resources) {
			resources = {
				shared,
				module: null,
				pipeline: null,
				params: null,
				paramWriter: INTERACTION_OUTLINE_LAYOUT.createWriter(),
			};
			this._resources.set(shared, resources);
			this._resourceSet.add(resources);
		}
		await shared.ensureCommonResources();
		if (!resources.module) {
			const shader = await loadPostProcessShaderPartComposite(
				"interactionOutline"
			);
			resources.module = await shared.compute.createShaderModule({
				label: "WebGPUInteractionOutlineShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!resources.pipeline) {
			resources.pipeline = shared.compute.createComputePipeline({
				label: "WebGPUInteractionOutlinePipeline",
				compute: {
					module: resources.module,
					entryPoint: "csMain",
				},
			});
		}
		if (!resources.params) {
			resources.params = shared.compute.createBuffer({
				label: "WebGPUInteractionOutlineParams",
				size: INTERACTION_OUTLINE_LAYOUT.byteSize,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
		}
		return resources;
	}
}

export class WebGPUGammaImplementation
	implements PostProcessPassImplementation<WebGPUGammaContext, EmptyOptions>
{
	public readonly id = "gamma:webgpu";
	public readonly metadata = {
		context: WEBGPU_PRESENT_POST_PROCESS_CONTEXT_METADATA,
	};

	public async warmup(context: WebGPUGammaContext | undefined): Promise<void> {
		await context?.warmupPresent?.();
	}

	public async execute(
		request: PostProcessPassRequest<EmptyOptions>,
		context: WebGPUGammaContext | undefined
	): Promise<PostProcessPassResult> {
		const source = context?.targets?.sceneColor;
		if (!source || !context?.presentToCanvas) {
			return { ran: false };
		}
		await context.presentToCanvas(
			source,
			request.frameContext.postProcess.isEnabled(GAMMA_PASS_ID)
		);
		return { ran: true };
	}
}

export class WebGLMotionBlurImplementation
	implements PostProcessPassImplementation<WebGLMotionBlurContext, MotionBlurOptions>
{
	public readonly id = "motion-blur:webgl";

	public warmup(context: WebGLMotionBlurContext | undefined): void {
		context?.programs.getMotionBlurProgram();
	}

	public execute(
		request: PostProcessPassRequest<MotionBlurOptions>,
		context: WebGLMotionBlurContext | undefined
	): PostProcessPassResult {
		if (!context?.sceneMotionTexture) {
			return { ran: false };
		}
		const target = resolveWebGLTarget(context);
		if (!target) {
			return { ran: false };
		}
		const options = request.options;
		const shutterScale = sanitizeFiniteClamped(
			options?.shutterScale,
			DEFAULT_MOTION_BLUR_OPTIONS.shutterScale,
			MOTION_BLUR_SHUTTER_SCALE_RANGE[0],
			MOTION_BLUR_SHUTTER_SCALE_RANGE[1]
		);
		const maxSamples = clamp(
			Math.round(
				finiteOr(options?.maxSamples, DEFAULT_MOTION_BLUR_OPTIONS.maxSamples)
			),
			MOTION_BLUR_MAX_SAMPLES_RANGE[0],
			MOTION_BLUR_MAX_SAMPLES_RANGE[1]
		);
		const velocityClamp = sanitizeFiniteClamped(
			options?.velocityClamp,
			DEFAULT_MOTION_BLUR_OPTIONS.velocityClamp,
			MOTION_BLUR_VELOCITY_CLAMP_RANGE[0],
			MOTION_BLUR_VELOCITY_CLAMP_RANGE[1]
		);
		const depthReject = sanitizeFiniteClamped(
			options?.depthReject,
			DEFAULT_MOTION_BLUR_OPTIONS.depthReject,
			MOTION_BLUR_DEPTH_REJECT_RANGE[0],
			MOTION_BLUR_DEPTH_REJECT_RANGE[1]
		);
		const centerWeight = sanitizeFiniteClamped(
			options?.centerWeight,
			DEFAULT_MOTION_BLUR_OPTIONS.centerWeight,
			MOTION_BLUR_CENTER_WEIGHT_RANGE[0],
			MOTION_BLUR_CENTER_WEIGHT_RANGE[1]
		);

		const gl = context.gl;
		const program = context.programs.getMotionBlurProgram();
		bindWebGLPostTarget(context, program.program, target.texture);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, target.source);
		gl.activeTexture(gl.TEXTURE1);
		gl.bindTexture(gl.TEXTURE_2D, context.sceneMotionTexture);
		const uniforms = program.uniforms;
		if (uniforms.sourceMap) gl.uniform1i(uniforms.sourceMap, 0);
		if (uniforms.motionDepthMap) gl.uniform1i(uniforms.motionDepthMap, 1);
		if (uniforms.texelSize) {
			gl.uniform2f(
				uniforms.texelSize,
				1 / Math.max(1, context.width),
				1 / Math.max(1, context.height)
			);
		}
		if (uniforms.motionParams) {
			gl.uniform4f(
				uniforms.motionParams,
				shutterScale,
				maxSamples,
				velocityClamp,
				depthReject
			);
		}
		if (uniforms.centerWeight) {
			gl.uniform1f(uniforms.centerWeight, centerWeight);
		}
		context.drawFullscreen();
		gl.bindVertexArray(null);
		context.publishColorTexture(target.texture);
		return { ran: true };
	}
}

export class WebGLDepthOfFieldImplementation
	implements PostProcessPassImplementation<WebGLDepthOfFieldContext, DOFOptions>
{
	public readonly id = "dof:webgl";

	public warmup(context: WebGLDepthOfFieldContext | undefined): void {
		context?.programs.getDOFProgram();
	}

	public execute(
		request: PostProcessPassRequest<DOFOptions>,
		context: WebGLDepthOfFieldContext | undefined
	): PostProcessPassResult {
		if (!context?.sceneMotionTexture) {
			return { ran: false };
		}
		const target = resolveWebGLTarget(context);
		if (!target) {
			return { ran: false };
		}
		const options = request.options;
		const focusDistance = Math.max(
			0.01,
			finiteOr(options?.focusDistance, DEFAULT_DOF_OPTIONS.focusDistance)
		);
		const focusRange = Math.max(
			0.001,
			finiteOr(options?.focusRange, DEFAULT_DOF_OPTIONS.focusRange)
		);
		const nearStrength = sanitizeFiniteClamped(
			options?.nearStrength,
			DEFAULT_DOF_OPTIONS.nearStrength,
			DOF_NEAR_FAR_STRENGTH_RANGE[0],
			DOF_NEAR_FAR_STRENGTH_RANGE[1]
		);
		const farStrength = sanitizeFiniteClamped(
			options?.farStrength,
			DEFAULT_DOF_OPTIONS.farStrength,
			DOF_NEAR_FAR_STRENGTH_RANGE[0],
			DOF_NEAR_FAR_STRENGTH_RANGE[1]
		);
		const maxBlurRadius = sanitizeFiniteClamped(
			options?.maxBlurRadius,
			DEFAULT_DOF_OPTIONS.maxBlurRadius,
			DOF_MAX_BLUR_RADIUS_RANGE[0],
			DOF_MAX_BLUR_RADIUS_RANGE[1]
		);
		const depthCurve = sanitizeFiniteClamped(
			options?.depthCurve,
			DEFAULT_DOF_OPTIONS.depthCurve,
			DOF_DEPTH_CURVE_RANGE[0],
			DOF_DEPTH_CURVE_RANGE[1]
		);
		const highlightThreshold = Math.max(
			0,
			finiteOr(options?.highlightThreshold, DEFAULT_DOF_OPTIONS.highlightThreshold)
		);
		const highlightGain = sanitizeFiniteClamped(
			options?.highlightGain,
			DEFAULT_DOF_OPTIONS.highlightGain,
			DOF_HIGHLIGHT_GAIN_RANGE[0],
			DOF_HIGHLIGHT_GAIN_RANGE[1]
		);
		const chromaticAberration = sanitizeFiniteClamped(
			options?.chromaticAberration,
			DEFAULT_DOF_OPTIONS.chromaticAberration,
			DOF_CHROMATIC_ABERRATION_RANGE[0],
			DOF_CHROMATIC_ABERRATION_RANGE[1]
		);

		const gl = context.gl;
		const program = context.programs.getDOFProgram();
		bindWebGLPostTarget(context, program.program, target.texture);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, target.source);
		gl.activeTexture(gl.TEXTURE1);
		gl.bindTexture(gl.TEXTURE_2D, context.sceneMotionTexture);
		const uniforms = program.uniforms;
		if (uniforms.sourceMap) gl.uniform1i(uniforms.sourceMap, 0);
		if (uniforms.motionDepthMap) gl.uniform1i(uniforms.motionDepthMap, 1);
		if (uniforms.texelSize) {
			gl.uniform2f(
				uniforms.texelSize,
				1 / Math.max(1, context.width),
				1 / Math.max(1, context.height)
			);
		}
		if (uniforms.focusParams) {
			gl.uniform4f(
				uniforms.focusParams,
				focusDistance,
				focusRange,
				nearStrength,
				farStrength
			);
		}
		if (uniforms.dofParams) {
			gl.uniform4f(
				uniforms.dofParams,
				maxBlurRadius,
				depthCurve,
				highlightThreshold,
				highlightGain
			);
		}
		if (uniforms.chromaticAberration) {
			gl.uniform1f(uniforms.chromaticAberration, chromaticAberration);
		}
		context.drawFullscreen();
		gl.bindVertexArray(null);
		context.publishColorTexture(target.texture);
		return { ran: true };
	}
}

export class WebGLToneMappingImplementation
	implements PostProcessPassImplementation<WebGLToneMappingContext, EmptyOptions>
{
	public readonly id = "tonemap:webgl";

	public warmup(context: WebGLToneMappingContext | undefined): void {
		context?.programs.getToneMappingProgram();
	}

	public execute(
		_request: PostProcessPassRequest<EmptyOptions>,
		context: WebGLToneMappingContext | undefined
	): PostProcessPassResult {
		const target = resolveWebGLTarget(context);
		if (!target) {
			return { ran: false };
		}
		const gl = context.gl;
		const program = context.programs.getToneMappingProgram();
		bindWebGLPostTarget(context, program.program, target.texture);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, target.source);
		if (program.uniforms.sourceMap) {
			gl.uniform1i(program.uniforms.sourceMap, 0);
		}
		context.drawFullscreen();
		gl.bindVertexArray(null);
		context.publishColorTexture(target.texture);
		return { ran: true };
	}
}

export class WebGLColorFilterImplementation
	implements PostProcessPassImplementation<WebGLColorFilterContext, ColorFilterOptions>
{
	public readonly id = "color-filter:webgl";

	public warmup(context: WebGLColorFilterContext | undefined): void {
		context?.programs.getColorFilterProgram();
	}

	public execute(
		request: PostProcessPassRequest<ColorFilterOptions>,
		context: WebGLColorFilterContext | undefined
	): PostProcessPassResult {
		const target = resolveWebGLTarget(context);
		if (!target) {
			return { ran: false };
		}
		const options = request.options;
		const brightness = sanitizeFiniteClamped(
			options?.brightness,
			DEFAULT_COLOR_FILTER_OPTIONS.brightness,
			-1,
			1
		);
		const saturation = sanitizeFiniteClamped(
			options?.saturation,
			DEFAULT_COLOR_FILTER_OPTIONS.saturation,
			0,
			2
		);
		const contrast = sanitizeFiniteClamped(
			options?.contrast,
			DEFAULT_COLOR_FILTER_OPTIONS.contrast,
			0,
			2
		);
		const temperature = sanitizeFiniteClamped(
			options?.temperature,
			DEFAULT_COLOR_FILTER_OPTIONS.temperature,
			-1,
			1
		);
		const tint = sanitizeFiniteClamped(
			options?.tint,
			DEFAULT_COLOR_FILTER_OPTIONS.tint,
			-1,
			1
		);

		const gl = context.gl;
		const program = context.programs.getColorFilterProgram();
		bindWebGLPostTarget(context, program.program, target.texture);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, target.source);
		if (program.uniforms.sourceMap) {
			gl.uniform1i(program.uniforms.sourceMap, 0);
		}
		if (program.uniforms.filterParams0) {
			gl.uniform4f(
				program.uniforms.filterParams0,
				brightness,
				saturation,
				contrast,
				temperature
			);
		}
		if (program.uniforms.filterParams1) {
			gl.uniform4f(program.uniforms.filterParams1, tint, 0, 0, 0);
		}
		context.drawFullscreen();
		gl.bindVertexArray(null);
		context.publishColorTexture(target.texture);
		return { ran: true };
	}
}

export class WebGLInteractionOutlineImplementation
	implements PostProcessPassImplementation<WebGLInteractionOutlineContext, EmptyOptions>
{
	public readonly id = "interaction-outline:webgl";
	private readonly _circleData = new Float32Array(
		MAX_INTERACTION_OUTLINE_CIRCLES * 4
	);

	public warmup(context: WebGLInteractionOutlineContext | undefined): void {
		context?.programs.getInteractionOutlineProgram();
	}

	public execute(
		request: PostProcessPassRequest<EmptyOptions>,
		context: WebGLInteractionOutlineContext | undefined
	): PostProcessPassResult {
		const state = request.frameContext.transient.get(
			INTERACTION_TRANSIENT_STATE_KEY
		);
		const selectedEntityIds = state?.selectedEntityIds ?? [];
		if (selectedEntityIds.length === 0) {
			return { ran: false };
		}
		const circles = collectProjectedOutlineCircles(
			request.frameContext,
			selectedEntityIds,
			MAX_INTERACTION_OUTLINE_CIRCLES
		);
		if (circles.length === 0) {
			return { ran: false };
		}
		const target = resolveWebGLTarget(context);
		if (!target) {
			return { ran: false };
		}

		let writeOffset = 0;
		for (const circle of circles) {
			this._circleData[writeOffset] = circle.centerX;
			this._circleData[writeOffset + 1] = circle.centerY;
			this._circleData[writeOffset + 2] = circle.radius;
			this._circleData[writeOffset + 3] = 0;
			writeOffset += 4;
		}

		const outlineColor = state?.outline?.color ?? { r: 255, g: 196, b: 64, a: 1 };
		const colorScale =
			Math.max(outlineColor.r, outlineColor.g, outlineColor.b) > 1 ? 255 : 1;
		const linearR = sRGBToLinear(
			clamp(outlineColor.r / Math.max(1, colorScale), 0, 1)
		);
		const linearG = sRGBToLinear(
			clamp(outlineColor.g / Math.max(1, colorScale), 0, 1)
		);
		const linearB = sRGBToLinear(
			clamp(outlineColor.b / Math.max(1, colorScale), 0, 1)
		);
		const alpha = clamp(
			finiteOr(state?.outline?.opacity, 0.9) * finiteOr(outlineColor.a, 1),
			0,
			1
		);
		const thickness = Math.max(1, finiteOr(state?.outline?.thickness, 2));
		const shapeCode = getInteractionOutlineShapeCode(state?.outline?.shape);

		const gl = context.gl;
		const program = context.programs.getInteractionOutlineProgram();
		bindWebGLPostTarget(context, program.program, target.texture);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, target.source);
		if (program.uniforms.sourceMap) {
			gl.uniform1i(program.uniforms.sourceMap, 0);
		}
		if (program.uniforms.outlineColor) {
			gl.uniform4f(program.uniforms.outlineColor, linearR, linearG, linearB, 1);
		}
		if (program.uniforms.outlineParams) {
			gl.uniform3f(program.uniforms.outlineParams, alpha, thickness, shapeCode);
		}
		if (program.uniforms.viewportSize) {
			gl.uniform2f(
				program.uniforms.viewportSize,
				Math.max(1, context.width),
				Math.max(1, context.height)
			);
		}
		if (program.uniforms.circleCount) {
			gl.uniform1i(program.uniforms.circleCount, circles.length);
		}
		if (program.uniforms.circles) {
			gl.uniform4fv(
				program.uniforms.circles,
				this._circleData.subarray(0, circles.length * 4)
			);
		}
		context.drawFullscreen();
		gl.bindVertexArray(null);
		context.publishColorTexture(target.texture);
		return { ran: true };
	}
}

export class WebGLGammaImplementation
	implements PostProcessPassImplementation<WebGLGammaContext, EmptyOptions>
{
	public readonly id = "gamma:webgl";

	public warmup(context: WebGLGammaContext | undefined): void {
		context?.programs.getPresentProgram();
	}

	public execute(
		request: PostProcessPassRequest<EmptyOptions>,
		context: WebGLGammaContext | undefined
	): PostProcessPassResult {
		const sourceTexture = context?.getSourceTexture();
		if (!context?.fullscreenVao || !sourceTexture) {
			return { ran: false };
		}
		const gl = context.gl;
		const program = context.programs.getPresentProgram();
		gl.bindFramebuffer(gl.FRAMEBUFFER, null);
		gl.viewport(0, 0, context.width, context.height);
		gl.useProgram(program.program);
		gl.bindVertexArray(context.fullscreenVao);
		gl.disable(gl.CULL_FACE);
		gl.disable(gl.DEPTH_TEST);
		gl.disable(gl.BLEND);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, sourceTexture);
		if (program.uniforms.sourceMap) {
			gl.uniform1i(program.uniforms.sourceMap, 0);
		}
		if (program.uniforms.applyGamma) {
			gl.uniform1i(
				program.uniforms.applyGamma,
				request.frameContext.postProcess.isEnabled(GAMMA_PASS_ID) ? 1 : 0
			);
		}
		context.drawFullscreen();
		gl.bindVertexArray(null);
		context.markPresented();
		return { ran: true };
	}
}

export interface MotionBlurPassConfig
	extends Omit<
		PostProcessPassConfig<MotionBlurOptions>,
		| "id"
		| "builtIn"
		| "warningLabel"
		| "placement"
		| "order"
		| "implementations"
	> {}

/**
 * Stateful logical motion blur pass.
 */
export class MotionBlurPass extends PostProcessPass<
	MotionBlurOptions,
	MotionBlurOptions
> {
	public constructor(config: MotionBlurPassConfig = {}) {
		super({
			...config,
			id: MOTION_BLUR_PASS_ID,
			builtIn: true,
			warningLabel: "motion blur",
			placement: "camera",
			order: 400,
			implementations: {
				webgpu: new WebGPUMotionBlurImplementation(),
				webgl: new WebGLMotionBlurImplementation(),
			},
		});
	}

	public override normalizeOptions(): MotionBlurOptions {
		return {
			...DEFAULT_MOTION_BLUR_OPTIONS,
			...this.getRawOptions(),
		};
	}

	public override getRequirements(): PostProcessPassRequirements {
		return { gBuffer: ["depth", "motion"] };
	}
}

export interface DepthOfFieldPassConfig
	extends Omit<
		PostProcessPassConfig<DOFOptions>,
		| "id"
		| "builtIn"
		| "warningLabel"
		| "placement"
		| "order"
		| "implementations"
	> {}

/**
 * Stateful logical depth of field pass.
 */
export class DepthOfFieldPass extends PostProcessPass<DOFOptions, DOFOptions> {
	public constructor(config: DepthOfFieldPassConfig = {}) {
		super({
			...config,
			id: DEPTH_OF_FIELD_PASS_ID,
			builtIn: true,
			warningLabel: "depth of field",
			placement: "camera",
			order: 410,
			implementations: {
				webgpu: new WebGPUDepthOfFieldImplementation(),
				webgl: new WebGLDepthOfFieldImplementation(),
			},
		});
	}

	public override normalizeOptions(): DOFOptions {
		return {
			...DEFAULT_DOF_OPTIONS,
			...this.getRawOptions(),
		};
	}

	public override getRequirements(): PostProcessPassRequirements {
		return { gBuffer: ["depth"] };
	}
}

/**
 * Stateful logical tone mapping pass.
 */
export class ToneMappingPass extends PostProcessPass<EmptyOptions, EmptyOptions> {
	public constructor(
		config: Omit<
			PostProcessPassConfig<EmptyOptions>,
			| "id"
			| "builtIn"
			| "warningLabel"
			| "placement"
			| "order"
			| "implementations"
		> = {}
	) {
		super({
			...config,
			id: TONE_MAPPING_PASS_ID,
			builtIn: true,
			warningLabel: "tone mapping",
			placement: "hdr",
			order: 600,
			implementations: {
				software: new SoftwareToneMappingImplementation(),
				webgpu: new WebGPUToneMappingImplementation(),
				webgl: new WebGLToneMappingImplementation(),
			},
		});
	}
}

export interface ColorFilterPassConfig
	extends Omit<
		PostProcessPassConfig<ColorFilterOptions>,
		| "id"
		| "builtIn"
		| "warningLabel"
		| "placement"
		| "order"
		| "implementations"
	> {}

/**
 * Stateful logical color filter pass.
 */
export class ColorFilterPass extends PostProcessPass<
	ColorFilterOptions,
	ColorFilterOptions
> {
	public constructor(config: ColorFilterPassConfig = {}) {
		super({
			...config,
			id: COLOR_FILTER_PASS_ID,
			builtIn: true,
			warningLabel: "color filter",
			placement: "ldr",
			order: 700,
			implementations: {
				software: new SoftwareColorFilterImplementation(),
				webgpu: new WebGPUColorFilterImplementation(),
				webgl: new WebGLColorFilterImplementation(),
			},
		});
	}

	public override normalizeOptions(): ColorFilterOptions {
		return {
			...DEFAULT_COLOR_FILTER_OPTIONS,
			...this.getRawOptions(),
		};
	}
}

/**
 * Stateful logical interaction outline pass.
 */
export class InteractionOutlinePass extends PostProcessPass<
	EmptyOptions,
	EmptyOptions
> {
	public constructor(
		config: Omit<
			PostProcessPassConfig<EmptyOptions>,
			| "id"
			| "builtIn"
			| "warningLabel"
			| "placement"
			| "order"
			| "implementations"
		> = {}
	) {
		super({
			...config,
			id: INTERACTION_OUTLINE_PASS_ID,
			builtIn: true,
			warningLabel: "interaction outline",
			placement: "overlay",
			order: 800,
			implementations: {
				software: new SoftwareInteractionOutlineImplementation(),
				webgpu: new WebGPUInteractionOutlineImplementation(),
				webgl: new WebGLInteractionOutlineImplementation(),
			},
		});
	}

	public override shouldExecute(
		request: PostProcessPassResolveRequest<EmptyOptions>
	): boolean {
		if (!request.frameContext) {
			return true;
		}
		const interactionState = request.frameContext.transient.get(
			INTERACTION_TRANSIENT_STATE_KEY
		);
		return (interactionState?.selectedEntityIds?.length ?? 0) > 0;
	}
}

/**
 * Stateful logical gamma correction pass.
 */
export class GammaPass extends PostProcessPass<EmptyOptions, EmptyOptions> {
	public constructor(
		config: Omit<
			PostProcessPassConfig<EmptyOptions>,
			| "id"
			| "builtIn"
			| "warningLabel"
			| "placement"
			| "order"
			| "implementations"
		> = {}
	) {
		super({
			...config,
			id: GAMMA_PASS_ID,
			builtIn: true,
			warningLabel: "gamma correction",
			placement: "present",
			order: 900,
			implementations: {
				software: new SoftwareGammaImplementation(),
				webgpu: new WebGPUGammaImplementation(),
				webgl: new WebGLGammaImplementation(),
			},
		});
	}
}

function resolveWebGPUTarget(
	targets: WebGPUPostProcessFrameTargets
): IRenderTexture {
	return targets.sceneColor === targets.postPong ?
			targets.postPing
		:	targets.postPong;
}

function publishWebGPUColorTarget(
	context: WebGPUScreenPostProcessContext,
	texture: IRenderTexture
): void {
	if (context.publishColorTarget) {
		context.publishColorTarget(texture);
	}
}

function uploadWebGPUMotionBlurParams(
	shared: PostProcessSharedContext,
	resources: WebGPUMotionBlurResources,
	width: number,
	height: number,
	shutterScale: number,
	maxSamples: number,
	velocityClamp: number,
	depthReject: number,
	centerWeight: number
): void {
	if (!resources.params) {
		return;
	}
	const data = resources.paramData;
	let changed = !resources.paramUploaded;
	changed =
		setWebGPUParamIfChanged(data, 0, 1 / Math.max(width, 1)) || changed;
	changed =
		setWebGPUParamIfChanged(data, 1, 1 / Math.max(height, 1)) || changed;
	changed = setWebGPUParamIfChanged(data, 2, shutterScale) || changed;
	changed = setWebGPUParamIfChanged(data, 3, maxSamples) || changed;
	changed = setWebGPUParamIfChanged(data, 4, velocityClamp) || changed;
	changed = setWebGPUParamIfChanged(data, 5, depthReject) || changed;
	changed = setWebGPUParamIfChanged(data, 6, centerWeight) || changed;
	changed = setWebGPUParamIfChanged(data, 7, 0) || changed;
	if (!changed) {
		return;
	}
	shared.compute.writeBuffer(resources.params, data);
	resources.paramUploaded = true;
}

function setWebGPUParamIfChanged(
	data: Float32Array<ArrayBuffer>,
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

interface ResolvedWebGLTarget {
	readonly source: WebGLTexture;
	readonly texture: WebGLTexture;
}

function resolveWebGLTarget(
	context: WebGLScreenPostProcessContext | undefined
): ResolvedWebGLTarget | null {
	if (
		!context?.postFramebuffer ||
		!context.sceneColorTexture ||
		!context.fullscreenVao
	) {
		return null;
	}
	const source = context.getSourceTexture();
	if (!source) {
		return null;
	}
	const texture = context.resolveTargetTexture(source);
	if (!texture) {
		return null;
	}
	return { source, texture };
}

function bindWebGLPostTarget(
	context: WebGLScreenPostProcessContext,
	program: WebGLProgram,
	targetTexture: WebGLTexture
): void {
	const gl = context.gl;
	gl.bindFramebuffer(gl.FRAMEBUFFER, context.postFramebuffer);
	context.bindColorTarget(targetTexture);
	gl.viewport(0, 0, context.width, context.height);
	gl.useProgram(program);
	gl.bindVertexArray(context.fullscreenVao);
	gl.disable(gl.CULL_FACE);
	gl.disable(gl.DEPTH_TEST);
	gl.disable(gl.BLEND);
}
