import {
	INTERACTION_TRANSIENT_STATE_KEY,
	type FrameContext,
} from "../../pipeline/types";
import { clamp, linearToSRGB, sRGBToLinear } from "../../maths/Common";
import { ceilDiv, finiteOr } from "../../maths/Misc";
import {
	MAX_INTERACTION_OUTLINE_CIRCLES,
	collectProjectedOutlineCircles,
} from "../../interaction/outlineProjection";
import {
	computeInteractionOutlineShapeDistance,
	getInteractionOutlineShapeCode,
	resolveInteractionOutlineShape,
} from "../../interaction/outlineShape";
import type { ICommandEncoder } from "../../renderers/ICommandEncoder";
import { DEFAULT_GAMMA } from "../../renderers/constants";
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
import { ShaderSource } from "../../shaders/ShaderSource";
import {
	PostProcessPass,
	type PostProcessPassConfig,
	type PostProcessPassResolveRequest,
} from "../PostProcessPass";
import { getRequiredBuiltinPostProcessOrderMetadata } from "../builtinMetadata";
import type {
	PostProcessPassImplementation,
	PostProcessPassRequest,
	PostProcessPassRequirements,
	PostProcessPassResult,
} from "../types";

type EmptyOptions = Record<string, never>;

export const MOTION_BLUR_PASS_ID = "motion-blur";
export const DEPTH_OF_FIELD_PASS_ID = "dof";
export const TONE_MAPPING_PASS_ID = "tonemap";
export const COLOR_FILTER_PASS_ID = "color-filter";
export const INTERACTION_OUTLINE_PASS_ID = "interaction-outline";
export const GAMMA_PASS_ID = "gamma";
export const MOTION_BLUR_PASS_ORDER =
	getRequiredBuiltinPostProcessOrderMetadata(MOTION_BLUR_PASS_ID);
export const DEPTH_OF_FIELD_PASS_ORDER =
	getRequiredBuiltinPostProcessOrderMetadata(DEPTH_OF_FIELD_PASS_ID);
export const TONE_MAPPING_PASS_ORDER =
	getRequiredBuiltinPostProcessOrderMetadata(TONE_MAPPING_PASS_ID);
export const COLOR_FILTER_PASS_ORDER =
	getRequiredBuiltinPostProcessOrderMetadata(COLOR_FILTER_PASS_ID);
export const INTERACTION_OUTLINE_PASS_ORDER =
	getRequiredBuiltinPostProcessOrderMetadata(INTERACTION_OUTLINE_PASS_ID);
export const GAMMA_PASS_ORDER =
	getRequiredBuiltinPostProcessOrderMetadata(GAMMA_PASS_ID);

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

interface IncrementalDirtyRect {
	minX: number;
	minY: number;
	maxX: number;
	maxY: number;
}

export class SoftwareToneMappingImplementation
	implements PostProcessPassImplementation<SoftwareBuiltinPostProcessContext>
{
	public readonly id = "tonemap:software";

	public execute(
		request: PostProcessPassRequest,
		_context: SoftwareBuiltinPostProcessContext | undefined
	): PostProcessPassResult {
		const pixels = request.frameContext.attachments.pixels;
		if (!pixels || pixels.length === 0) {
			return { ran: false };
		}
		const dirtyRects = resolveSoftwareDirtyRects(request.frameContext);
		const width = request.frameContext.attachments.width;
		forEachSoftwareDirtyRect(dirtyRects, (rect) => {
			for (let y = rect.minY; y <= rect.maxY; y++) {
				const row = y * width;
				for (let x = rect.minX; x <= rect.maxX; x++) {
					const index = (row + x) << 2;
					const red = applyAcesToneMap(pixels[index] / 255);
					const green = applyAcesToneMap(pixels[index + 1] / 255);
					const blue = applyAcesToneMap(pixels[index + 2] / 255);
					pixels[index] = Math.round(red * 255);
					pixels[index + 1] = Math.round(green * 255);
					pixels[index + 2] = Math.round(blue * 255);
				}
			}
		});
		return { ran: true };
	}
}

export class SoftwareColorFilterImplementation
	implements PostProcessPassImplementation<SoftwareBuiltinPostProcessContext>
{
	public readonly id = "color-filter:software";

	public execute(
		request: PostProcessPassRequest,
		_context: SoftwareBuiltinPostProcessContext | undefined
	): PostProcessPassResult {
		const pixels = request.frameContext.attachments.pixels;
		if (!pixels || pixels.length === 0) {
			return { ran: false };
		}
		const options = {
			...DEFAULT_COLOR_FILTER_OPTIONS,
			...((request.options as ColorFilterOptions | undefined) ?? {}),
		};
		const dirtyRects = resolveSoftwareDirtyRects(request.frameContext);
		const brightness = clamp(finiteOr(options.brightness, 0), -1, 1);
		const saturation = clamp(finiteOr(options.saturation, 1), 0, 2);
		const contrast = clamp(finiteOr(options.contrast, 1), 0, 2);
		const temperature = clamp(finiteOr(options.temperature, 0), -1, 1);
		const tint = clamp(finiteOr(options.tint, 0), -1, 1);
		const tempShiftR = temperature * 0.1 + tint * 0.05;
		const tempShiftG = -tint * 0.1;
		const tempShiftB = -temperature * 0.1 + tint * 0.05;
		const width = request.frameContext.attachments.width;
		forEachSoftwareDirtyRect(dirtyRects, (rect) => {
			for (let y = rect.minY; y <= rect.maxY; y++) {
				const row = y * width;
				for (let x = rect.minX; x <= rect.maxX; x++) {
					const index = (row + x) << 2;
					let red = pixels[index] / 255;
					let green = pixels[index + 1] / 255;
					let blue = pixels[index + 2] / 255;

					red += brightness;
					green += brightness;
					blue += brightness;

					const luma = red * 0.2126 + green * 0.7152 + blue * 0.0722;
					red = luma + (red - luma) * saturation;
					green = luma + (green - luma) * saturation;
					blue = luma + (blue - luma) * saturation;

					red = (red - 0.5) * contrast + 0.5;
					green = (green - 0.5) * contrast + 0.5;
					blue = (blue - 0.5) * contrast + 0.5;

					red += tempShiftR;
					green += tempShiftG;
					blue += tempShiftB;

					pixels[index] = Math.round(clamp(red, 0, 1) * 255);
					pixels[index + 1] = Math.round(clamp(green, 0, 1) * 255);
					pixels[index + 2] = Math.round(clamp(blue, 0, 1) * 255);
				}
			}
		});
		return { ran: true };
	}
}

export class SoftwareInteractionOutlineImplementation
	implements PostProcessPassImplementation<SoftwareBuiltinPostProcessContext>
{
	public readonly id = "interaction-outline:software";

	public execute(
		request: PostProcessPassRequest,
		_context: SoftwareBuiltinPostProcessContext | undefined
	): PostProcessPassResult {
		const frameContext = request.frameContext;
		const state = frameContext.transient.get(INTERACTION_TRANSIENT_STATE_KEY);
		const pixels = frameContext.attachments.pixels;
		if (!pixels || pixels.length === 0) {
			return { ran: false };
		}
		if (!state || state.selectedEntityIds.length === 0) {
			return { ran: true };
		}
		const width = Math.max(1, frameContext.attachments.width);
		const height = Math.max(1, frameContext.attachments.height);
		const dirtyRects = resolveSoftwareDirtyRects(frameContext);
		const outlineColor = state.outline?.color ?? { r: 255, g: 196, b: 64, a: 1 };
		const alpha = clamp(
			(state.outline?.opacity ?? 0.9) *
				(typeof outlineColor.a === "number" ? outlineColor.a : 1),
			0,
			1
		);
		const thickness = Math.max(1, Math.round(state.outline?.thickness ?? 2));
		const outlineShape = resolveInteractionOutlineShape(state.outline?.shape);
		const circles = collectProjectedOutlineCircles(
			frameContext,
			state.selectedEntityIds
		);
		for (const circle of circles) {
			const circleMinX = Math.max(
				0,
				Math.floor(circle.centerX - circle.radius - thickness)
			);
			const circleMinY = Math.max(
				0,
				Math.floor(circle.centerY - circle.radius - thickness)
			);
			const circleMaxX = Math.min(
				width - 1,
				Math.ceil(circle.centerX + circle.radius + thickness)
			);
			const circleMaxY = Math.min(
				height - 1,
				Math.ceil(circle.centerY + circle.radius + thickness)
			);
			if (
				!softwareRectIntersectsDirtyRects(
					circleMinX,
					circleMinY,
					circleMaxX,
					circleMaxY,
					dirtyRects
				)
			) {
				continue;
			}
			drawSoftwareInteractionOutlineShape(
				pixels,
				width,
				height,
				circle.centerX,
				circle.centerY,
				circle.radius,
				thickness,
				outlineShape,
				outlineColor.r,
				outlineColor.g,
				outlineColor.b,
				alpha,
				dirtyRects
			);
		}
		return { ran: true };
	}
}

export class SoftwareGammaImplementation
	implements PostProcessPassImplementation<SoftwareBuiltinPostProcessContext>
{
	public readonly id = "gamma:software";
	private readonly _sRGBLUT = new Uint8Array(256);
	private _lutBuilt = false;
	private _lastGamma = -1;

	public execute(
		request: PostProcessPassRequest,
		context: SoftwareBuiltinPostProcessContext | undefined
	): PostProcessPassResult {
		const width = request.frameContext.attachments.width;
		const height = request.frameContext.attachments.height;
		const canvasContext = context?.canvasContext ?? null;
		let pixels = request.frameContext.attachments.pixels;
		let imageData: ImageData | null = null;
		if (!pixels) {
			if (!canvasContext) {
				return { ran: false };
			}
			imageData = canvasContext.getImageData(0, 0, width, height);
			pixels = imageData.data;
		}
		if (pixels.length === 0) {
			return { ran: false };
		}
		const gamma = request.frameContext.postProcess.isEnabled(GAMMA_PASS_ID)
			? DEFAULT_GAMMA
			: 1;
		const dirtyRects = resolveSoftwareDirtyRects(request.frameContext);
		this._buildSRGBLUT(gamma);
		const lut = this._sRGBLUT;
		forEachSoftwareDirtyRect(dirtyRects, (rect) => {
			for (let y = rect.minY; y <= rect.maxY; y++) {
				const row = y * width;
				for (let x = rect.minX; x <= rect.maxX; x++) {
					const i = (row + x) << 2;
					pixels[i] = lut[pixels[i]];
					pixels[i + 1] = lut[pixels[i + 1]];
					pixels[i + 2] = lut[pixels[i + 2]];
				}
			}
		});
		if (imageData && canvasContext) {
			canvasContext.putImageData(imageData, 0, 0);
		}
		return { ran: true };
	}

	private _buildSRGBLUT(gamma: number): void {
		if (this._lutBuilt && this._lastGamma === gamma) {
			return;
		}
		const isStandardSRGB = Math.abs(gamma - DEFAULT_GAMMA) < 0.001;
		const invGamma = 1 / gamma;
		for (let i = 0; i < 256; i++) {
			const value = i / 255;
			this._sRGBLUT[i] =
				isStandardSRGB ?
					Math.round(linearToSRGB(value) * 255)
				:	Math.round(Math.pow(value, invGamma) * 255);
		}
		this._lutBuilt = true;
		this._lastGamma = gamma;
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
			const shader = await ShaderSource.load(
				"webgpu.postprocess.motionBlur.composite"
			);
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
			resources.pipeline = await shared.compute.createComputePipeline({
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
			const shader = await ShaderSource.load("webgpu.postprocess.dof.composite");
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
			resources.pipeline = await shared.compute.createComputePipeline({
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
			const shader = await ShaderSource.load(
				"webgpu.postprocess.toneMapping.composite"
			);
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
			resources.pipeline = await shared.compute.createComputePipeline({
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
			const shader = await ShaderSource.load(
				"webgpu.postprocess.colorFilter.composite"
			);
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
			resources.pipeline = await shared.compute.createComputePipeline({
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
			const shader = await ShaderSource.load(
				"webgpu.postprocess.interactionOutline.composite"
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
			resources.pipeline = await shared.compute.createComputePipeline({
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
			...MOTION_BLUR_PASS_ORDER,
			builtIn: true,
			warningLabel: "motion blur",
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
			...DEPTH_OF_FIELD_PASS_ORDER,
			builtIn: true,
			warningLabel: "depth of field",
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
			...TONE_MAPPING_PASS_ORDER,
			builtIn: true,
			warningLabel: "tone mapping",
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
			...COLOR_FILTER_PASS_ORDER,
			builtIn: true,
			warningLabel: "color filter",
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
			...INTERACTION_OUTLINE_PASS_ORDER,
			builtIn: true,
			warningLabel: "interaction outline",
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
			...GAMMA_PASS_ORDER,
			builtIn: true,
			warningLabel: "gamma correction",
			implementations: {
				software: new SoftwareGammaImplementation(),
				webgpu: new WebGPUGammaImplementation(),
				webgl: new WebGLGammaImplementation(),
			},
		});
	}
}

function resolveSoftwareDirtyRects(context: FrameContext): IncrementalDirtyRect[] {
	const width = Math.max(1, context.attachments.width);
	const height = Math.max(1, context.attachments.height);
	const incremental = context.incremental;
	if (
		!incremental.enabled ||
		incremental.forceFullFrame ||
		incremental.dirtyRects.length === 0
	) {
		return [{ minX: 0, minY: 0, maxX: width - 1, maxY: height - 1 }];
	}
	const dirtyRects: IncrementalDirtyRect[] = [];
	for (const rect of incremental.dirtyRects) {
		const minX = Math.max(0, Math.floor(rect.x));
		const minY = Math.max(0, Math.floor(rect.y));
		const maxX = Math.min(width - 1, Math.ceil(rect.x + rect.width) - 1);
		const maxY = Math.min(height - 1, Math.ceil(rect.y + rect.height) - 1);
		if (minX > maxX || minY > maxY) {
			continue;
		}
		dirtyRects.push({ minX, minY, maxX, maxY });
	}
	return dirtyRects;
}

function forEachSoftwareDirtyRect(
	dirtyRects: IncrementalDirtyRect[],
	callback: (rect: IncrementalDirtyRect) => void
): void {
	for (const rect of dirtyRects) {
		if (rect.minX > rect.maxX || rect.minY > rect.maxY) {
			continue;
		}
		callback(rect);
	}
}

function softwareRectIntersectsDirtyRects(
	minX: number,
	minY: number,
	maxX: number,
	maxY: number,
	dirtyRects: IncrementalDirtyRect[]
): boolean {
	for (const rect of dirtyRects) {
		if (
			maxX >= rect.minX &&
			minX <= rect.maxX &&
			maxY >= rect.minY &&
			minY <= rect.maxY
		) {
			return true;
		}
	}
	return false;
}

function applyAcesToneMap(value: number): number {
	const a = 2.51;
	const b = 0.03;
	const c = 2.43;
	const d = 0.59;
	const e = 0.14;
	const mapped = (value * (a * value + b)) / (value * (c * value + d) + e);
	return clamp(mapped, 0, 1);
}

function drawSoftwareInteractionOutlineShape(
	pixels: Uint8ClampedArray,
	width: number,
	height: number,
	centerX: number,
	centerY: number,
	radius: number,
	thickness: number,
	shape: "circle" | "square" | "diamond" | "octagon",
	red: number,
	green: number,
	blue: number,
	alpha: number,
	dirtyRects: IncrementalDirtyRect[]
): void {
	const minX = Math.max(0, Math.floor(centerX - radius - thickness));
	const minY = Math.max(0, Math.floor(centerY - radius - thickness));
	const maxX = Math.min(width - 1, Math.ceil(centerX + radius + thickness));
	const maxY = Math.min(height - 1, Math.ceil(centerY + radius + thickness));
	const inner = Math.max(0, radius - thickness * 0.5);
	const outer = radius + thickness * 0.5;

	for (let y = minY; y <= maxY; y++) {
		for (let x = minX; x <= maxX; x++) {
			if (!softwareRectIntersectsDirtyRects(x, y, x, y, dirtyRects)) {
				continue;
			}
			const dx = x + 0.5 - centerX;
			const dy = y + 0.5 - centerY;
			const shapeDistance = computeInteractionOutlineShapeDistance(
				dx,
				dy,
				shape
			);
			if (shapeDistance < inner || shapeDistance > outer) {
				continue;
			}
			const index = (y * width + x) << 2;
			const invAlpha = 1 - alpha;
			pixels[index] = Math.round(pixels[index] * invAlpha + red * alpha);
			pixels[index + 1] = Math.round(
				pixels[index + 1] * invAlpha + green * alpha
			);
			pixels[index + 2] = Math.round(
				pixels[index + 2] * invAlpha + blue * alpha
			);
			pixels[index + 3] = 255;
		}
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
