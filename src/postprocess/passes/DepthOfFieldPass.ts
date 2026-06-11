import { clamp } from "../../maths/Common";
import { ceilDiv, finiteOr } from "../../maths/Misc";
import {
	BufferUsage,
	type IComputePipeline,
	type IRenderBuffer,
	type IShaderModule,
} from "../../renderers/types";
import {
	WEBGPU_2D_COMPUTE_WORKGROUP_SIZE as WEBGPU_WORKGROUP_SIZE,
} from "../../renderers/webgpu/constants";
import {
	WEBGPU_SCREEN_POST_PROCESS_CONTEXT_METADATA,
} from "../../renderers/webgpu/WebGPUPostProcessContracts";
import type { PostProcessSharedContext } from "../../renderers/webgpu/postprocess/PostProcessSharedContext";
import {
	DOF_CHROMATIC_ABERRATION_RANGE,
	DOF_DEPTH_CURVE_RANGE,
	DOF_HIGHLIGHT_GAIN_RANGE,
	DOF_MAX_BLUR_RADIUS_RANGE,
	DOF_NEAR_FAR_STRENGTH_RANGE,
} from "../../renderers/webgl/constants";
import { sanitizeFiniteClamped } from "../../renderers/webgl/WebGLFrameMath";
import { ShaderSource } from "../../shaders/ShaderSource";
import { PostProcessPass, type PostProcessPassConfig } from "../PostProcessPass";
import type { PostProcessPassMetadata } from "../ordering";
import type {
	PostProcessPassImplementation,
	PostProcessPassRequest,
	PostProcessPassRequirements,
	PostProcessPassResult,
} from "../types";
import {
	bindWebGLPostTarget,
	publishWebGPUColorTarget,
	resolveWebGLTarget,
	resolveWebGPUTarget,
	type WebGLScreenPostProcessContext,
	type WebGPURuntimePostProcessContext,
} from "./ScreenPassShared";

export const DEPTH_OF_FIELD_PASS_ID = "dof";
export const DEPTH_OF_FIELD_PASS_ORDER = {
	id: DEPTH_OF_FIELD_PASS_ID,
	placement: "camera",
	order: 410,
	incremental: {
		firstPass: "dof",
		grade: "cinematic",
		inflationRadius: 32,
	},
} as const satisfies PostProcessPassMetadata;
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
export type WebGPUDepthOfFieldContext = WebGPURuntimePostProcessContext;
export type WebGLDepthOfFieldContext = WebGLScreenPostProcessContext;

interface WebGPUDepthOfFieldResources {
	shared: PostProcessSharedContext;
	module: IShaderModule | null;
	pipeline: IComputePipeline | null;
	params: IRenderBuffer | null;
	paramData: Float32Array<ArrayBuffer>;
}
/** @internal WebGPU implementation for the built-in depth-of-field pass. */
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
/** @internal WebGL implementation for the built-in depth-of-field pass. */
export class WebGLDepthOfFieldImplementation
	implements PostProcessPassImplementation<WebGLDepthOfFieldContext, DOFOptions>
{
	public readonly id = "dof:webgl";

	public warmup(context: WebGLDepthOfFieldContext | undefined): void {
		context?.programs.warmupDOFProgram();
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
		const program = context.programs.tryGetDOFProgram();
		if (!program) {
			return { ran: false };
		}
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
			incremental: config.incremental ?? DEPTH_OF_FIELD_PASS_ORDER.incremental,
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
