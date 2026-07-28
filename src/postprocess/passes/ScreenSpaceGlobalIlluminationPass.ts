import { CameraType } from "../../cameras/Camera";
import type { ICommandEncoder } from "../../backends/ICommandEncoder";
import {
	BufferUsage,
	type IBindingGroup,
	type IComputePipeline,
	type IRenderBuffer,
	type IRenderTexture,
	type IShaderModule,
} from "../../backends/types";
import {
	WEBGPU_2D_COMPUTE_WORKGROUP_SIZE as WORKGROUP_SIZE,
} from "../../backends/webgpu/constants";
import type {
	WebGPUPostProcessFrameTargets,
	WebGPUPostProcessServices,
} from "../../backends/webgpu/WebGPUPostProcessContracts";
import { clamp } from "../../maths/Common";
import { ceilDiv, finiteOr } from "../../maths/Misc";
import { ShaderSource } from "../../shaders/ShaderSource";
import {
	PostProcessPass,
	type PostProcessPassConfig,
	type PostProcessPassResolveRequest,
} from "../PostProcessPass";
import type { PostProcessScheduleEntry } from "../ordering";
import {
	POST_PROCESS_SAMPLED_READ,
	POST_PROCESS_STORAGE_WRITE,
	WEBGPU_HIZ_SHARED_RESOURCE,
	WEBGPU_VERSIONED_EXECUTION,
} from "../executionDeclarations";
import type {
	PostProcessExecutionDeclaration,
	PostProcessHistoryDescriptor,
	PostProcessHistorySlots,
	PostProcessPassImplementation,
	PostProcessPassRequest,
	PostProcessPassResult,
	PostProcessResourceAccessor,
	PostProcessTransientDescriptor,
} from "../types";

const DEFAULT_HISTORY_USAGE = ["sampled", "storage", "render-target"] as const;
const MOTION_HISTORY_USAGE = ["sampled", "copy-dst", "render-target"] as const;
const SSGI_HISTORY_ID = "ssgi";
const SSGI_DENOISE_A_ID = "ssgi:denoise-a";
const SSGI_DENOISE_B_ID = "ssgi:denoise-b";
const SSGI_TRACE_PARAM_FLOATS = 20;
const SSGI_DENOISE_PARAM_FLOATS = 8;
const SSGI_COMPOSE_PARAM_FLOATS = 8;

export const SCREEN_SPACE_GLOBAL_ILLUMINATION_PASS_ID = "ssgi";
export const SCREEN_SPACE_GLOBAL_ILLUMINATION_PASS_ORDER = {
	id: SCREEN_SPACE_GLOBAL_ILLUMINATION_PASS_ID,
	placement: "spatial",
	order: 110,
	incremental: {
		firstPass: "ssgi",
		grade: "cinematic",
		inflationRadius: 8,
	},
} as const satisfies PostProcessScheduleEntry;

export interface SSGIOptions {
	/** Internal trace-buffer divisor. Supported values are `1`, `2`, and `4`. */
	downsample?: 1 | 2 | 4;
	/** Cosine-weighted hemisphere rays traced per pixel. */
	raysPerPixel?: number;
	/** Maximum Hi-Z marching iterations per ray. */
	maxSteps?: number;
	/** Binary hit-refinement iterations after a ray reaches mip zero. */
	binarySearchSteps?: number;
	/** Maximum world-space trace distance. */
	maxDistance?: number;
	/** World-space depth tolerance used to accept ray hits. */
	thickness?: number;
	/** World-space surface offset used to suppress self-intersection. */
	normalBias?: number;
	/** Exponent applied to normalized ray-distance attenuation. */
	distanceFalloffExponent?: number;
	/** Screen-edge fade distance used to hide missing off-screen radiance. */
	edgeFade?: number;
	/** Indirect diffuse lighting multiplier. */
	intensity?: number;
	/** Temporal history blend factor. */
	historyWeight?: number;
	/** Relative depth threshold used to reject disoccluded history. */
	disocclusionDepthThreshold?: number;
	/** Maximum luminance ratio retained from valid temporal history. */
	historyClamp?: number;
	/** Radius of each separable bilateral denoise pass. */
	denoiseRadius?: number;
	/** Relative-depth edge sensitivity used by the denoiser. */
	denoiseDepthPhi?: number;
	/** World-normal edge sensitivity used by the denoiser. */
	denoiseNormalPhi?: number;
}

export const DEFAULT_SSGI_OPTIONS: Required<
	Pick<
		SSGIOptions,
		| "downsample"
		| "raysPerPixel"
		| "maxSteps"
		| "binarySearchSteps"
		| "maxDistance"
		| "thickness"
		| "normalBias"
		| "distanceFalloffExponent"
		| "edgeFade"
		| "intensity"
		| "historyWeight"
		| "disocclusionDepthThreshold"
		| "historyClamp"
		| "denoiseRadius"
		| "denoiseDepthPhi"
		| "denoiseNormalPhi"
	>
> = {
	downsample: 2,
	raysPerPixel: 1,
	maxSteps: 24,
	binarySearchSteps: 3,
	maxDistance: 8,
	thickness: 0.2,
	normalBias: 0.05,
	distanceFalloffExponent: 2,
	edgeFade: 0.1,
	intensity: 0.35,
	historyWeight: 0.9,
	disocclusionDepthThreshold: 0.02,
	historyClamp: 2,
	denoiseRadius: 2,
	denoiseDepthPhi: 24,
	denoiseNormalPhi: 16,
};

export type ResolvedSSGIOptions = Required<
	Pick<SSGIOptions, keyof typeof DEFAULT_SSGI_OPTIONS>
>;

/** @internal WebGPU context supplied to the built-in SSGI implementation. */
export interface WebGPUSSGIContext {
	readonly encoder?: ICommandEncoder;
	readonly targets?: WebGPUPostProcessFrameTargets;
	readonly shared: WebGPUPostProcessServices;
	readonly frameBinding?: IBindingGroup;
	readonly resources: PostProcessResourceAccessor<IRenderTexture>;
}

interface WebGPUSSGIResources {
	shared: WebGPUPostProcessServices;
	module: IShaderModule | null;
	tracePipeline: IComputePipeline | null;
	composePipeline: IComputePipeline | null;
	traceParams: IRenderBuffer | null;
	composeParams: IRenderBuffer | null;
	traceGroupLayout0: GPUBindGroupLayout | null;
	tracePipelineLayout: GPUPipelineLayout | null;
	traceParamData: Float32Array;
	composeParamData: Float32Array;
	frameIndex: number;
	historyContinuity: boolean;
}

/**
 * Resolves SSGI options with backend-independent clamping.
 *
 * @param options User-provided SSGI options.
 * @returns Fully resolved SSGI options.
 * @sideEffects None.
 */
export function resolveSSGIOptions(
	options?: SSGIOptions | null
): ResolvedSSGIOptions {
	const maxDistance = Math.max(
		0.001,
		positiveFiniteOr(
			options?.maxDistance,
			DEFAULT_SSGI_OPTIONS.maxDistance
		)
	);
	return {
		downsample: resolveSSGIDownsample(options?.downsample),
		raysPerPixel: clampInteger(
			options?.raysPerPixel,
			DEFAULT_SSGI_OPTIONS.raysPerPixel,
			1,
			4
		),
		maxSteps: clampInteger(
			options?.maxSteps,
			DEFAULT_SSGI_OPTIONS.maxSteps,
			4,
			64
		),
		binarySearchSteps: clampInteger(
			options?.binarySearchSteps,
			DEFAULT_SSGI_OPTIONS.binarySearchSteps,
			0,
			8
		),
		maxDistance,
		thickness: clamp(
			finiteOr(options?.thickness, DEFAULT_SSGI_OPTIONS.thickness),
			0.001,
			maxDistance
		),
		normalBias: Math.max(
			0,
			finiteOr(options?.normalBias, DEFAULT_SSGI_OPTIONS.normalBias)
		),
		distanceFalloffExponent: clamp(
			finiteOr(
				options?.distanceFalloffExponent,
				DEFAULT_SSGI_OPTIONS.distanceFalloffExponent
			),
			0.25,
			8
		),
		edgeFade: clamp(
			finiteOr(options?.edgeFade, DEFAULT_SSGI_OPTIONS.edgeFade),
			0,
			0.5
		),
		intensity: Math.max(
			0,
			finiteOr(options?.intensity, DEFAULT_SSGI_OPTIONS.intensity)
		),
		historyWeight: clamp(
			finiteOr(options?.historyWeight, DEFAULT_SSGI_OPTIONS.historyWeight),
			0,
			0.98
		),
		disocclusionDepthThreshold: clamp(
			finiteOr(
				options?.disocclusionDepthThreshold,
				DEFAULT_SSGI_OPTIONS.disocclusionDepthThreshold
			),
			0.001,
			0.25
		),
		historyClamp: clamp(
			finiteOr(options?.historyClamp, DEFAULT_SSGI_OPTIONS.historyClamp),
			1,
			16
		),
		denoiseRadius: clampInteger(
			options?.denoiseRadius,
			DEFAULT_SSGI_OPTIONS.denoiseRadius,
			1,
			4
		),
		denoiseDepthPhi: positiveFiniteOr(
			options?.denoiseDepthPhi,
			DEFAULT_SSGI_OPTIONS.denoiseDepthPhi
		),
		denoiseNormalPhi: positiveFiniteOr(
			options?.denoiseNormalPhi,
			DEFAULT_SSGI_OPTIONS.denoiseNormalPhi
		),
	};
}

/**
 * Packs trace and temporal parameters into a pre-allocated float array.
 *
 * @param target Twenty-float destination array.
 * @param width Trace target width.
 * @param height Trace target height.
 * @param options Resolved SSGI options.
 * @param maxHiZMip Maximum available Hi-Z mip index.
 * @param historyValid Whether temporal history may be sampled.
 * @param frameIndex Temporal stochastic sampling frame index.
 * @returns The supplied destination array.
 * @sideEffects Overwrites `target`.
 */
export function writeSSGITraceParams(
	target: Float32Array,
	width: number,
	height: number,
	options: ResolvedSSGIOptions,
	maxHiZMip: number,
	historyValid: boolean,
	frameIndex: number
): Float32Array {
	assertParamTarget(target, SSGI_TRACE_PARAM_FLOATS, "trace");
	target[0] = 1 / Math.max(width, 1);
	target[1] = 1 / Math.max(height, 1);
	target[2] = options.maxDistance;
	target[3] = options.thickness;
	target[4] = options.normalBias;
	target[5] = options.distanceFalloffExponent;
	target[6] = options.edgeFade;
	target[7] = options.raysPerPixel;
	target[8] = options.maxSteps;
	target[9] = options.binarySearchSteps;
	target[10] = Math.max(0, maxHiZMip);
	target[11] = frameIndex;
	target[12] = options.historyWeight;
	target[13] = options.disocclusionDepthThreshold;
	target[14] = options.historyClamp;
	target[15] = historyValid ? 1 : 0;
	target[16] = 0;
	target[17] = 0;
	target[18] = 0;
	target[19] = 0;
	return target;
}

/**
 * Packs the legacy SSGI denoise parameter layout.
 *
 * The shared WebGPU denoiser does not consume this layout. This helper remains
 * available so existing tooling that inspects resolved SSGI settings does not
 * break.
 *
 * @param target Eight-float destination array.
 * @param width Denoise target width.
 * @param height Denoise target height.
 * @param options Resolved SSGI options.
 * @returns The supplied destination array.
 * @sideEffects Overwrites `target`.
 */
export function writeSSGIDenoiseParams(
	target: Float32Array,
	width: number,
	height: number,
	options: ResolvedSSGIOptions
): Float32Array {
	assertParamTarget(target, SSGI_DENOISE_PARAM_FLOATS, "denoise");
	target[0] = 1 / Math.max(width, 1);
	target[1] = 1 / Math.max(height, 1);
	target[2] = options.denoiseRadius;
	target[3] = options.denoiseDepthPhi;
	target[4] = options.denoiseNormalPhi;
	target[5] = 0;
	target[6] = 0;
	target[7] = 0;
	return target;
}

/**
 * Packs full-resolution composition parameters into a pre-allocated array.
 *
 * @param target Eight-float destination array.
 * @param width Full-resolution output width.
 * @param height Full-resolution output height.
 * @param traceWidth Filtered SSGI width.
 * @param traceHeight Filtered SSGI height.
 * @param options Resolved SSGI options.
 * @returns The supplied destination array.
 * @sideEffects Overwrites `target`.
 */
export function writeSSGIComposeParams(
	target: Float32Array,
	width: number,
	height: number,
	traceWidth: number,
	traceHeight: number,
	options: ResolvedSSGIOptions
): Float32Array {
	assertParamTarget(target, SSGI_COMPOSE_PARAM_FLOATS, "compose");
	target[0] = 1 / Math.max(width, 1);
	target[1] = 1 / Math.max(height, 1);
	target[2] = 1 / Math.max(traceWidth, 1);
	target[3] = 1 / Math.max(traceHeight, 1);
	target[4] = options.intensity;
	target[5] = options.denoiseDepthPhi;
	target[6] = options.denoiseNormalPhi;
	target[7] = 0;
	return target;
}

/**
 * Resolves whether SSGI may consume temporal history this frame.
 *
 * @param histories Current frame history slots.
 * @param implementationContinuous Whether the previous implementation frame ran.
 * @returns `true` when radiance and motion history are valid and continuous.
 * @sideEffects None.
 */
export function resolveSSGIHistoryValid(
	histories: PostProcessHistorySlots,
	implementationContinuous: boolean
): boolean {
	return implementationContinuous &&
		(histories[SSGI_HISTORY_ID]?.valid ?? false) &&
		(histories.motion?.valid ?? false);
}

/**
 * Resolves dynamic history resources required by SSGI.
 *
 * @param request Current backend and resolved-option request.
 * @returns Half-resolution radiance history and full-resolution motion history.
 * @sideEffects None.
 */
export function resolveSSGIHistoryDescriptors(
	request: PostProcessPassResolveRequest<ResolvedSSGIOptions>
): readonly PostProcessHistoryDescriptor[] {
	const options = resolveSSGIOptions(request.options);
	const scale = 1 / options.downsample;
	return [
		{
			id: SSGI_HISTORY_ID,
			widthScale: scale,
			heightScale: scale,
			format: "rgba16float",
			usage: DEFAULT_HISTORY_USAGE,
		},
		{ id: "motion", usage: MOTION_HISTORY_USAGE },
	];
}

/**
 * Resolves half-resolution denoise resources required by SSGI.
 *
 * @param request Current backend and resolved-option request.
 * @returns Horizontal and vertical denoise targets.
 * @sideEffects None.
 */
export function resolveSSGITransientDescriptors(
	request: PostProcessPassResolveRequest<ResolvedSSGIOptions>
): readonly PostProcessTransientDescriptor[] {
	const options = resolveSSGIOptions(request.options);
	const scale = 1 / options.downsample;
	return [
		{
			id: SSGI_DENOISE_A_ID,
			widthScale: scale,
			heightScale: scale,
			format: "rgba16float",
		},
		{
			id: SSGI_DENOISE_B_ID,
			widthScale: scale,
			heightScale: scale,
			format: "rgba16float",
		},
	];
}

/** @internal WebGPU implementation for the built-in SSGI pass. */
export class WebGPUScreenSpaceGlobalIlluminationImplementation
	implements PostProcessPassImplementation<WebGPUSSGIContext>
{
	public readonly id = "ssgi:webgpu";
	private _resources = new Map<WebGPUPostProcessServices, WebGPUSSGIResources>();

	public describeExecution(
		request: PostProcessPassResolveRequest<ResolvedSSGIOptions>
	) {
		return {
			...WEBGPU_VERSIONED_EXECUTION,
			gBuffer: (
				["depth", "normal", "albedo", "metallic", "motion"] as const
			).map((semantic) => ({ semantic, ...POST_PROCESS_SAMPLED_READ })),
			histories: resolveSSGIHistoryDescriptors(request).map((descriptor) => ({
				descriptor,
				read: [POST_PROCESS_SAMPLED_READ],
				write:
					descriptor.id === SSGI_HISTORY_ID ?
						[POST_PROCESS_STORAGE_WRITE, POST_PROCESS_SAMPLED_READ]
					:	[POST_PROCESS_STORAGE_WRITE],
			})),
			transients: resolveSSGITransientDescriptors(request).map(
				(descriptor) => ({
					descriptor,
					uses: [POST_PROCESS_STORAGE_WRITE, POST_PROCESS_SAMPLED_READ],
				})
			),
			shared: [WEBGPU_HIZ_SHARED_RESOURCE],
		} satisfies PostProcessExecutionDeclaration;
	}

	public async warmup(context: WebGPUSSGIContext | undefined): Promise<void> {
		if (context) {
			await this._ensureResources(context.shared);
		}
	}

	public async execute(
		request: PostProcessPassRequest,
		context: WebGPUSSGIContext | undefined
	): Promise<PostProcessPassResult> {
		if (!context?.encoder || !context.targets || !context.frameBinding) {
			this._breakHistoryContinuity(context?.shared);
			return { ran: false };
		}
		if (request.frameContext.viewCamera.type === CameraType.Orthographic) {
			context.shared.warn(
				"webgpu-ssgi-orthographic-disabled",
				"WebGPU SSGI is disabled for orthographic cameras."
			);
			this._breakHistoryContinuity(context.shared);
			return { ran: false };
		}
		let ran = false;
		try {
			ran = await this._runSSGIKernel(request, context);
			if (ran) {
				await context.resources.copyGBufferToHistory("motion", "motion");
			}
		} catch (error) {
			this._breakHistoryContinuity(context.shared);
			throw error;
		}
		if (!ran) {
			this._breakHistoryContinuity(context.shared);
			return { ran: false };
		}
		const resources = this._resources.get(context.shared);
		if (resources) {
			resources.historyContinuity = true;
		}
		return { ran: true, updatedHistoryIds: [SSGI_HISTORY_ID, "motion"] };
	}

	public invalidate(): void {
		for (const resources of this._resources.values()) {
			resources.historyContinuity = false;
			this._destroyResources(resources);
		}
	}

	public destroy(): void {
		for (const resources of this._resources.values()) {
			this._destroyResources(resources);
		}
		this._resources.clear();
	}

	private async _runSSGIKernel(
		request: PostProcessPassRequest,
		context: WebGPUSSGIContext
	): Promise<boolean> {
		const resources = await this._ensureResources(context.shared);
		const hiZ = context.resources.getShared("backend:frame-hiz");
		const history = context.resources.getHistory(SSGI_HISTORY_ID);
		const motionHistory = context.resources.getHistory("motion");
		const denoiseA = context.resources.getTransient(SSGI_DENOISE_A_ID);
		const denoiseB = context.resources.getTransient(SSGI_DENOISE_B_ID);
		const normalTexture = context.resources.getGBuffer("normal");
		const depthTexture = context.resources.getGBuffer("depth");
		const albedoTexture = context.resources.getGBuffer("albedo");
		const motionTexture = context.resources.getGBuffer("motion");
		const input = context.resources.color.input;
		const output = context.resources.color.output;
		if (
			!context.encoder ||
			!context.targets ||
			!context.frameBinding ||
			!hiZ ||
			!history.read ||
			!history.write ||
			!motionHistory.read ||
			!motionHistory.write ||
			!denoiseA ||
			!denoiseB ||
			!normalTexture ||
			!depthTexture ||
			!albedoTexture ||
			!motionTexture ||
			!input ||
			!output ||
			!context.shared.sampler ||
			!resources.tracePipeline ||
			!resources.composePipeline ||
			!resources.traceParams ||
			!resources.composeParams
		) {
			return false;
		}

		const hiZMips = context.shared.getHiZBuilder().getMipViews(hiZ);
		if (hiZMips.length === 0) {
			return false;
		}
		resources.frameIndex = (resources.frameIndex + 1) % 1024;
		const options = resolveSSGIOptions(request.options as SSGIOptions);
		const historyValid = resolveSSGIHistoryValid(
			request.histories,
			resources.historyContinuity
		);
		context.shared.compute.writeBuffer(
			resources.traceParams,
			writeSSGITraceParams(
				resources.traceParamData,
				history.write.width,
				history.write.height,
				options,
				hiZMips.length - 1,
				historyValid,
				resources.frameIndex
			) as unknown as BufferSource
		);
		context.shared.compute.writeBuffer(
			resources.composeParams,
			writeSSGIComposeParams(
				resources.composeParamData,
				output.width,
				output.height,
				denoiseB.width,
				denoiseB.height,
				options
			) as unknown as BufferSource
		);

		const parity = resources.frameIndex & 1;
		let binding = context.shared.getCachedBindGroup(
			`ssgi-trace-${parity}`,
			resources.tracePipeline,
			[
				{ binding: 0, resource: input },
				{ binding: 1, resource: normalTexture },
				{ binding: 2, resource: depthTexture },
				{ binding: 3, resource: hiZ },
				{ binding: 4, resource: history.read },
				{ binding: 5, resource: motionHistory.read },
				{ binding: 6, resource: context.shared.sampler },
				{ binding: 7, resource: resources.traceParams },
				{ binding: 8, resource: history.write },
			],
			"WebGPUSSGI_TraceBinding"
		);
		context.encoder.beginComputePass({ label: "WebGPUSSGI_TraceTemporal" });
		context.encoder.setComputePipeline(resources.tracePipeline);
		context.encoder.setBindingGroup(0, binding);
		context.encoder.setBindingGroup(1, context.frameBinding);
		context.encoder.dispatchWorkgroups(
			ceilDiv(history.write.width, WORKGROUP_SIZE),
			ceilDiv(history.write.height, WORKGROUP_SIZE),
			1
		);
		context.encoder.endComputePass();

		const denoiseResult = await context.shared.getDenoiser().encode({
			scope: "ssgi",
			encoder: context.encoder,
			source: history.write,
			scratch: denoiseA,
			output: denoiseB,
			depth: depthTexture,
			normal: normalTexture,
			sampler: context.shared.sampler,
			options: {
				mode: "quality",
				signal: "radiance-confidence",
				radius: options.denoiseRadius,
				depthPhi: options.denoiseDepthPhi,
				normalPhi: options.denoiseNormalPhi,
				valuePhi: 2,
				confidenceFloor: 0.05,
			},
		});

		const targets = context.targets;
		binding = context.shared.getCachedBindGroup(
			`ssgi-compose-${output === targets.postPing ? "ping" : "pong"}`,
			resources.composePipeline,
			[
				{ binding: 0, resource: input },
				{ binding: 1, resource: denoiseResult.texture },
				{ binding: 2, resource: albedoTexture },
				{ binding: 3, resource: normalTexture },
				{ binding: 4, resource: depthTexture },
				{ binding: 5, resource: context.shared.sampler },
				{ binding: 6, resource: resources.composeParams },
				{ binding: 7, resource: output },
			],
			"WebGPUSSGI_ComposeBinding"
		);
		context.encoder.beginComputePass({ label: "WebGPUSSGI_Compose" });
		context.encoder.setComputePipeline(resources.composePipeline);
		context.encoder.setBindingGroup(0, binding);
		context.encoder.dispatchWorkgroups(
			ceilDiv(output.width, WORKGROUP_SIZE),
			ceilDiv(output.height, WORKGROUP_SIZE),
			1
		);
		context.encoder.endComputePass();
		return true;
	}

	private async _ensureResources(
		shared: WebGPUPostProcessServices
	): Promise<WebGPUSSGIResources> {
		let resources = this._resources.get(shared);
		if (!resources) {
			resources = {
				shared,
				module: null,
				tracePipeline: null,
				composePipeline: null,
				traceParams: null,
				composeParams: null,
				traceGroupLayout0: null,
				tracePipelineLayout: null,
				traceParamData: new Float32Array(SSGI_TRACE_PARAM_FLOATS),
				composeParamData: new Float32Array(SSGI_COMPOSE_PARAM_FLOATS),
				frameIndex: 0,
				historyContinuity: false,
			};
			this._resources.set(shared, resources);
		}
		await shared.ensureCommonResources();
		await shared.getHiZBuilder().ensureResources();
		await shared.getDenoiser().ensureResources();
		if (!resources.module) {
			const shader = await ShaderSource.load(
				"webgpu.postprocess.ssgi.composite"
			);
			resources.module = await shared.compute.createShaderModule({
				label: "WebGPUSSGIShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!resources.tracePipeline) {
			if (shared.frameBindGroupLayout) {
				resources.traceGroupLayout0 = shared.compute.createBindGroupLayout({
					label: "WebGPUSSGITrace_GroupLayout0",
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
							storageTexture: {
								format: "rgba16float",
								access: "write-only",
							},
						},
					],
				});
				resources.tracePipelineLayout = shared.compute.createPipelineLayout({
					label: "WebGPUSSGITrace_PipelineLayout",
					bindGroupLayouts: [
						resources.traceGroupLayout0,
						shared.frameBindGroupLayout,
					],
				});
				resources.tracePipeline = await shared.compute.createComputePipeline({
					label: "WebGPUSSGITracePipeline",
					layout: resources.tracePipelineLayout,
					compute: {
						module: resources.module,
						entryPoint: "csTraceTemporal",
					},
				});
			} else {
				resources.tracePipeline = await shared.compute.createComputePipeline({
					label: "WebGPUSSGITracePipeline",
					compute: {
						module: resources.module,
						entryPoint: "csTraceTemporal",
					},
				});
			}
		}
		if (!resources.composePipeline) {
			resources.composePipeline = await shared.compute.createComputePipeline({
				label: "WebGPUSSGIComposePipeline",
				compute: { module: resources.module, entryPoint: "csCompose" },
			});
		}
		if (!resources.traceParams) {
			resources.traceParams = shared.compute.createBuffer({
				label: "WebGPUSSGITraceParams",
				size: SSGI_TRACE_PARAM_FLOATS * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
		}
		if (!resources.composeParams) {
			resources.composeParams = shared.compute.createBuffer({
				label: "WebGPUSSGIComposeParams",
				size: SSGI_COMPOSE_PARAM_FLOATS * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
		}
		return resources;
	}

	private _breakHistoryContinuity(
		shared: WebGPUPostProcessServices | undefined
	): void {
		if (!shared) return;
		const resources = this._resources.get(shared);
		if (resources) {
			resources.historyContinuity = false;
		}
	}

	private _destroyResources(resources: WebGPUSSGIResources): void {
		resources.shared.destroyManagedResource(
			resources.tracePipeline,
			"SSGI trace pipeline"
		);
		resources.shared.destroyManagedResource(
			resources.composePipeline,
			"SSGI compose pipeline"
		);
		resources.shared.destroyManagedResource(
			resources.module,
			"SSGI shader module"
		);
		resources.shared.destroyManagedResource(
			resources.traceParams,
			"SSGI trace params buffer"
		);
		resources.shared.destroyManagedResource(
			resources.composeParams,
			"SSGI compose params buffer"
		);
		resources.shared.invalidateBindingsByPrefix("ssgi-");
		resources.module = null;
		resources.tracePipeline = null;
		resources.composePipeline = null;
		resources.traceParams = null;
		resources.composeParams = null;
		resources.traceGroupLayout0 = null;
		resources.tracePipelineLayout = null;
	}
}

export interface ScreenSpaceGlobalIlluminationPassConfig
	extends Omit<
		PostProcessPassConfig<SSGIOptions>,
		| "id"
		| "builtIn"
		| "label"
		| "placement"
		| "order"
		| "implementations"
	> {}

/**
 * Stateful logical SSGI pass shared by supported rendering backends.
 */
export class ScreenSpaceGlobalIlluminationPass extends PostProcessPass<
	SSGIOptions,
	ResolvedSSGIOptions
> {
	constructor(config: ScreenSpaceGlobalIlluminationPassConfig = {}) {
		super({
			...config,
			id: SCREEN_SPACE_GLOBAL_ILLUMINATION_PASS_ORDER.id,
			schedule: {
				placement:
					config.schedule?.placement ??
					SCREEN_SPACE_GLOBAL_ILLUMINATION_PASS_ORDER.placement,
				order: config.schedule?.order ?? SCREEN_SPACE_GLOBAL_ILLUMINATION_PASS_ORDER.order,
				incremental:
					config.schedule?.incremental ??
					SCREEN_SPACE_GLOBAL_ILLUMINATION_PASS_ORDER.incremental,
			},
			label: "SSGI",
			implementations: {
				webgpu: () => new WebGPUScreenSpaceGlobalIlluminationImplementation(),
			},
		});
	}

	public override normalizeOptions(): ResolvedSSGIOptions {
		return resolveSSGIOptions(this.getRawOptions());
	}
}

function resolveSSGIDownsample(value: unknown): 1 | 2 | 4 {
	const finite = finiteOr(value, DEFAULT_SSGI_OPTIONS.downsample);
	if (finite <= 1) return 1;
	if (finite <= 2) return 2;
	return 4;
}

function clampInteger(
	value: unknown,
	fallback: number,
	minimum: number,
	maximum: number
): number {
	return clamp(Math.floor(finiteOr(value, fallback)), minimum, maximum);
}

function positiveFiniteOr(value: unknown, fallback: number): number {
	const finite = finiteOr(value, fallback);
	return finite > 0 ? finite : fallback;
}

function assertParamTarget(
	target: Float32Array,
	expectedLength: number,
	label: string
): void {
	if (target.length !== expectedLength) {
		throw new Error(
			`SSGI ${label} parameter target must contain ${expectedLength} floats.`
		);
	}
}
