import { CameraType } from "../../cameras/Camera";
import type { ICommandEncoder } from "../../renderers/ICommandEncoder";
import {
	BufferUsage,
	type IBindingGroup,
	type IComputePipeline,
	type IRenderBuffer,
	type IRenderTexture,
	type IShaderModule,
} from "../../renderers/types";
import {
	WEBGPU_2D_COMPUTE_WORKGROUP_SIZE as WORKGROUP_SIZE,
} from "../../renderers/webgpu/constants";
import type { WebGPUPostProcessFrameTargets } from "../../renderers/webgpu/WebGPUPostProcessContracts";
import type { PostProcessSharedContext } from "../../renderers/webgpu/postprocess/PostProcessSharedContext";
import { ceilDiv, finiteOr } from "../../maths/Misc";
import { ShaderSource } from "../../shaders/ShaderSource";
import {
	PostProcessPass,
	type PostProcessPassConfig,
	type PostProcessPassResolveRequest,
} from "../PostProcessPass";
import type { PostProcessPassMetadata } from "../ordering";
import type {
	PostProcessHistoryDescriptor,
	PostProcessHistorySlots,
	PostProcessPassImplementation,
	PostProcessPassRequest,
	PostProcessPassResult,
	PostProcessPassRequirements,
	PostProcessTransientDescriptor,
} from "../types";

const DEFAULT_HISTORY_USAGE = ["sampled", "storage", "render-target"] as const;
const MOTION_HISTORY_USAGE = ["sampled", "copy-dst", "render-target"] as const;
export const SCREEN_SPACE_REFLECTIONS_PASS_ID = "ssr";
export const SCREEN_SPACE_REFLECTIONS_PASS_ORDER = {
	id: SCREEN_SPACE_REFLECTIONS_PASS_ID,
	placement: "temporal",
	order: 210,
	incremental: {
		firstPass: "ssr",
		grade: "cinematic",
		inflationRadius: 16,
	},
} as const satisfies PostProcessPassMetadata;
const WEBGPU_SSR_RAW_TRANSIENT_ID = "ssr:raw";

export interface SSROptions {
	/** Maximum ray-march iterations per reflection ray. */
	maxSteps?: number;
	/** Maximum view/world-space ray distance for screen-space reflections. */
	maxDistance?: number;
	/** Depth thickness tolerance used when matching ray hits to surfaces. */
	thickness?: number;
	/** Ray step stride. Higher values improve speed but can skip thin details. */
	stride?: number;
	/** Reflection contribution multiplier mixed into the scene color. */
	intensity?: number;
	/** Temporal history blend factor. Higher values stabilize but can ghost. */
	historyWeight?: number;
	/** Internal trace buffer scale divisor. Higher values improve speed. */
	downsample?: number;
	/** Refinement iterations after a ray hit is found. */
	binarySearchSteps?: number;
	/** Screen-edge fade distance that hides reflections near missing data. */
	edgeFade?: number;
	/** Maximum material roughness that may receive SSR. */
	maxRoughness?: number;
	/** Allows backend-specific experimental SSR options. */
	[key: string]: unknown;
}

export const DEFAULT_SSR_OPTIONS: Required<
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

export type ResolvedSSROptions = Required<
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
>;

/** @internal WebGPU context supplied to the built-in SSR implementation. */
export interface WebGPUSSRContext {
	readonly encoder?: ICommandEncoder;
	readonly targets?: WebGPUPostProcessFrameTargets;
	readonly shared: PostProcessSharedContext;
	readonly frameBinding?: IBindingGroup;
	readonly ssrRaw?: IRenderTexture | null;
	readonly hiZ?: IRenderTexture | null;
	readonly historyRead?: IRenderTexture | null;
	readonly historyWrite?: IRenderTexture | null;
	readonly motionHistoryRead?: IRenderTexture | null;
	readonly motionHistoryWrite?: IRenderTexture | null;
	publishColorTarget?(texture: IRenderTexture): void;
	writeMotionHistoryFromCurrent?(): void | Promise<void>;
}

interface WebGPUSSRResources {
	shared: PostProcessSharedContext;
	module: IShaderModule | null;
	tracePipeline: IComputePipeline | null;
	composePipeline: IComputePipeline | null;
	traceParams: IRenderBuffer | null;
	composeParams: IRenderBuffer | null;
	traceGroupLayout0: GPUBindGroupLayout | null;
	tracePipelineLayout: GPUPipelineLayout | null;
	frameIndex: number;
}

/**
 * Resolves SSR options with backend-independent clamping for history sizing.
 *
 * @param options User-provided SSR options.
 * @returns Fully resolved SSR options.
 * @sideEffects None.
 */
export function resolveSSROptions(options?: SSROptions | null): ResolvedSSROptions {
	return {
		downsample: clampDownsample(options?.downsample, DEFAULT_SSR_OPTIONS.downsample),
		maxSteps: finiteOr(options?.maxSteps, DEFAULT_SSR_OPTIONS.maxSteps),
		binarySearchSteps: finiteOr(
			options?.binarySearchSteps,
			DEFAULT_SSR_OPTIONS.binarySearchSteps
		),
		maxDistance: finiteOr(options?.maxDistance, DEFAULT_SSR_OPTIONS.maxDistance),
		thickness: finiteOr(options?.thickness, DEFAULT_SSR_OPTIONS.thickness),
		stride: finiteOr(options?.stride, DEFAULT_SSR_OPTIONS.stride),
		intensity: finiteOr(options?.intensity, DEFAULT_SSR_OPTIONS.intensity),
		historyWeight: finiteOr(
			options?.historyWeight,
			DEFAULT_SSR_OPTIONS.historyWeight
		),
		edgeFade: finiteOr(options?.edgeFade, DEFAULT_SSR_OPTIONS.edgeFade),
		maxRoughness: finiteOr(
			options?.maxRoughness,
			DEFAULT_SSR_OPTIONS.maxRoughness
		),
	};
}

/**
 * Resolves whether SSR may sample temporal history this frame.
 *
 * @param histories Current frame history slots.
 * @returns `true` when SSR color and motion histories are valid.
 * @sideEffects None.
 */
export function resolveSSRHistoryValid(histories: PostProcessHistorySlots): boolean {
	return (histories.ssr?.valid ?? false) && (histories.motion?.valid ?? false);
}

/**
 * Creates packed SSR trace shader parameters.
 *
 * @param width Trace target width.
 * @param height Trace target height.
 * @param options Resolved SSR options.
 * @param maxHiZMip Maximum available Hi-Z mip index.
 * @param historyValid Whether temporal history can be sampled.
 * @param frameIndex Temporal stochastic sampling frame index.
 * @returns Sixteen float parameters expected by the SSR trace shader.
 * @sideEffects None.
 */
export function createSSRTraceParams(
	width: number,
	height: number,
	options: ResolvedSSROptions,
	maxHiZMip: number,
	historyValid: boolean,
	frameIndex: number
): Float32Array {
	return new Float32Array([
		1 / Math.max(width, 1),
		1 / Math.max(height, 1),
		options.maxDistance,
		options.thickness,
		options.stride,
		options.intensity,
		options.maxRoughness,
		options.edgeFade,
		options.maxSteps,
		options.binarySearchSteps,
		maxHiZMip,
		options.historyWeight,
		historyValid ? 1 : 0,
		0.02,
		frameIndex,
		0,
	]);
}

/**
 * Resolves dynamic history resources required by SSR.
 *
 * @param request History resolution request for the current frame.
 * @returns SSR color history at the configured downsample plus full-size motion history.
 * @sideEffects None.
 */
export function resolveSSRHistoryDescriptors(
	request: PostProcessPassResolveRequest<ResolvedSSROptions>
): readonly PostProcessHistoryDescriptor[] {
	const options = resolveSSROptions(request.options);
	const scale = 1 / options.downsample;
	return [
		{
			id: "ssr",
			widthScale: scale,
			heightScale: scale,
			usage: DEFAULT_HISTORY_USAGE,
		},
		{ id: "motion", usage: MOTION_HISTORY_USAGE },
	];
}

/**
 * WebGPU implementation of the cross-backend screen-space reflections pass.
 */
/** @internal WebGPU implementation for the built-in SSR pass. */
export class WebGPUScreenSpaceReflectionsImplementation
	implements PostProcessPassImplementation<WebGPUSSRContext>
{
	public readonly id = "ssr:webgpu";
	public readonly metadata = {
		context: {
			backend: "webgpu",
			kind: "screen",
			publishColorTarget: true,
			frameBinding: true,
			histories: [
				{ property: "historyRead", historyId: "ssr", side: "read" },
				{ property: "historyWrite", historyId: "ssr", side: "write" },
				{ property: "motionHistoryRead", historyId: "motion", side: "read" },
				{ property: "motionHistoryWrite", historyId: "motion", side: "write" },
			],
			transients: [
				{
					property: "ssrRaw",
					transientId: WEBGPU_SSR_RAW_TRANSIENT_ID,
				},
			],
			requiresHiZ: true,
			motionHistoryCopy: {
				writeProperty: "motionHistoryWrite",
			},
		},
	} as const;
	private _resources = new Map<PostProcessSharedContext, WebGPUSSRResources>();

	public async warmup(context: WebGPUSSRContext | undefined): Promise<void> {
		if (context) {
			await this._ensureResources(context.shared);
		}
	}

	public async execute(
		request: PostProcessPassRequest,
		context: WebGPUSSRContext | undefined
	): Promise<PostProcessPassResult> {
		if (!context?.encoder || !context.targets || !context.frameBinding) {
			return { ran: false };
		}
		if (request.frameContext.viewCamera.type === CameraType.Orthographic) {
			context.shared.warn(
				"webgpu-ssr-orthographic-disabled",
				"WebGPU SSR is disabled for orthographic cameras."
			);
			return { ran: false };
		}
		const ran = await this._runSSRKernel(request, context);
		if (!ran) {
			return { ran: false };
		}
		await context.writeMotionHistoryFromCurrent?.();
		return { ran: true, updatedHistoryIds: ["ssr", "motion"] };
	}

	public invalidate(): void {
		for (const resources of this._resources.values()) {
			resources.shared.invalidateBindingsByPrefix("ssr-");
		}
	}

	public destroy(): void {
		for (const resources of this._resources.values()) {
			resources.shared.destroyManagedResource(
				resources.tracePipeline,
				"SSR trace pipeline"
			);
			resources.shared.destroyManagedResource(
				resources.composePipeline,
				"SSR compose pipeline"
			);
			resources.shared.destroyManagedResource(
				resources.module,
				"SSR shader module"
			);
			resources.shared.destroyManagedResource(
				resources.traceParams,
				"SSR trace params buffer"
			);
			resources.shared.destroyManagedResource(
				resources.composeParams,
				"SSR compose params buffer"
			);
			resources.shared.invalidateBindingsByPrefix("ssr-");
			resources.module = null;
			resources.tracePipeline = null;
			resources.composePipeline = null;
			resources.traceParams = null;
			resources.composeParams = null;
			resources.traceGroupLayout0 = null;
			resources.tracePipelineLayout = null;
		}
		this._resources.clear();
	}

	private async _runSSRKernel(
		request: PostProcessPassRequest,
		context: WebGPUSSRContext
	): Promise<boolean> {
		const resources = await this._ensureResources(context.shared);
		if (
			!context.encoder ||
			!context.targets ||
			!context.frameBinding ||
			!context.ssrRaw ||
			!context.hiZ ||
			!context.shared.sampler ||
			!resources.tracePipeline ||
			!resources.composePipeline ||
			!resources.traceParams ||
			!resources.composeParams ||
			!context.historyRead ||
			!context.historyWrite ||
			!context.motionHistoryRead ||
			!context.motionHistoryWrite
		) {
			return false;
		}

		const targets = context.targets;
		const ssrRaw = context.ssrRaw;
		const hiZ = context.hiZ;
		const hiZMips = context.shared.getHiZBuilder().getMipViews(hiZ);
		if (hiZMips.length === 0) {
			return false;
		}
		resources.frameIndex = (resources.frameIndex + 1) % 1024;
		const options = resolveSSROptions(request.options as SSROptions);
		context.shared.compute.writeBuffer(
			resources.traceParams,
			createSSRTraceParams(
				ssrRaw.width,
				ssrRaw.height,
				options,
				hiZMips.length - 1,
				resolveSSRHistoryValid(request.histories),
				resources.frameIndex
			) as unknown as BufferSource
		);

		let binding = context.shared.getCachedBindGroup(
			"ssr-trace",
			resources.tracePipeline,
			[
				{ binding: 0, resource: targets.sceneColor },
				{ binding: 1, resource: targets.gNormalRoughMetal },
				{ binding: 2, resource: targets.gMotionDepth },
				{ binding: 3, resource: hiZ },
				{ binding: 4, resource: context.historyRead },
				{ binding: 5, resource: context.motionHistoryRead },
				{ binding: 6, resource: context.shared.sampler },
				{ binding: 7, resource: resources.traceParams },
				{ binding: 8, resource: ssrRaw },
			],
			"WebGPUSSR_TraceBinding"
		);
		context.encoder.beginComputePass({ label: "WebGPUSSR_TraceTemporal" });
		context.encoder.setComputePipeline(resources.tracePipeline);
		context.encoder.setBindingGroup(0, binding);
		context.encoder.setBindingGroup(1, context.frameBinding);
		context.encoder.dispatchWorkgroups(
			ceilDiv(ssrRaw.width, WORKGROUP_SIZE),
			ceilDiv(ssrRaw.height, WORKGROUP_SIZE),
			1
		);
		context.encoder.endComputePass();

		await context.shared.getCopyHelper().copyTexture({
			encoder: context.encoder,
			source: ssrRaw,
			destination: context.historyWrite,
			cacheKey: "copy-ssr-history",
			label: "WebGPUPost_Copy",
		});

		const composeTarget =
			targets.sceneColor === targets.postPing ? targets.postPong : targets.postPing;
		context.shared.compute.writeBuffer(
			resources.composeParams,
			new Float32Array([
				1 / Math.max(composeTarget.width, 1),
				1 / Math.max(composeTarget.height, 1),
				0,
				0,
			]) as unknown as BufferSource
		);
		binding = context.shared.getCachedBindGroup(
			`ssr-compose-${composeTarget === targets.postPing ? "ping" : "pong"}`,
			resources.composePipeline,
			[
				{ binding: 0, resource: targets.sceneColor },
				{ binding: 1, resource: ssrRaw },
				{ binding: 2, resource: targets.gMotionDepth },
				{ binding: 3, resource: context.shared.sampler },
				{ binding: 4, resource: resources.composeParams },
				{ binding: 5, resource: composeTarget },
				{ binding: 6, resource: targets.planarReflectionMask },
			],
			"WebGPUSSR_ComposeBinding"
		);
		context.encoder.beginComputePass({ label: "WebGPUSSR_Compose" });
		context.encoder.setComputePipeline(resources.composePipeline);
		context.encoder.setBindingGroup(0, binding);
		context.encoder.dispatchWorkgroups(
			ceilDiv(composeTarget.width, WORKGROUP_SIZE),
			ceilDiv(composeTarget.height, WORKGROUP_SIZE),
			1
		);
		context.encoder.endComputePass();
		context.publishColorTarget?.(composeTarget);
		return true;
	}

	private async _ensureResources(
		shared: PostProcessSharedContext
	): Promise<WebGPUSSRResources> {
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
				frameIndex: 0,
			};
			this._resources.set(shared, resources);
		}
		await shared.ensureCommonResources();
		await shared.getHiZBuilder().ensureResources();
		if (!resources.module) {
			const shader = await ShaderSource.load("webgpu.postprocess.ssr.composite");
			resources.module = await shared.compute.createShaderModule({
				label: "WebGPUSSRShader",
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
				resources.tracePipelineLayout = shared.compute.createPipelineLayout({
					label: "WebGPUSSRTrace_PipelineLayout",
					bindGroupLayouts: [
						resources.traceGroupLayout0,
						shared.frameBindGroupLayout,
					],
				});
				resources.tracePipeline = await shared.compute.createComputePipeline({
					label: "WebGPUSSRTracePipeline",
					layout: resources.tracePipelineLayout,
					compute: { module: resources.module, entryPoint: "csTrace" },
				});
			} else {
				resources.tracePipeline = await shared.compute.createComputePipeline({
					label: "WebGPUSSRTracePipeline",
					compute: { module: resources.module, entryPoint: "csTrace" },
				});
			}
		}
		if (!resources.composePipeline) {
			resources.composePipeline = await shared.compute.createComputePipeline({
				label: "WebGPUSSRComposePipeline",
				compute: { module: resources.module, entryPoint: "csCompose" },
			});
		}
		if (!resources.traceParams) {
			resources.traceParams = shared.compute.createBuffer({
				label: "WebGPUSSRTraceParams",
				size: 16 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
		}
		if (!resources.composeParams) {
			resources.composeParams = shared.compute.createBuffer({
				label: "WebGPUSSRComposeParams",
				size: 4 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
		}
		return resources;
	}
}

export interface ScreenSpaceReflectionsPassConfig
	extends Omit<
		PostProcessPassConfig<SSROptions>,
		| "id"
		| "builtIn"
		| "warningLabel"
		| "placement"
		| "order"
		| "implementations"
	> {}

/**
 * Stateful logical screen-space reflections pass.
 */
export class ScreenSpaceReflectionsPass extends PostProcessPass<
	SSROptions,
	ResolvedSSROptions
> {
	public constructor(config: ScreenSpaceReflectionsPassConfig = {}) {
		super({
			...config,
			...SCREEN_SPACE_REFLECTIONS_PASS_ORDER,
			incremental:
				config.incremental ?? SCREEN_SPACE_REFLECTIONS_PASS_ORDER.incremental,
			warningLabel: "SSR",
			implementations: {
				webgpu: () => new WebGPUScreenSpaceReflectionsImplementation(),
			},
		});
	}

	public override normalizeOptions(): ResolvedSSROptions {
		return resolveSSROptions(this.getRawOptions());
	}

	public override getRequirements(): PostProcessPassRequirements {
		return { gBuffer: ["depth", "normal", "roughness", "metallic", "motion"] };
	}

	public override getHistoryDescriptors(
		request: PostProcessPassResolveRequest<ResolvedSSROptions>
	): readonly PostProcessHistoryDescriptor[] {
		return resolveSSRHistoryDescriptors(request);
	}

	public override getTransientResourceDescriptors(
		request: PostProcessPassResolveRequest<ResolvedSSROptions>
	): readonly PostProcessTransientDescriptor[] {
		if (request.backend !== "webgpu") {
			return [];
		}
		const options = resolveSSROptions(request.options);
		const scale = 1 / options.downsample;
		return [
			{
				id: WEBGPU_SSR_RAW_TRANSIENT_ID,
				widthScale: scale,
				heightScale: scale,
			},
		];
	}
}

function clampDownsample(value: unknown, fallback: number): number {
	if (typeof value !== "number" || !Number.isFinite(value)) {
		return fallback;
	}
	return Math.min(8, Math.max(1, Math.floor(value)));
}
