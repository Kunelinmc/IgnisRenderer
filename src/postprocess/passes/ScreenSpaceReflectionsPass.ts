import { CameraType } from "../../cameras/Camera";
import {
	DEFAULT_SSR_OPTIONS,
	type SSROptions,
} from "../../pipeline/types";
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
import type { WebGPUFrameTargets } from "../../renderers/webgpu/WebGPUPostProcessContracts";
import { WebGPUHiZPostProcessHelper } from "../../renderers/webgpu/postprocess/HiZPostProcessHelper";
import type { PostProcessSharedContext } from "../../renderers/webgpu/postprocess/PostProcessSharedContext";
import { ceilDiv, finiteOr } from "../../maths/Misc";
import { loadPostProcessShaderPartComposite } from "../../shaders/webgpu/shaderSource";
import {
	PostProcessPass,
	type PostProcessPassConfig,
	type PostProcessPassResolveRequest,
} from "../PostProcessPass";
import type {
	PostProcessHistoryDescriptor,
	PostProcessHistorySlots,
	PostProcessPassImplementation,
	PostProcessPassRequest,
	PostProcessPassResult,
	PostProcessPassRequirements,
} from "../types";

const DEFAULT_HISTORY_USAGE = ["sampled", "storage", "render-target"] as const;
const MOTION_HISTORY_USAGE = ["sampled", "copy-dst", "render-target"] as const;
export const SCREEN_SPACE_REFLECTIONS_PASS_ID = "ssr";

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

export interface WebGPUSSRContext {
	readonly encoder?: ICommandEncoder;
	readonly targets?: WebGPUFrameTargets;
	readonly shared: PostProcessSharedContext;
	readonly frameBinding?: IBindingGroup;
	publishColorTarget?(texture: IRenderTexture): void;
	writeMotionHistoryFromCurrent?(): void | Promise<void>;
}

interface WebGPUSSRResources {
	hiz: WebGPUHiZPostProcessHelper;
	module: IShaderModule | null;
	tracePipeline: IComputePipeline | null;
	composePipeline: IComputePipeline | null;
	traceParams: IRenderBuffer | null;
	composeParams: IRenderBuffer | null;
	copyModule: IShaderModule | null;
	copyPipeline: IComputePipeline | null;
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
export class WebGPUScreenSpaceReflectionsImplementation
	implements PostProcessPassImplementation<WebGPUSSRContext>
{
	public readonly id = "ssr:webgpu";
	private _resources = new WeakMap<PostProcessSharedContext, WebGPUSSRResources>();

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
		if (request.frameContext.camera.type === CameraType.Orthographic) {
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

	private async _runSSRKernel(
		request: PostProcessPassRequest,
		context: WebGPUSSRContext
	): Promise<boolean> {
		const resources = await this._ensureResources(context.shared);
		if (
			!context.encoder ||
			!context.targets ||
			!context.frameBinding ||
			!context.shared.sampler ||
			!resources.tracePipeline ||
			!resources.composePipeline ||
			!resources.traceParams ||
			!resources.composeParams
		) {
			return false;
		}

		const targets = context.targets;
		const hiZMips = await resources.hiz.build(context.encoder, targets);
		if (hiZMips.length === 0) {
			return false;
		}
		resources.frameIndex = (resources.frameIndex + 1) % 1024;
		const options = resolveSSROptions(request.options as SSROptions);
		context.shared.compute.writeBuffer(
			resources.traceParams,
			createSSRTraceParams(
				targets.ssrRaw.width,
				targets.ssrRaw.height,
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
				{ binding: 3, resource: targets.hiZ },
				{ binding: 4, resource: targets.ssrHistoryRead },
				{ binding: 5, resource: targets.motionHistoryRead },
				{ binding: 6, resource: context.shared.sampler },
				{ binding: 7, resource: resources.traceParams },
				{ binding: 8, resource: targets.ssrRaw },
			],
			"WebGPUSSR_TraceBinding"
		);
		context.encoder.beginComputePass({ label: "WebGPUSSR_TraceTemporal" });
		context.encoder.setComputePipeline(resources.tracePipeline);
		context.encoder.setBindingGroup(0, binding);
		context.encoder.setBindingGroup(1, context.frameBinding);
		context.encoder.dispatchWorkgroups(
			ceilDiv(targets.ssrRaw.width, WORKGROUP_SIZE),
			ceilDiv(targets.ssrRaw.height, WORKGROUP_SIZE),
			1
		);
		context.encoder.endComputePass();

		await this._copyTexture(
			context.shared,
			resources,
			context.encoder,
			targets.ssrRaw,
			targets.ssrHistoryWrite
		);

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
				{ binding: 1, resource: targets.ssrRaw },
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
				hiz: new WebGPUHiZPostProcessHelper(shared),
				module: null,
				tracePipeline: null,
				composePipeline: null,
				traceParams: null,
				composeParams: null,
				copyModule: null,
				copyPipeline: null,
				traceGroupLayout0: null,
				tracePipelineLayout: null,
				frameIndex: 0,
			};
			this._resources.set(shared, resources);
		}
		await resources.hiz.ensureResources();
		if (!resources.module) {
			const shader = await loadPostProcessShaderPartComposite("ssr");
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
				resources.tracePipeline = shared.compute.createComputePipeline({
					label: "WebGPUSSRTracePipeline",
					layout: resources.tracePipelineLayout,
					compute: { module: resources.module, entryPoint: "csTrace" },
				});
			} else {
				resources.tracePipeline = shared.compute.createComputePipeline({
					label: "WebGPUSSRTracePipeline",
					compute: { module: resources.module, entryPoint: "csTrace" },
				});
			}
		}
		if (!resources.composePipeline) {
			resources.composePipeline = shared.compute.createComputePipeline({
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

	private async _copyTexture(
		shared: PostProcessSharedContext,
		resources: WebGPUSSRResources,
		encoder: ICommandEncoder,
		src: IRenderTexture,
		dst: IRenderTexture
	): Promise<void> {
		if (src === dst) {
			return;
		}
		await this._ensureCopyResources(shared, resources);
		if (!resources.copyPipeline) {
			return;
		}
		const binding = shared.getCachedBindGroup(
			`copy-${src === dst ? "same" : "diff"}`,
			resources.copyPipeline,
			[
				{ binding: 0, resource: src },
				{ binding: 1, resource: dst },
			],
			"WebGPUPost_CopyBinding"
		);
		encoder.beginComputePass({ label: "WebGPUPost_Copy" });
		encoder.setComputePipeline(resources.copyPipeline);
		encoder.setBindingGroup(0, binding);
		encoder.dispatchWorkgroups(
			ceilDiv(dst.width, WORKGROUP_SIZE),
			ceilDiv(dst.height, WORKGROUP_SIZE),
			1
		);
		encoder.endComputePass();
	}

	private async _ensureCopyResources(
		shared: PostProcessSharedContext,
		resources: WebGPUSSRResources
	): Promise<void> {
		if (!resources.copyModule) {
			const shader = await loadPostProcessShaderPartComposite("copy");
			resources.copyModule = await shared.compute.createShaderModule({
				label: "WebGPUCopyShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!resources.copyPipeline) {
			resources.copyPipeline = shared.compute.createComputePipeline({
				label: "WebGPUCopyPipeline",
				compute: { module: resources.copyModule, entryPoint: "csMain" },
			});
		}
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
			id: SCREEN_SPACE_REFLECTIONS_PASS_ID,
			builtIn: true,
			warningLabel: "SSR",
			placement: "temporal",
			order: 210,
			implementations: {
				webgpu: new WebGPUScreenSpaceReflectionsImplementation(),
			},
		});
	}

	public override normalizeOptions(): ResolvedSSROptions {
		return resolveSSROptions(this.getRawOptions());
	}

	public override getRequirements(): PostProcessPassRequirements {
		return { gBuffer: ["depth", "normal", "motion"] };
	}

	public override getHistoryDescriptors(
		request: PostProcessPassResolveRequest<ResolvedSSROptions>
	): readonly PostProcessHistoryDescriptor[] {
		return resolveSSRHistoryDescriptors(request);
	}
}

function clampDownsample(value: unknown, fallback: number): number {
	if (typeof value !== "number" || !Number.isFinite(value)) {
		return fallback;
	}
	return Math.min(8, Math.max(1, Math.floor(value)));
}
