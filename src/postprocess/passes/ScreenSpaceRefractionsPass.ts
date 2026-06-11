import { materialUsesTransmission } from "../../materials/transparency";
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
import type { ICommandEncoder } from "../../renderers/ICommandEncoder";
import { ceilDiv, finiteOr } from "../../maths/Misc";
import { ShaderSource } from "../../shaders/ShaderSource";
import {
	PostProcessPass,
	type PostProcessPassConfig,
	type PostProcessPassResolveRequest,
} from "../PostProcessPass";
import type { PostProcessPassMetadata } from "../ordering";
import type {
	PostProcessPassImplementation,
	PostProcessPassRequest,
	PostProcessPassRequirements,
	PostProcessPassResult,
	PostProcessTransientDescriptor,
} from "../types";

export const SCREEN_SPACE_REFRACTIONS_PASS_ID = "ssrefraction";
export const SCREEN_SPACE_REFRACTIONS_PASS_ORDER = {
	id: SCREEN_SPACE_REFRACTIONS_PASS_ID,
	placement: "temporal",
	order: 215,
	incremental: {
		firstPass: "ssrefraction",
		grade: "cinematic",
		inflationRadius: 16,
	},
} as const satisfies PostProcessPassMetadata;
const WEBGPU_SSREFRACTION_RAW_TRANSIENT_ID = "ssrefraction:raw";
const WEBGPU_HIZ_TRANSIENT_ID = "hiz";
const WEBGPU_HIZ_TRANSIENT_USAGE = ["sampled", "storage"] as const;

export interface SSRefractionOptions {
	/** Maximum ray-march iterations per refraction ray. */
	maxSteps?: number;
	/** Maximum world/view-space ray distance for screen-space refractions. */
	maxDistance?: number;
	/** Depth tolerance used when matching refracted rays to opaque surfaces. */
	thickness?: number;
	/** Ray step stride. Higher values improve speed but can skip thin details. */
	stride?: number;
	/** Refraction contribution multiplier mixed into the scene color. */
	intensity?: number;
	/** Screen-edge fade distance that hides unreliable offscreen hits. */
	edgeFade?: number;
	/** Mip bias multiplier used for rough transmitted background sampling. */
	roughnessMipScale?: number;
	/** Internal trace buffer scale divisor. Higher values improve speed. */
	downsample?: number;
	/** Refinement iterations after a ray hit is found. */
	binarySearchSteps?: number;
	/** Allows backend-specific experimental refraction options. */
	[key: string]: unknown;
}

export const DEFAULT_SSREFRACTION_OPTIONS: Required<
	Pick<
		SSRefractionOptions,
		| "downsample"
		| "maxSteps"
		| "binarySearchSteps"
		| "maxDistance"
		| "thickness"
		| "stride"
		| "intensity"
		| "edgeFade"
		| "roughnessMipScale"
	>
> = {
	downsample: 1,
	maxSteps: 64,
	binarySearchSteps: 5,
	maxDistance: 50,
	thickness: 0.2,
	stride: 1,
	intensity: 1,
	edgeFade: 0.1,
	roughnessMipScale: 4,
};

export type ResolvedSSRefractionOptions = Required<
	Pick<
		SSRefractionOptions,
		| "downsample"
		| "maxSteps"
		| "binarySearchSteps"
		| "maxDistance"
		| "thickness"
		| "stride"
		| "intensity"
		| "edgeFade"
		| "roughnessMipScale"
	>
>;

/** @internal WebGPU context supplied to the built-in screen-space refraction implementation. */
export interface WebGPUSSRefractionContext {
	readonly encoder?: ICommandEncoder;
	readonly targets?: WebGPUPostProcessFrameTargets;
	readonly shared: PostProcessSharedContext;
	readonly frameBinding?: IBindingGroup;
	readonly refractionRaw?: IRenderTexture | null;
	readonly hiZ?: IRenderTexture | null;
	publishColorTarget?(texture: IRenderTexture): void;
}

interface WebGPUSSRefractionResources {
	shared: PostProcessSharedContext;
	module: IShaderModule | null;
	tracePipeline: IComputePipeline | null;
	composePipeline: IComputePipeline | null;
	traceParams: IRenderBuffer | null;
	composeParams: IRenderBuffer | null;
	traceGroupLayout0: GPUBindGroupLayout | null;
	tracePipelineLayout: GPUPipelineLayout | null;
}

/**
 * Resolves SSRf options with backend-independent clamping.
 *
 * @param options User-provided screen-space refraction options.
 * @returns Fully resolved screen-space refraction options.
 * @sideEffects None.
 */
export function resolveSSRefractionOptions(
	options?: SSRefractionOptions | null
): ResolvedSSRefractionOptions {
	return {
		downsample: clampDownsample(
			options?.downsample,
			DEFAULT_SSREFRACTION_OPTIONS.downsample
		),
		maxSteps: finiteOr(
			options?.maxSteps,
			DEFAULT_SSREFRACTION_OPTIONS.maxSteps
		),
		binarySearchSteps: finiteOr(
			options?.binarySearchSteps,
			DEFAULT_SSREFRACTION_OPTIONS.binarySearchSteps
		),
		maxDistance: finiteOr(
			options?.maxDistance,
			DEFAULT_SSREFRACTION_OPTIONS.maxDistance
		),
		thickness: finiteOr(
			options?.thickness,
			DEFAULT_SSREFRACTION_OPTIONS.thickness
		),
		stride: finiteOr(options?.stride, DEFAULT_SSREFRACTION_OPTIONS.stride),
		intensity: finiteOr(
			options?.intensity,
			DEFAULT_SSREFRACTION_OPTIONS.intensity
		),
		edgeFade: finiteOr(
			options?.edgeFade,
			DEFAULT_SSREFRACTION_OPTIONS.edgeFade
		),
		roughnessMipScale: finiteOr(
			options?.roughnessMipScale,
			DEFAULT_SSREFRACTION_OPTIONS.roughnessMipScale
		),
	};
}

/**
 * Creates packed SSRf trace shader parameters.
 *
 * @param width Trace target width.
 * @param height Trace target height.
 * @param options Resolved options.
 * @param maxHiZMip Maximum available Hi-Z mip index.
 * @returns Sixteen float parameters expected by the SSRf trace shader.
 * @sideEffects None.
 */
export function createSSRefractionTraceParams(
	width: number,
	height: number,
	options: ResolvedSSRefractionOptions,
	maxHiZMip: number
): Float32Array {
	return new Float32Array([
		1 / Math.max(width, 1),
		1 / Math.max(height, 1),
		options.maxDistance,
		options.thickness,
		options.stride,
		options.intensity,
		options.edgeFade,
		options.maxSteps,
		options.binarySearchSteps,
		maxHiZMip,
		options.roughnessMipScale,
		0,
		0,
		0,
		0,
		0,
	]);
}

/**
 * Resolves dynamic transient resources required by SSRf.
 *
 * @param request Transient resolution request for the current frame.
 * @returns Downsampled raw refraction target and shared full-chain Hi-Z target.
 * @sideEffects None.
 */
export function resolveSSRefractionTransientDescriptors(
	request: PostProcessPassResolveRequest<ResolvedSSRefractionOptions>
): readonly PostProcessTransientDescriptor[] {
	if (request.backend !== "webgpu") {
		return [];
	}
	const options = resolveSSRefractionOptions(request.options);
	const scale = 1 / options.downsample;
	return [
		{
			id: WEBGPU_SSREFRACTION_RAW_TRANSIENT_ID,
			widthScale: scale,
			heightScale: scale,
		},
		{
			id: WEBGPU_HIZ_TRANSIENT_ID,
			usage: WEBGPU_HIZ_TRANSIENT_USAGE,
			mipMode: "full-chain",
		},
	];
}

/**
 * WebGPU implementation of the cross-backend screen-space refractions pass.
 */
/** @internal WebGPU implementation for the built-in screen-space refraction pass. */
export class WebGPUScreenSpaceRefractionsImplementation
	implements PostProcessPassImplementation<WebGPUSSRefractionContext>
{
	public readonly id = "ssrefraction:webgpu";
	public readonly metadata = {
		context: {
			backend: "webgpu",
			kind: "screen",
			publishColorTarget: true,
			frameBinding: true,
			transients: [
				{
					property: "refractionRaw",
					transientId: WEBGPU_SSREFRACTION_RAW_TRANSIENT_ID,
				},
				{
					property: "hiZ",
					transientId: WEBGPU_HIZ_TRANSIENT_ID,
				},
			],
		},
	} as const;
	private _resources = new WeakMap<
		PostProcessSharedContext,
		WebGPUSSRefractionResources
	>();
	private _resourceSet = new Set<WebGPUSSRefractionResources>();

	public async warmup(
		context: WebGPUSSRefractionContext | undefined
	): Promise<void> {
		if (context) {
			await this._ensureResources(context.shared);
		}
	}

	public async execute(
		request: PostProcessPassRequest,
		context: WebGPUSSRefractionContext | undefined
	): Promise<PostProcessPassResult> {
		if (!context?.encoder || !context.targets || !context.frameBinding) {
			return { ran: false };
		}
		const ran = await this._runKernel(request, context);
		return ran ? { ran: true } : { ran: false };
	}

	public invalidate(): void {
		for (const resources of this._resourceSet) {
			resources.shared.invalidateBindingsByPrefix("ssrefraction-");
		}
	}

	public destroy(): void {
		for (const resources of this._resourceSet) {
			resources.shared.destroyManagedResource(
				resources.tracePipeline,
				"SSRf trace pipeline"
			);
			resources.shared.destroyManagedResource(
				resources.composePipeline,
				"SSRf compose pipeline"
			);
			resources.shared.destroyManagedResource(
				resources.module,
				"SSRf shader module"
			);
			resources.shared.destroyManagedResource(
				resources.traceParams,
				"SSRf trace params buffer"
			);
			resources.shared.destroyManagedResource(
				resources.composeParams,
				"SSRf compose params buffer"
			);
			resources.shared.invalidateBindingsByPrefix("ssrefraction-");
			resources.module = null;
			resources.tracePipeline = null;
			resources.composePipeline = null;
			resources.traceParams = null;
			resources.composeParams = null;
			resources.traceGroupLayout0 = null;
			resources.tracePipelineLayout = null;
		}
		this._resourceSet.clear();
		this._resources = new WeakMap<
			PostProcessSharedContext,
			WebGPUSSRefractionResources
		>();
	}

	private async _runKernel(
		request: PostProcessPassRequest,
		context: WebGPUSSRefractionContext
	): Promise<boolean> {
		const resources = await this._ensureResources(context.shared);
		const targets = context.targets;
		if (
			!context.encoder ||
			!context.frameBinding ||
			!context.refractionRaw ||
			!context.hiZ ||
			!context.shared.sampler ||
			!resources.tracePipeline ||
			!resources.composePipeline ||
			!resources.traceParams ||
			!resources.composeParams ||
			!targets ||
			!targets.postPing ||
			!targets.postPong ||
			!targets.gMotionDepth ||
			!targets.transmissionSceneColorCopy ||
			!targets.transmissionLighting ||
			!targets.gTransmissionSurface0 ||
			!targets.gTransmissionSurface1 ||
			!targets.gTransmissionSurface2
		) {
			return false;
		}

		const hiZMips = await context.shared.getHiZHelper().build({
			encoder: context.encoder,
			depth: targets.gMotionDepth,
			hiZ: context.hiZ,
		});
		if (hiZMips.length === 0) {
			return false;
		}

		const raw = context.refractionRaw;
		const options = resolveSSRefractionOptions(
			request.options as SSRefractionOptions
		);
		context.shared.compute.writeBuffer(
			resources.traceParams,
			createSSRefractionTraceParams(
				raw.width,
				raw.height,
				options,
				hiZMips.length - 1
			) as unknown as BufferSource
		);

		let binding = context.shared.getCachedBindGroup(
			"ssrefraction-trace",
			resources.tracePipeline,
			[
				{ binding: 0, resource: targets.transmissionSceneColorCopy },
				{ binding: 1, resource: targets.gTransmissionSurface0 },
				{ binding: 2, resource: targets.gTransmissionSurface1 },
				{ binding: 3, resource: targets.gTransmissionSurface2 },
				{ binding: 4, resource: targets.gMotionDepth },
				{ binding: 5, resource: context.hiZ },
				{ binding: 6, resource: context.shared.sampler },
				{ binding: 7, resource: resources.traceParams },
				{ binding: 8, resource: raw },
			],
			"WebGPUSSRefraction_TraceBinding"
		);
		context.encoder.beginComputePass({ label: "WebGPUSSRefraction_Trace" });
		context.encoder.setComputePipeline(resources.tracePipeline);
		context.encoder.setBindingGroup(0, binding);
		context.encoder.setBindingGroup(1, context.frameBinding);
		context.encoder.dispatchWorkgroups(
			ceilDiv(raw.width, WORKGROUP_SIZE),
			ceilDiv(raw.height, WORKGROUP_SIZE),
			1
		);
		context.encoder.endComputePass();

		const composeTarget =
			targets.sceneColor === targets.postPing ? targets.postPong : targets.postPing;
		context.shared.compute.writeBuffer(
			resources.composeParams,
			new Float32Array([
				1 / Math.max(composeTarget.width, 1),
				1 / Math.max(composeTarget.height, 1),
				options.intensity,
				0,
			]) as unknown as BufferSource
		);
		binding = context.shared.getCachedBindGroup(
			`ssrefraction-compose-${composeTarget === targets.postPing ? "ping" : "pong"}`,
			resources.composePipeline,
			[
				{ binding: 0, resource: targets.sceneColor },
				{ binding: 1, resource: raw },
				{ binding: 2, resource: targets.transmissionLighting },
				{ binding: 3, resource: context.shared.sampler },
				{ binding: 4, resource: resources.composeParams },
				{ binding: 5, resource: composeTarget },
			],
			"WebGPUSSRefraction_ComposeBinding"
		);
		context.encoder.beginComputePass({ label: "WebGPUSSRefraction_Compose" });
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
	): Promise<WebGPUSSRefractionResources> {
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
			};
			this._resources.set(shared, resources);
			this._resourceSet.add(resources);
		}
		await shared.getHiZHelper().ensureResources();
		if (!resources.module) {
			const shader =
				await ShaderSource.load(
					"webgpu.postprocess.screenSpaceRefractions.composite"
				);
			resources.module = await shared.compute.createShaderModule({
				label: "WebGPUSSRefractionShader",
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
					label: "WebGPUSSRefractionTrace_GroupLayout0",
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
					label: "WebGPUSSRefractionTrace_PipelineLayout",
					bindGroupLayouts: [
						resources.traceGroupLayout0,
						shared.frameBindGroupLayout,
					],
				});
				resources.tracePipeline = await shared.compute.createComputePipeline({
					label: "WebGPUSSRefractionTracePipeline",
					layout: resources.tracePipelineLayout,
					compute: { module: resources.module, entryPoint: "csTrace" },
				});
			} else {
				resources.tracePipeline = await shared.compute.createComputePipeline({
					label: "WebGPUSSRefractionTracePipeline",
					compute: { module: resources.module, entryPoint: "csTrace" },
				});
			}
		}
		if (!resources.composePipeline) {
			resources.composePipeline = await shared.compute.createComputePipeline({
				label: "WebGPUSSRefractionComposePipeline",
				compute: { module: resources.module, entryPoint: "csCompose" },
			});
		}
		if (!resources.traceParams) {
			resources.traceParams = shared.compute.createBuffer({
				label: "WebGPUSSRefractionTraceParams",
				size: 16 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
		}
		if (!resources.composeParams) {
			resources.composeParams = shared.compute.createBuffer({
				label: "WebGPUSSRefractionComposeParams",
				size: 4 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
		}
		return resources;
	}
}

export interface ScreenSpaceRefractionsPassConfig
	extends Omit<
		PostProcessPassConfig<SSRefractionOptions>,
		| "id"
		| "builtIn"
		| "warningLabel"
		| "placement"
		| "order"
		| "implementations"
	> {}

/**
 * Stateful logical screen-space refractions pass.
 */
export class ScreenSpaceRefractionsPass extends PostProcessPass<
	SSRefractionOptions,
	ResolvedSSRefractionOptions
> {
	public constructor(config: ScreenSpaceRefractionsPassConfig = {}) {
		super({
			...config,
			...SCREEN_SPACE_REFRACTIONS_PASS_ORDER,
			incremental:
				config.incremental ?? SCREEN_SPACE_REFRACTIONS_PASS_ORDER.incremental,
			warningLabel: "screen-space refractions",
			implementations: {
				webgpu: new WebGPUScreenSpaceRefractionsImplementation(),
			},
		});
	}

	public override normalizeOptions(): ResolvedSSRefractionOptions {
		return resolveSSRefractionOptions(this.getRawOptions());
	}

	public override getRequirements(): PostProcessPassRequirements {
		return { gBuffer: ["depth", "motion", "transmission"] };
	}

	public override getTransientResourceDescriptors(
		request: PostProcessPassResolveRequest<ResolvedSSRefractionOptions>
	): readonly PostProcessTransientDescriptor[] {
		return resolveSSRefractionTransientDescriptors(request);
	}

	public override shouldExecute(
		request: PostProcessPassResolveRequest<ResolvedSSRefractionOptions>
	): boolean {
		return (
			request.frameContext?.scene.transparentPackets.some((packet) =>
				materialUsesTransmission(packet.material)
			) ?? false
		);
	}
}

function clampDownsample(value: unknown, fallback: number): number {
	if (typeof value !== "number" || !Number.isFinite(value)) {
		return fallback;
	}
	return Math.min(8, Math.max(1, Math.floor(value)));
}
