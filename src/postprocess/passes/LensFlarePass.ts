import {
	BufferUsage,
	type IComputePipeline,
	type IRenderBuffer,
	type IRenderTexture,
	type IShaderModule,
} from "../../backends/types";
import type { WebGPUPostProcessServices } from "../../backends/webgpu/WebGPUPostProcessContracts";
import { WEBGPU_2D_COMPUTE_WORKGROUP_SIZE } from "../../backends/webgpu/constants";
import { ceilDiv, sanitizeFiniteClamped } from "../../maths/Misc";
import { ShaderSource } from "../../shaders/ShaderSource";
import { PostProcessPass, type PostProcessPassConfig } from "../PostProcessPass";
import {
	POST_PROCESS_SAMPLED_READ,
	POST_PROCESS_STORAGE_WRITE,
	WEBGPU_VERSIONED_EXECUTION,
} from "../executionDeclarations";
import type { PostProcessScheduleEntry } from "../ordering";
import type {
	PostProcessExecutionDeclaration,
	PostProcessPassImplementation,
	PostProcessPassRequest,
} from "../types";
import type { WebGPUScreenPostProcessContext } from "./ScreenPassShared";

export const LENS_FLARE_PASS_ID = "lens-flare";
export const LENS_FLARE_PASS_ORDER = {
	id: LENS_FLARE_PASS_ID,
	placement: "hdr",
	order: 490,
} as const satisfies PostProcessScheduleEntry;

export interface LensFlareOptions {
	/** Scene-linear luminance threshold, clamped to [0, 65504]. */
	threshold?: number;
	/** Soft threshold width in luminance units, clamped to [0, 65504]. */
	softKnee?: number;
	/** Overall HDR contribution in [0, 10]. Zero skips the effect. */
	intensity?: number;
	/** Total ghost contribution in [0, 4], independent of ghost count. */
	ghostIntensity?: number;
	/** Number of ghost samples, rounded and clamped to [1, 8]. */
	ghostCount?: number;
	/** Ghost spacing along the image-center vector, clamped to [0, 1]. */
	ghostSpacing?: number;
	/** Soft hexagonal aperture circumradius in image-height units, in [0.005, 0.08]. */
	ghostRadius?: number;
	/** Upper bound on ghost magnification, clamped to [1, 4]. */
	maxGhostScale?: number;
	/** Halo contribution, clamped to [0, 4]. */
	haloIntensity?: number;
	/** Halo radius as a fraction of image height, clamped to [0, 1]. */
	haloRadius?: number;
	/** Halo radial spread in image-height units, clamped to [0, 0.25]. */
	haloWidth?: number;
	/** RGB separation in image-height units, clamped to [0, 0.05]. */
	chromaticAberration?: number;
	/** Blur tap spacing in low-resolution pixels, clamped to [0, 8]. */
	blurRadius?: number;
	/** Source fade distance from image edges in UV units, clamped to [0, 0.5]. */
	edgeFade?: number;
	/** Width and height divisor, rounded and clamped to [1, 4]. */
	downsample?: number;
}

export type ResolvedLensFlareOptions = Required<LensFlareOptions>;

export const DEFAULT_LENS_FLARE_OPTIONS: Readonly<ResolvedLensFlareOptions> = Object.freeze({
	threshold: 1,
	softKnee: 0.5,
	intensity: 0.22,
	ghostIntensity: 0.75,
	ghostCount: 4,
	ghostSpacing: 0.3,
	ghostRadius: 0.035,
	maxGhostScale: 2,
	haloIntensity: 0.08,
	haloRadius: 0.32,
	haloWidth: 0.04,
	chromaticAberration: 0.0007,
	blurRadius: 0.75,
	edgeFade: 0.1,
	downsample: 4,
});

/**
 * Resolves lens flare settings without mutating the input.
 * @param options Partial settings; non-finite values use their defaults.
 * @returns Complete options clamped to the ranges documented by `LensFlareOptions`.
 */
export function resolveLensFlareOptions(
	options: LensFlareOptions = {}
): ResolvedLensFlareOptions {
	const defaults = DEFAULT_LENS_FLARE_OPTIONS;
	return {
		threshold: sanitizeFiniteClamped(options.threshold, defaults.threshold, 0, 65504),
		softKnee: sanitizeFiniteClamped(options.softKnee, defaults.softKnee, 0, 65504),
		intensity: sanitizeFiniteClamped(options.intensity, defaults.intensity, 0, 10),
		ghostIntensity: sanitizeFiniteClamped(options.ghostIntensity, defaults.ghostIntensity, 0, 4),
		ghostCount: Math.round(sanitizeFiniteClamped(options.ghostCount, defaults.ghostCount, 1, 8)),
		ghostSpacing: sanitizeFiniteClamped(options.ghostSpacing, defaults.ghostSpacing, 0, 1),
		ghostRadius: sanitizeFiniteClamped(options.ghostRadius, defaults.ghostRadius, 0.005, 0.08),
		maxGhostScale: sanitizeFiniteClamped(options.maxGhostScale, defaults.maxGhostScale, 1, 4),
		haloIntensity: sanitizeFiniteClamped(options.haloIntensity, defaults.haloIntensity, 0, 4),
		haloRadius: sanitizeFiniteClamped(options.haloRadius, defaults.haloRadius, 0, 1),
		haloWidth: sanitizeFiniteClamped(options.haloWidth, defaults.haloWidth, 0, 0.25),
		chromaticAberration: sanitizeFiniteClamped(
			options.chromaticAberration, defaults.chromaticAberration, 0, 0.05
		),
		blurRadius: sanitizeFiniteClamped(options.blurRadius, defaults.blurRadius, 0, 8),
		edgeFade: sanitizeFiniteClamped(options.edgeFade, defaults.edgeFade, 0, 0.5),
		downsample: Math.round(sanitizeFiniteClamped(options.downsample, defaults.downsample, 1, 4)),
	};
}

const STAGES = [
	"Extract", "AperturePrefilterH", "AperturePrefilterV", "Aperture",
	"Features", "BlurH", "BlurV", "Composite",
] as const;
type LensFlareStage = (typeof STAGES)[number];
const TRANSIENT_IDS = ["lens-flare:bright", "lens-flare:features", "lens-flare:scratch"];

interface LensFlareResources {
	readonly shared: WebGPUPostProcessServices;
	readonly pipelines: Partial<Record<LensFlareStage, IComputePipeline>>;
	readonly paramData: Float32Array<ArrayBuffer>;
	module: IShaderModule | null;
	params: IRenderBuffer | null;
}

/** @internal Post-process-owned WebGPU implementation; register `LensFlarePass` instead. */
export class WebGPULensFlareImplementation implements
	PostProcessPassImplementation<WebGPUScreenPostProcessContext, ResolvedLensFlareOptions>
{
	public readonly id = "lens-flare:webgpu";
	private readonly _resources = new Map<WebGPUPostProcessServices, LensFlareResources>();

	/** @internal Post-process planner declaration; configure `LensFlarePass` instead. */
	public describeExecution(
		request: { readonly options?: LensFlareOptions }
	): PostProcessExecutionDeclaration {
		const scale = 1 / resolveLensFlareOptions(request.options).downsample;
		return {
			...WEBGPU_VERSIONED_EXECUTION,
			transients: TRANSIENT_IDS.map((id) => ({
				descriptor: {
					id, widthScale: scale, heightScale: scale, format: "rgba16float",
				},
				uses: [POST_PROCESS_STORAGE_WRITE, POST_PROCESS_SAMPLED_READ],
			})),
		};
	}

	/** @internal Backend warmup hook; use `Renderer.warmup()` instead. */
	public async warmup(context: WebGPUScreenPostProcessContext | undefined): Promise<void> {
		if (context) await this._ensureResources(context.shared);
	}

	/** @internal Backend recording hook; render through `Renderer` instead. */
	public async execute(
		request: PostProcessPassRequest<ResolvedLensFlareOptions>,
		context: WebGPUScreenPostProcessContext | undefined
	) {
		const options = request.options;
		if (!context?.encoder || options.intensity === 0 ||
			(options.ghostIntensity === 0 && options.haloIntensity === 0)) {
			return { ran: false };
		}
		const input = context.resources.color.input;
		const output = context.resources.color.output;
		if (!input || !output) return { ran: false };
		const bright = context.resources.getTransient(TRANSIENT_IDS[0]);
		const features = context.resources.getTransient(TRANSIENT_IDS[1]);
		const scratch = context.resources.getTransient(TRANSIENT_IDS[2]);
		if (!bright || !features || !scratch) return { ran: false };
		const resources = await this._ensureResources(context.shared);
		if (!context.shared.sampler) return { ran: false };
		const data = resources.paramData;
		data[0] = options.threshold;
		data[1] = options.softKnee;
		data[2] = options.intensity;
		data[3] = options.ghostIntensity;
		data[4] = options.ghostCount;
		data[5] = options.ghostSpacing;
		data[6] = options.haloIntensity;
		data[7] = options.haloRadius;
		data[8] = options.haloWidth;
		data[9] = options.chromaticAberration;
		data[10] = options.blurRadius;
		data[11] = options.edgeFade;
		data[12] = input.width / Math.max(input.height, 1);
		data[13] = options.ghostRadius;
		data[14] = options.maxGhostScale;
		context.shared.compute.writeBuffer(resources.params!, data);
		recordDispatch(context, resources, "Extract", input, bright);
		recordDispatch(context, resources, "AperturePrefilterH", bright, scratch);
		recordDispatch(context, resources, "AperturePrefilterV", scratch, features);
		recordDispatch(context, resources, "Aperture", features, scratch);
		recordDispatch(context, resources, "Features", scratch, features, bright);
		recordDispatch(context, resources, "BlurH", features, scratch);
		recordDispatch(context, resources, "BlurV", scratch, features);
		recordDispatch(context, resources, "Composite", input, output, features);
		return { ran: true };
	}

	/** @internal Backend cache lifecycle; invalidate the logical `LensFlarePass` instead. */
	public invalidate(): void {
		this.destroy();
	}

	/** @internal Releases backend-owned state; remove the logical pass through its registry. */
	public destroy(): void {
		for (const resources of this._resources.values()) {
			resources.shared.invalidateBindingsByPrefix("lens-flare-");
			for (const pipeline of Object.values(resources.pipelines)) {
				resources.shared.destroyManagedResource(pipeline, "lens flare pipeline");
			}
			resources.shared.destroyManagedResource(resources.params, "lens flare parameters");
			resources.shared.destroyManagedResource(resources.module, "lens flare shader");
		}
		this._resources.clear();
	}

	private async _ensureResources(shared: WebGPUPostProcessServices): Promise<LensFlareResources> {
		let resources = this._resources.get(shared);
		if (!resources) {
			resources = { shared, pipelines: {}, paramData: new Float32Array(15),
				module: null, params: null };
			this._resources.set(shared, resources);
		}
		await shared.ensureCommonResources();
		if (!resources.module) {
			const shader = await ShaderSource.load("webgpu.postprocess.lensFlare");
			resources.module = await shared.compute.createShaderModule({
				label: "WebGPULensFlareShader", code: shader.source.code,
				sourceMap: shader.source.sourceMap, language: "wgsl",
				stage: "compute", sourceKind: "postprocess",
			});
		}
		for (const stage of STAGES) {
			resources.pipelines[stage] ??= await shared.compute.createComputePipeline({
				label: `WebGPULensFlare${stage}Pipeline`,
				compute: { module: resources.module, entryPoint: `cs${stage}` },
			});
		}
		resources.params ??= shared.compute.createBuffer({
			label: "WebGPULensFlareParams", size: resources.paramData.byteLength,
			usage: BufferUsage.Uniform | BufferUsage.CopyDst,
		});
		return resources;
	}
}

function recordDispatch(
	context: WebGPUScreenPostProcessContext,
	resources: LensFlareResources,
	stage: LensFlareStage,
	input: IRenderTexture,
	output: IRenderTexture,
	secondary?: IRenderTexture
): void {
	const pipeline = resources.pipelines[stage]!;
	const entries = [
		{ binding: 0, resource: input },
		{ binding: 1, resource: context.shared.sampler },
		{ binding: 2, resource: resources.params },
		{ binding: 3, resource: output },
	];
	if (secondary) entries.push({ binding: 4, resource: secondary });
	const label = `WebGPULensFlare${stage}`;
	const group = context.shared.getCachedBindGroup(
		`lens-flare-${stage}`, pipeline, entries, label
	);
	const encoder = context.encoder!;
	encoder.beginComputePass({ label });
	encoder.setComputePipeline(pipeline);
	encoder.setBindingGroup(0, group);
	encoder.dispatchWorkgroups(
		ceilDiv(output.width, WEBGPU_2D_COMPUTE_WORKGROUP_SIZE),
		ceilDiv(output.height, WEBGPU_2D_COMPUTE_WORKGROUP_SIZE),
		1
	);
	encoder.endComputePass();
}

export interface LensFlarePassConfig extends Omit<
	PostProcessPassConfig<LensFlareOptions>,
	"id" | "builtIn" | "label" | "implementations" | "alphaContract" | "colorContract"
> {}

/** Screen-space HDR ghosts and halos. Only WebGPU implements this effect. */
export class LensFlarePass extends PostProcessPass<LensFlareOptions, ResolvedLensFlareOptions> {
	public constructor(config: LensFlarePassConfig = {}) {
		super({
			...config,
			id: LENS_FLARE_PASS_ID,
			label: "Lens Flare",
			alphaContract: "premultiplied",
			colorContract: { input: "scene-linear-hdr", output: "scene-linear-hdr" },
			schedule: {
				placement: config.schedule?.placement ?? LENS_FLARE_PASS_ORDER.placement,
				order: config.schedule?.order ?? LENS_FLARE_PASS_ORDER.order,
			},
			implementations: { webgpu: () => new WebGPULensFlareImplementation() },
		});
	}

	public override normalizeOptions(): ResolvedLensFlareOptions {
		return resolveLensFlareOptions(this.getRawOptions());
	}
}
