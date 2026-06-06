import { clamp } from "../../maths/Common";
import { ceilDiv, finiteOr } from "../../maths/Misc";
import type { ICommandEncoder } from "../../renderers/ICommandEncoder";
import {
	BufferUsage,
	type IComputePipeline,
	type IRenderBuffer,
	type IRenderTexture,
	type IShaderModule,
} from "../../renderers/types";
import {
	WEBGPU_2D_COMPUTE_WORKGROUP_SIZE as WORKGROUP_SIZE,
} from "../../renderers/webgpu/constants";
import {
	WEBGPU_SCREEN_POST_PROCESS_CONTEXT_METADATA,
	type WebGPUPostProcessFrameTargets,
} from "../../renderers/webgpu/WebGPUPostProcessContracts";
import type { PostProcessSharedContext } from "../../renderers/webgpu/postprocess/PostProcessSharedContext";
import { loadPostProcessShaderPartComposite } from "../../shaders/webgpu/shaderSource";
import { PostProcessPass, type PostProcessPassConfig } from "../PostProcessPass";
import { defineBuiltinPostProcessOrder } from "../ordering";
import type {
	PostProcessPassImplementation,
	PostProcessPassRequest,
	PostProcessPassResult,
	PostProcessPassRequirements,
} from "../types";

const SSGI_MAX_SAMPLES = 16;
export const SCREEN_SPACE_GLOBAL_ILLUMINATION_PASS_ID = "ssgi";
export const SCREEN_SPACE_GLOBAL_ILLUMINATION_PASS_ORDER =
	defineBuiltinPostProcessOrder({
		id: SCREEN_SPACE_GLOBAL_ILLUMINATION_PASS_ID,
		placement: "spatial",
		order: 110,
	});

export interface SSGIOptions {
	/** Indirect-light sample count, clamped to backend limits. */
	samples?: number;
	/** Screen-space sampling radius for bounced light. */
	radius?: number;
	/** Indirect diffuse lighting multiplier. */
	intensity?: number;
	/** Distance falloff exponent for indirect samples. */
	falloff?: number;
	/** Depth sensitivity for rejecting samples across geometry breaks. */
	depthPhi?: number;
	/** Normal sensitivity for rejecting samples from unrelated surfaces. */
	normalPhi?: number;
	/** Albedo multiplier used to brighten diffuse bounce color. */
	albedoBoost?: number;
	/** Allows backend-specific experimental SSGI options. */
	[key: string]: unknown;
}

export const DEFAULT_SSGI_OPTIONS: Required<
	Pick<
		SSGIOptions,
		| "samples"
		| "radius"
		| "intensity"
		| "falloff"
		| "depthPhi"
		| "normalPhi"
		| "albedoBoost"
	>
> = {
	samples: 8,
	radius: 3,
	intensity: 0.35,
	falloff: 1.5,
	depthPhi: 1.25,
	normalPhi: 2,
	albedoBoost: 1,
};

export type ResolvedSSGIOptions = Required<
	Pick<
		SSGIOptions,
		| "samples"
		| "radius"
		| "intensity"
		| "falloff"
		| "depthPhi"
		| "normalPhi"
		| "albedoBoost"
	>
>;

export interface WebGPUSSGIContext {
	readonly encoder?: ICommandEncoder;
	readonly targets?: WebGPUPostProcessFrameTargets;
	readonly shared: PostProcessSharedContext;
	publishColorTarget?(texture: IRenderTexture): void;
}

interface WebGPUSSGIResources {
	shared: PostProcessSharedContext;
	module: IShaderModule | null;
	pipeline: IComputePipeline | null;
	params: IRenderBuffer | null;
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
	return {
		radius: clamp(finiteOr(options?.radius, DEFAULT_SSGI_OPTIONS.radius), 1, 6),
		samples: clamp(
			Math.floor(finiteOr(options?.samples, DEFAULT_SSGI_OPTIONS.samples)),
			1,
			SSGI_MAX_SAMPLES
		),
		intensity: Math.max(
			0,
			finiteOr(options?.intensity, DEFAULT_SSGI_OPTIONS.intensity)
		),
		falloff: Math.max(
			0.1,
			finiteOr(options?.falloff, DEFAULT_SSGI_OPTIONS.falloff)
		),
		depthPhi: Math.max(
			0.01,
			finiteOr(options?.depthPhi, DEFAULT_SSGI_OPTIONS.depthPhi)
		),
		normalPhi: Math.max(
			0.1,
			finiteOr(options?.normalPhi, DEFAULT_SSGI_OPTIONS.normalPhi)
		),
		albedoBoost: Math.max(
			0,
			finiteOr(options?.albedoBoost, DEFAULT_SSGI_OPTIONS.albedoBoost)
		),
	};
}

/**
 * Creates packed SSGI shader parameters.
 *
 * @param width Target width.
 * @param height Target height.
 * @param options Resolved SSGI options.
 * @returns Twelve float parameters expected by the SSGI compute shader.
 * @sideEffects None.
 */
export function createSSGIKernelParams(
	width: number,
	height: number,
	options: ResolvedSSGIOptions
): Float32Array {
	return new Float32Array([
		1 / Math.max(width, 1),
		1 / Math.max(height, 1),
		options.radius,
		options.intensity,
		options.falloff,
		options.depthPhi,
		options.normalPhi,
		options.albedoBoost,
		options.samples,
		0,
		0,
		0,
	]);
}

/**
 * WebGPU implementation of the screen-space global illumination pass.
 */
export class WebGPUScreenSpaceGlobalIlluminationImplementation
	implements PostProcessPassImplementation<WebGPUSSGIContext>
{
	public readonly id = "ssgi:webgpu";
	public readonly metadata = {
		context: WEBGPU_SCREEN_POST_PROCESS_CONTEXT_METADATA,
	};
	private _resources = new WeakMap<PostProcessSharedContext, WebGPUSSGIResources>();
	private _resourceSet = new Set<WebGPUSSGIResources>();

	public async warmup(context: WebGPUSSGIContext | undefined): Promise<void> {
		if (context) {
			await this._ensureResources(context.shared);
		}
	}

	public async execute(
		request: PostProcessPassRequest,
		context: WebGPUSSGIContext | undefined
	): Promise<PostProcessPassResult> {
		if (!context?.encoder || !context.targets) {
			return { ran: false };
		}
		const ran = await this._runSSGIKernel(request, context);
		return ran ? { ran: true } : { ran: false };
	}

	public invalidate(): void {
		for (const resources of this._resourceSet) {
			resources.shared.invalidateBindingsByPrefix("ssgi-");
		}
	}

	public destroy(): void {
		for (const resources of this._resourceSet) {
			resources.shared.destroyManagedResource(
				resources.pipeline,
				"SSGI pipeline"
			);
			resources.shared.destroyManagedResource(
				resources.module,
				"SSGI shader module"
			);
			resources.shared.destroyManagedResource(
				resources.params,
				"SSGI params buffer"
			);
			resources.shared.invalidateBindingsByPrefix("ssgi-");
			resources.module = null;
			resources.pipeline = null;
			resources.params = null;
		}
		this._resourceSet.clear();
		this._resources = new WeakMap<PostProcessSharedContext, WebGPUSSGIResources>();
	}

	private async _runSSGIKernel(
		request: PostProcessPassRequest,
		context: WebGPUSSGIContext
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
		const target =
			targets.sceneColor === targets.postPong ? targets.postPing : targets.postPong;
		const options = resolveSSGIOptions(request.options as SSGIOptions);
		context.shared.compute.writeBuffer(
			resources.params,
			createSSGIKernelParams(target.width, target.height, options) as unknown as BufferSource
		);
		const binding = context.shared.getCachedBindGroup(
			`ssgi-${target === targets.postPing ? "ping" : "pong"}`,
			resources.pipeline,
			[
				{ binding: 0, resource: targets.sceneColor },
				{ binding: 1, resource: targets.gAlbedoAlpha },
				{ binding: 2, resource: targets.gNormalRoughMetal },
				{ binding: 3, resource: targets.gMotionDepth },
				{ binding: 4, resource: context.shared.sampler },
				{ binding: 5, resource: resources.params },
				{ binding: 6, resource: target },
			],
			"WebGPUSSGI_Binding"
		);
		context.encoder.beginComputePass({ label: "WebGPUSSGI" });
		context.encoder.setComputePipeline(resources.pipeline);
		context.encoder.setBindingGroup(0, binding);
		context.encoder.dispatchWorkgroups(
			ceilDiv(target.width, WORKGROUP_SIZE),
			ceilDiv(target.height, WORKGROUP_SIZE),
			1
		);
		context.encoder.endComputePass();
		context.publishColorTarget?.(target);
		return true;
	}

	private async _ensureResources(
		shared: PostProcessSharedContext
	): Promise<WebGPUSSGIResources> {
		let resources = this._resources.get(shared);
		if (!resources) {
			resources = { shared, module: null, pipeline: null, params: null };
			this._resources.set(shared, resources);
			this._resourceSet.add(resources);
		}
		await shared.ensureCommonResources();
		if (!resources.module) {
			const shader = await loadPostProcessShaderPartComposite("ssgi");
			resources.module = await shared.compute.createShaderModule({
				label: "WebGPUSSGIShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!resources.pipeline) {
			resources.pipeline = shared.compute.createComputePipeline({
				label: "WebGPUSSGIPipeline",
				compute: { module: resources.module, entryPoint: "csMain" },
			});
		}
		if (!resources.params) {
			resources.params = shared.compute.createBuffer({
				label: "WebGPUSSGIParams",
				size: 12 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
		}
		return resources;
	}
}

export interface ScreenSpaceGlobalIlluminationPassConfig
	extends Omit<
		PostProcessPassConfig<SSGIOptions>,
		| "id"
		| "builtIn"
		| "warningLabel"
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
	public constructor(config: ScreenSpaceGlobalIlluminationPassConfig = {}) {
		super({
			...config,
			...SCREEN_SPACE_GLOBAL_ILLUMINATION_PASS_ORDER,
			builtIn: true,
			warningLabel: "SSGI",
			implementations: {
				webgpu: new WebGPUScreenSpaceGlobalIlluminationImplementation(),
			},
		});
	}

	public override normalizeOptions(): ResolvedSSGIOptions {
		return resolveSSGIOptions(this.getRawOptions());
	}

	public override getRequirements(): PostProcessPassRequirements {
		return { gBuffer: ["color", "depth", "normal", "albedo"] };
	}
}
