import { CameraType } from "../../cameras/Camera";
import {
	type DirectionalLight,
	type PointLight,
	type SpotLight,
} from "../../lights";
import type { ICommandEncoder } from "../../backends/ICommandEncoder";
import {
	MAX_EXPOSURE,
	MAX_VOLUMETRIC_LIGHTS,
	POST_PROCESS_NOISE_REFERENCE_WIDTH,
} from "../../backends/constants";
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
	WEBGPU_VOLUMETRIC_LIGHT_STRIDE_FLOATS as VOLUMETRIC_LIGHT_STRIDE_FLOATS,
} from "../../backends/webgpu/constants";
import { getWebGPUVolumetricLightLayout } from "../../backends/webgpu/bufferLayouts";
import type {
	WebGPUPostProcessFrameTargets,
	WebGPUPostProcessServices,
} from "../../backends/webgpu/WebGPUPostProcessContracts";
import { WEBGPU_VOLUMETRIC_LIGHTING_DATA } from "../../backends/webgpu/WebGPUFrameFeatureModules";
import type { WebGPUVolumetricLightingData } from "../../backends/webgpu/types";
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
	PostProcessPassImplementation,
	PostProcessPassRequest,
	PostProcessResourceAccessor,
	PostProcessPassResult,
	PostProcessTransientDescriptor,
} from "../types";

const DEFAULT_HISTORY_USAGE = ["sampled", "storage", "render-target"] as const;
const MOTION_HISTORY_USAGE = ["sampled", "copy-dst", "render-target"] as const;
const VOLUMETRIC_HISTORY_DESCRIPTORS = [{
	id: "volumetric",
	usage: DEFAULT_HISTORY_USAGE,
}, {
	id: "volumetric-reservoir",
	usage: DEFAULT_HISTORY_USAGE,
}, {
	id: "motion",
	usage: MOTION_HISTORY_USAGE,
}] as const satisfies readonly PostProcessHistoryDescriptor[];
export const VOLUMETRIC_LIGHTING_PASS_ID = "volumetric";
export const VOLUMETRIC_LIGHTING_PASS_ORDER = {
	id: VOLUMETRIC_LIGHTING_PASS_ID,
	placement: "atmosphere",
	order: 300,
	incremental: {
		firstPass: "volumetric",
		grade: "cinematic",
		inflationRadius: 16,
	},
} as const satisfies PostProcessScheduleEntry;
export interface VolumetricOptions {
	/** Ray-march step count. Higher values reduce banding at higher GPU cost. */
	samples?: number;
	/** Reserved backend-specific quality scale divisor. */
	downsample?: number;
	/** Overall scattering contribution added to the scene color. */
	weight?: number;
	/** Exposure multiplier applied to the accumulated light shaft result. */
	exposure?: number;
	/** Participating-media density. Higher values make fog volumes thicker. */
	airDensity?: number;
	/**
	 * Henyey-Greenstein phase anisotropy. Positive values emphasize forward
	 * scattering, negative values emphasize back scattering.
	 */
	anisotropy?: number;
	/** Maximum world-space ray distance sampled from the camera. */
	maxRayDistance?: number;
	/** Fraction of light scattered instead of absorbed, clamped to `[0, 1]`. */
	scatteringAlbedo?: number;
	/** Step interval for shadow lookups. Higher values reduce shadowing cost. */
	shadowSampleInterval?: number;
	/** Whether the provided depth buffer is already linearized. */
	isLinearDepth?: boolean;
	/** Enables depth-adaptive ray marching to spend samples where detail changes. */
	adaptiveSteps?: boolean;
	/** Enables bilateral upscaling for downsampled volumes. */
	useBilateralUpscale?: boolean;
	/** Depth tolerance for bilateral upscale; lower values preserve harder edges. */
	bilateralDepthSigma?: number;
	/** ReSTIR candidate count per pixel. Higher values improve light selection. */
	restirCandidates?: number;
	/** Temporal reservoir blend factor. Higher values stabilize but can ghost. */
	restirTemporalWeight?: number;
	/** Maximum ReSTIR reservoir weight scale to prevent bright outliers. */
	restirScaleClamp?: number;
	/** Allows backend-specific experimental volumetric options. */
	[key: string]: unknown;
}

export const DEFAULT_VOLUMETRIC_OPTIONS: Required<
	Pick<
		VolumetricOptions,
		| "samples"
		| "downsample"
		| "weight"
		| "exposure"
		| "airDensity"
		| "anisotropy"
		| "maxRayDistance"
		| "scatteringAlbedo"
		| "shadowSampleInterval"
		| "isLinearDepth"
		| "adaptiveSteps"
		| "useBilateralUpscale"
		| "bilateralDepthSigma"
		| "restirCandidates"
		| "restirTemporalWeight"
		| "restirScaleClamp"
	>
> = {
	samples: 32,
	downsample: 1,
	weight: 4,
	exposure: 1,
	airDensity: 1,
	anisotropy: 0.2,
	maxRayDistance: 300,
	scatteringAlbedo: 0.9,
	shadowSampleInterval: 2,
	isLinearDepth: true,
	adaptiveSteps: true,
	useBilateralUpscale: true,
	bilateralDepthSigma: 0.05,
	restirCandidates: 8,
	restirTemporalWeight: 0.8,
	restirScaleClamp: 24,
};

/** @internal WebGPU context supplied to the built-in volumetric lighting implementation. */
export interface WebGPUVolumetricLightingContext {
	readonly encoder?: ICommandEncoder;
	readonly targets?: WebGPUPostProcessFrameTargets;
	readonly shared: WebGPUPostProcessServices;
	readonly frameBinding?: IBindingGroup;
	readonly resources: PostProcessResourceAccessor<IRenderTexture>;
	getFrameData<T>(key: unknown): T | undefined;
}

interface WebGPUVolumetricResources {
	shared: WebGPUPostProcessServices;
	module: IShaderModule | null;
	pipeline: IComputePipeline | null;
	params: IRenderBuffer | null;
	lightBuffer: IRenderBuffer | null;
	lightCapacity: number;
	frameIndex: number;
	groupLayout0: GPUBindGroupLayout | null;
	pipelineLayout: GPUPipelineLayout | null;
}

/** @internal WebGPU implementation for the built-in volumetric lighting pass. */
export class WebGPUVolumetricLightingImplementation
	implements
		PostProcessPassImplementation<
			WebGPUVolumetricLightingContext,
			VolumetricOptions
		>
{
	public readonly id = "volumetric:webgpu";
	public describeExecution() {
		return {
			...WEBGPU_VERSIONED_EXECUTION,
			gBuffer: (["depth", "motion"] as const).map((semantic) => ({
				semantic,
				...POST_PROCESS_SAMPLED_READ,
			})),
			histories: VOLUMETRIC_HISTORY_DESCRIPTORS.map((descriptor) => ({
				descriptor,
				read: [POST_PROCESS_SAMPLED_READ],
				write: [POST_PROCESS_STORAGE_WRITE],
			})),
			shared: [WEBGPU_HIZ_SHARED_RESOURCE],
		} satisfies PostProcessExecutionDeclaration;
	}
	private _resources = new Map<
		WebGPUPostProcessServices,
		WebGPUVolumetricResources
	>();

	public async warmup(
		context: WebGPUVolumetricLightingContext | undefined
	): Promise<void> {
		if (context) {
			await this._ensureResources(context.shared);
		}
	}

	public async execute(
		request: PostProcessPassRequest<VolumetricOptions>,
		context: WebGPUVolumetricLightingContext | undefined
	): Promise<PostProcessPassResult> {
		if (
			!context?.encoder ||
			!context.targets ||
			!context.frameBinding
		) {
			return { ran: false };
		}
		const ran = await this._runVolumetricKernel(request, context);
		if (!ran) {
			return { ran: false };
		}
		await context.resources.copyGBufferToHistory("motion", "motion");
		return {
			ran: true,
			updatedHistoryIds: ["volumetric", "volumetric-reservoir", "motion"],
		};
	}

	public invalidate(): void {
		for (const resources of this._resources.values()) {
			resources.shared.invalidateBindingsByPrefix("volumetric-");
		}
	}

	public destroy(): void {
		for (const resources of this._resources.values()) {
			resources.shared.destroyManagedResource(
				resources.pipeline,
				"volumetric pipeline"
			);
			resources.shared.destroyManagedResource(
				resources.module,
				"volumetric shader module"
			);
			resources.shared.destroyManagedResource(
				resources.params,
				"volumetric params buffer"
			);
			resources.shared.destroyManagedResource(
				resources.lightBuffer,
				"volumetric light buffer"
			);
			resources.shared.invalidateBindingsByPrefix("volumetric-");
			resources.module = null;
			resources.pipeline = null;
			resources.params = null;
			resources.lightBuffer = null;
			resources.lightCapacity = 0;
			resources.frameIndex = 0;
			resources.groupLayout0 = null;
			resources.pipelineLayout = null;
		}
		this._resources.clear();
	}

	private async _runVolumetricKernel(
		request: PostProcessPassRequest<VolumetricOptions>,
		context: WebGPUVolumetricLightingContext
	): Promise<boolean> {
		if (request.frameContext.viewCamera.type === CameraType.Orthographic) {
			context.shared.warn(
				"webgpu-volumetric-orthographic-disabled",
				"WebGPU volumetric lighting is disabled for orthographic cameras."
			);
			return false;
		}
		const resources = await this._ensureResources(context.shared);
		const lightCount = this._updateLightBuffer(
			resources,
			context.getFrameData<WebGPUVolumetricLightingData>(
				WEBGPU_VOLUMETRIC_LIGHTING_DATA,
			),
		);
		const hiZ = context.resources.getShared(WEBGPU_HIZ_SHARED_RESOURCE.id);
		const history = context.resources.getHistory("volumetric");
		const reservoirHistory = context.resources.getHistory("volumetric-reservoir");
		const motionHistory = context.resources.getHistory("motion");
		const depthTexture = context.resources.getGBuffer("depth");
		const input = context.resources.color.input;
		if (
			!context.encoder ||
			!context.targets ||
			!context.frameBinding ||
			!hiZ ||
			!context.shared.sampler ||
			!resources.pipeline ||
			!resources.params ||
			!resources.lightBuffer ||
			!history.read ||
			!history.write ||
			!reservoirHistory.read ||
			!reservoirHistory.write ||
			!motionHistory.read ||
			!motionHistory.write ||
			!depthTexture ||
			!input
		) {
			return false;
		}
		const hiZMips = context.shared.getHiZBuilder().getMipViews(hiZ);
		if (hiZMips.length === 0) {
			return false;
		}

		const options = request.options ?? {};
		const samples = Math.max(
			1,
			Math.min(
				128,
				finiteOr(options.samples, DEFAULT_VOLUMETRIC_OPTIONS.samples)
			)
		);
		const weight = Math.max(
			0,
			finiteOr(options.weight, DEFAULT_VOLUMETRIC_OPTIONS.weight)
		);
		const exposure = Math.max(
			0,
			finiteOr(options.exposure, DEFAULT_VOLUMETRIC_OPTIONS.exposure)
		);
		const airDensity = Math.max(
			0.001,
			finiteOr(options.airDensity, DEFAULT_VOLUMETRIC_OPTIONS.airDensity)
		);
		const anisotropy = Math.max(
			-0.95,
			Math.min(
				0.95,
				finiteOr(options.anisotropy, DEFAULT_VOLUMETRIC_OPTIONS.anisotropy)
			)
		);
		const maxRayDistance = Math.max(
			0.1,
			finiteOr(
				options.maxRayDistance,
				DEFAULT_VOLUMETRIC_OPTIONS.maxRayDistance
			)
		);
		const scatteringAlbedo = Math.max(
			0,
			Math.min(
				1,
				finiteOr(
					options.scatteringAlbedo,
					DEFAULT_VOLUMETRIC_OPTIONS.scatteringAlbedo
				)
			)
		);
		const shadowSampleInterval = Math.max(
			1,
			Math.min(
				32,
				finiteOr(
					options.shadowSampleInterval,
					DEFAULT_VOLUMETRIC_OPTIONS.shadowSampleInterval
				)
			)
		);
		const adaptiveSteps = options.adaptiveSteps === false ? 0 : 1;
		const depthThickness = Math.max(
			0.01,
			finiteOr(
				options.bilateralDepthSigma,
				DEFAULT_VOLUMETRIC_OPTIONS.bilateralDepthSigma
			) * 8
		);
		const maxMip = Math.max(0, hiZMips.length - 1);
		const restirCandidates = Math.max(
			1,
			Math.min(
				64,
				finiteOr(
					options.restirCandidates,
					DEFAULT_VOLUMETRIC_OPTIONS.restirCandidates
				)
			)
		);
		const restirTemporalWeight = Math.max(
			0,
			Math.min(
				1,
				finiteOr(
					options.restirTemporalWeight,
					DEFAULT_VOLUMETRIC_OPTIONS.restirTemporalWeight
				)
			)
		);
		const restirScaleClamp = Math.max(
			1,
			finiteOr(
				options.restirScaleClamp,
				DEFAULT_VOLUMETRIC_OPTIONS.restirScaleClamp
			)
		);
		resources.frameIndex = (resources.frameIndex + 1) % 4096;

		context.shared.compute.writeBuffer(
			resources.params,
			new Float32Array([
				1 / Math.max(input.width, 1),
				1 / Math.max(input.height, 1),
				samples,
				weight,
				exposure,
				airDensity,
				anisotropy,
				maxRayDistance,
				scatteringAlbedo,
				shadowSampleInterval,
				adaptiveSteps,
				depthThickness,
				maxMip,
				0.75,
				(request.histories.volumetric?.valid ?? false) &&
					(request.histories.motion?.valid ?? false) ?
					1
				:	0,
				lightCount,
				restirCandidates,
				restirTemporalWeight,
				restirScaleClamp,
				resources.frameIndex,
			])
		);

		const target = context.resources.color.output;
		if (!target) return false;
		const binding = context.shared.getCachedBindGroup(
			`volumetric-${target === context.targets.postPing ? "ping" : "pong"}`,
			resources.pipeline,
			[
				{ binding: 0, resource: input },
				{ binding: 1, resource: depthTexture },
				{ binding: 2, resource: hiZ },
				{ binding: 3, resource: history.read },
				{ binding: 4, resource: motionHistory.read },
				{ binding: 5, resource: context.shared.sampler },
				{ binding: 6, resource: resources.params },
				{ binding: 7, resource: target },
				{ binding: 8, resource: history.write },
				{
					binding: 9,
					resource: reservoirHistory.read,
				},
				{
					binding: 10,
					resource: reservoirHistory.write,
				},
				{ binding: 11, resource: resources.lightBuffer },
			],
			"WebGPUVolumetric_Binding"
		);
		context.encoder.beginComputePass({ label: "WebGPUVolumetric" });
		context.encoder.setComputePipeline(resources.pipeline);
		context.encoder.setBindingGroup(0, binding);
		context.encoder.setBindingGroup(1, context.frameBinding);
		context.encoder.dispatchWorkgroups(
			ceilDiv(target.width, WORKGROUP_SIZE),
			ceilDiv(target.height, WORKGROUP_SIZE),
			1
		);
		context.encoder.endComputePass();
		return true;
	}

	private _updateLightBuffer(
		resources: WebGPUVolumetricResources,
		volumetricLighting: WebGPUVolumetricLightingData | undefined
	): number {
		const sourceLights = volumetricLighting?.lights ?? [];
		const clampedLightCount = Math.min(sourceLights.length, MAX_VOLUMETRIC_LIGHTS);
		if (sourceLights.length > MAX_VOLUMETRIC_LIGHTS) {
			resources.shared.warn(
				"webgpu-volumetric-light-count-clamped",
				`WebGPU volumetric lighting samples at most ${MAX_VOLUMETRIC_LIGHTS} effective surface lights; extra lights are skipped`
			);
		}
		this._ensureLightBufferCapacity(resources, clampedLightCount);
		if (!resources.lightBuffer) {
			return 0;
		}

		const packedCount = Math.max(1, clampedLightCount);
		const layout = getWebGPUVolumetricLightLayout(packedCount);
		const packed = layout.createWriter();
		packed.expectByteLength(
			packedCount * VOLUMETRIC_LIGHT_STRIDE_FLOATS * 4,
			"VolumetricLightBuffer"
		);
		for (let i = 0; i < clampedLightCount; i++) {
			const light = sourceLights[i];
			const isDirectional = light.type === 0;
			const isSpot = light.type === 2;
			packed.writeVec([i, "positionRange"], [
				light.position[0],
				light.position[1],
				light.position[2],
				isDirectional ? -1 : Math.max(light.range, 0.001),
			]);
			packed.writeVec([i, "directionOuter"], [
				light.direction[0],
				light.direction[1],
				light.direction[2],
				isSpot ? light.outerCos : -2,
			]);
			packed.writeVec([i, "colorInner"], [
				light.color[0],
				light.color[1],
				light.color[2],
				isSpot ? light.innerCos : -2,
			]);
		}
		if (clampedLightCount === 0) {
			packed.writeVec([0, "positionRange"], [0, 0, 0, -1]);
		}
		resources.shared.compute.writeBuffer(
			resources.lightBuffer,
			packed.toFloat32Array()
		);
		return clampedLightCount;
	}

	private _ensureLightBufferCapacity(
		resources: WebGPUVolumetricResources,
		lightCount: number
	): void {
		const required = Math.max(1, lightCount);
		if (resources.lightBuffer && resources.lightCapacity >= required) {
			return;
		}
		let capacity = Math.max(1, resources.lightCapacity);
		while (capacity < required) {
			capacity *= 2;
		}
		resources.lightBuffer?.destroy();
		resources.lightBuffer = resources.shared.compute.createBuffer({
			label: "WebGPUVolumetricLights",
			size: capacity * VOLUMETRIC_LIGHT_STRIDE_FLOATS * 4,
			usage: BufferUsage.Storage | BufferUsage.CopyDst,
		});
		resources.lightCapacity = capacity;
	}

	private async _ensureResources(
		shared: WebGPUPostProcessServices
	): Promise<WebGPUVolumetricResources> {
		let resources = this._resources.get(shared);
		if (!resources) {
			resources = {
				shared,
				module: null,
				pipeline: null,
				params: null,
				lightBuffer: null,
				lightCapacity: 0,
				frameIndex: 0,
				groupLayout0: null,
				pipelineLayout: null,
			};
			this._resources.set(shared, resources);
		}
		await shared.ensureCommonResources();
		await shared.getHiZBuilder().ensureResources();
		if (!resources.module) {
			const shader = await ShaderSource.load(
				"webgpu.postprocess.volumetric"
			);
			resources.module = await shared.compute.createShaderModule({
				label: "WebGPUVolumetricShader",
				code: shader.source.code,
				sourceMap: shader.source.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!resources.pipeline) {
			if (shared.frameBindGroupLayout) {
				resources.groupLayout0 = shared.compute.createBindGroupLayout({
					label: "WebGPUVolumetric_GroupLayout0",
					entries: [
						{ binding: 0, visibility: GPUShaderStage.COMPUTE, texture: {} },
						{ binding: 1, visibility: GPUShaderStage.COMPUTE, texture: {} },
						{ binding: 2, visibility: GPUShaderStage.COMPUTE, texture: {} },
						{ binding: 3, visibility: GPUShaderStage.COMPUTE, texture: {} },
						{ binding: 4, visibility: GPUShaderStage.COMPUTE, texture: {} },
						{ binding: 5, visibility: GPUShaderStage.COMPUTE, sampler: {} },
						{
							binding: 6,
							visibility: GPUShaderStage.COMPUTE,
							buffer: { type: "uniform" },
						},
						{
							binding: 7,
							visibility: GPUShaderStage.COMPUTE,
							storageTexture: { format: "rgba16float", access: "write-only" },
						},
						{
							binding: 8,
							visibility: GPUShaderStage.COMPUTE,
							storageTexture: { format: "rgba16float", access: "write-only" },
						},
						{ binding: 9, visibility: GPUShaderStage.COMPUTE, texture: {} },
						{
							binding: 10,
							visibility: GPUShaderStage.COMPUTE,
							storageTexture: { format: "rgba16float", access: "write-only" },
						},
						{
							binding: 11,
							visibility: GPUShaderStage.COMPUTE,
							buffer: { type: "read-only-storage" },
						},
					],
				});
				resources.pipelineLayout = shared.compute.createPipelineLayout({
					label: "WebGPUVolumetric_PipelineLayout",
					bindGroupLayouts: [
						resources.groupLayout0,
						shared.frameBindGroupLayout,
					],
				});
				resources.pipeline = await shared.compute.createComputePipeline({
					label: "WebGPUVolumetricPipeline",
					layout: resources.pipelineLayout,
					compute: { module: resources.module, entryPoint: "csMain" },
				});
			} else {
				resources.pipeline = await shared.compute.createComputePipeline({
					label: "WebGPUVolumetricPipeline",
					compute: { module: resources.module, entryPoint: "csMain" },
				});
			}
		}
		if (!resources.params) {
			resources.params = shared.compute.createBuffer({
				label: "WebGPUVolumetricParams",
				size: 20 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
		}
		return resources;
	}
}

export interface VolumetricLightingPassConfig
	extends Omit<
		PostProcessPassConfig<VolumetricOptions>,
		| "id"
		| "builtIn"
		| "label"
		| "placement"
		| "order"
		| "implementations"
	> {}

/**
 * Stateful logical volumetric lighting pass for WebGPU.
 */
export class VolumetricLightingPass extends PostProcessPass<
	VolumetricOptions,
	VolumetricOptions
> {
	public constructor(config: VolumetricLightingPassConfig = {}) {
		super({
			...config,
			id: VOLUMETRIC_LIGHTING_PASS_ORDER.id,
			schedule: {
				placement: config.schedule?.placement ?? VOLUMETRIC_LIGHTING_PASS_ORDER.placement,
				order: config.schedule?.order ?? VOLUMETRIC_LIGHTING_PASS_ORDER.order,
				incremental: config.schedule?.incremental ?? VOLUMETRIC_LIGHTING_PASS_ORDER.incremental,
			},
			label: "volumetric effects",
			colorContract: config.colorContract ?? {
				input: "scene-linear-hdr",
				output: "scene-linear-hdr",
			},
			implementations: {
				webgpu: () => new WebGPUVolumetricLightingImplementation(),
			},
		});
	}

	public override normalizeOptions(): VolumetricOptions {
		return {
			...DEFAULT_VOLUMETRIC_OPTIONS,
			...this.getRawOptions(),
		};
	}

}
