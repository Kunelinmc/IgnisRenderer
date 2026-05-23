import { CameraType } from "../../cameras/Camera";
import {
	DEFAULT_VOLUMETRIC_OPTIONS,
	type VolumetricOptions,
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
import type { PostProcessorLike } from "../../renderers/software/PostProcessor";
import {
	WEBGPU_2D_COMPUTE_WORKGROUP_SIZE as WORKGROUP_SIZE,
	WEBGPU_MAX_VOLUMETRIC_LIGHTS as MAX_VOLUMETRIC_LIGHTS,
	WEBGPU_VOLUMETRIC_LIGHT_STRIDE_FLOATS as VOLUMETRIC_LIGHT_STRIDE_FLOATS,
} from "../../renderers/webgpu/constants";
import { getWebGPUVolumetricLightLayout } from "../../renderers/webgpu/bufferLayouts";
import type { WebGPUFrameTargets } from "../../renderers/webgpu/WebGPUPostProcessContracts";
import { WebGPUHiZPostProcessHelper } from "../../renderers/webgpu/postprocess/HiZPostProcessHelper";
import type { PostProcessSharedContext } from "../../renderers/webgpu/postprocess/PostProcessSharedContext";
import type { WebGPULightingState } from "../../renderers/webgpu/types";
import { ceilDiv, finiteOr } from "../../maths/Misc";
import { loadPostProcessShaderPartComposite } from "../../shaders/webgpu/shaderSource";
import {
	PostProcessPass,
	type PostProcessPassConfig,
} from "../PostProcessPass";
import type {
	PostProcessHistoryDescriptor,
	PostProcessPassImplementation,
	PostProcessPassRequest,
	PostProcessPassRequirements,
	PostProcessPassResult,
} from "../types";

const DEFAULT_HISTORY_USAGE = ["sampled", "storage", "render-target"] as const;
const MOTION_HISTORY_USAGE = ["sampled", "copy-dst", "render-target"] as const;
export const VOLUMETRIC_LIGHTING_PASS_ID = "volumetric";

export interface SoftwareVolumetricLightingContext {
	readonly processor: PostProcessorLike | null;
	readonly canvasContext: CanvasRenderingContext2D | null;
}

export interface WebGPUVolumetricLightingContext {
	readonly encoder?: ICommandEncoder;
	readonly targets?: WebGPUFrameTargets;
	readonly shared: PostProcessSharedContext;
	readonly frameBinding?: IBindingGroup;
	readonly lightingState?: WebGPULightingState | null;
	publishColorTarget?(texture: IRenderTexture): void;
	writeMotionHistoryFromCurrent?(): void | Promise<void>;
}

interface WebGPUVolumetricResources {
	shared: PostProcessSharedContext;
	hiz: WebGPUHiZPostProcessHelper;
	module: IShaderModule | null;
	pipeline: IComputePipeline | null;
	params: IRenderBuffer | null;
	lightBuffer: IRenderBuffer | null;
	lightCapacity: number;
	frameIndex: number;
	groupLayout0: GPUBindGroupLayout | null;
	pipelineLayout: GPUPipelineLayout | null;
}

/**
 * Software implementation of volumetric lighting.
 */
export class SoftwareVolumetricLightingImplementation
	implements
		PostProcessPassImplementation<
			SoftwareVolumetricLightingContext,
			VolumetricOptions
		>
{
	public readonly id = "volumetric:software";

	public execute(
		request: PostProcessPassRequest<VolumetricOptions>,
		context: SoftwareVolumetricLightingContext | undefined
	): PostProcessPassResult {
		if (!context?.processor || !context.canvasContext) {
			return { ran: false };
		}
		context.processor.applyVolumetricLight(
			request.frameContext,
			context.canvasContext
		);
		return { ran: true };
	}
}

/**
 * WebGPU implementation of volumetric lighting.
 */
export class WebGPUVolumetricLightingImplementation
	implements
		PostProcessPassImplementation<
			WebGPUVolumetricLightingContext,
			VolumetricOptions
		>
{
	public readonly id = "volumetric:webgpu";
	private _resources = new WeakMap<
		PostProcessSharedContext,
		WebGPUVolumetricResources
	>();
	private _resourceSet = new Set<WebGPUVolumetricResources>();

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
		await context.writeMotionHistoryFromCurrent?.();
		return {
			ran: true,
			updatedHistoryIds: ["volumetric", "volumetric-reservoir", "motion"],
		};
	}

	public invalidate(): void {
		for (const resources of this._resourceSet) {
			resources.shared.invalidateBindingsByPrefix("volumetric-");
		}
	}

	public destroy(): void {
		for (const resources of this._resourceSet) {
			resources.hiz.destroy();
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
		this._resourceSet.clear();
		this._resources = new WeakMap<
			PostProcessSharedContext,
			WebGPUVolumetricResources
		>();
	}

	private async _runVolumetricKernel(
		request: PostProcessPassRequest<VolumetricOptions>,
		context: WebGPUVolumetricLightingContext
	): Promise<boolean> {
		if (request.frameContext.camera.type === CameraType.Orthographic) {
			context.shared.warn(
				"webgpu-volumetric-orthographic-disabled",
				"WebGPU volumetric lighting is disabled for orthographic cameras."
			);
			return false;
		}
		const resources = await this._ensureResources(context.shared);
		const lightCount = this._updateLightBuffer(
			resources,
			context.lightingState ?? null
		);
		if (
			!context.encoder ||
			!context.targets ||
			!context.frameBinding ||
			!context.shared.sampler ||
			!resources.pipeline ||
			!resources.params ||
			!resources.lightBuffer
		) {
			return false;
		}
		const hiZMips = await resources.hiz.build(context.encoder, context.targets);
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
				1 / Math.max(context.targets.sceneColor.width, 1),
				1 / Math.max(context.targets.sceneColor.height, 1),
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

		const target =
			context.targets.sceneColor === context.targets.postPong ?
				context.targets.postPing
			:	context.targets.postPong;
		const binding = context.shared.getCachedBindGroup(
			`volumetric-${target === context.targets.postPing ? "ping" : "pong"}`,
			resources.pipeline,
			[
				{ binding: 0, resource: context.targets.sceneColor },
				{ binding: 1, resource: context.targets.gMotionDepth },
				{ binding: 2, resource: context.targets.hiZ },
				{ binding: 3, resource: context.targets.volumetricHistoryRead },
				{ binding: 4, resource: context.targets.motionHistoryRead },
				{ binding: 5, resource: context.shared.sampler },
				{ binding: 6, resource: resources.params },
				{ binding: 7, resource: target },
				{ binding: 8, resource: context.targets.volumetricHistoryWrite },
				{
					binding: 9,
					resource: context.targets.volumetricReservoirHistoryRead,
				},
				{
					binding: 10,
					resource: context.targets.volumetricReservoirHistoryWrite,
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
		context.publishColorTarget?.(target);
		return true;
	}

	private _updateLightBuffer(
		resources: WebGPUVolumetricResources,
		lightingState: WebGPULightingState | null
	): number {
		const sourceLights = lightingState?.volumetricLights ?? [];
		const clampedLightCount = Math.min(sourceLights.length, MAX_VOLUMETRIC_LIGHTS);
		if (sourceLights.length > MAX_VOLUMETRIC_LIGHTS) {
			resources.shared.warn(
				"webgpu-volumetric-light-count-clamped",
				`WebGPU volumetric ReSTIR clamps light count to ${MAX_VOLUMETRIC_LIGHTS}; extra lights are skipped`
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
		shared: PostProcessSharedContext
	): Promise<WebGPUVolumetricResources> {
		let resources = this._resources.get(shared);
		if (!resources) {
			resources = {
				shared,
				hiz: new WebGPUHiZPostProcessHelper(shared),
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
			this._resourceSet.add(resources);
		}
		await resources.hiz.ensureResources();
		if (!resources.module) {
			const shader = await loadPostProcessShaderPartComposite("volumetric");
			resources.module = await shared.compute.createShaderModule({
				label: "WebGPUVolumetricShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
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
				resources.pipeline = shared.compute.createComputePipeline({
					label: "WebGPUVolumetricPipeline",
					layout: resources.pipelineLayout,
					compute: { module: resources.module, entryPoint: "csMain" },
				});
			} else {
				resources.pipeline = shared.compute.createComputePipeline({
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
		| "capabilityId"
		| "warningLabel"
		| "placement"
		| "order"
		| "implementations"
	> {}

/**
 * Stateful logical volumetric lighting pass shared by Software and WebGPU.
 */
export class VolumetricLightingPass extends PostProcessPass<
	VolumetricOptions,
	VolumetricOptions
> {
	public constructor(config: VolumetricLightingPassConfig = {}) {
		super({
			...config,
			id: VOLUMETRIC_LIGHTING_PASS_ID,
			builtIn: true,
			capabilityId: VOLUMETRIC_LIGHTING_PASS_ID,
			warningLabel: "volumetric effects",
			placement: "atmosphere",
			order: 300,
			implementations: {
				software: new SoftwareVolumetricLightingImplementation(),
				webgpu: new WebGPUVolumetricLightingImplementation(),
			},
		});
	}

	public override normalizeOptions(): VolumetricOptions {
		return {
			...DEFAULT_VOLUMETRIC_OPTIONS,
			...this.getRawOptions(),
		};
	}

	public override getRequirements(): PostProcessPassRequirements {
		return { gBuffer: ["depth", "motion"] };
	}

	public override getHistoryDescriptors(): readonly PostProcessHistoryDescriptor[] {
		return [
			{ id: "volumetric", usage: DEFAULT_HISTORY_USAGE },
			{ id: "volumetric-reservoir", usage: DEFAULT_HISTORY_USAGE },
			{ id: "motion", usage: MOTION_HISTORY_USAGE },
		];
	}
}
