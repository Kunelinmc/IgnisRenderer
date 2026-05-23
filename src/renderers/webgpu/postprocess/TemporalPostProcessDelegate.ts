import { CameraType } from "../../../cameras/Camera";
import type { FrameContext } from "../../../pipeline/types";
import {
	DEFAULT_VOLUMETRIC_OPTIONS,
	type VolumetricOptions,
} from "../../../pipeline/types";
import {
	BufferUsage,
	type IComputePipeline,
	type IRenderBuffer,
	type IShaderModule,
} from "../../types";
import { loadPostProcessShaderPartComposite } from "../../../shaders/webgpu/shaderSource";
import { ceilDiv, finiteOr } from "../../../maths/Misc";
import type { WebGPULightingState } from "../types";
import {
	WEBGPU_MAX_VOLUMETRIC_LIGHTS as MAX_VOLUMETRIC_LIGHTS,
	WEBGPU_VOLUMETRIC_LIGHT_STRIDE_FLOATS as VOLUMETRIC_LIGHT_STRIDE_FLOATS,
} from "../constants";
import { getWebGPUVolumetricLightLayout } from "../bufferLayouts";
import { WebGPUHiZPostProcessHelper } from "./HiZPostProcessHelper";
import { PostProcessSharedContext } from "./PostProcessSharedContext";
import type {
	WebGPUPostProcessRuntimePassRegistry,
	WebGPUPostProcessVolumetricExecuteRequest,
} from "./types";

const WORKGROUP_SIZE = 8;

export class TemporalPostProcessDelegate {
	private _shared: PostProcessSharedContext;
	private _hiz: WebGPUHiZPostProcessHelper;
	private _volumetricModule: IShaderModule | null = null;
	private _volumetricPipeline: IComputePipeline | null = null;
	private _volumetricParams: IRenderBuffer | null = null;
	private _volumetricLightBuffer: IRenderBuffer | null = null;
	private _volumetricLightCapacity = 0;
	private _volumetricFrameIndex = 0;
	private _volumetricGroupLayout0: GPUBindGroupLayout | null = null;
	private _volumetricPipelineLayout: GPUPipelineLayout | null = null;

	constructor(shared: PostProcessSharedContext) {
		this._shared = shared;
		this._hiz = new WebGPUHiZPostProcessHelper(shared);
	}

	/**
	 * Registers temporal post-process runtime passes with the owning runtime.
	 */
	public registerPasses(registry: WebGPUPostProcessRuntimePassRegistry): void {
		registry.registerRuntimePass({
			id: "volumetric",
			warmupHints: ["postprocess:volumetric", "postprocess:hiz"],
			warmup: async (hint) => {
				if (hint === "postprocess:hiz") {
					await this._hiz.ensureResources();
				} else {
					await this._ensureVolumetricResources();
				}
				return true;
			},
			execute: async (request) => {
				const historyUpdated = await this._executeVolumetric(
					request as WebGPUPostProcessVolumetricExecuteRequest
				);
				return { ran: historyUpdated, historyUpdated };
			},
			invalidateBindings: () => this.invalidateBindings(),
			onShaderRuntimeChanged: () => this.onShaderRuntimeChanged(),
		});
	}

	public invalidateBindings(): void {}

	public onShaderRuntimeChanged(): void {
		this._hiz.destroy();
		this._shared.destroyManagedResource(
			this._volumetricPipeline,
			"volumetric pipeline"
		);
		this._shared.destroyManagedResource(
			this._volumetricModule,
			"volumetric shader module"
		);
		this._volumetricModule = null;
		this._volumetricPipeline = null;
		this._shared.destroyManagedResource(
			this._volumetricParams,
			"volumetric params buffer"
		);
		this._volumetricParams = null;
		this._shared.destroyManagedResource(
			this._volumetricLightBuffer,
			"volumetric light buffer"
		);
		this._volumetricLightBuffer = null;
		this._volumetricLightCapacity = 0;
		this._volumetricFrameIndex = 0;
		this._volumetricGroupLayout0 = null;
		this._volumetricPipelineLayout = null;
	}

	public destroy(): void {
		this.onShaderRuntimeChanged();
	}

	private async _executeVolumetric(
		request: WebGPUPostProcessVolumetricExecuteRequest
	): Promise<boolean> {
		if (request.frameContext.camera.type === CameraType.Orthographic) {
			this._shared.warn(
				"webgpu-volumetric-orthographic-disabled",
				"WebGPU volumetric lighting is disabled for orthographic cameras."
			);
			return false;
		}
		await this._ensureVolumetricResources();
		const lightCount = this._updateVolumetricLightBuffer(request.lightingState);
		if (
			!this._shared.sampler ||
			!this._volumetricPipeline ||
			!this._volumetricParams ||
			!this._volumetricLightBuffer
		) {
			return false;
		}
		const hiZMips = await this._hiz.build(request.encoder, request.targets);
		if (hiZMips.length === 0) {
			return false;
		}

		const options = (request.options as VolumetricOptions) ?? {};
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
					options["restirCandidates"],
					DEFAULT_VOLUMETRIC_OPTIONS.restirCandidates
				)
			)
		);
		const restirTemporalWeight = Math.max(
			0,
			Math.min(
				1,
				finiteOr(
					options["restirTemporalWeight"],
					DEFAULT_VOLUMETRIC_OPTIONS.restirTemporalWeight
				)
			)
		);
		const restirScaleClamp = Math.max(
			1,
			finiteOr(
				options["restirScaleClamp"],
				DEFAULT_VOLUMETRIC_OPTIONS.restirScaleClamp
			)
		);
		this._volumetricFrameIndex = (this._volumetricFrameIndex + 1) % 4096;

		this._shared.compute.writeBuffer(
			this._volumetricParams,
			new Float32Array([
				1 / Math.max(request.targets.sceneColor.width, 1),
				1 / Math.max(request.targets.sceneColor.height, 1),
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
				request.historyValid ? 1 : 0,
				lightCount,
				restirCandidates,
				restirTemporalWeight,
				restirScaleClamp,
				this._volumetricFrameIndex,
			])
		);

		const target =
			request.targets.sceneColor === request.targets.postPong ?
				request.targets.postPing
			:	request.targets.postPong;
		const binding = this._shared.getCachedBindGroup(
			`volumetric-${target === request.targets.postPing ? "ping" : "pong"}`,
			this._volumetricPipeline,
			[
				{ binding: 0, resource: request.targets.sceneColor },
				{ binding: 1, resource: request.targets.gMotionDepth },
				{ binding: 2, resource: request.targets.hiZ },
				{ binding: 3, resource: request.targets.volumetricHistoryRead },
				{ binding: 4, resource: request.targets.motionHistoryRead },
				{ binding: 5, resource: this._shared.sampler },
				{ binding: 6, resource: this._volumetricParams },
				{ binding: 7, resource: target },
				{ binding: 8, resource: request.targets.volumetricHistoryWrite },
				{
					binding: 9,
					resource: request.targets.volumetricReservoirHistoryRead,
				},
				{
					binding: 10,
					resource: request.targets.volumetricReservoirHistoryWrite,
				},
				{ binding: 11, resource: this._volumetricLightBuffer },
			],
			"WebGPUVolumetric_Binding"
		);
		request.encoder.beginComputePass({ label: "WebGPUVolumetric" });
		request.encoder.setComputePipeline(this._volumetricPipeline);
		request.encoder.setBindingGroup(0, binding);
		request.encoder.setBindingGroup(1, request.frameBinding);
		request.encoder.dispatchWorkgroups(
			ceilDiv(target.width, WORKGROUP_SIZE),
			ceilDiv(target.height, WORKGROUP_SIZE),
			1
		);
		request.encoder.endComputePass();
		request.targets.sceneColor = target;
		return true;
	}

	private _updateVolumetricLightBuffer(
		lightingState: WebGPULightingState | null
	): number {
		const sourceLights = lightingState?.volumetricLights ?? [];
		const clampedLightCount = Math.min(sourceLights.length, MAX_VOLUMETRIC_LIGHTS);
		if (sourceLights.length > MAX_VOLUMETRIC_LIGHTS) {
			this._shared.warn(
				"webgpu-volumetric-light-count-clamped",
				`WebGPU volumetric ReSTIR clamps light count to ${MAX_VOLUMETRIC_LIGHTS}; extra lights are skipped`
			);
		}

		this._ensureVolumetricLightBufferCapacity(clampedLightCount);
		if (!this._volumetricLightBuffer) {
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

		this._shared.compute.writeBuffer(
			this._volumetricLightBuffer,
			packed.toFloat32Array()
		);
		return clampedLightCount;
	}

	private _ensureVolumetricLightBufferCapacity(lightCount: number): void {
		const required = Math.max(1, lightCount);
		if (
			this._volumetricLightBuffer &&
			this._volumetricLightCapacity >= required
		) {
			return;
		}

		let capacity = Math.max(1, this._volumetricLightCapacity);
		while (capacity < required) {
			capacity *= 2;
		}

		this._volumetricLightBuffer?.destroy();
		this._volumetricLightBuffer = this._shared.compute.createBuffer({
			label: "WebGPUVolumetricLights",
			size: capacity * VOLUMETRIC_LIGHT_STRIDE_FLOATS * 4,
			usage: BufferUsage.Storage | BufferUsage.CopyDst,
		});
		this._volumetricLightCapacity = capacity;
	}

	private async _ensureVolumetricResources(): Promise<void> {
		await this._hiz.ensureResources();
		if (!this._volumetricModule) {
			const shader = await loadPostProcessShaderPartComposite("volumetric");
			this._volumetricModule = await this._shared.compute.createShaderModule({
				label: "WebGPUVolumetricShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!this._volumetricPipeline) {
			if (this._shared.frameBindGroupLayout) {
				this._volumetricGroupLayout0 = this._shared.compute.createBindGroupLayout({
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
				this._volumetricPipelineLayout = this._shared.compute.createPipelineLayout(
					{
						label: "WebGPUVolumetric_PipelineLayout",
						bindGroupLayouts: [
							this._volumetricGroupLayout0,
							this._shared.frameBindGroupLayout,
						],
					}
				);
				this._volumetricPipeline = this._shared.compute.createComputePipeline({
					label: "WebGPUVolumetricPipeline",
					layout: this._volumetricPipelineLayout,
					compute: { module: this._volumetricModule, entryPoint: "csMain" },
				});
			} else {
				this._volumetricPipeline = this._shared.compute.createComputePipeline({
					label: "WebGPUVolumetricPipeline",
					compute: { module: this._volumetricModule, entryPoint: "csMain" },
				});
			}
		}
		if (!this._volumetricParams) {
			this._volumetricParams = this._shared.compute.createBuffer({
				label: "WebGPUVolumetricParams",
				size: 20 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
		}
	}
}
