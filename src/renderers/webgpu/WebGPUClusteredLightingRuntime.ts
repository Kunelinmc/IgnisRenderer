import { CameraType } from "../../cameras/Camera";
import type { PreparedScene } from "../../pipeline/types";
import type { ICommandEncoder } from "../ICommandEncoder";
import {
	BufferUsage,
	type IBindingGroup,
	type IComputePipeline,
	type IRenderBuffer,
} from "../types";
import type { IWebGPUComputeFacade } from "./computeFacade";
import { destroyResource, recordComputePass } from "./computeUtils";
import {
	WEBGPU_CLUSTERED_HEADER_FLAG_HAS_SHADOWED,
	WEBGPU_CLUSTERED_HEADER_FLAG_HAS_VOLUMETRIC,
	WEBGPU_CLUSTERED_HEADER_FLAG_OVERFLOW,
	WEBGPU_CLUSTERED_INDEX_LIGHT_MASK,
	WEBGPU_CLUSTERED_INDEX_SHADOW_BIT,
	WEBGPU_CLUSTERED_INDEX_TYPE_MASK,
	WEBGPU_CLUSTERED_INDEX_TYPE_SHIFT,
	WEBGPU_CLUSTERED_INDEX_VOLUMETRIC_BIT,
	WEBGPU_CLUSTERED_LIGHT_FLAG_AFFECTS_VOLUMETRIC,
	WEBGPU_CLUSTERED_LIGHT_FLAG_CASTS_SHADOW,
	WEBGPU_CLUSTERED_LIGHT_FLAG_TYPE_MASK,
	WEBGPU_CLUSTERED_LIGHT_TYPE_POINT,
	WEBGPU_CLUSTERED_LIGHT_TYPE_SPOT,
} from "./constants";
import {
	loadClusteredLightingCullShaderComposite,
} from "../../shaders/webgpu/shaderSource";
import type {
	WebGPUClusterGridParams,
	WebGPUFeatureState,
	WebGPULightingState,
} from "./types";

const CLUSTERED_SHADER_WORKGROUP_SIZE = 128;
const CLUSTERED_PARAMS_FLOATS = 12;
const CLUSTERED_LIGHT_STRIDE_FLOATS = 16;
const CLUSTERED_HEADER_STRIDE_UINTS = 4;

const CLUSTERED_PARAMS_DEFAULTS = {
	tileSizePx: 64,
	zSlices: 24,
	maxLights: 256,
	maxLightsPerCluster: 64,
} as const;

interface FrameClusterState {
	enabled: boolean;
	maxLights: number;
	maxLightsPerCluster: number;
	clusterCount: number;
	tilesX: number;
	tilesY: number;
	zSlices: number;
}

export class WebGPUClusteredLightingRuntime {
	private _compute: IWebGPUComputeFacade;
	private _warn: (key: string, message: string) => void;
	private _sceneLayout: GPUBindGroupLayout;
	private _frameLayout: GPUBindGroupLayout;

	private _state: FrameClusterState = {
		enabled: false,
		maxLights: 0,
		maxLightsPerCluster: CLUSTERED_PARAMS_DEFAULTS.maxLightsPerCluster,
		clusterCount: 1,
		tilesX: 1,
		tilesY: 1,
		zSlices: 1,
	};

	private _clusterParamsBuffer: IRenderBuffer | null = null;
	private _clusterLightBuffer: IRenderBuffer | null = null;
	private _clusterHeaderBuffer: IRenderBuffer | null = null;
	private _clusterIndexBuffer: IRenderBuffer | null = null;

	private _lightCapacity = 0;
	private _clusterCapacity = 0;
	private _indexCapacity = 0;

	private _sceneBinding: IBindingGroup | null = null;
	private _sceneBindingSources:
		| [IRenderBuffer, IRenderBuffer, IRenderBuffer, IRenderBuffer]
		| null = null;
	private _computeBinding: IBindingGroup | null = null;
	private _computeBindingSources:
		| [IRenderBuffer, IRenderBuffer, IRenderBuffer, IRenderBuffer]
		| null = null;

	private _computeShaderModule: any = null;
	private _computePipeline: IComputePipeline | null = null;
	private _computeGroupLayout0: GPUBindGroupLayout | null = null;
	private _computePipelineLayout: GPUPipelineLayout | null = null;
	private _built = false;

	constructor(
		computeFacade: IWebGPUComputeFacade,
		sceneLayout: GPUBindGroupLayout,
		frameLayout: GPUBindGroupLayout,
		warn: (key: string, message: string) => void
	) {
		this._compute = computeFacade;
		this._sceneLayout = sceneLayout;
		this._frameLayout = frameLayout;
		this._warn = warn;
	}

	public onShaderRuntimeChanged(): void {
		this._computeShaderModule = null;
		this._computePipeline = null;
		this._computeGroupLayout0 = null;
		this._computePipelineLayout = null;
		this._destroyBindingGroup(this._computeBinding);
		this._computeBinding = null;
		this._computeBindingSources = null;
	}

	public prepareFrame(
		frame: PreparedScene,
		features: WebGPUFeatureState,
		lighting: WebGPULightingState | null,
		renderWidth: number,
		renderHeight: number
	): void {
		this._built = false;
		const options = {
			tileSizePx:
				features.clusteredLightingOptions?.tileSizePx ??
				CLUSTERED_PARAMS_DEFAULTS.tileSizePx,
			zSlices:
				features.clusteredLightingOptions?.zSlices ??
				CLUSTERED_PARAMS_DEFAULTS.zSlices,
			maxLights:
				features.clusteredLightingOptions?.maxLights ??
				CLUSTERED_PARAMS_DEFAULTS.maxLights,
			maxLightsPerCluster:
				features.clusteredLightingOptions?.maxLightsPerCluster ??
				CLUSTERED_PARAMS_DEFAULTS.maxLightsPerCluster,
		};

		const renderW = Math.max(1, Math.floor(renderWidth));
		const renderH = Math.max(1, Math.floor(renderHeight));
		const tileSizePx = Math.max(8, Math.floor(options.tileSizePx));
		const tilesX = Math.max(1, Math.ceil(renderW / tileSizePx));
		const tilesY = Math.max(1, Math.ceil(renderH / tileSizePx));
		const zSlices = Math.max(1, Math.floor(options.zSlices));
		const clusterCount = Math.max(1, tilesX * tilesY * zSlices);
		const maxLightsPerCluster = Math.max(
			1,
			Math.floor(options.maxLightsPerCluster)
		);

		const isPerspective = frame.camera.type === CameraType.Perspective;
		if (features.enableClusteredLighting && !isPerspective) {
			this._warn(
				"webgpu-clustered-perspective-only",
				"WebGPU clustered lighting only supports perspective cameras; falling back to legacy forward lights"
			);
		}
		const near = Math.max(0.05, frame.camera.near ?? 0.1);
		const far = Math.max(near + 1e-3, frame.camera.far ?? near + 1);
		const logDenom = Math.log(far) - Math.log(near);
		const canCluster =
			features.enableClusteredLighting &&
			features.enableLighting &&
			isPerspective &&
			logDenom > 1e-6;

		const sourceLights = lighting?.clusteredLights ?? [];
		const requestedMaxLights = Math.max(1, Math.floor(options.maxLights));
		let maxLights = 0;
		if (canCluster) {
			maxLights = Math.min(sourceLights.length, requestedMaxLights);
			if (sourceLights.length > requestedMaxLights) {
				this._warn(
					"webgpu-clustered-light-budget",
					`WebGPU clustered lighting clamps lights to ${requestedMaxLights}; extra lights are skipped`
				);
			}
			if (maxLights > maxLightsPerCluster) {
				this._warn(
					"webgpu-clustered-overflow",
					`WebGPU clustered lighting may overflow cluster capacity (${maxLightsPerCluster}); overflowing entries are truncated`
				);
			}
		}

		const runtimeEnabled = canCluster && maxLights > 0;
		const activeClusterCount = runtimeEnabled ? clusterCount : 1;
		const activeMaxLightsPerCluster = runtimeEnabled ? maxLightsPerCluster : 1;
		const logScale = runtimeEnabled ? zSlices / logDenom : 0;
		const logBias = runtimeEnabled ? (-Math.log(near) * zSlices) / logDenom : 0;

		this._state = {
			enabled: runtimeEnabled,
			maxLights,
			maxLightsPerCluster: activeMaxLightsPerCluster,
			clusterCount: activeClusterCount,
			tilesX,
			tilesY,
			zSlices,
		};

		this._ensureBuffers(
			maxLights,
			activeClusterCount,
			activeMaxLightsPerCluster
		);
		const params = this._createClusterParamsBufferData({
			screenWidth: renderW,
			screenHeight: renderH,
			tilesX,
			tilesY,
			zSlices,
			clusterCount: activeClusterCount,
			near,
			far,
			logScale,
			logBias,
		});
		this._compute.writeBuffer(this._clusterParamsBuffer!, params);

		const lightData = this._createClusterLightBufferData(
			sourceLights,
			maxLights
		);
		this._compute.writeBuffer(this._clusterLightBuffer!, lightData);
	}

	public async build(
		encoder: ICommandEncoder,
		frameBinding: IBindingGroup
	): Promise<void> {
		if (this._built || !this._state.enabled || this._state.maxLights <= 0) {
			return;
		}
		await this._ensureComputeResources();
		if (!this._computePipeline || !this._computeBinding) {
			return;
		}

		recordComputePass(
			encoder,
			"WebGPUClusteredLightingCull",
			this._computePipeline,
			[
				{ index: 0, group: this._computeBinding },
				{ index: 1, group: frameBinding },
			],
			{
				x: Math.ceil(
					this._state.clusterCount / CLUSTERED_SHADER_WORKGROUP_SIZE
				),
				y: 1,
				z: 1,
			}
		);
		this._built = true;
	}

	public getSceneBinding(): IBindingGroup {
		if (
			!this._clusterParamsBuffer ||
			!this._clusterLightBuffer ||
			!this._clusterHeaderBuffer ||
			!this._clusterIndexBuffer
		) {
			this._ensureBuffers(0, 1, 1);
		}
		const paramsBuffer = this._clusterParamsBuffer!;
		const lightBuffer = this._clusterLightBuffer!;
		const headerBuffer = this._clusterHeaderBuffer!;
		const indexBuffer = this._clusterIndexBuffer!;
		const sources: [
			IRenderBuffer,
			IRenderBuffer,
			IRenderBuffer,
			IRenderBuffer,
		] = [paramsBuffer, lightBuffer, headerBuffer, indexBuffer];

		if (
			!this._sceneBinding ||
			!this._sceneBindingSources ||
			this._sceneBindingSources[0] !== sources[0] ||
			this._sceneBindingSources[1] !== sources[1] ||
			this._sceneBindingSources[2] !== sources[2] ||
			this._sceneBindingSources[3] !== sources[3]
		) {
			this._destroyBindingGroup(this._sceneBinding);
			this._sceneBinding = this._compute.createBindingGroup({
				label: "WebGPUClusteredSceneBinding",
				layout: this._sceneLayout,
				entries: [
					{ binding: 0, resource: paramsBuffer },
					{ binding: 1, resource: lightBuffer },
					{ binding: 2, resource: headerBuffer },
					{ binding: 3, resource: indexBuffer },
				],
			});
			this._sceneBindingSources = sources;
		}
		return this._sceneBinding;
	}

	public destroy(): void {
		this._destroyBindingGroup(this._sceneBinding);
		this._destroyBindingGroup(this._computeBinding);
		this._sceneBinding = null;
		this._computeBinding = null;
		this._sceneBindingSources = null;
		this._computeBindingSources = null;
		this._clusterParamsBuffer?.destroy();
		this._clusterParamsBuffer = null;
		this._clusterLightBuffer?.destroy();
		this._clusterLightBuffer = null;
		this._clusterHeaderBuffer?.destroy();
		this._clusterHeaderBuffer = null;
		this._clusterIndexBuffer?.destroy();
		this._clusterIndexBuffer = null;
		this._lightCapacity = 0;
		this._clusterCapacity = 0;
		this._indexCapacity = 0;
		this.onShaderRuntimeChanged();
	}

	private _ensureBuffers(
		maxLights: number,
		clusterCount: number,
		maxLightsPerCluster: number
	): void {
		if (!this._clusterParamsBuffer) {
			this._clusterParamsBuffer = this._compute.createBuffer({
				label: "WebGPUClusteredParams",
				size: CLUSTERED_PARAMS_FLOATS * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
		}

		const requiredLightCapacity = Math.max(1, maxLights);
		if (
			!this._clusterLightBuffer ||
			this._lightCapacity < requiredLightCapacity
		) {
			this._clusterLightBuffer?.destroy();
			this._clusterLightBuffer = this._compute.createBuffer({
				label: "WebGPUClusteredLights",
				size: requiredLightCapacity * CLUSTERED_LIGHT_STRIDE_FLOATS * 4,
				usage: BufferUsage.Storage | BufferUsage.CopyDst,
			});
			this._lightCapacity = requiredLightCapacity;
		}

		const requiredClusterCapacity = Math.max(1, clusterCount);
		if (
			!this._clusterHeaderBuffer ||
			this._clusterCapacity < requiredClusterCapacity
		) {
			this._clusterHeaderBuffer?.destroy();
			this._clusterHeaderBuffer = this._compute.createBuffer({
				label: "WebGPUClusteredHeaders",
				size: requiredClusterCapacity * CLUSTERED_HEADER_STRIDE_UINTS * 4,
				usage: BufferUsage.Storage | BufferUsage.CopyDst,
			});
			this._clusterCapacity = requiredClusterCapacity;
		}

		const requiredIndexCapacity = Math.max(
			1,
			requiredClusterCapacity * Math.max(1, maxLightsPerCluster)
		);
		if (
			!this._clusterIndexBuffer ||
			this._indexCapacity < requiredIndexCapacity
		) {
			this._clusterIndexBuffer?.destroy();
			this._clusterIndexBuffer = this._compute.createBuffer({
				label: "WebGPUClusteredIndices",
				size: requiredIndexCapacity * 4,
				usage: BufferUsage.Storage | BufferUsage.CopyDst,
			});
			this._indexCapacity = requiredIndexCapacity;
		}

		this._sceneBindingSources = null;
		this._computeBindingSources = null;
	}

	private async _ensureComputeResources(): Promise<void> {
		if (!this._computeShaderModule) {
			const shader = await loadClusteredLightingCullShaderComposite();
			this._computeShaderModule = await this._compute.createShaderModule({
				label: "WebGPUClusteredLightingCullShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "clustered",
			});
		}

		if (!this._computePipeline) {
			this._computeGroupLayout0 = this._compute.createBindGroupLayout({
				label: "WebGPUClusteredLighting_Group0",
				entries: [
					{
						binding: 0,
						visibility: GPUShaderStage.COMPUTE,
						buffer: { type: "uniform" },
					},
					{
						binding: 1,
						visibility: GPUShaderStage.COMPUTE,
						buffer: { type: "read-only-storage" },
					},
					{
						binding: 2,
						visibility: GPUShaderStage.COMPUTE,
						buffer: { type: "storage" },
					},
					{
						binding: 3,
						visibility: GPUShaderStage.COMPUTE,
						buffer: { type: "storage" },
					},
				],
			});
			this._computePipelineLayout = this._compute.createPipelineLayout({
				label: "WebGPUClusteredLighting_PipelineLayout",
				bindGroupLayouts: [this._computeGroupLayout0, this._frameLayout],
			});
			this._computePipeline = this._compute.createComputePipeline({
				label: "WebGPUClusteredLightingPipeline",
				layout: this._computePipelineLayout,
				compute: {
					module: this._computeShaderModule,
					entryPoint: "csMain",
				},
			});
		}

		this._ensureComputeBinding();
	}

	private _ensureComputeBinding(): void {
		if (
			!this._clusterParamsBuffer ||
			!this._clusterLightBuffer ||
			!this._clusterHeaderBuffer ||
			!this._clusterIndexBuffer ||
			!this._computeGroupLayout0
		) {
			return;
		}
		const sources: [
			IRenderBuffer,
			IRenderBuffer,
			IRenderBuffer,
			IRenderBuffer,
		] = [
			this._clusterParamsBuffer,
			this._clusterLightBuffer,
			this._clusterHeaderBuffer,
			this._clusterIndexBuffer,
		];

		if (
			this._computeBinding &&
			this._computeBindingSources &&
			this._computeBindingSources[0] === sources[0] &&
			this._computeBindingSources[1] === sources[1] &&
			this._computeBindingSources[2] === sources[2] &&
			this._computeBindingSources[3] === sources[3]
		) {
			return;
		}

		this._destroyBindingGroup(this._computeBinding);
		this._computeBinding = this._compute.createBindingGroup({
			label: "WebGPUClusteredLightingComputeBinding",
			layout: this._computeGroupLayout0,
			entries: [
				{ binding: 0, resource: sources[0] },
				{ binding: 1, resource: sources[1] },
				{ binding: 2, resource: sources[2] },
				{ binding: 3, resource: sources[3] },
			],
		});
		this._computeBindingSources = sources;
	}

	private _createClusterParamsBufferData(
		params: WebGPUClusterGridParams
	): ArrayBuffer {
		const buffer = new ArrayBuffer(CLUSTERED_PARAMS_FLOATS * 4);
		const u32 = new Uint32Array(buffer);
		const f32 = new Float32Array(buffer);
		u32[0] = params.screenWidth >>> 0;
		u32[1] = params.screenHeight >>> 0;
		u32[2] = params.tilesX >>> 0;
		u32[3] = params.tilesY >>> 0;
		u32[4] = params.zSlices >>> 0;
		u32[5] = params.clusterCount >>> 0;
		f32[6] = params.near;
		f32[7] = params.far;
		f32[8] = params.logScale;
		f32[9] = params.logBias;
		u32[10] = 0;
		u32[11] = 0;
		return buffer;
	}

	private _createClusterLightBufferData(
		sourceLights: WebGPULightingState["clusteredLights"],
		count: number
	): ArrayBuffer {
		const safeCount = Math.max(1, count);
		const buffer = new ArrayBuffer(
			safeCount * CLUSTERED_LIGHT_STRIDE_FLOATS * 4
		);
		// Use typed array views for bulk writes - much faster than
		// per-field DataView.setFloat32() calls (16 per light).
		const f32 = new Float32Array(buffer);
		const u32 = new Uint32Array(buffer);
		for (let i = 0; i < count; i++) {
			const light = sourceLights[i];
			const base = i * CLUSTERED_LIGHT_STRIDE_FLOATS;
			f32[base + 0] = light.position[0];
			f32[base + 1] = light.position[1];
			f32[base + 2] = light.position[2];
			f32[base + 3] = Math.max(light.range, 0.001);
			f32[base + 4] = light.direction[0];
			f32[base + 5] = light.direction[1];
			f32[base + 6] = light.direction[2];
			f32[base + 7] = light.outerCos;
			f32[base + 8] = light.color[0];
			f32[base + 9] = light.color[1];
			f32[base + 10] = light.color[2];
			f32[base + 11] = light.innerCos;

			const lightType =
				light.type === WEBGPU_CLUSTERED_LIGHT_TYPE_SPOT ?
					WEBGPU_CLUSTERED_LIGHT_TYPE_SPOT
				:	WEBGPU_CLUSTERED_LIGHT_TYPE_POINT;
			let packedFlags = lightType & WEBGPU_CLUSTERED_LIGHT_FLAG_TYPE_MASK;
			if (light.castsShadow) {
				packedFlags |= WEBGPU_CLUSTERED_LIGHT_FLAG_CASTS_SHADOW;
			}
			if (light.affectsVolumetric) {
				packedFlags |= WEBGPU_CLUSTERED_LIGHT_FLAG_AFFECTS_VOLUMETRIC;
			}
			u32[base + 12] = packedFlags >>> 0;
			u32[base + 13] = Math.max(0, light.shadowIndex | 0);
			u32[base + 14] = 0;
			u32[base + 15] = 0;
		}
		if (count < safeCount) {
			const base = count * CLUSTERED_LIGHT_STRIDE_FLOATS;
			u32[base + 12] = 0xffffffff;
		}
		return buffer;
	}

	private _destroyBindingGroup(group: IBindingGroup | null): void {
		destroyResource(group);
	}
}

export function packClusteredIndexRef(
	lightIndex: number,
	lightType: number,
	shadowed: boolean,
	volumetric: boolean
): number {
	let value =
		(Math.max(0, lightIndex) & WEBGPU_CLUSTERED_INDEX_LIGHT_MASK) |
		((Math.max(0, lightType) & WEBGPU_CLUSTERED_LIGHT_FLAG_TYPE_MASK) <<
			WEBGPU_CLUSTERED_INDEX_TYPE_SHIFT);
	if (shadowed) {
		value |= WEBGPU_CLUSTERED_INDEX_SHADOW_BIT;
	}
	if (volumetric) {
		value |= WEBGPU_CLUSTERED_INDEX_VOLUMETRIC_BIT;
	}
	return value >>> 0;
}

export function unpackClusteredIndexRef(value: number): {
	lightIndex: number;
	lightType: number;
	shadowed: boolean;
	volumetric: boolean;
} {
	const safeValue = value >>> 0;
	return {
		lightIndex: safeValue & WEBGPU_CLUSTERED_INDEX_LIGHT_MASK,
		lightType:
			(safeValue & WEBGPU_CLUSTERED_INDEX_TYPE_MASK) >>>
			WEBGPU_CLUSTERED_INDEX_TYPE_SHIFT,
		shadowed: (safeValue & WEBGPU_CLUSTERED_INDEX_SHADOW_BIT) !== 0,
		volumetric: (safeValue & WEBGPU_CLUSTERED_INDEX_VOLUMETRIC_BIT) !== 0,
	};
}

export function packClusterHeaderFlags(
	overflow: boolean,
	hasShadowed: boolean,
	hasVolumetric: boolean
): number {
	let flags = 0;
	if (overflow) {
		flags |= WEBGPU_CLUSTERED_HEADER_FLAG_OVERFLOW;
	}
	if (hasShadowed) {
		flags |= WEBGPU_CLUSTERED_HEADER_FLAG_HAS_SHADOWED;
	}
	if (hasVolumetric) {
		flags |= WEBGPU_CLUSTERED_HEADER_FLAG_HAS_VOLUMETRIC;
	}
	return flags >>> 0;
}
