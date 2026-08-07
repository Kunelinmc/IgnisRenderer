import { CameraType } from "../../cameras/Camera";
import type { PreparedScene } from "../../pipeline/types";
import { ShaderSource } from "../../shaders/ShaderSource";
import type { ICommandEncoder } from "../ICommandEncoder";
import {
	BufferUsage,
	type IBindingGroup,
	type IComputePipeline,
	type IRenderBuffer,
} from "../types";
import { WEBGPU_CLUSTER_GRID_PARAMS_LAYOUT } from "./bufferLayouts";
import type { IWebGPUComputeFacade } from "./ComputeFacade";
import {
	WEBGPU_CLUSTERED_AREA_STRIDE_FLOATS,
	WEBGPU_CLUSTERED_CULL_STRIDE_FLOATS,
	WEBGPU_CLUSTERED_HEADER_FLAG_HAS_SHADOWED,
	WEBGPU_CLUSTERED_HEADER_FLAG_HAS_VOLUMETRIC,
	WEBGPU_CLUSTERED_HEADER_FLAG_OVERFLOW,
	WEBGPU_CLUSTERED_HEADER_STRIDE_UINTS,
	WEBGPU_CLUSTERED_INDEX_LIGHT_MASK,
	WEBGPU_CLUSTERED_INDEX_SHADOW_BIT,
	WEBGPU_CLUSTERED_INDEX_TYPE_MASK,
	WEBGPU_CLUSTERED_INDEX_TYPE_SHIFT,
	WEBGPU_CLUSTERED_INDEX_VOLUMETRIC_BIT,
	WEBGPU_CLUSTERED_LIGHT_FLAG_AFFECTS_VOLUMETRIC,
	WEBGPU_CLUSTERED_LIGHT_FLAG_CASTS_SHADOW,
	WEBGPU_CLUSTERED_LIGHT_FLAG_TYPE_MASK,
	WEBGPU_CLUSTERED_LIGHT_TYPE_AREA,
	WEBGPU_CLUSTERED_LIGHT_TYPE_POINT,
	WEBGPU_CLUSTERED_LIGHT_TYPE_SPOT,
	WEBGPU_CLUSTERED_MAX_LIGHTS,
	WEBGPU_CLUSTERED_MAX_LIGHTS_PER_CLUSTER,
	WEBGPU_CLUSTERED_METADATA_STRIDE_UINTS,
	WEBGPU_CLUSTERED_PARAMS_FLOATS,
	WEBGPU_CLUSTERED_SHADER_WORKGROUP_SIZE,
	WEBGPU_CLUSTERED_VEC4_STRIDE_FLOATS,
} from "./constants";
import type {
	WebGPUClusteredCullingMode,
	WebGPUClusteredLightingData,
	WebGPUClusterGridParams,
	WebGPUFeatureState,
	WebGPUClusteredLightUniform,
} from "./types";

export interface ComputePassBinding {
	index: number;
	group: IBindingGroup;
}

export interface ComputePassDispatch {
	x: number;
	y: number;
	z: number;
}

const CLUSTERED_PARAMS_DEFAULTS = {
	tileSizePx: 64,
	zSlices: 24,
	maxLights: 256,
	maxLightsPerCluster: 64,
	cullingMode: "gather" as WebGPUClusteredCullingMode,
} as const;

const HASH_OFFSET = 0x811c9dc5;
const HASH_PRIME = 0x01000193;
const HASH_FLOAT_BUFFER = new ArrayBuffer(4);
const HASH_FLOAT_VIEW = new Float32Array(HASH_FLOAT_BUFFER);
const HASH_UINT_VIEW = new Uint32Array(HASH_FLOAT_BUFFER);

interface FrameClusterState {
	enabled: boolean;
	maxLights: number;
	maxLightsPerCluster: number;
	clusterCount: number;
	tilesX: number;
	tilesY: number;
	zSlices: number;
	cullingMode: WebGPUClusteredCullingMode;
}

interface ClusterBuffers {
	params: IRenderBuffer | null;
	positionRange: IRenderBuffer | null;
	directionOuter: IRenderBuffer | null;
	colorInner: IRenderBuffer | null;
	areaPayload: IRenderBuffer | null;
	metadata: IRenderBuffer | null;
	cullData: IRenderBuffer | null;
	headers: IRenderBuffer | null;
	indices: IRenderBuffer | null;
	sliceDepths: IRenderBuffer | null;
}

interface FrameSignatures {
	grid: number;
	cullView: number;
	position: number;
	direction: number;
	color: number;
	area: number;
	metadata: number;
	cullGeometry: number;
	cullScore: number;
}

function recordComputePass(
	encoder: ICommandEncoder,
	label: string,
	pipeline: IComputePipeline,
	bindings: ComputePassBinding[],
	dispatch: ComputePassDispatch
): void {
	const x = assertPositiveInteger(dispatch.x, "dispatch.x");
	const y = assertPositiveInteger(dispatch.y, "dispatch.y");
	const z = assertPositiveInteger(dispatch.z, "dispatch.z");
	encoder.beginComputePass({ label });
	encoder.setComputePipeline(pipeline);
	for (const binding of bindings) {
		encoder.setBindingGroup(binding.index, binding.group);
	}
	encoder.dispatchWorkgroups(x, y, z);
	encoder.endComputePass();
}

function assertPositiveInteger(value: number, name: string): number {
	if (!Number.isInteger(value) || value <= 0) {
		throw new Error(
			`recordComputePass() requires ${name} to be a positive integer, received ${value}.`
		);
	}
	return value;
}

function hashUint(hash: number, value: number): number {
	return Math.imul((hash ^ (value >>> 0)) >>> 0, HASH_PRIME) >>> 0;
}

function hashFloat(hash: number, value: number): number {
	HASH_FLOAT_VIEW[0] = Number.isFinite(value) ? value : 0;
	return hashUint(hash, HASH_UINT_VIEW[0]);
}

function hashString(hash: number, value: string): number {
	let next = hash;
	for (let i = 0; i < value.length; i++) {
		next = hashUint(next, value.charCodeAt(i));
	}
	return next;
}

function growCapacity(current: number, required: number): number {
	let capacity = Math.max(1, current);
	while (capacity < required) {
		capacity *= 2;
	}
	return capacity;
}

function finiteInteger(value: unknown, fallback: number): number {
	return typeof value === "number" && Number.isFinite(value) ?
		Math.floor(value)
	: fallback;
}

function resolveCullingMode(value: unknown): WebGPUClusteredCullingMode {
	return value === "scatter" ? "scatter" : "gather";
}

export class WebGPUClusteredLightingRuntime {
	private _compute: IWebGPUComputeFacade;
	private _sceneLayout: GPUBindGroupLayout;
	private _frameLayout: GPUBindGroupLayout;
	private _warn: (key: string, message: string) => void = () => {};
	private _warningKeys = new Set<string>();
	private _state: FrameClusterState = {
		enabled: false,
		maxLights: 0,
		maxLightsPerCluster: CLUSTERED_PARAMS_DEFAULTS.maxLightsPerCluster,
		clusterCount: 1,
		tilesX: 1,
		tilesY: 1,
		zSlices: 1,
		cullingMode: CLUSTERED_PARAMS_DEFAULTS.cullingMode,
	};
	private _buffers: ClusterBuffers = {
		params: null,
		positionRange: null,
		directionOuter: null,
		colorInner: null,
		areaPayload: null,
		metadata: null,
		cullData: null,
		headers: null,
		indices: null,
		sliceDepths: null,
	};
	private _lightCapacity = 0;
	private _clusterCapacity = 0;
	private _indexCapacity = 0;
	private _sliceCapacity = 0;
	private _positionData = new Float32Array(0);
	private _directionData = new Float32Array(0);
	private _colorData = new Float32Array(0);
	private _areaData = new Float32Array(0);
	private _metadataData = new Uint32Array(0);
	private _cullData = new Float32Array(0);
	private _sliceDepthData = new Float32Array(0);
	private _signatures: FrameSignatures | null = null;
	private _needsCullBuild = false;
	private _sceneBinding: IBindingGroup | null = null;
	private _sceneBindingSources: IRenderBuffer[] | null = null;
	private _computeBinding: IBindingGroup | null = null;
	private _computeBindingSources: IRenderBuffer[] | null = null;
	private _computeShaderModule: any = null;
	private _clearPipeline: IComputePipeline | null = null;
	private _scatterPipeline: IComputePipeline | null = null;
	private _finalizePipeline: IComputePipeline | null = null;
	private _gatherPipeline: IComputePipeline | null = null;
	private _overflowPipeline: IComputePipeline | null = null;
	private _computeGroupLayout0: GPUBindGroupLayout | null = null;
	private _computePipelineLayout: GPUPipelineLayout | null = null;
	private _built = false;

	constructor(
		computeFacade: IWebGPUComputeFacade,
		sceneLayout: GPUBindGroupLayout,
		frameLayout: GPUBindGroupLayout
	) {
		this._compute = computeFacade;
		this._sceneLayout = sceneLayout;
		this._frameLayout = frameLayout;
	}

	/**
	 * Installs the clustered-lighting warning sink.
	 *
	 * @internal Owned by the WebGPU frame-service warning bridge. Renderer
	 * consumers should use the configured logger instead.
	 */
	public onWarn(warn: (key: string, message: string) => void): void {
		this._warn = warn;
	}

	public onShaderRuntimeChanged(): void {
		this._computeShaderModule = null;
		this._clearPipeline = null;
		this._scatterPipeline = null;
		this._finalizePipeline = null;
		this._gatherPipeline = null;
		this._overflowPipeline = null;
		this._computeGroupLayout0 = null;
		this._computePipelineLayout = null;
		this._destroyBindingGroup(this._computeBinding);
		this._computeBinding = null;
		this._computeBindingSources = null;
	}

	public prepareFrame(
		frame: PreparedScene,
		features: WebGPUFeatureState,
		lighting: WebGPUClusteredLightingData | null,
		renderWidth: number,
		renderHeight: number
	): void {
		this._built = false;
		const requested = features.clusteredLightingOptions ?? {};
		const requestedMaxLights = Math.max(
			1,
			finiteInteger(requested.maxLights, CLUSTERED_PARAMS_DEFAULTS.maxLights)
		);
		const requestedMaxPerCluster = Math.max(
			1,
			finiteInteger(
				requested.maxLightsPerCluster,
				CLUSTERED_PARAMS_DEFAULTS.maxLightsPerCluster
			)
		);
		if (requestedMaxLights > WEBGPU_CLUSTERED_MAX_LIGHTS) {
			this._warnOnce(
				"webgpu-clustered-max-lights-limit",
				`WebGPU clustered lighting clamps maxLights to ${WEBGPU_CLUSTERED_MAX_LIGHTS}.`
			);
		}
		if (requestedMaxPerCluster > WEBGPU_CLUSTERED_MAX_LIGHTS_PER_CLUSTER) {
			this._warnOnce(
				"webgpu-clustered-max-per-cluster-limit",
				`WebGPU clustered lighting clamps maxLightsPerCluster to ${WEBGPU_CLUSTERED_MAX_LIGHTS_PER_CLUSTER}.`
			);
		}
		const maxLightBudget = Math.min(
			requestedMaxLights,
			WEBGPU_CLUSTERED_MAX_LIGHTS
		);
		const maxLightsPerCluster = Math.min(
			requestedMaxPerCluster,
			WEBGPU_CLUSTERED_MAX_LIGHTS_PER_CLUSTER
		);
		const cullingMode = resolveCullingMode(requested.cullingMode);
		const renderW = Math.max(1, Math.floor(renderWidth));
		const renderH = Math.max(1, Math.floor(renderHeight));
		const tileSizePx = Math.max(
			8,
			finiteInteger(requested.tileSizePx, CLUSTERED_PARAMS_DEFAULTS.tileSizePx)
		);
		const tilesX = Math.max(1, Math.ceil(renderW / tileSizePx));
		const tilesY = Math.max(1, Math.ceil(renderH / tileSizePx));
		const zSlices = Math.max(
			1,
			finiteInteger(requested.zSlices, CLUSTERED_PARAMS_DEFAULTS.zSlices)
		);
		const clusterCount = Math.max(1, tilesX * tilesY * zSlices);
		const isPerspective = frame.camera.type === CameraType.Perspective;
		if (features.enableClusteredLighting && !isPerspective) {
			this._warnOnce(
				"webgpu-clustered-perspective-only",
				"WebGPU clustered lighting only supports perspective cameras; falling back to legacy forward lights",
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
		const sourceLights = lighting?.lights ?? [];
		const maxLights = canCluster ? Math.min(sourceLights.length, maxLightBudget) : 0;
		if (
			sourceLights.length > maxLightBudget &&
			!(lighting?.warnings ?? []).some(
				(warning) => warning.key === "webgpu-clustered-light-budget"
			)
		) {
			this._warnOnce(
				"webgpu-clustered-light-budget",
				`WebGPU clustered lighting clamps lights to ${maxLightBudget}; extra lights are skipped`,
			);
		}
		const runtimeEnabled = canCluster && maxLights > 0;
		const activeClusterCount = runtimeEnabled ? clusterCount : 1;
		const activeMaxPerCluster = runtimeEnabled ? maxLightsPerCluster : 1;
		const logScale = runtimeEnabled ? zSlices / logDenom : 0;
		const logBias = runtimeEnabled ? (-Math.log(near) * zSlices) / logDenom : 0;
		this._state = {
			enabled: runtimeEnabled,
			maxLights,
			maxLightsPerCluster: activeMaxPerCluster,
			clusterCount: activeClusterCount,
			tilesX,
			tilesY,
			zSlices,
			cullingMode,
		};
		const buffersChanged = this._ensureBuffers(
			maxLights,
			activeClusterCount,
			activeMaxPerCluster,
			zSlices + 1
		);
		const params: WebGPUClusterGridParams = {
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
			lightCount: maxLights,
			maxLightsPerCluster: activeMaxPerCluster,
		};
		const signatures = this._packFrameData(
			frame,
			sourceLights,
			maxLights,
			params,
			cullingMode
		);
		const previous = this._signatures;
		if (!previous || previous.grid !== signatures.grid || buffersChanged) {
			this._compute.writeBuffer(
				this._buffers.params!,
				this._createClusterParamsBufferData(params)
			);
			this._compute.writeBuffer(
				this._buffers.sliceDepths!,
				this._sliceDepthData.subarray(0, zSlices + 1)
			);
		}
		this._writeSoABuffer(
			previous?.position,
			signatures.position,
			this._buffers.positionRange!,
			this._positionData,
			maxLights * WEBGPU_CLUSTERED_VEC4_STRIDE_FLOATS,
			buffersChanged
		);
		this._writeSoABuffer(
			previous?.direction,
			signatures.direction,
			this._buffers.directionOuter!,
			this._directionData,
			maxLights * WEBGPU_CLUSTERED_VEC4_STRIDE_FLOATS,
			buffersChanged
		);
		this._writeSoABuffer(
			previous?.color,
			signatures.color,
			this._buffers.colorInner!,
			this._colorData,
			maxLights * WEBGPU_CLUSTERED_VEC4_STRIDE_FLOATS,
			buffersChanged
		);
		this._writeSoABuffer(
			previous?.area,
			signatures.area,
			this._buffers.areaPayload!,
			this._areaData,
			maxLights * WEBGPU_CLUSTERED_AREA_STRIDE_FLOATS,
			buffersChanged
		);
		this._writeSoABuffer(
			previous?.metadata,
			signatures.metadata,
			this._buffers.metadata!,
			this._metadataData,
			maxLights * WEBGPU_CLUSTERED_METADATA_STRIDE_UINTS,
			buffersChanged
		);
		const cullChanged =
			!previous ||
			previous.cullGeometry !== signatures.cullGeometry ||
			(maxLights > activeMaxPerCluster &&
				previous.cullScore !== signatures.cullScore);
		this._writeSoABuffer(
			cullChanged ? undefined : signatures.cullGeometry,
			signatures.cullGeometry,
			this._buffers.cullData!,
			this._cullData,
			maxLights * WEBGPU_CLUSTERED_CULL_STRIDE_FLOATS,
			buffersChanged
		);
		this._needsCullBuild = runtimeEnabled && (
			buffersChanged ||
			!previous ||
			previous.grid !== signatures.grid ||
			previous.cullView !== signatures.cullView ||
			cullChanged
		);
		this._signatures = signatures;
	}

	public async build(
		encoder: ICommandEncoder,
		frameBinding: IBindingGroup
	): Promise<void> {
		if (this._built || !this._state.enabled || this._state.maxLights <= 0) {
			return;
		}
		if (!this._needsCullBuild) {
			this._built = true;
			return;
		}
		await this._ensureComputeResources();
		if (!this._computeBinding) {
			return;
		}
		const bindings = [
			{ index: 0, group: this._computeBinding },
			{ index: 1, group: frameBinding },
		];
		if (this._state.cullingMode === "gather") {
			if (!this._gatherPipeline || !this._overflowPipeline) {
				return;
			}
			const clusterDispatch = {
				x: this._state.tilesX,
				y: this._state.tilesY,
				z: this._state.zSlices,
			};
			recordComputePass(
				encoder,
				"WebGPUClusteredLightingGather",
				this._gatherPipeline,
				bindings,
				clusterDispatch
			);
			if (this._state.maxLights > this._state.maxLightsPerCluster) {
				recordComputePass(
					encoder,
					"WebGPUClusteredLightingOverflowResolve",
					this._overflowPipeline,
					bindings,
					clusterDispatch
				);
			}
		} else {
			if (!this._clearPipeline || !this._scatterPipeline || !this._finalizePipeline) {
				return;
			}
			const clusterDispatch = {
				x: Math.ceil(
					this._state.clusterCount / WEBGPU_CLUSTERED_SHADER_WORKGROUP_SIZE
				),
				y: 1,
				z: 1,
			};
			recordComputePass(
				encoder,
				"WebGPUClusteredLightingClear",
				this._clearPipeline,
				bindings,
				clusterDispatch
			);
			recordComputePass(
				encoder,
				"WebGPUClusteredLightingScatter",
				this._scatterPipeline,
				bindings,
				{
					x: Math.ceil(
						this._state.maxLights / WEBGPU_CLUSTERED_SHADER_WORKGROUP_SIZE
					),
					y: 1,
					z: 1,
				}
			);
			recordComputePass(
				encoder,
				"WebGPUClusteredLightingFinalize",
				this._finalizePipeline,
				bindings,
				clusterDispatch
			);
		}
		this._built = true;
		this._needsCullBuild = false;
	}

	public getSceneBinding(): IBindingGroup {
		if (!this._buffers.params) {
			this._ensureBuffers(0, 1, 1, 2);
		}
		const sources = this._getSceneBindingSources();
		if (!this._sameSources(this._sceneBindingSources, sources)) {
			this._destroyBindingGroup(this._sceneBinding);
			this._sceneBinding = this._compute.createBindingGroup({
				label: "WebGPUClusteredSceneBinding",
				layout: this._sceneLayout,
				entries: sources.map((resource, binding) => ({ binding, resource })),
			});
			this._sceneBindingSources = sources;
		}
		return this._sceneBinding!;
	}

	public destroy(): void {
		this._destroyBindingGroup(this._sceneBinding);
		this._destroyBindingGroup(this._computeBinding);
		this._sceneBinding = null;
		this._computeBinding = null;
		this._sceneBindingSources = null;
		this._computeBindingSources = null;
		for (const buffer of Object.values(this._buffers)) {
			buffer?.destroy();
		}
		for (const key of Object.keys(this._buffers) as Array<keyof ClusterBuffers>) {
			this._buffers[key] = null;
		}
		this._lightCapacity = 0;
		this._clusterCapacity = 0;
		this._indexCapacity = 0;
		this._sliceCapacity = 0;
		this._signatures = null;
		this.onShaderRuntimeChanged();
	}

	private _ensureBuffers(
		maxLights: number,
		clusterCount: number,
		maxLightsPerCluster: number,
		sliceCount: number
	): boolean {
		let changed = false;
		if (!this._buffers.params) {
			this._buffers.params = this._compute.createBuffer({
				label: "WebGPUClusteredParams",
				size: WEBGPU_CLUSTERED_PARAMS_FLOATS * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
			changed = true;
		}
		const requiredLights = Math.max(1, maxLights);
		if (this._lightCapacity < requiredLights) {
			const capacity = growCapacity(this._lightCapacity, requiredLights);
			this._replaceBuffer(
				"positionRange",
				"WebGPUClusteredPositionRange",
				capacity * WEBGPU_CLUSTERED_VEC4_STRIDE_FLOATS * 4
			);
			this._replaceBuffer(
				"directionOuter",
				"WebGPUClusteredDirectionOuter",
				capacity * WEBGPU_CLUSTERED_VEC4_STRIDE_FLOATS * 4
			);
			this._replaceBuffer(
				"colorInner",
				"WebGPUClusteredColorInner",
				capacity * WEBGPU_CLUSTERED_VEC4_STRIDE_FLOATS * 4
			);
			this._replaceBuffer(
				"areaPayload",
				"WebGPUClusteredAreaPayload",
				capacity * WEBGPU_CLUSTERED_AREA_STRIDE_FLOATS * 4
			);
			this._replaceBuffer(
				"metadata",
				"WebGPUClusteredMetadata",
				capacity * WEBGPU_CLUSTERED_METADATA_STRIDE_UINTS * 4
			);
			this._replaceBuffer(
				"cullData",
				"WebGPUClusteredCullData",
				capacity * WEBGPU_CLUSTERED_CULL_STRIDE_FLOATS * 4
			);
			this._positionData = new Float32Array(
				capacity * WEBGPU_CLUSTERED_VEC4_STRIDE_FLOATS
			);
			this._directionData = new Float32Array(
				capacity * WEBGPU_CLUSTERED_VEC4_STRIDE_FLOATS
			);
			this._colorData = new Float32Array(
				capacity * WEBGPU_CLUSTERED_VEC4_STRIDE_FLOATS
			);
			this._areaData = new Float32Array(
				capacity * WEBGPU_CLUSTERED_AREA_STRIDE_FLOATS
			);
			this._metadataData = new Uint32Array(
				capacity * WEBGPU_CLUSTERED_METADATA_STRIDE_UINTS
			);
			this._cullData = new Float32Array(
				capacity * WEBGPU_CLUSTERED_CULL_STRIDE_FLOATS
			);
			this._lightCapacity = capacity;
			changed = true;
		}
		const requiredClusters = Math.max(1, clusterCount);
		if (this._clusterCapacity < requiredClusters) {
			const capacity = growCapacity(this._clusterCapacity, requiredClusters);
			this._replaceBuffer(
				"headers",
				"WebGPUClusteredHeaders",
				capacity * WEBGPU_CLUSTERED_HEADER_STRIDE_UINTS * 4
			);
			this._clusterCapacity = capacity;
			changed = true;
		}
		const requiredIndices = Math.max(1, requiredClusters * maxLightsPerCluster);
		if (this._indexCapacity < requiredIndices) {
			const capacity = growCapacity(this._indexCapacity, requiredIndices);
			this._replaceBuffer(
				"indices",
				"WebGPUClusteredIndices",
				capacity * 4
			);
			this._indexCapacity = capacity;
			changed = true;
		}
		const requiredSlices = Math.max(2, sliceCount);
		if (this._sliceCapacity < requiredSlices) {
			const capacity = growCapacity(this._sliceCapacity, requiredSlices);
			this._replaceBuffer(
				"sliceDepths",
				"WebGPUClusteredSliceDepths",
				capacity * 4
			);
			this._sliceDepthData = new Float32Array(capacity);
			this._sliceCapacity = capacity;
			changed = true;
		}
		if (changed) {
			this._sceneBindingSources = null;
			this._computeBindingSources = null;
		}
		return changed;
	}

	private _replaceBuffer(
		key: keyof ClusterBuffers,
		label: string,
		size: number
	): void {
		this._buffers[key]?.destroy();
		this._buffers[key] = this._compute.createBuffer({
			label,
			size,
			usage: BufferUsage.Storage | BufferUsage.CopyDst,
		});
	}

	private _packFrameData(
		frame: PreparedScene,
		lights: readonly WebGPUClusteredLightUniform[],
		count: number,
		params: WebGPUClusterGridParams,
		mode: WebGPUClusteredCullingMode
	): FrameSignatures {
		let positionHash = HASH_OFFSET;
		let directionHash = HASH_OFFSET;
		let colorHash = HASH_OFFSET;
		let areaHash = HASH_OFFSET;
		let metadataHash = HASH_OFFSET;
		let cullGeometryHash = HASH_OFFSET;
		let cullScoreHash = HASH_OFFSET;
		for (let i = 0; i < count; i++) {
			const light = lights[i];
			const range = Math.max(light.range, 0.001);
			const right = light.right ?? [0, 0, 0];
			const up = light.up ?? [0, 0, 0];
			const normal = light.normal ?? [0, 1, 0];
			const width = Math.max(0, light.width ?? 0);
			const height = Math.max(0, light.height ?? 0);
			const areaScale = Math.max(0, light.areaScale ?? 0);
			const lightType =
				light.type === WEBGPU_CLUSTERED_LIGHT_TYPE_AREA ?
					WEBGPU_CLUSTERED_LIGHT_TYPE_AREA
				: light.type === WEBGPU_CLUSTERED_LIGHT_TYPE_SPOT ?
					WEBGPU_CLUSTERED_LIGHT_TYPE_SPOT
				:	WEBGPU_CLUSTERED_LIGHT_TYPE_POINT;
			let packedFlags = lightType & WEBGPU_CLUSTERED_LIGHT_FLAG_TYPE_MASK;
			if (light.castsShadow) packedFlags |= WEBGPU_CLUSTERED_LIGHT_FLAG_CASTS_SHADOW;
			if (light.affectsVolumetric) {
				packedFlags |= WEBGPU_CLUSTERED_LIGHT_FLAG_AFFECTS_VOLUMETRIC;
			}
			const luminance = Math.max(
				0,
				light.color[0] * 0.2126 +
					light.color[1] * 0.7152 +
					light.color[2] * 0.0722
			);
			const halfWidth = width * 0.5;
			const halfHeight = height * 0.5;
			const cullRadius =
				lightType === WEBGPU_CLUSTERED_LIGHT_TYPE_AREA ?
					range + Math.hypot(halfWidth, halfHeight)
				:	range;
			const vecBase = i * WEBGPU_CLUSTERED_VEC4_STRIDE_FLOATS;
			this._positionData.set(
				[light.position[0], light.position[1], light.position[2], range],
				vecBase
			);
			this._directionData.set(
				[
					light.direction[0],
					light.direction[1],
					light.direction[2],
					light.outerCos,
				],
				vecBase
			);
			this._colorData.set(
				[light.color[0], light.color[1], light.color[2], light.innerCos],
				vecBase
			);
			const areaBase = i * WEBGPU_CLUSTERED_AREA_STRIDE_FLOATS;
			this._areaData.set([right[0], right[1], right[2], width], areaBase);
			this._areaData.set([up[0], up[1], up[2], height], areaBase + 4);
			this._areaData.set(
				[normal[0], normal[1], normal[2], areaScale],
				areaBase + 8
			);
			const metadataBase = i * WEBGPU_CLUSTERED_METADATA_STRIDE_UINTS;
			this._metadataData[metadataBase] = packedFlags >>> 0;
			this._metadataData[metadataBase + 1] = Math.max(0, light.shadowIndex | 0);
			const cullBase = i * WEBGPU_CLUSTERED_CULL_STRIDE_FLOATS;
			this._cullData.set(
				[light.position[0], light.position[1], light.position[2], cullRadius],
				cullBase
			);
			this._cullData.set(
				[range, luminance, areaScale, light.innerCos],
				cullBase + 4
			);
			for (let component = 0; component < 4; component++) {
				positionHash = hashFloat(positionHash, this._positionData[vecBase + component]);
				directionHash = hashFloat(directionHash, this._directionData[vecBase + component]);
				colorHash = hashFloat(colorHash, this._colorData[vecBase + component]);
			}
			for (let component = 0; component < 12; component++) {
				areaHash = hashFloat(areaHash, this._areaData[areaBase + component]);
			}
			metadataHash = hashUint(metadataHash, packedFlags);
			metadataHash = hashUint(metadataHash, this._metadataData[metadataBase + 1]);
			for (let component = 0; component < 4; component++) {
				cullGeometryHash = hashFloat(
					cullGeometryHash,
					this._cullData[cullBase + component]
				);
			}
			cullGeometryHash = hashUint(cullGeometryHash, packedFlags);
			for (let component = 0; component < 3; component++) {
				cullGeometryHash = hashFloat(
					cullGeometryHash,
					this._directionData[vecBase + component]
				);
			}
			cullGeometryHash = hashFloat(cullGeometryHash, light.outerCos);
			cullGeometryHash = hashFloat(cullGeometryHash, light.innerCos);
			for (let component = 0; component < 12; component++) {
				cullGeometryHash = hashFloat(
					cullGeometryHash,
					this._areaData[areaBase + component]
				);
			}
			cullScoreHash = hashFloat(cullScoreHash, luminance);
		}
		let gridHash = HASH_OFFSET;
		for (const value of [
			params.screenWidth,
			params.screenHeight,
			params.tilesX,
			params.tilesY,
			params.zSlices,
			params.clusterCount,
			params.near,
			params.far,
			params.logScale,
			params.logBias,
			params.lightCount,
			params.maxLightsPerCluster,
		]) {
			gridHash = hashFloat(gridHash, value);
		}
		let cullViewHash = hashString(HASH_OFFSET, mode);
		for (const value of [
			frame.camera.fov,
			frame.camera.aspectRatio,
			frame.camera.type === CameraType.Perspective ? 1 : 0,
		]) {
			cullViewHash = hashFloat(cullViewHash, value);
		}
		for (const matrix of [
			frame.camera.viewMatrix,
			frame.camera.projectionMatrix,
			frame.camera.viewProjectionMatrix,
		]) {
			if (!matrix?.elements) continue;
			for (const row of matrix.elements) {
				for (const value of row) {
					cullViewHash = hashFloat(cullViewHash, value);
				}
			}
		}
		for (let slice = 0; slice <= params.zSlices; slice++) {
			const depth =
				params.logScale > 0 ?
					Math.exp((slice - params.logBias) / params.logScale)
				:	params.near;
			this._sliceDepthData[slice] = Math.min(
				params.far,
				Math.max(params.near, depth)
			);
		}
		return {
			grid: gridHash,
			cullView: cullViewHash,
			position: positionHash,
			direction: directionHash,
			color: colorHash,
			area: areaHash,
			metadata: metadataHash,
			cullGeometry: cullGeometryHash,
			cullScore: cullScoreHash,
		};
	}

	private _writeSoABuffer(
		previousSignature: number | undefined,
		nextSignature: number,
		buffer: IRenderBuffer,
		data: Float32Array | Uint32Array,
		length: number,
		force: boolean
	): void {
		if (!force && previousSignature === nextSignature) return;
		const safeLength = Math.max(1, length);
		const source = new Uint8Array(
			data.buffer as ArrayBuffer,
			data.byteOffset,
			safeLength * data.BYTES_PER_ELEMENT
		);
		this._compute.writeBuffer(buffer, source);
	}

	private async _ensureComputeResources(): Promise<void> {
		if (!this._computeShaderModule) {
			const shader = await ShaderSource.load(
				"webgpu.clusteredLightingCull.composite"
			);
			this._computeShaderModule = await this._compute.createShaderModule({
				label: "WebGPUClusteredLightingCullShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "clustered",
			});
		}
		if (!this._computeGroupLayout0) {
			this._computeGroupLayout0 = this._compute.createBindGroupLayout({
				label: "WebGPUClusteredLighting_Group0",
				entries: Array.from({ length: 8 }, (_, binding) => ({
					binding,
					visibility: GPUShaderStage.COMPUTE,
					buffer: {
						type:
							binding === 0 ? "uniform"
							: binding === 5 || binding === 6 ? "storage"
							: "read-only-storage",
					} as GPUBufferBindingLayout,
				})),
			});
			this._computePipelineLayout = this._compute.createPipelineLayout({
				label: "WebGPUClusteredLighting_PipelineLayout",
				bindGroupLayouts: [this._computeGroupLayout0, this._frameLayout],
			});
		}
		if (!this._clearPipeline) {
			[
				this._clearPipeline,
				this._scatterPipeline,
				this._finalizePipeline,
				this._gatherPipeline,
				this._overflowPipeline,
			] = await Promise.all([
				this._createComputePipeline("WebGPUClusteredLightingClearPipeline", "csClear"),
				this._createComputePipeline("WebGPUClusteredLightingScatterPipeline", "csScatter"),
				this._createComputePipeline("WebGPUClusteredLightingFinalizePipeline", "csFinalize"),
				this._createComputePipeline("WebGPUClusteredLightingGatherPipeline", "csGather"),
				this._createComputePipeline(
					"WebGPUClusteredLightingOverflowPipeline",
					"csResolveOverflow"
				),
			]);
		}
		this._ensureComputeBinding();
	}

	private _createComputePipeline(
		label: string,
		entryPoint: string
	): Promise<IComputePipeline> {
		return this._compute.createComputePipeline({
			label,
			layout: this._computePipelineLayout,
			compute: { module: this._computeShaderModule, entryPoint },
		});
	}

	private _ensureComputeBinding(): void {
		if (!this._computeGroupLayout0) return;
		const sources = this._getComputeBindingSources();
		if (this._sameSources(this._computeBindingSources, sources)) return;
		this._destroyBindingGroup(this._computeBinding);
		this._computeBinding = this._compute.createBindingGroup({
			label: "WebGPUClusteredLightingComputeBinding",
			layout: this._computeGroupLayout0,
			entries: sources.map((resource, binding) => ({ binding, resource })),
		});
		this._computeBindingSources = sources;
	}

	private _getSceneBindingSources(): IRenderBuffer[] {
		return [
			this._buffers.params!,
			this._buffers.positionRange!,
			this._buffers.directionOuter!,
			this._buffers.colorInner!,
			this._buffers.areaPayload!,
			this._buffers.metadata!,
			this._buffers.headers!,
			this._buffers.indices!,
		];
	}

	private _getComputeBindingSources(): IRenderBuffer[] {
		return [
			this._buffers.params!,
			this._buffers.cullData!,
			this._buffers.directionOuter!,
			this._buffers.areaPayload!,
			this._buffers.metadata!,
			this._buffers.headers!,
			this._buffers.indices!,
			this._buffers.sliceDepths!,
		];
	}

	private _sameSources(
		left: IRenderBuffer[] | null,
		right: IRenderBuffer[]
	): boolean {
		return !!left &&
			left.length === right.length &&
			left.every((source, index) => source === right[index]);
	}

	private _createClusterParamsBufferData(
		params: WebGPUClusterGridParams
	): ArrayBuffer {
		const writer = WEBGPU_CLUSTER_GRID_PARAMS_LAYOUT.createWriter();
		writer.expectByteLength(WEBGPU_CLUSTERED_PARAMS_FLOATS * 4, "ClusterGridParams");
		writer.writeU32("screenWidth", params.screenWidth >>> 0);
		writer.writeU32("screenHeight", params.screenHeight >>> 0);
		writer.writeU32("tilesX", params.tilesX >>> 0);
		writer.writeU32("tilesY", params.tilesY >>> 0);
		writer.writeU32("zSlices", params.zSlices >>> 0);
		writer.writeU32("clusterCount", params.clusterCount >>> 0);
		writer.writeF32("near", params.near);
		writer.writeF32("far", params.far);
		writer.writeF32("logScale", params.logScale);
		writer.writeF32("logBias", params.logBias);
		writer.writeU32("lightCount", Math.max(0, params.lightCount) >>> 0);
		writer.writeU32(
			"maxLightsPerCluster",
			Math.max(1, params.maxLightsPerCluster) >>> 0
		);
		return writer.toArrayBuffer();
	}

	private _destroyBindingGroup(group: IBindingGroup | null): void {
		const destroyFn = (group as { destroy?: () => void } | null)?.destroy;
		if (typeof destroyFn !== "function") return;
		destroyFn.call(group);
	}

	private _warnOnce(key: string, message: string): void {
		if (this._warningKeys.has(key)) return;
		this._warningKeys.add(key);
		this._warn(key, message);
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
	if (shadowed) value |= WEBGPU_CLUSTERED_INDEX_SHADOW_BIT;
	if (volumetric) value |= WEBGPU_CLUSTERED_INDEX_VOLUMETRIC_BIT;
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
	if (overflow) flags |= WEBGPU_CLUSTERED_HEADER_FLAG_OVERFLOW;
	if (hasShadowed) flags |= WEBGPU_CLUSTERED_HEADER_FLAG_HAS_SHADOWED;
	if (hasVolumetric) flags |= WEBGPU_CLUSTERED_HEADER_FLAG_HAS_VOLUMETRIC;
	return flags >>> 0;
}
