import { Frustum } from "../../maths/Frustum";
import {
	LightType,
	isShadowCastingLight,
	type ShadowCastingLight,
} from "../../lights";
import { Matrix4 } from "../../maths/Matrix4";
import {
	getPrimaryShadowMap,
	type ShadowMap,
	type ShadowRenderSet,
} from "../../lights/shadows/ShadowMapping";
import type {
	DrawPacket,
	FrameContext,
	PreparedScene,
} from "../../pipeline/types";
import {
	ANIMATION_WEBGPU_JOINT_MATRICES_KEY,
	ANIMATION_WEBGPU_MORPH_WEIGHTS_KEY,
} from "../../simulation/animation/types";
import { DEFAULT_PRIMITIVE_DRAW_TOPOLOGY } from "../../core/types";
import { resolveMaterialShadowTransmittance } from "../../materials/transparency";
import type { WebGPUBackend } from "../WebGPUBackend";
import type { ICommandEncoder } from "../ICommandEncoder";
import { createInlineCompositeShaderSource } from "../../shaders/runtime";
import {
	WEBGPU_MAX_DIRECTIONAL_LIGHTS,
	WEBGPU_MAX_MORPH_TARGETS,
	WEBGPU_MAX_SPOT_LIGHTS,
	WEBGPU_SHADOW_ATLAS_COLUMNS,
} from "./constants";
import { createWebGPUShadowVertexBufferLayout } from "./bufferLayouts";
import { getWebGPUShaderModule, getWebGPUTexture } from "./WebGPUResourceAccess";
import { tryGetNativeWebGPUCommandEncoder } from "./WebGPUCommandEncoder";
import { TextureFormat } from "../types";
import type {
	WebGPUGeometryHandle,
	WebGPUGeometryRegistry,
} from "./WebGPUGeometryRegistry";
import type { WebGPUShadowAtlasAllocator } from "./WebGPUShadowAtlasAllocator";
import type { IShaderModule } from "../types";

const WEBGPU_SHADOW_DEPTH_SHADER = /* wgsl */ `
const EPSILON: f32 = 1e-6;

struct AnimationParams {
	jointCount: u32,
	morphTargetCount: u32,
	jointStride: u32,
	morphWeightStride: u32,
}

struct ShadowInstanceData {
	instanceBaseOffset: u32,
	vertexBaseOffset: u32,
	jointBaseOffset: u32,
	morphWeightBaseOffset: u32,
	morphDeltaBaseOffset: u32,
	_pad0: u32,
	_pad1: u32,
	_pad2: u32,
}

struct ShadowVertexInput {
	@location(0) position: vec3<f32>,
	@location(5) joints0: vec4<f32>,
	@location(6) weights0: vec4<f32>,
	@location(7) joints1: vec4<f32>,
	@location(8) weights1: vec4<f32>,
}

struct ShadowVertexOutput {
	@builtin(position) position: vec4<f32>,
	@location(0) transmittance: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> shadowMvps: array<mat4x4<f32>>;
@group(0) @binding(1) var<storage, read> shadowInstances: array<ShadowInstanceData>;
@group(0) @binding(2) var<storage, read> shadowTransmittance: array<vec4<f32>>;
@group(1) @binding(0) var<uniform> animationParams: AnimationParams;
@group(1) @binding(1) var<storage, read> jointMatrices: array<mat4x4<f32>>;
@group(1) @binding(2) var<storage, read> morphWeights: array<f32>;
@group(1) @binding(3) var<storage, read> morphPositionDeltas: array<vec4<f32>>;

fn applyMorphPosition(
	basePosition: vec3<f32>,
	vertexIndex: u32,
	morphTargetCount: u32,
	morphWeightOffset: u32,
	morphDeltaOffset: u32
) -> vec3<f32> {
	if (morphTargetCount == 0u) {
		return basePosition;
	}

	let morphDeltaCount = arrayLength(&morphPositionDeltas);
	let morphWeightCount = arrayLength(&morphWeights);
	if (morphDeltaCount == 0u || morphWeightCount == 0u) {
		return basePosition;
	}

	let deltaBase = min(morphDeltaOffset, morphDeltaCount);
	let deltaRange = morphDeltaCount - deltaBase;
	let vertexCount = max(deltaRange / morphTargetCount, 1u);
	var position = basePosition;
	for (
		var targetIndex: u32 = 0u;
		targetIndex < morphTargetCount;
		targetIndex = targetIndex + 1u
	) {
		let weightIndex = morphWeightOffset + targetIndex;
		if (weightIndex >= morphWeightCount) {
			continue;
		}

		let weight = morphWeights[weightIndex];
		if (abs(weight) <= EPSILON) {
			continue;
		}

		let deltaIndex = deltaBase + targetIndex * vertexCount + vertexIndex;
		if (deltaIndex >= morphDeltaCount) {
			continue;
		}

		position += morphPositionDeltas[deltaIndex].xyz * weight;
	}

	return position;
}

fn applySkinningPosition(
	basePosition: vec3<f32>,
	jointIndices: array<f32, 8>,
	jointWeights: array<f32, 8>,
	jointCount: u32,
	jointOffset: u32
) -> vec3<f32> {
	if (jointCount == 0u) {
		return basePosition;
	}

	let matrixCount = arrayLength(&jointMatrices);
	if (matrixCount == 0u) {
		return basePosition;
	}

	var skinnedPosition = vec3<f32>(0.0);
	var weightSum = 0.0;
	for (var influence: u32 = 0u; influence < 8u; influence = influence + 1u) {
		let weight = jointWeights[influence];
		if (weight <= EPSILON) {
			continue;
		}

		let rawJoint = max(jointIndices[influence], 0.0);
		let jointIndex = u32(rawJoint + 0.5);
		if (jointIndex >= jointCount) {
			continue;
		}

		let matrixIndex = jointOffset + jointIndex;
		if (matrixIndex >= matrixCount) {
			continue;
		}

		let skinMatrix = jointMatrices[matrixIndex];
		skinnedPosition += (skinMatrix * vec4<f32>(basePosition, 1.0)).xyz * weight;
		weightSum += weight;
	}

	if (weightSum <= EPSILON) {
		return basePosition;
	}

	return skinnedPosition / weightSum;
}

@vertex
fn vsMain(
	input: ShadowVertexInput,
	@builtin(vertex_index) vertexIndex: u32,
	@builtin(instance_index) instanceIndex: u32
) -> ShadowVertexOutput {
	var output: ShadowVertexOutput;
	output.position = vec4<f32>(0.0);
	output.transmittance = vec4<f32>(1.0);
	let morphTargetCount = animationParams.morphTargetCount;
	let jointCount = animationParams.jointCount;

	let joints = array<f32, 8>(
		input.joints0.x,
		input.joints0.y,
		input.joints0.z,
		input.joints0.w,
		input.joints1.x,
		input.joints1.y,
		input.joints1.z,
		input.joints1.w
	);
	let weights = array<f32, 8>(
		input.weights0.x,
		input.weights0.y,
		input.weights0.z,
		input.weights0.w,
		input.weights1.x,
		input.weights1.y,
		input.weights1.z,
		input.weights1.w
	);

	let mvpCount = arrayLength(&shadowMvps);
	let instanceDataCount = arrayLength(&shadowInstances);
	if (mvpCount == 0u || instanceDataCount == 0u) {
		return output;
	}
	let safeInstanceIndex = min(instanceIndex, min(mvpCount, instanceDataCount) - 1u);
	let transmittanceCount = arrayLength(&shadowTransmittance);
	if (transmittanceCount > 0u) {
		output.transmittance = shadowTransmittance[min(safeInstanceIndex, transmittanceCount - 1u)];
	}
	let instanceData = shadowInstances[safeInstanceIndex];
	var localInstanceIndex = 0u;
	if (safeInstanceIndex >= instanceData.instanceBaseOffset) {
		localInstanceIndex = safeInstanceIndex - instanceData.instanceBaseOffset;
	}
	let jointOffset =
		instanceData.jointBaseOffset +
		localInstanceIndex * animationParams.jointStride;
	let morphWeightOffset =
		instanceData.morphWeightBaseOffset +
		localInstanceIndex * animationParams.morphWeightStride;
	var localVertexIndex = 0u;
	if (vertexIndex >= instanceData.vertexBaseOffset) {
		localVertexIndex = vertexIndex - instanceData.vertexBaseOffset;
	}

	let morphedPosition = applyMorphPosition(
		input.position,
		localVertexIndex,
		morphTargetCount,
		morphWeightOffset,
		instanceData.morphDeltaBaseOffset
	);
	let skinnedPosition = applySkinningPosition(
		morphedPosition,
		joints,
		weights,
		jointCount,
		jointOffset
	);
	output.position = shadowMvps[safeInstanceIndex] * vec4<f32>(skinnedPosition, 1.0);
	return output;
}

@fragment
fn fsTransmittance(input: ShadowVertexOutput) -> @location(0) vec4<f32> {
	return clamp(input.transmittance, vec4<f32>(0.0), vec4<f32>(1.0));
}
`;

interface ShadowRenderSlot {
	shadowMap: ShadowMap;
	renderSet: ShadowRenderSet;
	sliceIndex: number;
	tileX: number;
	tileY: number;
	localTileX: number;
	localTileY: number;
	localTileSpan: number;
	atlasBaseSize: number;
}

interface ShadowAnimationState {
	jointMatrices: Float32Array | null;
	morphWeights: Float32Array | null;
	morphTargetCount: number;
	morphPositionBuffer: GPUBuffer | null;
}

interface ShadowAnimationBindingEntry {
	paramsBuffer: GPUBuffer;
	jointBuffer: GPUBuffer;
	morphWeightBuffer: GPUBuffer;
	bindGroup: GPUBindGroup | null;
	jointCapacity: number;
	morphCapacity: number;
	morphPositionBuffer: GPUBuffer | null;
	lastUsedFrame: number;
}

interface ShadowDrawCandidate {
	packet: DrawPacket;
	geometry: WebGPUGeometryHandle;
	vertexBuffer: GPUBuffer;
	indexBuffer: GPUBuffer;
}

interface ShadowInstancedDrawBatch {
	candidate: ShadowDrawCandidate;
	animationBindGroup: GPUBindGroup;
	firstInstance: number;
	instanceCount: number;
}

const SHADOW_INSTANCE_DATA_UINTS = 8;

export class WebGPUShadowPass {
	private _backend: WebGPUBackend;
	private _geometryRegistry: WebGPUGeometryRegistry;
	private _shadowAtlases: WebGPUShadowAtlasAllocator;
	private _depthRemapMatrix = new Matrix4([
		[1, 0, 0, 0],
		[0, 1, 0, 0],
		[0, 0, 0.5, 0.5],
		[0, 0, 0, 1],
	]);
	private _shadowViewProjectionMatrix = Matrix4.identity();
	private _mvpMatrix = Matrix4.identity();
	private _instanceMvpData = new Float32Array(0);
	private _instanceMetaData = new Uint32Array(0);
	private _shaderModule: IShaderModule | null = null;
	private _shaderModulePromise: Promise<IShaderModule> | null = null;
	private _bindGroupLayout: GPUBindGroupLayout | null = null;
	private _animationBindGroupLayout: GPUBindGroupLayout | null = null;
	private _pipelineLayout: GPUPipelineLayout | null = null;
	private _pipeline: GPURenderPipeline | null = null;
	private _transmittancePipeline: GPURenderPipeline | null = null;
	private _instanceMvpBuffer: GPUBuffer | null = null;
	private _instanceMetaBuffer: GPUBuffer | null = null;
	private _instanceTransmittanceBuffer: GPUBuffer | null = null;
	private _instanceMvpBindGroup: GPUBindGroup | null = null;
	private _instanceMvpCapacity = 0;
	private _frustum = new Frustum();
	private _animationBindings = new Map<string, ShadowAnimationBindingEntry>();
	private _fallbackStorageBuffer: GPUBuffer | null = null;
	private _frameId = 0;
	private _instanceTransmittanceData = new Float32Array(0);

	constructor(
		backend: WebGPUBackend,
		geometryRegistry: WebGPUGeometryRegistry,
		shadowAtlases: WebGPUShadowAtlasAllocator
	) {
		this._backend = backend;
		this._geometryRegistry = geometryRegistry;
		this._shadowAtlases = shadowAtlases;
	}

	public async render(
		context: FrameContext,
		frameEncoder?: ICommandEncoder | null
	): Promise<void> {
		if (!context.features.enableShadows) return;

		const frame = context.scene;
		const shadowMaps = context.shadowMaps;
		const slots = this._collectShadowSlots(frame, shadowMaps);
		const maxShadowSize = getMaxShadowSize(slots);
		const requestedAtlasTileSize = Math.max(1, maxShadowSize);
		const atlasTexture =
			this._shadowAtlases.ensureAtlasForTileSize(requestedAtlasTileSize);
		const transmittanceAtlasTexture = this._shadowAtlases.transmittanceAtlas;
		const atlasTileSize = Math.max(1, this._shadowAtlases.tileSize);
		const atlasView = getWebGPUTexture(atlasTexture).view;
		const transmittanceAtlasView =
			transmittanceAtlasTexture ?
				getWebGPUTexture(transmittanceAtlasTexture).view
			:	null;
		if (!atlasView || !transmittanceAtlasView) return;

		await this._ensurePipelineResources();
		if (
			!this._pipeline ||
			!this._transmittancePipeline ||
			!this._bindGroupLayout ||
			!this._animationBindGroupLayout
		) {
			return;
		}

		this._frameId++;
		const drawCandidates = this._collectShadowDrawCandidates(
			frame.shadowCasterPackets
		);
		const transmitterCandidates = this._collectShadowDrawCandidates(
			frame.shadowTransmitterPackets
		);
		const animationBindingCache = new Map<string, GPUBindGroup | null>();

		const { commandEncoder, submitAtEnd } =
			this._resolveShadowCommandEncoder(frameEncoder);

		const passEncoder = commandEncoder.beginRenderPass({
			label: "WebGPUShadowPass",
			colorAttachments: [],
			depthStencilAttachment: {
				view: atlasView,
				depthClearValue: 1,
				depthLoadOp: "clear",
				depthStoreOp: "store",
			},
		});

		passEncoder.setPipeline(this._pipeline);

		for (const slot of slots) {
			const shadowMapSize = Math.max(1, slot.shadowMap.size | 0);
			const baseOffsetX = slot.tileX * atlasTileSize;
			const baseOffsetY = slot.tileY * atlasTileSize;
			const subTileSize =
				slot.localTileSpan > 1 ?
					Math.max(1, Math.floor(atlasTileSize / slot.localTileSpan))
				:	atlasTileSize;
			const viewportX = baseOffsetX + slot.localTileX * subTileSize;
			const viewportY = baseOffsetY + slot.localTileY * subTileSize;
			const viewportSize = Math.min(
				shadowMapSize,
				slot.localTileSpan > 1 ? subTileSize : atlasTileSize
			);
			passEncoder.setViewport(
				viewportX,
				viewportY,
				viewportSize,
				viewportSize,
				0,
				1
			);
			passEncoder.setScissorRect(
				viewportX,
				viewportY,
				viewportSize,
				viewportSize
			);
			const slotSlice = slot.renderSet.slices[slot.sliceIndex];
			if (slotSlice) {
				slotSlice.atlasRect = {
					offsetX: viewportX,
					offsetY: viewportY,
					size: viewportSize,
					tileSize: atlasTileSize,
					localTileX: slot.localTileX,
					localTileY: slot.localTileY,
					localTileSpan: slot.localTileSpan,
				};
			}
			Matrix4.multiply(
				this._depthRemapMatrix,
				slot.shadowMap.viewProjectionMatrix!,
				this._shadowViewProjectionMatrix
			);

			// Update frustum for current shadow map
			this._frustum.setFromMatrix(slot.shadowMap.viewProjectionMatrix!);

			this._drawShadowCasters(
				passEncoder,
				drawCandidates,
				this._shadowViewProjectionMatrix,
				context,
				animationBindingCache
			);
		}

		passEncoder.end();
		const transmittancePassEncoder = commandEncoder.beginRenderPass({
			label: "WebGPUShadowTransmittancePass",
			colorAttachments: [
				{
					view: transmittanceAtlasView,
					clearValue: { r: 1, g: 1, b: 1, a: 1 },
					loadOp: "clear",
					storeOp: "store",
				},
			],
			depthStencilAttachment: {
				view: atlasView,
				depthLoadOp: "load",
				depthStoreOp: "store",
			},
		});
		transmittancePassEncoder.setPipeline(this._transmittancePipeline);
		for (const slot of slots) {
			const shadowMapSize = Math.max(1, slot.shadowMap.size | 0);
			const baseOffsetX = slot.tileX * atlasTileSize;
			const baseOffsetY = slot.tileY * atlasTileSize;
			const subTileSize =
				slot.localTileSpan > 1 ?
					Math.max(1, Math.floor(atlasTileSize / slot.localTileSpan))
				:	atlasTileSize;
			const viewportX = baseOffsetX + slot.localTileX * subTileSize;
			const viewportY = baseOffsetY + slot.localTileY * subTileSize;
			const viewportSize = Math.min(
				shadowMapSize,
				slot.localTileSpan > 1 ? subTileSize : atlasTileSize
			);
			transmittancePassEncoder.setViewport(
				viewportX,
				viewportY,
				viewportSize,
				viewportSize,
				0,
				1
			);
			transmittancePassEncoder.setScissorRect(
				viewportX,
				viewportY,
				viewportSize,
				viewportSize
			);
			Matrix4.multiply(
				this._depthRemapMatrix,
				slot.shadowMap.viewProjectionMatrix!,
				this._shadowViewProjectionMatrix
			);
			this._frustum.setFromMatrix(slot.shadowMap.viewProjectionMatrix!);
			this._drawShadowTransmitters(
				transmittancePassEncoder,
				transmitterCandidates,
				this._shadowViewProjectionMatrix,
				context,
				animationBindingCache
			);
		}
		transmittancePassEncoder.end();
		if (submitAtEnd) {
			this._requireBackendQueue().submit([commandEncoder.finish()]);
		}
		this._trimAnimationResources();
	}

	public onShaderRuntimeChanged(): void {
		this._shaderModule = null;
		this._shaderModulePromise = null;
		this._pipeline = null;
		this._transmittancePipeline = null;
	}

	public async warmup(): Promise<void> {
		await this._ensurePipelineResources();
	}

	public destroy(): void {
		this._shaderModule = null;
		this._shaderModulePromise = null;
		this._bindGroupLayout = null;
		this._animationBindGroupLayout = null;
		this._pipelineLayout = null;
		this._pipeline = null;
		this._transmittancePipeline = null;
		this._instanceMvpBuffer?.destroy();
		this._instanceMetaBuffer?.destroy();
		this._instanceTransmittanceBuffer?.destroy();
		this._instanceMvpBuffer = null;
		this._instanceMetaBuffer = null;
		this._instanceTransmittanceBuffer = null;
		this._instanceMvpBindGroup = null;
		this._instanceMvpCapacity = 0;
		this._instanceMvpData = new Float32Array(0);
		this._instanceMetaData = new Uint32Array(0);
		this._instanceTransmittanceData = new Float32Array(0);
		for (const entry of this._animationBindings.values()) {
			entry.paramsBuffer.destroy();
			entry.jointBuffer.destroy();
			entry.morphWeightBuffer.destroy();
		}
		this._animationBindings.clear();
		this._fallbackStorageBuffer?.destroy();
		this._fallbackStorageBuffer = null;
		this._frameId = 0;
	}

	private _drawShadowCasters(
		passEncoder: GPURenderPassEncoder,
		drawCandidates: ShadowDrawCandidate[],
		viewProjectionMatrix: Matrix4,
		context: FrameContext,
		animationBindingCache: Map<string, GPUBindGroup | null>
	): void {
		const drawBatches = this._buildShadowDrawBatches(
			drawCandidates,
			viewProjectionMatrix,
			context,
			animationBindingCache,
			false
		);
		if (
			drawBatches.length === 0 ||
			!this._instanceMvpBindGroup
		) {
			return;
		}

		passEncoder.setBindGroup(0, this._instanceMvpBindGroup);
		for (const batch of drawBatches) {
			passEncoder.setVertexBuffer(0, batch.candidate.vertexBuffer);
			passEncoder.setIndexBuffer(batch.candidate.indexBuffer, "uint32");
			passEncoder.setBindGroup(1, batch.animationBindGroup);
			passEncoder.drawIndexed(
				batch.candidate.geometry.indexCount,
				batch.instanceCount,
				0,
				0,
				batch.firstInstance
			);
		}
	}

	private _drawShadowTransmitters(
		passEncoder: GPURenderPassEncoder,
		drawCandidates: ShadowDrawCandidate[],
		viewProjectionMatrix: Matrix4,
		context: FrameContext,
		animationBindingCache: Map<string, GPUBindGroup | null>
	): void {
		const drawBatches = this._buildShadowDrawBatches(
			drawCandidates,
			viewProjectionMatrix,
			context,
			animationBindingCache,
			true
		);
		if (
			drawBatches.length === 0 ||
			!this._instanceMvpBindGroup
		) {
			return;
		}

		passEncoder.setBindGroup(0, this._instanceMvpBindGroup);
		for (const batch of drawBatches) {
			passEncoder.setVertexBuffer(0, batch.candidate.vertexBuffer);
			passEncoder.setIndexBuffer(batch.candidate.indexBuffer, "uint32");
			passEncoder.setBindGroup(1, batch.animationBindGroup);
			passEncoder.drawIndexed(
				batch.candidate.geometry.indexCount,
				batch.instanceCount,
				0,
				0,
				batch.firstInstance
			);
		}
	}

	private _buildShadowDrawBatches(
		drawCandidates: ShadowDrawCandidate[],
		viewProjectionMatrix: Matrix4,
		context: FrameContext,
		animationBindingCache: Map<string, GPUBindGroup | null>,
		resolveTransmittance: boolean
	): ShadowInstancedDrawBatch[] {
		const drawBatches: ShadowInstancedDrawBatch[] = [];
		let instanceCount = 0;
		for (const candidate of drawCandidates) {
			const packet = candidate.packet;
			// Per-light Frustum Culling
			if (
				!this._frustum.intersectsSphere(
					packet.worldBounds.center,
					packet.worldBounds.radius
				)
			) {
				continue;
			}

			if (!animationBindingCache.has(packet.id)) {
				animationBindingCache.set(
					packet.id,
					this._resolveAnimationBinding(packet, candidate.geometry, context)
				);
			}
			const animationBindGroup =
				animationBindingCache.get(packet.id) ?? null;
			if (!animationBindGroup) continue;

			Matrix4.multiply(
				viewProjectionMatrix,
				packet.worldMatrix,
				this._mvpMatrix
			);

			this._ensureInstanceDataCapacity(instanceCount + 1);
			const mvpOffset = instanceCount * 16;
			this._setMatrixInArray(this._mvpMatrix, this._instanceMvpData, mvpOffset);
			const transmittanceOffset = instanceCount * 4;
			if (resolveTransmittance) {
				const transmittance = resolveMaterialShadowTransmittance(
					packet.material
				);
				this._instanceTransmittanceData[transmittanceOffset] =
					transmittance.r;
				this._instanceTransmittanceData[transmittanceOffset + 1] =
					transmittance.g;
				this._instanceTransmittanceData[transmittanceOffset + 2] =
					transmittance.b;
				this._instanceTransmittanceData[transmittanceOffset + 3] = 1;
			} else {
				this._instanceTransmittanceData[transmittanceOffset] = 1;
				this._instanceTransmittanceData[transmittanceOffset + 1] = 1;
				this._instanceTransmittanceData[transmittanceOffset + 2] = 1;
				this._instanceTransmittanceData[transmittanceOffset + 3] = 1;
			}
			const metaOffset = instanceCount * SHADOW_INSTANCE_DATA_UINTS;
			this._setShadowInstanceMetaInArray(
				this._instanceMetaData,
				metaOffset,
				instanceCount,
				0,
				0,
				0,
				0
			);
			instanceCount++;

			const lastBatch = drawBatches[drawBatches.length - 1];
			if (
				lastBatch &&
				lastBatch.animationBindGroup === animationBindGroup &&
				lastBatch.candidate.vertexBuffer === candidate.vertexBuffer &&
				lastBatch.candidate.indexBuffer === candidate.indexBuffer &&
				lastBatch.candidate.geometry.indexCount === candidate.geometry.indexCount
			) {
				lastBatch.instanceCount++;
				continue;
			}

			drawBatches.push({
				candidate,
				animationBindGroup,
				firstInstance: instanceCount - 1,
				instanceCount: 1,
			});
		}

		if (
			instanceCount === 0 ||
			!this._upsertShadowInstanceResources(instanceCount)
		) {
			return [];
		}
		this._requireBackendQueue().writeBuffer(
			this._instanceMvpBuffer!,
			0,
			this._instanceMvpData.subarray(
				0,
				instanceCount * 16
			) as Float32Array<ArrayBuffer>
		);
		this._requireBackendQueue().writeBuffer(
			this._instanceMetaBuffer!,
			0,
			this._instanceMetaData.subarray(
				0,
				instanceCount * SHADOW_INSTANCE_DATA_UINTS
			) as Uint32Array<ArrayBuffer>
		);
		this._requireBackendQueue().writeBuffer(
			this._instanceTransmittanceBuffer!,
			0,
			this._instanceTransmittanceData.subarray(
				0,
				instanceCount * 4
			) as Float32Array<ArrayBuffer>
		);

		return drawBatches;
	}

	private _ensureInstanceDataCapacity(instanceCount: number): void {
		const requiredLength = Math.max(1, instanceCount) * 16;
		if (this._instanceMvpData.length < requiredLength) {
			const nextLength = Math.max(requiredLength, this._instanceMvpData.length * 2);
			const next = new Float32Array(nextLength);
			next.set(this._instanceMvpData);
			this._instanceMvpData = next;
		}

		const requiredMetaLength =
			Math.max(1, instanceCount) * SHADOW_INSTANCE_DATA_UINTS;
		const requiredTransmittanceLength = Math.max(1, instanceCount) * 4;
		if (this._instanceTransmittanceData.length < requiredTransmittanceLength) {
			const nextTransmittanceLength = Math.max(
				requiredTransmittanceLength,
				this._instanceTransmittanceData.length * 2
			);
			const nextTransmittance = new Float32Array(nextTransmittanceLength);
			nextTransmittance.set(this._instanceTransmittanceData);
			this._instanceTransmittanceData = nextTransmittance;
		}
		if (this._instanceMetaData.length >= requiredMetaLength) {
			return;
		}
		const nextMetaLength = Math.max(
			requiredMetaLength,
			this._instanceMetaData.length * 2
		);
		const nextMeta = new Uint32Array(nextMetaLength);
		nextMeta.set(this._instanceMetaData);
		this._instanceMetaData = nextMeta;
	}

	private _upsertShadowInstanceResources(instanceCount: number): boolean {
		if (!this._bindGroupLayout) {
			return false;
		}
		const device = this._requireBackendDevice();
		const requiredCapacity = Math.max(1, instanceCount);

		if (
			!this._instanceMvpBuffer ||
			!this._instanceMetaBuffer ||
			!this._instanceTransmittanceBuffer ||
			requiredCapacity > this._instanceMvpCapacity
		) {
			this._instanceMvpBuffer?.destroy();
			this._instanceMetaBuffer?.destroy();
			this._instanceTransmittanceBuffer?.destroy();
			this._instanceMvpBuffer = device.createBuffer({
				label: "WebGPUShadowDepthMvpStorage",
				size: requiredCapacity * 16 * 4,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
			});
			this._instanceMetaBuffer = device.createBuffer({
				label: "WebGPUShadowDepthInstanceMeta",
				size: requiredCapacity * SHADOW_INSTANCE_DATA_UINTS * 4,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
			});
			this._instanceTransmittanceBuffer = device.createBuffer({
				label: "WebGPUShadowTransmittanceStorage",
				size: requiredCapacity * 4 * 4,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
			});
			this._instanceMvpBindGroup = null;
			this._instanceMvpCapacity = requiredCapacity;
		}

		if (
			!this._instanceMvpBindGroup &&
			this._instanceMvpBuffer &&
			this._instanceMetaBuffer
		) {
			this._instanceMvpBindGroup = device.createBindGroup({
				label: "WebGPUShadowDepthMvpBindGroup",
				layout: this._bindGroupLayout,
				entries: [
					{
						binding: 0,
						resource: { buffer: this._instanceMvpBuffer },
					},
					{
						binding: 1,
						resource: { buffer: this._instanceMetaBuffer },
					},
					{
						binding: 2,
						resource: { buffer: this._instanceTransmittanceBuffer },
					},
				],
			});
		}

		return (
			!!this._instanceMvpBuffer &&
			!!this._instanceMetaBuffer &&
			!!this._instanceTransmittanceBuffer &&
			!!this._instanceMvpBindGroup
		);
	}

	private _collectShadowDrawCandidates(
		packets: DrawPacket[]
	): ShadowDrawCandidate[] {
		const candidates: ShadowDrawCandidate[] = [];
		for (const packet of packets) {
			if (
				(packet.primitive.topology ?? DEFAULT_PRIMITIVE_DRAW_TOPOLOGY) !==
				DEFAULT_PRIMITIVE_DRAW_TOPOLOGY
			) {
				continue;
			}
			const geometry = this._geometryRegistry.getGeometry(packet.primitive);
			const vertexBuffer = (
				geometry.vertexBuffer as { _gpuResource?: GPUBuffer }
			)._gpuResource;
			const indexBuffer = (geometry.indexBuffer as { _gpuResource?: GPUBuffer })
				._gpuResource;
			if (!vertexBuffer || !indexBuffer) {
				continue;
			}
			candidates.push({
				packet,
				geometry,
				vertexBuffer,
				indexBuffer,
			});
		}
		return candidates;
	}

	private _resolveShadowCommandEncoder(
		frameEncoder?: ICommandEncoder | null
	): {
		commandEncoder: GPUCommandEncoder;
		submitAtEnd: boolean;
	} {
		const nativeCommandEncoder =
			tryGetNativeWebGPUCommandEncoder(frameEncoder);
		if (
			nativeCommandEncoder &&
			typeof (nativeCommandEncoder as GPUCommandEncoder).beginRenderPass ===
				"function" &&
			typeof (nativeCommandEncoder as GPUCommandEncoder).finish === "function"
		) {
			return {
				commandEncoder: nativeCommandEncoder as GPUCommandEncoder,
				submitAtEnd: false,
			};
		}

		return {
			commandEncoder: this._requireBackendDevice().createCommandEncoder({
				label: "WebGPUShadowEncoder",
			}),
			submitAtEnd: true,
		};
	}

	private _resolveAnimationBinding(
		packet: DrawPacket,
		geometry: WebGPUGeometryHandle,
		context: FrameContext
	): GPUBindGroup | null {
		if (!this._animationBindGroupLayout || !this._fallbackStorageBuffer) {
			return null;
		}
		const device = this._requireBackendDevice();
		const queue = this._requireBackendQueue();

		const key = packet.id;
		let entry = this._animationBindings.get(key);
		if (!entry) {
			entry = this._createAnimationBindingEntry(key);
			this._animationBindings.set(key, entry);
		}
		entry.lastUsedFrame = this._frameId;

		const state = this._resolveAnimationState(packet, geometry, context);
		const jointCount = Math.max(
			0,
			Math.floor((state.jointMatrices?.length ?? 0) / 16)
		);
		const morphCount = Math.min(
			Math.max(0, state.morphTargetCount),
			state.morphWeights?.length ?? state.morphTargetCount
		);
		const jointCapacity = Math.max(1, jointCount);
		const morphCapacity = Math.max(1, morphCount);

		let needsRebind = false;
		if (jointCapacity > entry.jointCapacity) {
			entry.jointBuffer.destroy();
			entry.jointBuffer = device.createBuffer({
				label: `WebGPUShadowJointBuffer_${key}`,
				size: jointCapacity * 16 * 4,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
			});
			entry.jointCapacity = jointCapacity;
			needsRebind = true;
		}
		if (morphCapacity > entry.morphCapacity) {
			entry.morphWeightBuffer.destroy();
			entry.morphWeightBuffer = device.createBuffer({
				label: `WebGPUShadowMorphWeightBuffer_${key}`,
				size: morphCapacity * 4,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
			});
			entry.morphCapacity = morphCapacity;
			needsRebind = true;
		}

		queue.writeBuffer(
			entry.paramsBuffer,
			0,
			new Uint32Array([jointCount, morphCount, jointCount, morphCount])
		);
		if (jointCount > 0 && state.jointMatrices) {
			queue.writeBuffer(
				entry.jointBuffer,
				0,
				state.jointMatrices.subarray(
					0,
					jointCount * 16
				) as Float32Array<ArrayBuffer>
			);
		}
		if (morphCount > 0 && state.morphWeights) {
			queue.writeBuffer(
				entry.morphWeightBuffer,
				0,
				state.morphWeights.subarray(0, morphCount) as Float32Array<ArrayBuffer>
			);
		}

		const morphPositionBuffer =
			state.morphPositionBuffer ?? this._fallbackStorageBuffer;
		if (entry.morphPositionBuffer !== morphPositionBuffer) {
			entry.morphPositionBuffer = morphPositionBuffer;
			needsRebind = true;
		}

		if (!entry.bindGroup || needsRebind) {
			entry.bindGroup = device.createBindGroup({
				label: `WebGPUShadowAnimationBinding_${key}`,
				layout: this._animationBindGroupLayout,
				entries: [
					{
						binding: 0,
						resource: { buffer: entry.paramsBuffer },
					},
					{
						binding: 1,
						resource: { buffer: entry.jointBuffer },
					},
					{
						binding: 2,
						resource: { buffer: entry.morphWeightBuffer },
					},
					{
						binding: 3,
						resource: { buffer: morphPositionBuffer },
					},
				],
			});
		}

		return entry.bindGroup;
	}

	private _resolveAnimationState(
		packet: DrawPacket,
		geometry: WebGPUGeometryHandle,
		context: FrameContext
	): ShadowAnimationState {
		const runtimeJointMap =
			context.transient.get(ANIMATION_WEBGPU_JOINT_MATRICES_KEY) ?? null;
		const runtimeMorphMap =
			context.transient.get(ANIMATION_WEBGPU_MORPH_WEIGHTS_KEY) ?? null;
		const runtimeJoint = runtimeJointMap?.get(packet.meshInstance.id) ?? null;
		let jointMatrices: Float32Array | null = null;
		if (runtimeJoint?.skeleton) {
			runtimeJoint.skeleton.updateJointMatrices(
				packet.meshInstance.worldMatrix
			);
			jointMatrices = runtimeJoint.skeleton.toFloat32Array(
				runtimeJoint.matrices
			);
		} else if (packet.meshInstance.skeleton) {
			packet.meshInstance.skeleton.updateJointMatrices(
				packet.meshInstance.worldMatrix
			);
			jointMatrices = packet.meshInstance.skeleton.toFloat32Array();
		}

		const runtimeMorph = runtimeMorphMap?.get(packet.primitive.id) ?? null;
		let morphTargetCount = Math.max(0, runtimeMorph?.targetCount ?? 0);
		let sourceMorphWeights: Float32Array | null = runtimeMorph?.weights ?? null;
		if (!sourceMorphWeights || morphTargetCount <= 0) {
			const primitiveIndex = packet.mesh.primitives.indexOf(packet.primitive);
			const instanceWeights =
				primitiveIndex >= 0 ?
					packet.meshInstance.morphWeights[primitiveIndex]
				:	null;
			sourceMorphWeights = instanceWeights ?? null;
			morphTargetCount = sourceMorphWeights?.length ?? 0;
		}
		morphTargetCount = Math.min(
			Math.max(0, morphTargetCount),
			geometry.morphTargetCount,
			WEBGPU_MAX_MORPH_TARGETS
		);

		let morphWeights: Float32Array | null = null;
		if (sourceMorphWeights && morphTargetCount > 0) {
			morphWeights = sourceMorphWeights.subarray(0, morphTargetCount);
		}

		const morphPositionBuffer = (
			geometry.morphPositionBuffer as { _gpuResource?: GPUBuffer } | null
		)?._gpuResource;

		return {
			jointMatrices,
			morphWeights,
			morphTargetCount,
			morphPositionBuffer: morphPositionBuffer ?? null,
		};
	}

	private _setMatrixInArray(
		matrix: Matrix4,
		target: Float32Array,
		offset: number
	): void {
		const elements = matrix.elements;
		target[offset] = elements[0][0];
		target[offset + 1] = elements[1][0];
		target[offset + 2] = elements[2][0];
		target[offset + 3] = elements[3][0];
		target[offset + 4] = elements[0][1];
		target[offset + 5] = elements[1][1];
		target[offset + 6] = elements[2][1];
		target[offset + 7] = elements[3][1];
		target[offset + 8] = elements[0][2];
		target[offset + 9] = elements[1][2];
		target[offset + 10] = elements[2][2];
		target[offset + 11] = elements[3][2];
		target[offset + 12] = elements[0][3];
		target[offset + 13] = elements[1][3];
		target[offset + 14] = elements[2][3];
		target[offset + 15] = elements[3][3];
	}

	private _setShadowInstanceMetaInArray(
		target: Uint32Array,
		offset: number,
		instanceBaseOffset: number,
		vertexBaseOffset: number,
		jointBaseOffset: number,
		morphWeightBaseOffset: number,
		morphDeltaBaseOffset: number
	): void {
		target[offset] = instanceBaseOffset >>> 0;
		target[offset + 1] = vertexBaseOffset >>> 0;
		target[offset + 2] = jointBaseOffset >>> 0;
		target[offset + 3] = morphWeightBaseOffset >>> 0;
		target[offset + 4] = morphDeltaBaseOffset >>> 0;
		target[offset + 5] = 0;
		target[offset + 6] = 0;
		target[offset + 7] = 0;
	}

	private _collectShadowSlots(
		scene: PreparedScene,
		shadowMaps: Map<ShadowCastingLight, ShadowRenderSet>
	): ShadowRenderSlot[] {
		const slots: ShadowRenderSlot[] = [];
		const atlasColumns = Math.max(1, WEBGPU_SHADOW_ATLAS_COLUMNS);
		let directionalIndex = 0;
		let spotIndex = 0;

		for (const light of scene.lights) {
			if (light.type === LightType.Directional) {
				if (directionalIndex >= WEBGPU_MAX_DIRECTIONAL_LIGHTS) continue;
				if (isShadowCastingLight(light)) {
					const renderSet = shadowMaps.get(light) ?? null;
					const shadowMap = getPrimaryShadowMap(renderSet);
					if (shadowMap?.viewProjectionMatrix && renderSet) {
						const globalTileIndex = directionalIndex;
						const tileX = globalTileIndex % atlasColumns;
						const tileY = Math.floor(globalTileIndex / atlasColumns);
						const isCSM = renderSet.effectiveStrategyType === "csm";
						const maxSlices = Math.min(
							renderSet.slices.length,
							4
						);
						if (isCSM && maxSlices > 1) {
							for (let sliceIndex = 0; sliceIndex < maxSlices; sliceIndex++) {
								const slice = renderSet.slices[sliceIndex];
								const sliceShadowMap = slice?.shadowMap ?? null;
								if (!sliceShadowMap?.viewProjectionMatrix) {
									continue;
								}
								const localTileX = sliceIndex % 2;
								const localTileY = Math.floor(sliceIndex / 2);
								slots.push({
									shadowMap: sliceShadowMap,
									renderSet,
									sliceIndex,
									tileX,
									tileY,
									localTileX,
									localTileY,
									localTileSpan: 2,
									atlasBaseSize: Math.max(1, renderSet.size | 0),
								});
							}
						} else {
							const primarySlice = renderSet.slices[0];
							slots.push({
								shadowMap,
								renderSet,
								sliceIndex: 0,
								tileX,
								tileY,
								localTileX: 0,
								localTileY: 0,
								localTileSpan: 1,
								atlasBaseSize: Math.max(
									1,
									primarySlice?.shadowMap.size ?? (renderSet.size | 0)
								),
							});
						}
					}
				}
				directionalIndex++;
				continue;
			}

			if (light.type === LightType.Spot) {
				if (spotIndex >= WEBGPU_MAX_SPOT_LIGHTS) continue;
				if (isShadowCastingLight(light)) {
					const renderSet = shadowMaps.get(light) ?? null;
					const shadowMap = getPrimaryShadowMap(renderSet);
					if (shadowMap?.viewProjectionMatrix && renderSet) {
						const globalTileIndex =
							WEBGPU_MAX_DIRECTIONAL_LIGHTS + spotIndex;
						slots.push({
							shadowMap,
							renderSet,
							sliceIndex: 0,
							tileX: globalTileIndex % atlasColumns,
							tileY: Math.floor(globalTileIndex / atlasColumns),
							localTileX: 0,
							localTileY: 0,
							localTileSpan: 1,
							atlasBaseSize: Math.max(
								1,
								renderSet.slices[0]?.shadowMap.size ?? (renderSet.size | 0)
							),
						});
					}
				}
				spotIndex++;
			}
		}

		return slots;
	}

	private async _ensurePipelineResources(): Promise<void> {
		if (
			this._pipeline &&
			this._transmittancePipeline &&
			this._bindGroupLayout &&
			this._animationBindGroupLayout &&
			this._fallbackStorageBuffer
		) {
			return;
		}

		const device = this._requireBackendDevice();
		const queue = this._requireBackendQueue();
		if (!this._shaderModule) {
			if (!this._shaderModulePromise) {
				const composite = createInlineCompositeShaderSource(
					WEBGPU_SHADOW_DEPTH_SHADER,
					"<webgpu-shadow-depth>",
					"source"
				);
				this._shaderModulePromise = this._backend.createShaderModule({
					label: "WebGPUShadowDepthShader",
					code: composite.code,
					sourceMap: composite.sourceMap,
					language: "wgsl",
					stage: "vertex",
					entryPoint: "vsMain",
					sourceKind: "shadow",
				});
			}
			try {
				this._shaderModule = await this._shaderModulePromise;
			} catch (error) {
				this._shaderModulePromise = null;
				throw error;
			}
		}

		if (!this._bindGroupLayout) {
			this._bindGroupLayout = device.createBindGroupLayout({
				label: "WebGPUShadowDepthBindGroupLayout",
				entries: [
					{
						binding: 0,
						visibility: GPUShaderStage.VERTEX,
						buffer: { type: "read-only-storage" },
					},
					{
						binding: 1,
						visibility: GPUShaderStage.VERTEX,
						buffer: { type: "read-only-storage" },
					},
					{
						binding: 2,
						visibility: GPUShaderStage.VERTEX,
						buffer: { type: "read-only-storage" },
					},
				],
			});
		}

		if (!this._animationBindGroupLayout) {
			this._animationBindGroupLayout = device.createBindGroupLayout({
				label: "WebGPUShadowAnimationBindGroupLayout",
				entries: [
					{
						binding: 0,
						visibility: GPUShaderStage.VERTEX,
						buffer: { type: "uniform" },
					},
					{
						binding: 1,
						visibility: GPUShaderStage.VERTEX,
						buffer: { type: "read-only-storage" },
					},
					{
						binding: 2,
						visibility: GPUShaderStage.VERTEX,
						buffer: { type: "read-only-storage" },
					},
					{
						binding: 3,
						visibility: GPUShaderStage.VERTEX,
						buffer: { type: "read-only-storage" },
					},
				],
			});
		}

		if (
			!this._pipelineLayout &&
			this._bindGroupLayout &&
			this._animationBindGroupLayout
		) {
			this._pipelineLayout = device.createPipelineLayout({
				label: "WebGPUShadowDepthPipelineLayout",
				bindGroupLayouts: [
					this._bindGroupLayout,
					this._animationBindGroupLayout,
				],
			});
		}

		if (!this._fallbackStorageBuffer) {
			this._fallbackStorageBuffer = device.createBuffer({
				label: "WebGPUShadowFallbackStorage",
				size: 16,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
			});
			queue.writeBuffer(
				this._fallbackStorageBuffer,
				0,
				new Float32Array(4)
			);
		}

		if (!this._pipeline && this._shaderModule && this._pipelineLayout) {
			this._pipeline = device.createRenderPipeline({
				label: "WebGPUShadowDepthPipeline",
				layout: this._pipelineLayout,
				vertex: {
					module: getWebGPUShaderModule(this._shaderModule),
					entryPoint: "vsMain",
					buffers: [createWebGPUShadowVertexBufferLayout()],
				},
				primitive: {
					topology: "triangle-list",
					cullMode: "none",
					frontFace: "ccw",
				},
				depthStencil: {
					format: "depth32float",
					depthWriteEnabled: true,
					depthCompare: "less",
				},
			});
		}
		if (
			!this._transmittancePipeline &&
			this._shaderModule &&
			this._pipelineLayout
		) {
			this._transmittancePipeline = device.createRenderPipeline({
				label: "WebGPUShadowTransmittancePipeline",
				layout: this._pipelineLayout,
				vertex: {
					module: getWebGPUShaderModule(this._shaderModule),
					entryPoint: "vsMain",
					buffers: [createWebGPUShadowVertexBufferLayout()],
				},
				fragment: {
					module: getWebGPUShaderModule(this._shaderModule),
					entryPoint: "fsTransmittance",
					targets: [
						{
							format: TextureFormat.RGBA16Float,
							blend: {
								color: {
									operation: "add",
									srcFactor: "zero",
									dstFactor: "src",
								},
								alpha: {
									operation: "add",
									srcFactor: "zero",
									dstFactor: "one",
								},
							},
						},
					],
				},
				primitive: {
					topology: "triangle-list",
					cullMode: "none",
					frontFace: "ccw",
				},
				depthStencil: {
					format: "depth32float",
					depthWriteEnabled: false,
					depthCompare: "less",
				},
			});
		}
	}

	private _createAnimationBindingEntry(
		key: string
	): ShadowAnimationBindingEntry {
		const device = this._requireBackendDevice();
		return {
			paramsBuffer: device.createBuffer({
				label: `WebGPUShadowAnimationParams_${key}`,
				size: 16,
				usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
			}),
			jointBuffer: device.createBuffer({
				label: `WebGPUShadowJointBuffer_${key}`,
				size: 16 * 4,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
			}),
			morphWeightBuffer: device.createBuffer({
				label: `WebGPUShadowMorphWeightBuffer_${key}`,
				size: 4,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
			}),
			bindGroup: null,
			jointCapacity: 1,
			morphCapacity: 1,
			morphPositionBuffer: null,
			lastUsedFrame: this._frameId,
		};
	}

	private _requireBackendDevice(): GPUDevice {
		const device = this._backend.device;
		if (!device) {
			throw new Error(
				"WebGPU backend is not initialized; shadow pass requires an active GPU device."
			);
		}
		return device;
	}

	private _requireBackendQueue(): GPUQueue {
		const queue = this._backend.queue;
		if (!queue) {
			throw new Error(
				"WebGPU backend is not initialized; shadow pass requires an active GPU queue."
			);
		}
		return queue;
	}

	private _trimAnimationResources(): void {
		const staleFrame = this._frameId - 120;
		for (const [key, entry] of this._animationBindings) {
			if (entry.lastUsedFrame >= staleFrame) continue;
			entry.paramsBuffer.destroy();
			entry.jointBuffer.destroy();
			entry.morphWeightBuffer.destroy();
			this._animationBindings.delete(key);
		}
	}
}

function getMaxShadowSize(slots: ShadowRenderSlot[]): number {
	let maxSize = 0;
	for (const slot of slots) {
		maxSize = Math.max(maxSize, slot.atlasBaseSize | 0);
	}
	return maxSize;
}
