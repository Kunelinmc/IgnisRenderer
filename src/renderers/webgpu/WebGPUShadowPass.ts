import { Frustum } from "../../maths/Frustum";
import {
	LightType,
	isShadowCastingLight,
	type ShadowCastingLight,
} from "../../lights";
import { Matrix4 } from "../../maths/Matrix4";
import type {
	DrawPacket,
	FrameContext,
	PreparedScene,
} from "../../pipeline/types";
import type { ShadowMap } from "../../utils/ShadowMapping";
import {
	ANIMATION_WEBGPU_JOINT_MATRICES_KEY,
	ANIMATION_WEBGPU_MORPH_WEIGHTS_KEY,
	type JointMatrixMap,
	type MorphWeightMap,
} from "../../simulation/animation/types";
import type { WebGPUBackend } from "../WebGPUBackend";
import {
	WEBGPU_MAX_DIRECTIONAL_LIGHTS,
	WEBGPU_MAX_MORPH_TARGETS,
	WEBGPU_MAX_SPOT_LIGHTS,
	WEBGPU_SCENE_VERTEX_STRIDE,
} from "./constants";
import { getWebGPUTexture } from "./WebGPUResourceAccess";
import type {
	WebGPUGeometryHandle,
	WebGPUGeometryRegistry,
} from "./WebGPUGeometryRegistry";
import type { WebGPUShadowAtlasAllocator } from "./WebGPUShadowAtlasAllocator";

const WEBGPU_SHADOW_DEPTH_SHADER = /* wgsl */ `
const EPSILON: f32 = 1e-6;

struct ShadowUniforms {
	mvp: mat4x4<f32>,
}

struct AnimationParams {
	jointCount: f32,
	morphTargetCount: f32,
	_pad0: f32,
	_pad1: f32,
}

struct ShadowVertexInput {
	@location(0) position: vec3<f32>,
	@location(5) joints0: vec4<f32>,
	@location(6) weights0: vec4<f32>,
	@location(7) joints1: vec4<f32>,
	@location(8) weights1: vec4<f32>,
}

@group(0) @binding(0) var<uniform> shadow: ShadowUniforms;
@group(1) @binding(0) var<uniform> animationParams: AnimationParams;
@group(1) @binding(1) var<storage, read> jointMatrices: array<mat4x4<f32>>;
@group(1) @binding(2) var<storage, read> morphWeights: array<f32>;
@group(1) @binding(3) var<storage, read> morphPositionDeltas: array<vec4<f32>>;

fn applyMorphPosition(
	basePosition: vec3<f32>,
	vertexIndex: u32,
	morphTargetCount: u32
) -> vec3<f32> {
	if (morphTargetCount == 0u) {
		return basePosition;
	}

	let morphDeltaCount = arrayLength(&morphPositionDeltas);
	let morphWeightCount = arrayLength(&morphWeights);
	if (morphDeltaCount == 0u || morphWeightCount == 0u) {
		return basePosition;
	}

	let vertexCount = max(morphDeltaCount / morphTargetCount, 1u);
	var position = basePosition;
	for (
		var targetIndex: u32 = 0u;
		targetIndex < morphTargetCount;
		targetIndex = targetIndex + 1u
	) {
		if (targetIndex >= morphWeightCount) {
			continue;
		}

		let weight = morphWeights[targetIndex];
		if (abs(weight) <= EPSILON) {
			continue;
		}

		let deltaIndex = targetIndex * vertexCount + vertexIndex;
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
	jointCount: u32
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
		if (jointIndex >= jointCount || jointIndex >= matrixCount) {
			continue;
		}

		let skinMatrix = jointMatrices[jointIndex];
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
	@builtin(vertex_index) vertexIndex: u32
) -> @builtin(position) vec4<f32> {
	let morphTargetCount = u32(animationParams.morphTargetCount + 0.5);
	let jointCount = u32(animationParams.jointCount + 0.5);

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

	let morphedPosition = applyMorphPosition(
		input.position,
		vertexIndex,
		morphTargetCount
	);
	let skinnedPosition = applySkinningPosition(
		morphedPosition,
		joints,
		weights,
		jointCount
	);

	return shadow.mvp * vec4<f32>(skinnedPosition, 1.0);
}
`;

interface ShadowRenderSlot {
	shadowMap: ShadowMap;
	tileX: number;
	tileY: number;
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
	private _uniformData = new Float32Array(16);
	private _shaderModule: GPUShaderModule | null = null;
	private _bindGroupLayout: GPUBindGroupLayout | null = null;
	private _animationBindGroupLayout: GPUBindGroupLayout | null = null;
	private _pipelineLayout: GPUPipelineLayout | null = null;
	private _pipeline: GPURenderPipeline | null = null;
	private _drawUniformBuffers: GPUBuffer[] = [];
	private _drawBindGroups: GPUBindGroup[] = [];
	private _drawResourceCursor = 0;
	private _frustum = new Frustum();
	private _animationBindings = new Map<string, ShadowAnimationBindingEntry>();
	private _fallbackStorageBuffer: GPUBuffer | null = null;
	private _frameId = 0;

	constructor(
		backend: WebGPUBackend,
		geometryRegistry: WebGPUGeometryRegistry,
		shadowAtlases: WebGPUShadowAtlasAllocator
	) {
		this._backend = backend;
		this._geometryRegistry = geometryRegistry;
		this._shadowAtlases = shadowAtlases;
	}

	public render(context: FrameContext): void {
		if (!context.features.enableShadows) return;

		const frame = context.scene;
		const shadowMaps = context.shadowMaps;
		const slots = this._collectShadowSlots(frame, shadowMaps);
		const maxShadowSize = getMaxShadowSize(slots);
		const atlasTileSize = Math.max(1, maxShadowSize);
		const atlasTexture =
			this._shadowAtlases.ensureAtlasForTileSize(atlasTileSize);
		const atlasView = getWebGPUTexture(atlasTexture).view;
		if (!atlasView) return;

		this._ensurePipelineResources();
		if (
			!this._pipeline ||
			!this._bindGroupLayout ||
			!this._animationBindGroupLayout
		) {
			return;
		}

		this._frameId++;
		this._drawResourceCursor = 0;

		const commandEncoder = this._backend.device.createCommandEncoder({
			label: "WebGPUShadowEncoder",
		});
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
			const viewportX = slot.tileX * atlasTileSize;
			const viewportY = slot.tileY * atlasTileSize;
			passEncoder.setViewport(
				viewportX,
				viewportY,
				shadowMapSize,
				shadowMapSize,
				0,
				1
			);
			passEncoder.setScissorRect(
				viewportX,
				viewportY,
				shadowMapSize,
				shadowMapSize
			);
			Matrix4.multiply(
				this._depthRemapMatrix,
				slot.shadowMap.viewProjectionMatrix!,
				this._shadowViewProjectionMatrix
			);

			// Update frustum for current shadow map
			this._frustum.setFromMatrix(slot.shadowMap.viewProjectionMatrix!);

			this._drawShadowCasters(
				passEncoder,
				frame.shadowCasterPackets,
				this._shadowViewProjectionMatrix,
				context
			);
		}

		passEncoder.end();
		this._backend.queue.submit([commandEncoder.finish()]);
		this._trimDrawResources();
		this._trimAnimationResources();
	}

	private _drawShadowCasters(
		passEncoder: GPURenderPassEncoder,
		packets: DrawPacket[],
		viewProjectionMatrix: Matrix4,
		context: FrameContext
	): void {
		for (const packet of packets) {
			// Per-light Frustum Culling
			if (
				!this._frustum.intersectsSphere(
					packet.worldBounds.center,
					packet.worldBounds.radius
				)
			) {
				continue;
			}

			const geometry = this._geometryRegistry.getGeometry(packet.primitive);
			const vertexBuffer = (
				geometry.vertexBuffer as { _gpuResource?: GPUBuffer }
			)._gpuResource;
			const indexBuffer = (geometry.indexBuffer as { _gpuResource?: GPUBuffer })
				._gpuResource;
			if (!vertexBuffer || !indexBuffer) continue;

			Matrix4.multiply(
				viewProjectionMatrix,
				packet.worldMatrix,
				this._mvpMatrix
			);
			const shadowBindGroup = this._nextDrawBindGroup();
			if (!shadowBindGroup) continue;
			this._writeUniformMatrix(this._mvpMatrix, shadowBindGroup.buffer);

			const animationBindGroup = this._resolveAnimationBinding(
				packet,
				geometry,
				context
			);
			if (!animationBindGroup) continue;

			passEncoder.setVertexBuffer(0, vertexBuffer);
			passEncoder.setIndexBuffer(indexBuffer, "uint32");
			passEncoder.setBindGroup(0, shadowBindGroup.group);
			passEncoder.setBindGroup(1, animationBindGroup);
			passEncoder.drawIndexed(geometry.indexCount);
		}
	}

	private _resolveAnimationBinding(
		packet: DrawPacket,
		geometry: WebGPUGeometryHandle,
		context: FrameContext
	): GPUBindGroup | null {
		if (!this._animationBindGroupLayout || !this._fallbackStorageBuffer) {
			return null;
		}

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
			entry.jointBuffer = this._backend.device.createBuffer({
				label: `WebGPUShadowJointBuffer_${key}`,
				size: jointCapacity * 16 * 4,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
			});
			entry.jointCapacity = jointCapacity;
			needsRebind = true;
		}
		if (morphCapacity > entry.morphCapacity) {
			entry.morphWeightBuffer.destroy();
			entry.morphWeightBuffer = this._backend.device.createBuffer({
				label: `WebGPUShadowMorphWeightBuffer_${key}`,
				size: morphCapacity * 4,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
			});
			entry.morphCapacity = morphCapacity;
			needsRebind = true;
		}

		this._backend.queue.writeBuffer(
			entry.paramsBuffer,
			0,
			new Float32Array([jointCount, morphCount, 0, 0])
		);
		if (jointCount > 0 && state.jointMatrices) {
			this._backend.queue.writeBuffer(
				entry.jointBuffer,
				0,
				state.jointMatrices.subarray(
					0,
					jointCount * 16
				) as Float32Array<ArrayBuffer>
			);
		}
		if (morphCount > 0 && state.morphWeights) {
			this._backend.queue.writeBuffer(
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
			entry.bindGroup = this._backend.device.createBindGroup({
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
			(context.transient.get(ANIMATION_WEBGPU_JOINT_MATRICES_KEY) as
				| JointMatrixMap
				| undefined) ?? null;
		const runtimeMorphMap =
			(context.transient.get(ANIMATION_WEBGPU_MORPH_WEIGHTS_KEY) as
				| MorphWeightMap
				| undefined) ?? null;
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

	private _writeUniformMatrix(matrix: Matrix4, buffer: GPUBuffer): void {
		const elements = matrix.elements;
		const data = this._uniformData;
		data[0] = elements[0][0];
		data[1] = elements[1][0];
		data[2] = elements[2][0];
		data[3] = elements[3][0];
		data[4] = elements[0][1];
		data[5] = elements[1][1];
		data[6] = elements[2][1];
		data[7] = elements[3][1];
		data[8] = elements[0][2];
		data[9] = elements[1][2];
		data[10] = elements[2][2];
		data[11] = elements[3][2];
		data[12] = elements[0][3];
		data[13] = elements[1][3];
		data[14] = elements[2][3];
		data[15] = elements[3][3];
		this._backend.queue.writeBuffer(buffer, 0, data);
	}

	private _collectShadowSlots(
		scene: PreparedScene,
		shadowMaps: Map<ShadowCastingLight, ShadowMap>
	): ShadowRenderSlot[] {
		const slots: ShadowRenderSlot[] = [];
		let directionalIndex = 0;
		let spotIndex = 0;

		for (const light of scene.lights) {
			if (light.type === LightType.Directional) {
				if (directionalIndex >= WEBGPU_MAX_DIRECTIONAL_LIGHTS) continue;
				if (isShadowCastingLight(light)) {
					const shadowMap = shadowMaps.get(light);
					if (shadowMap?.viewProjectionMatrix) {
						slots.push({
							shadowMap,
							tileX: directionalIndex,
							tileY: 0,
						});
					}
				}
				directionalIndex++;
				continue;
			}

			if (light.type === LightType.Spot) {
				if (spotIndex >= WEBGPU_MAX_SPOT_LIGHTS) continue;
				if (isShadowCastingLight(light)) {
					const shadowMap = shadowMaps.get(light);
					if (shadowMap?.viewProjectionMatrix) {
						slots.push({
							shadowMap,
							tileX: spotIndex,
							tileY: 1,
						});
					}
				}
				spotIndex++;
			}
		}

		return slots;
	}

	private _ensurePipelineResources(): void {
		if (
			this._pipeline &&
			this._bindGroupLayout &&
			this._animationBindGroupLayout &&
			this._fallbackStorageBuffer
		) {
			return;
		}

		const device = this._backend.device;
		if (!this._shaderModule) {
			this._shaderModule = device.createShaderModule({
				label: "WebGPUShadowDepthShader",
				code: WEBGPU_SHADOW_DEPTH_SHADER,
			});
		}

		if (!this._bindGroupLayout) {
			this._bindGroupLayout = device.createBindGroupLayout({
				label: "WebGPUShadowDepthBindGroupLayout",
				entries: [
					{
						binding: 0,
						visibility: GPUShaderStage.VERTEX,
						buffer: { type: "uniform" },
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
			this._backend.queue.writeBuffer(
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
					module: this._shaderModule,
					entryPoint: "vsMain",
					buffers: [
						{
							arrayStride: WEBGPU_SCENE_VERTEX_STRIDE,
							attributes: [
								{
									shaderLocation: 0,
									offset: 0,
									format: "float32x3",
								},
								{
									shaderLocation: 5,
									offset: 56,
									format: "float32x4",
								},
								{
									shaderLocation: 6,
									offset: 72,
									format: "float32x4",
								},
								{
									shaderLocation: 7,
									offset: 88,
									format: "float32x4",
								},
								{
									shaderLocation: 8,
									offset: 104,
									format: "float32x4",
								},
							],
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
					depthWriteEnabled: true,
					depthCompare: "less",
				},
			});
		}
	}

	private _nextDrawBindGroup(): {
		buffer: GPUBuffer;
		group: GPUBindGroup;
	} | null {
		if (!this._bindGroupLayout) return null;

		const slot = this._drawResourceCursor++;
		let buffer = this._drawUniformBuffers[slot];
		let group = this._drawBindGroups[slot];

		if (!buffer) {
			buffer = this._backend.device.createBuffer({
				label: `WebGPUShadowDepthUniforms_${slot}`,
				size: 16 * 4,
				usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
			});
			this._drawUniformBuffers[slot] = buffer;
		}

		if (!group) {
			group = this._backend.device.createBindGroup({
				label: `WebGPUShadowDepthBindGroup_${slot}`,
				layout: this._bindGroupLayout,
				entries: [
					{
						binding: 0,
						resource: { buffer },
					},
				],
			});
			this._drawBindGroups[slot] = group;
		}

		return { buffer, group };
	}

	private _createAnimationBindingEntry(
		key: string
	): ShadowAnimationBindingEntry {
		return {
			paramsBuffer: this._backend.device.createBuffer({
				label: `WebGPUShadowAnimationParams_${key}`,
				size: 16,
				usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
			}),
			jointBuffer: this._backend.device.createBuffer({
				label: `WebGPUShadowJointBuffer_${key}`,
				size: 16 * 4,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
			}),
			morphWeightBuffer: this._backend.device.createBuffer({
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

	private _trimDrawResources(): void {
		const used = this._drawResourceCursor;
		const allocated = this._drawUniformBuffers.length;
		// Trim when usage drops below 1/3 of allocated capacity and there
		// are at least 16 excess slots, to avoid trimming on small
		// fluctuations.
		if (allocated > 16 && used < allocated / 3) {
			const keep = Math.max(used, 8);
			for (let i = keep; i < allocated; i++) {
				this._drawUniformBuffers[i]?.destroy();
			}
			this._drawUniformBuffers.length = keep;
			this._drawBindGroups.length = keep;
		}
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
		maxSize = Math.max(maxSize, slot.shadowMap.size | 0);
	}
	return maxSize;
}
