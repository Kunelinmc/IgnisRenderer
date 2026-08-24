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
import type {
	PreparedShadowLight,
	PreparedShadowSlice,
	ShadowFramePlan,
} from "../../lights/shadows/ShadowFramePlan";
import {
	ANIMATION_JOINT_MATRICES_KEY,
	ANIMATION_MORPH_WEIGHTS_KEY,
} from "../../simulation/animation/types";
import { DEFAULT_PRIMITIVE_DRAW_TOPOLOGY } from "../../core/types";
import { resolveMaterialShadowTransmittance } from "../../materials/transparency";
import type { WebGPUDeviceResourceHost } from "./WebGPUDeviceResourceHost";
import type { ICommandEncoder } from "../ICommandEncoder";
import { ShaderSource } from "../../shaders/ShaderSource";
import {
	MAX_DIRECTIONAL_LIGHTS,
	MAX_SPOT_LIGHTS,
} from "../constants";
import { WEBGPU_SHADOW_ATLAS_COLUMNS } from "./constants";
import {
	getWebGPUBuffer,
	getWebGPURenderPipeline,
	getWebGPUTexture,
} from "./WebGPUResourceAccess";
import { tryGetNativeWebGPUCommandEncoder } from "./WebGPUCommandEncoder";
import { TextureFormat } from "../../core/TextureFormat";
import {
	PrimitiveTopology,
	type IRenderPipeline,
	type IRenderTexture,
	type IShaderModule,
} from "../types";
import type {
	WebGPUGeometryHandle,
	WebGPUGeometryRegistry,
} from "./WebGPUGeometryRegistry";
import type { WebGPUAnimationPayloadPool } from "./WebGPUAnimationPayloadPool";
import type { WebGPUShadowAtlasAllocator } from "./WebGPUShadowAtlasAllocator";
import type {
	WebGPUPagedShadowIndirectRenderResources,
	WebGPUPagedShadowResidentPage,
} from "./WebGPUPagedShadowTechnique";

interface ShadowRenderSlot {
	prepared: PreparedShadowLight;
	slice: PreparedShadowSlice;
	sliceIndex: number;
	tileX: number;
	tileY: number;
	localTileX: number;
	localTileY: number;
	localTileSpan: number;
	atlasBaseSize: number;
}

interface ShadowAnimationBindingEntry {
	bindGroup: GPUBindGroup | null;
	payloadGeneration: number;
	morphPositionBuffer: GPUBuffer | null;
	lastUsedFrame: number;
}

interface ShadowDrawCandidate {
	packet: DrawPacket;
	geometry: WebGPUGeometryHandle;
	vertexBindings: readonly { slot: number; buffer: GPUBuffer }[];
	indexBuffer: GPUBuffer;
	indexFormat: GPUIndexFormat;
}

interface ShadowInstancedDrawBatch {
	candidate: ShadowDrawCandidate;
	animationBindGroup: GPUBindGroup;
	firstInstance: number;
	instanceCount: number;
}

interface InstanceBufferGroup {
	mvpBuffer: GPUBuffer;
	metaBuffer: GPUBuffer;
	transmittanceBuffer: GPUBuffer;
	bindGroup: GPUBindGroup;
	capacity: number;
}

const SHADOW_INSTANCE_DATA_UINTS = 12;
const DRAW_INDEXED_INDIRECT_UINTS = 5;

/**
 * Encodes shadow-caster draws into targets selected by a shadow technique.
 *
 * This renderer owns shared draw pipelines and transient instance resources. It
 * does not own atlas allocation, paged residency, or frame-graph policy.
 */
export class WebGPUShadowCasterRenderer {
	private _backend: WebGPUDeviceResourceHost;
	private _geometryRegistry: WebGPUGeometryRegistry;
	private _animationPayloads: WebGPUAnimationPayloadPool;
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
	private _pipelines = new Map<string, IRenderPipeline>();
	private _transmittancePipelines = new Map<string, IRenderPipeline>();
	private _pagedClearShaderModule: IShaderModule | null = null;
	private _pagedClearShaderModulePromise: Promise<IShaderModule> | null = null;
	private _pagedClearBindGroupLayout: GPUBindGroupLayout | null = null;
	private _pagedClearPipelineLayout: GPUPipelineLayout | null = null;
	private _pagedClearPipeline: IRenderPipeline | null = null;
	private _pagedClearParamsBuffer: GPUBuffer | null = null;
	private _opaqueBufferGroups: InstanceBufferGroup[] = [];
	private _transmittanceBufferGroups: InstanceBufferGroup[] = [];
	private _pagedOpaqueBufferGroups: InstanceBufferGroup[] = [];
	private _pagedTransmittanceBufferGroups: InstanceBufferGroup[] = [];
	private _frustum = new Frustum();
	private _animationBindings = new Map<string, ShadowAnimationBindingEntry>();
	private _staticAnimationBindGroup: GPUBindGroup | null = null;
	private _frameId = 0;
	private _instanceTransmittanceData = new Float32Array(0);

	constructor(
		backend: WebGPUDeviceResourceHost,
		geometryRegistry: WebGPUGeometryRegistry,
		animationPayloads: WebGPUAnimationPayloadPool
	) {
		this._backend = backend;
		this._geometryRegistry = geometryRegistry;
		this._animationPayloads = animationPayloads;
	}

	/** @internal Encodes the atlas technique's caster work. */
	public async renderAtlas(
		context: FrameContext,
		shadowAtlases: WebGPUShadowAtlasAllocator,
		frameEncoder?: ICommandEncoder | null
	): Promise<void> {
		if (!context.features.enableShadows) return;

		const frame = context.scene;
		const slots = this._collectShadowSlots(frame, context.shadowPlan);
		if (slots.length === 0) return;
		const maxShadowSize = getMaxShadowSize(slots);
		const requestedAtlasTileSize = Math.max(1, maxShadowSize);
		const atlasTexture =
			shadowAtlases.ensureAtlasForTileSize(requestedAtlasTileSize);
		const transmittanceAtlasTexture = shadowAtlases.transmittanceAtlas;
		const atlasTileSize = Math.max(1, shadowAtlases.tileSize);
		const atlasView = getWebGPUTexture(atlasTexture).view;
		const transmittanceAtlasView =
			transmittanceAtlasTexture ?
				getWebGPUTexture(transmittanceAtlasTexture).view
			:	null;
		if (!atlasView || !transmittanceAtlasView) return;

		await this._ensurePipelineResources();
		if (
			!this._shaderModule ||
			!this._pipelineLayout ||
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
		await this._prepareGeometryPipelines(drawCandidates, false);
		await this._prepareGeometryPipelines(transmitterCandidates, true);
		const animationBindingCache = new Map<string, GPUBindGroup | null>();

		const { commandEncoder, submitAtEnd } =
			this._resolveShadowCommandEncoder(frameEncoder);

		const passEncoder = commandEncoder.beginRenderPass({
			label: "WebGPUAtlasShadowPass",
			colorAttachments: [],
			depthStencilAttachment: {
				view: atlasView,
				depthClearValue: 1,
				depthLoadOp: "clear",
				depthStoreOp: "store",
			},
		});

		let slotIndex = 0;
		for (const slot of slots) {
			const shadowMapSize = Math.max(1, slot.slice.resolution | 0);
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
			Matrix4.multiply(
				this._depthRemapMatrix,
				slot.slice.viewProjection,
				this._shadowViewProjectionMatrix
			);

			// Update frustum for current shadow map
			this._frustum.setFromMatrix(slot.slice.viewProjection);

			this._drawShadowCasters(
				passEncoder,
				drawCandidates,
				this._shadowViewProjectionMatrix,
				context,
				animationBindingCache,
				slotIndex
			);
			slotIndex++;
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
		slotIndex = 0;
		for (const slot of slots) {
			const shadowMapSize = Math.max(1, slot.slice.resolution | 0);
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
				slot.slice.viewProjection,
				this._shadowViewProjectionMatrix
			);
			this._frustum.setFromMatrix(slot.slice.viewProjection);
			this._drawShadowTransmitters(
				transmittancePassEncoder,
				transmitterCandidates,
				this._shadowViewProjectionMatrix,
				context,
				animationBindingCache,
				slotIndex
			);
			slotIndex++;
		}
		transmittancePassEncoder.end();
		if (submitAtEnd) {
			this._requireBackendQueue().submit([commandEncoder.finish()]);
		}
		this._trimAnimationResources();
	}

	/**
	 * @internal WebGPU paged shadow depth renderer.
	 */
	public async renderPagedDepthPages(
		context: FrameContext,
		pages: readonly WebGPUPagedShadowResidentPage[],
		physicalDepthAtlas: IRenderTexture,
		frameEncoder?: ICommandEncoder | null,
		shadowCasterPackets: readonly DrawPacket[] = context.scene.shadowCasterPackets
	): Promise<void> {
		if (!context.features.enableShadows || pages.length <= 0) {
			return;
		}
		const atlasView = getWebGPUTexture(physicalDepthAtlas).view;
		if (!atlasView) {
			return;
		}

		await this._ensurePipelineResources();
		if (
			!this._shaderModule ||
			!this._pipelineLayout ||
			!this._bindGroupLayout ||
			!this._animationBindGroupLayout
		) {
			return;
		}

		this._frameId++;
		const drawCandidates = this._collectShadowDrawCandidates(shadowCasterPackets);
		await this._prepareGeometryPipelines(drawCandidates, false);
		const animationBindingCache = new Map<string, GPUBindGroup | null>();
		const { commandEncoder, submitAtEnd } =
			this._resolveShadowCommandEncoder(frameEncoder);
		const passEncoder = commandEncoder.beginRenderPass({
			label: "WebGPUPagedShadowDepthPass",
			colorAttachments: [],
			depthStencilAttachment: {
				view: atlasView,
				depthLoadOp: "load",
				depthStoreOp: "store",
			},
		});

		const atlasSize = Math.max(
			1,
			Math.floor(
				physicalDepthAtlas.width ??
				pages.reduce(
					(max, page) =>
						Math.max(max, page.viewportX + page.viewportSize, page.viewportY + page.viewportSize),
					1
				)
			)
		);
		passEncoder.setViewport(0, 0, atlasSize, atlasSize, 0, 1);
		passEncoder.setScissorRect(0, 0, atlasSize, atlasSize);
		this._drawPagedShadowCasters(
			passEncoder,
			drawCandidates,
			pages,
			atlasSize,
			context,
			animationBindingCache
		);
		passEncoder.end();
		if (submitAtEnd) {
			this._requireBackendQueue().submit([commandEncoder.finish()]);
		}
		this._trimAnimationResources();
	}

	/**
	 * @internal WebGPU GPU-driven paged shadow depth renderer.
	 */
	public async renderPagedDepthIndirect(
		context: FrameContext,
		resources: WebGPUPagedShadowIndirectRenderResources,
		frameEncoder?: ICommandEncoder | null,
		shadowCasterPackets: readonly DrawPacket[] = context.scene.shadowCasterPackets
	): Promise<void> {
		if (
			!context.features.enableShadows ||
			resources.drawInstanceCapacity <= 0
		) {
			return;
		}
		const atlasView = getWebGPUTexture(resources.physicalDepthAtlas).view;
		if (!atlasView) {
			return;
		}

		await this._ensurePipelineResources();
		if (
			!this._shaderModule ||
			!this._pipelineLayout ||
			!this._bindGroupLayout ||
			!this._animationBindGroupLayout
		) {
			return;
		}

		this._frameId++;
		const indirectCandidates = this._collectShadowDrawCandidates(
			shadowCasterPackets
		);
		await this._prepareGeometryPipelines(indirectCandidates, false);
		// Update clear params buffer on queue
		const paramsArray = new Uint32Array([
			resources.physicalPageCount,
			resources.pageSize,
			resources.physicalGridSize,
			0, // pad
		]);
		this._requireBackendQueue().writeBuffer(
			this._pagedClearParamsBuffer!,
			0,
			paramsArray
		);

		const { commandEncoder, submitAtEnd } =
			this._resolveShadowCommandEncoder(frameEncoder);
		const passEncoder = commandEncoder.beginRenderPass({
			label: "WebGPUPagedShadowDepthIndirectPass",
			colorAttachments: [],
			depthStencilAttachment: {
				view: atlasView,
				depthLoadOp: "load",
				depthStoreOp: "store",
			},
		});

		const atlasSize = Math.max(1, resources.physicalAtlasSize);
		passEncoder.setViewport(0, 0, atlasSize, atlasSize, 0, 1);
		passEncoder.setScissorRect(0, 0, atlasSize, atlasSize);

		// Tombstones can dirty pages even when there are no current casters.
		if (this._pagedClearPipeline && this._pagedClearBindGroupLayout) {
			const clearBindGroup = this._requireBackendDevice().createBindGroup({
				label: "WebGPUPagedShadowClearBindGroup",
				layout: this._pagedClearBindGroupLayout,
				entries: [
					{
						binding: 0,
						resource: { buffer: this._pagedClearParamsBuffer! },
					},
					{
						binding: 1,
						resource: { buffer: getWebGPUBuffer(resources.dirtyPhysicalPages) },
					},
				],
			});
			passEncoder.setPipeline(getWebGPURenderPipeline(this._pagedClearPipeline));
			passEncoder.setBindGroup(0, clearBindGroup);
			passEncoder.drawIndirect(
				getWebGPUBuffer(resources.clearDrawIndirectArgsBuffer),
				0
			);
		}

		if (shadowCasterPackets.length > 0 && resources.drawCandidateCount > 0) {
			const animationBindingCache = new Map<string, GPUBindGroup | null>();
			const bindGroup = this._requireBackendDevice().createBindGroup({
				label: "WebGPUPagedShadowDepthIndirectBindGroup",
				layout: this._bindGroupLayout,
				entries: [
					{
						binding: 0,
						resource: { buffer: getWebGPUBuffer(resources.drawMvpBuffer) },
					},
					{
						binding: 1,
						resource: {
							buffer: getWebGPUBuffer(resources.drawInstanceMetaBuffer),
						},
					},
					{
						binding: 2,
						resource: {
							buffer: getWebGPUBuffer(resources.drawTransmittanceBuffer),
						},
					},
				],
			});
			passEncoder.setBindGroup(0, bindGroup);
			const indirectBuffer = getWebGPUBuffer(resources.drawIndirectArgsBuffer);
			const candidateLimit = Math.min(
				shadowCasterPackets.length,
				resources.drawCandidateCount
			);
			for (let candidateIndex = 0; candidateIndex < candidateLimit; candidateIndex++) {
				const candidate = this._collectShadowDrawCandidate(
					shadowCasterPackets[candidateIndex]
				);
				if (!candidate) {
					continue;
				}
				const pipeline = this._pipelines.get(candidate.geometry.shadowLayoutKey);
				if (!pipeline) continue;
				passEncoder.setPipeline(getWebGPURenderPipeline(pipeline));
				const packet = candidate.packet;
				if (!animationBindingCache.has(packet.id)) {
					animationBindingCache.set(
						packet.id,
						this._resolveAnimationBinding(packet, candidate.geometry, context)
					);
				}
				const animationBindGroup =
					animationBindingCache.get(packet.id) ?? null;
				if (!animationBindGroup) {
					continue;
				}
				bindShadowCandidate(passEncoder, candidate);
				passEncoder.setBindGroup(1, animationBindGroup);
				passEncoder.drawIndexedIndirect(
					indirectBuffer,
					candidateIndex * DRAW_INDEXED_INDIRECT_UINTS * 4
				);
			}
		}
		passEncoder.end();
		if (submitAtEnd) {
			this._requireBackendQueue().submit([commandEncoder.finish()]);
		}
		this._trimAnimationResources();
	}

	public onShaderRuntimeChanged(): void {
		this._destroyManagedResource(this._shaderModule);
		for (const pipeline of this._pipelines.values()) {
			this._destroyManagedResource(pipeline);
		}
		for (const pipeline of this._transmittancePipelines.values()) {
			this._destroyManagedResource(pipeline);
		}
		this._pipelines.clear();
		this._transmittancePipelines.clear();
		this._destroyManagedResource(this._pagedClearShaderModule);
		this._destroyManagedResource(this._pagedClearPipeline);
		this._shaderModule = null;
		this._shaderModulePromise = null;
		this._pagedClearShaderModule = null;
		this._pagedClearShaderModulePromise = null;
		this._pagedClearPipeline = null;
	}

	public async warmup(): Promise<void> {
		await this._ensurePipelineResources();
	}

	public destroy(): void {
		this._destroyManagedResource(this._shaderModule);
		for (const pipeline of this._pipelines.values()) {
			this._destroyManagedResource(pipeline);
		}
		for (const pipeline of this._transmittancePipelines.values()) {
			this._destroyManagedResource(pipeline);
		}
		this._pipelines.clear();
		this._transmittancePipelines.clear();
		this._destroyManagedResource(this._pagedClearShaderModule);
		this._destroyManagedResource(this._pagedClearPipeline);
		this._destroyManagedResource(this._pagedClearParamsBuffer);
		this._shaderModule = null;
		this._shaderModulePromise = null;
		this._bindGroupLayout = null;
		this._animationBindGroupLayout = null;
		this._staticAnimationBindGroup = null;
		this._pipelineLayout = null;
		this._pagedClearShaderModule = null;
		this._pagedClearShaderModulePromise = null;
		this._pagedClearBindGroupLayout = null;
		this._pagedClearPipelineLayout = null;
		this._pagedClearPipeline = null;
		this._pagedClearParamsBuffer = null;

		for (const group of this._opaqueBufferGroups) {
			if (group) {
				group.mvpBuffer.destroy();
				group.metaBuffer.destroy();
				group.transmittanceBuffer.destroy();
			}
		}
		this._opaqueBufferGroups = [];

		for (const group of this._transmittanceBufferGroups) {
			if (group) {
				group.mvpBuffer.destroy();
				group.metaBuffer.destroy();
				group.transmittanceBuffer.destroy();
			}
		}
		this._transmittanceBufferGroups = [];

		for (const group of this._pagedOpaqueBufferGroups) {
			if (group) {
				group.mvpBuffer.destroy();
				group.metaBuffer.destroy();
				group.transmittanceBuffer.destroy();
			}
		}
		this._pagedOpaqueBufferGroups = [];

		for (const group of this._pagedTransmittanceBufferGroups) {
			if (group) {
				group.mvpBuffer.destroy();
				group.metaBuffer.destroy();
				group.transmittanceBuffer.destroy();
			}
		}
		this._pagedTransmittanceBufferGroups = [];

		this._instanceMvpData = new Float32Array(0);
		this._instanceMetaData = new Uint32Array(0);
		this._instanceTransmittanceData = new Float32Array(0);
		this._animationBindings.clear();
		this._frameId = 0;
	}

	private _drawPagedShadowCasters(
		passEncoder: GPURenderPassEncoder,
		drawCandidates: ShadowDrawCandidate[],
		pages: readonly WebGPUPagedShadowResidentPage[],
		atlasSize: number,
		context: FrameContext,
		animationBindingCache: Map<string, GPUBindGroup | null>
	): void {
		const { batches: drawBatches, bindGroup } =
			this._buildPagedShadowDrawBatches(
				drawCandidates,
				pages,
				atlasSize,
				context,
				animationBindingCache
			);
		if (drawBatches.length === 0 || !bindGroup) {
			return;
		}

		passEncoder.setBindGroup(0, bindGroup);
		for (const batch of drawBatches) {
			const pipeline = this._pipelines.get(
				batch.candidate.geometry.shadowLayoutKey
			);
			if (!pipeline) continue;
			passEncoder.setPipeline(getWebGPURenderPipeline(pipeline));
			bindShadowCandidate(passEncoder, batch.candidate);
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

	private _buildPagedShadowDrawBatches(
		drawCandidates: ShadowDrawCandidate[],
		pages: readonly WebGPUPagedShadowResidentPage[],
		atlasSize: number,
		context: FrameContext,
		animationBindingCache: Map<string, GPUBindGroup | null>
	): { batches: ShadowInstancedDrawBatch[]; bindGroup: GPUBindGroup | null } {
		const drawBatches: ShadowInstancedDrawBatch[] = [];
		let instanceCount = 0;
		for (const candidate of drawCandidates) {
			const packet = candidate.packet;
			if (!animationBindingCache.has(packet.id)) {
				animationBindingCache.set(
					packet.id,
					this._resolveAnimationBinding(packet, candidate.geometry, context)
				);
			}
			const animationBindGroup =
				animationBindingCache.get(packet.id) ?? null;
			if (!animationBindGroup) {
				continue;
			}

			let firstInstance = -1;
			let pageInstanceCount = 0;
			for (const page of pages) {
				this._frustum.setFromMatrix(page.viewProjection);
				if (
					!this._frustum.intersectsSphere(
						packet.worldBounds.center,
						packet.worldBounds.radius
					)
				) {
					continue;
				}

				Matrix4.multiply(
					this._depthRemapMatrix,
					page.viewProjection,
					this._shadowViewProjectionMatrix
				);
				Matrix4.multiply(
					this._shadowViewProjectionMatrix,
					packet.worldMatrix,
					this._mvpMatrix
				);

				this._ensureInstanceDataCapacity(instanceCount + 1);
				const mvpOffset = instanceCount * 16;
				this._setMatrixInArray(this._mvpMatrix, this._instanceMvpData, mvpOffset);
				const transmittanceOffset = instanceCount * 4;
				this._instanceTransmittanceData[transmittanceOffset] = 1;
				this._instanceTransmittanceData[transmittanceOffset + 1] = 1;
				this._instanceTransmittanceData[transmittanceOffset + 2] = 1;
				this._instanceTransmittanceData[transmittanceOffset + 3] = 1;
				const metaOffset = instanceCount * SHADOW_INSTANCE_DATA_UINTS;
				this._setShadowInstanceMetaInArray(
					this._instanceMetaData,
					metaOffset,
					instanceCount,
					0,
					0,
					0,
					0,
					page.viewportX,
					page.viewportY,
					page.viewportSize,
					atlasSize,
					1
				);
				if (firstInstance < 0) {
					firstInstance = instanceCount;
				}
				instanceCount++;
				pageInstanceCount++;
			}

			if (pageInstanceCount <= 0 || firstInstance < 0) {
				continue;
			}
			const lastBatch = drawBatches[drawBatches.length - 1];
			if (
				lastBatch &&
				lastBatch.animationBindGroup === animationBindGroup &&
				lastBatch.candidate.geometry === candidate.geometry &&
				lastBatch.candidate.geometry.indexCount === candidate.geometry.indexCount &&
				lastBatch.firstInstance + lastBatch.instanceCount === firstInstance
			) {
				lastBatch.instanceCount += pageInstanceCount;
				continue;
			}
			drawBatches.push({
				candidate,
				animationBindGroup,
				firstInstance,
				instanceCount: pageInstanceCount,
			});
		}

		if (instanceCount === 0) {
			return { batches: [], bindGroup: null };
		}

		const group = this._upsertShadowInstanceResources(instanceCount, 0, false, true);
		if (!group) {
			return { batches: [], bindGroup: null };
		}
		this._requireBackendQueue().writeBuffer(
			group.mvpBuffer,
			0,
			this._instanceMvpData.subarray(0, instanceCount * 16) as Float32Array<ArrayBuffer>
		);
		this._requireBackendQueue().writeBuffer(
			group.metaBuffer,
			0,
			this._instanceMetaData.subarray(
				0,
				instanceCount * SHADOW_INSTANCE_DATA_UINTS
			) as Uint32Array<ArrayBuffer>
		);
		this._requireBackendQueue().writeBuffer(
			group.transmittanceBuffer,
			0,
			this._instanceTransmittanceData.subarray(
				0,
				instanceCount * 4
			) as Float32Array<ArrayBuffer>
		);
		return { batches: drawBatches, bindGroup: group.bindGroup };
	}

	private _drawShadowCasters(
		passEncoder: GPURenderPassEncoder,
		drawCandidates: ShadowDrawCandidate[],
		viewProjectionMatrix: Matrix4,
		context: FrameContext,
		animationBindingCache: Map<string, GPUBindGroup | null>,
		slotIndex: number
	): void {
		const { batches: drawBatches, bindGroup } = this._buildShadowDrawBatches(
			drawCandidates,
			viewProjectionMatrix,
			context,
			animationBindingCache,
			false,
			slotIndex,
			false
		);
		if (
			drawBatches.length === 0 ||
			!bindGroup
		) {
			return;
		}

		passEncoder.setBindGroup(0, bindGroup);
		for (const batch of drawBatches) {
			const pipeline = this._pipelines.get(
				batch.candidate.geometry.shadowLayoutKey
			);
			if (!pipeline) continue;
			passEncoder.setPipeline(getWebGPURenderPipeline(pipeline));
			bindShadowCandidate(passEncoder, batch.candidate);
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
		animationBindingCache: Map<string, GPUBindGroup | null>,
		slotIndex: number
	): void {
		const { batches: drawBatches, bindGroup } = this._buildShadowDrawBatches(
			drawCandidates,
			viewProjectionMatrix,
			context,
			animationBindingCache,
			true,
			slotIndex,
			true
		);
		if (
			drawBatches.length === 0 ||
			!bindGroup
		) {
			return;
		}

		passEncoder.setBindGroup(0, bindGroup);
		for (const batch of drawBatches) {
			const pipeline = this._transmittancePipelines.get(
				batch.candidate.geometry.shadowLayoutKey
			);
			if (!pipeline) continue;
			passEncoder.setPipeline(getWebGPURenderPipeline(pipeline));
			bindShadowCandidate(passEncoder, batch.candidate);
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
		resolveTransmittance: boolean,
		slotIndex: number,
		isTransmittance: boolean
	): { batches: ShadowInstancedDrawBatch[]; bindGroup: GPUBindGroup | null } {
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
				lastBatch.candidate.geometry === candidate.geometry &&
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

		if (instanceCount === 0) {
			return { batches: [], bindGroup: null };
		}

		const group = this._upsertShadowInstanceResources(
			instanceCount,
			slotIndex,
			isTransmittance
		);
		if (!group) {
			return { batches: [], bindGroup: null };
		}

		this._requireBackendQueue().writeBuffer(
			group.mvpBuffer,
			0,
			this._instanceMvpData.subarray(
				0,
				instanceCount * 16
			) as Float32Array<ArrayBuffer>
		);
		this._requireBackendQueue().writeBuffer(
			group.metaBuffer,
			0,
			this._instanceMetaData.subarray(
				0,
				instanceCount * SHADOW_INSTANCE_DATA_UINTS
			) as Uint32Array<ArrayBuffer>
		);
		this._requireBackendQueue().writeBuffer(
			group.transmittanceBuffer,
			0,
			this._instanceTransmittanceData.subarray(
				0,
				instanceCount * 4
			) as Float32Array<ArrayBuffer>
		);

		return { batches: drawBatches, bindGroup: group.bindGroup };
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

	private _upsertShadowInstanceResources(
		instanceCount: number,
		slotIndex: number,
		isTransmittance: boolean,
		isPaged = false
	): InstanceBufferGroup | null {
		if (!this._bindGroupLayout) {
			return null;
		}
		const device = this._requireBackendDevice();
		const requiredCapacity = Math.max(1, instanceCount);

		const groups = isPaged ?
			(isTransmittance ? this._pagedTransmittanceBufferGroups : this._pagedOpaqueBufferGroups) :
			(isTransmittance ? this._transmittanceBufferGroups : this._opaqueBufferGroups);
		let group = groups[slotIndex];

		if (
			!group ||
			requiredCapacity > group.capacity
		) {
			if (group) {
				group.mvpBuffer.destroy();
				group.metaBuffer.destroy();
				group.transmittanceBuffer.destroy();
			}
			const mvpBuffer = device.createBuffer({
				label: `WebGPUShadowDepthMvpStorage_${isPaged ? "paged_" : ""}${isTransmittance ? "trans" : "opaque"}_${slotIndex}`,
				size: requiredCapacity * 16 * 4,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
			});
			const metaBuffer = device.createBuffer({
				label: `WebGPUShadowDepthInstanceMeta_${isPaged ? "paged_" : ""}${isTransmittance ? "trans" : "opaque"}_${slotIndex}`,
				size: requiredCapacity * SHADOW_INSTANCE_DATA_UINTS * 4,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
			});
			const transmittanceBuffer = device.createBuffer({
				label: `WebGPUShadowTransmittanceStorage_${isPaged ? "paged_" : ""}${isTransmittance ? "trans" : "opaque"}_${slotIndex}`,
				size: requiredCapacity * 4 * 4,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
			});

			const bindGroup = device.createBindGroup({
				label: `WebGPUShadowDepthMvpBindGroup_${isPaged ? "paged_" : ""}${isTransmittance ? "trans" : "opaque"}_${slotIndex}`,
				layout: this._bindGroupLayout,
				entries: [
					{
						binding: 0,
						resource: { buffer: mvpBuffer },
					},
					{
						binding: 1,
						resource: { buffer: metaBuffer },
					},
					{
						binding: 2,
						resource: { buffer: transmittanceBuffer },
					},
				],
			});

			group = {
				mvpBuffer,
				metaBuffer,
				transmittanceBuffer,
				bindGroup,
				capacity: requiredCapacity,
			};
			groups[slotIndex] = group;
		}

		return group;
	}

	private _collectShadowDrawCandidates(
		packets: readonly DrawPacket[]
	): ShadowDrawCandidate[] {
		const candidates: ShadowDrawCandidate[] = [];
		for (const packet of packets) {
			const candidate = this._collectShadowDrawCandidate(packet);
			if (candidate) {
				candidates.push(candidate);
			}
		}
		return candidates;
	}

	private _collectShadowDrawCandidate(
		packet: DrawPacket
	): ShadowDrawCandidate | null {
		if (
			(packet.primitive.topology ?? DEFAULT_PRIMITIVE_DRAW_TOPOLOGY) !==
			DEFAULT_PRIMITIVE_DRAW_TOPOLOGY
		) {
			return null;
		}
		const geometry = this._geometryRegistry.getGeometry(packet.primitive);
		const vertexBindings = geometry.shadowVertexBindings.map((binding) => ({
			slot: binding.slot,
			buffer: getWebGPUBuffer(binding.buffer),
		}));
		const indexBuffer = getWebGPUBuffer(geometry.indexBuffer);
		return {
			packet,
			geometry,
			vertexBindings,
			indexBuffer,
			indexFormat: geometry.indexFormat,
		};
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
		if (!this._animationBindGroupLayout || !this._staticAnimationBindGroup) {
			return null;
		}
		const device = this._requireBackendDevice();
		const jointMap =
			context.transient.get(ANIMATION_JOINT_MATRICES_KEY) ?? null;
		const morphMap =
			context.transient.get(ANIMATION_MORPH_WEIGHTS_KEY) ?? null;
		const payload = this._animationPayloads.getShadowPayload(
			packet,
			geometry,
			jointMap,
			morphMap
		);
		if (payload.generation === 0) {
			this._animationBindings.delete(packet.id);
			return this._staticAnimationBindGroup;
		}

		const key = packet.id;
		let entry = this._animationBindings.get(key);
		if (!entry) {
			entry = {
				bindGroup: null,
				payloadGeneration: -1,
				morphPositionBuffer: null,
				lastUsedFrame: this._frameId,
			};
			this._animationBindings.set(key, entry);
		}
		entry.lastUsedFrame = this._frameId;

		const morphPositionBuffer =
			geometry.morphPositionBuffer ?
				getWebGPUBuffer(geometry.morphPositionBuffer)
			: getWebGPUBuffer(this._animationPayloads.getFallbackStorageBuffer());
		let needsRebind = entry.payloadGeneration !== payload.generation;
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
						resource: { buffer: getWebGPUBuffer(payload.paramsBuffer) },
					},
					{
						binding: 1,
						resource: { buffer: getWebGPUBuffer(payload.jointMatricesBuffer) },
					},
					{
						binding: 2,
						resource: { buffer: getWebGPUBuffer(payload.morphWeightsBuffer) },
					},
					{
						binding: 3,
						resource: { buffer: morphPositionBuffer },
					},
				],
			});
			entry.payloadGeneration = payload.generation;
		}

		return entry.bindGroup;
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
		morphDeltaBaseOffset: number,
		atlasOffsetX: number = 0,
		atlasOffsetY: number = 0,
		atlasPageSize: number = 0,
		atlasSize: number = 0,
		flags: number = 0
	): void {
		target[offset] = instanceBaseOffset >>> 0;
		target[offset + 1] = vertexBaseOffset >>> 0;
		target[offset + 2] = jointBaseOffset >>> 0;
		target[offset + 3] = morphWeightBaseOffset >>> 0;
		target[offset + 4] = morphDeltaBaseOffset >>> 0;
		target[offset + 5] = atlasOffsetX >>> 0;
		target[offset + 6] = atlasOffsetY >>> 0;
		target[offset + 7] = atlasPageSize >>> 0;
		target[offset + 8] = atlasSize >>> 0;
		target[offset + 9] = flags >>> 0;
		target[offset + 10] = 0;
		target[offset + 11] = 0;
	}

	private _collectShadowSlots(
		scene: PreparedScene,
		plan: ShadowFramePlan
	): ShadowRenderSlot[] {
		const slots: ShadowRenderSlot[] = [];
		const atlasLightIds = new Set(
			plan.jobs
				.filter((job) =>
					job.technique === "atlas" || job.technique === "atlas-fallback"
				)
				.map((job) => plan.lights[job.lightIndex]?.lightId)
				.filter((lightId): lightId is string => typeof lightId === "string")
		);
		const atlasColumns = Math.max(1, WEBGPU_SHADOW_ATLAS_COLUMNS);
		let directionalIndex = 0;
		let spotIndex = 0;

		for (const light of scene.lights) {
			if (light.type === LightType.Directional) {
				if (directionalIndex >= MAX_DIRECTIONAL_LIGHTS) continue;
				if (isShadowCastingLight(light)) {
					const prepared = plan.lights.find((candidate) => candidate.light === light);
					if (!prepared || (prepared.storage === "paged" && !atlasLightIds.has(light.id))) {
						directionalIndex++;
						continue;
					}
					if (prepared.slices.length > 0) {
						const globalTileIndex = directionalIndex;
						const tileX = globalTileIndex % atlasColumns;
						const tileY = Math.floor(globalTileIndex / atlasColumns);
						const isCSM = prepared.effectiveTechnique === "cascaded";
						const maxSlices = Math.min(
							prepared.slices.length,
							4
						);
						if (isCSM && maxSlices > 1) {
							for (let sliceIndex = 0; sliceIndex < maxSlices; sliceIndex++) {
								const slice = prepared.slices[sliceIndex];
								if (!slice) continue;
								const localTileX = sliceIndex % 2;
								const localTileY = Math.floor(sliceIndex / 2);
								slots.push({
									prepared,
									slice,
									sliceIndex,
									tileX,
									tileY,
									localTileX,
									localTileY,
									localTileSpan: 2,
									atlasBaseSize: Math.max(1, prepared.effectiveResolution | 0),
								});
							}
						} else {
							const primarySlice = prepared.slices[0];
							slots.push({
								prepared,
								slice: primarySlice,
								sliceIndex: 0,
								tileX,
								tileY,
								localTileX: 0,
								localTileY: 0,
								localTileSpan: 1,
								atlasBaseSize: Math.max(
									1,
									primarySlice?.resolution ?? prepared.effectiveResolution
								),
							});
						}
					}
				}
				directionalIndex++;
				continue;
			}

			if (light.type === LightType.Spot) {
				if (spotIndex >= MAX_SPOT_LIGHTS) continue;
				if (isShadowCastingLight(light)) {
					const prepared = plan.lights.find((candidate) => candidate.light === light);
					if (!prepared || (prepared.storage === "paged" && !atlasLightIds.has(light.id))) {
						spotIndex++;
						continue;
					}
					const primarySlice = prepared.slices[0];
					if (primarySlice) {
						const globalTileIndex =
							MAX_DIRECTIONAL_LIGHTS + spotIndex;
						slots.push({
							prepared,
							slice: primarySlice,
							sliceIndex: 0,
							tileX: globalTileIndex % atlasColumns,
							tileY: Math.floor(globalTileIndex / atlasColumns),
							localTileX: 0,
							localTileY: 0,
							localTileSpan: 1,
							atlasBaseSize: Math.max(
								1,
								primarySlice.resolution ?? prepared.effectiveResolution
							),
						});
					}
				}
				spotIndex++;
			}
		}

		return slots;
	}

	private async _prepareGeometryPipelines(
		candidates: readonly ShadowDrawCandidate[],
		transmittance: boolean
	): Promise<void> {
		const unique = new Map<string, WebGPUGeometryHandle>();
		for (const candidate of candidates) {
			unique.set(candidate.geometry.shadowLayoutKey, candidate.geometry);
		}
		await Promise.all(
			[...unique.values()].map((geometry) =>
				this._getOrCreateGeometryPipeline(geometry, transmittance)
			)
		);
	}

	private async _getOrCreateGeometryPipeline(
		geometry: WebGPUGeometryHandle,
		transmittance: boolean
	): Promise<IRenderPipeline> {
		const cache = transmittance ?
			this._transmittancePipelines : this._pipelines;
		const cached = cache.get(geometry.shadowLayoutKey);
		if (cached) return cached;
		if (!this._shaderModule || !this._pipelineLayout) {
			throw new Error("WebGPU shadow pipeline resources are unavailable.");
		}

		const pipeline = await this._backend.createPipeline({
			label: transmittance ?
				`WebGPUShadowTransmittancePipeline_${geometry.skinProfile}`
			: `WebGPUShadowDepthPipeline_${geometry.skinProfile}`,
			layout: this._pipelineLayout,
			vertex: {
				module: this._shaderModule,
				entryPoint: "vsMain",
				buffers: [...geometry.shadowVertexLayouts],
			},
			fragment: {
				module: this._shaderModule,
				entryPoint: transmittance ? "fsTransmittance" : "fsDepthClip",
				targets: transmittance ? [
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
				] : [],
			},
			primitive: {
				topology: PrimitiveTopology.TriangleList,
				cullMode: "none",
				frontFace: "ccw",
			},
			depthStencil: {
				format: TextureFormat.Depth32Float,
				depthWriteEnabled: !transmittance,
				depthCompare: "less",
			},
		});
		cache.set(geometry.shadowLayoutKey, pipeline);
		return pipeline;
	}

	private async _ensurePipelineResources(): Promise<void> {
		if (
			this._shaderModule &&
			this._pipelineLayout &&
			this._bindGroupLayout &&
			this._animationBindGroupLayout &&
			this._staticAnimationBindGroup
		) {
			return;
		}

		const device = this._requireBackendDevice();
		if (!this._shaderModule) {
			if (!this._shaderModulePromise) {
				this._shaderModulePromise = ShaderSource.load(
					"webgpu.shadow.depth"
				).then((artifact) =>
					this._backend.createShaderModule({
						label: "WebGPUShadowDepthShader",
						code: artifact.source.code,
						sourceMap: artifact.source.sourceMap,
						language: "wgsl",
						stage: "vertex",
						entryPoint: "vsMain",
						sourceKind: "shadow",
					})
				);
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

		if (!this._staticAnimationBindGroup && this._animationBindGroupLayout) {
			const fallback = this._animationPayloads.getStaticShadowPayload();
			const fallbackStorage = getWebGPUBuffer(fallback.jointMatricesBuffer);
			this._staticAnimationBindGroup = device.createBindGroup({
				label: "WebGPUShadowStaticAnimationBinding",
				layout: this._animationBindGroupLayout,
				entries: [
					{
						binding: 0,
						resource: { buffer: getWebGPUBuffer(fallback.paramsBuffer) },
					},
					{ binding: 1, resource: { buffer: fallbackStorage } },
					{ binding: 2, resource: { buffer: fallbackStorage } },
					{ binding: 3, resource: { buffer: fallbackStorage } },
				],
			});
		}

		if (!this._pagedClearShaderModule) {
			if (!this._pagedClearShaderModulePromise) {
				this._pagedClearShaderModulePromise = ShaderSource.load(
					"webgpu.shadow.pagedShadowClear"
				).then((artifact) =>
					this._backend.createShaderModule({
						label: "WebGPUPagedShadowClearShader",
						code: artifact.source.code,
						sourceMap: artifact.source.sourceMap,
						language: "wgsl",
						stage: "vertex",
						entryPoint: "vsMain",
						sourceKind: "shadow",
					})
				);
			}
			try {
				this._pagedClearShaderModule = await this._pagedClearShaderModulePromise;
			} catch (error) {
				this._pagedClearShaderModulePromise = null;
				throw error;
			}
		}

		if (!this._pagedClearBindGroupLayout) {
			this._pagedClearBindGroupLayout = device.createBindGroupLayout({
				label: "WebGPUPagedShadowClearBindGroupLayout",
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
				],
			});
		}

		if (!this._pagedClearPipelineLayout && this._pagedClearBindGroupLayout) {
			this._pagedClearPipelineLayout = device.createPipelineLayout({
				label: "WebGPUPagedShadowClearPipelineLayout",
				bindGroupLayouts: [this._pagedClearBindGroupLayout],
			});
		}

		if (!this._pagedClearParamsBuffer) {
			this._pagedClearParamsBuffer = device.createBuffer({
				label: "WebGPUPagedShadowClearParams",
				size: 16,
				usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
			});
		}

		if (!this._pagedClearPipeline && this._pagedClearShaderModule && this._pagedClearPipelineLayout) {
			this._pagedClearPipeline = await this._backend.createPipeline({
				label: "WebGPUPagedShadowClearPipeline",
				layout: this._pagedClearPipelineLayout,
				vertex: {
					module: this._pagedClearShaderModule,
					entryPoint: "vsMain",
					buffers: [],
				},
				primitive: {
					topology: PrimitiveTopology.TriangleList,
					cullMode: "none",
					frontFace: "ccw",
				},
				depthStencil: {
					format: TextureFormat.Depth32Float,
					depthWriteEnabled: true,
					depthCompare: "always",
				},
			});
		}
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

	private _destroyManagedResource(resource: unknown): void {
		const destroyFn = (resource as { destroy?: () => void } | null)?.destroy;
		if (typeof destroyFn === "function") {
			destroyFn.call(resource);
		}
	}

	private _trimAnimationResources(): void {
		const staleFrame = this._frameId - 120;
		for (const [key, entry] of this._animationBindings) {
			if (entry.lastUsedFrame >= staleFrame) continue;
			this._animationBindings.delete(key);
		}
	}
}

function bindShadowCandidate(
	passEncoder: GPURenderPassEncoder,
	candidate: ShadowDrawCandidate
): void {
	for (const binding of candidate.vertexBindings) {
		passEncoder.setVertexBuffer(binding.slot, binding.buffer);
	}
	passEncoder.setIndexBuffer(candidate.indexBuffer, candidate.indexFormat);
}

function getMaxShadowSize(slots: ShadowRenderSlot[]): number {
	let maxSize = 0;
	for (const slot of slots) {
		maxSize = Math.max(maxSize, slot.atlasBaseSize | 0);
	}
	return maxSize;
}
