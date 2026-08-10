import { clamp } from "../../maths/Common";
import { ParticleBlendMode } from "../../particles";
import type { ParticleRenderBatch } from "../../particles/ParticleRenderBatch";
import { PARTICLE_TRANSIENT_BATCHES_KEY } from "../../pipeline/types";
import type { FrameContext } from "../../pipeline/types";
import { ShaderSource } from "../../shaders/ShaderSource";
import type { ICommandEncoder } from "../ICommandEncoder";
import {
	BufferUsage,
	TextureFormat,
	type IBindingGroup,
	type IRenderBuffer,
	type IRenderPipeline,
	type IRenderTexture,
	type ISampler,
	type IShaderModule,
} from "../types";
import { WEBGPU_PARTICLE_VERTEX_LAYOUTS } from "./bufferLayouts";
import {
	WEBGPU_PARTICLE_BINDING_SAMPLER,
	WEBGPU_PARTICLE_BINDING_TEXTURE,
	WEBGPU_PARTICLE_BINDING_UV_TRANSFORM,
	WEBGPU_PARTICLE_INSTANCE_FLOATS,
	WEBGPU_PARTICLE_INSTANCE_STRIDE,
	WEBGPU_PARTICLE_QUAD_VERTICES,
	WEBGPU_PARTICLE_UV_UNIFORM_SIZE,
} from "./constants";
import type { WebGPUDeviceResourceHost } from "./WebGPUDeviceResourceHost";
import type { WebGPUPipelineLayouts } from "./WebGPUPipelineLayouts";
import type { WebGPUSceneTargetMode } from "./WebGPUPipelineLibrary";
import type {
	WebGPUParticlePassTargets,
	WebGPUParticleRenderOptions,
	WebGPUParticleBillboardRenderer,
	WebGPUPreparedFrameResources,
	WebGPUTextureResourceProvider,
} from "./WebGPUResourceContracts";
import {
	WEBGPU_PARTICLE_DRAW_BATCHES_KEY,
	type WebGPUParticleDrawBatch,
} from "./types";

const WEBGPU_PARTICLE_BINDING_CACHE_MAX_AGE_FRAMES = 120;
const WEBGPU_PARTICLE_BINDING_CACHE_MAX_ENTRIES = 128;

interface WebGPUParticleBindingCacheEntry {
	group: IBindingGroup;
	texture: IRenderTexture;
	sampler: ISampler;
	uvTransformBuffer: IRenderBuffer;
	lastUsedFrame: number;
}

/**
 * Owns WebGPU billboard particle resources and pass recording.
 *
 * @internal Owned by `WebGPUFrameServiceOwner`; applications should use
 * `Renderer.renderFrame()`.
 */
export class WebGPUParticleRenderResources implements WebGPUParticleBillboardRenderer {
	private _particleShaderModule: IShaderModule | null = null;
	private _particleQuadBuffer: IRenderBuffer | null = null;
	private _particleInstanceBuffer: IRenderBuffer | null = null;
	private _particleInstanceCapacity = 0;
	private _particlePipelineAlpha = new Map<string, IRenderPipeline>();
	private _particlePipelineOITAlpha = new Map<string, IRenderPipeline>();
	private _particlePipelineAdditive = new Map<string, IRenderPipeline>();
	private _particleBindingCache = new Map<string, WebGPUParticleBindingCacheEntry>();
	private _frameId = 0;
	private _destroyed = false;

	constructor(
		private readonly _backend: WebGPUDeviceResourceHost,
		private readonly _layouts: WebGPUPipelineLayouts,
		private readonly _textures: WebGPUTextureResourceProvider,
	) {}

	/** Advances particle binding-cache lifetime for a new frame. */
	public beginFrame(): void {
		this._frameId++;
		this._evictParticleBindings();
	}

	/** Compiles the legacy particle resources required by warmup. */
	public async warmup(mode: WebGPUSceneTargetMode): Promise<void> {
		await this._ensureParticleResources(mode, 1, "legacy", 1);
	}

	public async renderParticles(
		encoder: ICommandEncoder,
		context: FrameContext,
		targets: WebGPUParticlePassTargets,
		frameResources: WebGPUPreparedFrameResources,
		mode: WebGPUSceneTargetMode,
		options: WebGPUParticleRenderOptions = {},
	): Promise<number> {
		if (!frameResources?.frameBinding) {
			throw new Error(
				"WebGPUParticleRenderResources.renderParticles() requires prepared frame resources.",
			);
		}
		const includeBlendModes = options.includeBlendModes ?? null;
		const pipelineMode = options.pipelineMode ?? "legacy";
		const sampleCount = this._normalizeSampleCount(targets.sampleCount);
		const particlePipelineKey = this._createParticlePipelineCacheKey(mode, sampleCount);
		const gpuBatches = context.transient.get(WEBGPU_PARTICLE_DRAW_BATCHES_KEY);
		if (gpuBatches && gpuBatches.length > 0) {
			const drawBatches = gpuBatches.filter((batch) => {
				if (batch.instanceCount <= 0) {
					return false;
				}
				if (!includeBlendModes || includeBlendModes.length === 0) {
					return true;
				}
				return includeBlendModes.includes(batch.blendMode);
			});
			if (drawBatches.length > 0) {
				return this._renderParticlesFromGPUBatches(
					encoder,
					context,
					targets,
					frameResources,
					mode,
					pipelineMode,
					sampleCount,
					drawBatches,
				);
			}
		}
		const batches = context.transient.get(PARTICLE_TRANSIENT_BATCHES_KEY);
		if (!batches || batches.length === 0) return 0;

		const drawBatches = batches.filter((batch) => {
			if (batch.particles.length <= 0) {
				return false;
			}
			if (!includeBlendModes || includeBlendModes.length === 0) {
				return true;
			}
			return includeBlendModes.includes(batch.blendMode);
		});
		if (drawBatches.length === 0) return 0;

		const totalParticles = drawBatches.reduce(
			(sum, batch) => sum + batch.particles.length,
			0,
		);
		if (totalParticles <= 0) return 0;

		await this._ensureParticleResources(
			mode,
			totalParticles,
			pipelineMode,
			sampleCount,
		);
		if (!this._particleInstanceBuffer || !this._particleQuadBuffer) return 0;

		const instanceData = new Float32Array(
			totalParticles * WEBGPU_PARTICLE_INSTANCE_FLOATS,
		);
		const drawRanges: Array<{
			batch: ParticleRenderBatch;
			firstInstance: number;
			instanceCount: number;
		}> = [];
		const activeCacheKeys = new Set<string>();

		let particleOffset = 0;
		for (const batch of drawBatches) {
			const firstInstance = particleOffset;
			for (const particle of batch.particles) {
				const offset = particleOffset * WEBGPU_PARTICLE_INSTANCE_FLOATS;
				instanceData[offset] = particle.position.x;
				instanceData[offset + 1] = particle.position.y;
				instanceData[offset + 2] = particle.position.z;
				instanceData[offset + 3] = Math.max(0.001, particle.size);
				instanceData[offset + 4] = particle.color.r / 255;
				instanceData[offset + 5] = particle.color.g / 255;
				instanceData[offset + 6] = particle.color.b / 255;
				instanceData[offset + 7] = clamp(particle.color.a);
				instanceData[offset + 8] = particle.uvRect.u0;
				instanceData[offset + 9] = particle.uvRect.v0;
				instanceData[offset + 10] = particle.uvRect.u1;
				instanceData[offset + 11] = particle.uvRect.v1;
				instanceData[offset + 12] = particle.rotation;
				instanceData[offset + 13] = batch.receiveShadows ? 1 : 0;
				instanceData[offset + 14] = 0;
				instanceData[offset + 15] = 0;
				particleOffset++;
			}

			drawRanges.push({
				batch,
				firstInstance,
				instanceCount: batch.particles.length,
			});
		}

		this._backend.writeBuffer(this._particleInstanceBuffer, instanceData);
		const pipelines = this._resolveParticlePipelines(particlePipelineKey, pipelineMode);
		if (!pipelines) return 0;

		encoder.beginRenderPass({
			label: targets.label,
			colorAttachments: targets.colorAttachments,
			depthStencilAttachment: {
				view: targets.depth,
				depthLoadOp: "load",
				depthStoreOp: "store",
			},
		} as any);
		encoder.setBindingGroup(0, frameResources.frameBinding);
		encoder.setVertexBuffer(0, this._particleQuadBuffer);
		encoder.setVertexBuffer(1, this._particleInstanceBuffer);
		const dirtyRects = this._resolveParticleDirtyRects(context, targets);

		for (const range of drawRanges) {
			const particleBinding = await this._resolveParticleBinding(
				range.batch,
				activeCacheKeys,
			);
			const pipeline =
				pipelineMode === "oit"
					? pipelines.oitAlpha!
					: range.batch.blendMode === ParticleBlendMode.Additive
						? pipelines.additive!
						: pipelines.alpha!;
			encoder.setPipeline(pipeline);
			encoder.setBindingGroup(1, particleBinding);
			for (const rect of dirtyRects) {
				encoder.setScissorRect?.(rect.x, rect.y, rect.width, rect.height);
				encoder.draw(6, range.instanceCount, 0, range.firstInstance);
			}
		}

		encoder.endRenderPass();
		this._evictParticleBindings(activeCacheKeys);
		return totalParticles;
	}

	public onShaderRuntimeChanged(): void {
		if (this._destroyed) return;
		this._particleShaderModule = null;
		this._particlePipelineAlpha.clear();
		this._particlePipelineOITAlpha.clear();
		this._particlePipelineAdditive.clear();
		this._clearParticleBindingCache();
	}

	public destroy(): void {
		if (this._destroyed) return;
		this._clearParticleBindingCache();
		this._particleShaderModule = null;
		this._particlePipelineAlpha.clear();
		this._particlePipelineOITAlpha.clear();
		this._particlePipelineAdditive.clear();
		this._particleQuadBuffer?.destroy();
		this._particleQuadBuffer = null;
		this._particleInstanceBuffer?.destroy();
		this._particleInstanceBuffer = null;
		this._particleInstanceCapacity = 0;
		this._destroyed = true;
	}

	private async _renderParticlesFromGPUBatches(
		encoder: ICommandEncoder,
		context: FrameContext,
		targets: WebGPUParticlePassTargets,
		frameResources: WebGPUPreparedFrameResources,
		mode: WebGPUSceneTargetMode,
		pipelineMode: "legacy" | "oit",
		sampleCount: number,
		drawBatches: readonly WebGPUParticleDrawBatch[],
	): Promise<number> {
		const totalParticles = drawBatches.reduce(
			(sum, batch) => sum + batch.instanceCount,
			0,
		);
		if (totalParticles <= 0) return 0;
		await this._ensureParticleResources(mode, 0, pipelineMode, sampleCount);
		if (!this._particleQuadBuffer) return 0;
		const pipelineKey = this._createParticlePipelineCacheKey(mode, sampleCount);
		const pipelines = this._resolveParticlePipelines(pipelineKey, pipelineMode);
		if (!pipelines) return 0;

		encoder.beginRenderPass({
			label: targets.label,
			colorAttachments: targets.colorAttachments,
			depthStencilAttachment: {
				view: targets.depth,
				depthLoadOp: "load",
				depthStoreOp: "store",
			},
		} as any);
		encoder.setBindingGroup(0, frameResources.frameBinding);
		encoder.setVertexBuffer(0, this._particleQuadBuffer);
		const dirtyRects = this._resolveParticleDirtyRects(context, targets);
		const activeCacheKeys = new Set<string>();

		for (const batch of drawBatches) {
			const particleBinding = await this._resolveParticleBinding(
				batch,
				activeCacheKeys,
			);
			const pipeline =
				pipelineMode === "oit"
					? pipelines.oitAlpha!
					: batch.blendMode === ParticleBlendMode.Additive
						? pipelines.additive!
						: pipelines.alpha!;
			encoder.setPipeline(pipeline);
			encoder.setBindingGroup(1, particleBinding);
			encoder.setVertexBuffer(1, batch.instanceBuffer);
			for (const rect of dirtyRects) {
				encoder.setScissorRect?.(rect.x, rect.y, rect.width, rect.height);
				if (typeof encoder.drawIndirect === "function") {
					encoder.drawIndirect(batch.indirectBuffer, batch.indirectOffset);
				} else {
					encoder.draw(6, batch.instanceCount, 0, 0);
				}
			}
		}

		encoder.endRenderPass();
		this._evictParticleBindings(activeCacheKeys);
		return totalParticles;
	}

	private async _resolveParticleBinding(
		batch: Pick<ParticleRenderBatch, "systemId" | "texture">,
		activeCacheKeys: Set<string>,
	): Promise<IBindingGroup> {
		const texture = await this._textures.getTextureForSlotAsync(batch.texture, 0);
		const sampler = this._textures.getSamplerForTexture(batch.texture);
		const cacheKey = `particle_${batch.systemId}`;
		activeCacheKeys.add(cacheKey);
		const cachedBinding = this._particleBindingCache.get(cacheKey);
		const uvTransformBuffer =
			cachedBinding?.uvTransformBuffer ??
			this._backend.createBuffer({
				size: WEBGPU_PARTICLE_UV_UNIFORM_SIZE,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
				label: `ParticleUVTransform_${batch.systemId}`,
			});
		let particleBinding: IBindingGroup;
		if (
			cachedBinding &&
			cachedBinding.texture === texture &&
			cachedBinding.sampler === sampler
		) {
			particleBinding = cachedBinding.group;
			cachedBinding.lastUsedFrame = this._frameId;
		} else {
			this._destroyBindingGroup(cachedBinding?.group ?? null);
			particleBinding = this._backend.createBindingGroup({
				layout: this._layouts.particleBindGroupLayout,
				entries: [
					{ binding: WEBGPU_PARTICLE_BINDING_TEXTURE, resource: texture },
					{ binding: WEBGPU_PARTICLE_BINDING_SAMPLER, resource: sampler },
					{
						binding: WEBGPU_PARTICLE_BINDING_UV_TRANSFORM,
						resource: uvTransformBuffer,
					},
				],
				label: `ParticleBinding_${batch.systemId}`,
			});
			this._particleBindingCache.set(cacheKey, {
				group: particleBinding,
				texture,
				sampler,
				uvTransformBuffer,
				lastUsedFrame: this._frameId,
			});
		}
		this._backend.writeBuffer(
			uvTransformBuffer,
			this._createParticleUVTransformData(batch.texture),
		);
		return particleBinding;
	}

	private _resolveParticlePipelines(
		cacheKey: string,
		pipelineMode: "legacy" | "oit",
	): {
		alpha: IRenderPipeline | null;
		oitAlpha: IRenderPipeline | null;
		additive: IRenderPipeline | null;
	} | null {
		const pipelines = {
			alpha: this._particlePipelineAlpha.get(cacheKey) ?? null,
			oitAlpha: this._particlePipelineOITAlpha.get(cacheKey) ?? null,
			additive: this._particlePipelineAdditive.get(cacheKey) ?? null,
		};
		if (pipelineMode === "oit") {
			return pipelines.oitAlpha ? pipelines : null;
		}
		return pipelines.alpha && pipelines.additive ? pipelines : null;
	}

	private _resolveParticleDirtyRects(
		context: FrameContext,
		targets: WebGPUParticlePassTargets,
	): Array<{ x: number; y: number; width: number; height: number }> {
		const targetView =
			targets.colorAttachments.find((attachment) => attachment.view)?.view ??
			targets.depth;
		const viewSize = targetView as { width?: number; height?: number };
		const targetWidth = Math.max(
			1,
			Math.floor(
				typeof viewSize?.width === "number"
					? viewSize.width
					: context.attachments.width,
			),
		);
		const targetHeight = Math.max(
			1,
			Math.floor(
				typeof viewSize?.height === "number"
					? viewSize.height
					: context.attachments.height,
			),
		);
		const hasIncrementalRects =
			context.incremental?.enabled &&
			!context.incremental.forceFullFrame &&
			(context.incremental.dirtyRects?.length ?? 0) > 0;
		if (!hasIncrementalRects) {
			return [{ x: 0, y: 0, width: targetWidth, height: targetHeight }];
		}
		return context.incremental!.dirtyRects
			.map((rect) => {
				const minX = Math.max(
					0,
					Math.floor(
						(rect.x * targetWidth) / Math.max(1, context.attachments.width),
					),
				);
				const minY = Math.max(
					0,
					Math.floor(
						(rect.y * targetHeight) / Math.max(1, context.attachments.height),
					),
				);
				const maxX = Math.min(
					targetWidth,
					Math.ceil(
						((rect.x + rect.width) * targetWidth) /
							Math.max(1, context.attachments.width),
					),
				);
				const maxY = Math.min(
					targetHeight,
					Math.ceil(
						((rect.y + rect.height) * targetHeight) /
							Math.max(1, context.attachments.height),
					),
				);
				return {
					x: minX,
					y: minY,
					width: maxX - minX,
					height: maxY - minY,
				};
			})
			.filter((rect) => rect.width > 0 && rect.height > 0);
	}

	private _evictParticleBindings(activeCacheKeys?: Set<string>): void {
		const staleFrameThreshold =
			this._frameId - WEBGPU_PARTICLE_BINDING_CACHE_MAX_AGE_FRAMES;
		for (const [cacheKey, entry] of this._particleBindingCache.entries()) {
			if (activeCacheKeys?.has(cacheKey)) continue;
			if (entry.lastUsedFrame > staleFrameThreshold) continue;
			this._destroyParticleBindingEntry(cacheKey, entry);
		}

		if (this._particleBindingCache.size <= WEBGPU_PARTICLE_BINDING_CACHE_MAX_ENTRIES) {
			return;
		}
		const evictionCandidates = Array.from(this._particleBindingCache.entries())
			.filter(([cacheKey]) => !activeCacheKeys?.has(cacheKey))
			.sort((left, right) => left[1].lastUsedFrame - right[1].lastUsedFrame);
		while (
			this._particleBindingCache.size > WEBGPU_PARTICLE_BINDING_CACHE_MAX_ENTRIES &&
			evictionCandidates.length > 0
		) {
			const [cacheKey, entry] = evictionCandidates.shift()!;
			this._destroyParticleBindingEntry(cacheKey, entry);
		}
	}

	private _clearParticleBindingCache(): void {
		for (const [cacheKey, entry] of this._particleBindingCache.entries()) {
			this._destroyParticleBindingEntry(cacheKey, entry);
		}
		this._particleBindingCache.clear();
	}

	private _destroyParticleBindingEntry(
		cacheKey: string,
		entry: WebGPUParticleBindingCacheEntry,
	): void {
		this._destroyBindingGroup(entry.group);
		entry.uvTransformBuffer.destroy();
		this._particleBindingCache.delete(cacheKey);
	}

	private _destroyBindingGroup(group: IBindingGroup | null): void {
		const destroy = (group as { destroy?: () => void } | null)?.destroy;
		if (typeof destroy === "function") destroy.call(group);
	}

	private async _ensureParticleResources(
		mode: WebGPUSceneTargetMode,
		totalParticles: number,
		pipelineMode: "legacy" | "oit",
		sampleCount: number,
	): Promise<void> {
		if (!this._particleShaderModule) {
			const shader = await ShaderSource.load("webgpu.particle.composite");
			this._particleShaderModule = await this._backend.createShaderModule({
				label: "WebGPUParticleShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "unknown",
				sourceKind: "particle",
			});
		}

		if (!this._particleQuadBuffer) {
			this._particleQuadBuffer = this._backend.createBuffer({
				size: WEBGPU_PARTICLE_QUAD_VERTICES.byteLength,
				usage: BufferUsage.Vertex | BufferUsage.CopyDst,
				label: "WebGPUParticleQuad",
			});
			this._backend.writeBuffer(
				this._particleQuadBuffer,
				WEBGPU_PARTICLE_QUAD_VERTICES,
			);
		}

		this._ensureParticleInstanceBuffer(totalParticles);
		if (pipelineMode === "oit") {
			await this._ensureParticlePipeline(mode, "oit-alpha", sampleCount);
		} else {
			await this._ensureParticlePipeline(mode, "alpha", sampleCount);
			await this._ensureParticlePipeline(mode, "additive", sampleCount);
		}
	}

	private _ensureParticleInstanceBuffer(totalParticles: number): void {
		if (totalParticles <= this._particleInstanceCapacity) return;
		const resolvedParticles = Number.isFinite(totalParticles)
			? Math.max(1, Math.floor(totalParticles))
			: 1;
		const maxCapacity = Math.max(
			256,
			Math.floor(Number.MAX_SAFE_INTEGER / WEBGPU_PARTICLE_INSTANCE_STRIDE),
		);
		if (resolvedParticles > maxCapacity) {
			throw new Error(
				`Particle instance request ${resolvedParticles} exceeds max capacity ${maxCapacity}.`,
			);
		}
		const exponent = Math.ceil(Math.log2(resolvedParticles));
		const pow2Capacity = Math.pow(2, exponent);
		const nextCapacity = Math.max(
			256,
			Math.min(maxCapacity, Number.isFinite(pow2Capacity) ? pow2Capacity : 256),
		);
		this._particleInstanceBuffer?.destroy();
		this._particleInstanceBuffer = this._backend.createBuffer({
			size: nextCapacity * WEBGPU_PARTICLE_INSTANCE_STRIDE,
			usage: BufferUsage.Vertex | BufferUsage.CopyDst,
			label: "WebGPUParticleInstances",
		});
		this._particleInstanceCapacity = nextCapacity;
	}

	private _ensureParticlePipeline(
		mode: WebGPUSceneTargetMode,
		pipelineType: "alpha" | "additive" | "oit-alpha",
		sampleCount: number,
	): Promise<void> {
		const cache =
			pipelineType === "additive"
				? this._particlePipelineAdditive
				: pipelineType === "oit-alpha"
					? this._particlePipelineOITAlpha
					: this._particlePipelineAlpha;
		const cacheKey = this._createParticlePipelineCacheKey(mode, sampleCount);
		if (cache.has(cacheKey) || !this._particleShaderModule) return Promise.resolve();

		const blend =
			pipelineType === "additive"
				? {
						color: {
							srcFactor: "src-alpha",
							dstFactor: "one",
							operation: "add",
						},
						alpha: {
							srcFactor: "one",
							dstFactor: "one",
							operation: "add",
						},
					}
				: {
						color: {
							srcFactor: "src-alpha",
							dstFactor: "one-minus-src-alpha",
							operation: "add",
						},
						alpha: {
							srcFactor: "one",
							dstFactor: "one-minus-src-alpha",
							operation: "add",
						},
					};
		const colorFormat =
			mode === "mrt" || mode === "color"
				? TextureFormat.RGBA16Float
				: this._backend.canvasFormat;
		const depthFormat =
			mode === "mrt" || mode === "color"
				? TextureFormat.Depth32Float
				: this._backend.canvasDepthFormat;
		const fragmentEntryPoint = pipelineType === "oit-alpha" ? "fsMainOIT" : "fsMain";
		const fragmentTargets =
			pipelineType === "oit-alpha"
				? [
						{
							format: TextureFormat.RGBA16Float,
							blend: {
								color: {
									srcFactor: "one",
									dstFactor: "one",
									operation: "add",
								},
								alpha: {
									srcFactor: "one",
									dstFactor: "one",
									operation: "add",
								},
							},
						},
						{
							format: TextureFormat.R8Unorm,
							blend: {
								color: {
									srcFactor: "zero",
									dstFactor: "one-minus-src",
									operation: "add",
								},
								alpha: {
									srcFactor: "zero",
									dstFactor: "one-minus-src",
									operation: "add",
								},
							},
						},
					]
				: [{ format: colorFormat, blend }];

		return this._backend
			.createPipeline({
				layout: this._layouts.particlePipelineLayout,
				label: `WebGPUParticlePipeline_${pipelineType}_${mode}`,
				vertex: {
					module: this._particleShaderModule,
					entryPoint: "vsMain",
					buffers: WEBGPU_PARTICLE_VERTEX_LAYOUTS,
				},
				fragment: {
					module: this._particleShaderModule,
					entryPoint: fragmentEntryPoint,
					targets: fragmentTargets,
				},
				primitive: {
					topology: "triangle-list" as any,
					cullMode: "none",
					frontFace: "ccw",
				},
				depthStencil: {
					format: depthFormat,
					depthWriteEnabled: false,
					depthCompare: "less",
				},
				sampleCount,
			} as any)
			.then((pipeline) => {
				cache.set(cacheKey, pipeline);
			});
	}

	private _createParticlePipelineCacheKey(
		mode: WebGPUSceneTargetMode,
		sampleCount: number,
	): string {
		return `${mode}|msaa:${sampleCount}`;
	}

	private _normalizeSampleCount(sampleCount: number): number {
		if (!Number.isFinite(sampleCount)) {
			throw new Error("WebGPU particle pass sampleCount must be a finite number.");
		}
		return Math.max(1, Math.floor(sampleCount));
	}

	private _createParticleUVTransformData(texture: ParticleRenderBatch["texture"]) {
		const repeatX = texture?.repeat.x ?? 1;
		const repeatY = texture?.repeat.y ?? 1;
		const offsetX = texture?.offset.x ?? 0;
		const offsetY = texture?.offset.y ?? 0;
		const rotation = texture?.rotation ?? 0;
		return new Float32Array([
			repeatX,
			repeatY,
			offsetX,
			offsetY,
			Math.cos(rotation),
			Math.sin(rotation),
			0,
			0,
		]);
	}
}
