import type { DrawPacket, FrameContext } from "../../../pipeline/types";
import { Logger } from "../../../foundation/Logger";
import { materialUsesTransmission } from "../../../materials/transparency";
import { ParticleBlendMode } from "../../../particles";
import {
	AddressMode,
	FilterMode,
	TextureFormat,
	type IBindingGroup,
	type IRenderPipeline,
	type IRenderTexture,
	type ISampler,
	type IShaderModule,
} from "../../types";
import type { WebGPUBackend } from "../../WebGPUBackend";
import {
	submitWebGPUDraws,
} from "../WebGPUDrawSubmission";
import type {
	WebGPURenderResources,
} from "../WebGPURenderResources";
import type { WebGPUSceneTargetMode } from "../WebGPUScenePassDescriptors";
import { ShaderSource } from "../../../shaders/ShaderSource";
import type { WebGPUFrameGraphRecordingContext } from "./WebGPUFrameGraphRecordingContext";

export interface WebGPUOITPassCallbacks {
	readonly recordingContext: WebGPUFrameGraphRecordingContext;
	recordLegacyMainPass(
		context: FrameContext,
		packets: DrawPacket[],
		clearAttachments: boolean,
		allowEarlyZPrepass: boolean
	): Promise<void>;
	drawTransmissionFallback(
		context: FrameContext,
		packets: DrawPacket[]
	): Promise<void>;
	warnDisabled(key: string, message: string): void;
}

/**
 * Records weighted blended OIT transparent, particle, and resolve work.
 */
export class WebGPUOITPass {
	private readonly _backend: WebGPUBackend;
	private readonly _resources: WebGPURenderResources;
	private readonly _recordingContext: WebGPUFrameGraphRecordingContext;
	private readonly _callbacks: WebGPUOITPassCallbacks;
	private _resolveShaderModule: IShaderModule | null = null;
	private _resolvePipeline: IRenderPipeline | null = null;
	private _resolveSampler: ISampler | null = null;
	private _resolveBinding: IBindingGroup | null = null;
	private _resolveBindingScene: IRenderTexture | null = null;
	private _resolveBindingAccum: IRenderTexture | null = null;
	private _resolveBindingReveal: IRenderTexture | null = null;
	private _hasContributors = false;
	private _transmissionPackets: DrawPacket[] = [];
	private _needsTransmissionAfterParticles = false;

	public constructor(
		backend: WebGPUBackend,
		resources: WebGPURenderResources,
		callbacks: WebGPUOITPassCallbacks
	) {
		this._backend = backend;
		this._resources = resources;
		this._recordingContext = callbacks.recordingContext;
		this._callbacks = callbacks;
	}

	public resetFrameState(): void {
		this._hasContributors = false;
		this._transmissionPackets = [];
		this._needsTransmissionAfterParticles = false;
	}

	public invalidateBindings(): void {
		this._destroyBindingGroup(this._resolveBinding);
		this._resolveBinding = null;
		this._resolveBindingScene = null;
		this._resolveBindingAccum = null;
		this._resolveBindingReveal = null;
	}

	public onShaderRuntimeChanged(): void {
		this._destroyManagedResource(this._resolveShaderModule);
		this._destroyManagedResource(this._resolvePipeline);
		this._destroyManagedResource(this._resolveSampler);
		this._resolveShaderModule = null;
		this._resolvePipeline = null;
		this._resolveSampler = null;
		this.invalidateBindings();
	}

	public destroy(): void {
		this.onShaderRuntimeChanged();
	}

	public async recordTransparentPass(context: FrameContext): Promise<void> {
		const encoder = this._recordingContext.getEncoder();
		const targets = this._recordingContext.getFrameTargets();
		if (!encoder) {
			return;
		}
		if (!targets) {
			await this._callbacks.recordLegacyMainPass(
				context,
				context.scene.transparentPackets,
				false,
				false
			);
			return;
		}
		const frameResources = this._recordingContext.requireFrameResources();
		await this._resources.buildClusteredLighting(encoder, frameResources);
		const { oitPackets, transmissionPackets } =
			this._partitionTransparentPackets(context.scene.transparentPackets);
		this._transmissionPackets = transmissionPackets;
		this._needsTransmissionAfterParticles =
			(context.scene.particleSystems?.length ?? 0) > 0;
		this._hasContributors = false;
		if (oitPackets.length > 0) {
			this._clearTargets();
			const draws = await this._drawPackets(context, oitPackets);
			this._hasContributors = draws > 0;
		}
		if (!this._needsTransmissionAfterParticles) {
			if (this._hasContributors) {
				await this._resolveComposition(context);
			}
			await this._callbacks.drawTransmissionFallback(
				context,
				this._transmissionPackets
			);
			this._transmissionPackets = [];
			this._hasContributors = false;
		}
	}

	public async recordParticlePass(context: FrameContext): Promise<void> {
		const encoder = this._recordingContext.getEncoder();
		const targets = this._recordingContext.getFrameTargets();
		if (!encoder || !targets?.oitAccum || !targets.oitReveal) {
			return;
		}
		const msaaTargets = this._recordingContext.getMSAATargets();
		const depthAttachment = msaaTargets?.depth ?? targets.depth;
		const sceneTargetMode = this._resolveSceneTargetMode();
		if (!this._hasContributors) {
			this._clearTargets();
		}
		const alphaParticleCount = await this._resources.renderParticles(
			encoder,
			context,
			{
				label: "WebGPUParticlesOIT",
				colorAttachments: [
					{
						view: targets.oitAccum,
						loadOp: "load",
						storeOp: "store",
					},
					{
						view: targets.oitReveal,
						loadOp: "load",
						storeOp: "store",
					},
				],
				depth: depthAttachment,
			},
			this._recordingContext.requireFrameResources(),
			sceneTargetMode,
			{
				includeBlendModes: [ParticleBlendMode.Alpha],
				pipelineMode: "oit",
			}
		);
		if (alphaParticleCount > 0) {
			this._hasContributors = true;
		}
		if (this._hasContributors) {
			await this._resolveComposition(context);
		}
		if (this._transmissionPackets.length > 0) {
			await this._callbacks.drawTransmissionFallback(
				context,
				this._transmissionPackets
			);
		}
		await this._resources.renderParticles(
			encoder,
			context,
			{
				label: "WebGPUParticlesMRT_Additive",
				colorAttachments: [
					{
						view: msaaTargets?.sceneColorMain ?? targets.sceneColorMain,
						resolveTarget: msaaTargets ? targets.sceneColorMain : undefined,
						loadOp: "load",
						storeOp: "store",
					},
				],
				depth: depthAttachment,
			},
			this._recordingContext.requireFrameResources(),
			sceneTargetMode,
			{
				includeBlendModes: [ParticleBlendMode.Additive],
				pipelineMode: "legacy",
			}
		);
		this.resetFrameState();
	}

	private _partitionTransparentPackets(packets: DrawPacket[]): {
		oitPackets: DrawPacket[];
		transmissionPackets: DrawPacket[];
	} {
		const oitPackets: DrawPacket[] = [];
		const transmissionPackets: DrawPacket[] = [];
		for (const packet of packets) {
			if (materialUsesTransmission(packet.material)) {
				transmissionPackets.push(packet);
				continue;
			}
			oitPackets.push(packet);
		}
		return {
			oitPackets,
			transmissionPackets,
		};
	}

	private _clearTargets(): void {
		const encoder = this._recordingContext.getEncoder();
		const targets = this._recordingContext.getFrameTargets();
		if (!encoder || !targets?.oitAccum || !targets.oitReveal) {
			return;
		}
		encoder.beginRenderPass({
			label: "WebGPUOITClear",
			colorAttachments: [
				{
					view: targets.oitAccum,
					clearValue: { r: 0, g: 0, b: 0, a: 0 },
					loadOp: "clear",
					storeOp: "store",
				},
				{
					view: targets.oitReveal,
					clearValue: { r: 1, g: 1, b: 1, a: 1 },
					loadOp: "clear",
					storeOp: "store",
				},
			],
		});
		encoder.endRenderPass();
	}

	private async _drawPackets(
		context: FrameContext,
		packets: DrawPacket[]
	): Promise<number> {
		const encoder = this._recordingContext.getEncoder();
		const targets = this._recordingContext.getFrameTargets();
		if (!encoder || !targets?.oitAccum || !targets.oitReveal || packets.length <= 0) {
			return 0;
		}
		const frameResources = this._recordingContext.requireFrameResources();
		const depthAttachment =
			this._recordingContext.getMSAATargets()?.depth ?? targets.depth;
		encoder.beginRenderPass({
			label: "WebGPUOITDraw",
			colorAttachments: [
				{
					view: targets.oitAccum,
					loadOp: "load",
					storeOp: "store",
				},
				{
					view: targets.oitReveal,
					loadOp: "load",
					storeOp: "store",
				},
			],
			depthStencilAttachment: {
				view: depthAttachment,
				depthLoadOp: "load",
				depthStoreOp: "store",
			},
		});
		const dirtyRects = this._recordingContext.resolveDirtyRects(
			context,
			targets.sceneColorMain.width,
			targets.sceneColorMain.height
		);
		const sceneTargetMode = this._resolveSceneTargetMode();
		const submission = await submitWebGPUDraws({
			encoder,
			resources: this._resources,
			frameResources,
			packets,
			dirtyRects,
			selectPacketsForRect: (candidatePackets, rect) =>
				this._recordingContext.selectTransparentSubsetForRect(
					context,
					candidatePackets,
					rect
				),
			resolveDrawOptions: () => ({
				sceneTargetMode,
				transparentPipelineMode: "oit",
			}),
		});
		encoder.endRenderPass();
		return submission.drawCount;
	}

	private async _ensureResolveResources(): Promise<void> {
		if (!this._resolveShaderModule) {
			const composite = await ShaderSource.load(
				"webgpu.utility.oitResolve.composite"
			);
			this._resolveShaderModule = await this._backend.createShaderModule({
				label: "WebGPUOITResolveShader",
				code: composite.code,
				sourceMap: composite.sourceMap,
				language: "wgsl",
				stage: "unknown",
				sourceKind: "postprocess",
			});
		}
		if (!this._resolvePipeline) {
			this._resolvePipeline = this._backend.createPipeline({
				label: "WebGPUOITResolvePipeline",
				vertex: {
					module: this._resolveShaderModule,
					entryPoint: "vsMain",
				},
				fragment: {
					module: this._resolveShaderModule,
					entryPoint: "fsMain",
					targets: [{ format: TextureFormat.RGBA16Float }],
				},
				primitive: {
					topology: "triangle-list" as any,
					cullMode: "none",
					frontFace: "ccw",
				},
			} as any);
		}
		if (!this._resolveSampler) {
			this._resolveSampler = this._backend.createSampler({
				label: "WebGPUOITResolveSampler",
				magFilter: FilterMode.Linear,
				minFilter: FilterMode.Linear,
				mipmapFilter: FilterMode.Linear,
				addressModeU: AddressMode.ClampToEdge,
				addressModeV: AddressMode.ClampToEdge,
			});
		}
	}

	private _copySceneColorForResolve(): boolean {
		const encoder = this._recordingContext.getEncoder();
		const targets = this._recordingContext.getFrameTargets();
		if (!encoder || !targets?.oitSceneColorCopy) {
			return false;
		}
		if (typeof encoder.copyTextureToTexture !== "function") {
			this._callbacks.warnDisabled(
				"webgpu-oit-disabled-runtime",
				"WebGPU OIT requires in-frame texture-copy support; falling back to legacy transparent rendering."
			);
			return false;
		}
		try {
			encoder.copyTextureToTexture(
				{ texture: targets.sceneColorMain },
				{ texture: targets.oitSceneColorCopy },
				{
					width: Math.max(1, this._recordingContext.getTargetWidth()),
					height: Math.max(1, this._recordingContext.getTargetHeight()),
					depthOrArrayLayers: 1,
				}
			);
			return true;
		} catch (error) {
			const key = "webgpu-oit-copy-scene-color-failed";
			Logger.warn(
				`[${key}] WebGPU OIT scene-color copy failed; falling back to legacy transparent rendering. ${String(error)}`,
				{ scope: "WebGPUFrameExecutor", onceKey: key }
			);
			return false;
		}
	}

	private async _resolveComposition(context: FrameContext): Promise<void> {
		const encoder = this._recordingContext.getEncoder();
		const targets = this._recordingContext.getFrameTargets();
		if (
			!encoder ||
			!targets?.oitSceneColorCopy ||
			!targets.oitAccum ||
			!targets.oitReveal ||
			!this._hasContributors
		) {
			return;
		}
		if (!this._copySceneColorForResolve()) {
			return;
		}
		await this._ensureResolveResources();
		if (!this._resolvePipeline || !this._resolveSampler) {
			return;
		}
		if (
			!this._resolveBinding ||
			this._resolveBindingScene !== targets.oitSceneColorCopy ||
			this._resolveBindingAccum !== targets.oitAccum ||
			this._resolveBindingReveal !== targets.oitReveal
		) {
			this._destroyBindingGroup(this._resolveBinding);
			this._resolveBinding = this._backend.createBindingGroup({
				pipeline: this._resolvePipeline,
				layoutIndex: 0,
				entries: [
					{ binding: 0, resource: targets.oitSceneColorCopy },
					{ binding: 1, resource: targets.oitAccum },
					{ binding: 2, resource: targets.oitReveal },
					{ binding: 3, resource: this._resolveSampler },
				],
				label: "WebGPUOITResolveBinding",
			});
			this._resolveBindingScene = targets.oitSceneColorCopy;
			this._resolveBindingAccum = targets.oitAccum;
			this._resolveBindingReveal = targets.oitReveal;
		}
		encoder.beginRenderPass({
			label: "WebGPUOITResolvePass",
			colorAttachments: [
				{
					view: targets.sceneColorMain,
					loadOp: "load",
					storeOp: "store",
				},
			],
		});
		encoder.setPipeline(this._resolvePipeline);
		encoder.setBindingGroup(0, this._resolveBinding);
		const dirtyRects = this._recordingContext.resolveDirtyRects(
			context,
			targets.sceneColorMain.width,
			targets.sceneColorMain.height
		);
		for (const rect of dirtyRects) {
			encoder.setScissorRect?.(rect.x, rect.y, rect.width, rect.height);
			encoder.draw(3);
		}
		encoder.endRenderPass();
	}

	private _resolveSceneTargetMode(): Exclude<WebGPUSceneTargetMode, "single"> {
		return this._recordingContext.getSceneTargetMode() === "color" ?
				"color"
			:	"mrt";
	}

	private _destroyBindingGroup(group: IBindingGroup | null): void {
		const destroyFn = (group as { destroy?: () => void } | null)?.destroy;
		if (typeof destroyFn === "function") {
			destroyFn.call(group);
		}
	}

	private _destroyManagedResource(resource: unknown): void {
		const destroyFn = (resource as { destroy?: () => void } | null)?.destroy;
		if (typeof destroyFn === "function") {
			destroyFn.call(resource);
		}
	}
}
