import type {
	DirtyRect,
} from "../../../pipeline/incremental";
import type {
	DrawPacket,
	FrameContext,
} from "../../../pipeline/types";
import type { PreparedFramePacketSet } from "../../../pipeline/FramePacketContributorRegistry";
import { materialSupportsWebGPUDeferredLighting } from "../material";
import { GBufferSlot } from "../constants";
import {
	getDefaultWebGPUDrawBindings,
	submitWebGPUDraws,
} from "../WebGPUDrawSubmission";
import type {
	WebGPUPreparedFrameResources,
	WebGPUParticleBillboardRenderer,
	WebGPUSceneResourceProvider,
} from "../WebGPUResourceContracts";
import type { WebGPUSceneTargetMode } from "../WebGPUScenePassDescriptors";
import type { WebGPUFrameTargets } from "../WebGPUFrameTargetContracts";
import type { WebGPUFrameHost } from "./WebGPUFrameHost";
import {
	TextureFormat,
	type IBindingGroup,
	type IRenderTexture,
} from "../../types";
import type { WebGPUDepthDirtyClearPass } from "./WebGPUDepthDirtyClearPass";
import type { WebGPUFrameExecutionContext } from "./WebGPUFrameExecutionContext";
import { Logger } from "../../../foundation/Logger";

export interface WebGPUDeferredOpaqueFrameState {
	readonly fallbackPackets: DrawPacket[];
	readonly clearSceneColor: boolean;
	readonly lightingEnabled: boolean;
}

export interface WebGPUScenePassRecorderCallbacks {
	getGBufferWriteBinding(): IBindingGroup;
	preflightDeferredFrame(context: FrameContext): Promise<void>;
}

/** @internal Concrete leaf port for scene recording data. */
export class WebGPUSceneRecordingPort {
	private _frame: Pick<
		WebGPUFrameExecutionContext,
		"configuration" | "resources" | "targets" | "commands" |
		"earlyZPrepassEnabled" | "dirtyRects"
	> | null = null;

	public bind(frame: WebGPUFrameExecutionContext): void {
		this._frame = frame;
	}

	public close(): void {
		this._frame = null;
	}

	public getEncoder() { return this._require().commands.encoder; }
	public getFrameTargets() { return this._require().targets.frameTargets; }
	public getMSAATargets() { return this._require().targets.msaaTargets; }
	public getTargetWidth() { return this._require().targets.width; }
	public getTargetHeight() { return this._require().targets.height; }
	public getSampleCount() { return this._require().targets.sampleCount; }
	public getSceneTargetMode() { return this._require().targets.sceneTargetMode; }
	public getDeferredGBufferLayout() {
		return this._require().configuration.deferredGBufferLayout;
	}
	public isMRTEnabled() { return this._require().configuration.mrtSupported; }
	public isEarlyZPrepassEnabled() { return this._require().earlyZPrepassEnabled; }
	public requireFrameResources() { return this._require().resources; }
	public isIncrementalPartial(context: FrameContext | null) {
		return this._require().dirtyRects.isIncrementalPartial(context);
	}
	public resolveDirtyRects(context: FrameContext | null, width: number, height: number) {
		return this._require().dirtyRects.resolveDirtyRects(context, width, height);
	}
	public selectPacketsForRect(
		context: FrameContext,
		packets: DrawPacket[],
		rect: DirtyRect,
	) {
		return this._require().dirtyRects.selectPacketsForRect(context, packets, rect);
	}
	public selectTransparentSubsetForRect(
		context: FrameContext,
		packets: DrawPacket[],
		rect: DirtyRect,
	) {
		return this._require().dirtyRects.selectTransparentSubsetForRect(
			context,
			packets,
			rect,
		);
	}

	private _require() {
		if (!this._frame) throw new Error("WebGPU scene recording frame is not active.");
		return this._frame;
	}
}

/**
 * Records WebGPU scene draw passes while the runtime orchestrates graph nodes.
 */
export class WebGPUScenePassRecorder {
	private readonly _host: WebGPUFrameHost;
	private readonly _sceneResources: WebGPUSceneResourceProvider;
	private readonly _particleRenderer: WebGPUParticleBillboardRenderer;
	private readonly _framePort = new WebGPUSceneRecordingPort();
	private readonly _depthDirtyClearPass: WebGPUDepthDirtyClearPass;
	private readonly _callbacks: WebGPUScenePassRecorderCallbacks;

	public constructor(
		host: WebGPUFrameHost,
		sceneResources: WebGPUSceneResourceProvider,
		particleRenderer: WebGPUParticleBillboardRenderer,
		depthDirtyClearPass: WebGPUDepthDirtyClearPass,
		callbacks: WebGPUScenePassRecorderCallbacks
	) {
		this._host = host;
		this._sceneResources = sceneResources;
		this._particleRenderer = particleRenderer;
		this._depthDirtyClearPass = depthDirtyClearPass;
		this._callbacks = callbacks;
	}

	/** @internal Binds the narrow frame data used by scene leaf recorders. */
	public bindFrame(frame: WebGPUFrameExecutionContext): void {
		this._framePort.bind(frame);
	}

	/** @internal Releases the active frame identity after commit or abort. */
	public closeFrame(): void {
		this._framePort.close();
	}

	/**
	 * Records the opaque scene node and returns deferred state when lighting
	 * resolution must continue in later graph nodes.
	 *
	 * @param context Current frame context.
	 * @param deferredEnabled Whether deferred opaque lighting is active.
	 * @returns Deferred opaque state, or `null` when opaque work is complete.
	 * @sideEffects Records one or more WebGPU render passes.
	 */
	public async recordOpaque(
		context: FrameContext,
		framePackets: PreparedFramePacketSet,
		deferredEnabled: boolean
	): Promise<WebGPUDeferredOpaqueFrameState | null> {
		const targets = this._framePort.getFrameTargets();
		const opaquePackets = framePackets.opaque.slice();
		if (
			!deferredEnabled ||
			!this._framePort.isMRTEnabled() ||
			!targets
		) {
			await this.recordMainPass(
				context,
				opaquePackets,
				true,
				true
			);
			return null;
		}

		const deferredPackets: DrawPacket[] = [];
		const fallbackPackets: DrawPacket[] = [];
		for (const packet of opaquePackets) {
			if (materialSupportsWebGPUDeferredLighting(packet.material)) {
				deferredPackets.push(packet);
			} else {
				fallbackPackets.push(packet);
			}
		}

		if (deferredPackets.length <= 0 && fallbackPackets.length > 0) {
			await this.recordMainPass(context, fallbackPackets, true, true);
			return null;
		}

		try {
			await this._preflightDeferredPackets(context, deferredPackets);
		} catch (error) {
			Logger.warn(
				"[webgpu-deferred-runtime-fallback] Deferred pipeline or binding " +
					`creation failed; using the legacy MRT forward path. ${String(error)}`,
				{
					scope: "WebGPUScenePassRecorder",
					onceKey: "webgpu-deferred-runtime-fallback",
				}
			);
			await this.recordMainPass(context, opaquePackets, true, true);
			return null;
		}

		const deferredResult = await this._recordDeferredGBufferPass(
			context,
			deferredPackets,
			true,
			true
		);
		return {
			fallbackPackets,
			clearSceneColor: deferredResult?.clearSceneColor ?? false,
			lightingEnabled: deferredResult !== null,
		};
	}

	private async _preflightDeferredPackets(
		context: FrameContext,
		packets: readonly DrawPacket[]
	): Promise<void> {
		const frameResources = this._framePort.requireFrameResources();
		const deferredGBufferLayout = this._framePort.getDeferredGBufferLayout();
		this._callbacks.getGBufferWriteBinding();
		await this._callbacks.preflightDeferredFrame(context);
		for (const packet of packets) {
			await this._sceneResources.getDrawResources(packet, frameResources, {
				sceneTargetMode: "gbuffer",
				deferredGBufferLayout,
				sampleCount: 1,
				drawMode: "default",
			});
		}
	}

	/**
	 * Records a forward scene pass for the currently selected target mode.
	 *
	 * @param context Current frame context.
	 * @param packets Draw packets to submit.
	 * @param clearAttachments Whether this pass owns initial clears.
	 * @param allowEarlyZPrepass Whether early-Z can run before color output.
	 * @returns Nothing.
	 * @sideEffects Records WebGPU render passes and clustered lighting work.
	 */
	public async recordMainPass(
		context: FrameContext,
		packets: DrawPacket[],
		clearAttachments: boolean,
		allowEarlyZPrepass: boolean
	): Promise<void> {
		const encoder = this._framePort.getEncoder();
		if (!encoder) return;
		const frameResources = this._framePort.requireFrameResources();
		await this._sceneResources.buildClusteredLighting(encoder, frameResources);
		if (
			!this._framePort.isMRTEnabled() ||
			!this._framePort.getFrameTargets()
		) {
			await this.recordLegacyMainPass(
				context,
				packets,
				clearAttachments,
				allowEarlyZPrepass
			);
			return;
		}
		if (this._framePort.getSceneTargetMode() === "color") {
			await this._recordColorMainPass(
				context,
				packets,
				clearAttachments,
				allowEarlyZPrepass,
				frameResources
			);
			return;
		}
		await this._recordMRTMainPass(
			context,
			packets,
			clearAttachments,
			allowEarlyZPrepass,
			frameResources
		);
	}

	/**
	 * Records a canvas-backed forward pass.
	 *
	 * @param context Current frame context.
	 * @param packets Draw packets to submit.
	 * @param clearAttachments Whether this pass owns initial clears.
	 * @param allowEarlyZPrepass Whether early-Z can run before color output.
	 * @returns Nothing.
	 * @sideEffects Records a render pass against the canvas attachments.
	 */
	public async recordLegacyMainPass(
		context: FrameContext,
		packets: DrawPacket[],
		clearAttachments: boolean,
		allowEarlyZPrepass: boolean
	): Promise<void> {
		const encoder = this._framePort.getEncoder();
		if (!encoder) return;
		const frameResources = this._framePort.requireFrameResources();
		await this._sceneResources.buildClusteredLighting(encoder, frameResources);
		const incrementalPartial =
			this._framePort.isIncrementalPartial(context);
		const colorTexture = this._host.getCanvasColorTexture();
		const depthTexture = this._host.getCanvasDepthTexture();
		const shouldClearAttachments = clearAttachments && !incrementalPartial;
		const dirtyRects = this._framePort.resolveDirtyRects(
			context,
			colorTexture.width,
			colorTexture.height
		);
		let depthPartialReuseApplied = false;
		if (incrementalPartial && dirtyRects.length > 0) {
			depthPartialReuseApplied = await this._depthDirtyClearPass.record(
				encoder,
				depthTexture,
				this._host.canvasDepthFormat,
				1,
				dirtyRects
			);
		}
		const shouldRunEarlyZ =
			allowEarlyZPrepass &&
			this._framePort.isEarlyZPrepassEnabled() &&
			packets.length > 0;
		const earlyZPacketIds =
			shouldRunEarlyZ ?
				await this._recordEarlyZPrepass(
					context,
					packets,
					dirtyRects,
					"single",
					depthTexture,
					this._resolveLegacyMainDepthLoadOp(
						depthPartialReuseApplied,
						incrementalPartial,
						shouldClearAttachments,
						false
					)
				)
			:	new Set<string>();
		const earlyZExecuted = earlyZPacketIds.size > 0;

		encoder.beginRenderPass({
			colorAttachments: [
				{
					view: colorTexture,
					clearValue: { r: 0, g: 0, b: 0, a: 1 },
					loadOp: shouldClearAttachments ? "clear" : "load",
					storeOp: "store",
				},
			],
			depthStencilAttachment: {
				view: depthTexture,
				depthClearValue: 1,
				depthLoadOp: this._resolveLegacyMainDepthLoadOp(
					depthPartialReuseApplied,
					incrementalPartial,
					shouldClearAttachments,
					earlyZExecuted
				),
				depthStoreOp: "store",
			},
		});

		if (shouldClearAttachments) {
			const environmentResources =
				await this._sceneResources.getEnvironmentResources(frameResources, "single", {
					sampleCount: 1,
				});
			if (environmentResources) {
				encoder.setPipeline(environmentResources.pipeline);
				encoder.setBindingGroup(0, environmentResources.frameBinding);
				encoder.draw(3);
			}
		}

		await submitWebGPUDraws({
			encoder,
			resources: this._sceneResources,
			frameResources,
			packets,
			dirtyRects,
			selectPacketsForRect: (candidatePackets, rect) =>
				this._framePort.selectPacketsForRect(
					context,
					candidatePackets,
					rect
				),
			resolveDrawOptions: (packet) => ({
				sceneTargetMode: "single",
				sampleCount: 1,
				drawMode:
					earlyZExecuted && earlyZPacketIds.has(packet.id) ?
						"early-z-color"
					:	"default",
			}),
		});

		encoder.endRenderPass();
	}

	/**
	 * Records transparent transmission fallback packets.
	 *
	 * @param context Current frame context.
	 * @param packets Transmission packets to draw.
	 * @returns Nothing.
	 * @sideEffects Records a forward transmission render pass.
	 */
	public async drawTransmissionPackets(
		context: FrameContext,
		packets: DrawPacket[]
	): Promise<void> {
		const encoder = this._framePort.getEncoder();
		if (!encoder || packets.length <= 0) {
			return;
		}
		const frameResources = this._framePort.requireFrameResources();
		await this._sceneResources.buildClusteredLighting(encoder, frameResources);
		const targets = this._framePort.getFrameTargets();
		if (!this._framePort.isMRTEnabled() || !targets) {
			await this.recordLegacyMainPass(context, packets, false, false);
			return;
		}
		if (this._hasTransmissionCaptureTargets(targets)) {
			const captured = await this._recordTransmissionCapture(
				context,
				packets,
				targets
			);
			if (captured) {
				return;
			}
		}
		const msaaTargets = this._framePort.getMSAATargets();
		if (this._framePort.getSceneTargetMode() === "color") {
			const sceneColorAttachment =
				msaaTargets?.sceneColorMain ?? targets.sceneColorMain;
			const depthAttachment = msaaTargets?.depth ?? targets.depth;
			encoder.beginRenderPass({
				label: "WebGPUTransmissionColor",
				colorAttachments: [
					{
						view: sceneColorAttachment,
						resolveTarget: msaaTargets ? targets.sceneColorMain : undefined,
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
			const dirtyRects = this._framePort.resolveDirtyRects(
				context,
				sceneColorAttachment.width,
				sceneColorAttachment.height
			);
			await submitWebGPUDraws({
				encoder,
				resources: this._sceneResources,
				frameResources,
				packets,
				dirtyRects,
				selectPacketsForRect: (candidatePackets, rect) =>
					this._framePort.selectTransparentSubsetForRect(
						context,
						candidatePackets,
						rect
					),
				resolveDrawOptions: () => ({
					sceneTargetMode: "color",
					transparentPipelineMode: "transmission",
					sampleCount: this._framePort.getSampleCount(),
				}),
			});
			encoder.endRenderPass();
			return;
		}
		const sceneColorAttachment =
			msaaTargets?.sceneColorMain ?? targets.sceneColorMain;
		const gAlbedoAttachment =
			msaaTargets?.gAlbedoAlpha ?? targets.gAlbedoAlpha;
		const gNormalAttachment =
			msaaTargets?.gNormalRoughMetal ?? targets.gNormalRoughMetal;
		const gEmissiveAttachment =
			msaaTargets?.gEmissiveOcclusion ?? targets.gEmissiveOcclusion;
		const gMotionAttachment =
			msaaTargets?.gMotionDepth ?? targets.gMotionDepth;
		const depthAttachment = msaaTargets?.depth ?? targets.depth;
		encoder.beginRenderPass({
			label: "WebGPUTransmissionMRT",
			colorAttachments: [
				{
					view: sceneColorAttachment,
					resolveTarget: msaaTargets ? targets.sceneColorMain : undefined,
					loadOp: "load",
					storeOp: "store",
				},
				{
					view: gAlbedoAttachment,
					resolveTarget: msaaTargets ? targets.gAlbedoAlpha : undefined,
					loadOp: "load",
					storeOp: "store",
				},
				{
					view: gNormalAttachment,
					resolveTarget: msaaTargets ? targets.gNormalRoughMetal : undefined,
					loadOp: "load",
					storeOp: "store",
				},
				{
					view: gEmissiveAttachment,
					resolveTarget: msaaTargets ? targets.gEmissiveOcclusion : undefined,
					loadOp: "load",
					storeOp: "store",
				},
				{
					view: gMotionAttachment,
					resolveTarget: msaaTargets ? targets.gMotionDepth : undefined,
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
		const dirtyRects = this._framePort.resolveDirtyRects(
			context,
			targets.oitAccum?.width ?? sceneColorAttachment.width,
			targets.oitAccum?.height ?? sceneColorAttachment.height
		);
		await submitWebGPUDraws({
			encoder,
			resources: this._sceneResources,
			frameResources,
			packets,
			dirtyRects,
			selectPacketsForRect: (candidatePackets, rect) =>
				this._framePort.selectTransparentSubsetForRect(
					context,
					candidatePackets,
					rect
				),
			resolveDrawOptions: () => ({
				sceneTargetMode: "mrt",
				transparentPipelineMode: "transmission",
				sampleCount: this._framePort.getSampleCount(),
			}),
		});
		encoder.endRenderPass();
	}

	private _hasTransmissionCaptureTargets(
		targets: WebGPUFrameTargets,
	): boolean {
		return !!(
			targets.transmissionSceneColorCopy &&
			targets.transmissionLighting &&
			targets.gTransmissionSurface0 &&
			targets.gTransmissionSurface1 &&
			targets.gTransmissionSurface2 &&
			targets.transmissionDepth
		);
	}

	private async _recordTransmissionCapture(
		context: FrameContext,
		packets: DrawPacket[],
		targets: WebGPUFrameTargets,
	): Promise<boolean> {
		const encoder = this._framePort.getEncoder();
		if (
			!encoder ||
			!encoder.copyTextureToTexture ||
			!targets.transmissionSceneColorCopy ||
			!targets.transmissionLighting ||
			!targets.gTransmissionSurface0 ||
			!targets.gTransmissionSurface1 ||
			!targets.gTransmissionSurface2 ||
			!targets.transmissionDepth
		) {
			return false;
		}
		const frameResources = this._framePort.requireFrameResources();
		encoder.copyTextureToTexture(
			{ texture: targets.sceneColorMain },
			{ texture: targets.transmissionSceneColorCopy },
			{
				width: targets.sceneColorMain.width,
				height: targets.sceneColorMain.height,
				depthOrArrayLayers: 1,
			}
		);
		encoder.copyTextureToTexture(
			{ texture: targets.depth, aspect: "depth-only" },
			{ texture: targets.transmissionDepth, aspect: "depth-only" },
			{
				width: targets.depth.width,
				height: targets.depth.height,
				depthOrArrayLayers: 1,
			}
		);
		encoder.beginRenderPass({
			label: "WebGPUTransmissionCapture",
			colorAttachments: [
				{
					view: targets.transmissionLighting,
					clearValue: { r: 0, g: 0, b: 0, a: 0 },
					loadOp: "clear",
					storeOp: "store",
				},
				{
					view: targets.gTransmissionSurface0,
					clearValue: { r: 0, g: 0, b: 0, a: 0 },
					loadOp: "clear",
					storeOp: "store",
				},
				{
					view: targets.gTransmissionSurface1,
					clearValue: { r: 0, g: 0, b: 0, a: 0 },
					loadOp: "clear",
					storeOp: "store",
				},
				{
					view: targets.gTransmissionSurface2,
					clearValue: { r: 0, g: 0, b: 0, a: 0 },
					loadOp: "clear",
					storeOp: "store",
				},
			],
			depthStencilAttachment: {
				view: targets.transmissionDepth,
				depthLoadOp: "load",
				depthStoreOp: "store",
			},
		});
		await submitWebGPUDraws({
			encoder,
			resources: this._sceneResources,
			frameResources,
			packets,
			resolveDrawOptions: () => ({
				sceneTargetMode: "mrt",
				transparentPipelineMode: "transmission-capture",
				sampleCount: 1,
			}),
		});
		encoder.endRenderPass();
		return true;
	}

	/**
	 * Records legacy particle rendering for single/color/MRT scene targets.
	 *
	 * @param context Current frame context.
	 * @returns Nothing.
	 * @sideEffects Records particle draw passes through render resources.
	 */
	public async recordParticlePass(context: FrameContext): Promise<void> {
		const encoder = this._framePort.getEncoder();
		if (!encoder) return;
		const frameResources = this._framePort.requireFrameResources();
		const targets = this._framePort.getFrameTargets();

		if (this._framePort.isMRTEnabled() && targets) {
			const msaaTargets = this._framePort.getMSAATargets();
			const sceneTargetMode =
				this._framePort.getSceneTargetMode() === "color" ?
					"color"
				:	"mrt";
			await this._particleRenderer.renderParticles(
				encoder,
				context,
				{
					label: "WebGPUParticlesMRT",
				sampleCount: this._framePort.getSampleCount(),
					colorAttachments: [
						{
							view:
								msaaTargets?.sceneColorMain ??
								targets.sceneColorMain,
							resolveTarget:
								msaaTargets ? targets.sceneColorMain : undefined,
							clearValue: { r: 0, g: 0, b: 0, a: 1 },
							loadOp: "load",
							storeOp: "store",
						},
					],
					depth: msaaTargets?.depth ?? targets.depth,
				},
				frameResources,
				sceneTargetMode,
				{
					pipelineMode: "legacy",
				}
			);
			return;
		}

		await this._particleRenderer.renderParticles(
			encoder,
			context,
			{
				label: "WebGPUParticlesSingle",
				sampleCount: 1,
				colorAttachments: [
					{
						view: this._host.getCanvasColorTexture(),
						clearValue: { r: 0, g: 0, b: 0, a: 1 },
						loadOp: "load",
						storeOp: "store",
					},
				],
				depth: this._host.getCanvasDepthTexture(),
			},
			frameResources,
			"single",
			{
				pipelineMode: "legacy",
			}
		);
	}

	private async _recordDeferredGBufferPass(
		context: FrameContext,
		packets: DrawPacket[],
		clearAttachments: boolean,
		allowEarlyZPrepass: boolean
	): Promise<{ clearSceneColor: boolean } | null> {
		const encoder = this._framePort.getEncoder();
		const targets = this._framePort.getFrameTargets();
		if (!encoder || !targets) {
			return null;
		}
		const deferredGBufferLayout = this._framePort.getDeferredGBufferLayout();
		if (
			deferredGBufferLayout === "extended" &&
			(
				!targets.gSpecular ||
				!targets.gCoatSheen ||
				!targets.gSheenReflectance ||
				!targets.gMaterialExt0 ||
				!targets.gMaterialExt3
			)
		) {
			await this.recordMainPass(context, packets, clearAttachments, true);
			return null;
		}

		const frameResources = this._framePort.requireFrameResources();
		await this._sceneResources.buildClusteredLighting(encoder, frameResources);
		const incrementalPartial =
			this._framePort.isIncrementalPartial(context);
		const sceneColorAttachment = targets.sceneColorMain;
		const depthAttachment = targets.depth;
		const dirtyRects = this._framePort.resolveDirtyRects(
			context,
			sceneColorAttachment.width,
			sceneColorAttachment.height
		);
		const shouldClearAttachments = clearAttachments && !incrementalPartial;
		let depthPartialReuseApplied = false;
		if (incrementalPartial && dirtyRects.length > 0) {
			depthPartialReuseApplied = await this._depthDirtyClearPass.record(
				encoder,
				depthAttachment,
				TextureFormat.Depth32Float,
				1,
				dirtyRects
			);
		}

		let environmentDrawn = false;
		if (shouldClearAttachments) {
			const environmentResources =
				await this._sceneResources.getEnvironmentResources(
					frameResources,
					"gbuffer",
					{ sampleCount: 1 },
				);
			if (environmentResources) {
				encoder.beginRenderPass({
					label: "WebGPUEnvironmentDeferred",
					colorAttachments: [
						{
							view: sceneColorAttachment,
							clearValue: { r: 0, g: 0, b: 0, a: 1 },
							loadOp: "clear",
							storeOp: "store",
						},
					],
					depthStencilAttachment: {
						view: depthAttachment,
						depthClearValue: 1,
						depthLoadOp: "clear",
						depthStoreOp: "store",
					},
				});
				encoder.setPipeline(environmentResources.pipeline);
				encoder.setBindingGroup(0, environmentResources.frameBinding);
				encoder.draw(3);
				encoder.endRenderPass();
				environmentDrawn = true;
			}
		}

		const shouldRunEarlyZ =
			allowEarlyZPrepass &&
			this._framePort.isEarlyZPrepassEnabled() &&
			packets.length > 0;
		const earlyZPacketIds =
			shouldRunEarlyZ ?
				await this._recordEarlyZPrepass(
					context,
					packets,
					dirtyRects,
					"gbuffer",
					depthAttachment,
					this._resolveMRTMainDepthLoadOp(
						depthPartialReuseApplied,
						incrementalPartial,
						shouldClearAttachments,
						environmentDrawn,
						false
					)
				)
			:	new Set<string>();
		const earlyZExecuted = earlyZPacketIds.size > 0;
		const gbufferWriteBinding = this._callbacks.getGBufferWriteBinding();
		const colorAttachments = [];
		colorAttachments[GBufferSlot.AlbedoAlpha] = {
			view: targets.gAlbedoAlpha,
			clearValue: { r: 0, g: 0, b: 0, a: 0 },
			loadOp: shouldClearAttachments ? "clear" : "load",
			storeOp: "store",
		};
		colorAttachments[GBufferSlot.NormalRoughMetal] = {
			view: targets.gNormalRoughMetal,
			clearValue: { r: 0.5, g: 0.5, b: 1, a: 0 },
			loadOp: shouldClearAttachments ? "clear" : "load",
			storeOp: "store",
		};
		colorAttachments[GBufferSlot.EmissiveOcclusion] = {
			view: targets.gEmissiveOcclusion,
			clearValue: { r: 0, g: 0, b: 0, a: 1 },
			loadOp: shouldClearAttachments ? "clear" : "load",
			storeOp: "store",
		};
		colorAttachments[GBufferSlot.MotionDepth] = {
			view: targets.gMotionDepth,
			clearValue: { r: 0, g: 0, b: 0, a: 0 },
			loadOp: shouldClearAttachments ? "clear" : "load",
			storeOp: "store",
		};
		if (deferredGBufferLayout === "extended") {
			colorAttachments[GBufferSlot.Specular] = {
				view: targets.gSpecular,
				clearValue: { r: 0, g: 0, b: 0, a: 0 },
				loadOp: shouldClearAttachments ? "clear" : "load",
				storeOp: "store",
			};
			colorAttachments[GBufferSlot.CoatSheen] = {
				view: targets.gCoatSheen,
				clearValue: { r: 0, g: 0, b: 0, a: 0 },
				loadOp: shouldClearAttachments ? "clear" : "load",
				storeOp: "store",
			};
			colorAttachments[GBufferSlot.SheenReflectance] = {
				view: targets.gSheenReflectance,
				clearValue: { r: 0, g: 0, b: 0, a: 0 },
				loadOp: shouldClearAttachments ? "clear" : "load",
				storeOp: "store",
			};
		}

		encoder.beginRenderPass({
			label:
				shouldClearAttachments ?
					"WebGPUGBuffer_Clear"
				:	"WebGPUGBuffer_Load",
			colorAttachments,
			depthStencilAttachment: {
				view: depthAttachment,
				depthClearValue: 1,
				depthLoadOp: this._resolveMRTMainDepthLoadOp(
					depthPartialReuseApplied,
					incrementalPartial,
					shouldClearAttachments,
					environmentDrawn,
					earlyZExecuted
				),
				depthStoreOp: "store",
			},
		});

		await submitWebGPUDraws({
			encoder,
			resources: this._sceneResources,
			frameResources,
			packets,
			dirtyRects,
			selectPacketsForRect: (candidatePackets, rect) =>
				this._framePort.selectPacketsForRect(
					context,
					candidatePackets,
					rect
				),
			resolveDrawOptions: (packet) => ({
				sceneTargetMode: "gbuffer",
				deferredGBufferLayout,
				sampleCount: 1,
				drawMode:
					earlyZExecuted && earlyZPacketIds.has(packet.id) ?
						"early-z-color"
					:	"default",
			}),
			resolveBindings: (draw) => [
				...getDefaultWebGPUDrawBindings(draw),
				{ slot: 3, group: gbufferWriteBinding },
			],
		});

		encoder.endRenderPass();
		return {
			clearSceneColor: shouldClearAttachments && !environmentDrawn,
		};
	}

	private async _recordMRTMainPass(
		context: FrameContext,
		packets: DrawPacket[],
		clearAttachments: boolean,
		allowEarlyZPrepass: boolean,
		frameResources: WebGPUPreparedFrameResources
	): Promise<void> {
		const encoder = this._framePort.getEncoder();
		const targets = this._framePort.getFrameTargets();
		if (!encoder || !targets) {
			return;
		}
		const msaaTargets = this._framePort.getMSAATargets();
		const sceneColorAttachment =
			msaaTargets?.sceneColorMain ?? targets.sceneColorMain;
		const gAlbedoAttachment =
			msaaTargets?.gAlbedoAlpha ?? targets.gAlbedoAlpha;
		const gNormalAttachment =
			msaaTargets?.gNormalRoughMetal ?? targets.gNormalRoughMetal;
		const gEmissiveAttachment =
			msaaTargets?.gEmissiveOcclusion ?? targets.gEmissiveOcclusion;
		const gMotionAttachment =
			msaaTargets?.gMotionDepth ?? targets.gMotionDepth;
		const depthAttachment = msaaTargets?.depth ?? targets.depth;
		const incrementalPartial =
			this._framePort.isIncrementalPartial(context);
		const dirtyRects = this._framePort.resolveDirtyRects(
			context,
			sceneColorAttachment.width,
			sceneColorAttachment.height
		);
		const shouldClearAttachments = clearAttachments && !incrementalPartial;
		let depthPartialReuseApplied = false;
		if (incrementalPartial && dirtyRects.length > 0) {
			depthPartialReuseApplied = await this._depthDirtyClearPass.record(
				encoder,
				depthAttachment,
				TextureFormat.Depth32Float,
				msaaTargets ? this._framePort.getSampleCount() : 1,
				dirtyRects
			);
		}

		let environmentDrawn = false;
		if (shouldClearAttachments) {
			const environmentResources =
				await this._sceneResources.getEnvironmentResources(frameResources, "mrt", {
					sampleCount: this._framePort.getSampleCount(),
				});
			if (environmentResources) {
				encoder.beginRenderPass({
					label: "WebGPUEnvironmentMRT",
					colorAttachments: [
						{
							view: sceneColorAttachment,
							resolveTarget:
								msaaTargets ? targets.sceneColorMain : undefined,
							clearValue: { r: 0, g: 0, b: 0, a: 1 },
							loadOp: "clear",
							storeOp: "store",
						},
					],
					depthStencilAttachment: {
						view: depthAttachment,
						depthClearValue: 1,
						depthLoadOp: "clear",
						depthStoreOp: "store",
					},
				});
				encoder.setPipeline(environmentResources.pipeline);
				encoder.setBindingGroup(0, environmentResources.frameBinding);
				encoder.draw(3);
				encoder.endRenderPass();
				environmentDrawn = true;
			}
		}
		const shouldRunEarlyZ =
			allowEarlyZPrepass &&
			this._framePort.isEarlyZPrepassEnabled() &&
			packets.length > 0;
		const earlyZPacketIds =
			shouldRunEarlyZ ?
				await this._recordEarlyZPrepass(
					context,
					packets,
					dirtyRects,
					"mrt",
					depthAttachment,
					this._resolveMRTMainDepthLoadOp(
						depthPartialReuseApplied,
						incrementalPartial,
						shouldClearAttachments,
						environmentDrawn,
						false
					)
				)
			:	new Set<string>();
		const earlyZExecuted = earlyZPacketIds.size > 0;

		encoder.beginRenderPass({
			label:
				shouldClearAttachments ? "WebGPUMainMRT_Clear" : "WebGPUMainMRT_Load",
			colorAttachments: [
				{
					view: sceneColorAttachment,
					resolveTarget: msaaTargets ? targets.sceneColorMain : undefined,
					clearValue: { r: 0, g: 0, b: 0, a: 1 },
					loadOp: shouldClearAttachments && !environmentDrawn ? "clear" : "load",
					storeOp: "store",
				},
				{
					view: gAlbedoAttachment,
					resolveTarget: msaaTargets ? targets.gAlbedoAlpha : undefined,
					clearValue: { r: 0, g: 0, b: 0, a: 1 },
					loadOp: shouldClearAttachments ? "clear" : "load",
					storeOp: "store",
				},
				{
					view: gNormalAttachment,
					resolveTarget: msaaTargets ? targets.gNormalRoughMetal : undefined,
					clearValue: { r: 0.5, g: 0.5, b: 1, a: 0 },
					loadOp: shouldClearAttachments ? "clear" : "load",
					storeOp: "store",
				},
				{
					view: gEmissiveAttachment,
					resolveTarget: msaaTargets ? targets.gEmissiveOcclusion : undefined,
					clearValue: { r: 0, g: 0, b: 0, a: 1 },
					loadOp: shouldClearAttachments ? "clear" : "load",
					storeOp: "store",
				},
				{
					view: gMotionAttachment,
					resolveTarget: msaaTargets ? targets.gMotionDepth : undefined,
					clearValue: { r: 0, g: 0, b: 0, a: 0 },
					loadOp: shouldClearAttachments ? "clear" : "load",
					storeOp: "store",
				},
			],
			depthStencilAttachment: {
				view: depthAttachment,
				depthClearValue: 1,
				depthLoadOp: this._resolveMRTMainDepthLoadOp(
					depthPartialReuseApplied,
					incrementalPartial,
					shouldClearAttachments,
					environmentDrawn,
					earlyZExecuted
				),
				depthStoreOp: "store",
			},
		});

		await submitWebGPUDraws({
			encoder,
			resources: this._sceneResources,
			frameResources,
			packets,
			dirtyRects,
			selectPacketsForRect: (candidatePackets, rect) =>
				this._framePort.selectPacketsForRect(
					context,
					candidatePackets,
					rect
				),
			resolveDrawOptions: (packet) => ({
				sceneTargetMode: "mrt",
				sampleCount: this._framePort.getSampleCount(),
				drawMode:
					earlyZExecuted && earlyZPacketIds.has(packet.id) ?
						"early-z-color"
					:	"default",
			}),
		});

		encoder.endRenderPass();
	}

	private async _recordColorMainPass(
		context: FrameContext,
		packets: DrawPacket[],
		clearAttachments: boolean,
		allowEarlyZPrepass: boolean,
		frameResources: WebGPUPreparedFrameResources
	): Promise<void> {
		const encoder = this._framePort.getEncoder();
		const targets = this._framePort.getFrameTargets();
		if (!encoder || !targets) {
			return;
		}
		const msaaTargets = this._framePort.getMSAATargets();
		const sceneColorAttachment =
			msaaTargets?.sceneColorMain ?? targets.sceneColorMain;
		const depthAttachment = msaaTargets?.depth ?? targets.depth;
		const incrementalPartial =
			this._framePort.isIncrementalPartial(context);
		const dirtyRects = this._framePort.resolveDirtyRects(
			context,
			sceneColorAttachment.width,
			sceneColorAttachment.height
		);
		const shouldClearAttachments = clearAttachments && !incrementalPartial;
		let depthPartialReuseApplied = false;
		if (incrementalPartial && dirtyRects.length > 0) {
			depthPartialReuseApplied = await this._depthDirtyClearPass.record(
				encoder,
				depthAttachment,
				TextureFormat.Depth32Float,
				msaaTargets ? this._framePort.getSampleCount() : 1,
				dirtyRects
			);
		}

		let environmentDrawn = false;
		if (shouldClearAttachments) {
			const environmentResources =
				await this._sceneResources.getEnvironmentResources(frameResources, "color", {
					sampleCount: this._framePort.getSampleCount(),
				});
			if (environmentResources) {
				encoder.beginRenderPass({
					label: "WebGPUEnvironmentColor",
					colorAttachments: [
						{
							view: sceneColorAttachment,
							resolveTarget:
								msaaTargets ? targets.sceneColorMain : undefined,
							clearValue: { r: 0, g: 0, b: 0, a: 1 },
							loadOp: "clear",
							storeOp: "store",
						},
					],
					depthStencilAttachment: {
						view: depthAttachment,
						depthClearValue: 1,
						depthLoadOp: "clear",
						depthStoreOp: "store",
					},
				});
				encoder.setPipeline(environmentResources.pipeline);
				encoder.setBindingGroup(0, environmentResources.frameBinding);
				encoder.draw(3);
				encoder.endRenderPass();
				environmentDrawn = true;
			}
		}

		const shouldRunEarlyZ =
			allowEarlyZPrepass &&
			this._framePort.isEarlyZPrepassEnabled() &&
			packets.length > 0;
		const earlyZPacketIds =
			shouldRunEarlyZ ?
				await this._recordEarlyZPrepass(
					context,
					packets,
					dirtyRects,
					"color",
					depthAttachment,
					this._resolveMRTMainDepthLoadOp(
						depthPartialReuseApplied,
						incrementalPartial,
						shouldClearAttachments,
						environmentDrawn,
						false
					)
				)
			:	new Set<string>();
		const earlyZExecuted = earlyZPacketIds.size > 0;

		encoder.beginRenderPass({
			label:
				shouldClearAttachments ?
					"WebGPUMainColor_Clear"
				:	"WebGPUMainColor_Load",
			colorAttachments: [
				{
					view: sceneColorAttachment,
					resolveTarget: msaaTargets ? targets.sceneColorMain : undefined,
					clearValue: { r: 0, g: 0, b: 0, a: 1 },
					loadOp: shouldClearAttachments && !environmentDrawn ? "clear" : "load",
					storeOp: "store",
				},
			],
			depthStencilAttachment: {
				view: depthAttachment,
				depthClearValue: 1,
				depthLoadOp: this._resolveMRTMainDepthLoadOp(
					depthPartialReuseApplied,
					incrementalPartial,
					shouldClearAttachments,
					environmentDrawn,
					earlyZExecuted
				),
				depthStoreOp: "store",
			},
		});

		await submitWebGPUDraws({
			encoder,
			resources: this._sceneResources,
			frameResources,
			packets,
			dirtyRects,
			selectPacketsForRect: (candidatePackets, rect) =>
				this._framePort.selectPacketsForRect(
					context,
					candidatePackets,
					rect
				),
			resolveDrawOptions: (packet) => ({
				sceneTargetMode: "color",
				sampleCount: this._framePort.getSampleCount(),
				drawMode:
					earlyZExecuted && earlyZPacketIds.has(packet.id) ?
						"early-z-color"
					:	"default",
			}),
		});

		encoder.endRenderPass();
	}

	private async _recordEarlyZPrepass(
		context: FrameContext,
		packets: DrawPacket[],
		dirtyRects: DirtyRect[],
		sceneTargetMode: WebGPUSceneTargetMode,
		depthAttachment: IRenderTexture,
		depthLoadOp: "clear" | "load"
	): Promise<Set<string>> {
		const prepassedPacketIds = new Set<string>();
		const encoder = this._framePort.getEncoder();
		if (!encoder || packets.length <= 0) {
			return prepassedPacketIds;
		}
		encoder.beginRenderPass({
			label:
				sceneTargetMode === "gbuffer" ? "WebGPUEarlyZPrepassGBuffer"
				: sceneTargetMode === "mrt" ?
					"WebGPUEarlyZPrepassMRT"
				: sceneTargetMode === "color" ?
					"WebGPUEarlyZPrepassColor"
				:	"WebGPUEarlyZPrepassSingle",
			colorAttachments: [],
			depthStencilAttachment: {
				view: depthAttachment,
				depthClearValue: 1,
				depthLoadOp,
				depthStoreOp: "store",
			},
		});

		const submission = await submitWebGPUDraws({
			encoder,
			resources: this._sceneResources,
			frameResources: this._framePort.requireFrameResources(),
			packets,
			dirtyRects,
			selectPacketsForRect: (candidatePackets, rect) =>
				this._framePort.selectPacketsForRect(
					context,
					candidatePackets,
					rect
				),
			resolveDrawOptions: () => ({
				sceneTargetMode,
				drawMode: "early-z-prepass",
				sampleCount:
					sceneTargetMode === "single"
						? 1
						: this._framePort.getSampleCount(),
			}),
		});

		encoder.endRenderPass();
		return submission.submittedPacketIds;
	}

	private _resolveMRTMainDepthLoadOp(
		depthPartialReuseApplied: boolean,
		incrementalPartial: boolean,
		shouldClearAttachments: boolean,
		environmentDrawn: boolean,
		earlyZExecuted: boolean
	): "load" | "clear" {
		if (earlyZExecuted || depthPartialReuseApplied) {
			return "load";
		}
		return incrementalPartial || (shouldClearAttachments && !environmentDrawn) ?
				"clear"
			:	"load";
	}

	private _resolveLegacyMainDepthLoadOp(
		depthPartialReuseApplied: boolean,
		incrementalPartial: boolean,
		shouldClearAttachments: boolean,
		earlyZExecuted: boolean
	): "load" | "clear" {
		if (earlyZExecuted || depthPartialReuseApplied) {
			return "load";
		}
		return incrementalPartial || shouldClearAttachments ? "clear" : "load";
	}
}
