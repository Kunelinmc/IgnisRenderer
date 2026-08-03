import type { DrawPacket, FrameContext } from "../../../pipeline/types";
import { ParticleBlendMode } from "../../../particles";
import { ShaderSource } from "../../../shaders/ShaderSource";
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
import { submitWebGPUDraws } from "../WebGPUDrawSubmission";
import type {
	WebGPUParticleBillboardRenderer,
	WebGPUSceneResourceProvider,
} from "../WebGPUResourceContracts";
import type { WebGPUSceneTargetMode } from "../WebGPUScenePassDescriptors";
import type { WebGPUFrameGraphRecordingContext } from "./WebGPUFrameGraphRecordingContext";
import type { WebGPUFrameNodeRuntime } from "./WebGPUFrameNodeRuntimes";
import type { WebGPUFrameSession } from "./WebGPUFrameSession";
import type { WebGPUFrameHost } from "./WebGPUFrameHost";
import type { WebGPUScenePassRecorder } from "./WebGPUScenePassRecorder";

export interface WebGPUFrameDiagnosticSink {
	warnOnce(code: string, message: string, cause?: unknown): void;
}

/**
 * Owns WebGPU transparent rendering, OIT resolve resources, and transparency
 * graph-node execution for one backend runtime.
 *
 * @internal Owned by the WebGPU frame runtime. Applications should use
 * `Renderer.renderFrame()` instead.
 */
export class WebGPUTransparencyRuntime implements WebGPUFrameNodeRuntime {
	public readonly id = "transparency";
	public readonly executors = {
		"transparent-forward": async (_node: unknown, session: WebGPUFrameSession) =>
			this._recordTransparentForward(session),
		"oit-prepare": async (_node: unknown, session: WebGPUFrameSession) =>
			this._prepareOIT(session),
		"oit-clear": async (_node: unknown, session: WebGPUFrameSession) =>
			this._clearOITTargets(session),
		"oit-mesh-accumulate": async (_node: unknown, session: WebGPUFrameSession) =>
			this._recordOITMeshAccumulation(session),
		"oit-particle-accumulate": async (_node: unknown, session: WebGPUFrameSession) =>
			this._recordOITParticleAccumulation(session),
		"oit-resolve": async (_node: unknown, session: WebGPUFrameSession) =>
			this._resolveOIT(session),
		transmission: async (_node: unknown, session: WebGPUFrameSession) =>
			this._recordTransmission(session),
		"particle-alpha-forward": async (_node: unknown, session: WebGPUFrameSession) =>
			this._recordLegacyParticles(session, [ParticleBlendMode.Alpha]),
		"particle-additive": async (_node: unknown, session: WebGPUFrameSession) =>
			this._recordLegacyParticles(session, [ParticleBlendMode.Additive]),
	};

	private _resolveShaderModule: IShaderModule | null = null;
	private _resolvePipeline: IRenderPipeline | null = null;
	private _resolveSampler: ISampler | null = null;
	private _resolveBinding: IBindingGroup | null = null;
	private _resolveBindingScene: IRenderTexture | null = null;
	private _resolveBindingAccum: IRenderTexture | null = null;
	private _resolveBindingReveal: IRenderTexture | null = null;

	public constructor(
		private readonly _host: WebGPUFrameHost,
		private readonly _sceneResources: WebGPUSceneResourceProvider,
		private readonly _particleRenderer: WebGPUParticleBillboardRenderer,
		private readonly _recordingContext: WebGPUFrameGraphRecordingContext,
		private readonly _sceneRecorder: WebGPUScenePassRecorder,
		private readonly _diagnostics: WebGPUFrameDiagnosticSink,
	) {}

	public beginFrame(_context: FrameContext): void {}

	public invalidateFrameResources(): void {
		this._destroyBindingGroup(this._resolveBinding);
		this._resolveBinding = null;
		this._resolveBindingScene = null;
		this._resolveBindingAccum = null;
		this._resolveBindingReveal = null;
	}

	public onShaderRuntimeChanged(): void {
		this.invalidateFrameResources();
		this._destroyManagedResource(this._resolvePipeline);
		this._destroyManagedResource(this._resolveSampler);
		this._destroyManagedResource(this._resolveShaderModule);
		this._resolvePipeline = null;
		this._resolveSampler = null;
		this._resolveShaderModule = null;
	}

	public destroy(): void {
		this.onShaderRuntimeChanged();
	}

	private async _recordTransparentForward(session: WebGPUFrameSession): Promise<void> {
		const analysis = this._requireTransparencyAnalysis(session);
		if (analysis.legacyTransparentPackets.length <= 0) return;
		await this._sceneRecorder.recordMainPass(
			session.context,
			analysis.legacyTransparentPackets.slice(),
			false,
			false,
		);
	}

	private _prepareOIT(session: WebGPUFrameSession): void {
		if (session.transparencyMode !== "oit") return;
		const encoder = session.encoder;
		const targets = this._recordingContext.getFrameTargets();
		if (!encoder || !targets?.oitSceneColorCopy) {
			this._fallbackToLegacy(session, "webgpu-oit-disabled-runtime", "WebGPU OIT frame targets are unavailable; using legacy transparent rendering.");
			return;
		}
		try {
			encoder.copyTextureToTexture(
				{ texture: targets.sceneColorMain },
				{ texture: targets.oitSceneColorCopy },
				{
					width: Math.max(1, this._recordingContext.getTargetWidth()),
					height: Math.max(1, this._recordingContext.getTargetHeight()),
					depthOrArrayLayers: 1,
				},
			);
		} catch (error) {
			this._fallbackToLegacy(
				session,
				"webgpu-oit-copy-scene-color-failed",
				"WebGPU OIT scene-color copy failed; using legacy transparent rendering.",
				error,
			);
		}
	}

	private _clearOITTargets(session: WebGPUFrameSession): void {
		if (session.transparencyMode !== "oit") return;
		const encoder = session.encoder;
		const targets = this._recordingContext.getFrameTargets();
		if (!encoder || !targets?.oitAccum || !targets.oitReveal) {
			this._fallbackToLegacy(session, "webgpu-oit-disabled-runtime", "WebGPU OIT accumulation targets are unavailable; using legacy transparent rendering.");
			return;
		}
		encoder.beginRenderPass({
			label: "WebGPUOITClear",
			colorAttachments: [
				{ view: targets.oitAccum, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" },
				{ view: targets.oitReveal, clearValue: { r: 1, g: 1, b: 1, a: 1 }, loadOp: "clear", storeOp: "store" },
			],
		});
		encoder.endRenderPass();
	}

	private async _recordOITMeshAccumulation(session: WebGPUFrameSession): Promise<void> {
		const analysis = this._requireTransparencyAnalysis(session);
		if (analysis.oitPackets.length <= 0) return;
		if (session.transparencyMode !== "oit") {
			await this._sceneRecorder.recordMainPass(session.context, analysis.oitPackets.slice(), false, false);
			return;
		}
		const encoder = session.encoder;
		const targets = this._recordingContext.getFrameTargets();
		if (!encoder || !targets?.oitAccum || !targets.oitReveal) {
			this._fallbackToLegacy(session, "webgpu-oit-disabled-runtime", "WebGPU OIT accumulation targets are unavailable; using legacy transparent rendering.");
			await this._sceneRecorder.recordMainPass(session.context, analysis.oitPackets.slice(), false, false);
			return;
		}
		const frameResources = this._recordingContext.requireFrameResources();
		await this._sceneResources.buildClusteredLighting(encoder, frameResources);
		const depthAttachment = this._recordingContext.getMSAATargets()?.depth ?? targets.depth;
		encoder.beginRenderPass({
			label: "WebGPUOITMeshAccumulate",
			colorAttachments: [
				{ view: targets.oitAccum, loadOp: "load", storeOp: "store" },
				{ view: targets.oitReveal, loadOp: "load", storeOp: "store" },
			],
			depthStencilAttachment: {
				view: depthAttachment,
				depthLoadOp: "load",
				depthStoreOp: "store",
			},
		});
		const dirtyRects = this._recordingContext.resolveDirtyRects(
			session.context,
			targets.sceneColorMain.width,
			targets.sceneColorMain.height,
		);
		await submitWebGPUDraws({
			encoder,
			resources: this._sceneResources,
			frameResources,
			packets: analysis.oitPackets.slice(),
			dirtyRects,
			selectPacketsForRect: (packets, rect) =>
				this._recordingContext.selectTransparentSubsetForRect(session.context, packets, rect),
			resolveDrawOptions: () => ({
				sceneTargetMode: this._resolveSceneTargetMode(),
				transparentPipelineMode: "oit",
			}),
		});
		encoder.endRenderPass();
	}

	private async _recordOITParticleAccumulation(session: WebGPUFrameSession): Promise<void> {
		if (session.transparencyMode !== "oit") {
			await this._recordLegacyParticles(session, [ParticleBlendMode.Alpha]);
			return;
		}
		const encoder = session.encoder;
		const targets = this._recordingContext.getFrameTargets();
		if (!encoder || !targets?.oitAccum || !targets.oitReveal) {
			this._fallbackToLegacy(session, "webgpu-oit-disabled-runtime", "WebGPU OIT accumulation targets are unavailable; using legacy transparent rendering.");
			await this._recordLegacyParticles(session, [ParticleBlendMode.Alpha]);
			return;
		}
		await this._particleRenderer.renderParticles(
			encoder,
			session.context,
			{
				label: "WebGPUParticlesOIT",
				sampleCount: 1,
				colorAttachments: [
					{ view: targets.oitAccum, loadOp: "load", storeOp: "store" },
					{ view: targets.oitReveal, loadOp: "load", storeOp: "store" },
				],
				depth: targets.depth,
			},
			this._recordingContext.requireFrameResources(),
			this._resolveSceneTargetMode(),
			{ includeBlendModes: [ParticleBlendMode.Alpha], pipelineMode: "oit" },
		);
	}

	private async _resolveOIT(session: WebGPUFrameSession): Promise<void> {
		if (session.transparencyMode !== "oit") return;
		const encoder = session.encoder;
		const targets = this._recordingContext.getFrameTargets();
		if (!encoder || !targets?.oitSceneColorCopy || !targets.oitAccum || !targets.oitReveal) {
			this._fallbackToLegacy(session, "webgpu-oit-disabled-runtime", "WebGPU OIT resolve targets are unavailable; using legacy transparent rendering.");
			return;
		}
		await this._ensureResolveResources();
		if (!this._resolvePipeline || !this._resolveSampler) return;
		if (
			!this._resolveBinding ||
			this._resolveBindingScene !== targets.oitSceneColorCopy ||
			this._resolveBindingAccum !== targets.oitAccum ||
			this._resolveBindingReveal !== targets.oitReveal
		) {
			this.invalidateFrameResources();
			this._resolveBinding = this._host.createBindingGroup({
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
			colorAttachments: [{ view: targets.sceneColorMain, loadOp: "load", storeOp: "store" }],
		});
		encoder.setPipeline(this._resolvePipeline);
		encoder.setBindingGroup(0, this._resolveBinding);
		for (const rect of this._recordingContext.resolveDirtyRects(
			session.context,
			targets.sceneColorMain.width,
			targets.sceneColorMain.height,
		)) {
			encoder.setScissorRect?.(rect.x, rect.y, rect.width, rect.height);
			encoder.draw(3);
		}
		encoder.endRenderPass();
	}

	private async _recordTransmission(session: WebGPUFrameSession): Promise<void> {
		const packets = this._requireTransparencyAnalysis(session).transmissionPackets;
		if (packets.length > 0) {
			await this._sceneRecorder.drawTransmissionPackets(session.context, packets.slice());
		}
	}

	private async _recordLegacyParticles(
		session: WebGPUFrameSession,
		includeBlendModes: readonly ParticleBlendMode[],
	): Promise<void> {
		const encoder = session.encoder;
		const targets = this._recordingContext.getFrameTargets();
		if (!encoder || !targets) return;
		const msaaTargets = this._recordingContext.getMSAATargets();
		await this._particleRenderer.renderParticles(
			encoder,
			session.context,
			{
				label: includeBlendModes[0] === ParticleBlendMode.Additive ?
					"WebGPUParticlesMRT_Additive" : "WebGPUParticlesMRT_Alpha",
				sampleCount: this._recordingContext.getTargetMSAASampleCount(),
				colorAttachments: [{
					view: msaaTargets?.sceneColorMain ?? targets.sceneColorMain,
					resolveTarget: msaaTargets ? targets.sceneColorMain : undefined,
					loadOp: "load",
					storeOp: "store",
				}],
				depth: msaaTargets?.depth ?? targets.depth,
			},
			this._recordingContext.requireFrameResources(),
			this._resolveSceneTargetMode(),
			{ includeBlendModes, pipelineMode: "legacy" },
		);
	}

	private async _ensureResolveResources(): Promise<void> {
		if (!this._resolveShaderModule) {
			const source = await ShaderSource.load("webgpu.utility.oitResolve.composite");
			this._resolveShaderModule = await this._host.createShaderModule({
				label: "WebGPUOITResolveShader", code: source.code, sourceMap: source.sourceMap,
				language: "wgsl", stage: "unknown", sourceKind: "postprocess",
			});
		}
		if (!this._resolvePipeline) {
			this._resolvePipeline = await this._host.createPipeline({
				label: "WebGPUOITResolvePipeline",
				vertex: { module: this._resolveShaderModule, entryPoint: "vsMain" },
				fragment: { module: this._resolveShaderModule, entryPoint: "fsMain", targets: [{ format: TextureFormat.RGBA16Float }] },
				primitive: { topology: "triangle-list" as any, cullMode: "none", frontFace: "ccw" },
			} as any);
		}
		if (!this._resolveSampler) {
			this._resolveSampler = this._host.createSampler({
				label: "WebGPUOITResolveSampler", magFilter: FilterMode.Linear,
				minFilter: FilterMode.Linear, mipmapFilter: FilterMode.Linear,
				addressModeU: AddressMode.ClampToEdge, addressModeV: AddressMode.ClampToEdge,
			});
		}
	}

	private _fallbackToLegacy(
		session: WebGPUFrameSession,
		code: string,
		message: string,
		cause?: unknown,
	): void {
		if (session.transparencyMode === "legacy-runtime-fallback") return;
		session.transparencyMode = "legacy-runtime-fallback";
		this._diagnostics.warnOnce(code, message, cause);
	}

	private _requireTransparencyAnalysis(session: WebGPUFrameSession) {
		if (!session.analysis) throw new Error("WebGPU transparency execution requires frame analysis.");
		return session.analysis.transparency;
	}

	private _resolveSceneTargetMode(): Exclude<WebGPUSceneTargetMode, "single"> {
		return this._recordingContext.getSceneTargetMode() === "color" ? "color" : "mrt";
	}

	private _destroyBindingGroup(group: IBindingGroup | null): void {
		const destroy = (group as { destroy?: () => void } | null)?.destroy;
		if (typeof destroy === "function") destroy.call(group);
	}

	private _destroyManagedResource(resource: unknown): void {
		const destroy = (resource as { destroy?: () => void } | null)?.destroy;
		if (typeof destroy === "function") destroy.call(resource);
	}
}
