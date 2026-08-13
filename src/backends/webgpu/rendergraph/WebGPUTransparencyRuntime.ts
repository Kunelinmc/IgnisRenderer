import type { DrawPacket, FrameContext } from "../../../pipeline/types";
import { ParticleBlendMode } from "../../../particles";
import { materialUsesTransmission } from "../../../materials/transparency";
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
import type { WebGPUFrameGraphModule } from "./WebGPUFrameGraphModule";
import type {
	WebGPUFrameGraphContribution,
	WebGPUFrameModulePlanningInput,
} from "./WebGPUFrameGraphModule";
import {
	defineWebGPUFrameMessage,
	type WebGPUFrameMessageHandler,
} from "./WebGPUFrameMessage";
import {
	WEBGPU_FRAME_CONFIGURATION_DEMAND_MESSAGE,
	WEBGPU_FRAME_CONFIGURATION_REQUEST_MESSAGE,
	WEBGPU_FRAME_LOGICAL_RESOURCES,
	WEBGPU_FRAME_FEATURE_STATES,
	WEBGPU_FRAME_CONTEXT_MESSAGE,
	WEBGPU_FRAME_PACKETS_MESSAGE,
} from "./WebGPUFrameMessages";
import {
	WEBGPU_MRT_COLOR_BYTES_PER_SAMPLE,
	WEBGPU_MRT_COLOR_TARGET_COUNT,
} from "../constants";
import { WebGPUTransparencyFramePlanner } from "./WebGPUTransparencyFramePlanner";
import type { WebGPUFrameSession } from "./WebGPUFrameSession";
import type { WebGPUFrameHost } from "./WebGPUFrameHost";

interface WebGPUTransparencyScenePort {
	recordMainPass(
		context: FrameContext,
		packets: DrawPacket[],
		clearAttachments: boolean,
		allowEarlyZPrepass: boolean,
	): Promise<void>;
	drawTransmissionPackets(context: FrameContext, packets: DrawPacket[]): Promise<void>;
}

export interface WebGPUFrameDiagnosticSink {
	warnOnce(code: string, message: string, cause?: unknown): void;
}

export interface WebGPUTransparencyAnalysis {
	readonly oitPackets: readonly DrawPacket[];
	readonly transmissionPackets: readonly DrawPacket[];
	readonly legacyTransparentPackets: readonly DrawPacket[];
	readonly hasAlphaBillboardParticles: boolean;
	readonly hasAdditiveBillboardParticles: boolean;
	readonly hasOITContributors: boolean;
}

export const WEBGPU_TRANSPARENCY_FEATURE_ANALYSIS =
	defineWebGPUFrameMessage<WebGPUTransparencyAnalysis>({
		id: "webgpu:transparency-analysis",
		ownerId: "transparency",
		phase: "analysis",
	});

export function analyzeWebGPUTransparency(
	context: FrameContext,
	transparentPackets: readonly DrawPacket[],
): WebGPUTransparencyAnalysis {
	const oitPackets: DrawPacket[] = [];
	const transmissionPackets: DrawPacket[] = [];
	for (const packet of transparentPackets) {
		if (materialUsesTransmission(packet.material)) transmissionPackets.push(packet);
		else oitPackets.push(packet);
	}
	let hasAlphaBillboardParticles = false;
	let hasAdditiveBillboardParticles = false;
	for (const system of context.scene.particleSystems ?? []) {
		if (system.visible === false) continue;
		if (!system.templates) {
			hasAlphaBillboardParticles = true;
			hasAdditiveBillboardParticles = true;
			continue;
		}
		for (const template of system.templates) {
			if (template.shape.kind !== "billboard") continue;
			if (template.shape.blendMode === ParticleBlendMode.Additive) {
				hasAdditiveBillboardParticles = true;
			} else {
				hasAlphaBillboardParticles = true;
			}
		}
	}
	return {
		oitPackets,
		transmissionPackets,
		legacyTransparentPackets: oitPackets,
		hasAlphaBillboardParticles,
		hasAdditiveBillboardParticles,
		hasOITContributors: oitPackets.length > 0 || hasAlphaBillboardParticles,
	};
}

/**
 * Owns WebGPU transparent rendering, OIT resolve resources, and transparency
 * graph-node execution for one backend runtime.
 *
 * @internal Owned by the WebGPU frame runtime. Applications should use
 * `Renderer.renderFrame()` instead.
 */
export class WebGPUTransparencyRuntime implements WebGPUFrameGraphModule {
	public readonly id = "transparency";
	public readonly planningInputs = [{
		descriptor: WEBGPU_FRAME_CONFIGURATION_DEMAND_MESSAGE,
		required: false,
	}] as const;
	public readonly messageHandlers: readonly WebGPUFrameMessageHandler[] = [{
		id: "analyze",
		moduleId: this.id,
		phase: "analysis",
		inputs: [
			{ descriptor: WEBGPU_FRAME_CONTEXT_MESSAGE },
			{ descriptor: WEBGPU_FRAME_PACKETS_MESSAGE },
		],
		outputs: [WEBGPU_TRANSPARENCY_FEATURE_ANALYSIS],
		run: (messages, publisher) => publisher.publish(
			WEBGPU_TRANSPARENCY_FEATURE_ANALYSIS,
			analyzeWebGPUTransparency(
				messages.get(WEBGPU_FRAME_CONTEXT_MESSAGE),
				messages.get(WEBGPU_FRAME_PACKETS_MESSAGE).transparent,
			),
		),
	}, {
		id: "configure",
		moduleId: this.id,
		phase: "configuration",
		inputs: [
			{ descriptor: WEBGPU_TRANSPARENCY_FEATURE_ANALYSIS },
			{ descriptor: WEBGPU_FRAME_CONFIGURATION_REQUEST_MESSAGE },
		],
		outputs: [WEBGPU_FRAME_CONFIGURATION_DEMAND_MESSAGE],
		run: (messages, publisher) => {
			const analysis = messages.get(WEBGPU_TRANSPARENCY_FEATURE_ANALYSIS);
			const request = messages.get(WEBGPU_FRAME_CONFIGURATION_REQUEST_MESSAGE);
			const mrtSupported =
				request.capabilities.maxColorAttachments >= WEBGPU_MRT_COLOR_TARGET_COUNT &&
				request.capabilities.maxColorAttachmentBytesPerSample >=
					WEBGPU_MRT_COLOR_BYTES_PER_SAMPLE;
			const requested = request.context.features.enableOIT === true;
			const active =
				mrtSupported && request.options.samplePlan.sampleCount === 1 &&
				request.options.supportsInFrameTextureCopy && requested &&
				analysis.hasOITContributors;
			const diagnostics = [];
			if (requested && analysis.hasOITContributors && !active) {
				diagnostics.push({
					code: request.options.samplePlan.sampleCount > 1
						? "webgpu-oit-disabled-msaa"
						: !mrtSupported
							? "webgpu-oit-disabled-mrt-unavailable"
							: "webgpu-oit-disabled-runtime",
					message: request.options.samplePlan.sampleCount > 1
						? "WebGPU OIT v1 only supports sampleCount=1; falling back to legacy transparent rendering."
						: !mrtSupported
							? "WebGPU OIT requires MRT scene targets; falling back to legacy transparent rendering."
							: "WebGPU OIT requires in-frame texture-copy support; falling back to legacy transparent rendering.",
				});
			}
			publisher.publish(WEBGPU_FRAME_CONFIGURATION_DEMAND_MESSAGE, {
				source: this.id,
				targetClass: active ? "color" :
					analysis.transmissionPackets.length > 0 ? "color" : "single",
				featureStates: {
					[WEBGPU_FRAME_FEATURE_STATES.oitActive]: active,
					[WEBGPU_FRAME_FEATURE_STATES.transparencyMode]:
						active ? "oit" : "legacy",
				},
				resources: active
					? [{ id: WEBGPU_FRAME_LOGICAL_RESOURCES.oitTargets }]
					: [],
				hasOITMeshContributors: analysis.oitPackets.length > 0,
				hasTransmissionPackets: analysis.transmissionPackets.length > 0,
				hasAlphaBillboardParticles: analysis.hasAlphaBillboardParticles,
				hasAdditiveBillboardParticles: analysis.hasAdditiveBillboardParticles,
				diagnostics,
			});
		},
	}];
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
	private readonly _planner = new WebGPUTransparencyFramePlanner();

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
		private readonly _scene: WebGPUTransparencyScenePort,
		private readonly _diagnostics: WebGPUFrameDiagnosticSink,
	) {}

	public planStage(
		input: WebGPUFrameModulePlanningInput,
	): readonly WebGPUFrameGraphContribution[] {
		const demand = input.messages
			.getAll(WEBGPU_FRAME_CONFIGURATION_DEMAND_MESSAGE)
			.find((candidate) => candidate.source === this.id);
		const nodes = this._planner.plan(
			input.pass,
			input.context,
			{
				...input.state,
				hasOITMeshContributors: demand?.hasOITMeshContributors === true,
				hasTransmissionPackets: demand?.hasTransmissionPackets === true,
				hasAlphaBillboardParticles:
					demand?.hasAlphaBillboardParticles === true,
				hasAdditiveBillboardParticles:
					demand?.hasAdditiveBillboardParticles === true,
			},
		);
		return nodes.length > 0 ? [{ lane: "transparent", nodes }] : [];
	}

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
		await this._scene.recordMainPass(
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
			await this._scene.recordMainPass(session.context, analysis.oitPackets.slice(), false, false);
			return;
		}
		const encoder = session.encoder;
		const targets = this._recordingContext.getFrameTargets();
		if (!encoder || !targets?.oitAccum || !targets.oitReveal) {
			this._fallbackToLegacy(session, "webgpu-oit-disabled-runtime", "WebGPU OIT accumulation targets are unavailable; using legacy transparent rendering.");
			await this._scene.recordMainPass(session.context, analysis.oitPackets.slice(), false, false);
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
				sampleCount: 1,
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
			await this._scene.drawTransmissionPackets(session.context, packets.slice());
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
				sampleCount: this._recordingContext.getSampleCount(),
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
		return session.messages.get(WEBGPU_TRANSPARENCY_FEATURE_ANALYSIS);
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
