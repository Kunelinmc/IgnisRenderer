import type { FrameContext, FramePass } from "../../../pipeline/types";
import type { FramePreparationRequirements } from "../../../pipeline/FrameRequirements";
import type { PostProcessColorDomain } from "../../../postprocess/PostProcessPass";
import type { WebGLSceneRuntime } from "../WebGLSceneRuntime";
import type { WebGLShadowRuntime } from "../WebGLShadowRuntime";
import type { WebGLTransparencyRuntime } from "../WebGLTransparencyRuntime";
import type { WebGLFrameTargetManager } from "../WebGLFrameTargetManager";
import type { WebGLFrameSession } from "../WebGLFrameSession";
import type { WebGLFullscreenRenderer } from "../WebGLFullscreenRenderer";
import type {
	WebGLPostProcessServices,
} from "../WebGLPostProcessServices";

/**
 * Lifecycle coordination owned by the frame services container. The adapter
 * forwards graph-facing lifecycle calls back to the owner without requiring
 * its concrete type.
 */
export interface WebGLFrameLifecycleCoordinator {
	beginFrame(context: FrameContext, materialGBufferRequested: boolean): void;
	beginTemporalFrame(
		context: FrameContext,
		frameRequirements: FramePreparationRequirements,
	): void;
	finishFrame(): void;
	abortFrame(): void;
	hasCustomRenderPass(pass: FramePass, context: FrameContext): boolean;
	executeCustomRenderPass(
		pass: FramePass,
		context: FrameContext,
	): Promise<void>;
}

/**
 * Implements the WebGL frame-graph execution facade by delegating each node
 * kind directly to its owning runtime. Keeps renderer-level pass orchestration
 * out of the frame services container.
 */
export class WebGLFrameNodeAdapter {
	private readonly _lifecycle: WebGLFrameLifecycleCoordinator;
	private readonly _scene: WebGLSceneRuntime;
	private readonly _shadow: WebGLShadowRuntime;
	private readonly _transparency: WebGLTransparencyRuntime;
	private readonly _targets: WebGLFrameTargetManager;
	private readonly _session: WebGLFrameSession;
	private readonly _fullscreen: WebGLFullscreenRenderer;
	private readonly _postProcess: WebGLPostProcessServices;

	constructor(services: {
		lifecycle: WebGLFrameLifecycleCoordinator;
		scene: WebGLSceneRuntime;
		shadow: WebGLShadowRuntime;
		transparency: WebGLTransparencyRuntime;
		targets: WebGLFrameTargetManager;
		session: WebGLFrameSession;
		fullscreen: WebGLFullscreenRenderer;
		postProcess: WebGLPostProcessServices;
	}) {
		this._lifecycle = services.lifecycle;
		this._scene = services.scene;
		this._shadow = services.shadow;
		this._transparency = services.transparency;
		this._targets = services.targets;
		this._session = services.session;
		this._fullscreen = services.fullscreen;
		this._postProcess = services.postProcess;
	}

	public beginFrame(
		context: FrameContext,
		materialGBufferRequested: boolean,
	): void {
		this._lifecycle.beginFrame(context, materialGBufferRequested);
	}

	public beginTemporalFrame(
		context: FrameContext,
		frameRequirements: FramePreparationRequirements,
	): void {
		this._lifecycle.beginTemporalFrame(context, frameRequirements);
	}

	public finishFrame(): void {
		this._lifecycle.finishFrame();
	}

	public abortFrame(): void {
		this._lifecycle.abortFrame();
	}

	public hasCustomRenderPass(pass: FramePass, context: FrameContext): boolean {
		return this._lifecycle.hasCustomRenderPass(pass, context);
	}

	public executeCustomRenderPass(
		pass: FramePass,
		context: FrameContext,
	): Promise<void> {
		return this._lifecycle.executeCustomRenderPass(pass, context);
	}

	public isOITActive(): boolean {
		return this._transparency.isActive();
	}

	public hasPresentedInFrame(): boolean {
		return this._session.presented;
	}

	public setPostProcessInitialColorDomain(domain: PostProcessColorDomain): void {
		this._postProcess.setInitialColorDomain(domain);
	}

	public collectFrameGraphResources(): readonly string[] {
		const resources = new Set(this._targets.collectGraphResources());
		for (const descriptor of this._shadow.describeGraphResources().resources) {
			resources.add(descriptor.id);
		}
		return Array.from(resources);
	}

	public collectFrameGraphResourceCatalog(includeShadowResources = true) {
		const frameCatalog = this._targets.collectGraphResourceCatalog();
		if (!includeShadowResources) return frameCatalog;
		const shadowCatalog = this._shadow.describeGraphResources();
		return Object.freeze({
			resources: Object.freeze([
				...frameCatalog.resources,
				...shadowCatalog.resources,
			]),
			bindings: Object.freeze([
				...frameCatalog.bindings,
				...shadowCatalog.bindings,
			]),
		});
	}

	public clearFrameTargets(context: FrameContext): void {
		this._scene.clearFrameTargets(context);
	}

	public renderEnvironmentNode(context: FrameContext): void {
		this._scene.renderEnvironment(context);
	}

	public renderShadowNode(context: FrameContext): void {
		this._shadow.renderPreparedFrame(context);
	}

	public renderOpaqueDepthPrepass(context: FrameContext): Set<string> {
		return this._scene.renderOpaqueDepthPrepass(context);
	}

	public renderOpaqueScene(
		context: FrameContext,
		earlyZPacketIds: ReadonlySet<string>,
	): void {
		this._scene.renderOpaque(context, earlyZPacketIds);
	}

	public renderTransparentLegacy(context: FrameContext): void {
		this._transparency.renderLegacyTransparent(context);
	}

	public prepareTransmissionDepth(context: FrameContext): void {
		this._transparency.prepareTransmissionDepth(context);
	}

	public renderLegacyTransparentSegment(
		context: FrameContext,
		start: number,
		end: number,
	): void {
		this._transparency.renderLegacyTransparentSegment(context, start, end);
	}

	public copyTransmissionBackground(context: FrameContext): void {
		this._transparency.copyTransmissionBackground(context);
	}

	public renderTransmissionPacket(context: FrameContext, index: number): void {
		this._transparency.renderTransmissionPacket(context, index);
	}

	public prepareOITTransparent(context: FrameContext): void {
		this._transparency.prepareTransparent(context);
	}

	public renderOITTransparentAccum(context: FrameContext): void {
		this._transparency.renderTransparentAccum(context);
	}

	public renderOITTransparentReveal(context: FrameContext): void {
		this._transparency.renderTransparentReveal(context);
	}

	public copySceneColorForOIT(context: FrameContext): void {
		this._transparency.copySceneColor(context);
	}

	public resolveOIT(context: FrameContext): void {
		this._transparency.resolve(context);
	}

	public renderOITLegacyTransparent(context: FrameContext): void {
		this._transparency.renderLegacy(context);
	}

	public prepareOITParticles(): void {
		this._transparency.prepareParticles();
	}

	public renderOITParticleAccum(context: FrameContext): void {
		this._transparency.renderParticleAccum(context);
	}

	public renderOITParticleReveal(context: FrameContext): void {
		this._transparency.renderParticleReveal(context);
	}

	public renderParticlesLegacy(context: FrameContext): void {
		this._transparency.renderParticlesLegacy(context);
	}

	public renderOITAdditiveParticles(context: FrameContext): void {
		this._transparency.renderAdditiveParticles(context);
	}

	public presentFrame(): void {
		if (!this._session.presented) {
			this._fullscreen.present(this._session.context, true);
		}
	}
}
