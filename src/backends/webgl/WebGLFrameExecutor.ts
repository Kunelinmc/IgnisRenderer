import type { FrameContext, FramePass } from "../../pipeline/types";
import type {
	LogicalGBufferBridge,
	PostProcessPassCompletion,
	PostProcessPassExecutionContextRequest,
	PostProcessPassRequest,
	PostProcessPassResult,
	PostProcessResourceDescriptor,
	PostProcessResourceHandle,
} from "../../postprocess";
import type { RenderTargetReadbackOptions } from "../../rendering/CustomRenderTargets";
import type { TextureReadbackResult } from "../IComputeRuntime";
import type { ShaderBackendCompileStage, ShaderRuntime } from "../../shaders/runtime";

import {
	WebGLFrameServiceOwner,
	type WebGLFrameServiceOwnerOptions,
} from "./WebGLFrameServiceOwner";
import type { WebGLWarmupCoordinator } from "./WebGLWarmupCoordinator";

export type WebGLFrameExecutorOptions = WebGLFrameServiceOwnerOptions;

/**
 * Thin frame lifecycle facade over context-scoped WebGL services.
 *
 * @internal Owned by `WebGLBackend`; applications must use `Renderer`.
 */
export class WebGLFrameExecutor {
	private readonly _services: WebGLFrameServiceOwner;

	public constructor(
		gl: WebGL2RenderingContext,
		shaderRuntime?: ShaderRuntime,
		shaderCompileStage?: ShaderBackendCompileStage,
		options: WebGLFrameExecutorOptions = {},
	) {
		this._services = new WebGLFrameServiceOwner(
			gl,
			shaderRuntime,
			shaderCompileStage,
			options,
		);
	}

	/** @internal Context-scoped services used by WebGL graph node runtimes. */
	public get services(): WebGLFrameServiceOwner {
		return this._services;
	}

	public get warmupCoordinator(): WebGLWarmupCoordinator {
		return this._services.warmupCoordinator;
	}

	/** @internal Compatibility diagnostic for backend configuration tests. */
	public get _enableEarlyZPrepass(): boolean {
		return this._services.enableEarlyZPrepass;
	}

	public beginFrame(context: FrameContext): void {
		this._services.beginFrame(context);
	}

	public clearFrameTargets(context: FrameContext): void {
		this._services.clearFrameTargets(context);
	}

	public renderEnvironmentNode(context: FrameContext): void {
		this._services.renderEnvironmentNode(context);
	}

	public isOITActive(): boolean {
		return this._services.isOITActive();
	}

	public hasPresentedInFrame(): boolean {
		return this._services.hasPresentedInFrame();
	}

	public collectFrameGraphResources(): readonly string[] {
		return this._services.collectFrameGraphResources();
	}

	public renderShadowNode(context: FrameContext): void {
		this._services.renderShadowNode(context);
	}

	public renderOpaqueDepthPrepass(context: FrameContext): Set<string> {
		return this._services.renderOpaqueDepthPrepass(context);
	}

	public renderOpaqueScene(
		context: FrameContext,
		earlyZPacketIds: ReadonlySet<string>,
	): void {
		this._services.renderOpaqueScene(context, earlyZPacketIds);
	}

	public renderTransparentLegacy(context: FrameContext): void {
		this._services.renderTransparentLegacy(context);
	}

	public prepareOITTransparent(context: FrameContext): void {
		this._services.prepareOITTransparent(context);
	}

	public renderOITTransparentAccum(context: FrameContext): void {
		this._services.renderOITTransparentAccum(context);
	}

	public renderOITTransparentReveal(context: FrameContext): void {
		this._services.renderOITTransparentReveal(context);
	}

	public resolveOIT(context: FrameContext): void {
		this._services.resolveOIT(context);
	}

	public renderOITLegacyTransparent(context: FrameContext): void {
		this._services.renderOITLegacyTransparent(context);
	}

	public prepareOITParticles(): void {
		this._services.prepareOITParticles();
	}

	public renderOITParticleAccum(context: FrameContext): void {
		this._services.renderOITParticleAccum(context);
	}

	public renderOITParticleReveal(context: FrameContext): void {
		this._services.renderOITParticleReveal(context);
	}

	public renderParticlesLegacy(context: FrameContext): void {
		this._services.renderParticlesLegacy(context);
	}

	public renderOITAdditiveParticles(context: FrameContext): void {
		this._services.renderOITAdditiveParticles(context);
	}

	public presentFrame(): void {
		this._services.presentFrame();
	}

	public finishFrame(): void {
		this._services.finishFrame();
	}

	public abortFrame(): void {
		this._services.abortFrame();
	}

	public hasCustomRenderPass(pass: FramePass, context: FrameContext): boolean {
		return this._services.hasCustomRenderPass(pass, context);
	}

	public executeCustomRenderPass(
		pass: FramePass,
		context: FrameContext,
	): Promise<void> {
		return this._services.executeCustomRenderPass(pass, context);
	}

	public readCustomRenderTargetColor(
		id: string,
		attachmentIndex?: number,
		options?: RenderTargetReadbackOptions,
	): Promise<TextureReadbackResult> {
		return this._services.readCustomRenderTargetColor(
			id,
			attachmentIndex,
			options,
		);
	}

	public createPostProcessResource(
		desc: PostProcessResourceDescriptor,
	): PostProcessResourceHandle {
		return this._services.createPostProcessResource(desc);
	}

	public destroyPostProcessResource(handle: PostProcessResourceHandle): void {
		this._services.destroyPostProcessResource(handle);
	}

	public createGBufferBridge(context: FrameContext): LogicalGBufferBridge {
		return this._services.createGBufferBridge(context);
	}

	public getPassExecutionContext(
		request: PostProcessPassExecutionContextRequest,
	): unknown {
		return this._services.getPassExecutionContext(request);
	}

	public beginPostProcessFrame(): void {
		this._services.beginPostProcessFrame();
	}

	public endPostProcessFrame(): void {
		this._services.endPostProcessFrame();
	}

	public abortPostProcessFrame(): void {
		this._services.abortPostProcessFrame();
	}

	public completePostProcessPass(
		request: PostProcessPassRequest,
		result: PostProcessPassResult,
	): PostProcessPassCompletion {
		return this._services.completePostProcessPass(request, result);
	}

	public resize(width: number, height: number): void {
		this._services.resize(width, height);
	}

	public destroy(): void {
		this._services.destroy();
	}
}
