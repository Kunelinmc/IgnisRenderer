import type { DrawPacket, FrameContext } from "../../pipeline/types";

import type { WebGLFrameTargetManager } from "./WebGLFrameTargetManager";
import {
	renderWebGLEarlyZPrepass,
	renderWebGLPackets,
	type WebGLSceneRenderOptions,
	type WebGLScenePassHost,
} from "./WebGLScenePass";

const TRANSPARENT_SCENE_CLEAR = new Float32Array([0, 0, 0, 0]);

export interface WebGLSceneRuntimeServices {
	readonly gl: WebGL2RenderingContext;
	readonly host: WebGLScenePassHost;
	readonly targets: WebGLFrameTargetManager;
	readonly enableEarlyZPrepass: boolean;
	getWidth(): number;
	getHeight(): number;
	isIncrementalPartial(context: FrameContext): boolean;
	resolveDirtyRects(
		context: FrameContext,
		width: number,
		height: number,
	): Array<{ x: number; y: number; width: number; height: number }>;
	setScissorRect(
		x: number,
		y: number,
		width: number,
		height: number,
		viewportHeight: number,
	): void;
	renderEnvironment(context: FrameContext): void;
	renderLegacyTransparent(context: FrameContext): void;
}

/** Owns scene-node execution and frame-to-frame model transform state. */
export class WebGLSceneRuntime {
	public readonly modelMatrixCache = new Map<string, Float32Array>();
	public readonly modelMatrixKeysThisFrame = new Set<string>();
	private readonly _services: WebGLSceneRuntimeServices;

	public constructor(services: WebGLSceneRuntimeServices) {
		this._services = services;
	}

	public beginFrame(): void {
		this.modelMatrixKeysThisFrame.clear();
	}

	public abortFrame(): void {
		this.modelMatrixKeysThisFrame.clear();
	}

	public clearFrameTargets(context: FrameContext): void {
		const { gl, targets } = this._services;
		const width = this._services.getWidth();
		const height = this._services.getHeight();
		gl.bindFramebuffer(gl.FRAMEBUFFER, targets._sceneFramebuffer);
		gl.viewport(0, 0, width, height);
		gl.disable(gl.BLEND);
		gl.enable(gl.DEPTH_TEST);
		gl.depthMask(true);

		const drawBuffers = targets._materialGBufferEnabled ?
			[
				gl.COLOR_ATTACHMENT0,
				gl.COLOR_ATTACHMENT1,
				gl.COLOR_ATTACHMENT2,
				gl.COLOR_ATTACHMENT3,
				gl.COLOR_ATTACHMENT4,
			]
		: targets._sceneNormalTexture ?
			[gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1, gl.COLOR_ATTACHMENT2]
		: [gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1];
		gl.drawBuffers(drawBuffers);
		gl.clearColor(0, 0, 0, 1);
		gl.clearDepth(1);

		if (!this._services.isIncrementalPartial(context)) {
			gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
			if (context.presentationAlphaMode === "premultiplied") {
				gl.clearBufferfv(gl.COLOR, 0, TRANSPARENT_SCENE_CLEAR);
			}
			return;
		}
		const dirtyRects = this._services.resolveDirtyRects(context, width, height);
		if (dirtyRects.length === 0) return;
		gl.enable(gl.SCISSOR_TEST);
		for (const rect of dirtyRects) {
			this._services.setScissorRect(
				rect.x,
				rect.y,
				rect.width,
				rect.height,
				height,
			);
			gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
			if (context.presentationAlphaMode === "premultiplied") {
				gl.clearBufferfv(gl.COLOR, 0, TRANSPARENT_SCENE_CLEAR);
			}
		}
		gl.disable(gl.SCISSOR_TEST);
	}

	public renderEnvironment(context: FrameContext): void {
		if (
			this._services.isIncrementalPartial(context) ||
			!context.features.enableEnvironment ||
			!context.scene.environment.backgroundEnabled ||
			!context.scene.environment.backgroundTexture
		) {
			return;
		}
		this._services.renderEnvironment(context);
	}

	public renderOpaqueDepthPrepass(context: FrameContext): Set<string> {
		return this.renderEarlyZ(context, context.scene.opaquePackets);
	}

	public renderOpaque(context: FrameContext, earlyZPacketIds: ReadonlySet<string>): void {
		this.renderPackets(context, context.scene.opaquePackets, false, {
			earlyZPacketIds,
		});
	}

	public renderLegacyTransparent(context: FrameContext): void {
		this._services.renderLegacyTransparent(context);
	}

	public renderPackets(
		context: FrameContext,
		packets: DrawPacket[],
		transparent: boolean,
		options: WebGLSceneRenderOptions = {},
	): void {
		renderWebGLPackets(
			this._services.host,
			context,
			packets,
			transparent,
			options,
		);
	}

	public renderEarlyZ(context: FrameContext, packets: DrawPacket[]): Set<string> {
		if (!this._services.enableEarlyZPrepass || packets.length === 0) {
			return new Set<string>();
		}
		return renderWebGLEarlyZPrepass(this._services.host, context, packets);
	}

	public finishFrame(): void {
		if (this.modelMatrixCache.size > this.modelMatrixKeysThisFrame.size) {
			for (const cacheKey of this.modelMatrixCache.keys()) {
				if (!this.modelMatrixKeysThisFrame.has(cacheKey)) {
					this.modelMatrixCache.delete(cacheKey);
				}
			}
		}
		this.modelMatrixKeysThisFrame.clear();
	}

	public destroy(): void {
		this.modelMatrixCache.clear();
		this.modelMatrixKeysThisFrame.clear();
	}
}
