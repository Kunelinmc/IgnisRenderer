import { getTextureFormatInfo } from "../../core/TextureFormat";
import type { TextureFormat } from "../../core/TextureFormat";
import type { IRenderTexture, TextureDesc } from "../types";
import type { CustomRenderTargetExecutionTarget, CustomRenderTargetResourceFactory } from "../../rendering/CustomRenderTargets";
import type { WebGLResourceStore } from "./WebGLResourceStore";
import {
	resolveWebGLColorRenderTargetFormat,
	resolveWebGLDepthRenderTargetFormat,
} from "./WebGLRenderTargetFormat";

/**
 * Supplies WebGL native operations to the CustomRenderTarget resource allocation.
 * @internal WebGL allocation bridge; applications use `Renderer.renderTargets`.
 */
export class WebGLRenderTargetResources implements CustomRenderTargetResourceFactory {
	private readonly _framebuffers = new WeakMap<CustomRenderTargetExecutionTarget, WebGLFramebuffer>();

	public constructor(
		private readonly _gl: WebGL2RenderingContext,
		private readonly _resources: WebGLResourceStore,
		private readonly _floatColorSupported: boolean,
	) {}

	/** @internal WebGL texture allocation; CustomRenderTarget runtime owns the returned texture. */
	public createTexture(descriptor: TextureDesc): IRenderTexture {
		this._resources.assertAlive();
		if ((descriptor.sampleCount ?? 1) !== 1) {
			throw new Error("WebGL render target sampleCount must be 1.");
		}
		if (getTextureFormatInfo(descriptor.format).formatClass === "depth") {
			return createWebGLDepthTexture(
				this._gl, this._resources, descriptor.width, descriptor.height,
				descriptor.format, descriptor.label ?? "WebGLTarget_Depth",
			);
		}
		return createWebGLColorTexture(
			this._gl, this._resources, descriptor.width, descriptor.height,
			descriptor.format, this._floatColorSupported, descriptor.label ?? "WebGLTarget_Color",
		);
	}

	/** @internal Registers private framebuffer state; CustomRenderTarget runtime owns its disposal. */
	public initializeTarget(target: CustomRenderTargetExecutionTarget): () => void {
		const gl = this._gl;
		this._resources.assertAlive();
		const framebuffer = gl.createFramebuffer();
		if (!framebuffer) throw new Error("Failed to create WebGL render target framebuffer.");
		try {
			gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
			for (let index = 0; index < target.color.length; index++) {
				gl.framebufferTexture2D(
					gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + index, gl.TEXTURE_2D,
					this._resources.getTexture(target.color[index].texture).texture, 0,
				);
			}
			if (target.depth) {
				gl.framebufferTexture2D(
					gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D,
					this._resources.getTexture(target.depth.texture).texture, 0,
				);
			}
			gl.drawBuffers(target.color.map((_, index) => gl.COLOR_ATTACHMENT0 + index));
			const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
			if (status !== gl.FRAMEBUFFER_COMPLETE) {
				throw new Error(
					`WebGL custom framebuffer "${target.id}" is incomplete ` +
						`(status=0x${status.toString(16)}).`
				);
			}
			this._framebuffers.set(target, framebuffer);
			return () => {
				this._framebuffers.delete(target);
				gl.deleteFramebuffer(framebuffer);
			};
		} catch (error) {
			gl.deleteFramebuffer(framebuffer);
			throw error;
		} finally {
			gl.bindFramebuffer(gl.FRAMEBUFFER, null);
		}
	}

	/** @internal Resolves a live owned framebuffer; applications use `Renderer.renderTargets`. */
	public getFramebuffer(target: CustomRenderTargetExecutionTarget): WebGLFramebuffer {
		this._resources.assertAlive();
		const framebuffer = this._framebuffers.get(target);
		if (!framebuffer) throw new Error("WebGL render target is foreign or destroyed.");
		return framebuffer;
	}
}

/** @internal WebGL attachment allocation; applications use `Renderer.renderTargets`. */
export function createWebGLColorTexture(
	gl: WebGL2RenderingContext,
	resources: WebGLResourceStore,
	width: number,
	height: number,
	format: TextureFormat,
	floatColorSupported: boolean,
	label: string
): IRenderTexture {
	const texture = gl.createTexture();
	if (!texture) {
		throw new Error(`Failed to create ${label}.`);
	}
	const resolved = resolveWebGLColorRenderTargetFormat(
		gl,
		format,
		floatColorSupported
	);
	if (!resolved) {
		gl.deleteTexture(texture);
		throw new Error(`Unsupported WebGL custom render target format "${format}".`);
	}
	try {
		gl.bindTexture(gl.TEXTURE_2D, texture);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
		gl.texImage2D(
			gl.TEXTURE_2D,
			0,
			resolved.internalFormat,
			width,
			height,
			0,
			resolved.format,
			resolved.allocationType,
			null
		);
		return resources.createTexture({
			width,
			height,
			format,
			requestedFormat: format,
		}, { texture, colorFormat: resolved, maxMipLevel: 0 }, () => gl.deleteTexture(texture));
	} catch (error) {
		gl.deleteTexture(texture);
		throw error;
	} finally {
		gl.bindTexture(gl.TEXTURE_2D, null);
	}
}

/** @internal WebGL attachment allocation; applications use `Renderer.renderTargets`. */
export function createWebGLDepthTexture(
	gl: WebGL2RenderingContext,
	resources: WebGLResourceStore,
	width: number,
	height: number,
	format: TextureFormat,
	label: string
): IRenderTexture {
	const texture = gl.createTexture();
	if (!texture) {
		throw new Error(`Failed to create ${label}.`);
	}
	const resolved = resolveWebGLDepthRenderTargetFormat(gl, format);
	if (!resolved) {
		gl.deleteTexture(texture);
		throw new Error(`Unsupported WebGL custom depth format "${format}".`);
	}
	try {
		gl.bindTexture(gl.TEXTURE_2D, texture);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
		gl.texImage2D(
			gl.TEXTURE_2D,
			0,
			resolved.internalFormat,
			width,
			height,
			0,
			resolved.format,
			resolved.type,
			null
		);
		return resources.createTexture({
			width,
			height,
			format,
			requestedFormat: format,
		}, { texture, maxMipLevel: 0 }, () => gl.deleteTexture(texture));
	} catch (error) {
		gl.deleteTexture(texture);
		throw error;
	} finally {
		gl.bindTexture(gl.TEXTURE_2D, null);
	}
}
