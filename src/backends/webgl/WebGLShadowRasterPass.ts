import { Logger } from "../../foundation/Logger";
import { resolveMaterialShadowTransmittance } from "../../materials/transparency";
import { Matrix4 } from "../../maths/Matrix4";
import type { DrawPacket } from "../../pipeline/types";

import { isFiniteMatrix, toColumnMajorMat4 } from "./WebGLFrameMath";
import type {
	WebGLShadowDepthProgram,
	WebGLShadowTransmittanceProgram,
} from "./WebGLProgramLibrary";

function logWebGLShadowRasterWarning(key: string, message: string): void {
	Logger.warn(`[${key}] ${message}`, {
		scope: "WebGLShadowRasterPass",
		onceKey: key,
	});
}

/** @internal A fully resolved atlas slice consumed by the WebGL shadow rasterizer. */
export interface WebGLShadowRasterSlice {
	readonly kind: "directional" | "spot";
	readonly lightIndex: number;
	readonly cascadeIndex: number;
	readonly viewportX: number;
	readonly viewportY: number;
	readonly viewportWidth: number;
	readonly viewportHeight: number;
	readonly viewProjectionMatrix: Matrix4;
}

/** @internal Runtime-owned, allocation-conscious input for one shadow raster frame. */
export interface WebGLShadowRasterPlan {
	readonly atlasTileSize: number;
	readonly atlasWidth: number;
	readonly atlasHeight: number;
	readonly slices: readonly WebGLShadowRasterSlice[];
	readonly sliceCount: number;
	readonly casterPackets: readonly DrawPacket[];
	readonly transmitterPackets: readonly DrawPacket[];
	readonly baselineFramebuffer: WebGLFramebuffer | null;
	readonly baselineViewportWidth: number;
	readonly baselineViewportHeight: number;
}

/** @internal Native target state returned to the owning shadow runtime. */
export interface WebGLShadowRasterPreparedState {
	readonly framebuffer: WebGLFramebuffer | null;
	readonly atlasTexture: WebGLTexture | null;
	readonly transmittanceTexture: WebGLTexture | null;
	readonly atlasTileSize: number;
	readonly depthProgramAvailable: boolean;
}

/** @internal Narrow device dependencies for the plan-only shadow rasterizer. */
export interface WebGLShadowRasterPassHost {
	readonly gl: WebGL2RenderingContext;
	readonly programs: {
		getShadowDepthProgram(): WebGLShadowDepthProgram;
		getShadowTransmittanceProgram(): WebGLShadowTransmittanceProgram;
		tryGetShadowDepthProgram?(): WebGLShadowDepthProgram | null;
		tryGetShadowTransmittanceProgram?():
			WebGLShadowTransmittanceProgram | null;
	};
	readonly geometry: {
		getGeometry(packet: DrawPacket): {
			vao: WebGLVertexArrayObject;
			topology: number;
			indexCount: number;
			indexType: number;
		} | null;
	};
	readonly maxTextureSize: number;
}

/** Executes prepared WebGL shadow raster plans and owns their native atlas targets. */
export class WebGLShadowRasterPass {
	private readonly _host: WebGLShadowRasterPassHost;
	private _shadowFramebuffer: WebGLFramebuffer | null = null;
	private _shadowAtlasTexture: WebGLTexture | null = null;
	private _shadowTransmittanceTexture: WebGLTexture | null = null;
	private _shadowAtlasTileSize = 0;
	private _shadowMvpMatrix = Matrix4.identity();
	private _depthProgram: WebGLShadowDepthProgram | null = null;
	private _transmittanceProgram: WebGLShadowTransmittanceProgram | null = null;

	public constructor(host: WebGLShadowRasterPassHost) {
		this._host = host;
	}

	/** Prepares native targets and shader programs without inspecting frame state. */
	public prepare(plan: WebGLShadowRasterPlan): WebGLShadowRasterPreparedState {
		try {
			this._ensureShadowTargets(plan);
			this._depthProgram = this._resolveDepthProgram();
			this._transmittanceProgram = this._resolveTransmittanceProgram();
			return {
				framebuffer: this._shadowFramebuffer,
				atlasTexture: this._shadowAtlasTexture,
				transmittanceTexture: this._shadowTransmittanceTexture,
				atlasTileSize: this._shadowAtlasTileSize,
				depthProgramAvailable: !!this._depthProgram,
			};
		} finally {
			this._restoreFrameBaseline(plan);
		}
	}

	/** Renders one previously prepared plan and restores the frame baseline. */
	public render(plan: WebGLShadowRasterPlan): void {
		if (
			!this._shadowFramebuffer ||
			!this._shadowAtlasTexture ||
			!this._depthProgram ||
			plan.atlasTileSize !== this._shadowAtlasTileSize
		) return;

		const gl = this._host.gl;
		try {
			gl.bindFramebuffer(gl.FRAMEBUFFER, this._shadowFramebuffer);
			gl.useProgram(this._depthProgram.program);
			gl.disable(gl.BLEND);
			gl.enable(gl.DEPTH_TEST);
			gl.depthMask(true);
			gl.colorMask(false, false, false, false);
			gl.enable(gl.SCISSOR_TEST);
			gl.clearDepth(1);
			gl.clear(gl.DEPTH_BUFFER_BIT);

			for (let index = 0; index < plan.sliceCount; index++) {
				this._renderDepthSlice(this._depthProgram, plan, plan.slices[index]);
			}

			gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
			gl.colorMask(true, true, true, true);
			gl.depthMask(false);
			gl.enable(gl.BLEND);
			gl.blendFuncSeparate(gl.ZERO, gl.SRC_COLOR, gl.ZERO, gl.ONE);
			gl.clearColor(1, 1, 1, 1);
			gl.disable(gl.SCISSOR_TEST);
			gl.clear(gl.COLOR_BUFFER_BIT);
			if (!this._transmittanceProgram) return;

			gl.enable(gl.SCISSOR_TEST);
			gl.useProgram(this._transmittanceProgram.program);
			for (let index = 0; index < plan.sliceCount; index++) {
				this._renderTransmittanceSlice(
					this._transmittanceProgram,
					plan,
					plan.slices[index],
				);
			}
		} finally {
			this._restoreFrameBaseline(plan);
		}
	}

	public destroy(): void {
		this._depthProgram = null;
		this._transmittanceProgram = null;
		this._destroyShadowTargets();
	}

	private _renderDepthSlice(
		program: WebGLShadowDepthProgram,
		plan: WebGLShadowRasterPlan,
		slice: WebGLShadowRasterSlice | undefined,
	): void {
		if (!slice) return;
		this._setSliceViewport(slice);
		this._host.gl.clear(this._host.gl.DEPTH_BUFFER_BIT);
		for (const packet of plan.casterPackets) {
			this._drawShadowPacket(program, packet, slice.viewProjectionMatrix);
		}
	}

	private _renderTransmittanceSlice(
		program: WebGLShadowTransmittanceProgram,
		plan: WebGLShadowRasterPlan,
		slice: WebGLShadowRasterSlice | undefined,
	): void {
		if (!slice || plan.transmitterPackets.length === 0) return;
		this._setSliceViewport(slice);
		for (const packet of plan.transmitterPackets) {
			this._drawShadowTransmittancePacket(
				program,
				packet,
				slice.viewProjectionMatrix,
			);
		}
	}

	private _setSliceViewport(slice: WebGLShadowRasterSlice): void {
		const gl = this._host.gl;
		gl.viewport(
			slice.viewportX,
			slice.viewportY,
			slice.viewportWidth,
			slice.viewportHeight,
		);
		gl.scissor(
			slice.viewportX,
			slice.viewportY,
			slice.viewportWidth,
			slice.viewportHeight,
		);
	}

	private _drawShadowPacket(
		shadowProgram: WebGLShadowDepthProgram,
		packet: DrawPacket,
		viewProjectionMatrix: Matrix4,
	): void {
		if (packet.meshInstance.skeleton) {
			logWebGLShadowRasterWarning(
				"webgl-shadow-skinning-unsupported",
				"WebGL shadow raster pass does not support skinning yet; skipping " +
					`mesh instance ${packet.meshInstance.id}`,
			);
			return;
		}
		if (!isFiniteMatrix(packet.worldMatrix)) return;
		const geometry = this._host.geometry.getGeometry(packet);
		if (!geometry || geometry.topology !== this._host.gl.TRIANGLES) return;

		Matrix4.multiply(viewProjectionMatrix, packet.worldMatrix, this._shadowMvpMatrix);
		const gl = this._host.gl;
		if (shadowProgram.uniforms.mvp) {
			gl.uniformMatrix4fv(
				shadowProgram.uniforms.mvp,
				false,
				toColumnMajorMat4(this._shadowMvpMatrix),
			);
		}
		gl.disable(gl.CULL_FACE);
		gl.bindVertexArray(geometry.vao);
		gl.drawElements(
			geometry.topology,
			geometry.indexCount,
			geometry.indexType,
			0,
		);
		gl.bindVertexArray(null);
	}

	private _drawShadowTransmittancePacket(
		shadowProgram: WebGLShadowTransmittanceProgram,
		packet: DrawPacket,
		viewProjectionMatrix: Matrix4,
	): void {
		const geometry = this._host.geometry.getGeometry(packet);
		if (!geometry || geometry.topology !== this._host.gl.TRIANGLES) return;
		if (!isFiniteMatrix(packet.worldMatrix) || !isFiniteMatrix(viewProjectionMatrix)) {
			return;
		}
		const gl = this._host.gl;
		Matrix4.multiply(viewProjectionMatrix, packet.worldMatrix, this._shadowMvpMatrix);
		gl.uniformMatrix4fv(
			shadowProgram.uniforms.mvp,
			false,
			toColumnMajorMat4(this._shadowMvpMatrix),
		);
		const transmittance = resolveMaterialShadowTransmittance(packet.material);
		gl.uniform3f(
			shadowProgram.uniforms.transmittance,
			transmittance.r,
			transmittance.g,
			transmittance.b,
		);
		gl.disable(gl.CULL_FACE);
		gl.bindVertexArray(geometry.vao);
		gl.drawElements(
			geometry.topology,
			geometry.indexCount,
			geometry.indexType,
			0,
		);
		gl.bindVertexArray(null);
	}

	private _resolveDepthProgram(): WebGLShadowDepthProgram | null {
		return typeof this._host.programs.tryGetShadowDepthProgram === "function" ?
			this._host.programs.tryGetShadowDepthProgram()
		: this._host.programs.getShadowDepthProgram();
	}

	private _resolveTransmittanceProgram(): WebGLShadowTransmittanceProgram | null {
		return typeof this._host.programs.tryGetShadowTransmittanceProgram === "function" ?
			this._host.programs.tryGetShadowTransmittanceProgram()
		: this._host.programs.getShadowTransmittanceProgram();
	}

	private _ensureShadowTargets(plan: WebGLShadowRasterPlan): void {
		const tileSize = plan.atlasTileSize;
		if (
			this._shadowFramebuffer &&
			this._shadowAtlasTexture &&
			this._shadowTransmittanceTexture &&
			this._shadowAtlasTileSize === tileSize
		) return;
		const atlasWidth = plan.atlasWidth;
		const atlasHeight = plan.atlasHeight;
		if (atlasWidth > this._host.maxTextureSize || atlasHeight > this._host.maxTextureSize) {
			throw new Error(
				`WebGL shadow atlas ${atlasWidth}x${atlasHeight} exceeds MAX_TEXTURE_SIZE=${this._host.maxTextureSize}`,
			);
		}

		const gl = this._host.gl;
		const shadowTexture = gl.createTexture();
		const transmittanceTexture = gl.createTexture();
		const shadowFramebuffer = gl.createFramebuffer();
		if (!shadowTexture || !transmittanceTexture || !shadowFramebuffer) {
			if (shadowTexture) gl.deleteTexture(shadowTexture);
			if (transmittanceTexture) gl.deleteTexture(transmittanceTexture);
			if (shadowFramebuffer) gl.deleteFramebuffer(shadowFramebuffer);
			throw new Error("Failed to create WebGL shadow atlas targets");
		}

		try {
			this._allocateDepthAtlas(shadowTexture, atlasWidth, atlasHeight);
			this._allocateTransmittanceAtlas(
				transmittanceTexture,
				atlasWidth,
				atlasHeight,
			);
			gl.bindFramebuffer(gl.FRAMEBUFFER, shadowFramebuffer);
			gl.framebufferTexture2D(
				gl.FRAMEBUFFER,
				gl.DEPTH_ATTACHMENT,
				gl.TEXTURE_2D,
				shadowTexture,
				0,
			);
			gl.framebufferTexture2D(
				gl.FRAMEBUFFER,
				gl.COLOR_ATTACHMENT0,
				gl.TEXTURE_2D,
				transmittanceTexture,
				0,
			);
			gl.drawBuffers([gl.NONE]);
			gl.readBuffer(gl.NONE);
			const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
			if (status !== gl.FRAMEBUFFER_COMPLETE) {
				throw new Error(
					`WebGL shadow framebuffer is incomplete (status=0x${status.toString(16)})`,
				);
			}
		} catch (error) {
			gl.deleteFramebuffer(shadowFramebuffer);
			gl.deleteTexture(shadowTexture);
			gl.deleteTexture(transmittanceTexture);
			throw error;
		}

		this._destroyShadowTargets();
		this._shadowFramebuffer = shadowFramebuffer;
		this._shadowAtlasTexture = shadowTexture;
		this._shadowTransmittanceTexture = transmittanceTexture;
		this._shadowAtlasTileSize = tileSize;
	}

	private _allocateDepthAtlas(
		texture: WebGLTexture,
		width: number,
		height: number,
	): void {
		const gl = this._host.gl;
		gl.bindTexture(gl.TEXTURE_2D, texture);
		this._setAtlasTextureParameters();
		gl.texImage2D(
			gl.TEXTURE_2D,
			0,
			gl.DEPTH_COMPONENT24,
			width,
			height,
			0,
			gl.DEPTH_COMPONENT,
			gl.UNSIGNED_INT,
			null,
		);
	}

	private _allocateTransmittanceAtlas(
		texture: WebGLTexture,
		width: number,
		height: number,
	): void {
		const gl = this._host.gl;
		gl.bindTexture(gl.TEXTURE_2D, texture);
		this._setAtlasTextureParameters();
		gl.texImage2D(
			gl.TEXTURE_2D,
			0,
			gl.RGBA8,
			width,
			height,
			0,
			gl.RGBA,
			gl.UNSIGNED_BYTE,
			null,
		);
	}

	private _setAtlasTextureParameters(): void {
		const gl = this._host.gl;
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
	}

	private _destroyShadowTargets(): void {
		const gl = this._host.gl;
		if (this._shadowFramebuffer) gl.deleteFramebuffer(this._shadowFramebuffer);
		if (this._shadowAtlasTexture) gl.deleteTexture(this._shadowAtlasTexture);
		if (this._shadowTransmittanceTexture) {
			gl.deleteTexture(this._shadowTransmittanceTexture);
		}
		this._shadowFramebuffer = null;
		this._shadowAtlasTexture = null;
		this._shadowTransmittanceTexture = null;
		this._shadowAtlasTileSize = 0;
	}

	private _restoreFrameBaseline(plan: WebGLShadowRasterPlan): void {
		const gl = this._host.gl;
		gl.bindTexture(gl.TEXTURE_2D, null);
		gl.disable(gl.BLEND);
		gl.disable(gl.CULL_FACE);
		gl.depthMask(true);
		gl.disable(gl.SCISSOR_TEST);
		gl.colorMask(true, true, true, true);
		gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
		gl.bindVertexArray(null);
		gl.bindFramebuffer(gl.FRAMEBUFFER, plan.baselineFramebuffer);
		gl.viewport(0, 0, plan.baselineViewportWidth, plan.baselineViewportHeight);
	}
}
