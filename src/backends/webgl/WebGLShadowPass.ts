import type { Material } from "../../materials/Material";
import { Matrix4 } from "../../maths/Matrix4";
import { resolveMaterialShadowTransmittance } from "../../materials/transparency";
import type { DrawPacket, FrameContext } from "../../pipeline/types";
import { PARTICLE_TRANSIENT_BATCHES_KEY } from "../../pipeline/types";
import { hasParticleShadowCastingBatches } from "../../pipeline/ParticleShadowVolume";
import {
	MAX_DIRECTIONAL_LIGHTS,
	MAX_SPOT_LIGHTS,
} from "../constants";
import {
	WEBGL_SHADOW_ATLAS_COLUMNS,
	WEBGL_SHADOW_ATLAS_ROWS,
} from "./constants";
import {
	getMaxShadowSize,
	isFiniteMatrix,
	toColumnMajorMat4,
} from "./WebGLFrameMath";
import type { WebGLLightState, WebGLShadowData } from "./WebGLLightCollector";
import type {
	WebGLShadowDepthProgram,
	WebGLShadowTransmittanceProgram,
} from "./WebGLProgramLibrary";
import type { WebGLRenderPass } from "./WebGLPassLifecycle";
import { Logger } from "../../foundation/Logger";

function logWebGLShadowPassWarning(key: string, message: string): void {
	Logger.warn(`[${key}] ${message}`, {
		scope: "WebGLShadowPass",
		onceKey: key,
	});
}

export interface WebGLShadowPassTargets {
	readonly framebuffer: WebGLFramebuffer | null;
	readonly atlasTexture: WebGLTexture | null;
	readonly transmittanceTexture: WebGLTexture | null;
	readonly atlasTileSize: number;
}

export interface WebGLShadowPassHost {
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
	getLightState(): WebGLLightState | null;
	getSceneFramebuffer(): WebGLFramebuffer | null;
	getViewportSize(): { width: number; height: number };
	getMaxTextureSize(): number;
}

export class WebGLShadowPass implements WebGLRenderPass {
	private readonly _host: WebGLShadowPassHost;
	private _shadowFramebuffer: WebGLFramebuffer | null = null;
	private _shadowAtlasTexture: WebGLTexture | null = null;
	private _shadowTransmittanceTexture: WebGLTexture | null = null;
	private _shadowAtlasTileSize = 0;
	private _shadowMvpMatrix = Matrix4.identity();

	public constructor(host: WebGLShadowPassHost) {
		this._host = host;
	}

	public getTargets(): WebGLShadowPassTargets {
		return {
			framebuffer: this._shadowFramebuffer,
			atlasTexture: this._shadowAtlasTexture,
			transmittanceTexture: this._shadowTransmittanceTexture,
			atlasTileSize: this._shadowAtlasTileSize,
		};
	}

	public render(context: FrameContext): void {
		if (!context.features.enableShadows) {
			return;
		}
		const lights = this._host.getLightState();
		if (!lights) {
			return;
		}
		const maxShadowSize = Math.max(
			getMaxShadowSize(lights.directionalShadows),
			getMaxShadowSize(lights.spotShadows)
		);
		const particleBatches = context.transient.get(PARTICLE_TRANSIENT_BATCHES_KEY);
		const hasParticleShadowVolumes = hasParticleShadowCastingBatches(
			particleBatches
		);
		if (
			maxShadowSize <= 0 ||
			(context.scene.shadowCasterPackets.length <= 0 &&
				context.scene.shadowTransmitterPackets.length <= 0 &&
				!hasParticleShadowVolumes)
		) {
			this._shadowAtlasTileSize = 0;
			return;
		}

		this._ensureShadowTargets(maxShadowSize);
		if (!this._shadowFramebuffer || !this._shadowAtlasTexture) {
			this._shadowAtlasTileSize = 0;
			return;
		}

		const gl = this._host.gl;
		const shadowProgram =
			typeof this._host.programs.tryGetShadowDepthProgram === "function" ?
				this._host.programs.tryGetShadowDepthProgram()
			:	this._host.programs.getShadowDepthProgram();
		if (!shadowProgram) {
			this._shadowAtlasTileSize = 0;
			return;
		}

		this._shadowAtlasTileSize = maxShadowSize;
		for (const shadow of lights.directionalShadows) {
			shadow.atlasTileSize = maxShadowSize;
		}
		for (const shadow of lights.spotShadows) {
			shadow.atlasTileSize = maxShadowSize;
		}

		const packets = context.scene.shadowCasterPackets;
		const transmitterProgram =
			typeof this._host.programs.tryGetShadowTransmittanceProgram === "function" ?
				this._host.programs.tryGetShadowTransmittanceProgram()
			:	this._host.programs.getShadowTransmittanceProgram();
		const transmitterPackets = context.scene.shadowTransmitterPackets;

		gl.bindFramebuffer(gl.FRAMEBUFFER, this._shadowFramebuffer);
		gl.useProgram(shadowProgram.program);
		gl.disable(gl.BLEND);
		gl.enable(gl.DEPTH_TEST);
		gl.depthMask(true);
		gl.colorMask(false, false, false, false);
		gl.enable(gl.SCISSOR_TEST);
		gl.clearDepth(1);
		gl.clear(gl.DEPTH_BUFFER_BIT);

		const directionalCount = Math.min(
			MAX_DIRECTIONAL_LIGHTS,
			lights.directionalShadows.length
		);
		for (let i = 0; i < directionalCount; i++) {
			const shadow = lights.directionalShadows[i];
			const isCSM =
				shadow?.enabled &&
				shadow.strategyType === "csm" &&
				shadow.cascadeCount > 1;
			const cascadeCount =
				isCSM ? Math.max(1, Math.min(4, shadow.cascadeCount | 0)) : 1;
			for (let cascadeIndex = 0; cascadeIndex < cascadeCount; cascadeIndex++) {
				this._renderShadowSlice(
					shadowProgram,
					packets,
					shadow,
					i,
					cascadeIndex
				);
			}
		}

		const spotCount = Math.min(MAX_SPOT_LIGHTS, lights.spotShadows.length);
		for (let i = 0; i < spotCount; i++) {
			this._renderShadowSlice(
				shadowProgram,
				packets,
				lights.spotShadows[i],
				MAX_DIRECTIONAL_LIGHTS + i,
				0
			);
		}

		gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
		gl.colorMask(true, true, true, true);
		gl.depthMask(false);
		gl.enable(gl.BLEND);
		gl.blendFuncSeparate(gl.ZERO, gl.SRC_COLOR, gl.ZERO, gl.ONE);
		gl.clearColor(1, 1, 1, 1);
		gl.disable(gl.SCISSOR_TEST);
		gl.clear(gl.COLOR_BUFFER_BIT);
		if (!transmitterProgram) {
			this._restoreAfterRender();
			return;
		}
		gl.enable(gl.SCISSOR_TEST);
		gl.useProgram(transmitterProgram.program);
		for (let i = 0; i < directionalCount; i++) {
			const shadow = lights.directionalShadows[i];
			const isCSM =
				shadow?.enabled &&
				shadow.strategyType === "csm" &&
				shadow.cascadeCount > 1;
			const cascadeCount =
				isCSM ? Math.max(1, Math.min(4, shadow.cascadeCount | 0)) : 1;
			for (let cascadeIndex = 0; cascadeIndex < cascadeCount; cascadeIndex++) {
				this._renderShadowTransmittanceSlice(
					transmitterProgram,
					transmitterPackets,
					shadow,
					i,
					cascadeIndex
				);
			}
		}
		for (let i = 0; i < spotCount; i++) {
			this._renderShadowTransmittanceSlice(
				transmitterProgram,
				transmitterPackets,
				lights.spotShadows[i],
				MAX_DIRECTIONAL_LIGHTS + i,
				0
			);
		}

		this._restoreAfterRender();
	}

	public drawShadowPacket(
		shadowProgram: WebGLShadowDepthProgram,
		packet: DrawPacket,
		viewProjectionMatrix: Matrix4
	): void {
		this._drawShadowPacket(shadowProgram, packet, viewProjectionMatrix);
	}

	public drawShadowTransmittancePacket(
		shadowProgram: WebGLShadowTransmittanceProgram,
		packet: DrawPacket,
		viewProjectionMatrix: Matrix4
	): void {
		this._drawShadowTransmittancePacket(
			shadowProgram,
			packet,
			viewProjectionMatrix
		);
	}

	public invalidate(): void {
		this.destroy();
	}

	public destroy(): void {
		this._destroyShadowTargets();
	}

	private _renderShadowSlice(
		shadowProgram: WebGLShadowDepthProgram,
		packets: DrawPacket[],
		shadow: WebGLShadowData | undefined,
		tileIndex: number,
		cascadeIndex: number = 0
	): void {
		const resolved = this._resolveShadowTile(shadow, tileIndex, cascadeIndex);
		if (!resolved) {
			return;
		}
		const gl = this._host.gl;
		gl.viewport(
			resolved.viewportX,
			resolved.viewportY,
			resolved.shadowSize,
			resolved.shadowSize
		);
		gl.scissor(
			resolved.viewportX,
			resolved.viewportY,
			resolved.shadowSize,
			resolved.shadowSize
		);
		gl.clear(gl.DEPTH_BUFFER_BIT);

		for (const packet of packets) {
			this._drawShadowPacket(
				shadowProgram,
				packet,
				resolved.shadowViewProjection
			);
		}
	}

	private _renderShadowTransmittanceSlice(
		shadowProgram: WebGLShadowTransmittanceProgram,
		packets: DrawPacket[],
		shadow: WebGLShadowData | undefined,
		tileIndex: number,
		cascadeIndex: number = 0
	): void {
		if (packets.length <= 0) {
			return;
		}
		const resolved = this._resolveShadowTile(shadow, tileIndex, cascadeIndex);
		if (!resolved) {
			return;
		}
		const gl = this._host.gl;
		gl.viewport(
			resolved.viewportX,
			resolved.viewportY,
			resolved.shadowSize,
			resolved.shadowSize
		);
		gl.scissor(
			resolved.viewportX,
			resolved.viewportY,
			resolved.shadowSize,
			resolved.shadowSize
		);

		for (const packet of packets) {
			this._drawShadowTransmittancePacket(
				shadowProgram,
				packet,
				resolved.shadowViewProjection
			);
		}
	}

	private _resolveShadowTile(
		shadow: WebGLShadowData | undefined,
		tileIndex: number,
		cascadeIndex: number
	): {
		shadowViewProjection: Matrix4;
		shadowSize: number;
		viewportX: number;
		viewportY: number;
	} | null {
		if (!shadow?.enabled) {
			return null;
		}

		const isCSM = shadow.strategyType === "csm" && shadow.cascadeCount > 1;
		const resolvedCascadeCount =
			isCSM ? Math.max(1, Math.min(4, shadow.cascadeCount | 0)) : 1;
		const clampedCascadeIndex = Math.max(
			0,
			Math.min(resolvedCascadeCount - 1, cascadeIndex | 0)
		);
		const cascadeMatrices = shadow.cascadeViewProjectionMatrices ?? [];
		const shadowViewProjection =
			isCSM ?
				cascadeMatrices[clampedCascadeIndex]
			:	shadow.viewProjectionMatrix;
		if (!shadowViewProjection) {
			return null;
		}
		const localTileSpan = isCSM ? 2 : 1;
		const subTileSize = Math.max(
			1,
			Math.floor(this._shadowAtlasTileSize / localTileSpan)
		);
		const shadowSize = Math.max(
			1,
			Math.min(shadow.shadowMapSize | 0, subTileSize)
		);
		const cascadeSplits = shadow.cascadeSplits ?? [];
		const cascadeSplit = cascadeSplits[clampedCascadeIndex] ?? [0, 0, 0, 0];
		const localTileX =
			isCSM ? Math.max(0, Math.min(1, Math.floor(cascadeSplit[2] + 0.5))) : 0;
		const localTileY =
			isCSM ? Math.max(0, Math.min(1, Math.floor(cascadeSplit[3] + 0.5))) : 0;
		const atlasColumns = Math.max(1, WEBGL_SHADOW_ATLAS_COLUMNS);
		const tileX = tileIndex % atlasColumns;
		const tileY = Math.floor(tileIndex / atlasColumns);
		const viewportX = tileX * this._shadowAtlasTileSize + localTileX * subTileSize;
		const viewportY = tileY * this._shadowAtlasTileSize + localTileY * subTileSize;
		return {
			shadowViewProjection,
			shadowSize,
			viewportX,
			viewportY,
		};
	}

	private _drawShadowPacket(
		shadowProgram: WebGLShadowDepthProgram,
		packet: DrawPacket,
		viewProjectionMatrix: Matrix4
	): void {
		if (packet.meshInstance.skeleton) {
			logWebGLShadowPassWarning(
				"webgl-shadow-skinning-unsupported",
				`WebGL shadow pass does not support skinning yet; skipping mesh instance ${packet.meshInstance.id}`
			);
			return;
		}
		if (!isFiniteMatrix(packet.worldMatrix)) {
			return;
		}

		const geometry = this._host.geometry.getGeometry(packet);
		if (!geometry || geometry.topology !== this._host.gl.TRIANGLES) {
			return;
		}

		Matrix4.multiply(
			viewProjectionMatrix,
			packet.worldMatrix,
			this._shadowMvpMatrix
		);
		const gl = this._host.gl;
		if (shadowProgram.uniforms.mvp) {
			gl.uniformMatrix4fv(
				shadowProgram.uniforms.mvp,
				false,
				toColumnMajorMat4(this._shadowMvpMatrix)
			);
		}

		gl.disable(gl.CULL_FACE);
		gl.bindVertexArray(geometry.vao);
		gl.drawElements(
			geometry.topology,
			geometry.indexCount,
			geometry.indexType,
			0
		);
		gl.bindVertexArray(null);
	}

	private _drawShadowTransmittancePacket(
		shadowProgram: WebGLShadowTransmittanceProgram,
		packet: DrawPacket,
		viewProjectionMatrix: Matrix4
	): void {
		const geometry = this._host.geometry.getGeometry(packet);
		if (!geometry || !isFiniteMatrix(viewProjectionMatrix)) {
			return;
		}
		const gl = this._host.gl;
		Matrix4.multiply(
			viewProjectionMatrix,
			packet.worldMatrix,
			this._shadowMvpMatrix
		);
		gl.uniformMatrix4fv(
			shadowProgram.uniforms.mvp,
			false,
			toColumnMajorMat4(this._shadowMvpMatrix)
		);
		const transmittance = resolveMaterialShadowTransmittance(packet.material);
		gl.uniform3f(
			shadowProgram.uniforms.transmittance,
			transmittance.r,
			transmittance.g,
			transmittance.b
		);
		gl.disable(gl.CULL_FACE);
		gl.bindVertexArray(geometry.vao);
		gl.drawElements(
			geometry.topology,
			geometry.indexCount,
			geometry.indexType,
			0
		);
		gl.bindVertexArray(null);
	}

	private _ensureShadowTargets(tileSize: number): void {
		if (
			this._shadowFramebuffer &&
			this._shadowAtlasTexture &&
			this._shadowAtlasTileSize === tileSize
		) {
			return;
		}

		const atlasWidth = tileSize * WEBGL_SHADOW_ATLAS_COLUMNS;
		const atlasHeight = tileSize * WEBGL_SHADOW_ATLAS_ROWS;
		const maxTextureSize = this._host.getMaxTextureSize();
		if (atlasWidth > maxTextureSize || atlasHeight > maxTextureSize) {
			throw new Error(
				`WebGL shadow atlas ${atlasWidth}x${atlasHeight} exceeds MAX_TEXTURE_SIZE=${maxTextureSize}`
			);
		}

		this._destroyShadowTargets();
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

		gl.bindTexture(gl.TEXTURE_2D, shadowTexture);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
		gl.texImage2D(
			gl.TEXTURE_2D,
			0,
			gl.DEPTH_COMPONENT24,
			atlasWidth,
			atlasHeight,
			0,
			gl.DEPTH_COMPONENT,
			gl.UNSIGNED_INT,
			null
		);
		gl.bindTexture(gl.TEXTURE_2D, null);

		gl.bindTexture(gl.TEXTURE_2D, transmittanceTexture);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
		gl.texImage2D(
			gl.TEXTURE_2D,
			0,
			gl.RGBA8,
			atlasWidth,
			atlasHeight,
			0,
			gl.RGBA,
			gl.UNSIGNED_BYTE,
			null
		);
		gl.bindTexture(gl.TEXTURE_2D, null);

		gl.bindFramebuffer(gl.FRAMEBUFFER, shadowFramebuffer);
		gl.framebufferTexture2D(
			gl.FRAMEBUFFER,
			gl.DEPTH_ATTACHMENT,
			gl.TEXTURE_2D,
			shadowTexture,
			0
		);
		gl.framebufferTexture2D(
			gl.FRAMEBUFFER,
			gl.COLOR_ATTACHMENT0,
			gl.TEXTURE_2D,
			transmittanceTexture,
			0
		);
		gl.drawBuffers([gl.NONE]);
		gl.readBuffer(gl.NONE);
		const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
		gl.bindFramebuffer(gl.FRAMEBUFFER, null);
		if (status !== gl.FRAMEBUFFER_COMPLETE) {
			gl.deleteFramebuffer(shadowFramebuffer);
			gl.deleteTexture(shadowTexture);
			gl.deleteTexture(transmittanceTexture);
			throw new Error(
				`WebGL shadow framebuffer is incomplete (status=0x${status.toString(16)})`
			);
		}

		this._shadowFramebuffer = shadowFramebuffer;
		this._shadowAtlasTexture = shadowTexture;
		this._shadowTransmittanceTexture = transmittanceTexture;
		this._shadowAtlasTileSize = tileSize;
	}

	private _destroyShadowTargets(): void {
		const gl = this._host.gl;
		if (this._shadowFramebuffer) {
			gl.deleteFramebuffer(this._shadowFramebuffer);
			this._shadowFramebuffer = null;
		}
		if (this._shadowAtlasTexture) {
			gl.deleteTexture(this._shadowAtlasTexture);
			this._shadowAtlasTexture = null;
		}
		if (this._shadowTransmittanceTexture) {
			gl.deleteTexture(this._shadowTransmittanceTexture);
			this._shadowTransmittanceTexture = null;
		}
		this._shadowAtlasTileSize = 0;
	}

	private _restoreAfterRender(): void {
		const gl = this._host.gl;
		const { width, height } = this._host.getViewportSize();
		gl.disable(gl.BLEND);
		gl.depthMask(true);
		gl.disable(gl.SCISSOR_TEST);
		gl.colorMask(true, true, true, true);
		gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
		gl.bindVertexArray(null);
		gl.bindFramebuffer(gl.FRAMEBUFFER, this._host.getSceneFramebuffer());
		gl.viewport(0, 0, width, height);
	}
}
