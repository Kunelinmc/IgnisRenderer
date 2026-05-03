import type { Material } from "../../materials/Material";
import { Matrix4 } from "../../maths/Matrix4";
import type { DrawPacket, FrameContext } from "../../pipeline/types";
import { PARTICLE_TRANSIENT_BATCHES_KEY } from "../../pipeline/types";
import { hasParticleShadowCastingBatches } from "../../pipeline/ParticleShadowVolume";
import {
	WEBGL_MAX_DIRECTIONAL_LIGHTS,
	WEBGL_MAX_SPOT_LIGHTS,
	WEBGL_SHADOW_ATLAS_COLUMNS,
} from "./constants";
import { getMaxShadowSize, isFiniteMatrix, toColumnMajorMat4 } from "./WebGLFrameMath";
import type { WebGLLightState, WebGLShadowData } from "./WebGLLightCollector";
import type { WebGLShadowDepthProgram } from "./WebGLProgramLibrary";
import { Logger } from "../../foundation/Logger";

function logWebGLShadowPassWarning(key: string, message: string): void {
	Logger.warn(`[${key}] ${message}`, {
		scope: "WebGLShadowPass",
		onceKey: key,
	});
}

export interface WebGLShadowPassHost {
	_gl: WebGL2RenderingContext;
	_programs: {
		getShadowDepthProgram(): WebGLShadowDepthProgram;
	};
	_geometry: {
		getGeometry(packet: DrawPacket): {
			vao: WebGLVertexArrayObject;
			topology: number;
			indexCount: number;
			indexType: number;
		} | null;
	};
	_lightState: WebGLLightState | null;
	_shadowFramebuffer: WebGLFramebuffer | null;
	_shadowAtlasTexture: WebGLTexture | null;
	_shadowAtlasTileSize: number;
	_shadowMvpMatrix: Matrix4;
	_sceneFramebuffer: WebGLFramebuffer | null;
	_width: number;
	_height: number;
	_ensureShadowTargets(tileSize: number): void;
	_setCullMode(material: Material): void;
	_renderShadowSlice(
		shadowProgram: WebGLShadowDepthProgram,
		packets: DrawPacket[],
		shadow: WebGLShadowData | undefined,
		tileIndex: number,
		cascadeIndex?: number
	): void;
	_drawShadowPacket(
		shadowProgram: WebGLShadowDepthProgram,
		packet: DrawPacket,
		viewProjectionMatrix: Matrix4
	): void;
}

export function renderWebGLShadows(
	host: WebGLShadowPassHost,
	context: FrameContext
): void {
	if (!context.features.enableShadows) {
		return;
	}
	const lights = host._lightState;
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
		(context.scene.shadowCasterPackets.length <= 0 && !hasParticleShadowVolumes)
	) {
		host._shadowAtlasTileSize = 0;
		return;
	}

	host._ensureShadowTargets(maxShadowSize);
	if (!host._shadowFramebuffer || !host._shadowAtlasTexture) {
		host._shadowAtlasTileSize = 0;
		return;
	}

	host._shadowAtlasTileSize = maxShadowSize;
	for (const shadow of lights.directionalShadows) {
		shadow.atlasTileSize = maxShadowSize;
	}
	for (const shadow of lights.spotShadows) {
		shadow.atlasTileSize = maxShadowSize;
	}

	const gl = host._gl;
	const shadowProgram = host._programs.getShadowDepthProgram();
	const packets = context.scene.shadowCasterPackets;

	gl.bindFramebuffer(gl.FRAMEBUFFER, host._shadowFramebuffer);
	gl.useProgram(shadowProgram.program);
	gl.disable(gl.BLEND);
	gl.enable(gl.DEPTH_TEST);
	gl.depthMask(true);
	gl.colorMask(false, false, false, false);
	gl.enable(gl.SCISSOR_TEST);
	gl.clearDepth(1);
	gl.clear(gl.DEPTH_BUFFER_BIT);

	const directionalCount = Math.min(
		WEBGL_MAX_DIRECTIONAL_LIGHTS,
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
			host._renderShadowSlice(
				shadowProgram,
				packets,
				shadow,
				i,
				cascadeIndex
			);
		}
	}

	const spotCount = Math.min(WEBGL_MAX_SPOT_LIGHTS, lights.spotShadows.length);
	for (let i = 0; i < spotCount; i++) {
		host._renderShadowSlice(
			shadowProgram,
			packets,
			lights.spotShadows[i],
			WEBGL_MAX_DIRECTIONAL_LIGHTS + i,
			0
		);
	}

	gl.disable(gl.SCISSOR_TEST);
	gl.colorMask(true, true, true, true);
	gl.bindVertexArray(null);
	gl.bindFramebuffer(gl.FRAMEBUFFER, host._sceneFramebuffer);
	gl.viewport(0, 0, host._width, host._height);
}

export function renderWebGLShadowSlice(
	host: WebGLShadowPassHost,
	shadowProgram: WebGLShadowDepthProgram,
	packets: DrawPacket[],
	shadow: WebGLShadowData | undefined,
	tileIndex: number,
	cascadeIndex: number = 0
): void {
	if (!shadow?.enabled) {
		return;
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
		return;
	}
	const localTileSpan = isCSM ? 2 : 1;
	const subTileSize = Math.max(
		1,
		Math.floor(host._shadowAtlasTileSize / localTileSpan)
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
	const viewportX = tileX * host._shadowAtlasTileSize + localTileX * subTileSize;
	const viewportY = tileY * host._shadowAtlasTileSize + localTileY * subTileSize;
	const gl = host._gl;
	gl.viewport(viewportX, viewportY, shadowSize, shadowSize);
	gl.scissor(viewportX, viewportY, shadowSize, shadowSize);
	gl.clear(gl.DEPTH_BUFFER_BIT);

	for (const packet of packets) {
		host._drawShadowPacket(shadowProgram, packet, shadowViewProjection);
	}
}

export function drawWebGLShadowPacket(
	host: WebGLShadowPassHost,
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

	const geometry = host._geometry.getGeometry(packet);
	if (!geometry || geometry.topology !== host._gl.TRIANGLES) {
		return;
	}

	Matrix4.multiply(viewProjectionMatrix, packet.worldMatrix, host._shadowMvpMatrix);
	const gl = host._gl;
	if (shadowProgram.uniforms.mvp) {
		gl.uniformMatrix4fv(
			shadowProgram.uniforms.mvp,
			false,
			toColumnMajorMat4(host._shadowMvpMatrix)
		);
	}

	host._setCullMode(packet.material);
	gl.bindVertexArray(geometry.vao);
	gl.drawElements(
		geometry.topology,
		geometry.indexCount,
		geometry.indexType,
		0
	);
	gl.bindVertexArray(null);
}

