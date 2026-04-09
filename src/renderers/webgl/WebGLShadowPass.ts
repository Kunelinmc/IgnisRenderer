import type { Material } from "../../materials/Material";
import { Matrix4 } from "../../maths/Matrix4";
import type { DrawPacket, FrameContext } from "../../pipeline/types";
import {
	WEBGL_MAX_DIRECTIONAL_LIGHTS,
	WEBGL_MAX_SPOT_LIGHTS,
	WEBGL_SHADOW_ATLAS_COLUMNS,
} from "./constants";
import { getMaxShadowSize, isFiniteMatrix, toColumnMajorMat4 } from "./WebGLFrameMath";
import type { WebGLLightState, WebGLShadowData } from "./WebGLLightCollector";
import type { WebGLShadowDepthProgram } from "./WebGLProgramLibrary";

type WarnFn = (key: string, message: string) => void;

export interface WebGLShadowPassHost {
	_gl: WebGL2RenderingContext;
	_logWarning: WarnFn;
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
		tileIndex: number
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
	if (maxShadowSize <= 0 || context.scene.shadowCasterPackets.length <= 0) {
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
		host._renderShadowSlice(
			shadowProgram,
			packets,
			lights.directionalShadows[i],
			i
		);
	}

	const spotCount = Math.min(WEBGL_MAX_SPOT_LIGHTS, lights.spotShadows.length);
	for (let i = 0; i < spotCount; i++) {
		host._renderShadowSlice(
			shadowProgram,
			packets,
			lights.spotShadows[i],
			WEBGL_MAX_DIRECTIONAL_LIGHTS + i
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
	tileIndex: number
): void {
	if (!shadow?.enabled || !shadow.viewProjectionMatrix) {
		return;
	}

	const shadowSize = Math.max(1, shadow.shadowMapSize | 0);
	const atlasColumns = Math.max(1, WEBGL_SHADOW_ATLAS_COLUMNS);
	const tileX = tileIndex % atlasColumns;
	const tileY = Math.floor(tileIndex / atlasColumns);
	const viewportX = tileX * host._shadowAtlasTileSize;
	const viewportY = tileY * host._shadowAtlasTileSize;
	const gl = host._gl;
	gl.viewport(viewportX, viewportY, shadowSize, shadowSize);
	gl.scissor(viewportX, viewportY, shadowSize, shadowSize);
	gl.clear(gl.DEPTH_BUFFER_BIT);

	for (const packet of packets) {
		host._drawShadowPacket(shadowProgram, packet, shadow.viewProjectionMatrix);
	}
}

export function drawWebGLShadowPacket(
	host: WebGLShadowPassHost,
	shadowProgram: WebGLShadowDepthProgram,
	packet: DrawPacket,
	viewProjectionMatrix: Matrix4
): void {
	if (packet.meshInstance.skeleton) {
		host._logWarning(
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
