import {
	AlphaMode,
	type Material,
} from "../../materials/Material";
import { ShaderMaterial } from "../../materials/ShaderMaterial";
import type { DrawPacket, FrameContext } from "../../pipeline/types";
import {
	isFiniteMatrix,
	toColumnMajorMat3,
	toColumnMajorMat4,
	toFiniteColumnMajorMat4,
} from "./WebGLFrameMath";
import { resolveMaterialUniforms } from "./WebGLMaterialUniformResolver";
import type { WebGLSceneProgram } from "./WebGLProgramLibrary";

const WEBGL_TEXTURE_UNIT_BASE_MAP = 0;
const WEBGL_TEXTURE_UNIT_CUSTOM_START = 8;

type WarnFn = (key: string, message: string) => void;

export interface WebGLScenePassHost {
	_gl: WebGL2RenderingContext;
	_logWarning: WarnFn;
	_programs: {
		getSceneProgram(material?: Material): WebGLSceneProgram;
	};
	_geometry: {
		getGeometry(packet: DrawPacket): {
			vao: WebGLVertexArrayObject;
			topology: number;
			indexCount: number;
			indexType: number;
		} | null;
	};
	_textures: {
		getBaseColorTexture(texture: any | null): {
			texture: WebGLTexture | null;
			isLinear: boolean;
		};
	};
	_sceneFramebuffer: WebGLFramebuffer | null;
	_sceneNormalTexture: WebGLTexture | null;
	_width: number;
	_height: number;
	_maxTextureImageUnits: number;
	_modelMatrixCache: Map<string, Float32Array>;
	_modelMatrixKeysThisFrame: Set<string>;
	_prevViewProjection: Float32Array | null;
	_taaHistoryValid: boolean;
	_isIncrementalPartial(context: FrameContext | null | undefined): boolean;
	_resolveDirtyRects(
		context: FrameContext | null | undefined,
		viewportWidth: number,
		viewportHeight: number
	): Array<{ x: number; y: number; width: number; height: number }>;
	_resolvePacketsForRect(
		context: FrameContext,
		packets: DrawPacket[],
		rect: { x: number; y: number; width: number; height: number }
	): DrawPacket[];
	_setScissorRect(
		x: number,
		y: number,
		width: number,
		height: number,
		viewportHeight: number
	): void;
	_bindGlobalUniforms(sceneProgram: WebGLSceneProgram, context: FrameContext): void;
	_setCullMode(material: Material): void;
	_drawPacket(
		sceneProgram: WebGLSceneProgram,
		packet: DrawPacket,
		transparentPass: boolean,
		context: FrameContext
	): void;
	_bindShaderMaterialTextures(
		sceneProgram: WebGLSceneProgram,
		material: Material
	): void;
}

export function renderWebGLPackets(
	host: WebGLScenePassHost,
	context: FrameContext,
	packets: DrawPacket[],
	transparent: boolean
): void {
	if (packets.length === 0) {
		return;
	}
	if (!host._sceneFramebuffer) {
		return;
	}

	const gl = host._gl;
	const incrementalPartial = host._isIncrementalPartial(context);
	const dirtyRects = host._resolveDirtyRects(context, host._width, host._height);
	if (incrementalPartial && dirtyRects.length === 0) {
		return;
	}
	gl.bindFramebuffer(gl.FRAMEBUFFER, host._sceneFramebuffer);
	if (!transparent && host._sceneNormalTexture) {
		gl.drawBuffers([
			gl.COLOR_ATTACHMENT0,
			gl.COLOR_ATTACHMENT1,
			gl.COLOR_ATTACHMENT2,
		]);
	} else {
		gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
	}
	gl.activeTexture(gl.TEXTURE0);

	gl.enable(gl.DEPTH_TEST);
	gl.depthMask(!transparent);
	if (transparent) {
		gl.enable(gl.BLEND);
		gl.blendFuncSeparate(
			gl.SRC_ALPHA,
			gl.ONE_MINUS_SRC_ALPHA,
			gl.ONE,
			gl.ONE_MINUS_SRC_ALPHA
		);
	} else {
		gl.disable(gl.BLEND);
	}

	const drawPackets = (): void => {
		let activeProgram: WebGLSceneProgram | null = null;
		for (const packet of packets) {
			const sceneProgram = host._programs.getSceneProgram(packet.material);
			if (activeProgram !== sceneProgram) {
				gl.useProgram(sceneProgram.program);
				host._bindGlobalUniforms(sceneProgram, context);
				activeProgram = sceneProgram;
			}
			host._drawPacket(sceneProgram, packet, transparent, context);
		}
	};
	if (incrementalPartial) {
		gl.enable(gl.SCISSOR_TEST);
		for (const rect of dirtyRects) {
			const rectPackets = host._resolvePacketsForRect(context, packets, rect);
			if (rectPackets.length === 0) {
				continue;
			}
			host._setScissorRect(
				rect.x,
				rect.y,
				rect.width,
				rect.height,
				host._height
			);
			let activeProgram: WebGLSceneProgram | null = null;
			for (const packet of rectPackets) {
				const sceneProgram = host._programs.getSceneProgram(packet.material);
				if (activeProgram !== sceneProgram) {
					gl.useProgram(sceneProgram.program);
					host._bindGlobalUniforms(sceneProgram, context);
					activeProgram = sceneProgram;
				}
				host._drawPacket(sceneProgram, packet, transparent, context);
			}
		}
		gl.disable(gl.SCISSOR_TEST);
	} else {
		drawPackets();
	}

	const currentViewProjection = toFiniteColumnMajorMat4(
		context.camera.viewProjectionMatrix
	);
	if (!currentViewProjection) {
		host._logWarning(
			"webgl-camera-view-projection-invalid",
			"WebGL camera view-projection matrix is non-finite; resetting temporal history."
		);
		host._prevViewProjection = null;
		host._taaHistoryValid = false;
	} else if (host._prevViewProjection) {
		host._prevViewProjection.set(currentViewProjection);
	} else {
		host._prevViewProjection = currentViewProjection;
	}

	gl.depthMask(true);
	gl.disable(gl.BLEND);
	gl.bindVertexArray(null);
}

export function drawWebGLPacket(
	host: WebGLScenePassHost,
	sceneProgram: WebGLSceneProgram,
	packet: DrawPacket,
	transparentPass: boolean,
	context: FrameContext
): void {
	const gl = host._gl;
	const material = packet.material;
	const alphaMode = material.alphaMode ?? AlphaMode.Opaque;
	const requiresTransparent = alphaMode === AlphaMode.Blend;
	if (transparentPass !== requiresTransparent) {
		return;
	}

	if (packet.meshInstance.skeleton) {
		host._logWarning(
			"webgl-skinning-unsupported",
			`WebGL backend does not support skinning yet; skipping mesh instance ${packet.meshInstance.id}`
		);
		return;
	}
	if (!isFiniteMatrix(packet.worldMatrix)) {
		host._logWarning(
			"webgl-world-matrix-invalid",
			`WebGL packet ${packet.id} has non-finite world matrix; skipping`
		);
		return;
	}

	const geometry = host._geometry.getGeometry(packet);
	if (!geometry) {
		return;
	}

	const uniforms = resolveMaterialUniforms(material);
	const normalMatrix = toColumnMajorMat3(packet.normalMatrix);
	if (!normalMatrix) {
		host._logWarning(
			"webgl-normal-matrix-invalid",
			`WebGL packet ${packet.id} has invalid normal matrix; skipping`
		);
		return;
	}

	const resolvedMap = host._textures.getBaseColorTexture(uniforms.baseMap);
	gl.activeTexture(gl.TEXTURE0 + WEBGL_TEXTURE_UNIT_BASE_MAP);
	gl.bindTexture(gl.TEXTURE_2D, resolvedMap.texture);
	host._bindShaderMaterialTextures(sceneProgram, material);

	host._setCullMode(material);
	gl.bindVertexArray(geometry.vao);
	if (sceneProgram.uniforms.model) {
		gl.uniformMatrix4fv(
			sceneProgram.uniforms.model,
			false,
			toColumnMajorMat4(packet.worldMatrix)
		);
	}
	if (sceneProgram.uniforms.normalMatrix) {
		gl.uniformMatrix3fv(sceneProgram.uniforms.normalMatrix, false, normalMatrix);
	}
	if (sceneProgram.uniforms.prevModel) {
		const cacheKey = packet.id;
		host._modelMatrixKeysThisFrame.add(cacheKey);
		let cached = host._modelMatrixCache.get(cacheKey);
		gl.uniformMatrix4fv(
			sceneProgram.uniforms.prevModel,
			false,
			cached ?? toColumnMajorMat4(packet.worldMatrix)
		);
		if (!cached) {
			cached = toColumnMajorMat4(packet.worldMatrix);
			host._modelMatrixCache.set(cacheKey, cached);
		} else {
			cached.set(toColumnMajorMat4(packet.worldMatrix));
		}
	}
	if (sceneProgram.uniforms.shadingModel) {
		gl.uniform1i(sceneProgram.uniforms.shadingModel, uniforms.shadingModel);
	}
	if (sceneProgram.uniforms.baseColor) {
		gl.uniform4fv(sceneProgram.uniforms.baseColor, uniforms.baseColor);
	}
	if (sceneProgram.uniforms.emissive) {
		gl.uniform4f(
			sceneProgram.uniforms.emissive,
			uniforms.emissive[0],
			uniforms.emissive[1],
			uniforms.emissive[2],
			1
		);
	}
	if (sceneProgram.uniforms.pbr) {
		gl.uniform4fv(sceneProgram.uniforms.pbr, uniforms.pbr);
	}
	if (sceneProgram.uniforms.phong) {
		gl.uniform4fv(sceneProgram.uniforms.phong, uniforms.phong);
	}
	if (sceneProgram.uniforms.alpha) {
		gl.uniform4fv(sceneProgram.uniforms.alpha, uniforms.alpha);
	}
	if (sceneProgram.uniforms.baseMap) {
		gl.uniform1i(sceneProgram.uniforms.baseMap, WEBGL_TEXTURE_UNIT_BASE_MAP);
	}
	if (sceneProgram.uniforms.hasBaseMap) {
		gl.uniform1i(sceneProgram.uniforms.hasBaseMap, uniforms.baseMap ? 1 : 0);
	}
	if (sceneProgram.uniforms.baseMapIsLinear) {
		gl.uniform1i(
			sceneProgram.uniforms.baseMapIsLinear,
			resolvedMap.isLinear ? 1 : 0
		);
	}
	if (sceneProgram.uniforms.doubleSided) {
		gl.uniform1i(
			sceneProgram.uniforms.doubleSided,
			material.doubleSided || material.cullMode === "none" ? 1 : 0
		);
	}

	gl.drawElements(
		geometry.topology,
		geometry.indexCount,
		geometry.indexType,
		0
	);
	gl.bindVertexArray(null);
}

export function bindWebGLShaderMaterialTextures(
	host: WebGLScenePassHost,
	sceneProgram: WebGLSceneProgram,
	material: Material
): void {
	if (!(material instanceof ShaderMaterial)) {
		return;
	}
	const bindings = material.getTextureBindings();
	if (bindings.length <= 0) {
		return;
	}

	const gl = host._gl;
	let textureUnit = WEBGL_TEXTURE_UNIT_CUSTOM_START;
	const boundUniforms = new Set<string>();
	for (const binding of bindings) {
		if (boundUniforms.has(binding.webglUniform)) {
			continue;
		}
		boundUniforms.add(binding.webglUniform);
		const uniform = sceneProgram.uniforms.customSamplers[binding.webglUniform];
		if (!uniform) {
			continue;
		}
		if (textureUnit >= host._maxTextureImageUnits) {
			host._logWarning(
				`webgl-shader-material-texture-unit-limit-${material.shaderId}`,
				`ShaderMaterial ${material.name} custom textures exceed MAX_TEXTURE_IMAGE_UNITS=${host._maxTextureImageUnits}; extra bindings are ignored.`
			);
			break;
		}
		const resolved = host._textures.getBaseColorTexture(binding.texture);
		gl.activeTexture(gl.TEXTURE0 + textureUnit);
		gl.bindTexture(gl.TEXTURE_2D, resolved.texture);
		gl.uniform1i(uniform, textureUnit);
		textureUnit++;
	}
	gl.activeTexture(gl.TEXTURE0 + WEBGL_TEXTURE_UNIT_BASE_MAP);
}
