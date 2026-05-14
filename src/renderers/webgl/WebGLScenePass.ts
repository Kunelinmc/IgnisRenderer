import {
	materialWritesDepth,
	type Material,
} from "../../materials/Material";
import { isMaterialTransparentPass } from "../../materials/transparency";
import {
	ShaderMaterial,
	type ShaderTargetMode,
	type ResolvedShaderMaterialUniformBinding,
} from "../../materials/ShaderMaterial";
import type { DrawPacket, FrameContext } from "../../pipeline/types";
import {
	isFiniteMatrix,
	toColumnMajorMat3,
	toColumnMajorMat4,
	toFiniteColumnMajorMat4,
} from "./WebGLFrameMath";
import {
	resolveMaterialUniforms,
	resolveTextureUVTransform,
} from "./WebGLMaterialUniformResolver";
import type { WebGLSceneProgram } from "./WebGLProgramLibrary";
import { Logger } from "../../foundation/Logger";

const WEBGL_TEXTURE_UNIT_BASE_MAP = 0;
const WEBGL_TEXTURE_UNIT_NORMAL_MAP = 8;
const WEBGL_TEXTURE_UNIT_METALLIC_ROUGHNESS_MAP = 9;
const WEBGL_TEXTURE_UNIT_EMISSIVE_MAP = 10;
const WEBGL_TEXTURE_UNIT_OCCLUSION_MAP = 11;
const WEBGL_TEXTURE_UNIT_IRIDESCENCE_MAP = 12;
const WEBGL_TEXTURE_UNIT_IRIDESCENCE_THICKNESS_MAP = 15;
const WEBGL_TEXTURE_UNIT_CUSTOM_START = 17;

function logWebGLScenePassWarning(key: string, message: string): void {
	Logger.warn(`[${key}] ${message}`, {
		scope: "WebGLScenePass",
		onceKey: key,
	});
}

export interface WebGLScenePassHost {
	_gl: WebGL2RenderingContext;
	_programs: {
		getSceneProgram(
			material?: Material,
			mode?: ShaderTargetMode
		): WebGLSceneProgram;
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
	_oitPassMode: 0 | 1 | 2;
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
	_bindShaderMaterialUniforms(
		sceneProgram: WebGLSceneProgram,
		material: Material
	): void;
}

export interface WebGLSceneRenderOptions {
	framebuffer?: WebGLFramebuffer | null;
	drawBuffers?: number[];
	blendMode?: "legacy" | "oit-accum" | "oit-reveal";
	oitPassMode?: 0 | 1 | 2;
}

export function renderWebGLPackets(
	host: WebGLScenePassHost,
	context: FrameContext,
	packets: DrawPacket[],
	transparent: boolean,
	options: WebGLSceneRenderOptions = {}
): void {
	if (packets.length === 0) {
		return;
	}
	const framebuffer = options.framebuffer ?? host._sceneFramebuffer;
	if (!framebuffer) {
		return;
	}

	const gl = host._gl;
	const incrementalPartial = host._isIncrementalPartial(context);
	const dirtyRects = host._resolveDirtyRects(context, host._width, host._height);
	if (incrementalPartial && dirtyRects.length === 0) {
		return;
	}
	gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
	const drawBuffers =
		options.drawBuffers ??
		(!transparent && host._sceneNormalTexture ?
			[gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1, gl.COLOR_ATTACHMENT2]
		:	[gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
	const sceneProgramMode: ShaderTargetMode =
		drawBuffers.length >= 3 ? "mrt" : "single";
	gl.drawBuffers(drawBuffers);
	gl.activeTexture(gl.TEXTURE0);

	gl.enable(gl.DEPTH_TEST);
	gl.depthMask(!transparent);
	if (transparent) {
		gl.enable(gl.BLEND);
		switch (options.blendMode) {
			case "oit-accum":
				gl.blendFuncSeparate(gl.ONE, gl.ONE, gl.ONE, gl.ONE);
				break;
			case "oit-reveal":
				gl.blendFuncSeparate(
					gl.ZERO,
					gl.ONE_MINUS_SRC_ALPHA,
					gl.ZERO,
					gl.ONE_MINUS_SRC_ALPHA
				);
				break;
			default:
				gl.blendFuncSeparate(
					gl.SRC_ALPHA,
					gl.ONE_MINUS_SRC_ALPHA,
					gl.ONE,
					gl.ONE_MINUS_SRC_ALPHA
				);
				break;
		}
	} else {
		gl.disable(gl.BLEND);
	}
	const previousOITPassMode = host._oitPassMode;
	host._oitPassMode = options.oitPassMode ?? 0;
	try {
		const drawPackets = (): void => {
			let activeProgram: WebGLSceneProgram | null = null;
			for (const packet of packets) {
				const sceneProgram = host._programs.getSceneProgram(
					packet.material,
					sceneProgramMode
				);
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
					const sceneProgram = host._programs.getSceneProgram(
						packet.material,
						sceneProgramMode
					);
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
	} finally {
		host._oitPassMode = previousOITPassMode;
	}

	const currentViewProjection = toFiniteColumnMajorMat4(
		context.camera.viewProjectionMatrix
	);
	if (!currentViewProjection) {
		logWebGLScenePassWarning(
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
	const requiresTransparent = isMaterialTransparentPass(material);
	if (transparentPass !== requiresTransparent) {
		return;
	}

	if (packet.meshInstance.skeleton) {
		logWebGLScenePassWarning(
			"webgl-skinning-unsupported",
			`WebGL backend does not support skinning yet; skipping mesh instance ${packet.meshInstance.id}`
		);
		return;
	}
	if (!isFiniteMatrix(packet.worldMatrix)) {
		logWebGLScenePassWarning(
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
	const baseMapUVTransform = resolveTextureUVTransform(uniforms.baseMap);
	const metallicRoughnessMapUVTransform = resolveTextureUVTransform(
		uniforms.metallicRoughnessMap
	);
	const normalMapUVTransform = resolveTextureUVTransform(uniforms.normalMap);
	const emissiveMapUVTransform = resolveTextureUVTransform(uniforms.emissiveMap);
	const occlusionMapUVTransform = resolveTextureUVTransform(uniforms.occlusionMap);
	const iridescenceMapUVTransform = resolveTextureUVTransform(
		uniforms.iridescenceMap
	);
	const iridescenceThicknessMapUVTransform = resolveTextureUVTransform(
		uniforms.iridescenceThicknessMap
	);
	const normalMatrix = toColumnMajorMat3(packet.normalMatrix);
	if (!normalMatrix) {
		logWebGLScenePassWarning(
			"webgl-normal-matrix-invalid",
			`WebGL packet ${packet.id} has invalid normal matrix; skipping`
		);
		return;
	}

	const resolvedMap = host._textures.getBaseColorTexture(uniforms.baseMap);
	const resolvedNormalMap = host._textures.getBaseColorTexture(uniforms.normalMap);
	const resolvedMetallicRoughnessMap = host._textures.getBaseColorTexture(
		uniforms.metallicRoughnessMap
	);
	const resolvedEmissiveMap = host._textures.getBaseColorTexture(
		uniforms.emissiveMap
	);
	const resolvedOcclusionMap = host._textures.getBaseColorTexture(
		uniforms.occlusionMap
	);
	const resolvedIridescenceMap = host._textures.getBaseColorTexture(
		uniforms.iridescenceMap
	);
	const resolvedIridescenceThicknessMap = host._textures.getBaseColorTexture(
		uniforms.iridescenceThicknessMap
	);
	gl.activeTexture(gl.TEXTURE0 + WEBGL_TEXTURE_UNIT_BASE_MAP);
	gl.bindTexture(gl.TEXTURE_2D, resolvedMap.texture);
	gl.activeTexture(gl.TEXTURE0 + WEBGL_TEXTURE_UNIT_NORMAL_MAP);
	gl.bindTexture(gl.TEXTURE_2D, resolvedNormalMap.texture);
	gl.activeTexture(gl.TEXTURE0 + WEBGL_TEXTURE_UNIT_METALLIC_ROUGHNESS_MAP);
	gl.bindTexture(gl.TEXTURE_2D, resolvedMetallicRoughnessMap.texture);
	gl.activeTexture(gl.TEXTURE0 + WEBGL_TEXTURE_UNIT_EMISSIVE_MAP);
	gl.bindTexture(gl.TEXTURE_2D, resolvedEmissiveMap.texture);
	gl.activeTexture(gl.TEXTURE0 + WEBGL_TEXTURE_UNIT_OCCLUSION_MAP);
	gl.bindTexture(gl.TEXTURE_2D, resolvedOcclusionMap.texture);
	gl.activeTexture(gl.TEXTURE0 + WEBGL_TEXTURE_UNIT_IRIDESCENCE_MAP);
	gl.bindTexture(gl.TEXTURE_2D, resolvedIridescenceMap.texture);
	gl.activeTexture(gl.TEXTURE0 + WEBGL_TEXTURE_UNIT_IRIDESCENCE_THICKNESS_MAP);
	gl.bindTexture(gl.TEXTURE_2D, resolvedIridescenceThicknessMap.texture);
	host._bindShaderMaterialTextures(sceneProgram, material);
	host._bindShaderMaterialUniforms(sceneProgram, material);

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
	if (sceneProgram.uniforms.transmissionVolume) {
		gl.uniform4fv(
			sceneProgram.uniforms.transmissionVolume,
			uniforms.transmissionVolume
		);
	}
	if (sceneProgram.uniforms.iridescence) {
		gl.uniform4fv(sceneProgram.uniforms.iridescence, uniforms.iridescence);
	}
	if (sceneProgram.uniforms.attenuationColor) {
		gl.uniform4fv(
			sceneProgram.uniforms.attenuationColor,
			uniforms.attenuationColor
		);
	}
	if (sceneProgram.uniforms.phong) {
		gl.uniform4fv(sceneProgram.uniforms.phong, uniforms.phong);
	}
	if (sceneProgram.uniforms.alpha) {
		gl.uniform4fv(sceneProgram.uniforms.alpha, uniforms.alpha);
	}
	if (sceneProgram.uniforms.oitPassMode) {
		gl.uniform1i(sceneProgram.uniforms.oitPassMode, host._oitPassMode);
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
	if (sceneProgram.uniforms.baseMapUV) {
		gl.uniform1i(sceneProgram.uniforms.baseMapUV, uniforms.baseMapUV);
	}
	if (sceneProgram.uniforms.baseMapTransformA) {
		gl.uniform4f(
			sceneProgram.uniforms.baseMapTransformA,
			baseMapUVTransform.repeatX,
			baseMapUVTransform.repeatY,
			baseMapUVTransform.offsetX,
			baseMapUVTransform.offsetY
		);
	}
	if (sceneProgram.uniforms.baseMapTransformB) {
		gl.uniform2f(
			sceneProgram.uniforms.baseMapTransformB,
			baseMapUVTransform.cosRotation,
			baseMapUVTransform.sinRotation
		);
	}
	if (sceneProgram.uniforms.metallicRoughnessMap) {
		gl.uniform1i(
			sceneProgram.uniforms.metallicRoughnessMap,
			WEBGL_TEXTURE_UNIT_METALLIC_ROUGHNESS_MAP
		);
	}
	if (sceneProgram.uniforms.hasMetallicRoughnessMap) {
		gl.uniform1i(
			sceneProgram.uniforms.hasMetallicRoughnessMap,
			uniforms.metallicRoughnessMap ? 1 : 0
		);
	}
	if (sceneProgram.uniforms.metallicRoughnessMapUV) {
		gl.uniform1i(
			sceneProgram.uniforms.metallicRoughnessMapUV,
			uniforms.metallicRoughnessMapUV
		);
	}
	if (sceneProgram.uniforms.metallicRoughnessMapTransformA) {
		gl.uniform4f(
			sceneProgram.uniforms.metallicRoughnessMapTransformA,
			metallicRoughnessMapUVTransform.repeatX,
			metallicRoughnessMapUVTransform.repeatY,
			metallicRoughnessMapUVTransform.offsetX,
			metallicRoughnessMapUVTransform.offsetY
		);
	}
	if (sceneProgram.uniforms.metallicRoughnessMapTransformB) {
		gl.uniform2f(
			sceneProgram.uniforms.metallicRoughnessMapTransformB,
			metallicRoughnessMapUVTransform.cosRotation,
			metallicRoughnessMapUVTransform.sinRotation
		);
	}
	if (sceneProgram.uniforms.normalMap) {
		gl.uniform1i(sceneProgram.uniforms.normalMap, WEBGL_TEXTURE_UNIT_NORMAL_MAP);
	}
	if (sceneProgram.uniforms.hasNormalMap) {
		gl.uniform1i(sceneProgram.uniforms.hasNormalMap, uniforms.normalMap ? 1 : 0);
	}
	if (sceneProgram.uniforms.normalMapUV) {
		gl.uniform1i(sceneProgram.uniforms.normalMapUV, uniforms.normalMapUV);
	}
	if (sceneProgram.uniforms.normalMapTransformA) {
		gl.uniform4f(
			sceneProgram.uniforms.normalMapTransformA,
			normalMapUVTransform.repeatX,
			normalMapUVTransform.repeatY,
			normalMapUVTransform.offsetX,
			normalMapUVTransform.offsetY
		);
	}
	if (sceneProgram.uniforms.normalMapTransformB) {
		gl.uniform2f(
			sceneProgram.uniforms.normalMapTransformB,
			normalMapUVTransform.cosRotation,
			normalMapUVTransform.sinRotation
		);
	}
	if (sceneProgram.uniforms.normalScale) {
		gl.uniform1f(sceneProgram.uniforms.normalScale, uniforms.normalScale);
	}
	if (sceneProgram.uniforms.emissiveMap) {
		gl.uniform1i(sceneProgram.uniforms.emissiveMap, WEBGL_TEXTURE_UNIT_EMISSIVE_MAP);
	}
	if (sceneProgram.uniforms.hasEmissiveMap) {
		gl.uniform1i(sceneProgram.uniforms.hasEmissiveMap, uniforms.emissiveMap ? 1 : 0);
	}
	if (sceneProgram.uniforms.emissiveMapIsLinear) {
		gl.uniform1i(
			sceneProgram.uniforms.emissiveMapIsLinear,
			resolvedEmissiveMap.isLinear ? 1 : 0
		);
	}
	if (sceneProgram.uniforms.emissiveMapUV) {
		gl.uniform1i(sceneProgram.uniforms.emissiveMapUV, uniforms.emissiveMapUV);
	}
	if (sceneProgram.uniforms.emissiveMapTransformA) {
		gl.uniform4f(
			sceneProgram.uniforms.emissiveMapTransformA,
			emissiveMapUVTransform.repeatX,
			emissiveMapUVTransform.repeatY,
			emissiveMapUVTransform.offsetX,
			emissiveMapUVTransform.offsetY
		);
	}
	if (sceneProgram.uniforms.emissiveMapTransformB) {
		gl.uniform2f(
			sceneProgram.uniforms.emissiveMapTransformB,
			emissiveMapUVTransform.cosRotation,
			emissiveMapUVTransform.sinRotation
		);
	}
	if (sceneProgram.uniforms.occlusionMap) {
		gl.uniform1i(sceneProgram.uniforms.occlusionMap, WEBGL_TEXTURE_UNIT_OCCLUSION_MAP);
	}
	if (sceneProgram.uniforms.hasOcclusionMap) {
		gl.uniform1i(sceneProgram.uniforms.hasOcclusionMap, uniforms.occlusionMap ? 1 : 0);
	}
	if (sceneProgram.uniforms.occlusionMapUV) {
		gl.uniform1i(sceneProgram.uniforms.occlusionMapUV, uniforms.occlusionMapUV);
	}
	if (sceneProgram.uniforms.occlusionMapTransformA) {
		gl.uniform4f(
			sceneProgram.uniforms.occlusionMapTransformA,
			occlusionMapUVTransform.repeatX,
			occlusionMapUVTransform.repeatY,
			occlusionMapUVTransform.offsetX,
			occlusionMapUVTransform.offsetY
		);
	}
	if (sceneProgram.uniforms.occlusionMapTransformB) {
		gl.uniform2f(
			sceneProgram.uniforms.occlusionMapTransformB,
			occlusionMapUVTransform.cosRotation,
			occlusionMapUVTransform.sinRotation
		);
	}
	if (sceneProgram.uniforms.occlusionStrength) {
		gl.uniform1f(sceneProgram.uniforms.occlusionStrength, uniforms.occlusionStrength);
	}
	if (sceneProgram.uniforms.iridescenceMap) {
		gl.uniform1i(
			sceneProgram.uniforms.iridescenceMap,
			WEBGL_TEXTURE_UNIT_IRIDESCENCE_MAP
		);
	}
	if (sceneProgram.uniforms.hasIridescenceMap) {
		gl.uniform1i(
			sceneProgram.uniforms.hasIridescenceMap,
			uniforms.iridescenceMap ? 1 : 0
		);
	}
	if (sceneProgram.uniforms.iridescenceMapUV) {
		gl.uniform1i(
			sceneProgram.uniforms.iridescenceMapUV,
			uniforms.iridescenceMapUV
		);
	}
	if (sceneProgram.uniforms.iridescenceMapTransformA) {
		gl.uniform4f(
			sceneProgram.uniforms.iridescenceMapTransformA,
			iridescenceMapUVTransform.repeatX,
			iridescenceMapUVTransform.repeatY,
			iridescenceMapUVTransform.offsetX,
			iridescenceMapUVTransform.offsetY
		);
	}
	if (sceneProgram.uniforms.iridescenceMapTransformB) {
		gl.uniform2f(
			sceneProgram.uniforms.iridescenceMapTransformB,
			iridescenceMapUVTransform.cosRotation,
			iridescenceMapUVTransform.sinRotation
		);
	}
	if (sceneProgram.uniforms.iridescenceThicknessMap) {
		gl.uniform1i(
			sceneProgram.uniforms.iridescenceThicknessMap,
			WEBGL_TEXTURE_UNIT_IRIDESCENCE_THICKNESS_MAP
		);
	}
	if (sceneProgram.uniforms.hasIridescenceThicknessMap) {
		gl.uniform1i(
			sceneProgram.uniforms.hasIridescenceThicknessMap,
			uniforms.iridescenceThicknessMap ? 1 : 0
		);
	}
	if (sceneProgram.uniforms.iridescenceThicknessMapUV) {
		gl.uniform1i(
			sceneProgram.uniforms.iridescenceThicknessMapUV,
			uniforms.iridescenceThicknessMapUV
		);
	}
	if (sceneProgram.uniforms.iridescenceThicknessMapTransformA) {
		gl.uniform4f(
			sceneProgram.uniforms.iridescenceThicknessMapTransformA,
			iridescenceThicknessMapUVTransform.repeatX,
			iridescenceThicknessMapUVTransform.repeatY,
			iridescenceThicknessMapUVTransform.offsetX,
			iridescenceThicknessMapUVTransform.offsetY
		);
	}
	if (sceneProgram.uniforms.iridescenceThicknessMapTransformB) {
		gl.uniform2f(
			sceneProgram.uniforms.iridescenceThicknessMapTransformB,
			iridescenceThicknessMapUVTransform.cosRotation,
			iridescenceThicknessMapUVTransform.sinRotation
		);
	}
	if (sceneProgram.uniforms.doubleSided) {
		gl.uniform1i(
			sceneProgram.uniforms.doubleSided,
			material.doubleSided || material.cullMode === "none" ? 1 : 0
		);
	}

	gl.depthMask(!transparentPass && materialWritesDepth(material));
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
			logWebGLScenePassWarning(
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

export function bindWebGLShaderMaterialUniforms(
	host: WebGLScenePassHost,
	sceneProgram: WebGLSceneProgram,
	material: Material
): void {
	if (!(material instanceof ShaderMaterial)) {
		return;
	}
	const bindings = material.getUniformBindings();
	if (bindings.length <= 0) {
		return;
	}
	const gl = host._gl;
	for (const binding of bindings) {
		const uniform = sceneProgram.uniforms.customUniforms[binding.webglUniform];
		if (!uniform) {
			continue;
		}
		bindWebGLShaderMaterialUniform(gl, uniform, binding);
	}
}

function bindWebGLShaderMaterialUniform(
	gl: WebGL2RenderingContext,
	uniform: WebGLUniformLocation,
	binding: ResolvedShaderMaterialUniformBinding
): void {
	const value = binding.value;
	switch (binding.type) {
		case "f32":
			gl.uniform1f(uniform, value as number);
			break;
		case "i32":
			gl.uniform1i(uniform, value as number);
			break;
		case "u32":
			gl.uniform1ui(uniform, value as number);
			break;
		case "vec2f":
			gl.uniform2fv(uniform, value as readonly number[]);
			break;
		case "vec3f":
			gl.uniform3fv(uniform, value as readonly number[]);
			break;
		case "vec4f":
			gl.uniform4fv(uniform, value as readonly number[]);
			break;
		case "vec2i":
			gl.uniform2iv(uniform, value as readonly number[]);
			break;
		case "vec3i":
			gl.uniform3iv(uniform, value as readonly number[]);
			break;
		case "vec4i":
			gl.uniform4iv(uniform, value as readonly number[]);
			break;
		case "vec2u":
			gl.uniform2uiv(uniform, value as readonly number[]);
			break;
		case "vec3u":
			gl.uniform3uiv(uniform, value as readonly number[]);
			break;
		case "vec4u":
			gl.uniform4uiv(uniform, value as readonly number[]);
			break;
		case "mat4x4f":
			gl.uniformMatrix4fv(
				uniform,
				false,
				toColumnMajorMat4(value as number[][])
			);
			break;
	}
}

