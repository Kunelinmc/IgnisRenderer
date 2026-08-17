import type { Material } from "../../materials/Material";
import {
	isMaterialTransparentPass,
	materialUsesTransmission,
} from "../../materials/transparency";
import {
	ShaderMaterial,
	type ShaderTargetMode,
	type ResolvedShaderMaterialUniformBinding,
} from "../../materials/ShaderMaterial";
import { Matrix4 } from "../../maths/Matrix4";
import type { Matrix3Arr } from "../../maths/types";
import type { DrawPacket, FrameContext } from "../../pipeline/types";
import {
	resolveMaterialUniforms,
	resolveTextureUVTransform,
} from "./WebGLMaterialUniformResolver";
import type { WebGLSceneProgram } from "./WebGLSceneProgram";
import type {
	WebGLSceneDepthVariantDescriptor,
	WebGLSceneVariantDescriptor,
} from "./WebGLSceneProgramVariants";
import { Logger } from "../../foundation/Logger";
import { getWebGLSceneSamplerUnit } from "./WebGLSceneSamplerLayout";

function logWebGLScenePassWarning(key: string, message: string): void {
	Logger.warn(`[${key}] ${message}`, {
		scope: "WebGLScenePass",
		onceKey: key,
	});
}

function toColumnMajorMat3(
	matrix: Matrix4 | Matrix3Arr
): Float32Array | null {
	const rows = Array.isArray(matrix) ? matrix : matrix.elements;
	if (rows.length < 3) {
		return null;
	}
	const values = [
		rows[0][0],
		rows[1][0],
		rows[2][0],
		rows[0][1],
		rows[1][1],
		rows[2][1],
		rows[0][2],
		rows[1][2],
		rows[2][2],
	];
	for (let i = 0; i < values.length; i++) {
		if (!Number.isFinite(values[i])) {
			return null;
		}
	}
	return new Float32Array(values);
}

export interface WebGLScenePassHost {
	_gl: WebGL2RenderingContext;
	_scenePrograms: {
		getSceneProgram(
			material?: Material,
			mode?: ShaderTargetMode,
			variant?: WebGLSceneVariantDescriptor
		): WebGLSceneProgram;
		getSceneDepthPrepassProgram(
			material?: Material,
			mode?: ShaderTargetMode,
			variant?: WebGLSceneDepthVariantDescriptor
		): WebGLSceneProgram | null;
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
	_postColorTexture: WebGLTexture | null;
	_transmissionBackgroundTexture: WebGLTexture | null;
	_transmissionDepthTexture: WebGLTexture | null;
	_materialGBufferEnabled: boolean;
	_oitPassMode: 0 | 1 | 2;
	_width: number;
	_height: number;
	_activeDrawBuffers?: number[];
	_modelMatrixCache: Map<string, Float32Array>;
	_modelMatrixKeysThisFrame: Set<string>;
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
	getShadowSamplingState(): { readonly enabled: boolean };
	_setCullMode(material: Material): void;
	_drawPacket(
		sceneProgram: WebGLSceneProgram,
		packet: DrawPacket,
		transparentPass: boolean,
		context: FrameContext,
		options?: WebGLPacketDrawOptions
	): void;
	_bindShaderMaterialTextures(
		sceneProgram: WebGLSceneProgram,
		material: Material
	): void;
	_bindShaderMaterialUniforms(
		sceneProgram: WebGLSceneProgram,
		material: Material
	): void;
	_resolveSceneProgramVariant?(
		context: FrameContext,
		packet: DrawPacket,
		mode: ShaderTargetMode
	): WebGLSceneVariantDescriptor | null;
	_resolveSceneDepthPrepassVariant?(
		packet: DrawPacket
	): WebGLSceneDepthVariantDescriptor | null;
}

export interface WebGLSceneRenderOptions {
	framebuffer?: WebGLFramebuffer | null;
	drawBuffers?: number[];
	blendMode?: "legacy" | "disabled" | "oit-accum" | "oit-reveal";
	oitPassMode?: 0 | 1 | 2;
	earlyZPacketIds?: ReadonlySet<string>;
}

export interface WebGLPacketDrawOptions {
	earlyZColor?: boolean;
}

const WEBGL_EARLY_Z_COLOR_DRAW_OPTIONS: WebGLPacketDrawOptions = {
	earlyZColor: true,
};

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
		(!transparent && host._materialGBufferEnabled ?
			[
				gl.COLOR_ATTACHMENT0,
				gl.COLOR_ATTACHMENT1,
				gl.COLOR_ATTACHMENT2,
				gl.COLOR_ATTACHMENT3,
				gl.COLOR_ATTACHMENT4,
			]
		: !transparent && host._sceneNormalTexture ?
			[gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1, gl.COLOR_ATTACHMENT2]
		:	[gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
	const sceneProgramMode: ShaderTargetMode =
		drawBuffers.length >= 3 ? "mrt" : "single";
	gl.drawBuffers(drawBuffers);
	host._activeDrawBuffers = drawBuffers;
	gl.activeTexture(gl.TEXTURE0);

	gl.enable(gl.DEPTH_TEST);
	gl.depthFunc(gl.LESS);
	gl.depthMask(!transparent);
	if (transparent) {
		if (options.blendMode === "disabled") gl.disable(gl.BLEND);
		else gl.enable(gl.BLEND);
		switch (options.blendMode) {
			case "disabled":
				break;
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
				const sceneProgram = host._scenePrograms.getSceneProgram(
					packet.material,
					sceneProgramMode,
					resolveSceneProgramVariant(host, context, packet, sceneProgramMode)
				);
				if (activeProgram !== sceneProgram) {
					gl.useProgram(sceneProgram.program);
					host._bindGlobalUniforms(sceneProgram, context);
					activeProgram = sceneProgram;
				}
				host._drawPacket(
					sceneProgram,
					packet,
					transparent,
					context,
					!transparent && options.earlyZPacketIds?.has(packet.id) ?
						WEBGL_EARLY_Z_COLOR_DRAW_OPTIONS
					:	undefined
				);
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
					const sceneProgram = host._scenePrograms.getSceneProgram(
						packet.material,
						sceneProgramMode,
						resolveSceneProgramVariant(
							host,
							context,
							packet,
							sceneProgramMode
						)
					);
					if (activeProgram !== sceneProgram) {
						gl.useProgram(sceneProgram.program);
						host._bindGlobalUniforms(sceneProgram, context);
						activeProgram = sceneProgram;
					}
					host._drawPacket(
						sceneProgram,
						packet,
						transparent,
						context,
						!transparent && options.earlyZPacketIds?.has(packet.id) ?
							WEBGL_EARLY_Z_COLOR_DRAW_OPTIONS
						:	undefined
					);
				}
			}
			gl.disable(gl.SCISSOR_TEST);
		} else {
			drawPackets();
		}
	} finally {
		host._oitPassMode = previousOITPassMode;
	}

	gl.depthMask(true);
	gl.depthFunc(gl.LESS);
	gl.disable(gl.BLEND);
	gl.bindVertexArray(null);
}

export function renderWebGLEarlyZPrepass(
	host: WebGLScenePassHost,
	context: FrameContext,
	packets: DrawPacket[]
): Set<string> {
	const prepassedPacketIds = new Set<string>();
	if (packets.length === 0) {
		return prepassedPacketIds;
	}
	if (!host._sceneFramebuffer) {
		return prepassedPacketIds;
	}

	const gl = host._gl;
	const incrementalPartial = host._isIncrementalPartial(context);
	const dirtyRects = host._resolveDirtyRects(context, host._width, host._height);
	if (incrementalPartial && dirtyRects.length === 0) {
		return prepassedPacketIds;
	}
	const drawBuffers =
		host._materialGBufferEnabled ?
			[
				gl.COLOR_ATTACHMENT0,
				gl.COLOR_ATTACHMENT1,
				gl.COLOR_ATTACHMENT2,
				gl.COLOR_ATTACHMENT3,
				gl.COLOR_ATTACHMENT4,
			]
		: host._sceneNormalTexture ?
			[gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1, gl.COLOR_ATTACHMENT2]
		:	[gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1];
	const sceneProgramMode: ShaderTargetMode =
		drawBuffers.length >= 3 ? "mrt" : "single";

	gl.bindFramebuffer(gl.FRAMEBUFFER, host._sceneFramebuffer);
	gl.drawBuffers([gl.NONE]);
	gl.colorMask(false, false, false, false);
	gl.activeTexture(gl.TEXTURE0);
	gl.enable(gl.DEPTH_TEST);
	gl.disable(gl.BLEND);
	gl.depthMask(true);
	gl.depthFunc(gl.LESS);
	try {
		const drawDepthPackets = (depthPackets: DrawPacket[]): void => {
			let activeProgram: WebGLSceneProgram | null = null;
			for (const packet of depthPackets) {
				const depthProgram = resolveWebGLDepthPrepassProgram(
					host,
					packet,
					sceneProgramMode
				);
				if (!depthProgram) {
					continue;
				}
				if (activeProgram !== depthProgram) {
					gl.useProgram(depthProgram.program);
					host._bindGlobalUniforms(depthProgram, context);
					activeProgram = depthProgram;
				}
				if (drawWebGLDepthPrepassPacket(host, depthProgram, packet)) {
					prepassedPacketIds.add(packet.id);
				}
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
				drawDepthPackets(rectPackets);
			}
			gl.disable(gl.SCISSOR_TEST);
		} else {
			drawDepthPackets(packets);
		}
	} finally {
		gl.disable(gl.SCISSOR_TEST);
		gl.colorMask(true, true, true, true);
		gl.drawBuffers(drawBuffers);
		host._activeDrawBuffers = drawBuffers;
		gl.depthMask(true);
		gl.depthFunc(gl.LESS);
		gl.bindVertexArray(null);
	}
	return prepassedPacketIds;
}

export function drawWebGLPacket(
	host: WebGLScenePassHost,
	sceneProgram: WebGLSceneProgram,
	packet: DrawPacket,
	transparentPass: boolean,
	context: FrameContext,
	options: WebGLPacketDrawOptions = {}
): void {
	const gl = host._gl;
	const samplerUnit = (name: string): number =>
		getWebGLSceneSamplerUnit(sceneProgram.samplerLayout, name);
	const isSamplerActive = (name: string): boolean =>
		Object.prototype.hasOwnProperty.call(sceneProgram.samplerLayout.units, name);
	const bindSceneTexture = (
		name: string,
		texture: WebGLTexture | null,
	): void => {
		if (!isSamplerActive(name)) return;
		gl.activeTexture(gl.TEXTURE0 + samplerUnit(name));
		gl.bindTexture(gl.TEXTURE_2D, texture);
	};
	const material = packet.material;
	const requiresTransparent = isMaterialTransparentPass(material);
	if (transparentPass !== requiresTransparent) {
		return;
	}

	const activeDrawBuffers = host._activeDrawBuffers;

	if (packet.meshInstance.skeleton) {
		logWebGLScenePassWarning(
			"webgl-skinning-unsupported",
			`WebGL backend does not support skinning yet; skipping mesh instance ${packet.meshInstance.id}`
		);
		return;
	}
	if (!Matrix4.isFinite(packet.worldMatrix)) {
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
	const specularMapUVTransform = resolveTextureUVTransform(uniforms.specularMap);
	const specularColorMapUVTransform = resolveTextureUVTransform(
		uniforms.specularColorMap,
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
	const anisotropyMapUVTransform = resolveTextureUVTransform(
		uniforms.anisotropyMap
	);
	const extensionMaps = [
		["Clearcoat", uniforms.clearcoatMap, uniforms.clearcoatMapUV],
		["ClearcoatRoughness", uniforms.clearcoatRoughnessMap, uniforms.clearcoatRoughnessMapUV],
		["ClearcoatNormal", uniforms.clearcoatNormalMap, uniforms.clearcoatNormalMapUV],
		["SheenColor", uniforms.sheenColorMap, uniforms.sheenColorMapUV],
		["SheenRoughness", uniforms.sheenRoughnessMap, uniforms.sheenRoughnessMapUV],
		["Transmission", uniforms.transmissionMap, uniforms.transmissionMapUV],
		["Thickness", uniforms.thicknessMap, uniforms.thicknessMapUV],
	] as const;
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
	const resolvedSpecularMap = host._textures.getBaseColorTexture(uniforms.specularMap);
	const resolvedSpecularColorMap = host._textures.getBaseColorTexture(
		uniforms.specularColorMap
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
	const resolvedAnisotropyMap = host._textures.getBaseColorTexture(
		uniforms.anisotropyMap
	);
	const pbrExtensionUniforms = sceneProgram.uniforms.pbrExtensionUniforms ?? {};
	const transmissionBackgroundMap =
		pbrExtensionUniforms.uTransmissionBackgroundMap;
	const hasTransmissionBackgroundMap =
		pbrExtensionUniforms.uHasTransmissionBackgroundMap;
	const transmissionBackgroundInvSize =
		pbrExtensionUniforms.uTransmissionBackgroundInvSize;
	const transmissionDepthMap = pbrExtensionUniforms.uTransmissionDepthMap;
	const hasTransmissionDepthMap = pbrExtensionUniforms.uHasTransmissionDepthMap;
	const transmissionModelScale = pbrExtensionUniforms.uTransmissionModelScale;
	if (
		materialUsesTransmission(material) &&
		host._transmissionBackgroundTexture &&
		isSamplerActive("uTransmissionBackgroundMap")
	) {
		const unit = samplerUnit("uTransmissionBackgroundMap");
		gl.activeTexture(gl.TEXTURE0 + unit);
		gl.bindTexture(gl.TEXTURE_2D, host._transmissionBackgroundTexture);
		if (transmissionBackgroundMap) {
			gl.uniform1i(transmissionBackgroundMap, unit);
		}
		if (hasTransmissionBackgroundMap) {
			gl.uniform1i(hasTransmissionBackgroundMap, 1);
		}
		if (transmissionBackgroundInvSize) {
			gl.uniform2f(
				transmissionBackgroundInvSize,
				1 / Math.max(host._width, 1),
				1 / Math.max(host._height, 1),
			);
		}
		if (
			host._transmissionDepthTexture &&
			isSamplerActive("uTransmissionDepthMap")
		) {
			const depthUnit = samplerUnit("uTransmissionDepthMap");
			gl.activeTexture(gl.TEXTURE0 + depthUnit);
			gl.bindTexture(gl.TEXTURE_2D, host._transmissionDepthTexture);
			if (transmissionDepthMap) gl.uniform1i(transmissionDepthMap, depthUnit);
			if (hasTransmissionDepthMap) gl.uniform1i(hasTransmissionDepthMap, 1);
		} else if (hasTransmissionDepthMap) {
			gl.uniform1i(hasTransmissionDepthMap, 0);
		}
	} else if (hasTransmissionBackgroundMap) {
		gl.uniform1i(hasTransmissionBackgroundMap, 0);
		if (hasTransmissionDepthMap) gl.uniform1i(hasTransmissionDepthMap, 0);
	}
	for (let index = 0; index < extensionMaps.length; index++) {
		const [prefix, texture, uv] = extensionMaps[index];
		const samplerName = `u${prefix}Map`;
		if (!isSamplerActive(samplerName)) continue;
		const sampler = pbrExtensionUniforms[samplerName];
		const hasMap = pbrExtensionUniforms[`uHas${prefix}Map`];
		const uvUniform = pbrExtensionUniforms[`u${prefix}MapUV`];
		const transformA = pbrExtensionUniforms[`u${prefix}MapTransformA`];
		const transformB = pbrExtensionUniforms[`u${prefix}MapTransformB`];
		const transform = resolveTextureUVTransform(texture);
		const unit = samplerUnit(samplerName);
		const resolved = host._textures.getBaseColorTexture(texture);
		gl.activeTexture(gl.TEXTURE0 + unit);
		gl.bindTexture(gl.TEXTURE_2D, resolved.texture);
		if (sampler) gl.uniform1i(sampler, unit);
		if (hasMap) gl.uniform1i(hasMap, texture ? 1 : 0);
		if (uvUniform) gl.uniform1i(uvUniform, uv);
		if (transformA) {
			gl.uniform4f(
				transformA,
				transform.repeatX,
				transform.repeatY,
				transform.offsetX,
				transform.offsetY,
			);
		}
		if (transformB) {
			gl.uniform2f(transformB, transform.cosRotation, transform.sinRotation);
		}
	}
	const canBindAnisotropyMap = !!uniforms.anisotropyMap;
	bindSceneTexture(
		"uBaseMap",
		resolvedMap.texture,
	);
	bindSceneTexture(
		"uNormalMap",
		resolvedNormalMap.texture,
	);
	bindSceneTexture(
		"uMetallicRoughnessMap",
		resolvedMetallicRoughnessMap.texture,
	);
	bindSceneTexture(
		"uSpecularMap",
		resolvedSpecularMap.texture,
	);
	bindSceneTexture(
		"uSpecularColorMap",
		resolvedSpecularColorMap.texture,
	);
	bindSceneTexture(
		"uEmissiveMap",
		resolvedEmissiveMap.texture,
	);
	bindSceneTexture(
		"uOcclusionMap",
		resolvedOcclusionMap.texture,
	);
	bindSceneTexture(
		"uIridescenceMap",
		resolvedIridescenceMap.texture,
	);
	bindSceneTexture(
		"uIridescenceThicknessMap",
		resolvedIridescenceThicknessMap.texture,
	);
	bindSceneTexture(
		"uAnisotropyMap",
		resolvedAnisotropyMap.texture,
	);
	host._bindShaderMaterialTextures(sceneProgram, material);
	host._bindShaderMaterialUniforms(sceneProgram, material);

	host._setCullMode(material);
	gl.bindVertexArray(geometry.vao);
	if (sceneProgram.uniforms.model) {
		gl.uniformMatrix4fv(
			sceneProgram.uniforms.model,
			false,
			Matrix4.toColumnMajorArray(packet.worldMatrix)
		);
	}
	if (sceneProgram.uniforms.normalMatrix) {
		gl.uniformMatrix3fv(sceneProgram.uniforms.normalMatrix, false, normalMatrix);
	}
	if (sceneProgram.uniforms.enableShadows) {
		gl.uniform1i(
			sceneProgram.uniforms.enableShadows,
			context.features.enableShadows &&
				host.getShadowSamplingState().enabled &&
				packet.primitive.receiveShadows !== false ? 1 : 0
		);
	}
	if (sceneProgram.uniforms.prevModel) {
		const cacheKey = packet.id;
		host._modelMatrixKeysThisFrame.add(cacheKey);
		let cached = host._modelMatrixCache.get(cacheKey);
		gl.uniformMatrix4fv(
			sceneProgram.uniforms.prevModel,
			false,
			cached ?? Matrix4.toColumnMajorArray(packet.worldMatrix)
		);
		if (!cached) {
			cached = Matrix4.toColumnMajorArray(packet.worldMatrix);
			host._modelMatrixCache.set(cacheKey, cached);
		} else {
			cached.set(Matrix4.toColumnMajorArray(packet.worldMatrix));
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
	if (sceneProgram.uniforms.specular) {
		gl.uniform4fv(sceneProgram.uniforms.specular, uniforms.specular);
	}
	const clearcoatUniform = pbrExtensionUniforms.uClearcoat;
	if (clearcoatUniform) gl.uniform4fv(clearcoatUniform, uniforms.clearcoat);
	const sheenUniform = pbrExtensionUniforms.uSheen;
	if (sheenUniform) gl.uniform4fv(sheenUniform, uniforms.sheen);
	if (sceneProgram.uniforms.specularMap) {
		gl.uniform1i(sceneProgram.uniforms.specularMap, samplerUnit("uSpecularMap"));
	}
	if (sceneProgram.uniforms.hasSpecularMap) {
		gl.uniform1i(sceneProgram.uniforms.hasSpecularMap, uniforms.specularMap ? 1 : 0);
	}
	if (sceneProgram.uniforms.specularMapUV) {
		gl.uniform1i(sceneProgram.uniforms.specularMapUV, uniforms.specularMapUV);
	}
	if (sceneProgram.uniforms.specularColorMap) {
		gl.uniform1i(
			sceneProgram.uniforms.specularColorMap,
			samplerUnit("uSpecularColorMap")
		);
	}
	if (sceneProgram.uniforms.hasSpecularColorMap) {
		gl.uniform1i(
			sceneProgram.uniforms.hasSpecularColorMap,
			uniforms.specularColorMap ? 1 : 0
		);
	}
	if (sceneProgram.uniforms.specularColorMapUV) {
		gl.uniform1i(
			sceneProgram.uniforms.specularColorMapUV,
			uniforms.specularColorMapUV
		);
	}
	for (const [prefix, transform] of [
		["uSpecularMapTransform", specularMapUVTransform],
		["uSpecularColorMapTransform", specularColorMapUVTransform],
	] as const) {
		const transformA = pbrExtensionUniforms[`${prefix}A`];
		const transformB = pbrExtensionUniforms[`${prefix}B`];
		if (transformA) {
			gl.uniform4f(
				transformA,
				transform.repeatX,
				transform.repeatY,
				transform.offsetX,
				transform.offsetY,
			);
		}
		if (transformB) {
			gl.uniform2f(transformB, transform.cosRotation, transform.sinRotation);
		}
	}
	if (sceneProgram.uniforms.transmissionVolume) {
		gl.uniform4fv(
			sceneProgram.uniforms.transmissionVolume,
			uniforms.transmissionVolume
		);
	}
	if (transmissionModelScale) {
		const elements = packet.worldMatrix.elements;
		const scaleX = Math.hypot(elements[0][0], elements[1][0], elements[2][0]);
		const scaleY = Math.hypot(elements[0][1], elements[1][1], elements[2][1]);
		const scaleZ = Math.hypot(elements[0][2], elements[1][2], elements[2][2]);
		gl.uniform1f(
			transmissionModelScale,
			Math.max(scaleX, scaleY, scaleZ, 0.0001),
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
	if (sceneProgram.uniforms.anisotropy) {
		gl.uniform4fv(sceneProgram.uniforms.anisotropy, uniforms.anisotropy);
	}
	if (sceneProgram.uniforms.phong) {
		gl.uniform4fv(sceneProgram.uniforms.phong, uniforms.phong);
	}
	if (sceneProgram.uniforms.phongAmbient) {
		gl.uniform4fv(sceneProgram.uniforms.phongAmbient, uniforms.phongAmbient);
	}
	if (sceneProgram.uniforms.alpha) {
		gl.uniform4fv(sceneProgram.uniforms.alpha, uniforms.alpha);
	}
	if (sceneProgram.uniforms.oitPassMode) {
		gl.uniform1i(sceneProgram.uniforms.oitPassMode, host._oitPassMode);
	}
	if (sceneProgram.uniforms.baseMap) {
		gl.uniform1i(sceneProgram.uniforms.baseMap, samplerUnit("uBaseMap"));
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
			samplerUnit("uMetallicRoughnessMap")
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
		gl.uniform1i(sceneProgram.uniforms.normalMap, samplerUnit("uNormalMap"));
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
		gl.uniform1i(sceneProgram.uniforms.emissiveMap, samplerUnit("uEmissiveMap"));
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
		gl.uniform1i(sceneProgram.uniforms.occlusionMap, samplerUnit("uOcclusionMap"));
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
			samplerUnit("uIridescenceMap")
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
			samplerUnit("uIridescenceThicknessMap")
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
	if (sceneProgram.uniforms.hasAnisotropyMap) {
		gl.uniform1i(
			sceneProgram.uniforms.hasAnisotropyMap,
			canBindAnisotropyMap ? 1 : 0
		);
	}
	if (sceneProgram.uniforms.anisotropyMap) {
		gl.uniform1i(
			sceneProgram.uniforms.anisotropyMap,
			samplerUnit("uAnisotropyMap"),
		);
	}
	if (sceneProgram.uniforms.anisotropyMapUV) {
		gl.uniform1i(
			sceneProgram.uniforms.anisotropyMapUV,
			uniforms.anisotropyMapUV
		);
	}
	if (sceneProgram.uniforms.anisotropyMapTransformA) {
		gl.uniform4f(
			sceneProgram.uniforms.anisotropyMapTransformA,
			anisotropyMapUVTransform.repeatX,
			anisotropyMapUVTransform.repeatY,
			anisotropyMapUVTransform.offsetX,
			anisotropyMapUVTransform.offsetY
		);
	}
	if (sceneProgram.uniforms.anisotropyMapTransformB) {
		gl.uniform2f(
			sceneProgram.uniforms.anisotropyMapTransformB,
			anisotropyMapUVTransform.cosRotation,
			anisotropyMapUVTransform.sinRotation
		);
	}
	if (sceneProgram.uniforms.doubleSided) {
		gl.uniform1i(
			sceneProgram.uniforms.doubleSided,
			material.doubleSided || material.cullMode === "none" ? 1 : 0
		);
	}

	const earlyZColor =
		options.earlyZColor === true && !transparentPass && material.depthWrite;
	gl.depthFunc(earlyZColor ? gl.LEQUAL : gl.LESS);
	gl.depthMask(earlyZColor ? false : !transparentPass && material.depthWrite);
	const colorOutputCount =
		sceneProgram.colorOutputCount ??
		(sceneProgram.targetMode === "single" ? 1 : 3);
	const compatibleDrawBuffers =
		activeDrawBuffers && activeDrawBuffers.length > colorOutputCount ?
			activeDrawBuffers.slice(0, colorOutputCount)
		:	null;
	if (compatibleDrawBuffers) gl.drawBuffers(compatibleDrawBuffers);
	try {
		gl.drawElements(
			geometry.topology,
			geometry.indexCount,
			geometry.indexType,
			0
		);
	} finally {
		if (compatibleDrawBuffers && activeDrawBuffers) {
			gl.drawBuffers(activeDrawBuffers);
		}
	}
	gl.bindVertexArray(null);
}

function resolveWebGLDepthPrepassProgram(
	host: WebGLScenePassHost,
	packet: DrawPacket,
	mode: ShaderTargetMode
): WebGLSceneProgram | null {
	const material = packet.material;
	if (isMaterialTransparentPass(material) || material.depthWrite === false) {
		return null;
	}
	if (packet.meshInstance.skeleton || !Matrix4.isFinite(packet.worldMatrix)) {
		return null;
	}
	const geometry = host._geometry.getGeometry(packet);
	if (!geometry || geometry.topology !== host._gl.TRIANGLES) {
		return null;
	}
	return host._scenePrograms.getSceneDepthPrepassProgram(
		material,
		mode,
		resolveSceneDepthPrepassVariant(host, packet)
	);
}

function resolveSceneProgramVariant(
	host: WebGLScenePassHost,
	context: FrameContext,
	packet: DrawPacket,
	mode: ShaderTargetMode
): WebGLSceneVariantDescriptor | null {
	return host._resolveSceneProgramVariant?.(context, packet, mode) ?? null;
}

function resolveSceneDepthPrepassVariant(
	host: WebGLScenePassHost,
	packet: DrawPacket
): WebGLSceneDepthVariantDescriptor | null {
	return host._resolveSceneDepthPrepassVariant?.(packet) ?? null;
}

export function drawWebGLDepthPrepassPacket(
	host: WebGLScenePassHost,
	sceneProgram: WebGLSceneProgram,
	packet: DrawPacket
): boolean {
	const gl = host._gl;
	const material = packet.material;
	if (isMaterialTransparentPass(material) || material.depthWrite === false) {
		return false;
	}
	if (packet.meshInstance.skeleton || !Matrix4.isFinite(packet.worldMatrix)) {
		return false;
	}
	const geometry = host._geometry.getGeometry(packet);
	if (!geometry || geometry.topology !== gl.TRIANGLES) {
		return false;
	}
	const normalMatrix = sceneProgram.uniforms.normalMatrix ?
		toColumnMajorMat3(packet.normalMatrix)
	:	null;
	if (sceneProgram.uniforms.normalMatrix && !normalMatrix) {
		logWebGLScenePassWarning(
			"webgl-depth-prepass-normal-matrix-invalid",
			`WebGL packet ${packet.id} has invalid normal matrix; skipping depth prepass`
		);
		return false;
	}

	const uniforms = resolveMaterialUniforms(material);
	const baseMapUVTransform = resolveTextureUVTransform(uniforms.baseMap);
	const resolvedMap = host._textures.getBaseColorTexture(uniforms.baseMap);
	const baseMapUnit = sceneProgram.uniforms.baseMap ?
		getWebGLSceneSamplerUnit(sceneProgram.samplerLayout, "uBaseMap")
	: 0;

	gl.activeTexture(gl.TEXTURE0 + baseMapUnit);
	gl.bindTexture(gl.TEXTURE_2D, resolvedMap.texture);
	host._bindShaderMaterialTextures(sceneProgram, material);
	host._bindShaderMaterialUniforms(sceneProgram, material);

	host._setCullMode(material);
	gl.bindVertexArray(geometry.vao);
	if (sceneProgram.uniforms.model) {
		gl.uniformMatrix4fv(
			sceneProgram.uniforms.model,
			false,
			Matrix4.toColumnMajorArray(packet.worldMatrix)
		);
	}
	if (sceneProgram.uniforms.normalMatrix && normalMatrix) {
		gl.uniformMatrix3fv(sceneProgram.uniforms.normalMatrix, false, normalMatrix);
	}
	if (sceneProgram.uniforms.baseColor) {
		gl.uniform4fv(sceneProgram.uniforms.baseColor, uniforms.baseColor);
	}
	if (sceneProgram.uniforms.alpha) {
		gl.uniform4fv(sceneProgram.uniforms.alpha, uniforms.alpha);
	}
	if (sceneProgram.uniforms.baseMap) {
		gl.uniform1i(sceneProgram.uniforms.baseMap, baseMapUnit);
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
	if (sceneProgram.uniforms.doubleSided) {
		gl.uniform1i(
			sceneProgram.uniforms.doubleSided,
			material.doubleSided || material.cullMode === "none" ? 1 : 0
		);
	}

	gl.depthFunc(gl.LESS);
	gl.depthMask(true);
	gl.drawElements(
		geometry.topology,
		geometry.indexCount,
		geometry.indexType,
		0
	);
	gl.bindVertexArray(null);
	return true;
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
		const textureUnit = getWebGLSceneSamplerUnit(
			sceneProgram.samplerLayout,
			binding.webglUniform,
		);
		const resolved = host._textures.getBaseColorTexture(binding.texture);
		gl.activeTexture(gl.TEXTURE0 + textureUnit);
		gl.bindTexture(gl.TEXTURE_2D, resolved.texture);
		gl.uniform1i(uniform, textureUnit);
	}
	gl.activeTexture(gl.TEXTURE0);
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
				Matrix4.toColumnMajorArray(value as number[][])
			);
			break;
	}
}
