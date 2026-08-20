import type { DrawPacket } from "../../pipeline/types";
import type { IPrimitive } from "../../core/types";
import { DEFAULT_PRIMITIVE_DRAW_TOPOLOGY } from "../../core/types";
import { Logger } from "../../foundation/Logger";

export interface WebGLGeometryHandle {
	vao: WebGLVertexArrayObject;
	vertexBuffer: WebGLBuffer;
	skinBuffer: WebGLBuffer | null;
	indexBuffer: WebGLBuffer;
	indexCount: number;
	indexType: number;
	topology: number;
	vertexCount: number;
	skinProfile: WebGLSkinProfile;
	morphTargetCount: number;
	morphSemanticMask: number;
	morphPositionTexture: WebGLTexture | null;
	morphNormalTexture: WebGLTexture | null;
	morphTextureWidth: number;
}

export type WebGLSkinProfile = "static" | "skin4" | "skin8";

export const WEBGL_MAX_MORPH_TARGETS = 8;
export const WEBGL_MORPH_POSITION_BIT = 1 << 0;
export const WEBGL_MORPH_NORMAL_BIT = 1 << 1;

interface UploadPrimitiveResult {
	handle: WebGLGeometryHandle | null;
	cacheFailure: boolean;
}

interface WebGLCachedGeometryEntry {
	handle: WebGLGeometryHandle | null;
	geometryVersion: number;
	cacheFailure: boolean;
}

type WebGLGeometryWarn = (key: string, message: string) => void;
const WEBGL_SCENE_VERTEX_FLOATS = 18;
const WEBGL_SCENE_VERTEX_STRIDE = WEBGL_SCENE_VERTEX_FLOATS * 4;

export class WebGLGeometryRegistry {
	private _gl: WebGL2RenderingContext;
	private _cache = new WeakMap<IPrimitive, WebGLCachedGeometryEntry>();
	private _owned = new Set<WebGLGeometryHandle>();
	private _warnCallback: WebGLGeometryWarn | null = null;
	private readonly _maxTextureSize: number;

	constructor(gl: WebGL2RenderingContext, warn?: WebGLGeometryWarn) {
		this._gl = gl;
		this._warnCallback = warn ?? null;
		const maxTextureSize = gl.getParameter?.(gl.MAX_TEXTURE_SIZE);
		this._maxTextureSize = Number.isFinite(maxTextureSize) ?
			Math.max(1, Math.floor(maxTextureSize)) : 4096;
	}

	public getGeometry(packet: DrawPacket): WebGLGeometryHandle | null {
		const primitive = packet.primitive;
		const geometryVersion = primitive.geometryVersion ?? 0;
		const cached = this._cache.get(primitive);
		if (cached && cached.geometryVersion === geometryVersion) {
			return cached.handle;
		}
		if (cached?.handle) {
			this._owned.delete(cached.handle);
			this._destroyHandle(cached.handle);
		}
		const uploaded = this._uploadPrimitive(packet);
		if (uploaded.handle) {
			this._owned.add(uploaded.handle);
		}
		if (uploaded.handle || uploaded.cacheFailure) {
			this._cache.set(primitive, {
				handle: uploaded.handle,
				geometryVersion,
				cacheFailure: uploaded.cacheFailure,
			});
		}
		return uploaded.handle;
	}

	public destroy(): void {
		for (const handle of this._owned) {
			this._destroyHandle(handle);
		}
		this._owned.clear();
	}

	private _destroyHandle(handle: WebGLGeometryHandle): void {
		this._gl.deleteBuffer(handle.vertexBuffer);
		if (handle.skinBuffer) this._gl.deleteBuffer(handle.skinBuffer);
		this._gl.deleteBuffer(handle.indexBuffer);
		if (handle.morphPositionTexture) {
			this._gl.deleteTexture(handle.morphPositionTexture);
		}
		if (handle.morphNormalTexture) {
			this._gl.deleteTexture(handle.morphNormalTexture);
		}
		this._gl.deleteVertexArray(handle.vao);
	}

	private _uploadPrimitive(packet: DrawPacket): UploadPrimitiveResult {
		const primitive = packet.primitive;
		const geometry = primitive.geometry;
		const positions = geometry.positions;
		const indices = geometry.indices;
		const primitiveLabel = `${primitive.id}:${packet.id}`;

		if (!positions || positions.length < 3 || positions.length % 3 !== 0) {
			const key = `webgl-geometry-invalid-positions-${primitive.id}`;
			const message = `WebGL geometry ${primitiveLabel} has invalid position data; skipping`;
			this._warn(key, message);
			return { handle: null, cacheFailure: true };
		}
		if (!indices || indices.length <= 0) {
			const key = `webgl-geometry-empty-indices-${primitive.id}`;
			const message = `WebGL geometry ${primitiveLabel} has no indices; skipping`;
			this._warn(key, message);
			return { handle: null, cacheFailure: true };
		}
		if (!isFiniteArray(positions)) {
			const key = `webgl-geometry-nonfinite-positions-${primitive.id}`;
			const message = `WebGL geometry ${primitiveLabel} contains non-finite position values; skipping`;
			this._warn(key, message);
			return { handle: null, cacheFailure: true };
		}
		if (geometry.normals && !isFiniteArray(geometry.normals)) {
			const key = `webgl-geometry-nonfinite-normals-${primitive.id}`;
			const message = `WebGL geometry ${primitiveLabel} contains non-finite normal values; skipping`;
			this._warn(key, message);
			return { handle: null, cacheFailure: true };
		}
		if (geometry.uv0 && !isFiniteArray(geometry.uv0)) {
			const key = `webgl-geometry-nonfinite-uv-${primitive.id}`;
			const message = `WebGL geometry ${primitiveLabel} contains non-finite UV values; skipping`;
			this._warn(key, message);
			return { handle: null, cacheFailure: true };
		}
		if (geometry.uv1 && !isFiniteArray(geometry.uv1)) {
			const key = `webgl-geometry-nonfinite-uv1-${primitive.id}`;
			const message = `WebGL geometry ${primitiveLabel} contains non-finite UV1 values; skipping`;
			this._warn(key, message);
			return { handle: null, cacheFailure: true };
		}
		if (geometry.uv2 && !isFiniteArray(geometry.uv2)) {
			const key = `webgl-geometry-nonfinite-uv2-${primitive.id}`;
			const message = `WebGL geometry ${primitiveLabel} contains non-finite UV2 values; skipping`;
			this._warn(key, message);
			return { handle: null, cacheFailure: true };
		}
		if (geometry.uv3 && !isFiniteArray(geometry.uv3)) {
			const key = `webgl-geometry-nonfinite-uv3-${primitive.id}`;
			const message = `WebGL geometry ${primitiveLabel} contains non-finite UV3 values; skipping`;
			this._warn(key, message);
			return { handle: null, cacheFailure: true };
		}
		if (geometry.tangents && !isFiniteArray(geometry.tangents)) {
			const key = `webgl-geometry-nonfinite-tangents-${primitive.id}`;
			const message = `WebGL geometry ${primitiveLabel} contains non-finite tangent values; skipping`;
			this._warn(key, message);
			return { handle: null, cacheFailure: true };
		}

		const vertexCount = (positions.length / 3) | 0;
		const skinProfile = resolveWebGLSkinProfile(geometry);
		const morphTargetCount = Math.min(
			WEBGL_MAX_MORPH_TARGETS,
			geometry.morphTargets?.length ?? 0,
		);
		const morphTexelCount = morphTargetCount * vertexCount;
		if (morphTexelCount > this._maxTextureSize * this._maxTextureSize) {
			const key = `webgl-morph-texture-overflow-${primitive.id}`;
			const message =
				`WebGL morph payload for primitive ${primitive.id} requires ` +
				`${morphTexelCount} texels, exceeding device capacity; skipping`;
			this._warn(key, message);
			return { handle: null, cacheFailure: true };
		}
		const maxIndex = getMaxIndex(indices);
		if (!Number.isFinite(maxIndex) || maxIndex < 0 || maxIndex >= vertexCount) {
			const key = `webgl-geometry-index-range-${primitive.id}`;
			const message = `WebGL geometry ${primitiveLabel} index data exceeds vertex range; skipping`;
			this._warn(key, message);
			return { handle: null, cacheFailure: true };
		}

		const interleaved = new Float32Array(vertexCount * WEBGL_SCENE_VERTEX_FLOATS);
		const normals = geometry.normals;
		const uv0 = geometry.uv0;
		const uv1 = geometry.uv1;
		const uv2 = geometry.uv2;
		const uv3 = geometry.uv3;
		const tangents = geometry.tangents;
		for (let i = 0; i < vertexCount; i++) {
			const srcPos = i * 3;
			const srcUv = i * 2;
			const srcTangent = i * 4;
			const dst = i * WEBGL_SCENE_VERTEX_FLOATS;
			interleaved[dst] = positions[srcPos];
			interleaved[dst + 1] = positions[srcPos + 1];
			interleaved[dst + 2] = positions[srcPos + 2];
			interleaved[dst + 3] = normals?.[srcPos] ?? 0;
			interleaved[dst + 4] = normals?.[srcPos + 1] ?? 0;
			interleaved[dst + 5] = normals?.[srcPos + 2] ?? 1;
			interleaved[dst + 6] = uv0?.[srcUv] ?? 0;
			interleaved[dst + 7] = uv0?.[srcUv + 1] ?? 0;
			interleaved[dst + 8] = uv1?.[srcUv] ?? interleaved[dst + 6];
			interleaved[dst + 9] = uv1?.[srcUv + 1] ?? interleaved[dst + 7];
			interleaved[dst + 10] = uv2?.[srcUv] ?? interleaved[dst + 6];
			interleaved[dst + 11] = uv2?.[srcUv + 1] ?? interleaved[dst + 7];
			interleaved[dst + 12] = uv3?.[srcUv] ?? interleaved[dst + 6];
			interleaved[dst + 13] = uv3?.[srcUv + 1] ?? interleaved[dst + 7];
			interleaved[dst + 14] = tangents?.[srcTangent] ?? 0;
			interleaved[dst + 15] = tangents?.[srcTangent + 1] ?? 0;
			interleaved[dst + 16] = tangents?.[srcTangent + 2] ?? 0;
			interleaved[dst + 17] = tangents?.[srcTangent + 3] ?? 0;
		}

		const gl = this._gl;
		const vao = gl.createVertexArray();
		const vertexBuffer = gl.createBuffer();
		const skinBuffer = skinProfile === "static" ? null : gl.createBuffer();
		const indexBuffer = gl.createBuffer();
		if (!vao || !vertexBuffer || !indexBuffer || (skinProfile !== "static" && !skinBuffer)) {
			if (vao) gl.deleteVertexArray(vao);
			if (vertexBuffer) gl.deleteBuffer(vertexBuffer);
			if (skinBuffer) gl.deleteBuffer(skinBuffer);
			if (indexBuffer) gl.deleteBuffer(indexBuffer);
			const key = `webgl-geometry-upload-failed-${primitive.id}`;
			const message = `Failed to allocate WebGL buffers for primitive ${primitive.id}; skipping`;
			this._warn(key, message);
			return { handle: null, cacheFailure: false };
		}

		gl.bindVertexArray(vao);
		gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
		gl.bufferData(gl.ARRAY_BUFFER, interleaved, gl.STATIC_DRAW);
		gl.enableVertexAttribArray(0);
		gl.vertexAttribPointer(0, 3, gl.FLOAT, false, WEBGL_SCENE_VERTEX_STRIDE, 0);
		gl.enableVertexAttribArray(1);
		gl.vertexAttribPointer(1, 3, gl.FLOAT, false, WEBGL_SCENE_VERTEX_STRIDE, 12);
		gl.enableVertexAttribArray(2);
		gl.vertexAttribPointer(2, 2, gl.FLOAT, false, WEBGL_SCENE_VERTEX_STRIDE, 24);
		gl.enableVertexAttribArray(3);
		gl.vertexAttribPointer(3, 2, gl.FLOAT, false, WEBGL_SCENE_VERTEX_STRIDE, 32);
		gl.enableVertexAttribArray(4);
		gl.vertexAttribPointer(4, 2, gl.FLOAT, false, WEBGL_SCENE_VERTEX_STRIDE, 40);
		gl.enableVertexAttribArray(5);
		gl.vertexAttribPointer(5, 2, gl.FLOAT, false, WEBGL_SCENE_VERTEX_STRIDE, 48);
		gl.enableVertexAttribArray(6);
		gl.vertexAttribPointer(6, 4, gl.FLOAT, false, WEBGL_SCENE_VERTEX_STRIDE, 56);
		if (skinBuffer) {
			const skinData = packWebGLSkinData(geometry, vertexCount, skinProfile);
			const skinStride = skinProfile === "skin8" ? 64 : 32;
			gl.bindBuffer(gl.ARRAY_BUFFER, skinBuffer);
			gl.bufferData(gl.ARRAY_BUFFER, skinData, gl.STATIC_DRAW);
			gl.enableVertexAttribArray(7);
			gl.vertexAttribPointer(7, 4, gl.FLOAT, false, skinStride, 0);
			gl.enableVertexAttribArray(8);
			gl.vertexAttribPointer(8, 4, gl.FLOAT, false, skinStride, 16);
			if (skinProfile === "skin8") {
				gl.enableVertexAttribArray(9);
				gl.vertexAttribPointer(9, 4, gl.FLOAT, false, skinStride, 32);
				gl.enableVertexAttribArray(10);
				gl.vertexAttribPointer(10, 4, gl.FLOAT, false, skinStride, 48);
			}
		}

		gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
		let indexType: number = gl.UNSIGNED_INT;
		let indexData: Uint32Array | Uint16Array;
		if (maxIndex <= 65535) {
			indexType = gl.UNSIGNED_SHORT;
			indexData = new Uint16Array(indices.length);
			for (let i = 0; i < indices.length; i++) {
				indexData[i] = indices[i];
			}
		} else {
			indexData = new Uint32Array(indices);
		}
		gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indexData, gl.STATIC_DRAW);

		gl.bindVertexArray(null);
		gl.bindBuffer(gl.ARRAY_BUFFER, null);
		gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
		const morph = this._uploadMorphTargets(geometry, vertexCount, morphTargetCount);
		if (!morph) {
			gl.deleteVertexArray(vao);
			gl.deleteBuffer(vertexBuffer);
			if (skinBuffer) gl.deleteBuffer(skinBuffer);
			gl.deleteBuffer(indexBuffer);
			this._warn(
				`webgl-morph-texture-allocation-failed-${primitive.id}`,
				`Failed to allocate WebGL morph textures for primitive ${primitive.id}; skipping`,
			);
			return { handle: null, cacheFailure: false };
		}

		return {
			handle: {
				vao,
				vertexBuffer,
				skinBuffer,
				indexBuffer,
				indexCount: indexData.length,
				indexType,
				topology: mapTopology(gl, primitive.topology),
				vertexCount,
				skinProfile,
				morphTargetCount,
				morphSemanticMask: morph.semanticMask,
				morphPositionTexture: morph.positionTexture,
				morphNormalTexture: morph.normalTexture,
				morphTextureWidth: morph.width,
			},
			cacheFailure: false,
		};
	}

	private _uploadMorphTargets(
		geometry: IPrimitive["geometry"],
		vertexCount: number,
		targetCount: number,
	): {
		semanticMask: number;
		positionTexture: WebGLTexture | null;
		normalTexture: WebGLTexture | null;
		width: number;
	} | null {
		if (targetCount <= 0) {
			return { semanticMask: 0, positionTexture: null, normalTexture: null, width: 1 };
		}
		const targets = geometry.morphTargets ?? [];
		const hasPositions = targets.slice(0, targetCount).some((target) => !!target.positions);
		const hasNormals = targets.slice(0, targetCount).some((target) => !!target.normals);
		const texelCount = targetCount * vertexCount;
		const width = Math.min(this._maxTextureSize, nextPowerOfTwo(Math.max(1, texelCount)));
		const height = Math.ceil(texelCount / width);
		let positionTexture: WebGLTexture | null = null;
		let normalTexture: WebGLTexture | null = null;
		try {
			if (hasPositions) {
				positionTexture = this._createMorphTexture(
					targets,
					targetCount,
					vertexCount,
					"positions",
					width,
					height,
				);
			}
			if (hasNormals) {
				normalTexture = this._createMorphTexture(
					targets,
					targetCount,
					vertexCount,
					"normals",
					width,
					height,
				);
			}
		} catch {
			if (positionTexture) this._gl.deleteTexture(positionTexture);
			if (normalTexture) this._gl.deleteTexture(normalTexture);
			return null;
		}
		return {
			semanticMask:
				(hasPositions ? WEBGL_MORPH_POSITION_BIT : 0) |
				(hasNormals ? WEBGL_MORPH_NORMAL_BIT : 0),
			positionTexture,
			normalTexture,
			width,
		};
	}

	private _createMorphTexture(
		targets: NonNullable<IPrimitive["geometry"]["morphTargets"]>,
		targetCount: number,
		vertexCount: number,
		semantic: "positions" | "normals",
		width: number,
		height: number,
	): WebGLTexture {
		const gl = this._gl;
		const texture = gl.createTexture();
		if (!texture) throw new Error("Unable to allocate morph texture");
		const data = new Float32Array(width * height * 4);
		for (let targetIndex = 0; targetIndex < targetCount; targetIndex++) {
			const source = targets[targetIndex]?.[semantic];
			if (!source) continue;
			for (let vertexIndex = 0; vertexIndex < vertexCount; vertexIndex++) {
				const sourceOffset = vertexIndex * 3;
				const targetOffset = (targetIndex * vertexCount + vertexIndex) * 4;
				data[targetOffset] = source[sourceOffset] ?? 0;
				data[targetOffset + 1] = source[sourceOffset + 1] ?? 0;
				data[targetOffset + 2] = source[sourceOffset + 2] ?? 0;
			}
		}
		gl.bindTexture(gl.TEXTURE_2D, texture);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
		gl.texImage2D(
			gl.TEXTURE_2D,
			0,
			gl.RGBA32F,
			width,
			height,
			0,
			gl.RGBA,
			gl.FLOAT,
			data,
		);
		gl.bindTexture(gl.TEXTURE_2D, null);
		return texture;
	}

	private _warn(key: string, message: string): void {
		this._warnCallback?.(key, message);
		Logger.warn(`[${key}] ${message}`, {
			scope: "WebGLGeometryRegistry",
			onceKey: key,
		});
	}
}

export function resolveWebGLSkinProfile(
	geometry: IPrimitive["geometry"],
): WebGLSkinProfile {
	if (geometry.joints1 || geometry.weights1) return "skin8";
	if (geometry.joints0 || geometry.weights0) return "skin4";
	return "static";
}

function packWebGLSkinData(
	geometry: IPrimitive["geometry"],
	vertexCount: number,
	profile: WebGLSkinProfile,
): Float32Array {
	const floatsPerVertex = profile === "skin8" ? 16 : 8;
	const data = new Float32Array(vertexCount * floatsPerVertex);
	for (let vertexIndex = 0; vertexIndex < vertexCount; vertexIndex++) {
		const source = vertexIndex * 4;
		const target = vertexIndex * floatsPerVertex;
		for (let component = 0; component < 4; component++) {
			data[target + component] = geometry.joints0?.[source + component] ?? 0;
			data[target + 4 + component] = geometry.weights0?.[source + component] ?? 0;
			if (profile === "skin8") {
				data[target + 8 + component] = geometry.joints1?.[source + component] ?? 0;
				data[target + 12 + component] = geometry.weights1?.[source + component] ?? 0;
			}
		}
	}
	return data;
}

function nextPowerOfTwo(value: number): number {
	let result = 1;
	while (result < value) result *= 2;
	return result;
}

function mapTopology(gl: WebGL2RenderingContext, topology?: string): number {
	switch (topology ?? DEFAULT_PRIMITIVE_DRAW_TOPOLOGY) {
		case "triangle-list":
			return gl.TRIANGLES;
		case "line-list":
			return gl.LINES;
		case "point-list":
			return gl.POINTS;
		default:
			return gl.TRIANGLES;
	}
}

function getMaxIndex(indices: Uint32Array): number {
	let maxIndex = -1;
	for (let i = 0; i < indices.length; i++) {
		const index = indices[i];
		if (index > maxIndex) {
			maxIndex = index;
		}
	}
	return maxIndex;
}

function isFiniteArray(data: ArrayLike<number>): boolean {
	for (let i = 0; i < data.length; i++) {
		if (!Number.isFinite(data[i])) {
			return false;
		}
	}
	return true;
}
