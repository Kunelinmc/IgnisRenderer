import type { DrawPacket } from "../../pipeline/types";
import type { IPrimitive } from "../../core/types";
import { DEFAULT_PRIMITIVE_DRAW_TOPOLOGY } from "../../core/types";
import { Logger } from "../../foundation/Logger";

export interface WebGLGeometryHandle {
	vao: WebGLVertexArrayObject;
	vertexBuffer: WebGLBuffer;
	indexBuffer: WebGLBuffer;
	indexCount: number;
	indexType: number;
	topology: number;
}

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
const WEBGL_SCENE_VERTEX_FLOATS = 10;
const WEBGL_SCENE_VERTEX_STRIDE = WEBGL_SCENE_VERTEX_FLOATS * 4;

export class WebGLGeometryRegistry {
	private _gl: WebGL2RenderingContext;
	private _cache = new WeakMap<IPrimitive, WebGLCachedGeometryEntry>();
	private _owned = new Set<WebGLGeometryHandle>();
	private _warnCallback: WebGLGeometryWarn | null = null;

	constructor(gl: WebGL2RenderingContext, warn?: WebGLGeometryWarn) {
		this._gl = gl;
		this._warnCallback = warn ?? null;
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
		this._gl.deleteBuffer(handle.indexBuffer);
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

		const vertexCount = (positions.length / 3) | 0;
		const maxIndex = getMaxIndex(indices);
		if (!Number.isFinite(maxIndex) || maxIndex < 0 || maxIndex >= vertexCount) {
			const key = `webgl-geometry-index-range-${primitive.id}`;
			const message = `WebGL geometry ${primitiveLabel} index data exceeds vertex range; skipping`;
			this._warn(key, message);
			return { handle: null, cacheFailure: true };
		}

		if ((geometry.morphTargets?.length ?? 0) > 0) {
			const key = `webgl-geometry-morph-fallback-${primitive.id}`;
			const message = `WebGL backend does not support morph targets yet; rendering base geometry for primitive ${primitive.id}`;
			this._warn(key, message);
		}

		const interleaved = new Float32Array(vertexCount * WEBGL_SCENE_VERTEX_FLOATS);
		const normals = geometry.normals;
		const uv0 = geometry.uv0;
		const uv1 = geometry.uv1;
		for (let i = 0; i < vertexCount; i++) {
			const srcPos = i * 3;
			const srcUv = i * 2;
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
		}

		const gl = this._gl;
		const vao = gl.createVertexArray();
		const vertexBuffer = gl.createBuffer();
		const indexBuffer = gl.createBuffer();
		if (!vao || !vertexBuffer || !indexBuffer) {
			if (vao) gl.deleteVertexArray(vao);
			if (vertexBuffer) gl.deleteBuffer(vertexBuffer);
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

		return {
			handle: {
				vao,
				vertexBuffer,
				indexBuffer,
				indexCount: indexData.length,
				indexType,
				topology: mapTopology(gl, primitive.topology),
			},
			cacheFailure: false,
		};
	}

	private _warn(key: string, message: string): void {
		this._warnCallback?.(key, message);
		Logger.warn(`[${key}] ${message}`, {
			scope: "WebGLGeometryRegistry",
			onceKey: key,
		});
	}
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
