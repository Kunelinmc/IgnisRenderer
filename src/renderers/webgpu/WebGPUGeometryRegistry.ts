import type { IPrimitive } from "../../core/types";
import { BufferUsage, type IRenderBuffer } from "../types";
import type { WebGPUBackend } from "../WebGPUBackend";
import { GeometryBuilder } from "../../models/GeometryBuilder";

export interface WebGPUGeometryHandle {
	vertexBuffer: IRenderBuffer;
	indexBuffer: IRenderBuffer;
	indexCount: number;
	wireframeIndexBuffer: IRenderBuffer;
	wireframeIndexCount: number;
}

export class WebGPUGeometryRegistry {
	private _backend: WebGPUBackend;
	private _cache = new WeakMap<IPrimitive, WebGPUGeometryHandle>();

	constructor(backend: WebGPUBackend) {
		this._backend = backend;
	}

	public getGeometry(primitive: IPrimitive): WebGPUGeometryHandle {
		let cached = this._cache.get(primitive);
		if (!cached) {
			cached = this._uploadGeometry(primitive);
			this._cache.set(primitive, cached);
		}
		return cached;
	}

	private _uploadGeometry(primitive: IPrimitive): WebGPUGeometryHandle {
		const geometry = primitive.geometry;
		const vertexCount = GeometryBuilder.getVertexCount(geometry);
		const vertexData = new Float32Array(vertexCount * 14);

		for (let vertexIndex = 0; vertexIndex < vertexCount; vertexIndex++) {
			const sourcePosition = vertexIndex * 3;
			const sourceUv = vertexIndex * 2;
			const sourceTangent = vertexIndex * 4;
			const base = vertexIndex * 14;

			vertexData[base] = geometry.positions[sourcePosition];
			vertexData[base + 1] = geometry.positions[sourcePosition + 1];
			vertexData[base + 2] = geometry.positions[sourcePosition + 2];

			vertexData[base + 3] = geometry.normals?.[sourcePosition] ?? 0;
			vertexData[base + 4] = geometry.normals?.[sourcePosition + 1] ?? 0;
			vertexData[base + 5] = geometry.normals?.[sourcePosition + 2] ?? 0;

			vertexData[base + 6] = geometry.uv0?.[sourceUv] ?? 0;
			vertexData[base + 7] = geometry.uv0?.[sourceUv + 1] ?? 0;

			vertexData[base + 8] = geometry.tangents?.[sourceTangent] ?? 0;
			vertexData[base + 9] = geometry.tangents?.[sourceTangent + 1] ?? 0;
			vertexData[base + 10] = geometry.tangents?.[sourceTangent + 2] ?? 0;
			vertexData[base + 11] = geometry.tangents?.[sourceTangent + 3] ?? 0;

			vertexData[base + 12] = geometry.uv1?.[sourceUv] ?? 0;
			vertexData[base + 13] = geometry.uv1?.[sourceUv + 1] ?? 0;
		}

		const indexCount = geometry.indices.length;
		const triangleCount = Math.floor(indexCount / 3);
		const wireframeIndices = new Uint32Array(triangleCount * 6);
		let wIdx = 0;
		for (let i = 0; i < indexCount; i += 3) {
			const i0 = geometry.indices[i];
			const i1 = geometry.indices[i + 1];
			const i2 = geometry.indices[i + 2];
			wireframeIndices[wIdx++] = i0;
			wireframeIndices[wIdx++] = i1;
			wireframeIndices[wIdx++] = i1;
			wireframeIndices[wIdx++] = i2;
			wireframeIndices[wIdx++] = i2;
			wireframeIndices[wIdx++] = i0;
		}

		const vertexBuffer = this._backend.createBuffer({
			size: vertexData.byteLength,
			usage: BufferUsage.Vertex | BufferUsage.CopyDst,
			label: `VertexBuffer_${primitive.id}`,
		});
		const indexBuffer = this._backend.createBuffer({
			size: geometry.indices.byteLength,
			usage: BufferUsage.Index | BufferUsage.CopyDst,
			label: `IndexBuffer_${primitive.id}`,
		});
		const wireframeIndexBuffer = this._backend.createBuffer({
			size: wireframeIndices.byteLength,
			usage: BufferUsage.Index | BufferUsage.CopyDst,
			label: `WireframeIndexBuffer_${primitive.id}`,
		});

		this._backend.writeBuffer(vertexBuffer, new Float32Array(vertexData));
		this._backend.writeBuffer(indexBuffer, new Uint32Array(geometry.indices));
		this._backend.writeBuffer(wireframeIndexBuffer, wireframeIndices);

		return {
			vertexBuffer,
			indexBuffer,
			indexCount,
			wireframeIndexBuffer,
			wireframeIndexCount: wireframeIndices.length,
		};
	}
}
