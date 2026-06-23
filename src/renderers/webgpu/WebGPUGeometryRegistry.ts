import {
	DEFAULT_PRIMITIVE_DRAW_TOPOLOGY,
	type IPrimitive,
	type PrimitiveDrawTopology,
} from "../../core/types";
import { BufferUsage, type IRenderBuffer } from "../types";
import type { WebGPUBackend } from "../WebGPUBackend";
import { GeometryBuilder } from "../../meshes/GeometryBuilder";
import {
	WEBGPU_SCENE_VERTEX_FLOAT_OFFSET,
	WEBGPU_SCENE_VERTEX_FLOATS,
} from "./constants";

export interface WebGPUGeometryHandle {
	vertexBuffer: IRenderBuffer;
	indexBuffer: IRenderBuffer;
	indexCount: number;
	topology: PrimitiveDrawTopology;
	wireframeIndexBuffer: IRenderBuffer;
	wireframeIndexCount: number;
	vertexCount: number;
	morphTargetCount: number;
	morphPositionBuffer: IRenderBuffer | null;
	morphNormalBuffer: IRenderBuffer | null;
}

interface WebGPUCachedGeometryEntry {
	handle: WebGPUGeometryHandle;
	geometryVersion: number;
}

export class WebGPUGeometryRegistry {
	private _backend: WebGPUBackend;
	private _cache = new WeakMap<IPrimitive, WebGPUCachedGeometryEntry>();
	private _owned = new Set<WebGPUGeometryHandle>();
	private _finalizationRegistry: FinalizationRegistry<WebGPUGeometryHandle> | null =
		typeof FinalizationRegistry === "function" ?
			new FinalizationRegistry((handle) => {
				this._owned.delete(handle);
				this._destroyHandle(handle);
			})
		:	null;

	constructor(backend: WebGPUBackend) {
		this._backend = backend;
	}

	public getGeometry(primitive: IPrimitive): WebGPUGeometryHandle {
		const geometryVersion = primitive.geometryVersion ?? 0;
		let cached = this._cache.get(primitive);
		if (!cached || cached.geometryVersion !== geometryVersion) {
			if (cached) {
				this._finalizationRegistry?.unregister(primitive as unknown as object);
				this._owned.delete(cached.handle);
				this._destroyHandle(cached.handle);
			}
			const handle = this._uploadGeometry(primitive);
			cached = {
				handle,
				geometryVersion,
			};
			this._cache.set(primitive, cached);
			this._owned.add(handle);
			this._finalizationRegistry?.register(
				primitive as unknown as object,
				handle,
				primitive as unknown as object
			);
		}
		return cached.handle;
	}

	public releaseGeometry(primitive: IPrimitive): void {
		const cached = this._cache.get(primitive);
		if (!cached) {
			return;
		}

		this._cache.delete(primitive);
		this._finalizationRegistry?.unregister(primitive as unknown as object);
		this._owned.delete(cached.handle);
		this._destroyHandle(cached.handle);
	}

	public destroy(): void {
		for (const handle of this._owned) {
			this._destroyHandle(handle);
		}
		this._owned.clear();
		this._cache = new WeakMap<IPrimitive, WebGPUCachedGeometryEntry>();
		this._finalizationRegistry = null;
	}

	private _destroyHandle(handle: WebGPUGeometryHandle): void {
		handle.vertexBuffer.destroy();
		handle.indexBuffer.destroy();
		handle.wireframeIndexBuffer.destroy();
		handle.morphPositionBuffer?.destroy();
		handle.morphNormalBuffer?.destroy();
	}

	private _uploadGeometry(primitive: IPrimitive): WebGPUGeometryHandle {
		const geometry = primitive.geometry;
		const topology = primitive.topology ?? DEFAULT_PRIMITIVE_DRAW_TOPOLOGY;
		const vertexCount = GeometryBuilder.getVertexCount(geometry);
		const vertexData = new Float32Array(
			vertexCount * WEBGPU_SCENE_VERTEX_FLOATS
		);
		const vertexOffset = WEBGPU_SCENE_VERTEX_FLOAT_OFFSET;

		for (let vertexIndex = 0; vertexIndex < vertexCount; vertexIndex++) {
			const sourcePosition = vertexIndex * 3;
			const sourceUv = vertexIndex * 2;
			const sourceTangent = vertexIndex * 4;
			const sourceJoint = vertexIndex * 4;
			const base = vertexIndex * WEBGPU_SCENE_VERTEX_FLOATS;

			vertexData[base + vertexOffset.position] =
				geometry.positions[sourcePosition];
			vertexData[base + vertexOffset.position + 1] =
				geometry.positions[sourcePosition + 1];
			vertexData[base + vertexOffset.position + 2] =
				geometry.positions[sourcePosition + 2];

			vertexData[base + vertexOffset.normal] =
				geometry.normals?.[sourcePosition] ?? 0;
			vertexData[base + vertexOffset.normal + 1] =
				geometry.normals?.[sourcePosition + 1] ?? 0;
			vertexData[base + vertexOffset.normal + 2] =
				geometry.normals?.[sourcePosition + 2] ?? 0;

			vertexData[base + vertexOffset.uv0] = geometry.uv0?.[sourceUv] ?? 0;
			vertexData[base + vertexOffset.uv0 + 1] =
				geometry.uv0?.[sourceUv + 1] ?? 0;

			vertexData[base + vertexOffset.tangent] =
				geometry.tangents?.[sourceTangent] ?? 0;
			vertexData[base + vertexOffset.tangent + 1] =
				geometry.tangents?.[sourceTangent + 1] ?? 0;
			vertexData[base + vertexOffset.tangent + 2] =
				geometry.tangents?.[sourceTangent + 2] ?? 0;
			vertexData[base + vertexOffset.tangent + 3] =
				geometry.tangents?.[sourceTangent + 3] ?? 0;

			vertexData[base + vertexOffset.uv1] = geometry.uv1?.[sourceUv] ?? 0;
			vertexData[base + vertexOffset.uv1 + 1] =
				geometry.uv1?.[sourceUv + 1] ?? 0;

			vertexData[base + vertexOffset.joints0] = Number(
				geometry.joints0?.[sourceJoint] ?? 0
			);
			vertexData[base + vertexOffset.joints0 + 1] = Number(
				geometry.joints0?.[sourceJoint + 1] ?? 0
			);
			vertexData[base + vertexOffset.joints0 + 2] = Number(
				geometry.joints0?.[sourceJoint + 2] ?? 0
			);
			vertexData[base + vertexOffset.joints0 + 3] = Number(
				geometry.joints0?.[sourceJoint + 3] ?? 0
			);

			vertexData[base + vertexOffset.weights0] =
				geometry.weights0?.[sourceJoint] ?? 0;
			vertexData[base + vertexOffset.weights0 + 1] =
				geometry.weights0?.[sourceJoint + 1] ?? 0;
			vertexData[base + vertexOffset.weights0 + 2] =
				geometry.weights0?.[sourceJoint + 2] ?? 0;
			vertexData[base + vertexOffset.weights0 + 3] =
				geometry.weights0?.[sourceJoint + 3] ?? 0;

			vertexData[base + vertexOffset.joints1] = Number(
				geometry.joints1?.[sourceJoint] ?? 0
			);
			vertexData[base + vertexOffset.joints1 + 1] = Number(
				geometry.joints1?.[sourceJoint + 1] ?? 0
			);
			vertexData[base + vertexOffset.joints1 + 2] = Number(
				geometry.joints1?.[sourceJoint + 2] ?? 0
			);
			vertexData[base + vertexOffset.joints1 + 3] = Number(
				geometry.joints1?.[sourceJoint + 3] ?? 0
			);

			vertexData[base + vertexOffset.weights1] =
				geometry.weights1?.[sourceJoint] ?? 0;
			vertexData[base + vertexOffset.weights1 + 1] =
				geometry.weights1?.[sourceJoint + 1] ?? 0;
			vertexData[base + vertexOffset.weights1 + 2] =
				geometry.weights1?.[sourceJoint + 2] ?? 0;
			vertexData[base + vertexOffset.weights1 + 3] =
				geometry.weights1?.[sourceJoint + 3] ?? 0;

			vertexData[base + vertexOffset.uv2] = geometry.uv2?.[sourceUv] ?? 0;
			vertexData[base + vertexOffset.uv2 + 1] =
				geometry.uv2?.[sourceUv + 1] ?? 0;
			vertexData[base + vertexOffset.uv3] = geometry.uv3?.[sourceUv] ?? 0;
			vertexData[base + vertexOffset.uv3 + 1] =
				geometry.uv3?.[sourceUv + 1] ?? 0;
		}

		const indexCount = geometry.indices.length;
		const wireframeIndices =
			topology === DEFAULT_PRIMITIVE_DRAW_TOPOLOGY ?
				createTriangleWireframeIndices(geometry.indices)
			:	new Uint32Array(0);

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
			size: Math.max(4, wireframeIndices.byteLength),
			usage: BufferUsage.Index | BufferUsage.CopyDst,
			label: `WireframeIndexBuffer_${primitive.id}`,
		});

		this._backend.writeBuffer(vertexBuffer, vertexData as any);
		this._backend.writeBuffer(indexBuffer, geometry.indices as any);
		this._backend.writeBuffer(
			wireframeIndexBuffer,
			(wireframeIndices.length > 0 ? wireframeIndices : new Uint32Array([0])) as any
		);

		const morphTargets = geometry.morphTargets ?? [];
		const morphTargetCount = Math.min(8, morphTargets.length);
		let morphPositionBuffer: IRenderBuffer | null = null;
		let morphNormalBuffer: IRenderBuffer | null = null;
		if (morphTargetCount > 0) {
			const morphPositionData = new Float32Array(
				morphTargetCount * vertexCount * 4
			);
			const morphNormalData = new Float32Array(
				morphTargetCount * vertexCount * 4
			);

			for (let targetIndex = 0; targetIndex < morphTargetCount; targetIndex++) {
				const target = morphTargets[targetIndex];
				for (let vertexIndex = 0; vertexIndex < vertexCount; vertexIndex++) {
					const offset = (targetIndex * vertexCount + vertexIndex) * 4;
					const positionOffset = vertexIndex * 3;
					morphPositionData[offset] = target.positions?.[positionOffset] ?? 0;
					morphPositionData[offset + 1] =
						target.positions?.[positionOffset + 1] ?? 0;
					morphPositionData[offset + 2] =
						target.positions?.[positionOffset + 2] ?? 0;
					morphPositionData[offset + 3] = 0;

					morphNormalData[offset] = target.normals?.[positionOffset] ?? 0;
					morphNormalData[offset + 1] =
						target.normals?.[positionOffset + 1] ?? 0;
					morphNormalData[offset + 2] =
						target.normals?.[positionOffset + 2] ?? 0;
					morphNormalData[offset + 3] = 0;
				}
			}

			morphPositionBuffer = this._backend.createBuffer({
				size: morphPositionData.byteLength,
				usage: BufferUsage.Storage | BufferUsage.CopyDst,
				label: `MorphPositionBuffer_${primitive.id}`,
			});
			morphNormalBuffer = this._backend.createBuffer({
				size: morphNormalData.byteLength,
				usage: BufferUsage.Storage | BufferUsage.CopyDst,
				label: `MorphNormalBuffer_${primitive.id}`,
			});
			this._backend.writeBuffer(morphPositionBuffer, morphPositionData as any);
			this._backend.writeBuffer(morphNormalBuffer, morphNormalData as any);
		}

		return {
			vertexBuffer,
			indexBuffer,
			indexCount,
			topology,
			wireframeIndexBuffer,
			wireframeIndexCount: wireframeIndices.length,
			vertexCount,
			morphTargetCount,
			morphPositionBuffer,
			morphNormalBuffer,
		};
	}
}

function createTriangleWireframeIndices(indices: Uint32Array): Uint32Array {
	const triangleCount = Math.floor(indices.length / 3);
	const wireframeIndices = new Uint32Array(triangleCount * 6);
	let cursor = 0;
	for (let i = 0; i < triangleCount; i++) {
		const base = i * 3;
		const i0 = indices[base];
		const i1 = indices[base + 1];
		const i2 = indices[base + 2];
		wireframeIndices[cursor++] = i0;
		wireframeIndices[cursor++] = i1;
		wireframeIndices[cursor++] = i1;
		wireframeIndices[cursor++] = i2;
		wireframeIndices[cursor++] = i2;
		wireframeIndices[cursor++] = i0;
	}
	return wireframeIndices;
}
