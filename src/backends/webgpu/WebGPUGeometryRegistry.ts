import {
	DEFAULT_PRIMITIVE_DRAW_TOPOLOGY,
	type IPrimitive,
	type PrimitiveDrawTopology,
} from "../../core/types";
import { GeometryBuilder } from "../../meshes/GeometryBuilder";
import {
	BufferUsage,
	type IndexFormat,
	type IRenderBuffer,
	type VertexBufferLayout,
} from "../types";
import type { WebGPUDeviceResourceHost } from "./WebGPUDeviceResourceHost";
import {
	WEBGPU_GEOMETRY_DEFAULT_SLOT,
	WEBGPU_GEOMETRY_POSITION_SLOT,
	WEBGPU_GEOMETRY_SKIN_SLOT,
	WEBGPU_GEOMETRY_SURFACE_SLOT,
	packWebGPUVertexGeometry,
	type WebGPUSkinProfile,
} from "./WebGPUGeometryPacking";
import { WEBGPU_MAX_MORPH_TARGETS } from "./constants";

export const WEBGPU_MORPH_POSITION_BIT = 1 << 0;
export const WEBGPU_MORPH_NORMAL_BIT = 1 << 1;

/** @internal WebGPU geometry vertex binding selected by the registry. */
export interface WebGPUVertexBufferBinding {
	readonly slot: number;
	readonly buffer: IRenderBuffer;
}

/** @internal WebGPU geometry resources owned by `WebGPUGeometryRegistry`. */
export interface WebGPUGeometryHandle {
	positionBuffer: IRenderBuffer;
	surfaceBuffer: IRenderBuffer | null;
	skinBuffer: IRenderBuffer | null;
	vertexBindings: readonly WebGPUVertexBufferBinding[];
	shadowVertexBindings: readonly WebGPUVertexBufferBinding[];
	sceneVertexLayouts: readonly VertexBufferLayout[];
	shadowVertexLayouts: readonly VertexBufferLayout[];
	layoutKey: string;
	shadowLayoutKey: string;
	skinProfile: WebGPUSkinProfile;
	vertexByteLength: number;
	indexBuffer: IRenderBuffer;
	indexFormat: IndexFormat;
	indexByteLength: number;
	indexCount: number;
	topology: PrimitiveDrawTopology;
	wireframeIndexBuffer: IRenderBuffer | null;
	wireframeIndexFormat: IndexFormat;
	wireframeIndexByteLength: number;
	wireframeIndexCount: number;
	vertexCount: number;
	morphTargetCount: number;
	morphSemanticMask: number;
	morphPositionBuffer: IRenderBuffer | null;
	morphNormalBuffer: IRenderBuffer | null;
	morphByteLength: number;
	/** @internal Retained only for lazy wireframe generation. */
	readonly sourceIndices: Uint32Array;
}

interface WebGPUCachedGeometryEntry {
	handle: WebGPUGeometryHandle;
	geometryVersion: number;
}

interface PackedIndexData {
	data: Uint16Array | Uint32Array;
	format: IndexFormat;
}

/** @internal Owns device-lifetime WebGPU geometry resources. */
export class WebGPUGeometryRegistry {
	private _backend: WebGPUDeviceResourceHost;
	private _cache = new WeakMap<IPrimitive, WebGPUCachedGeometryEntry>();
	private _owned = new Set<WebGPUGeometryHandle>();
	private _defaultVertexBuffer: IRenderBuffer | null = null;
	private _finalizationRegistry: FinalizationRegistry<WebGPUGeometryHandle> | null =
		typeof FinalizationRegistry === "function" ?
			new FinalizationRegistry((handle) => {
				this._owned.delete(handle);
				this._destroyHandle(handle);
			})
		: null;

	constructor(backend: WebGPUDeviceResourceHost) {
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
			cached = { handle, geometryVersion };
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

	/**
	 * Returns geometry with its lazily generated wireframe index buffer.
	 *
	 * @internal Used by WebGPU scene draw preparation for wireframe materials.
	 * @param primitive Primitive whose canonical triangle edges are required.
	 * @returns The cached geometry handle with wireframe resources populated.
	 * @sideEffects May allocate and upload one deduplicated index buffer.
	 */
	public getWireframeGeometry(primitive: IPrimitive): WebGPUGeometryHandle {
		const handle = this.getGeometry(primitive);
		if (
			handle.topology === DEFAULT_PRIMITIVE_DRAW_TOPOLOGY &&
			!handle.wireframeIndexBuffer
		) {
			this._uploadWireframeIndices(handle);
		}
		return handle;
	}

	public releaseGeometry(primitive: IPrimitive): void {
		const cached = this._cache.get(primitive);
		if (!cached) return;
		this._cache.delete(primitive);
		this._finalizationRegistry?.unregister(primitive as unknown as object);
		this._owned.delete(cached.handle);
		this._destroyHandle(cached.handle);
	}

	public destroy(): void {
		for (const handle of this._owned) this._destroyHandle(handle);
		this._owned.clear();
		this._defaultVertexBuffer?.destroy();
		this._defaultVertexBuffer = null;
		this._cache = new WeakMap<IPrimitive, WebGPUCachedGeometryEntry>();
		this._finalizationRegistry = null;
	}

	private _destroyHandle(handle: WebGPUGeometryHandle): void {
		handle.positionBuffer.destroy();
		handle.surfaceBuffer?.destroy();
		handle.skinBuffer?.destroy();
		handle.indexBuffer.destroy();
		handle.wireframeIndexBuffer?.destroy();
		handle.morphPositionBuffer?.destroy();
		handle.morphNormalBuffer?.destroy();
	}

	private _uploadGeometry(primitive: IPrimitive): WebGPUGeometryHandle {
		const geometry = primitive.geometry;
		const topology = primitive.topology ?? DEFAULT_PRIMITIVE_DRAW_TOPOLOGY;
		const vertexCount = GeometryBuilder.getVertexCount(geometry);
		const packed = packWebGPUVertexGeometry(geometry, vertexCount);
		const defaultVertexBuffer = this._getDefaultVertexBuffer(packed.defaultData);

		const positionBuffer = this._createAndWriteBuffer(
			packed.position.data,
			BufferUsage.Vertex | BufferUsage.CopyDst,
			`PositionVertexBuffer_${primitive.id}`
		);
		const surfaceBuffer = packed.surface ?
			this._createAndWriteBuffer(
				packed.surface.data,
				BufferUsage.Vertex | BufferUsage.CopyDst,
				`SurfaceVertexBuffer_${primitive.id}`
			)
		: null;
		const skinBuffer = packed.skin ?
			this._createAndWriteBuffer(
				packed.skin.data,
				BufferUsage.Vertex | BufferUsage.CopyDst,
				`SkinVertexBuffer_${primitive.id}`
			)
		: null;
		const vertexBindings = createVertexBindings(
			positionBuffer,
			surfaceBuffer,
			skinBuffer,
			defaultVertexBuffer
		);
		const shadowVertexBindings = createShadowVertexBindings(
			positionBuffer,
			skinBuffer,
			defaultVertexBuffer
		);

		const packedIndices = packIndexData(geometry.indices);
		const indexBuffer = this._createAndWriteBuffer(
			packedIndices.data,
			BufferUsage.Index | BufferUsage.CopyDst,
			`IndexBuffer_${primitive.id}`
		);
		const morph = this._uploadMorphTargets(primitive, vertexCount);

		return {
			positionBuffer,
			surfaceBuffer,
			skinBuffer,
			vertexBindings,
			shadowVertexBindings,
			sceneVertexLayouts: packed.sceneLayouts,
			shadowVertexLayouts: packed.shadowLayouts,
			layoutKey: packed.layoutKey,
			shadowLayoutKey: packed.shadowLayoutKey,
			skinProfile: packed.skinProfile,
			vertexByteLength: packed.vertexByteLength,
			indexBuffer,
			indexFormat: packedIndices.format,
			indexByteLength: packedIndices.data.byteLength,
			indexCount: packedIndices.data.length,
			topology,
			wireframeIndexBuffer: null,
			wireframeIndexFormat: "uint16",
			wireframeIndexByteLength: 0,
			wireframeIndexCount: 0,
			vertexCount,
			morphTargetCount: morph.targetCount,
			morphSemanticMask: morph.semanticMask,
			morphPositionBuffer: morph.positionBuffer,
			morphNormalBuffer: morph.normalBuffer,
			morphByteLength: morph.byteLength,
			sourceIndices: geometry.indices,
		};
	}

	private _uploadMorphTargets(
		primitive: IPrimitive,
		vertexCount: number
	): {
		targetCount: number;
		semanticMask: number;
		positionBuffer: IRenderBuffer | null;
		normalBuffer: IRenderBuffer | null;
		byteLength: number;
	} {
		const targets = primitive.geometry.morphTargets ?? [];
		const targetCount = Math.min(WEBGPU_MAX_MORPH_TARGETS, targets.length);
		const activeTargets = targets.slice(0, targetCount);
		const hasPositions = activeTargets.some((target) => !!target.positions);
		const hasNormals = activeTargets.some((target) => !!target.normals);
		let semanticMask = 0;
		let positionBuffer: IRenderBuffer | null = null;
		let normalBuffer: IRenderBuffer | null = null;
		let byteLength = 0;

		if (hasPositions) {
			const data = packMorphSemantic(targets, targetCount, vertexCount, "positions");
			positionBuffer = this._createAndWriteBuffer(
				data,
				BufferUsage.Storage | BufferUsage.CopyDst,
				`MorphPositionBuffer_${primitive.id}`
			);
			semanticMask |= WEBGPU_MORPH_POSITION_BIT;
			byteLength += data.byteLength;
		}
		if (hasNormals) {
			const data = packMorphSemantic(targets, targetCount, vertexCount, "normals");
			normalBuffer = this._createAndWriteBuffer(
				data,
				BufferUsage.Storage | BufferUsage.CopyDst,
				`MorphNormalBuffer_${primitive.id}`
			);
			semanticMask |= WEBGPU_MORPH_NORMAL_BIT;
			byteLength += data.byteLength;
		}

		return { targetCount, semanticMask, positionBuffer, normalBuffer, byteLength };
	}

	private _uploadWireframeIndices(handle: WebGPUGeometryHandle): void {
		const data = createDeduplicatedWireframeIndices(handle.sourceIndices);
		handle.wireframeIndexBuffer = this._createAndWriteBuffer(
			data.data,
			BufferUsage.Index | BufferUsage.CopyDst,
			"WireframeIndexBuffer"
		);
		handle.wireframeIndexFormat = data.format;
		handle.wireframeIndexByteLength = data.data.byteLength;
		handle.wireframeIndexCount = data.data.length;
	}

	private _getDefaultVertexBuffer(data: Uint8Array): IRenderBuffer {
		if (!this._defaultVertexBuffer) {
			this._defaultVertexBuffer = this._createAndWriteBuffer(
				data,
				BufferUsage.Vertex | BufferUsage.CopyDst,
				"WebGPUDefaultVertexBuffer"
			);
		}
		return this._defaultVertexBuffer;
	}

	private _createAndWriteBuffer(
		data: ArrayBufferView,
		usage: BufferUsage,
		label: string
	): IRenderBuffer {
		const alignedByteLength = Math.max(4, alignTo4(data.byteLength));
		const buffer = this._backend.createBuffer({
			size: alignedByteLength,
			usage,
			label,
		});
		let writeData: ArrayBufferView = data;
		if (data.byteLength === 0) {
			writeData = new Uint32Array([0]);
		} else if ((data.byteLength & 0x3) !== 0) {
			const padded = new Uint8Array(alignedByteLength);
			padded.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
			writeData = padded;
		}
		this._backend.writeBuffer(buffer, writeData as any);
		return buffer;
	}
}

function alignTo4(value: number): number {
	return (value + 3) & ~3;
}

function createVertexBindings(
	positionBuffer: IRenderBuffer,
	surfaceBuffer: IRenderBuffer | null,
	skinBuffer: IRenderBuffer | null,
	defaultBuffer: IRenderBuffer
): WebGPUVertexBufferBinding[] {
	const bindings: WebGPUVertexBufferBinding[] = [
		{ slot: WEBGPU_GEOMETRY_POSITION_SLOT, buffer: positionBuffer },
		{
			slot: WEBGPU_GEOMETRY_SURFACE_SLOT,
			buffer: surfaceBuffer ?? defaultBuffer,
		},
		{
			slot: WEBGPU_GEOMETRY_SKIN_SLOT,
			buffer: skinBuffer ?? defaultBuffer,
		},
		{ slot: WEBGPU_GEOMETRY_DEFAULT_SLOT, buffer: defaultBuffer },
	];
	return bindings;
}

function createShadowVertexBindings(
	positionBuffer: IRenderBuffer,
	skinBuffer: IRenderBuffer | null,
	defaultBuffer: IRenderBuffer
): WebGPUVertexBufferBinding[] {
	return [
		{ slot: WEBGPU_GEOMETRY_POSITION_SLOT, buffer: positionBuffer },
		{ slot: WEBGPU_GEOMETRY_SURFACE_SLOT, buffer: defaultBuffer },
		{
			slot: WEBGPU_GEOMETRY_SKIN_SLOT,
			buffer: skinBuffer ?? defaultBuffer,
		},
		{ slot: WEBGPU_GEOMETRY_DEFAULT_SLOT, buffer: defaultBuffer },
	];
}

function packIndexData(indices: Uint32Array): PackedIndexData {
	let maxIndex = 0;
	for (let i = 0; i < indices.length; i++) maxIndex = Math.max(maxIndex, indices[i]);
	if (maxIndex <= 65535) {
		return { data: Uint16Array.from(indices), format: "uint16" };
	}
	return { data: new Uint32Array(indices), format: "uint32" };
}

function packMorphSemantic(
	targets: NonNullable<IPrimitive["geometry"]["morphTargets"]>,
	targetCount: number,
	vertexCount: number,
	semantic: "positions" | "normals"
): Float32Array {
	const data = new Float32Array(targetCount * vertexCount * 3);
	for (let targetIndex = 0; targetIndex < targetCount; targetIndex++) {
		const source = targets[targetIndex][semantic];
		if (!source) continue;
		const targetOffset = targetIndex * vertexCount * 3;
		for (let component = 0; component < vertexCount * 3; component++) {
			data[targetOffset + component] = source[component] ?? 0;
		}
	}
	return data;
}

function createDeduplicatedWireframeIndices(indices: Uint32Array): PackedIndexData {
	const triangleCount = Math.floor(indices.length / 3);
	const edgeCapacity = triangleCount * 3;
	if (edgeCapacity === 0) return { data: new Uint16Array(0), format: "uint16" };

	let tableCapacity = 1;
	while (tableCapacity < edgeCapacity * 2) tableCapacity *= 2;
	const occupied = new Uint8Array(tableCapacity);
	const edgeMin = new Uint32Array(tableCapacity);
	const edgeMax = new Uint32Array(tableCapacity);
	const output = new Uint32Array(edgeCapacity * 2);
	const mask = tableCapacity - 1;
	let cursor = 0;
	let maxIndex = 0;

	const insert = (left: number, right: number): void => {
		const min = Math.min(left, right) >>> 0;
		const max = Math.max(left, right) >>> 0;
		let slot = (Math.imul(min, 0x9e3779b1) ^ Math.imul(max, 0x85ebca6b)) & mask;
		while (occupied[slot]) {
			if (edgeMin[slot] === min && edgeMax[slot] === max) return;
			slot = (slot + 1) & mask;
		}
		occupied[slot] = 1;
		edgeMin[slot] = min;
		edgeMax[slot] = max;
		output[cursor++] = left;
		output[cursor++] = right;
		maxIndex = Math.max(maxIndex, left, right);
	};

	for (let triangleIndex = 0; triangleIndex < triangleCount; triangleIndex++) {
		const base = triangleIndex * 3;
		const i0 = indices[base];
		const i1 = indices[base + 1];
		const i2 = indices[base + 2];
		insert(i0, i1);
		insert(i1, i2);
		insert(i2, i0);
	}

	if (maxIndex <= 65535) {
		const compact = new Uint16Array(cursor);
		for (let i = 0; i < cursor; i++) compact[i] = output[i];
		return { data: compact, format: "uint16" };
	}
	return { data: output.slice(0, cursor), format: "uint32" };
}
