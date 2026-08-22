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

export const DEFAULT_DEFERRED_GEOMETRY_UPLOADS_PER_FRAME = 16;
export const DEFAULT_DEFERRED_GEOMETRY_UPLOAD_BYTES_PER_FRAME = 32 * 1024 * 1024;

export interface WebGLGeometryRegistryOptions {
	/**
	 * Upload scheduling mode. `deferred` queues first-use uploads into a
	 * frame-budgeted queue instead of blocking the draw pass.
	 */
	readonly uploadScheduling?: "immediate" | "deferred";
	/** Hard cap on uploads processed per `processPendingUploads` call. */
	readonly maxUploadsPerFrame?: number;
	/** Estimated-bytes budget per `processPendingUploads` call. */
	readonly maxUploadBytesPerFrame?: number;
	/** Invoked for packets that need another frame to complete their upload. */
	readonly onUploadPending?: (packets: readonly DrawPacket[]) => void;
}

interface UploadPrimitiveResult {
	handle: WebGLGeometryHandle | null;
	cacheFailure: boolean;
}

interface PendingGeometryUpload {
	primitive: IPrimitive;
	packet: DrawPacket;
	packetsById: Map<string, DrawPacket>;
	geometryVersion: number;
	estimatedBytes: number;
	queued: boolean;
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
	private _uploadScheduling: "immediate" | "deferred";
	private _maxUploadsPerFrame: number;
	private _maxUploadBytesPerFrame: number;
	private _onUploadPending: ((packets: readonly DrawPacket[]) => void) | null;
	private _pendingUploadQueue: PendingGeometryUpload[] = [];
	private _pendingUploadsByPrimitive = new Map<IPrimitive, PendingGeometryUpload>();

	constructor(
		gl: WebGL2RenderingContext,
		warn?: WebGLGeometryWarn,
		options: WebGLGeometryRegistryOptions = {}
	) {
		this._gl = gl;
		this._warnCallback = warn ?? null;
		const maxTextureSize = gl.getParameter?.(gl.MAX_TEXTURE_SIZE);
		this._maxTextureSize = Number.isFinite(maxTextureSize) ?
			Math.max(1, Math.floor(maxTextureSize)) : 4096;
		this._uploadScheduling = options.uploadScheduling ?? "immediate";
		this._maxUploadsPerFrame = Math.max(
			1,
			Math.floor(
				options.maxUploadsPerFrame ??
					DEFAULT_DEFERRED_GEOMETRY_UPLOADS_PER_FRAME
			)
		);
		this._maxUploadBytesPerFrame = Math.max(
			1,
			Math.floor(
				options.maxUploadBytesPerFrame ??
					DEFAULT_DEFERRED_GEOMETRY_UPLOAD_BYTES_PER_FRAME
			)
		);
		this._onUploadPending = options.onUploadPending ?? null;
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
		if (this._uploadScheduling === "deferred") {
			this._queueDeferredUpload(primitive, packet, geometryVersion);
			return null;
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
		this._pendingUploadQueue.length = 0;
		this._pendingUploadsByPrimitive.clear();
		for (const handle of this._owned) {
			this._destroyHandle(handle);
		}
		this._owned.clear();
	}

	/**
	 * Returns the number of queued geometry uploads waiting for frame-budgeted
	 * processing. Reading this value has no side effects.
	 */
	public get pendingUploadCount(): number {
		return this._pendingUploadQueue.length;
	}

	/**
	 * Processes deferred geometry uploads at the start of a WebGL frame.
	 *
	 * The method respects the configured upload count and estimated-byte
	 * budgets. If queued uploads remain after processing, `onUploadPending`
	 * is called so the renderer can schedule another frame.
	 */
	public beginFrame(): void {
		this.processPendingUploads();
	}

	public processPendingUploads(): void {
		if (this._pendingUploadQueue.length === 0) {
			return;
		}

		let uploads = 0;
		let uploadedBytes = 0;
		while (this._pendingUploadQueue.length > 0) {
			const pending = this._pendingUploadQueue[0];
			const wouldExceedUploadCount = uploads >= this._maxUploadsPerFrame;
			const wouldExceedByteBudget =
				uploadedBytes + pending.estimatedBytes >
				this._maxUploadBytesPerFrame;
			if (
				uploads > 0 &&
				(wouldExceedUploadCount || wouldExceedByteBudget)
			) {
				break;
			}

			this._pendingUploadQueue.shift();
			pending.queued = false;
			if (this._pendingUploadsByPrimitive.get(pending.primitive) === pending) {
				this._pendingUploadsByPrimitive.delete(pending.primitive);
			}
			if (
				pending.geometryVersion !== (pending.primitive.geometryVersion ?? 0)
			) {
				continue;
			}

			const uploaded = this._uploadPrimitive(pending.packet);
			if (!uploaded.handle && !uploaded.cacheFailure) {
				continue;
			}
			if (uploaded.handle) {
				this._owned.add(uploaded.handle);
			}
			this._cache.set(pending.primitive, {
				handle: uploaded.handle,
				geometryVersion: pending.geometryVersion,
				cacheFailure: uploaded.cacheFailure,
			});
			uploads++;
			uploadedBytes += pending.estimatedBytes;
		}

		if (this._pendingUploadQueue.length > 0) {
			this._notifyQueuedUploadsPending();
		}
	}

	private _queueDeferredUpload(
		primitive: IPrimitive,
		packet: DrawPacket,
		geometryVersion: number
	): void {
		const existing = this._pendingUploadsByPrimitive.get(primitive);
		if (existing) {
			if (existing.geometryVersion !== geometryVersion) {
				existing.geometryVersion = geometryVersion;
				existing.estimatedBytes =
					estimateWebGLGeometryUploadBytes(primitive);
			}
			existing.packet = packet;
			const alreadyTracked = existing.packetsById.has(packet.id);
			existing.packetsById.set(packet.id, packet);
			if (!alreadyTracked) {
				this._notifyUploadPending([packet]);
			}
			return;
		}
		const pending: PendingGeometryUpload = {
			primitive,
			packet,
			packetsById: new Map([[packet.id, packet]]),
			geometryVersion,
			estimatedBytes: estimateWebGLGeometryUploadBytes(primitive),
			queued: true,
		};
		this._pendingUploadsByPrimitive.set(primitive, pending);
		this._pendingUploadQueue.push(pending);
		this._notifyUploadPending([packet]);
	}

	private _notifyQueuedUploadsPending(): void {
		if (!this._onUploadPending) return;
		const packets: DrawPacket[] = [];
		for (const pending of this._pendingUploadQueue) {
			packets.push(...pending.packetsById.values());
		}
		this._notifyUploadPending(packets);
	}

	private _notifyUploadPending(packets: readonly DrawPacket[]): void {
		if (packets.length === 0) return;
		this._onUploadPending?.(packets);
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
		// Fused validation + interleave pass: each source component is checked
		// once as it is written, replacing separate whole-array finite scans.
		// Reports the earliest offending vertex's attribute when several
		// attributes contain non-finite values.
		let nonFiniteAttribute: WebGLNonFiniteAttribute | null = null;
		const scanState: { attribute: WebGLNonFiniteAttribute } = {
			attribute: "positions",
		};
		interleaveLoop:
		for (let i = 0; i < vertexCount; i++) {
			const srcPos = i * 3;
			const srcUv = i * 2;
			const srcTangent = i * 4;
			const dst = i * WEBGL_SCENE_VERTEX_FLOATS;
			if (!copyCheckedComponents(interleaved, dst, positions, srcPos, 3, WEBGL_XYZ_DEFAULTS, "positions", scanState)) {
				nonFiniteAttribute = "positions";
				break interleaveLoop;
			}
			if (!copyCheckedComponents(interleaved, dst + 3, normals, srcPos, 3, WEBGL_NORMAL_DEFAULTS, "normals", scanState)) {
				nonFiniteAttribute = "normals";
				break interleaveLoop;
			}
			if (!copyCheckedComponents(interleaved, dst + 6, uv0, srcUv, 2, WEBGL_UV_DEFAULTS, "uv0", scanState)) {
				nonFiniteAttribute = "uv0";
				break interleaveLoop;
			}
			if (uv1) {
				if (!copyCheckedComponents(interleaved, dst + 8, uv1, srcUv, 2, WEBGL_UV_DEFAULTS, "uv1", scanState)) {
					nonFiniteAttribute = "uv1";
					break interleaveLoop;
				}
			} else {
				interleaved[dst + 8] = interleaved[dst + 6];
				interleaved[dst + 9] = interleaved[dst + 7];
			}
			if (uv2) {
				if (!copyCheckedComponents(interleaved, dst + 10, uv2, srcUv, 2, WEBGL_UV_DEFAULTS, "uv2", scanState)) {
					nonFiniteAttribute = "uv2";
					break interleaveLoop;
				}
			} else {
				interleaved[dst + 10] = interleaved[dst + 6];
				interleaved[dst + 11] = interleaved[dst + 7];
			}
			if (uv3) {
				if (!copyCheckedComponents(interleaved, dst + 12, uv3, srcUv, 2, WEBGL_UV_DEFAULTS, "uv3", scanState)) {
					nonFiniteAttribute = "uv3";
					break interleaveLoop;
				}
			} else {
				interleaved[dst + 12] = interleaved[dst + 6];
				interleaved[dst + 13] = interleaved[dst + 7];
			}
			if (!copyCheckedComponents(interleaved, dst + 14, tangents, srcTangent, 4, WEBGL_TANGENT_DEFAULTS, "tangents", scanState)) {
				nonFiniteAttribute = "tangents";
				break interleaveLoop;
			}
		}
		if (nonFiniteAttribute) {
			const diagnostic =
				WEBGL_NONFINITE_ATTRIBUTE_DIAGNOSTICS[nonFiniteAttribute];
			this._warn(
				`webgl-geometry-nonfinite-${diagnostic.key}-${primitive.id}`,
				`WebGL geometry ${primitiveLabel} contains non-finite ${diagnostic.label} values; skipping`,
			);
			return { handle: null, cacheFailure: true };
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
			indexData = new Uint16Array(indices);
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

/**
 * Estimates GPU bytes for a primitive upload without touching GL state.
 * Used to budget deferred uploads; over-estimates index data (always assumes
 * 32-bit indices) so budget decisions stay conservative.
 */
function estimateWebGLGeometryUploadBytes(primitive: IPrimitive): number {
	const geometry = primitive.geometry;
	const vertexCount = Math.max(0, ((geometry.positions?.length ?? 0) / 3) | 0);
	let bytes = vertexCount * WEBGL_SCENE_VERTEX_FLOATS * 4;
	bytes += (geometry.indices?.length ?? 0) * 4;
	const skinProfile = resolveWebGLSkinProfile(geometry);
	if (skinProfile === "skin4") {
		bytes += vertexCount * 32;
	} else if (skinProfile === "skin8") {
		bytes += vertexCount * 64;
	}
	const morphTargetCount = Math.min(
		WEBGL_MAX_MORPH_TARGETS,
		geometry.morphTargets?.length ?? 0,
	);
	if (morphTargetCount > 0) {
		// Position + normal RGBA32F morph texels per vertex, padded estimate.
		bytes += morphTargetCount * vertexCount * 32;
	}
	return bytes;
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

type WebGLNonFiniteAttribute =
	| "positions"
	| "normals"
	| "uv0"
	| "uv1"
	| "uv2"
	| "uv3"
	| "tangents";

const WEBGL_NONFINITE_ATTRIBUTE_DIAGNOSTICS: Record<
	WebGLNonFiniteAttribute,
	{ key: string; label: string }
> = {
	positions: { key: "positions", label: "position" },
	normals: { key: "normals", label: "normal" },
	uv0: { key: "uv", label: "UV" },
	uv1: { key: "uv1", label: "UV1" },
	uv2: { key: "uv2", label: "UV2" },
	uv3: { key: "uv3", label: "UV3" },
	tangents: { key: "tangents", label: "tangent" },
};

const WEBGL_XYZ_DEFAULTS = [0, 0, 0];
const WEBGL_NORMAL_DEFAULTS = [0, 0, 1];
const WEBGL_UV_DEFAULTS = [0, 0];
const WEBGL_TANGENT_DEFAULTS = [0, 0, 0, 0];

/**
 * Copies `componentCount` components from an optional attribute source into
 * the interleaved target while validating finiteness. Missing sources and
 * out-of-range reads fall back to per-component defaults, matching legacy
 * upload behavior. Returns `false` when a non-finite value was found; in that
 * case `state.attribute` names the offending attribute.
 */
function copyCheckedComponents(
	target: Float32Array,
	targetOffset: number,
	source: ArrayLike<number> | null | undefined,
	sourceOffset: number,
	componentCount: number,
	defaults: readonly number[],
	attribute: WebGLNonFiniteAttribute,
	state: { attribute: WebGLNonFiniteAttribute },
): boolean {
	if (!source) {
		for (let i = 0; i < componentCount; i++) {
			target[targetOffset + i] = defaults[i];
		}
		return true;
	}
	for (let i = 0; i < componentCount; i++) {
		const value = source[sourceOffset + i];
		if (value === undefined) {
			target[targetOffset + i] = defaults[i];
			continue;
		}
		if (!Number.isFinite(value)) {
			state.attribute = attribute;
			return false;
		}
		target[targetOffset + i] = value;
	}
	return true;
}
