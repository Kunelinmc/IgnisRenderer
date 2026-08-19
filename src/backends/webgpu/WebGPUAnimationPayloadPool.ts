import type { DrawPacket } from "../../pipeline/types";
import type {
	JointMatrixMap,
	MorphWeightMap,
} from "../../simulation/animation/types";
import {
	BufferUsage,
	type IRenderBuffer,
} from "../types";
import type { WebGPUDeviceResourceHost } from "./WebGPUDeviceResourceHost";
import type { WebGPUGeometryHandle } from "./WebGPUGeometryRegistry";

const ANIMATION_RESOURCE_RELEASE_DELAY_FRAMES = 60;
const FALLBACK_STORAGE_BYTE_SIZE = 2 * 16 * 4;
const ANIMATION_PARAMS_BYTE_SIZE = 8 * 4;

interface ResolvedAnimationSource {
	jointMatrices: Float32Array | null;
	morphWeights: Float32Array | null;
	jointCount: number;
	morphCount: number;
	vertexCount: number;
	morphSemanticMask: number;
	revision: number;
	revisionReliable: boolean;
}

interface AnimationPayloadEntry {
	jointBuffer: IRenderBuffer | null;
	morphBuffer: IRenderBuffer | null;
	sceneParamsBuffer: IRenderBuffer | null;
	shadowParamsBuffer: IRenderBuffer | null;
	jointCapacity: number;
	morphCapacity: number;
	jointCount: number;
	morphCount: number;
	vertexCount: number;
	morphSemanticMask: number;
	jointCurrent: Float32Array | null;
	jointPrevious: Float32Array | null;
	morphCurrent: Float32Array | null;
	morphPrevious: Float32Array | null;
	jointPayload: Float32Array | null;
	morphPayload: Float32Array | null;
	fallbackJointScratch: Float32Array | null;
	sceneParamsData: Float32Array;
	shadowParamsData: Uint32Array;
	jointSettlePending: boolean;
	morphSettlePending: boolean;
	jointInactiveSinceFrame: number;
	morphInactiveSinceFrame: number;
	inactiveSinceFrame: number;
	lastPreparedFrame: number;
	lastUsedFrame: number;
	lastRevision: number;
	revisionReliable: boolean;
	storageGeneration: number;
	sceneGeneration: number;
	shadowGeneration: number;
}

/** @internal Shared WebGPU animation storage returned to scene bindings. */
export interface WebGPUSceneAnimationPayload {
	readonly generation: number;
	readonly paramsBuffer: IRenderBuffer;
	readonly jointMatricesBuffer: IRenderBuffer;
	readonly morphWeightsBuffer: IRenderBuffer;
	readonly jointCount: number;
	readonly morphCount: number;
}

/** @internal Shared WebGPU animation storage returned to shadow bindings. */
export interface WebGPUShadowAnimationPayload {
	readonly generation: number;
	readonly paramsBuffer: IRenderBuffer;
	readonly jointMatricesBuffer: IRenderBuffer;
	readonly morphWeightsBuffer: IRenderBuffer;
	readonly jointCount: number;
	readonly morphCount: number;
}

/** @internal Diagnostic snapshot for WebGPU animation payload ownership. */
export interface WebGPUAnimationPayloadPoolDebugStats {
	readonly frame: number;
	readonly entryCount: number;
	readonly activeEntryCount: number;
	readonly dormantEntryCount: number;
	readonly staticEntryCount: number;
	readonly jointBufferCount: number;
	readonly morphBufferCount: number;
	readonly sceneParamsBufferCount: number;
	readonly shadowParamsBufferCount: number;
	readonly liveBufferCount: number;
	readonly liveByteLength: number;
	readonly totalBufferCreates: number;
	readonly totalBufferDestroys: number;
	readonly totalUploadCalls: number;
	readonly totalUploadedBytes: number;
	readonly totalSkippedUploads: number;
	readonly capacityRebuilds: number;
	readonly graceReleases: number;
}

/**
 * Owns packet animation payload buffers shared by WebGPU scene and shadow draws.
 *
 * @internal Owned by `WebGPUFrameServiceOwner`; feature runtimes retain their
 * own bind groups and consume only the returned payload views.
 */
export class WebGPUAnimationPayloadPool {
	private readonly _backend: WebGPUDeviceResourceHost;
	private readonly _entries = new Map<string, AnimationPayloadEntry>();
	private readonly _fallbackStorageBuffer: IRenderBuffer;
	private readonly _fallbackSceneParamsBuffer: IRenderBuffer;
	private readonly _fallbackShadowParamsBuffer: IRenderBuffer;
	private _currentFrame = 0;
	private _nextGeneration = 1;
	private _destroyed = false;
	private _totalBufferCreates = 0;
	private _totalBufferDestroys = 0;
	private _totalUploadCalls = 0;
	private _totalUploadedBytes = 0;
	private _totalSkippedUploads = 0;
	private _capacityRebuilds = 0;
	private _graceReleases = 0;

	constructor(backend: WebGPUDeviceResourceHost) {
		this._backend = backend;
		this._fallbackStorageBuffer = this._createBuffer(
			FALLBACK_STORAGE_BYTE_SIZE,
			BufferUsage.Storage | BufferUsage.CopyDst,
			"WebGPUAnimationFallbackStorage",
		);
		this._fallbackSceneParamsBuffer = this._createBuffer(
			ANIMATION_PARAMS_BYTE_SIZE,
			BufferUsage.Uniform | BufferUsage.CopyDst,
			"WebGPUAnimationFallbackSceneParams",
		);
		this._fallbackShadowParamsBuffer = this._createBuffer(
			ANIMATION_PARAMS_BYTE_SIZE,
			BufferUsage.Uniform | BufferUsage.CopyDst,
			"WebGPUAnimationFallbackShadowParams",
		);
	}

	/** @internal Advances the logical frame and releases long-unused entries. */
	public beginFrame(): void {
		if (this._destroyed) return;
		this._currentFrame++;
		for (const [packetId, entry] of this._entries) {
			if (
				entry.lastUsedFrame > 0 &&
				this._currentFrame - entry.lastUsedFrame >= ANIMATION_RESOURCE_RELEASE_DELAY_FRAMES
			) {
				this._releaseEntryResources(entry, true);
				this._entries.delete(packetId);
			}
		}
	}

	/**
	 * Resolves the scene-compatible animation buffers for one packet.
	 *
	 * @internal Used by WebGPU scene draw preparation.
	 */
	public getScenePayload(
		packet: DrawPacket,
		geometry: WebGPUGeometryHandle,
		jointMap: JointMatrixMap | null,
		morphMap: MorphWeightMap | null,
	): WebGPUSceneAnimationPayload {
		const entry = this._prepareEntry(packet, geometry, jointMap, morphMap);
		if (!this._isActive(entry)) {
			return {
				generation: 0,
				paramsBuffer: this._fallbackSceneParamsBuffer,
				jointMatricesBuffer: this._fallbackStorageBuffer,
				morphWeightsBuffer: this._fallbackStorageBuffer,
				jointCount: 0,
				morphCount: 0,
			};
		}

		if (!entry.sceneParamsBuffer) {
			entry.sceneParamsBuffer = this._createBuffer(
				ANIMATION_PARAMS_BYTE_SIZE,
				BufferUsage.Uniform | BufferUsage.CopyDst,
				`ModelAnimationParams_${packet.id}`,
			);
			entry.sceneGeneration = this._allocateGeneration();
			this._writeSceneParams(entry, true);
		} else {
			this._writeSceneParams(entry, false);
		}

		return {
			generation: Math.max(entry.storageGeneration, entry.sceneGeneration),
			paramsBuffer: entry.sceneParamsBuffer,
			jointMatricesBuffer: entry.jointBuffer ?? this._fallbackStorageBuffer,
			morphWeightsBuffer: entry.morphBuffer ?? this._fallbackStorageBuffer,
			jointCount: entry.jointCount,
			morphCount: entry.morphCount,
		};
	}

	/**
	 * Resolves the shadow-compatible animation buffers for one packet.
	 *
	 * @internal Used by `WebGPUShadowCasterRenderer`.
	 */
	public getShadowPayload(
		packet: DrawPacket,
		geometry: WebGPUGeometryHandle,
		jointMap: JointMatrixMap | null,
		morphMap: MorphWeightMap | null,
	): WebGPUShadowAnimationPayload {
		const entry = this._prepareEntry(packet, geometry, jointMap, morphMap);
		if (!this._isActive(entry)) {
			return {
				generation: 0,
				paramsBuffer: this._fallbackShadowParamsBuffer,
				jointMatricesBuffer: this._fallbackStorageBuffer,
				morphWeightsBuffer: this._fallbackStorageBuffer,
				jointCount: 0,
				morphCount: 0,
			};
		}

		if (!entry.shadowParamsBuffer) {
			entry.shadowParamsBuffer = this._createBuffer(
				ANIMATION_PARAMS_BYTE_SIZE,
				BufferUsage.Uniform | BufferUsage.CopyDst,
				`WebGPUShadowAnimationParams_${packet.id}`,
			);
			entry.shadowGeneration = this._allocateGeneration();
			this._writeShadowParams(entry, true);
		} else {
			this._writeShadowParams(entry, false);
		}

		return {
			generation: Math.max(entry.storageGeneration, entry.shadowGeneration),
			paramsBuffer: entry.shadowParamsBuffer,
			jointMatricesBuffer: entry.jointBuffer ?? this._fallbackStorageBuffer,
			morphWeightsBuffer: entry.morphBuffer ?? this._fallbackStorageBuffer,
			jointCount: entry.jointCount,
			morphCount: entry.morphCount,
		};
	}

	/** @internal Returns zero storage for feature-local optional bindings. */
	public getFallbackStorageBuffer(): IRenderBuffer {
		return this._fallbackStorageBuffer;
	}

	/** @internal Returns the shared no-deformation shadow payload. */
	public getStaticShadowPayload(): WebGPUShadowAnimationPayload {
		return {
			generation: 0,
			paramsBuffer: this._fallbackShadowParamsBuffer,
			jointMatricesBuffer: this._fallbackStorageBuffer,
			morphWeightsBuffer: this._fallbackStorageBuffer,
			jointCount: 0,
			morphCount: 0,
		};
	}

	/** @internal Returns deterministic allocation and upload diagnostics. */
	public getDebugStats(): WebGPUAnimationPayloadPoolDebugStats {
		let activeEntryCount = 0;
		let dormantEntryCount = 0;
		let staticEntryCount = 0;
		let jointBufferCount = 0;
		let morphBufferCount = 0;
		let sceneParamsBufferCount = 0;
		let shadowParamsBufferCount = 0;
		let liveByteLength =
			this._fallbackStorageBuffer.size +
			this._fallbackSceneParamsBuffer.size +
			this._fallbackShadowParamsBuffer.size;

		for (const entry of this._entries.values()) {
			if (this._isActive(entry)) {
				activeEntryCount++;
			} else if (this._ownsResources(entry)) {
				dormantEntryCount++;
			} else {
				staticEntryCount++;
			}
			if (entry.jointBuffer) {
				jointBufferCount++;
				liveByteLength += entry.jointBuffer.size;
			}
			if (entry.morphBuffer) {
				morphBufferCount++;
				liveByteLength += entry.morphBuffer.size;
			}
			if (entry.sceneParamsBuffer) {
				sceneParamsBufferCount++;
				liveByteLength += entry.sceneParamsBuffer.size;
			}
			if (entry.shadowParamsBuffer) {
				shadowParamsBufferCount++;
				liveByteLength += entry.shadowParamsBuffer.size;
			}
		}

		const liveBufferCount =
			3 +
			jointBufferCount +
			morphBufferCount +
			sceneParamsBufferCount +
			shadowParamsBufferCount;
		return {
			frame: this._currentFrame,
			entryCount: this._entries.size,
			activeEntryCount,
			dormantEntryCount,
			staticEntryCount,
			jointBufferCount,
			morphBufferCount,
			sceneParamsBufferCount,
			shadowParamsBufferCount,
			liveBufferCount,
			liveByteLength,
			totalBufferCreates: this._totalBufferCreates,
			totalBufferDestroys: this._totalBufferDestroys,
			totalUploadCalls: this._totalUploadCalls,
			totalUploadedBytes: this._totalUploadedBytes,
			totalSkippedUploads: this._totalSkippedUploads,
			capacityRebuilds: this._capacityRebuilds,
			graceReleases: this._graceReleases,
		};
	}

	/** @internal Releases every buffer owned by the payload pool. */
	public destroy(): void {
		if (this._destroyed) return;
		this._destroyed = true;
		for (const entry of this._entries.values()) {
			this._releaseEntryResources(entry, false);
		}
		this._entries.clear();
		this._destroyBuffer(this._fallbackStorageBuffer);
		this._destroyBuffer(this._fallbackSceneParamsBuffer);
		this._destroyBuffer(this._fallbackShadowParamsBuffer);
	}

	private _prepareEntry(
		packet: DrawPacket,
		geometry: WebGPUGeometryHandle,
		jointMap: JointMatrixMap | null,
		morphMap: MorphWeightMap | null,
	): AnimationPayloadEntry {
		let entry = this._entries.get(packet.id);
		if (!entry) {
			entry = createAnimationPayloadEntry();
			this._entries.set(packet.id, entry);
		}
		entry.lastUsedFrame = this._currentFrame;
		if (entry.lastPreparedFrame === this._currentFrame) return entry;

		const source = this._resolveSource(entry, packet, geometry, jointMap, morphMap);
		const wasActive = this._isActive(entry);
		const revisionChanged =
			source.revisionReliable &&
			(!entry.revisionReliable || entry.lastRevision !== source.revision);

		const jointChanged = this._updateJointState(entry, source, wasActive, revisionChanged);
		const morphChanged = this._updateMorphState(entry, source, wasActive, revisionChanged);

		entry.vertexCount = source.vertexCount;
		entry.morphSemanticMask = source.morphSemanticMask;
		entry.lastRevision = source.revision;
		entry.revisionReliable = source.revisionReliable;
		entry.lastPreparedFrame = this._currentFrame;

		if (this._isActive(entry)) {
			entry.inactiveSinceFrame = -1;
		} else if (entry.inactiveSinceFrame < 0) {
			entry.inactiveSinceFrame = this._currentFrame;
		}

		if (!jointChanged && !morphChanged && this._isActive(entry)) {
			this._totalSkippedUploads++;
		}

		this._releaseInactiveResources(entry);
		return entry;
	}

	private _resolveSource(
		entry: AnimationPayloadEntry,
		packet: DrawPacket,
		geometry: WebGPUGeometryHandle,
		jointMap: JointMatrixMap | null,
		morphMap: MorphWeightMap | null,
	): ResolvedAnimationSource {
		const runtimeJoint = jointMap?.get(packet.meshInstance.id) ?? null;
		let jointMatrices = runtimeJoint?.matrices ?? null;
		let jointReliable = !!runtimeJoint;
		if (!jointMatrices && packet.meshInstance.skeleton) {
			packet.meshInstance.skeleton.updateJointMatrices(packet.meshInstance.worldMatrix);
			const requiredLength = packet.meshInstance.skeleton.jointCount * 16;
			if (
				!entry.fallbackJointScratch ||
				entry.fallbackJointScratch.length !== requiredLength
			) {
				entry.fallbackJointScratch = new Float32Array(requiredLength);
			}
			jointMatrices = packet.meshInstance.skeleton.toFloat32Array(entry.fallbackJointScratch);
			jointReliable = false;
		}
		const jointCount = Math.max(0, Math.floor((jointMatrices?.length ?? 0) / 16));

		const runtimeMorph = morphMap?.get(packet.id) ?? null;
		let morphWeights = runtimeMorph?.weights ?? null;
		let requestedMorphCount = Math.max(0, runtimeMorph?.targetCount ?? 0);
		let morphReliable = !!runtimeMorph;
		if (!morphWeights || requestedMorphCount <= 0) {
			const primitiveIndex = packet.mesh.primitives.indexOf(packet.primitive);
			morphWeights =
				primitiveIndex >= 0
					? (packet.meshInstance.morphWeights[primitiveIndex] ?? null)
					: null;
			requestedMorphCount = morphWeights?.length ?? 0;
			morphReliable = false;
		}
		const morphCount = Math.min(
			Math.max(0, requestedMorphCount),
			morphWeights?.length ?? 0,
			geometry.morphTargetCount,
		);

		return {
			jointMatrices,
			morphWeights,
			jointCount,
			morphCount,
			vertexCount: geometry.vertexCount,
			morphSemanticMask: geometry.morphSemanticMask,
			revision: packet.deformationRevision ?? 0,
			revisionReliable:
				(jointCount <= 0 || jointReliable) && (morphCount <= 0 || morphReliable),
		};
	}

	private _updateJointState(
		entry: AnimationPayloadEntry,
		source: ResolvedAnimationSource,
		wasActive: boolean,
		revisionChanged: boolean,
	): boolean {
		if (source.jointCount <= 0 || !source.jointMatrices) {
			if (entry.jointCount > 0) {
				entry.jointCount = 0;
				entry.jointSettlePending = false;
				entry.jointInactiveSinceFrame = this._currentFrame;
				entry.storageGeneration = this._allocateGeneration();
				return true;
			}
			return false;
		}

		const resumed = entry.jointCount <= 0 || !wasActive;
		const rebuilt = this._ensureJointCapacity(entry, packetSafeCount(source.jointCount));
		entry.jointInactiveSinceFrame = -1;
		const changed =
			resumed ||
			rebuilt ||
			revisionChanged ||
			(!source.revisionReliable &&
				!floatPrefixEquals(
					entry.jointCurrent,
					source.jointMatrices,
					source.jointCount * 16,
				));

		if (changed) {
			this._ensureJointArrays(entry);
			if (resumed || rebuilt || !entry.jointCurrent || entry.jointCount <= 0) {
				copyFloatPrefix(source.jointMatrices, entry.jointCurrent!, source.jointCount * 16);
				copyFloatPrefix(entry.jointCurrent!, entry.jointPrevious!, source.jointCount * 16);
				entry.jointSettlePending = false;
			} else {
				copyFloatPrefix(entry.jointCurrent, entry.jointPrevious!, entry.jointCount * 16);
				copyFloatPrefix(source.jointMatrices, entry.jointCurrent, source.jointCount * 16);
				entry.jointSettlePending = true;
			}
			entry.jointCount = source.jointCount;
			this._writeJointPayload(entry);
			return true;
		}

		if (entry.jointSettlePending) {
			copyFloatPrefix(entry.jointCurrent!, entry.jointPrevious!, entry.jointCount * 16);
			entry.jointSettlePending = false;
			this._writeJointPayload(entry);
			return true;
		}
		return false;
	}

	private _updateMorphState(
		entry: AnimationPayloadEntry,
		source: ResolvedAnimationSource,
		wasActive: boolean,
		revisionChanged: boolean,
	): boolean {
		if (source.morphCount <= 0 || !source.morphWeights) {
			if (entry.morphCount > 0) {
				entry.morphCount = 0;
				entry.morphSettlePending = false;
				entry.morphInactiveSinceFrame = this._currentFrame;
				entry.storageGeneration = this._allocateGeneration();
				return true;
			}
			return false;
		}

		const resumed = entry.morphCount <= 0 || !wasActive;
		const rebuilt = this._ensureMorphCapacity(entry, packetSafeCount(source.morphCount));
		entry.morphInactiveSinceFrame = -1;
		const changed =
			resumed ||
			rebuilt ||
			revisionChanged ||
			(!source.revisionReliable &&
				!floatPrefixEquals(entry.morphCurrent, source.morphWeights, source.morphCount));

		if (changed) {
			this._ensureMorphArrays(entry);
			if (resumed || rebuilt || !entry.morphCurrent || entry.morphCount <= 0) {
				copyFloatPrefix(source.morphWeights, entry.morphCurrent!, source.morphCount);
				copyFloatPrefix(entry.morphCurrent!, entry.morphPrevious!, source.morphCount);
				entry.morphSettlePending = false;
			} else {
				copyFloatPrefix(entry.morphCurrent, entry.morphPrevious!, entry.morphCount);
				copyFloatPrefix(source.morphWeights, entry.morphCurrent, source.morphCount);
				entry.morphSettlePending = true;
			}
			entry.morphCount = source.morphCount;
			this._writeMorphPayload(entry);
			return true;
		}

		if (entry.morphSettlePending) {
			copyFloatPrefix(entry.morphCurrent!, entry.morphPrevious!, entry.morphCount);
			entry.morphSettlePending = false;
			this._writeMorphPayload(entry);
			return true;
		}
		return false;
	}

	private _ensureJointCapacity(entry: AnimationPayloadEntry, capacity: number): boolean {
		if (entry.jointBuffer && capacity <= entry.jointCapacity) return false;
		if (entry.jointBuffer) this._destroyBuffer(entry.jointBuffer);
		entry.jointBuffer = this._createBuffer(
			capacity * 2 * 16 * 4,
			BufferUsage.Storage | BufferUsage.CopyDst,
			"WebGPUAnimationJointPayload",
		);
		entry.jointCapacity = capacity;
		entry.jointPayload = new Float32Array(capacity * 2 * 16);
		entry.jointCurrent = resizeFloatArray(entry.jointCurrent, capacity * 16);
		entry.jointPrevious = resizeFloatArray(entry.jointPrevious, capacity * 16);
		entry.storageGeneration = this._allocateGeneration();
		this._capacityRebuilds++;
		return true;
	}

	private _ensureMorphCapacity(entry: AnimationPayloadEntry, capacity: number): boolean {
		if (entry.morphBuffer && capacity <= entry.morphCapacity) return false;
		if (entry.morphBuffer) this._destroyBuffer(entry.morphBuffer);
		entry.morphBuffer = this._createBuffer(
			capacity * 2 * 4,
			BufferUsage.Storage | BufferUsage.CopyDst,
			"WebGPUAnimationMorphPayload",
		);
		entry.morphCapacity = capacity;
		entry.morphPayload = new Float32Array(capacity * 2);
		entry.morphCurrent = resizeFloatArray(entry.morphCurrent, capacity);
		entry.morphPrevious = resizeFloatArray(entry.morphPrevious, capacity);
		entry.storageGeneration = this._allocateGeneration();
		this._capacityRebuilds++;
		return true;
	}

	private _ensureJointArrays(entry: AnimationPayloadEntry): void {
		entry.jointCurrent = resizeFloatArray(entry.jointCurrent, entry.jointCapacity * 16);
		entry.jointPrevious = resizeFloatArray(entry.jointPrevious, entry.jointCapacity * 16);
	}

	private _ensureMorphArrays(entry: AnimationPayloadEntry): void {
		entry.morphCurrent = resizeFloatArray(entry.morphCurrent, entry.morphCapacity);
		entry.morphPrevious = resizeFloatArray(entry.morphPrevious, entry.morphCapacity);
	}

	private _writeJointPayload(entry: AnimationPayloadEntry): void {
		const payload = entry.jointPayload!;
		payload.fill(0);
		payload.set(entry.jointCurrent!.subarray(0, entry.jointCount * 16), 0);
		payload.set(
			entry.jointPrevious!.subarray(0, entry.jointCount * 16),
			entry.jointCapacity * 16,
		);
		this._writeBuffer(entry.jointBuffer!, payload);
	}

	private _writeMorphPayload(entry: AnimationPayloadEntry): void {
		const payload = entry.morphPayload!;
		payload.fill(0);
		payload.set(entry.morphCurrent!.subarray(0, entry.morphCount), 0);
		payload.set(entry.morphPrevious!.subarray(0, entry.morphCount), entry.morphCapacity);
		this._writeBuffer(entry.morphBuffer!, payload);
	}

	private _writeSceneParams(entry: AnimationPayloadEntry, force: boolean): void {
		const next = [
			entry.jointCount,
			entry.morphCount,
			Math.max(1, entry.jointCapacity),
			Math.max(1, entry.morphCapacity),
			entry.vertexCount,
			entry.morphSemanticMask,
			0,
			0,
		];
		if (!force && setArrayValues(entry.sceneParamsData, next) === false) return;
		if (force) setArrayValues(entry.sceneParamsData, next);
		this._writeBuffer(entry.sceneParamsBuffer!, entry.sceneParamsData);
	}

	private _writeShadowParams(entry: AnimationPayloadEntry, force: boolean): void {
		const next = [
			entry.jointCount,
			entry.morphCount,
			entry.jointCount,
			entry.morphCount,
			entry.vertexCount,
			entry.morphSemanticMask,
			0,
			0,
		];
		if (!force && setArrayValues(entry.shadowParamsData, next) === false) return;
		if (force) setArrayValues(entry.shadowParamsData, next);
		this._writeBuffer(entry.shadowParamsBuffer!, entry.shadowParamsData);
	}

	private _releaseInactiveResources(entry: AnimationPayloadEntry): void {
		if (
			entry.jointBuffer &&
			entry.jointCount <= 0 &&
			hasGraceElapsed(this._currentFrame, entry.jointInactiveSinceFrame)
		) {
			this._destroyBuffer(entry.jointBuffer);
			entry.jointBuffer = null;
			entry.jointCapacity = 0;
			entry.jointCurrent = null;
			entry.jointPrevious = null;
			entry.jointPayload = null;
			this._graceReleases++;
		}
		if (
			entry.morphBuffer &&
			entry.morphCount <= 0 &&
			hasGraceElapsed(this._currentFrame, entry.morphInactiveSinceFrame)
		) {
			this._destroyBuffer(entry.morphBuffer);
			entry.morphBuffer = null;
			entry.morphCapacity = 0;
			entry.morphCurrent = null;
			entry.morphPrevious = null;
			entry.morphPayload = null;
			this._graceReleases++;
		}

		if (
			!this._isActive(entry) &&
			hasGraceElapsed(this._currentFrame, entry.inactiveSinceFrame)
		) {
			if (entry.sceneParamsBuffer) {
				this._destroyBuffer(entry.sceneParamsBuffer);
				entry.sceneParamsBuffer = null;
				entry.sceneGeneration = this._allocateGeneration();
				this._graceReleases++;
			}
			if (entry.shadowParamsBuffer) {
				this._destroyBuffer(entry.shadowParamsBuffer);
				entry.shadowParamsBuffer = null;
				entry.shadowGeneration = this._allocateGeneration();
				this._graceReleases++;
			}
		}
	}

	private _releaseEntryResources(entry: AnimationPayloadEntry, graceRelease: boolean): void {
		for (const buffer of [
			entry.jointBuffer,
			entry.morphBuffer,
			entry.sceneParamsBuffer,
			entry.shadowParamsBuffer,
		]) {
			if (buffer) {
				this._destroyBuffer(buffer);
				if (graceRelease) this._graceReleases++;
			}
		}
	}

	private _isActive(entry: AnimationPayloadEntry): boolean {
		return entry.jointCount > 0 || entry.morphCount > 0;
	}

	private _ownsResources(entry: AnimationPayloadEntry): boolean {
		return !!(
			entry.jointBuffer ||
			entry.morphBuffer ||
			entry.sceneParamsBuffer ||
			entry.shadowParamsBuffer
		);
	}

	private _allocateGeneration(): number {
		return this._nextGeneration++;
	}

	private _createBuffer(size: number, usage: BufferUsage, label: string): IRenderBuffer {
		this._totalBufferCreates++;
		return this._backend.createBuffer({ size, usage, label });
	}

	private _destroyBuffer(buffer: IRenderBuffer): void {
		buffer.destroy();
		this._totalBufferDestroys++;
	}

	private _writeBuffer(buffer: IRenderBuffer, data: Float32Array | Uint32Array): void {
		this._backend.writeBuffer(buffer, data as Float32Array<ArrayBuffer>);
		this._totalUploadCalls++;
		this._totalUploadedBytes += data.byteLength;
	}
}

function createAnimationPayloadEntry(): AnimationPayloadEntry {
	return {
		jointBuffer: null,
		morphBuffer: null,
		sceneParamsBuffer: null,
		shadowParamsBuffer: null,
		jointCapacity: 0,
		morphCapacity: 0,
		jointCount: 0,
		morphCount: 0,
		vertexCount: 0,
		morphSemanticMask: 0,
		jointCurrent: null,
		jointPrevious: null,
		morphCurrent: null,
		morphPrevious: null,
		jointPayload: null,
		morphPayload: null,
		fallbackJointScratch: null,
		sceneParamsData: new Float32Array(8),
		shadowParamsData: new Uint32Array(8),
		jointSettlePending: false,
		morphSettlePending: false,
		jointInactiveSinceFrame: -1,
		morphInactiveSinceFrame: -1,
		inactiveSinceFrame: -1,
		lastPreparedFrame: -1,
		lastUsedFrame: 0,
		lastRevision: -1,
		revisionReliable: false,
		storageGeneration: 0,
		sceneGeneration: 0,
		shadowGeneration: 0,
	};
}

function packetSafeCount(value: number): number {
	return Math.max(1, Math.floor(value));
}

function hasGraceElapsed(currentFrame: number, inactiveSinceFrame: number): boolean {
	return (
		inactiveSinceFrame >= 0 &&
		currentFrame - inactiveSinceFrame + 1 >= ANIMATION_RESOURCE_RELEASE_DELAY_FRAMES
	);
}

function resizeFloatArray(
	current: Float32Array | null,
	length: number
): Float32Array {
	if (current?.length === length) return current;
	const next = new Float32Array(length);
	if (current) next.set(current.subarray(0, Math.min(current.length, length)));
	return next;
}

function floatPrefixEquals(
	left: Float32Array | null,
	right: Float32Array,
	length: number
): boolean {
	if (!left || left.length < length || right.length < length) return false;
	for (let i = 0; i < length; i++) {
		if (left[i] !== right[i]) return false;
	}
	return true;
}

function copyFloatPrefix(
	source: Float32Array,
	target: Float32Array,
	length: number
): void {
	target.set(source.subarray(0, length), 0);
}

function setArrayValues(
	target: Float32Array | Uint32Array,
	values: readonly number[]
): boolean {
	let changed = false;
	for (let i = 0; i < values.length; i++) {
		if (target[i] !== values[i]) {
			target[i] = values[i];
			changed = true;
		}
	}
	return changed;
}
