import { Logger } from "../../foundation/Logger";
import type { DrawPacket, FrameContext } from "../../pipeline/types";
import {
	ANIMATION_JOINT_MATRICES_KEY,
	ANIMATION_MORPH_WEIGHTS_KEY,
	type JointMatrixMap,
	type MorphWeightMap,
} from "../../simulation/animation/types";

import { ANIMATION_RESOURCE_RELEASE_DELAY_FRAMES } from "../constants";
import type { WebGLGeometryHandle } from "./WebGLGeometryRegistry";
import {
	supportsWebGLVertexTextureCount,
	type WebGLVertexTextureUnitLayout,
} from "./WebGLVertexTextureUnits";

export interface WebGLAnimationUniforms {
	animationPayload: WebGLUniformLocation | null;
	morphPositionDeltas: WebGLUniformLocation | null;
	morphNormalDeltas: WebGLUniformLocation | null;
	animationCounts: WebGLUniformLocation | null;
	animationOffsets: WebGLUniformLocation | null;
	animationTextureWidths: WebGLUniformLocation | null;
}

export interface WebGLAnimationPayloadDebugStats {
	readonly entryCount: number;
	readonly textureCount: number;
	readonly liveByteLength: number;
	readonly allocations: number;
	readonly releases: number;
	readonly uploads: number;
	readonly skippedUploads: number;
	readonly rebuilds: number;
	readonly graceReleases: number;
}

interface PayloadEntry {
	texture: WebGLTexture | null;
	width: number;
	height: number;
	jointCapacity: number;
	morphCapacity: number;
	jointCount: number;
	morphCount: number;
	currentJoints: Float32Array;
	previousJoints: Float32Array;
	currentMorph: Float32Array;
	previousMorph: Float32Array;
	payload: Float32Array;
	jointSettlePending: boolean;
	morphSettlePending: boolean;
	preparedFrame: number;
	lastUsedFrame: number;
	lastRevision: number;
	view: WebGLAnimationPayloadView | null;
}

interface WebGLAnimationPayloadView {
	readonly texture: WebGLTexture;
	readonly textureWidth: number;
	readonly jointCount: number;
	readonly morphCount: number;
	readonly currentJointOffset: number;
	readonly previousJointOffset: number;
	readonly currentMorphOffset: number;
	readonly previousMorphOffset: number;
}

/** @internal Owns current/previous WebGL animation textures per draw packet. */
export class WebGLAnimationPayloadPool {
	private readonly _gl: WebGL2RenderingContext;
	private readonly _units: WebGLVertexTextureUnitLayout;
	private readonly _maxTextureSize: number;
	private readonly _entries = new Map<string, PayloadEntry>();
	private _zeroTexture: WebGLTexture | null = null;
	private _jointMap: JointMatrixMap | null = null;
	private _morphMap: MorphWeightMap | null = null;
	private _frame = 0;
	private _allocations = 0;
	private _releases = 0;
	private _uploads = 0;
	private _skippedUploads = 0;
	private _rebuilds = 0;
	private _graceReleases = 0;

	public constructor(
		gl: WebGL2RenderingContext,
		units: WebGLVertexTextureUnitLayout,
		maxTextureSize: number,
	) {
		this._gl = gl;
		this._units = units;
		this._maxTextureSize = Math.max(1, Math.floor(maxTextureSize));
	}

	public beginFrame(context: FrameContext): void {
		this._frame++;
		this._jointMap = context.transient.get(ANIMATION_JOINT_MATRICES_KEY) ?? null;
		this._morphMap = context.transient.get(ANIMATION_MORPH_WEIGHTS_KEY) ?? null;
		for (const [id, entry] of this._entries) {
			const inactiveFrameCount = this._frame - entry.lastUsedFrame;
			if (inactiveFrameCount <= ANIMATION_RESOURCE_RELEASE_DELAY_FRAMES) continue;
			this._releaseEntry(entry);
			this._entries.delete(id);
			this._graceReleases++;
		}
	}

	public bind(
		uniforms: WebGLAnimationUniforms,
		packet: DrawPacket,
		geometry: WebGLGeometryHandle,
	): boolean {
		const requiredVertexTextures =
			(uniforms.animationPayload ? 1 : 0) +
			(uniforms.morphPositionDeltas ? 1 : 0) +
			(uniforms.morphNormalDeltas ? 1 : 0);
		if (requiredVertexTextures === 0) return true;
		if (!supportsWebGLVertexTextureCount(this._units, requiredVertexTextures)) {
			this._warn(
				"webgl-animation-vertex-texture-unavailable",
				`WebGL packet ${packet.submission.id} requires ${requiredVertexTextures} vertex ` +
					"texture units; skipping",
			);
			return false;
		}
		const zeroTexture = this._getZeroTexture();
		if (!zeroTexture) return false;
		const view = this._prepare(packet, geometry);
		if (!view) return false;
		const gl = this._gl;
		if (geometry.skinProfile === "skin4") {
			gl.vertexAttrib4f(9, 0, 0, 0, 0);
			gl.vertexAttrib4f(10, 0, 0, 0, 0);
		}
		if (uniforms.animationPayload) {
			this._bindTexture(this._units.animationPayload, view.texture);
		}
		if (uniforms.morphPositionDeltas) {
			this._bindTexture(
				this._units.morphPosition,
				geometry.morphPositionTexture ?? zeroTexture,
			);
		}
		if (uniforms.morphNormalDeltas) {
			this._bindTexture(
				this._units.morphNormal,
				geometry.morphNormalTexture ?? zeroTexture,
			);
		}
		if (uniforms.animationPayload) {
			gl.uniform1i(uniforms.animationPayload, this._units.animationPayload);
		}
		if (uniforms.morphPositionDeltas) {
			gl.uniform1i(uniforms.morphPositionDeltas, this._units.morphPosition);
		}
		if (uniforms.morphNormalDeltas) {
			gl.uniform1i(uniforms.morphNormalDeltas, this._units.morphNormal);
		}
		if (uniforms.animationCounts) {
			gl.uniform4i(
				uniforms.animationCounts,
				view.jointCount,
				view.morphCount,
				geometry.vertexCount,
				geometry.morphSemanticMask,
			);
		}
		if (uniforms.animationOffsets) {
			gl.uniform4i(
				uniforms.animationOffsets,
				view.currentJointOffset,
				view.previousJointOffset,
				view.currentMorphOffset,
				view.previousMorphOffset,
			);
		}
		if (uniforms.animationTextureWidths) {
			gl.uniform4i(
				uniforms.animationTextureWidths,
				view.textureWidth,
				geometry.morphTextureWidth,
				geometry.morphTextureWidth,
				0,
			);
		}
		return true;
	}

	public getDebugStats(): WebGLAnimationPayloadDebugStats {
		let textureCount = this._zeroTexture ? 1 : 0;
		let liveByteLength = this._zeroTexture ? 16 : 0;
		for (const entry of this._entries.values()) {
			if (!entry.texture) continue;
			textureCount++;
			liveByteLength += entry.width * entry.height * 16;
		}
		return {
			entryCount: this._entries.size,
			textureCount,
			liveByteLength,
			allocations: this._allocations,
			releases: this._releases,
			uploads: this._uploads,
			skippedUploads: this._skippedUploads,
			rebuilds: this._rebuilds,
			graceReleases: this._graceReleases,
		};
	}

	public destroy(): void {
		for (const entry of this._entries.values()) this._releaseEntry(entry);
		this._entries.clear();
		if (this._zeroTexture) this._gl.deleteTexture(this._zeroTexture);
		this._zeroTexture = null;
		this._jointMap = null;
		this._morphMap = null;
	}

	private _prepare(
		packet: DrawPacket,
		geometry: WebGLGeometryHandle,
	): WebGLAnimationPayloadView | null {
		let entry = this._entries.get(packet.submission.id);
		if (!entry) {
			entry = createPayloadEntry();
			this._entries.set(packet.submission.id, entry);
		}
		entry.lastUsedFrame = this._frame;
		if (entry.preparedFrame === this._frame && entry.view) return entry.view;

		const deformation = packet.submission.deformation;
		const runtimeJoint = deformation.jointPayloadKey ?
			this._jointMap?.get(deformation.jointPayloadKey) ?? null
			: null;
		if (deformation.jointPayloadKey && !runtimeJoint) {
			this._warn(
				`webgl-missing-joint-payload-${packet.submission.id}`,
				`WebGL packet ${packet.submission.id} is missing active joint payload; skipping`,
			);
			return null;
		}
		let joints = runtimeJoint?.matrices ?? null;
		const runtimeMorph = deformation.morphPayloadKey ?
			this._morphMap?.get(deformation.morphPayloadKey) ?? null
			: null;
		if (deformation.morphPayloadKey && !runtimeMorph) {
			this._warn(
				`webgl-missing-morph-payload-${packet.submission.id}`,
				`WebGL packet ${packet.submission.id} is missing active morph payload; skipping`,
			);
			return null;
		}
		let morphWeights = runtimeMorph?.weights ?? null;
		let morphCount = runtimeMorph?.targetCount ?? 0;
		const jointCount = Math.floor((joints?.length ?? 0) / 16);
		morphCount = Math.min(
			Math.max(0, morphCount),
			morphWeights?.length ?? 0,
			geometry.morphTargetCount,
		);
		if (jointCount <= 0 && morphCount <= 0) {
			entry.jointCount = 0;
			entry.morphCount = 0;
			entry.jointSettlePending = false;
			entry.morphSettlePending = false;
			entry.preparedFrame = this._frame;
			const zeroTexture = this._getZeroTexture();
			if (!zeroTexture) return null;
			entry.view = createFallbackView(zeroTexture);
			return entry.view;
		}
		const rebuilt = this._ensureCapacity(entry, jointCount, morphCount);
		if (rebuilt === null) {
			this._warn(
				`webgl-animation-payload-overflow-${packet.submission.id}`,
				`WebGL animation payload for packet ${packet.submission.id} exceeds texture limits; skipping`,
			);
			return null;
		}
		const revisionChanged = entry.lastRevision !== (packet.submission.deformation.revision ?? 0);
		const jointChanged = updateHistory(
			entry.currentJoints,
			entry.previousJoints,
			joints,
			jointCount * 16,
			entry.jointCount,
			rebuilt || revisionChanged,
			false,
			entry.jointSettlePending,
		);
		entry.jointSettlePending = jointChanged.settlePending;
		const morphChanged = updateHistory(
			entry.currentMorph,
			entry.previousMorph,
			morphWeights,
			morphCount,
			entry.morphCount,
			rebuilt || revisionChanged,
			false,
			entry.morphSettlePending,
		);
		entry.morphSettlePending = morphChanged.settlePending;
		entry.jointCount = jointCount;
		entry.morphCount = morphCount;
		entry.lastRevision = packet.submission.deformation.revision ?? 0;
		if (rebuilt || jointChanged.changed || morphChanged.changed) {
			this._writePayload(entry);
		} else {
			this._skippedUploads++;
		}
		entry.preparedFrame = this._frame;
		entry.view = createView(entry);
		return entry.view;
	}

	private _ensureCapacity(
		entry: PayloadEntry,
		jointCount: number,
		morphCount: number,
	): boolean | null {
		const jointCapacity = growCapacity(entry.jointCapacity, jointCount);
		const morphCapacity = growCapacity(entry.morphCapacity, morphCount);
		if (
			entry.texture &&
			jointCapacity === entry.jointCapacity &&
			morphCapacity === entry.morphCapacity
		) return false;
		const texelCount = jointCapacity * 8 + morphCapacity * 2;
		if (texelCount > this._maxTextureSize * this._maxTextureSize) return null;
		const width = Math.min(this._maxTextureSize, nextPowerOfTwo(Math.max(1, texelCount)));
		const height = Math.ceil(texelCount / width);
		if (!entry.texture) {
			entry.texture = this._gl.createTexture();
			if (!entry.texture) return null;
			this._allocations++;
		} else {
			this._rebuilds++;
		}
		entry.width = width;
		entry.height = height;
		entry.jointCapacity = jointCapacity;
		entry.morphCapacity = morphCapacity;
		entry.currentJoints = resize(entry.currentJoints, jointCapacity * 16);
		entry.previousJoints = resize(entry.previousJoints, jointCapacity * 16);
		entry.currentMorph = resize(entry.currentMorph, morphCapacity);
		entry.previousMorph = resize(entry.previousMorph, morphCapacity);
		entry.payload = new Float32Array(width * height * 4);
		this._initializeTexture(entry.texture, width, height, null);
		return true;
	}

	private _writePayload(entry: PayloadEntry): void {
		const payload = entry.payload;
		payload.fill(0);
		payload.set(entry.currentJoints.subarray(0, entry.jointCount * 16), 0);
		payload.set(
			entry.previousJoints.subarray(0, entry.jointCount * 16),
			entry.jointCapacity * 16,
		);
		let texelOffset = entry.jointCapacity * 8;
		for (let index = 0; index < entry.morphCount; index++) {
			payload[(texelOffset + index) * 4] = entry.currentMorph[index];
		}
		texelOffset += entry.morphCapacity;
		for (let index = 0; index < entry.morphCount; index++) {
			payload[(texelOffset + index) * 4] = entry.previousMorph[index];
		}
		const gl = this._gl;
		gl.bindTexture(gl.TEXTURE_2D, entry.texture);
		gl.texSubImage2D(
			gl.TEXTURE_2D,
			0,
			0,
			0,
			entry.width,
			entry.height,
			gl.RGBA,
			gl.FLOAT,
			payload,
		);
		gl.bindTexture(gl.TEXTURE_2D, null);
		this._uploads++;
	}

	private _initializeTexture(
		texture: WebGLTexture,
		width: number,
		height: number,
		data: Float32Array | null,
	): void {
		const gl = this._gl;
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
	}

	private _bindTexture(unit: number, texture: WebGLTexture): void {
		const gl = this._gl;
		gl.activeTexture(gl.TEXTURE0 + unit);
		gl.bindTexture(gl.TEXTURE_2D, texture);
	}

	private _getZeroTexture(): WebGLTexture | null {
		if (this._zeroTexture) return this._zeroTexture;
		const texture = this._gl.createTexture?.();
		if (!texture) {
			this._warn(
				"webgl-animation-texture-allocation-failed",
				"WebGL animation fallback texture allocation failed; skipping animated packets",
			);
			return null;
		}
		this._zeroTexture = texture;
		this._initializeTexture(texture, 1, 1, new Float32Array(4));
		return texture;
	}

	private _releaseEntry(entry: PayloadEntry): void {
		if (!entry.texture) return;
		this._gl.deleteTexture(entry.texture);
		entry.texture = null;
		this._releases++;
	}

	private _warn(key: string, message: string): void {
		Logger.warn(`[${key}] ${message}`, {
			scope: "WebGLAnimationPayloadPool",
			onceKey: key,
		});
	}
}

function createPayloadEntry(): PayloadEntry {
	return {
		texture: null,
		width: 1,
		height: 1,
		jointCapacity: 0,
		morphCapacity: 0,
		jointCount: 0,
		morphCount: 0,
		currentJoints: new Float32Array(0),
		previousJoints: new Float32Array(0),
		currentMorph: new Float32Array(0),
		previousMorph: new Float32Array(0),
		payload: new Float32Array(0),
		jointSettlePending: false,
		morphSettlePending: false,
		preparedFrame: -1,
		lastUsedFrame: -1,
		lastRevision: -1,
		view: null,
	};
}

function updateHistory(
	current: Float32Array,
	previous: Float32Array,
	source: Float32Array | null,
	count: number,
	previousCount: number,
	forceChanged: boolean,
	compareSource: boolean,
	settlePending: boolean,
): { changed: boolean; settlePending: boolean } {
	if (!source || count <= 0) {
		return { changed: previousCount > 0, settlePending: false };
	}
	const changed =
		previousCount <= 0 ||
		forceChanged ||
		(compareSource && !prefixEquals(current, source, count));
	if (changed) {
		if (previousCount <= 0) {
			current.set(source.subarray(0, count), 0);
			previous.set(source.subarray(0, count), 0);
			return { changed: true, settlePending: false };
		}
		previous.set(current.subarray(0, Math.min(previousCount, count)), 0);
		current.set(source.subarray(0, count), 0);
		return { changed: true, settlePending: true };
	}
	if (settlePending) {
		previous.set(current.subarray(0, count), 0);
		return { changed: true, settlePending: false };
	}
	return { changed: false, settlePending: false };
}

function createView(entry: PayloadEntry): WebGLAnimationPayloadView {
	const previousJointOffset = entry.jointCapacity * 4;
	const currentMorphOffset = entry.jointCapacity * 8;
	return {
		texture: entry.texture!,
		textureWidth: entry.width,
		jointCount: entry.jointCount,
		morphCount: entry.morphCount,
		currentJointOffset: 0,
		previousJointOffset,
		currentMorphOffset,
		previousMorphOffset: currentMorphOffset + entry.morphCapacity,
	};
}

function createFallbackView(texture: WebGLTexture): WebGLAnimationPayloadView {
	return {
		texture,
		textureWidth: 1,
		jointCount: 0,
		morphCount: 0,
		currentJointOffset: 0,
		previousJointOffset: 0,
		currentMorphOffset: 0,
		previousMorphOffset: 0,
	};
}

function prefixEquals(left: Float32Array, right: Float32Array, count: number): boolean {
	if (left.length < count || right.length < count) return false;
	for (let index = 0; index < count; index++) {
		if (left[index] !== right[index]) return false;
	}
	return true;
}

function resize(source: Float32Array, size: number): Float32Array {
	if (source.length === size) return source;
	const target = new Float32Array(size);
	target.set(source.subarray(0, Math.min(source.length, size)));
	return target;
}

function growCapacity(current: number, required: number): number {
	if (required <= 0) return current;
	let capacity = Math.max(1, current);
	while (capacity < required) capacity *= 2;
	return capacity;
}

function nextPowerOfTwo(value: number): number {
	let result = 1;
	while (result < value) result *= 2;
	return result;
}
