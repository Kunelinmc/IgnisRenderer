import type { DrawPacket } from "../../pipeline/types";
import { ShaderMaterial } from "../../materials/ShaderMaterial";
import {
	isMaterialTransparentPass,
	materialUsesTransmission,
} from "../../materials/transparency";
import { Matrix4 } from "../../maths/Matrix4";
import type { Matrix3Arr } from "../../maths/types";
import {
	BufferUsage,
	type IBindingGroup,
	type IRenderBuffer,
	type IRenderPipeline,
} from "../types";
import {
	WEBGPU_MODEL_BINDING_ANIMATION_PARAMS,
	WEBGPU_MODEL_BINDING_JOINT_MATRICES,
	WEBGPU_MODEL_BINDING_MORPH_NORMAL,
	WEBGPU_MODEL_BINDING_MORPH_POSITION,
	WEBGPU_MODEL_BINDING_MORPH_WEIGHTS,
	WEBGPU_MODEL_BINDING_SHADER_UNIFORMS,
	WEBGPU_MODEL_BINDING_STATIC_INSTANCES,
	WEBGPU_TEXTURE_DEDICATED_SAMPLER_SLOT_COUNT,
} from "./constants";
import { writeModelUniformData, createModelUniformWriter } from "./packing";
import type { WebGPUAnimationPayloadPool } from "./WebGPUAnimationPayloadPool";
import type { WebGPUDeviceResourceHost } from "./WebGPUDeviceResourceHost";
import type { WebGPUGeometryHandle } from "./WebGPUGeometryRegistry";
import type { WebGPUPipelineLayouts } from "./WebGPUPipelineLayouts";
import type { WebGPUScenePipelineDrawMode } from "./WebGPUPipelineLibrary";
import type { WebGPUResolvedMaterialSnapshot } from "./WebGPUMaterialSnapshotCache";
import type { WebGPUMaterialUniformData } from "./types";

const STATIC_INSTANCE_FLOATS = 52;
const STATIC_INSTANCE_INITIAL_CAPACITY = 256;
const STATIC_MATERIAL_BINDING_LIMIT = 4096;
const STATIC_HISTORY_LIMIT = 65_536;
const IDENTITY_MATRIX = Matrix4.identity();

interface StaticMaterialBindingEntry {
	readonly uniformBuffer: IRenderBuffer;
	readonly bindingGroup: IBindingGroup;
}

interface StaticHistoryEntry {
	readonly matrix: Float32Array;
}

/** @internal Static scene draw state backed by the frame instance arena. */
export interface WebGPUStaticDrawState {
	readonly modelBinding: IBindingGroup;
	readonly firstInstance: number;
	readonly batchKey: string;
}

/** @internal Owns static instance records and per-material batch bindings. */
export class WebGPUStaticMeshBatcher {
	private _instanceBuffer: IRenderBuffer;
	private readonly _fallbackUniformBuffer: IRenderBuffer;
	private _instanceData: Float32Array;
	private _capacity = STATIC_INSTANCE_INITIAL_CAPACITY;
	private _count = 0;
	private _packetIndices = new Map<DrawPacket, number>();
	private _pendingHistory = new Map<string, number>();
	private readonly _history = new Map<string, StaticHistoryEntry>();
	private readonly _materialBindings = new Map<
		WebGPUMaterialUniformData,
		StaticMaterialBindingEntry
	>();
	private readonly _objectIds = new WeakMap<object, number>();
	private _nextObjectId = 1;
	private _reserved = false;

	public constructor(
		private readonly _backend: WebGPUDeviceResourceHost,
		private readonly _layouts: WebGPUPipelineLayouts,
		private readonly _animations: WebGPUAnimationPayloadPool,
	) {
		this._instanceData = new Float32Array(this._capacity * STATIC_INSTANCE_FLOATS);
		this._instanceBuffer = this._createInstanceBuffer(this._capacity);
		this._fallbackUniformBuffer = this._backend.createBuffer({
			size: 16,
			usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			label: "WebGPUStaticBatchFallbackUniform",
		});
		this._backend.writeBuffer(this._fallbackUniformBuffer, new Float32Array(4));
	}

	public beginFrame(): void {
		this._count = 0;
		this._packetIndices = new Map();
		this._pendingHistory = new Map();
		this._reserved = false;
	}

	public preparePackets(packets: readonly DrawPacket[]): void {
		if (this._reserved) return;
		this._reserved = true;
		let eligibleCount = 0;
		for (const packet of packets) {
			if (this._isPotentiallyEligible(packet)) eligibleCount++;
		}
		this._ensureCapacity(eligibleCount);
		for (const packet of packets) {
			if (this._isPotentiallyEligible(packet)) this._appendPacket(packet);
		}
		if (this._count > 0) {
			this._backend.writeBuffer(
				this._instanceBuffer,
				this._instanceData.subarray(0, this._count * STATIC_INSTANCE_FLOATS) as any,
			);
		}
	}

	public getDrawState(
		packet: DrawPacket,
		pipeline: IRenderPipeline,
		geometry: WebGPUGeometryHandle,
		snapshot: WebGPUResolvedMaterialSnapshot,
		drawMode: WebGPUScenePipelineDrawMode,
	): WebGPUStaticDrawState | null {
		if (!this._isEligible(packet, geometry, drawMode)) return null;
		let firstInstance = this._packetIndices.get(packet);
		if (firstInstance === undefined) {
			this._ensureCapacity(this._count + 1);
			firstInstance = this._appendPacket(packet);
			this._backend.writeBuffer(
				this._instanceBuffer,
				this._instanceData.subarray(
					firstInstance * STATIC_INSTANCE_FLOATS,
					(firstInstance + 1) * STATIC_INSTANCE_FLOATS,
				) as any,
				firstInstance * STATIC_INSTANCE_FLOATS * 4,
			);
		}
		const modelBinding = this._getMaterialBinding(snapshot);
		const batchKey = [
			this._getObjectId(pipeline as object),
			this._getObjectId(geometry.indexBuffer as object),
			geometry.indexFormat,
			geometry.layoutKey,
			this._getObjectId(snapshot.data as object),
		].join("|");
		return { modelBinding, firstInstance, batchKey };
	}

	public commitFrame(): void {
		for (const [packetId, instanceIndex] of this._pendingHistory) {
			const offset = instanceIndex * STATIC_INSTANCE_FLOATS;
			let entry = this._history.get(packetId);
			if (!entry) entry = { matrix: new Float32Array(16) };
			entry.matrix.set(this._instanceData.subarray(offset, offset + 16));
			this._history.delete(packetId);
			this._history.set(packetId, entry);
		}
		while (this._history.size > STATIC_HISTORY_LIMIT) {
			const oldest = this._history.keys().next().value as string | undefined;
			if (!oldest) break;
			this._history.delete(oldest);
		}
		this._pendingHistory.clear();
	}

	public abortFrame(): void {
		this._pendingHistory.clear();
	}

	public getDebugStats(): {
		readonly instanceCount: number;
		readonly instanceCapacity: number;
		readonly materialBindings: number;
		readonly historyEntries: number;
	} {
		return {
			instanceCount: this._count,
			instanceCapacity: this._capacity,
			materialBindings: this._materialBindings.size,
			historyEntries: this._history.size,
		};
	}

	public destroy(): void {
		this._destroyMaterialBindings();
		this._instanceBuffer.destroy();
		this._fallbackUniformBuffer.destroy();
		this._history.clear();
		this._packetIndices.clear();
		this._pendingHistory.clear();
	}

	private _isEligible(
		packet: DrawPacket,
		geometry: WebGPUGeometryHandle,
		drawMode: WebGPUScenePipelineDrawMode,
	): boolean {
		return !(
			drawMode === "reflection-capture" ||
			drawMode === "planar-reflection-composite" ||
			!this._isPotentiallyEligible(packet) ||
			geometry.skinProfile !== "static" ||
			geometry.morphTargetCount > 0
		);
	}

	private _isPotentiallyEligible(packet: DrawPacket): boolean {
		const material = packet.material;
		const geometry = packet.primitive.geometry;
		return !(
			material instanceof ShaderMaterial ||
			isMaterialTransparentPass(material) ||
			materialUsesTransmission(material) ||
			material.wireframe ||
			material.reflectivity > 0 ||
			material.mirrorPlane !== null ||
			packet.meshInstance.skeleton ||
			(geometry.morphTargets?.length ?? 0) > 0 ||
			geometry.joints0 ||
			geometry.joints1 ||
			geometry.weights0 ||
			geometry.weights1
		);
	}

	private _appendPacket(packet: DrawPacket): number {
		const index = this._count++;
		this._packetIndices.set(packet, index);
		const offset = index * STATIC_INSTANCE_FLOATS;
		writeMatrix(this._instanceData, offset, packet.worldMatrix);
		if (packet.previousWorldMatrix) {
			writeMatrix(this._instanceData, offset + 16, packet.previousWorldMatrix);
		} else {
			const previous = this._history.get(packet.id)?.matrix ?? null;
			if (previous) {
				this._instanceData.set(previous, offset + 16);
			} else {
				this._instanceData.copyWithin(offset + 16, offset, offset + 16);
			}
		}
		writeNormalMatrix(this._instanceData, offset + 32, packet.normalMatrix);
		this._instanceData[offset + 48] = packet.meshInstance.renderLayers >>> 0;
		this._instanceData[offset + 49] = packet.primitive.receiveShadows === false ? 0 : 1;
		this._instanceData[offset + 50] = 0;
		this._instanceData[offset + 51] = 0;
		this._pendingHistory.set(packet.id, index);
		return index;
	}

	private _getMaterialBinding(
		snapshot: WebGPUResolvedMaterialSnapshot,
	): IBindingGroup {
		const cached = this._materialBindings.get(snapshot.data);
		if (cached) {
			this._materialBindings.delete(snapshot.data);
			this._materialBindings.set(snapshot.data, cached);
			return cached.bindingGroup;
		}
		const uniformBuffer = this._backend.createBuffer({
			size: writeModelUniformData(
				createModelUniformWriter(),
				IDENTITY_MATRIX,
				IDENTITY_MATRIX,
				snapshot.data,
				IDENTITY_MATRIX,
				1,
				true,
				true,
			).byteLength,
			usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			label: "WebGPUStaticBatchMaterialUniform",
		});
		const uniformData = writeModelUniformData(
			createModelUniformWriter(),
			IDENTITY_MATRIX,
			IDENTITY_MATRIX,
			snapshot.data,
			IDENTITY_MATRIX,
			1,
			true,
			true,
		);
		this._backend.writeBuffer(uniformBuffer, uniformData);
		const animation = this._animations.getStaticScenePayload();
		const fallbackStorage = this._animations.getFallbackStorageBuffer();
		const entries: Array<{ binding: number; resource: any }> = [
			{ binding: 0, resource: uniformBuffer },
		];
		for (let index = 0; index < snapshot.textures.length; index++) {
			entries.push({ binding: 1 + index * 2, resource: snapshot.textures[index] });
			if (index < WEBGPU_TEXTURE_DEDICATED_SAMPLER_SLOT_COUNT) {
				entries.push({ binding: 2 + index * 2, resource: snapshot.samplers[index] });
			}
		}
		entries.push(
			{ binding: WEBGPU_MODEL_BINDING_ANIMATION_PARAMS, resource: animation.paramsBuffer },
			{ binding: WEBGPU_MODEL_BINDING_JOINT_MATRICES, resource: fallbackStorage },
			{ binding: WEBGPU_MODEL_BINDING_MORPH_WEIGHTS, resource: fallbackStorage },
			{ binding: WEBGPU_MODEL_BINDING_MORPH_POSITION, resource: fallbackStorage },
			{ binding: WEBGPU_MODEL_BINDING_MORPH_NORMAL, resource: fallbackStorage },
			{
				binding: WEBGPU_MODEL_BINDING_SHADER_UNIFORMS,
				resource: this._fallbackUniformBuffer,
			},
			{ binding: WEBGPU_MODEL_BINDING_STATIC_INSTANCES, resource: this._instanceBuffer },
		);
		const bindingGroup = this._backend.createBindingGroup({
			label: "WebGPUStaticBatchMaterialBinding",
			layout: this._layouts.modelBindGroupLayout,
			entries,
			cache: false,
		});
		this._materialBindings.set(snapshot.data, { uniformBuffer, bindingGroup });
		while (this._materialBindings.size > STATIC_MATERIAL_BINDING_LIMIT) {
			const oldest = this._materialBindings.entries().next().value as
				| [WebGPUMaterialUniformData, StaticMaterialBindingEntry]
				| undefined;
			if (!oldest) break;
			this._materialBindings.delete(oldest[0]);
			oldest[1].uniformBuffer.destroy();
		}
		return bindingGroup;
	}

	private _ensureCapacity(required: number): void {
		if (required <= this._capacity) return;
		let nextCapacity = this._capacity;
		while (nextCapacity < required) nextCapacity *= 2;
		const nextData = new Float32Array(nextCapacity * STATIC_INSTANCE_FLOATS);
		nextData.set(this._instanceData.subarray(0, this._count * STATIC_INSTANCE_FLOATS));
		this._instanceData = nextData;
		this._capacity = nextCapacity;
		this._instanceBuffer.destroy();
		this._instanceBuffer = this._createInstanceBuffer(nextCapacity);
		this._destroyMaterialBindings();
	}

	private _createInstanceBuffer(capacity: number): IRenderBuffer {
		return this._backend.createBuffer({
			size: capacity * STATIC_INSTANCE_FLOATS * 4,
			usage: BufferUsage.Storage | BufferUsage.CopyDst,
			label: "WebGPUStaticInstanceArena",
		});
	}

	private _destroyMaterialBindings(): void {
		for (const entry of this._materialBindings.values()) {
			entry.uniformBuffer.destroy();
			(entry.bindingGroup as { destroy?: () => void }).destroy?.();
		}
		this._materialBindings.clear();
	}

	private _getObjectId(value: object): number {
		let id = this._objectIds.get(value);
		if (id !== undefined) return id;
		id = this._nextObjectId++;
		this._objectIds.set(value, id);
		return id;
	}
}

function writeMatrix(target: Float32Array, offset: number, matrix: Matrix4): void {
	const source = matrix.elements;
	for (let column = 0; column < 4; column++) {
		for (let row = 0; row < 4; row++) {
			target[offset + column * 4 + row] = source[row][column];
		}
	}
}

function writeNormalMatrix(
	target: Float32Array,
	offset: number,
	matrix: Matrix4 | Matrix3Arr,
): void {
	const source = matrix instanceof Matrix4 ? matrix.elements : matrix;
	for (let column = 0; column < 4; column++) {
		for (let row = 0; row < 4; row++) {
			target[offset + column * 4 + row] =
				row < 3 && column < 3 ? source[row][column] : (row === column ? 1 : 0);
		}
	}
}
