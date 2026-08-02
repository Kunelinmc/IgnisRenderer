import { DEFAULT_PRIMITIVE_DRAW_TOPOLOGY } from "../../core/types";
import { AlphaMode } from "../../materials/Material";
import { isMaterialTransparentPass } from "../../materials/transparency";
import { Matrix4 } from "../../maths/Matrix4";
import { Quaternion } from "../../maths/Quaternion";
import type { Matrix3Arr } from "../../maths/types";
import type { MeshInstance } from "../../meshes";
import {
	DRAW_PACKET_FLAG_REFLECTIVE,
	DRAW_PACKET_FLAG_SHADOW_CASTER,
	DRAW_PACKET_FLAG_SHADOW_TRANSMITTER,
	DRAW_PACKET_FLAG_TRANSPARENT,
	PARTICLE_MESH_TRANSIENT_BATCHES_KEY,
	defineTransientKey,
} from "../../pipeline/types";
import type {
	DrawPacket,
	FrameContext,
	ParticleMeshRenderBatch,
	ParticleMeshRenderItem,
} from "../../pipeline/types";

/** @internal Current-view mesh-particle packets shared by WebGPU frame consumers. */
export interface WebGPUParticleMeshFramePackets {
	readonly all: readonly DrawPacket[];
	readonly opaque: readonly DrawPacket[];
	readonly transparent: readonly DrawPacket[];
	readonly shadowCasters: readonly DrawPacket[];
	readonly shadowTransmitters: readonly DrawPacket[];
	readonly reflective: readonly DrawPacket[];
}

export const WEBGPU_PARTICLE_MESH_FRAME_PACKETS_KEY =
	defineTransientKey<WebGPUParticleMeshFramePackets>(
		"webgpu:particle-mesh-frame-packets",
	);

const UNPREPARED_PARTICLE_MESH_FRAME_PACKETS: WebGPUParticleMeshFramePackets =
	Object.freeze({
		all: Object.freeze([]),
		opaque: Object.freeze([]),
		transparent: Object.freeze([]),
		shadowCasters: Object.freeze([]),
		shadowTransmitters: Object.freeze([]),
		reflective: Object.freeze([]),
	});

/**
 * Prepares the active view's mesh-particle packets once after simulation.
 *
 * @internal Owned by WebGPU frame and capture preparation. Callers must use a
 * distinct transient store for each view and frame.
 */
export function prepareWebGPUParticleMeshFramePackets(
	context: FrameContext,
): WebGPUParticleMeshFramePackets {
	const prepared = context.transient.get(WEBGPU_PARTICLE_MESH_FRAME_PACKETS_KEY);
	if (prepared) return prepared;

	const batches = context.transient.get(PARTICLE_MESH_TRANSIENT_BATCHES_KEY);
	if (!batches || batches.length === 0) {
		const empty = createEmptyParticleMeshFramePackets();
		context.transient.set(
			WEBGPU_PARTICLE_MESH_FRAME_PACKETS_KEY,
			empty,
		);
		return empty;
	}

	const all: DrawPacket[] = [];
	for (const batch of batches) {
		for (let particleIndex = 0; particleIndex < batch.particles.length; particleIndex++) {
			all.push(
				createParticleMeshPacket(
					batch,
					batch.particles[particleIndex],
					particleIndex,
				),
			);
		}
	}
	all.sort(compareParticleMeshPackets);

	const opaque: DrawPacket[] = [];
	const transparent: DrawPacket[] = [];
	const shadowCasters: DrawPacket[] = [];
	const shadowTransmitters: DrawPacket[] = [];
	const reflective: DrawPacket[] = [];
	for (const packet of all) {
		const flags = packet.passFlags;
		if ((flags & DRAW_PACKET_FLAG_TRANSPARENT) !== 0) {
			transparent.push(packet);
		} else {
			opaque.push(packet);
		}
		if ((flags & DRAW_PACKET_FLAG_SHADOW_CASTER) !== 0) {
			shadowCasters.push(packet);
		}
		if ((flags & DRAW_PACKET_FLAG_SHADOW_TRANSMITTER) !== 0) {
			shadowTransmitters.push(packet);
		}
		if ((flags & DRAW_PACKET_FLAG_REFLECTIVE) !== 0) {
			reflective.push(packet);
		}
	}

	const result: WebGPUParticleMeshFramePackets = {
		all,
		opaque,
		transparent,
		shadowCasters,
		shadowTransmitters,
		reflective,
	};
	context.transient.set(WEBGPU_PARTICLE_MESH_FRAME_PACKETS_KEY, result);
	return result;
}

/** @internal Returns packets prepared for the active WebGPU frame view. */
export function getWebGPUParticleMeshFramePackets(
	context: FrameContext,
): WebGPUParticleMeshFramePackets {
	return (
		context.transient.get(WEBGPU_PARTICLE_MESH_FRAME_PACKETS_KEY) ??
		UNPREPARED_PARTICLE_MESH_FRAME_PACKETS
	);
}

function createEmptyParticleMeshFramePackets(): WebGPUParticleMeshFramePackets {
	return {
		all: [],
		opaque: [],
		transparent: [],
		shadowCasters: [],
		shadowTransmitters: [],
		reflective: [],
	};
}

function createParticleMeshPacket(
	batch: ParticleMeshRenderBatch,
	particle: ParticleMeshRenderItem,
	particleIndex: number,
): DrawPacket {
	const material = batch.material;
	const isTransparent = isMaterialTransparentPass(material);
	const isReflective = material.reflectivity > 0 && material.mirrorPlane !== null;
	const supportsShadowCasting =
		(batch.primitive.topology ?? DEFAULT_PRIMITIVE_DRAW_TOPOLOGY) ===
		DEFAULT_PRIMITIVE_DRAW_TOPOLOGY;
	const currentWorldMatrix = createParticleMeshWorldMatrix(
		particle.position,
		particle.rotation,
		particle.size,
	);
	const previousWorldMatrix = createParticleMeshWorldMatrix(
		particle.previousPosition,
		particle.previousRotation,
		particle.size,
	);
	const normalMatrix = Matrix4.normalMatrix(currentWorldMatrix) as Matrix3Arr;
	const worldCenter = Matrix4.transformPoint(
		currentWorldMatrix,
		batch.primitive.boundingSphere.center,
	);
	const meshInstance = createParticleMeshInstance(batch, currentWorldMatrix);

	let passFlags = 0;
	if (isTransparent) {
		passFlags |= DRAW_PACKET_FLAG_TRANSPARENT;
		if (batch.castShadows && supportsShadowCasting) {
			passFlags |= DRAW_PACKET_FLAG_SHADOW_TRANSMITTER;
		}
	} else if (batch.castShadows && supportsShadowCasting) {
		passFlags |= DRAW_PACKET_FLAG_SHADOW_CASTER;
	}
	if (isReflective) {
		passFlags |= DRAW_PACKET_FLAG_REFLECTIVE;
	}

	return {
		id: [
			"particleMesh",
			batch.systemId,
			batch.templateIndex,
			batch.primitive.id,
			particleIndex,
		].join(":"),
		meshInstance,
		mesh: batch.mesh,
		primitive: batch.primitive,
		material,
		geometry: batch.primitive.geometry,
		worldMatrix: currentWorldMatrix,
		previousWorldMatrix,
		normalMatrix,
		worldBounds: {
			center: {
				x: worldCenter.x,
				y: worldCenter.y,
				z: worldCenter.z,
			},
			radius: batch.primitive.boundingSphere.radius * Math.max(0.001, particle.size),
		},
		sortDepth: particle.depth,
		pipelineKey: [
			material.type,
			material.shading,
			material.alphaMode ?? AlphaMode.Opaque,
			material.doubleSided ? "double" : "single",
			material.depthWrite ? "depth-write" : "depth-read",
		].join(":"),
		passFlags,
	};
}

function createParticleMeshWorldMatrix(
	position: { x: number; y: number; z: number },
	rotation: number,
	size: number,
): Matrix4 {
	const scale = Math.max(0.001, size);
	return Matrix4.compose(position, Quaternion.fromEuler(0, 0, rotation), {
		x: scale,
		y: scale,
		z: scale,
	});
}

function createParticleMeshInstance(
	batch: ParticleMeshRenderBatch,
	worldMatrix: Matrix4,
): MeshInstance {
	return {
		id: `particleMeshInstance:${batch.systemId}:${batch.templateIndex}`,
		mesh: batch.mesh,
		skeleton: null,
		morphWeights: batch.mesh.defaultMorphWeights,
		renderLayers: 1,
		worldMatrix,
	} as MeshInstance;
}

function compareParticleMeshPackets(left: DrawPacket, right: DrawPacket): number {
	const leftTransparent = (left.passFlags & DRAW_PACKET_FLAG_TRANSPARENT) !== 0;
	const rightTransparent = (right.passFlags & DRAW_PACKET_FLAG_TRANSPARENT) !== 0;
	if (leftTransparent !== rightTransparent) return leftTransparent ? 1 : -1;
	if (leftTransparent && left.sortDepth !== right.sortDepth) {
		return right.sortDepth - left.sortDepth;
	}
	if (!leftTransparent) {
		const keyCompare = left.pipelineKey.localeCompare(right.pipelineKey);
		if (keyCompare !== 0) return keyCompare;
		if (left.material !== right.material) {
			return left.material.name.localeCompare(right.material.name);
		}
	}
	return left.id.localeCompare(right.id);
}
