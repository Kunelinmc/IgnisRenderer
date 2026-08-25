import { DEFAULT_PRIMITIVE_DRAW_TOPOLOGY } from "../core/types";
import { AlphaMode } from "../materials/Material";
import { isMaterialTransparentPass } from "../materials/transparency";
import { Matrix4 } from "../maths/Matrix4";
import { Quaternion } from "../maths/Quaternion";
import type { Matrix3Arr } from "../maths/types";
import type {
	ParticleMeshRenderBatch,
	ParticleMeshRenderItem,
} from "../particles/ParticleRenderBatch";
import {
	DRAW_PACKET_FLAG_REFLECTIVE,
	DRAW_PACKET_FLAG_SHADOW_CASTER,
	DRAW_PACKET_FLAG_SHADOW_RECEIVER,
	DRAW_PACKET_FLAG_SHADOW_TRANSMITTER,
	DRAW_PACKET_FLAG_TRANSPARENT,
	PARTICLE_MESH_TRANSIENT_BATCHES_KEY,
	type DrawPacket,
	type DrawGeometryBinding,
	type FrameContext,
} from "./types";
import { defineTransientKey } from "../foundation/TransientStore";

const particleGeometryBindings = new WeakMap<object, DrawGeometryBinding>();

/** @internal Identifies the camera view for which frame packets are prepared. */
export type FramePacketViewPurpose =
	| "main"
	| "probe-capture"
	| "planar-reflection"
	| "render-target-view";

/** @internal Complete draw-packet lists for one prepared frame view. */
export interface PreparedFramePacketSet {
	readonly all: readonly DrawPacket[];
	readonly opaque: readonly DrawPacket[];
	readonly transparent: readonly DrawPacket[];
	readonly shadowCasters: readonly DrawPacket[];
	readonly shadowTransmitters: readonly DrawPacket[];
	readonly reflective: readonly DrawPacket[];
}

/** @internal Creates a packet set containing only a prepared scene's baseline packets. */
export function createBaselineFramePacketSet(
	context: FrameContext,
): PreparedFramePacketSet {
	return {
		all: [
			...context.scene.opaquePackets,
			...context.scene.transparentPackets,
		],
		opaque: context.scene.opaquePackets.slice(),
		transparent: context.scene.transparentPackets.slice(),
		shadowCasters: context.scene.shadowCasterPackets.slice(),
		shadowTransmitters: context.scene.shadowTransmitterPackets.slice(),
		reflective: context.scene.reflectivePackets.slice(),
	};
}

interface PreparedFramePacketCacheRecord {
	readonly scene: FrameContext["scene"];
	readonly viewCamera: FrameContext["viewCamera"];
	readonly packets: PreparedFramePacketSet;
}

const PREPARED_FRAME_PACKET_SET_KEY =
	defineTransientKey<Map<FramePacketViewPurpose, PreparedFramePacketCacheRecord>>(
		"pipeline:prepared-frame-packet-set",
	);

/**
 * Prepares and caches the effective draw-packet lists for one frame view purpose.
 *
 * @param context Current frame context.
 * @param purpose View purpose ("main", "probe-capture", or "planar-reflection"). Defaults to "main".
 * @returns View-local prepared frame packet set.
 * @internal Stateless module function composing baseline scene packets and transient dynamic packets.
 */
export function prepareFramePackets(
	context: FrameContext,
	purpose: FramePacketViewPurpose = "main",
): PreparedFramePacketSet {
	let cacheMap = context.transient.get(PREPARED_FRAME_PACKET_SET_KEY);
	if (!cacheMap) {
		cacheMap = new Map();
		context.transient.set(PREPARED_FRAME_PACKET_SET_KEY, cacheMap);
	}

	const cached = cacheMap.get(purpose);
	if (
		cached &&
		cached.scene === context.scene &&
		cached.viewCamera === context.viewCamera
	) {
		return cached.packets;
	}

	const all: DrawPacket[] = [
		...context.scene.opaquePackets,
		...context.scene.transparentPackets,
	];
	const opaque = context.scene.opaquePackets.slice();
	const transparent = context.scene.transparentPackets.slice();
	const shadowCasters = context.scene.shadowCasterPackets.slice();
	const shadowTransmitters = context.scene.shadowTransmitterPackets.slice();
	const reflective = context.scene.reflectivePackets.slice();

	if (purpose !== "planar-reflection") {
		const particleMeshBatches = context.transient.get(
			PARTICLE_MESH_TRANSIENT_BATCHES_KEY,
		);
		if (particleMeshBatches && particleMeshBatches.length > 0) {
			const particlePackets: DrawPacket[] = [];
			for (const batch of particleMeshBatches) {
				for (
					let particleIndex = 0;
					particleIndex < batch.particles.length;
					particleIndex++
				) {
					const particle = batch.particles[particleIndex];
					const sortDepth = resolveParticleMeshSortDepth(
						context,
						particle,
						purpose,
					);
					if (sortDepth === null) {
						continue;
					}
					particlePackets.push(
						createParticleMeshPacket(
							batch,
							particle,
							particleIndex,
							sortDepth,
						),
					);
				}
			}
			particlePackets.sort(compareParticleMeshPackets);
			for (const packet of particlePackets) {
				all.push(packet);
				const flags = packet.submission.passFlags;
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
		}
	}

	const packets: PreparedFramePacketSet = {
		all,
		opaque,
		transparent,
		shadowCasters,
		shadowTransmitters,
		reflective,
	};
	cacheMap.set(purpose, {
		scene: context.scene,
		viewCamera: context.viewCamera,
		packets,
	});
	return packets;
}

function resolveParticleMeshSortDepth(
	context: FrameContext,
	particle: ParticleMeshRenderItem,
	purpose: FramePacketViewPurpose,
): number | null {
	if (purpose !== "probe-capture") {
		return particle.depth;
	}
	const cameraSpace = Matrix4.transformPoint(
		context.viewCamera.viewMatrix,
		particle.position,
	);
	const depth = -cameraSpace.z;
	return depth > 0 ? depth : null;
}

function createParticleMeshPacket(
	batch: ParticleMeshRenderBatch,
	particle: ParticleMeshRenderItem,
	particleIndex: number,
	sortDepth: number,
): DrawPacket {
	const material = batch.material;
	const isTransparent = isMaterialTransparentPass(material);
	const isReflective =
		material.reflectivity > 0 && material.mirrorPlane !== null;
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
	const id = [
		"particleMesh",
		batch.systemId,
		batch.templateIndex,
		batch.primitive.id,
		particleIndex,
	].join(":");

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
	if (batch.primitive.receiveShadows !== false) {
		passFlags |= DRAW_PACKET_FLAG_SHADOW_RECEIVER;
	}

	return {
		submission: {
			id,
			source: {
				kind: "particle-mesh",
				systemId: batch.systemId,
				templateIndex: batch.templateIndex,
				particleIndex,
			},
			geometry: resolveParticleGeometryBinding(batch),
			instance: {
				renderLayers: 1,
				worldMatrix: currentWorldMatrix,
				previousWorldMatrix,
				normalMatrix,
			},
			material: {
				effective: material,
				revision: material.revision,
				pipelineKey: [
					material.type,
					material.shading,
					material.alphaMode ?? AlphaMode.Opaque,
					material.doubleSided ? "double" : "single",
					material.depthWrite ? "depth-write" : "depth-read",
				].join(":"),
			},
			deformation: {
				mode: "none",
				revision: 0,
				jointPayloadKey: null,
				morphPayloadKey: null,
			},
			worldBounds: {
				center: {
					x: worldCenter.x,
					y: worldCenter.y,
					z: worldCenter.z,
				},
				radius:
					batch.primitive.boundingSphere.radius *
					Math.max(0.001, particle.size),
			},
			passFlags,
		},
		sortDepth,
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

function compareParticleMeshPackets(
	left: DrawPacket,
	right: DrawPacket,
): number {
	const leftTransparent =
		(left.submission.passFlags & DRAW_PACKET_FLAG_TRANSPARENT) !== 0;
	const rightTransparent =
		(right.submission.passFlags & DRAW_PACKET_FLAG_TRANSPARENT) !== 0;
	if (leftTransparent !== rightTransparent) return leftTransparent ? 1 : -1;
	if (leftTransparent && left.sortDepth !== right.sortDepth) {
		return right.sortDepth - left.sortDepth;
	}
	if (!leftTransparent) {
		const keyCompare = left.submission.material.pipelineKey.localeCompare(
			right.submission.material.pipelineKey,
		);
		if (keyCompare !== 0) return keyCompare;
		if (left.submission.material.effective !== right.submission.material.effective) {
			return left.submission.material.effective.name.localeCompare(
				right.submission.material.effective.name,
			);
		}
	}
	return left.submission.id.localeCompare(right.submission.id);
}

function resolveParticleGeometryBinding(
	batch: ParticleMeshRenderBatch,
): DrawGeometryBinding {
	const primitive = batch.primitive;
	const topology = primitive.topology ?? DEFAULT_PRIMITIVE_DRAW_TOPOLOGY;
	const cached = particleGeometryBindings.get(primitive);
	if (
		cached &&
		cached.data === primitive.geometry &&
		cached.version === (primitive.geometryVersion ?? 0) &&
		cached.topology === topology
	) {
		return cached;
	}
	const binding: DrawGeometryBinding = {
		resourceKey: primitive,
		id: primitive.id,
		data: primitive.geometry,
		version: primitive.geometryVersion ?? 0,
		topology,
	};
	particleGeometryBindings.set(primitive, binding);
	return binding;
}
