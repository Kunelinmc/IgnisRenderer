import { DEFAULT_PRIMITIVE_DRAW_TOPOLOGY } from "../../core/types";
import { AlphaMode } from "../../materials/Material";
import { isMaterialTransparentPass } from "../../materials/transparency";
import { Matrix4 } from "../../maths/Matrix4";
import { Quaternion } from "../../maths/Quaternion";
import type { Matrix3Arr } from "../../maths/types";
import type { MeshInstance } from "../../meshes";
import type {
	ParticleMeshRenderBatch,
	ParticleMeshRenderItem,
} from "../../particles/ParticleRenderBatch";
import {
	DRAW_PACKET_FLAG_REFLECTIVE,
	DRAW_PACKET_FLAG_SHADOW_CASTER,
	DRAW_PACKET_FLAG_SHADOW_TRANSMITTER,
	DRAW_PACKET_FLAG_TRANSPARENT,
	PARTICLE_MESH_TRANSIENT_BATCHES_KEY,
	type DrawPacket,
} from "../../pipeline/types";
import type {
	FramePacketContributor,
	FramePacketContributorContext,
	FramePacketSink,
} from "../../pipeline/FramePacketContributorRegistry";

/** @internal Produces WebGPU mesh-particle packets for main and probe views. */
export class WebGPUParticleMeshPacketContributor
	implements FramePacketContributor
{
	public readonly id = "webgpu:particle-mesh";

	public supports(context: FramePacketContributorContext): boolean {
		return context.purpose !== "planar-reflection";
	}

	public contribute(
		context: FramePacketContributorContext,
		sink: FramePacketSink,
	): void {
		const batches = context.frameContext.transient.get(
			PARTICLE_MESH_TRANSIENT_BATCHES_KEY,
		);
		if (!batches || batches.length <= 0) {
			return;
		}

		const packets: DrawPacket[] = [];
		for (const batch of batches) {
			for (let particleIndex = 0; particleIndex < batch.particles.length; particleIndex++) {
				const particle = batch.particles[particleIndex];
				const sortDepth = this._resolveSortDepth(context, particle);
				if (sortDepth === null) {
					continue;
				}
				packets.push(
					createParticleMeshPacket(
						batch,
						particle,
						particleIndex,
						sortDepth,
					),
				);
			}
		}
		packets.sort(compareParticleMeshPackets);
		for (const packet of packets) {
			sink.add(packet);
		}
	}

	private _resolveSortDepth(
		context: FramePacketContributorContext,
		particle: ParticleMeshRenderItem,
	): number | null {
		if (context.purpose !== "probe-capture") {
			return particle.depth;
		}
		const cameraSpace = Matrix4.transformPoint(
			context.frameContext.viewCamera.viewMatrix,
			particle.position,
		);
		const depth = -cameraSpace.z;
		return depth > 0 ? depth : null;
	}
}

function createParticleMeshPacket(
	batch: ParticleMeshRenderBatch,
	particle: ParticleMeshRenderItem,
	particleIndex: number,
	sortDepth: number,
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
		sortDepth,
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
