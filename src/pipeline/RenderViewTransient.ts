import type { Camera } from "../cameras/Camera";
import { createTransientStore, type TransientStore } from "../foundation/TransientStore";
import { Matrix4 } from "../maths/Matrix4";
import type {
	ParticleRenderBatch,
	ParticleRenderItem,
} from "../particles/ParticleRenderBatch";
import {
	PARTICLE_MESH_TRANSIENT_BATCHES_KEY,
	PARTICLE_TRANSIENT_BATCHES_KEY,
} from "./types";

/** Creates view-local transient particle ordering without mutating main-view work. */
export function createRenderViewTransient(
	source: TransientStore,
	camera: Camera,
): TransientStore {
	const target = createTransientStore(source);
	const sourceBatches = source.get(PARTICLE_TRANSIENT_BATCHES_KEY) ?? [];
	const batches: ParticleRenderBatch[] = [];
	for (const batch of sourceBatches) {
		const particles: ParticleRenderItem[] = [];
		for (const particle of batch.particles) {
			const cameraSpace = Matrix4.transformPoint(camera.viewMatrix, particle.position);
			const depth = -cameraSpace.z;
			if (depth <= 0) continue;
			particles.push({
				position: { ...particle.position },
				size: particle.size,
				color: { ...particle.color },
				rotation: particle.rotation,
				depth,
				uvRect: { ...particle.uvRect },
			});
		}
		particles.sort((left, right) => right.depth - left.depth);
		if (particles.length === 0) continue;
		batches.push({
			systemId: batch.systemId,
			blendMode: batch.blendMode,
			texture: batch.texture,
			receiveShadows: batch.receiveShadows,
			castShadows: batch.castShadows,
			shadowDensity: batch.shadowDensity,
			shadowSoftness: batch.shadowSoftness,
			particles,
		});
	}
	target.set(PARTICLE_TRANSIENT_BATCHES_KEY, batches);
	target.set(
		PARTICLE_MESH_TRANSIENT_BATCHES_KEY,
		source.get(PARTICLE_MESH_TRANSIENT_BATCHES_KEY) ?? [],
	);
	return target;
}
