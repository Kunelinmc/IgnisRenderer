import {
	ParticleBlendMode,
	type ParticleTemplate,
	type ParticleSystem,
} from "../particles";
import type { ShadowMap } from "../lights/shadows/ShadowMapping";
import type { IVector3 } from "../maths/types";
import { Matrix4 } from "../maths/Matrix4";
import type { ParticleRenderBatch } from "./types";

export interface ParticleShadowVolumeResolution {
	width: number;
	height: number;
	depth: number;
}

export interface ParticleShadowVolumeGrid {
	resolution: ParticleShadowVolumeResolution;
	density: Float32Array;
	active: boolean;
}

export interface ParticleShadowVolumeDescriptor {
	lightIndex: number;
	sliceIndex: number;
	resolution: ParticleShadowVolumeResolution;
	densityScale: number;
	softnessScale: number;
}

export interface ParticleShadowVolumeBounds {
	center: IVector3;
	radius: number;
}

export const DEFAULT_PARTICLE_SHADOW_VOLUME_RESOLUTION:
	ParticleShadowVolumeResolution = Object.freeze({
		width: 64,
		height: 64,
		depth: 32,
	});

const MIN_PARTICLE_SHADOW_ALPHA = 1e-4;
const MIN_PARTICLE_SHADOW_SIZE = 1e-4;
const MIN_PARTICLE_SHADOW_DENSITY = 1e-6;

export function isParticleShadowCastingBatch(
	batch: Pick<ParticleRenderBatch, "blendMode" | "castShadows" | "shadowDensity">
): boolean {
	return (
		batch.castShadows === true &&
		batch.blendMode !== ParticleBlendMode.Additive &&
		batch.shadowDensity > MIN_PARTICLE_SHADOW_DENSITY
	);
}

export function hasParticleShadowCastingBatches(
	batches: readonly ParticleRenderBatch[] | null | undefined
): boolean {
	if (!batches) return false;
	return batches.some(
		(batch) =>
			isParticleShadowCastingBatch(batch) &&
			batch.particles.some(
				(particle) =>
					particle.size > MIN_PARTICLE_SHADOW_SIZE &&
					particle.color.a > MIN_PARTICLE_SHADOW_ALPHA
			)
	);
}

export function hasParticleShadowCasters(
	systems: readonly ParticleSystem[] | null | undefined
): boolean {
	if (!systems) return false;
	return systems.some(
		(system) =>
			system.visible !== false &&
			hasParticleSystemShadowCaster(system)
	);
}

export function resolveParticleShadowCasterBounds(
	systems: readonly ParticleSystem[] | null | undefined
): ParticleShadowVolumeBounds | null {
	if (!systems || systems.length === 0) {
		return null;
	}

	let minX = Infinity;
	let minY = Infinity;
	let minZ = Infinity;
	let maxX = -Infinity;
	let maxY = -Infinity;
	let maxZ = -Infinity;

	for (const system of systems) {
		if (
			system.visible === false ||
			!hasParticleSystemShadowCaster(system)
		) {
			continue;
		}
		system.updateWorldMatrix(system.parent?.worldMatrix);
		const center = system.getWorldPosition();
		const radius = estimateParticleSystemShadowRadius(system);
		if (!Number.isFinite(radius) || radius <= 0) {
			continue;
		}
		minX = Math.min(minX, center.x - radius);
		minY = Math.min(minY, center.y - radius);
		minZ = Math.min(minZ, center.z - radius);
		maxX = Math.max(maxX, center.x + radius);
		maxY = Math.max(maxY, center.y + radius);
		maxZ = Math.max(maxZ, center.z + radius);
	}

	if (!Number.isFinite(minX) || !Number.isFinite(maxX)) {
		return null;
	}

	const center = {
		x: (minX + maxX) * 0.5,
		y: (minY + maxY) * 0.5,
		z: (minZ + maxZ) * 0.5,
	};
	const sizeX = maxX - minX;
	const sizeY = maxY - minY;
	const sizeZ = maxZ - minZ;
	const radius = Math.sqrt(sizeX * sizeX + sizeY * sizeY + sizeZ * sizeZ) * 0.5;
	if (!Number.isFinite(radius) || radius <= 0) {
		return null;
	}

	return { center, radius };
}

export function mergeParticleShadowBounds(
	baseBounds: ParticleShadowVolumeBounds,
	particleBounds: ParticleShadowVolumeBounds | null
): ParticleShadowVolumeBounds {
	if (!particleBounds) {
		return baseBounds;
	}

	const baseRadius = Math.max(0, baseBounds.radius);
	const particleRadius = Math.max(0, particleBounds.radius);
	let minX = baseBounds.center.x - baseRadius;
	let minY = baseBounds.center.y - baseRadius;
	let minZ = baseBounds.center.z - baseRadius;
	let maxX = baseBounds.center.x + baseRadius;
	let maxY = baseBounds.center.y + baseRadius;
	let maxZ = baseBounds.center.z + baseRadius;

	minX = Math.min(minX, particleBounds.center.x - particleRadius);
	minY = Math.min(minY, particleBounds.center.y - particleRadius);
	minZ = Math.min(minZ, particleBounds.center.z - particleRadius);
	maxX = Math.max(maxX, particleBounds.center.x + particleRadius);
	maxY = Math.max(maxY, particleBounds.center.y + particleRadius);
	maxZ = Math.max(maxZ, particleBounds.center.z + particleRadius);

	const center = {
		x: (minX + maxX) * 0.5,
		y: (minY + maxY) * 0.5,
		z: (minZ + maxZ) * 0.5,
	};
	const sizeX = maxX - minX;
	const sizeY = maxY - minY;
	const sizeZ = maxZ - minZ;
	const radius = Math.sqrt(sizeX * sizeX + sizeY * sizeY + sizeZ * sizeZ) * 0.5;
	return { center, radius };
}

export function createParticleShadowVolumeGrid(
	resolution: ParticleShadowVolumeResolution =
		DEFAULT_PARTICLE_SHADOW_VOLUME_RESOLUTION
): ParticleShadowVolumeGrid {
	const width = Math.max(1, Math.floor(resolution.width));
	const height = Math.max(1, Math.floor(resolution.height));
	const depth = Math.max(1, Math.floor(resolution.depth));
	return {
		resolution: { width, height, depth },
		density: new Float32Array(width * height * depth),
		active: false,
	};
}

export function clearParticleShadowVolumeGrid(
	grid: ParticleShadowVolumeGrid
): void {
	grid.density.fill(0);
	grid.active = false;
}

export function injectParticleBatchIntoShadowVolume(
	grid: ParticleShadowVolumeGrid,
	shadowMap: ShadowMap,
	batch: ParticleRenderBatch
): void {
	if (!isParticleShadowCastingBatch(batch) || !shadowMap.viewProjectionMatrix) {
		return;
	}

	const { width, height, depth } = grid.resolution;
	const density = grid.density;
	const softness = Math.max(0.001, batch.shadowSoftness);
	const densityScale = Math.max(0, batch.shadowDensity);

	for (const particle of batch.particles) {
		const alpha = particle.color.a;
		const size = particle.size;
		if (alpha <= MIN_PARTICLE_SHADOW_ALPHA || size <= MIN_PARTICLE_SHADOW_SIZE) {
			continue;
		}
		const center = Matrix4.transformPoint(
			shadowMap.viewProjectionMatrix,
			particle.position
		);
		const w = center.w ?? 0;
		if (w <= 1e-6) {
			continue;
		}
		const ndcX = center.x / w;
		const ndcY = center.y / w;
		const ndcZ = center.z / w;
		if (ndcX < -1.25 || ndcX > 1.25 || ndcY < -1.25 || ndcY > 1.25 || ndcZ < -1.25 || ndcZ > 1.25) {
			continue;
		}

		const radiusNdc = estimateParticleRadiusNDC(
			shadowMap.viewProjectionMatrix,
			particle.position,
			size * 0.5,
			w
		);
		if (radiusNdc <= 0) {
			continue;
		}

		const cx = (ndcX * 0.5 + 0.5) * (width - 1);
		const cy = (0.5 - ndcY * 0.5) * (height - 1);
		const cz = (ndcZ * 0.5 + 0.5) * (depth - 1);
		const rx = Math.max(1, radiusNdc * 0.5 * width);
		const ry = Math.max(1, radiusNdc * 0.5 * height);
		const rz = Math.max(1, radiusNdc * 0.5 * depth);
		const minX = Math.max(0, Math.floor(cx - rx));
		const maxX = Math.min(width - 1, Math.ceil(cx + rx));
		const minY = Math.max(0, Math.floor(cy - ry));
		const maxY = Math.min(height - 1, Math.ceil(cy + ry));
		const minZ = Math.max(0, Math.floor(cz - rz));
		const maxZ = Math.min(depth - 1, Math.ceil(cz + rz));
		const contribution = alpha * densityScale;

		for (let z = minZ; z <= maxZ; z++) {
			const dz = (z - cz) / rz;
			const zBase = z * width * height;
			for (let y = minY; y <= maxY; y++) {
				const dy = (y - cy) / ry;
				const rowBase = zBase + y * width;
				for (let x = minX; x <= maxX; x++) {
					const dx = (x - cx) / rx;
					const distSq = dx * dx + dy * dy + dz * dz;
					if (distSq > 1) {
						continue;
					}
					const falloff = Math.pow(1 - Math.sqrt(distSq), softness);
					density[rowBase + x] += contribution * falloff;
					grid.active = true;
				}
			}
		}
	}
}

export function sampleParticleShadowVolumeTransmittance(
	grid: ParticleShadowVolumeGrid | null | undefined,
	shadowMap: ShadowMap,
	worldPoint: IVector3
): number {
	if (!grid?.active || !shadowMap.viewProjectionMatrix) {
		return 1;
	}

	const clip = Matrix4.transformPoint(shadowMap.viewProjectionMatrix, worldPoint);
	const w = clip.w ?? 0;
	if (w <= 1e-6) {
		return 1;
	}
	const ndcX = clip.x / w;
	const ndcY = clip.y / w;
	const ndcZ = clip.z / w;
	if (ndcX < -1 || ndcX > 1 || ndcY < -1 || ndcY > 1 || ndcZ < -1 || ndcZ > 1) {
		return 1;
	}

	const { width, height, depth } = grid.resolution;
	const x = Math.max(0, Math.min(width - 1, Math.round((ndcX * 0.5 + 0.5) * (width - 1))));
	const y = Math.max(0, Math.min(height - 1, Math.round((0.5 - ndcY * 0.5) * (height - 1))));
	const zMax = Math.max(0, Math.min(depth - 1, Math.round((ndcZ * 0.5 + 0.5) * (depth - 1))));
	let opticalDepth = 0;
	for (let z = 0; z <= zMax; z++) {
		opticalDepth += grid.density[z * width * height + y * width + x];
	}
	opticalDepth /= Math.max(1, depth);
	return Math.exp(-Math.max(0, opticalDepth));
}

function estimateParticleSystemShadowRadius(system: ParticleSystem): number {
	const emit = system.emit ?? {};
	const spawnRadius = Math.max(0, emit.spawnRadius ?? 0);
	let lifetime = 0;
	let speed = 0;
	let startSize = 0;
	let sizeScale = 1;

	if (Array.isArray(system.templates)) {
		for (const template of system.templates) {
			if (template.shape.kind !== "billboard") continue;
			lifetime = Math.max(
				lifetime,
				template.lifetimeRange[0],
				template.lifetimeRange[1]
			);
			speed = Math.max(speed, template.speedRange[0], template.speedRange[1]);
			startSize = Math.max(
				startSize,
				template.sizeRange[0],
				template.sizeRange[1]
			);
			if ((template.sizeOverLifetime?.length ?? 0) > 0) {
				sizeScale = Math.max(
					sizeScale,
					...template.sizeOverLifetime!.map((key) => key.value)
				);
			}
		}
	} else {
		const lifetimeRange = emit.lifetimeRange ?? [0.5, 1.5];
		const speedRange = emit.speedRange ?? [2, 5];
		const sizeRange = emit.sizeRange ?? [0.5, 1];
		lifetime = Math.max(0, lifetimeRange[0], lifetimeRange[1]);
		speed = Math.max(0, speedRange[0], speedRange[1]);
		startSize = Math.max(0, sizeRange[0], sizeRange[1]);
		const sizeOverLifetime = system.sizeOverLifetime ?? [];
		sizeScale =
			sizeOverLifetime.length > 0 ?
				Math.max(0, ...sizeOverLifetime.map((key) => key.value))
			:	1;
	}
	const gravity = system.gravity ?? { x: 0, y: 0, z: 0 };
	const gravityDistance =
		0.5 * Math.hypot(gravity.x, gravity.y, gravity.z) * lifetime * lifetime;
	return spawnRadius + speed * lifetime + gravityDistance + startSize * sizeScale;
}

function hasParticleSystemShadowCaster(system: ParticleSystem): boolean {
	if (Array.isArray(system.templates)) {
		return system.templates.some(isParticleTemplateShadowCaster);
	}
	return (
		system.castShadows === true &&
		system.blendMode !== ParticleBlendMode.Additive &&
		system.shadowDensity > MIN_PARTICLE_SHADOW_DENSITY
	);
}

function isParticleTemplateShadowCaster(
	template: ParticleTemplate
): boolean {
	return (
		template.shape.kind === "billboard" &&
		(template.castShadows ?? true) === true &&
		(template.shape.blendMode ?? ParticleBlendMode.Alpha) !==
			ParticleBlendMode.Additive &&
		(template.shadowDensity ?? 1) > MIN_PARTICLE_SHADOW_DENSITY
	);
}

function estimateParticleRadiusNDC(
	viewProjection: Matrix4,
	position: IVector3,
	radius: number,
	centerW: number
): number {
	if (radius <= 0 || centerW <= 1e-6) {
		return 0;
	}
	const px = Matrix4.transformPoint(viewProjection, {
		x: position.x + radius,
		y: position.y,
		z: position.z,
	});
	const py = Matrix4.transformPoint(viewProjection, {
		x: position.x,
		y: position.y + radius,
		z: position.z,
	});
	const pz = Matrix4.transformPoint(viewProjection, {
		x: position.x,
		y: position.y,
		z: position.z + radius,
	});
	const c = Matrix4.transformPoint(viewProjection, position);
	if ((px.w ?? 0) <= 1e-6 || (py.w ?? 0) <= 1e-6 || (pz.w ?? 0) <= 1e-6) {
		return 0;
	}
	const cx = c.x / centerW;
	const cy = c.y / centerW;
	const cz = c.z / centerW;
	return Math.max(
		Math.hypot(px.x / px.w - cx, px.y / px.w - cy, px.z / px.w - cz),
		Math.hypot(py.x / py.w - cx, py.y / py.w - cy, py.z / py.w - cz),
		Math.hypot(pz.x / pz.w - cx, pz.y / pz.w - cy, pz.z / pz.w - cz)
	);
}
