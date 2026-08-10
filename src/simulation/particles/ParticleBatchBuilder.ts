import { Matrix4 } from "../../maths/Matrix4";
import { clamp } from "../../maths/Common";
import {
	ParticleBlendMode,
	ParticleSpaceMode,
	type ParticleTemplate,
	type ParticleGradientKey,
	type ParticleSystem,
} from "../../particles";
import type {
	ParticleMeshRenderBatch,
	ParticleMeshRenderItem,
	ParticleRenderBatch,
	ParticleUVRect,
} from "../../particles/ParticleRenderBatch";
import type { FrameContext } from "../../pipeline/types";
import type { RGBA } from "../../foundation/Color";
import { cloneColor } from "./ParticleSimulationCore";
import type { RuntimeParticle, SystemRuntimeState } from "./types";

const FULL_UV_RECT: ParticleUVRect = { u0: 0, v0: 0, u1: 1, v1: 1 };

export interface ParticleBuildResult {
	billboardBatches: ParticleRenderBatch[];
	meshBatches: ParticleMeshRenderBatch[];
}

export class ParticleBatchBuilder {
	public buildBatches(
		system: ParticleSystem,
		runtime: SystemRuntimeState,
		context: FrameContext,
		renderSortRatio: number
	): ParticleBuildResult {
		const billboardBatches = new Map<number, ParticleRenderBatch>();
		const meshBatches = new Map<string, ParticleMeshRenderBatch>();
		const cameraView = context.viewCamera.viewMatrix;
		const systemPosition = system.getWorldPosition();

		for (const particle of runtime.particles) {
			const template = system.templates[particle.templateIndex];
			if (!template) continue;
			const worldPosition = this._resolveWorldPosition(
				system,
				systemPosition,
				particle.position
			);
			const previousWorldPosition = this._resolveWorldPosition(
				system,
				systemPosition,
				particle.previousPosition
			);

			const cameraSpace = Matrix4.transformPoint(cameraView, worldPosition);
			const depth = -cameraSpace.z;
			if (depth <= 0) continue;

			const lifeT = clamp(particle.age / particle.lifetime);
			const sizeMultiplier = this._sampleNumberGradient(
				template.sizeOverLifetime,
				lifeT,
				1
			);
			const size = particle.startSize * Math.max(0, sizeMultiplier);

			if (size <= 0) continue;

			if (template.shape.kind === "mesh") {
				this._pushMeshParticle(
					meshBatches,
					system,
					template,
					particle.templateIndex,
					{
						templateIndex: particle.templateIndex,
						position: worldPosition,
						previousPosition: previousWorldPosition,
						size,
						rotation: particle.rotation,
						previousRotation: particle.previousRotation,
						depth,
					}
				);
				continue;
			}

			const color = this._sampleColorGradient(
				template.colorOverLifetime,
				lifeT,
				particle.startColor
			);
			if (color.a <= 0) continue;

			const batch = this._getOrCreateBillboardBatch(
				billboardBatches,
				system,
				template,
				particle.templateIndex
			);
			batch.particles.push({
				templateIndex: particle.templateIndex,
				position: worldPosition,
				previousPosition: previousWorldPosition,
				size,
				color,
				rotation: particle.rotation,
				previousRotation: particle.previousRotation,
				depth,
				uvRect: this._resolveAtlasUVRect(template, particle),
			});
		}

		const ratio = clamp(renderSortRatio);
		const billboardResults = Array.from(billboardBatches.values());
		for (const batch of billboardResults) {
			this._sortAndTrim(batch.particles, ratio);
		}

		const meshResults = Array.from(meshBatches.values());
		for (const batch of meshResults) {
			this._sortAndTrim(batch.particles, ratio);
		}

		return {
			billboardBatches: billboardResults.filter(
				(batch) => batch.particles.length > 0
			),
			meshBatches: meshResults.filter((batch) => batch.particles.length > 0),
		};
	}

	public buildBatch(
		system: ParticleSystem,
		runtime: SystemRuntimeState,
		context: FrameContext,
		renderSortRatio: number
	): ParticleRenderBatch {
		const result = this.buildBatches(system, runtime, context, renderSortRatio);
		return result.billboardBatches[0] ?? {
			kind: "billboard",
			systemId: system.id,
			templateIndex: 0,
			templateId: system.templates[0]?.id,
			blendMode: ParticleBlendMode.Alpha,
			texture: null,
			receiveShadows: true,
			castShadows: true,
			shadowDensity: 1,
			shadowSoftness: 1,
			particles: [],
		};
	}

	private _resolveWorldPosition(
		system: ParticleSystem,
		systemPosition: { x: number; y: number; z: number },
		position: { x: number; y: number; z: number }
	): { x: number; y: number; z: number } {
		return system.space === ParticleSpaceMode.Local ?
				{
					x: position.x + systemPosition.x,
					y: position.y + systemPosition.y,
					z: position.z + systemPosition.z,
				}
			:	{
					x: position.x,
					y: position.y,
					z: position.z,
				};
	}

	private _getOrCreateBillboardBatch(
		batches: Map<number, ParticleRenderBatch>,
		system: ParticleSystem,
		template: ParticleTemplate,
		templateIndex: number
	): ParticleRenderBatch {
		let batch = batches.get(templateIndex);
		if (batch) return batch;
		const shape = template.shape.kind === "billboard" ? template.shape : null;
		const blendMode = shape?.blendMode ?? ParticleBlendMode.Alpha;
		batch = {
			kind: "billboard",
			systemId: system.id,
			templateIndex,
			templateId: template.id,
			blendMode,
			texture: shape?.texture ?? null,
			receiveShadows: template.receiveShadows ?? true,
			castShadows:
				(template.castShadows ?? true) &&
				blendMode !== ParticleBlendMode.Additive,
			shadowDensity: Math.max(0, template.shadowDensity ?? 1),
			shadowSoftness: Math.max(0, template.shadowSoftness ?? 1),
			particles: [],
		};
		batches.set(templateIndex, batch);
		return batch;
	}

	private _pushMeshParticle(
		batches: Map<string, ParticleMeshRenderBatch>,
		system: ParticleSystem,
		template: ParticleTemplate,
		templateIndex: number,
		item: ParticleMeshRenderItem
	): void {
		if (template.shape.kind !== "mesh") return;
		const mesh = template.shape.mesh;
		for (const primitive of mesh.primitives) {
			if (primitive.visible === false) continue;
			const key = `${templateIndex}:${primitive.id}:${primitive.material.name}`;
			let batch = batches.get(key);
			if (!batch) {
				batch = {
					kind: "mesh",
					systemId: system.id,
					templateIndex,
					templateId: template.id,
					mesh,
					primitive,
					material: primitive.material,
					receiveShadows: template.receiveShadows ?? true,
					castShadows:
						(template.castShadows ?? true) &&
						primitive.castShadows !== false,
					shadowDensity: Math.max(0, template.shadowDensity ?? 1),
					shadowSoftness: Math.max(0, template.shadowSoftness ?? 1),
					particles: [],
				};
				batches.set(key, batch);
			}
			batch.particles.push(item);
		}
	}

	private _sortAndTrim<T extends { depth: number }>(
		particles: T[],
		ratio: number
	): void {
		particles.sort((left, right) => right.depth - left.depth);
		if (ratio <= 0) {
			particles.length = 0;
		} else if (ratio < 1 && particles.length > 0) {
			const visibleCount = Math.max(1, Math.floor(particles.length * ratio));
			particles.length = Math.min(particles.length, visibleCount);
		}
	}

	private _resolveAtlasUVRect(
		template: ParticleTemplate,
		particle: RuntimeParticle
	): ParticleUVRect {
		const atlas =
			template.shape.kind === "billboard" ? template.shape.atlas : null;
		if (!atlas) return FULL_UV_RECT;

		const rows = Math.max(1, atlas.rows | 0);
		const columns = Math.max(1, atlas.columns | 0);
		const fps = Math.max(0, atlas.fps);
		const frameCount = rows * columns;
		if (frameCount <= 1 || fps <= 0) return FULL_UV_RECT;

		const rawFrame = Math.floor(particle.age * fps);
		const frame =
			atlas.loop === false ?
				Math.min(frameCount - 1, rawFrame)
			:	((rawFrame % frameCount) + frameCount) % frameCount;
		const column = frame % columns;
		const row = Math.floor(frame / columns);
		const u0 = column / columns;
		const v0 = row / rows;

		return {
			u0,
			v0,
			u1: (column + 1) / columns,
			v1: (row + 1) / rows,
		};
	}

	private _sampleNumberGradient(
		keys: ParticleGradientKey<number>[] | undefined,
		t: number,
		fallback: number
	): number {
		if (!keys || keys.length === 0) return fallback;
		const sorted = keys.slice().sort((left, right) => left.t - right.t);
		if (t <= sorted[0].t) return sorted[0].value;
		if (t >= sorted[sorted.length - 1].t) {
			return sorted[sorted.length - 1].value;
		}

		for (let i = 1; i < sorted.length; i++) {
			const left = sorted[i - 1];
			const right = sorted[i];
			if (t > right.t) continue;
			const span = right.t - left.t || 1;
			const localT = (t - left.t) / span;
			return left.value + (right.value - left.value) * localT;
		}

		return fallback;
	}

	private _sampleColorGradient(
		keys: ParticleGradientKey<RGBA>[] | undefined,
		t: number,
		fallback: RGBA
	): RGBA {
		if (!keys || keys.length === 0) {
			return cloneColor(fallback);
		}
		const sorted = keys.slice().sort((left, right) => left.t - right.t);
		if (t <= sorted[0].t) return cloneColor(sorted[0].value);
		if (t >= sorted[sorted.length - 1].t) {
			return cloneColor(sorted[sorted.length - 1].value);
		}

		for (let i = 1; i < sorted.length; i++) {
			const left = sorted[i - 1];
			const right = sorted[i];
			if (t > right.t) continue;
			const span = right.t - left.t || 1;
			const localT = (t - left.t) / span;
			return {
				r: left.value.r + (right.value.r - left.value.r) * localT,
				g: left.value.g + (right.value.g - left.value.g) * localT,
				b: left.value.b + (right.value.b - left.value.b) * localT,
				a: left.value.a + (right.value.a - left.value.a) * localT,
			};
		}

		return cloneColor(fallback);
	}
}
