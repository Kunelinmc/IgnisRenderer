import { Matrix4 } from "../../maths/Matrix4"
import { clamp } from "../../maths/Common"
import {
	ParticleSpaceMode,
	type ParticleGradientKey,
	type ParticleSystem,
} from "../../particles"
import type { FrameContext, ParticleRenderBatch, ParticleRenderItem, ParticleUVRect } from "../../pipeline/types"
import type { RGBA } from "../../utils/Color"
import { cloneColor } from "./ParticleSimulationCore"
import type { RuntimeParticle, SystemRuntimeState } from "./types"

const FULL_UV_RECT: ParticleUVRect = { u0: 0, v0: 0, u1: 1, v1: 1 }

export class ParticleBatchBuilder {
	public buildBatch(
		system: ParticleSystem,
		runtime: SystemRuntimeState,
		context: FrameContext,
		renderSortRatio: number
	): ParticleRenderBatch {
		const particles: ParticleRenderItem[] = []
		const cameraView = context.camera.viewMatrix
		const systemPosition = system.position

		for (const particle of runtime.particles) {
			const worldPosition =
				system.space === ParticleSpaceMode.Local
					? {
							x: particle.position.x + systemPosition.x,
							y: particle.position.y + systemPosition.y,
							z: particle.position.z + systemPosition.z,
						}
					: {
							x: particle.position.x,
							y: particle.position.y,
							z: particle.position.z,
						}

			const cameraSpace = Matrix4.transformPoint(cameraView, worldPosition)
			const depth = -cameraSpace.z
			if (depth <= 0) continue

			const lifeT = clamp(particle.age / particle.lifetime)
			const sizeMultiplier = this._sampleNumberGradient(
				system.sizeOverLifetime,
				lifeT,
				1
			)
			const size = particle.startSize * Math.max(0, sizeMultiplier)
			const color = this._sampleColorGradient(
				system.colorOverLifetime,
				lifeT,
				particle.startColor
			)

			if (size <= 0 || color.a <= 0) continue

			particles.push({
				position: worldPosition,
				size,
				color,
				rotation: particle.rotation,
				depth,
				uvRect: this._resolveAtlasUVRect(system, particle),
			})
		}

		particles.sort((left, right) => right.depth - left.depth)
		const ratio = clamp(renderSortRatio)
		if (ratio <= 0) {
			particles.length = 0
		} else if (ratio < 1 && particles.length > 0) {
			const visibleCount = Math.max(1, Math.floor(particles.length * ratio))
			particles.length = Math.min(particles.length, visibleCount)
		}

		return {
			systemId: system.id,
			blendMode: system.blendMode,
			texture: system.texture,
			receiveShadows: system.receiveShadows,
			particles,
		}
	}

	private _resolveAtlasUVRect(
		system: ParticleSystem,
		particle: RuntimeParticle
	): ParticleUVRect {
		const atlas = system.atlas
		if (!atlas) return FULL_UV_RECT

		const rows = Math.max(1, atlas.rows | 0)
		const columns = Math.max(1, atlas.columns | 0)
		const fps = Math.max(0, atlas.fps)
		const frameCount = rows * columns
		if (frameCount <= 1 || fps <= 0) return FULL_UV_RECT

		const rawFrame = Math.floor(particle.age * fps)
		const frame =
			atlas.loop === false
				? Math.min(frameCount - 1, rawFrame)
				: ((rawFrame % frameCount) + frameCount) % frameCount
		const column = frame % columns
		const row = Math.floor(frame / columns)
		const u0 = column / columns
		const v0 = row / rows

		return {
			u0,
			v0,
			u1: (column + 1) / columns,
			v1: (row + 1) / rows,
		}
	}

	private _sampleNumberGradient(
		keys: ParticleGradientKey<number>[] | undefined,
		t: number,
		fallback: number
	): number {
		if (!keys || keys.length === 0) return fallback
		const sorted = keys.slice().sort((left, right) => left.t - right.t)
		if (t <= sorted[0].t) return sorted[0].value
		if (t >= sorted[sorted.length - 1].t) {
			return sorted[sorted.length - 1].value
		}

		for (let i = 1; i < sorted.length; i++) {
			const left = sorted[i - 1]
			const right = sorted[i]
			if (t > right.t) continue
			const span = right.t - left.t || 1
			const localT = (t - left.t) / span
			return left.value + (right.value - left.value) * localT
		}

		return fallback
	}

	private _sampleColorGradient(
		keys: ParticleGradientKey<RGBA>[] | undefined,
		t: number,
		fallback: RGBA
	): RGBA {
		if (!keys || keys.length === 0) {
			return cloneColor(fallback)
		}
		const sorted = keys.slice().sort((left, right) => left.t - right.t)
		if (t <= sorted[0].t) return cloneColor(sorted[0].value)
		if (t >= sorted[sorted.length - 1].t) {
			return cloneColor(sorted[sorted.length - 1].value)
		}

		for (let i = 1; i < sorted.length; i++) {
			const left = sorted[i - 1]
			const right = sorted[i]
			if (t > right.t) continue
			const span = right.t - left.t || 1
			const localT = (t - left.t) / span
			return {
				r: left.value.r + (right.value.r - left.value.r) * localT,
				g: left.value.g + (right.value.g - left.value.g) * localT,
				b: left.value.b + (right.value.b - left.value.b) * localT,
				a: left.value.a + (right.value.a - left.value.a) * localT,
			}
		}

		return cloneColor(fallback)
	}
}

