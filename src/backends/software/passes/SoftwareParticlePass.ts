import { CameraType } from "../../../cameras/Camera";
import { isShadowCastingLight } from "../../../lights";
import { Matrix4 } from "../../../maths/Matrix4";
import { ParticleBlendMode } from "../../../particles";
import {
	type ParticleRenderBatch,
	type ParticleRenderItem,
} from "../../../particles/ParticleRenderBatch";
import { clamp } from "../../../maths/Common";
import type { SoftwarePassLike } from "./types";
import { Logger } from "../../../foundation/Logger";
import type { SoftwarePassContext } from "../SoftwareFrameServices";
import type { SoftwareFrameView } from "../SoftwareFrameView";
import type { SoftwareShadowSampler } from "../SoftwareShadowContracts";

const MIN_PARTICLE_PIXEL_RADIUS = 0.5;
const PARTICLE_RADIAL_FADE_START = 0.4;
const PARTICLE_RADIAL_FADE_END = 0.5;
const PARTICLE_RADIAL_FADE_RANGE =
	PARTICLE_RADIAL_FADE_END - PARTICLE_RADIAL_FADE_START;
const PARTICLE_ALPHA_CUTOFF = 0.001;
const MIN_PARTICLE_WORLD_SIZE = 0.001;
const PROJECTION_EPSILON = 1e-6;

export class SoftwareParticlePass implements SoftwarePassLike {
	public render(context: SoftwarePassContext): void {
		warnSkippedMeshParticles(context);
		const batches = context.services.particles.batches;
		if (!batches || batches.length === 0) return;
		const frame = context.frame;
		if (frame.clipRegions.length === 0) {
			return;
		}

		for (const batch of batches) {
			for (const particle of batch.particles) {
				this._drawParticle(
					frame,
					batch,
					particle,
					context.services.shadow.sampler,
				);
			}
		}
	}

	private _drawParticle(
		frame: SoftwareFrameView,
		batch: ParticleRenderBatch,
		particle: ParticleRenderItem,
		sampleShadow: SoftwareShadowSampler,
	): void {
		const attachments = frame.attachments;
		const width = attachments.width;
		const height = attachments.height;
		const pixels = attachments.color;
		const depthBuffer = attachments.depthBuffer;
		const viewPosition = Matrix4.transformPoint(frame.camera.viewMatrix, particle.position);
		const depth = -viewPosition.z;
		if (depth <= 0) return;

		const clip = Matrix4.transformPoint(frame.camera.projectionMatrix, viewPosition);
		const w = clip.w ?? 0;
		if (Math.abs(w) < PROJECTION_EPSILON) return;

		const ndcX = clip.x / w;
		const ndcY = clip.y / w;
		const centerX = (ndcX * 0.5 + 0.5) * width;
		const centerY = (0.5 - ndcY * 0.5) * height;

		const radiusPx = this._resolveParticleRadiusPx(frame, particle.size, depth);
		if (radiusPx <= 0) return;

		const minX = Math.max(0, Math.floor(centerX - radiusPx));
		const maxX = Math.min(width - 1, Math.ceil(centerX + radiusPx));
		const minY = Math.max(0, Math.floor(centerY - radiusPx));
		const maxY = Math.min(height - 1, Math.ceil(centerY + radiusPx));
		if (minX > maxX || minY > maxY) return;
		const shadowVisibility =
			batch.receiveShadows && frame.features.enableShadows
				? this._resolveShadowVisibility(frame, particle, sampleShadow)
				: 1;
		const baseAlpha = clamp(particle.color.a);
		const rotation = particle.rotation;
		const cosRot = Math.cos(rotation);
		const sinRot = Math.sin(rotation);

		for (const region of frame.clipRegions) {
			const regionMinX = Math.max(minX, region.minX);
			const regionMaxX = Math.min(maxX, region.maxXExclusive - 1);
			const regionMinY = Math.max(minY, region.minY);
			const regionMaxY = Math.min(maxY, region.maxYExclusive - 1);
			if (regionMinX > regionMaxX || regionMinY > regionMaxY) continue;
			for (let y = regionMinY; y <= regionMaxY; y++) {
				const row = y * width;
				for (let x = regionMinX; x <= regionMaxX; x++) {
					const bufferIndex = row + x;
					if (depth >= depthBuffer[bufferIndex]) continue;

					let u = (x + 0.5 - (centerX - radiusPx)) / (radiusPx * 2);
					let v = (y + 0.5 - (centerY - radiusPx)) / (radiusPx * 2);
					if (u < 0 || u > 1 || v < 0 || v > 1) continue;

					if (rotation !== 0) {
						const localX = u - 0.5;
						const localY = v - 0.5;
						const rotatedX = localX * cosRot - localY * sinRot;
						const rotatedY = localX * sinRot + localY * cosRot;
						u = rotatedX + 0.5;
						v = rotatedY + 0.5;
						if (u < 0 || u > 1 || v < 0 || v > 1) continue;
					}

					let texR = 255;
					let texG = 255;
					let texB = 255;
					let texA = 255;
					if (batch.texture) {
						const uvRect = particle.uvRect;
						// Keep particle UV composition consistent with WebGPU:
						// atlas rect first, then Texture.sample() applies repeat/rotation/offset.
						const atlasU = uvRect.u0 + (uvRect.u1 - uvRect.u0) * u;
						const atlasV = uvRect.v0 + (uvRect.v1 - uvRect.v0) * v;
						const sampled = batch.texture.sample(atlasU, atlasV);
						texR = sampled.r;
						texG = sampled.g;
						texB = sampled.b;
						texA = sampled.a;
					}

					// Procedural soft radial falloff (circle mask)
					const dist = Math.sqrt((u - 0.5) ** 2 + (v - 0.5) ** 2);
					const radialMask = Math.max(
						0,
						1 -
							Math.min(
								1,
								(dist - PARTICLE_RADIAL_FADE_START) / PARTICLE_RADIAL_FADE_RANGE
							)
					);
					const alpha = baseAlpha * (texA / 255) * radialMask;
					if (alpha <= PARTICLE_ALPHA_CUTOFF) continue;

					const srcR = (particle.color.r / 255) * (texR / 255) * shadowVisibility;
					const srcG = (particle.color.g / 255) * (texG / 255) * shadowVisibility;
					const srcB = (particle.color.b / 255) * (texB / 255) * shadowVisibility;

					const pixelIndex = bufferIndex << 2;
					if (batch.blendMode === ParticleBlendMode.Additive) {
						pixels[pixelIndex] += srcR * alpha;
						pixels[pixelIndex + 1] += srcG * alpha;
						pixels[pixelIndex + 2] += srcB * alpha;
						pixels[pixelIndex + 3] =
							alpha + pixels[pixelIndex + 3] * (1 - alpha);
						continue;
					}

					const invAlpha = 1 - alpha;
					pixels[pixelIndex] = srcR * alpha + pixels[pixelIndex] * invAlpha;
					pixels[pixelIndex + 1] = srcG * alpha + pixels[pixelIndex + 1] * invAlpha;
					pixels[pixelIndex + 2] = srcB * alpha + pixels[pixelIndex + 2] * invAlpha;
					pixels[pixelIndex + 3] =
						alpha + pixels[pixelIndex + 3] * invAlpha;
				}
			}
		}
	}

	private _resolveParticleRadiusPx(
		frame: SoftwareFrameView,
		size: number,
		depth: number,
	): number {
		if (frame.camera.type === CameraType.Orthographic) {
			return Math.max(MIN_PARTICLE_PIXEL_RADIUS, size * 0.5);
		}

		const halfFovRadians = (frame.camera.fov * Math.PI) / 360;
		const tanHalfFov = Math.tan(halfFovRadians) || PROJECTION_EPSILON;
		const focalLength = (frame.attachments.height * 0.5) / tanHalfFov;
		const pixelSize = (Math.max(MIN_PARTICLE_WORLD_SIZE, size) * focalLength) / depth;
		return Math.max(MIN_PARTICLE_PIXEL_RADIUS, pixelSize * 0.5);
	}

	private _resolveShadowVisibility(
		frame: SoftwareFrameView,
		particle: ParticleRenderItem,
		sampleShadow: SoftwareShadowSampler,
	): number {
		let visibility = 1;
		for (const light of frame.scene.lights) {
			if (!isShadowCastingLight(light)) continue;
			const sampled = sampleShadow(light, particle.position, null);
			const shadow = clamp((sampled.r + sampled.g + sampled.b) / 3);
			visibility *= shadow;
		}
		return clamp(visibility);
	}
}

function warnSkippedMeshParticles(context: SoftwarePassContext): void {
	const meshBatches = context.services.particles.meshBatches;
	if (!meshBatches || meshBatches.length === 0) {
		return;
	}
	Logger.warn(
		[
			"[software-particle-mesh-skipped] SoftwareBackend skips mesh particle",
			"templates; use WebGPUBackend to render particle meshes.",
		].join(" "),
		{
			scope: "SoftwareParticlePass",
			onceKey: "software-particle-mesh-skipped",
		}
	);
}
