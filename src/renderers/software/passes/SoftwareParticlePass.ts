import { CameraType } from "../../../cameras/Camera";
import { isShadowCastingLight } from "../../../lights";
import { Matrix4 } from "../../../maths/Matrix4";
import { ParticleBlendMode } from "../../../particles";
import {
	PARTICLE_TRANSIENT_BATCHES_KEY,
	type FrameContext,
	type ParticleRenderBatch,
	type ParticleRenderItem,
} from "../../../pipeline/types";
import { CoreConstants } from "../constants";
import {
	createSoftwareShadowSampler,
	getSoftwareShadowRuntimeMap,
} from "../shadows";
import { clamp } from "../../../maths/Common";

const MIN_PARTICLE_PIXEL_RADIUS = 0.5;

export class SoftwareParticlePass {
	public render(context: FrameContext): void {
		const batches = context.transient.get(PARTICLE_TRANSIENT_BATCHES_KEY) as
			| ParticleRenderBatch[]
			| undefined;
		if (!batches || batches.length === 0) return;
		if (!context.attachments.pixels || !context.attachments.depthBuffer) return;

		const runtimeMap = getSoftwareShadowRuntimeMap(context.transient);
		const sampleShadow = createSoftwareShadowSampler(
			context.shadowMaps,
			runtimeMap
		);

		for (const batch of batches) {
			for (const particle of batch.particles) {
				this._drawParticle(context, batch, particle, sampleShadow);
			}
		}
	}

	private _drawParticle(
		context: FrameContext,
		batch: ParticleRenderBatch,
		particle: ParticleRenderItem,
		sampleShadow: ReturnType<typeof createSoftwareShadowSampler>
	): void {
		const attachments = context.attachments;
		const width = attachments.width;
		const height = attachments.height;
		const pixels = attachments.pixels!;
		const depthBuffer = attachments.depthBuffer!;
		const viewPosition = Matrix4.transformPoint(
			context.camera.viewMatrix,
			particle.position
		);
		const depth = -viewPosition.z;
		if (depth <= 0) return;

		const clip = Matrix4.transformPoint(
			context.camera.projectionMatrix,
			viewPosition
		);
		const w = clip.w ?? 0;
		if (Math.abs(w) < CoreConstants.EPSILON) return;

		const ndcX = clip.x / w;
		const ndcY = clip.y / w;
		const centerX = (ndcX * 0.5 + 0.5) * width;
		const centerY = (0.5 - ndcY * 0.5) * height;

		const radiusPx = this._resolveParticleRadiusPx(
			context,
			particle.size,
			depth
		);
		if (radiusPx <= 0) return;

		const minX = Math.max(0, Math.floor(centerX - radiusPx));
		const maxX = Math.min(width - 1, Math.ceil(centerX + radiusPx));
		const minY = Math.max(0, Math.floor(centerY - radiusPx));
		const maxY = Math.min(height - 1, Math.ceil(centerY + radiusPx));
		if (minX > maxX || minY > maxY) return;

		const shadowVisibility =
			batch.receiveShadows && context.features.enableShadows
				? this._resolveShadowVisibility(context, particle, sampleShadow)
				: 1;
		const baseAlpha = clamp(particle.color.a);
		const rotation = particle.rotation;
		const cosRot = Math.cos(rotation);
		const sinRot = Math.sin(rotation);

		for (let y = minY; y <= maxY; y++) {
			const row = y * width;
			for (let x = minX; x <= maxX; x++) {
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
					1 - Math.min(1, (dist - 0.4) / (0.5 - 0.4))
				);
				const alpha = baseAlpha * (texA / 255) * radialMask;
				if (alpha <= 0.001) continue;

				const srcR = particle.color.r * (texR / 255) * shadowVisibility;
				const srcG = particle.color.g * (texG / 255) * shadowVisibility;
				const srcB = particle.color.b * (texB / 255) * shadowVisibility;

				const pixelIndex = bufferIndex << 2;
				if (batch.blendMode === ParticleBlendMode.Additive) {
					pixels[pixelIndex] = Math.min(255, pixels[pixelIndex] + srcR * alpha);
					pixels[pixelIndex + 1] = Math.min(
						255,
						pixels[pixelIndex + 1] + srcG * alpha
					);
					pixels[pixelIndex + 2] = Math.min(
						255,
						pixels[pixelIndex + 2] + srcB * alpha
					);
					pixels[pixelIndex + 3] = 255;
					continue;
				}

				const invAlpha = 1 - alpha;
				pixels[pixelIndex] = srcR * alpha + pixels[pixelIndex] * invAlpha;
				pixels[pixelIndex + 1] =
					srcG * alpha + pixels[pixelIndex + 1] * invAlpha;
				pixels[pixelIndex + 2] =
					srcB * alpha + pixels[pixelIndex + 2] * invAlpha;
				pixels[pixelIndex + 3] = 255;
			}
		}
	}

	private _resolveParticleRadiusPx(
		context: FrameContext,
		size: number,
		depth: number
	): number {
		if (context.camera.type === CameraType.Orthographic) {
			return Math.max(MIN_PARTICLE_PIXEL_RADIUS, size * 0.5);
		}

		const halfFovRadians = (context.camera.fov * Math.PI) / 360;
		const tanHalfFov = Math.tan(halfFovRadians) || CoreConstants.EPSILON;
		const focalLength = (context.attachments.height * 0.5) / tanHalfFov;
		const pixelSize = (Math.max(0.001, size) * focalLength) / depth;
		return Math.max(MIN_PARTICLE_PIXEL_RADIUS, pixelSize * 0.5);
	}

	private _resolveShadowVisibility(
		context: FrameContext,
		particle: ParticleRenderItem,
		sampleShadow: ReturnType<typeof createSoftwareShadowSampler>
	): number {
		let visibility = 1;
		for (const light of context.scene.lights) {
			if (!isShadowCastingLight(light)) continue;
			const sampled = sampleShadow(light, particle.position, null);
			const shadow = clamp((sampled.r + sampled.g + sampled.b) / 3);
			visibility *= shadow;
		}
		return clamp(visibility);
	}
}
