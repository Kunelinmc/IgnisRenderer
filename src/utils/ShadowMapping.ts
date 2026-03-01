/**
 * Shadow Mapping utilities
 */

import { ShadowConstants } from "../core/Constants";

import { Matrix4 } from "../maths/Matrix4";
import { Vector3 } from "../maths/Vector3";
import type { IVector3 } from "../maths/types";
import type { ShadowCastingLight } from "../lights";
import type { RGB } from "./Color";

export interface ShadowParams {
	shadowBias?: number;
	shadowSlopeBias?: number;
	shadowNormalBias?: number;
	shadowNormalBiasMin?: number;
	shadowTexelBias?: number;
	shadowMaxBias?: number;
	shadowPCF?: number;
	shadowStrength?: number;

	// PCSS (Percentage-Closer Soft Shadows) extensions
	shadowRadius?: number; // Maximum radius for PCF/Search (in texels). > 0 enables PCSS.
	shadowSamples?: number; // Number of samples for PCF block (default 16)
	shadowSearchSamples?: number; // Number of samples for Blocker search (default 16)

	[key: string]: unknown;
}

function getVogelSample(index: number, numSamples: number, theta: number) {
	const goldenAngle = 2.400049405230919;
	const r = Math.sqrt((index + 0.5) / numSamples);
	const angle = index * goldenAngle + theta;
	return { x: r * Math.cos(angle), y: r * Math.sin(angle) };
}

export type ShadowCameraMatrix = Matrix4;

export interface ShadowMapContext {
	worldPoint: IVector3;
	normal?: IVector3 | null;
	viewProjectionMatrix: Matrix4 | null;
	projectionMatrix?: Matrix4 | null;
	latestLightDir: IVector3;
	buffer: Float32Array;
	transmissionBuffer?: Float32Array;
	size: number;
	params: ShadowParams;
}

export class ShadowMap {
	public size: number;
	public buffer: Float32Array;
	public transmissionBuffer: Float32Array;
	public viewMatrix: Matrix4 | null = null;
	public projectionMatrix: Matrix4 | null = null;
	public viewProjectionMatrix: Matrix4 | null = null;
	public params: ShadowParams;
	public latestLightDir: IVector3 = { x: 0, y: -1, z: 0 };

	constructor(size = 1024, params: ShadowParams = {}) {
		this.size = size;
		this.buffer = new Float32Array(size * size);
		this.transmissionBuffer = new Float32Array(size * size * 3);
		this.clear();

		// Default shadow parameters
		this.params = {
			shadowBias: 0.008,
			shadowSlopeBias: 0.03,
			shadowNormalBias: 1.0,
			shadowNormalBiasMin: 0.05,
			shadowTexelBias: 1.0,
			shadowMaxBias: 0.05,
			shadowPCF: 1,
			shadowStrength: 1,
			...params,
		};
	}

	public clear(): void {
		this.buffer.fill(Infinity);
		this.transmissionBuffer.fill(1.0);
	}

	/**
	 * Set the light camera matrices
	 */
	public setLightCamera(
		light: ShadowCastingLight,
		sceneBoundingBox: { center: IVector3; radius: number },
		worldMatrix?: Matrix4
	): void {
		if (!light.shadow) return;

		const config = light.shadow.setupShadowCamera({
			sceneBounds: sceneBoundingBox,
			worldMatrix: worldMatrix ?? light.worldMatrix,
		});

		if (config) {
			this.viewMatrix = config.view;
			this.projectionMatrix = config.projection;
			this.latestLightDir = config.lightDir;
		}

		if (this.projectionMatrix && this.viewMatrix) {
			this.viewProjectionMatrix = Matrix4.multiply(
				this.projectionMatrix,
				this.viewMatrix
			);
		}
	}

	/**
	 * Get shadow factor for a world point (RGB for colored transmission)
	 */
	public getShadowFactor(worldPoint: IVector3, normal?: IVector3 | null): RGB {
		return ShadowMap._calculateShadowFactor({
			worldPoint,
			normal,
			viewProjectionMatrix: this.viewProjectionMatrix,
			projectionMatrix: this.projectionMatrix,
			latestLightDir: this.latestLightDir,
			buffer: this.buffer,
			transmissionBuffer: this.transmissionBuffer,
			size: this.size,
			params: this.params,
		});
	}

	private static _calculateShadowFactor(ctx: ShadowMapContext): RGB {
		const {
			worldPoint,
			normal,
			viewProjectionMatrix,
			latestLightDir,
			buffer,
			transmissionBuffer,
			size,
			params,
		} = ctx;

		if (!viewProjectionMatrix) return { r: 1.0, g: 1.0, b: 1.0 };

		const L = Vector3.normalize({
			x: -latestLightDir.x,
			y: -latestLightDir.y,
			z: -latestLightDir.z,
		});

		const normalBias = params.shadowNormalBias ?? 1.0;
		const normalBiasMin = params.shadowNormalBiasMin ?? 0.05;

		let offsetPoint = worldPoint;
		if (normal) {
			const N = Vector3.normalize(normal);
			const cosTheta = Math.max(0, Vector3.dot(N, L));
			const normalOffset =
				normalBiasMin + (normalBias - normalBiasMin) * (1.0 - cosTheta);
			offsetPoint = {
				x: worldPoint.x + N.x * normalOffset,
				y: worldPoint.y + N.y * normalOffset,
				z: worldPoint.z + N.z * normalOffset,
			};
		} else {
			// Volumetric bias: simple constant offset along light direction
			const volumeOffset = normalBiasMin;
			offsetPoint = {
				x: worldPoint.x + L.x * volumeOffset,
				y: worldPoint.y + L.y * volumeOffset,
				z: worldPoint.z + L.z * volumeOffset,
			};
		}

		const lightSpacePos = Matrix4.transformPoint(
			viewProjectionMatrix,
			offsetPoint
		);
		const w = lightSpacePos.w;
		if (w <= ShadowConstants.MIN_CLIP_W) return { r: 1.0, g: 1.0, b: 1.0 };
		const invW = 1 / w;
		const ndcX = lightSpacePos.x * invW;
		const ndcY = lightSpacePos.y * invW;
		const ndcZ = lightSpacePos.z * invW;

		const u = ndcX * 0.5 + 0.5;
		const v = 0.5 - ndcY * 0.5;
		const currentDepth = ndcZ;

		if (
			u < 0 ||
			u > 1 ||
			v < 0 ||
			v > 1 ||
			currentDepth < ShadowConstants.MIN_NDC_DEPTH ||
			currentDepth > ShadowConstants.MAX_NDC_DEPTH
		) {
			return { r: 1.0, g: 1.0, b: 1.0 };
		}

		const constantBias = params.shadowBias ?? 0.008;
		const slopeBias = params.shadowSlopeBias ?? 0.03;
		const texelBias = (params.shadowTexelBias ?? 1.0) * (2.0 / size);
		const maxBias = params.shadowMaxBias ?? 0.05;

		// For perspective: d = m23 / (ndcZ + m22)
		// For ortho: d = (m23 - ndcZ) / m22
		const m = ctx.projectionMatrix ? ctx.projectionMatrix.elements : null;
		const isPerspective = m ? Math.abs(m[3][2] + 1.0) < 1e-6 : false;

		const linearizeDepth = (ndcZ: number): number => {
			if (!m) return ndcZ;
			if (isPerspective) {
				return m[2][3] / (ndcZ + m[2][2]);
			}
			return (m[2][3] - ndcZ) / m[2][2];
		};

		// Note: Slope bias is only effective with a surface normal
		const bias = normal
			? Math.min(
					maxBias,
					constantBias +
						slopeBias * (1.0 - Vector3.dot(Vector3.normalize(normal), L)) +
						texelBias
				)
			: Math.min(maxBias, constantBias + texelBias);

		const samples = Math.max(1, Math.floor(params.shadowPCF ?? 1));
		const texelSize = 1.0 / size;
		const strength = Math.max(0, Math.min(1, params.shadowStrength ?? 1.0));

		const pcfRadiusParams = params.shadowRadius ?? 0;

		let visibilityR = 0;
		let visibilityG = 0;
		let visibilityB = 0;
		let validSampleCount = 0;

		if (pcfRadiusParams > 0) {
			// PCSS / Poisson Disk Soft Shadow
			// Interleaved gradient noise for per-pixel rotation
			const theta =
				(worldPoint.x * 12.9898 +
					worldPoint.y * 78.233 +
					worldPoint.z * 37.719) %
				(Math.PI * 2);

			const numSearchSamples = Math.floor(params.shadowSearchSamples ?? 16);
			const numSamples = Math.floor(params.shadowSamples ?? 16);
			const maxRadiusUV = pcfRadiusParams * texelSize;

			// 1. Blocker search
			let numBlockers = 0;
			let avgBlockerDepth = 0;
			for (let i = 0; i < numSearchSamples; i++) {
				const offset = getVogelSample(i, numSearchSamples, theta);
				const su = u + offset.x * maxRadiusUV;
				const sv = v + offset.y * maxRadiusUV;
				if (su >= 0 && su <= 1 && sv >= 0 && sv <= 1) {
					const tx = Math.max(
						0,
						Math.min(size - 1, Math.floor(su * (size - 1)))
					);
					const ty = Math.max(
						0,
						Math.min(size - 1, Math.floor(sv * (size - 1)))
					);
					const shadowDepth = buffer[ty * size + tx];
					if (currentDepth - bias > shadowDepth) {
						numBlockers++;
						avgBlockerDepth += shadowDepth;
					}
				}
			}

			if (numBlockers > 0) {
				avgBlockerDepth /= numBlockers;

				// 2. Penumbra size estimation
				// We use linear depth to get a physically meaningful ratio
				const linCurrent = linearizeDepth(currentDepth);
				const linBlocker = linearizeDepth(avgBlockerDepth);

				let penumbraRatio = 1.0;
				if (linCurrent > linBlocker) {
					// Traditional PCSS ratio: (d_receiver - d_blocker) / d_blocker
					// For orthographic lights, we use a constant factor as they don't have a 1/d divergence
					const divergence = isPerspective ? linBlocker || 1e-6 : 100.0; // 100 is an arbitrary scale for ortho
					penumbraRatio = (linCurrent - linBlocker) / divergence;
					penumbraRatio = Math.max(0.0, Math.min(1.0, penumbraRatio));
				} else {
					penumbraRatio = 0;
				}

				const filterRadiusUV = maxRadiusUV * penumbraRatio;

				// If penumbra is tiny, skip filtering and just return unshadowed or colored
				if (filterRadiusUV < texelSize * 0.1) {
					return this._calculateShadowFactor({
						...ctx,
						params: { ...params, shadowRadius: 0 },
					});
				}

				// 3. PCF Filtering with Vogel Disk
				for (let i = 0; i < numSamples; i++) {
					const offset = getVogelSample(i, numSamples, theta);
					const su = u + offset.x * filterRadiusUV;
					const sv = v + offset.y * filterRadiusUV;
					if (su < 0 || su > 1 || sv < 0 || sv > 1) continue;

					const tx = Math.max(
						0,
						Math.min(size - 1, Math.floor(su * (size - 1)))
					);
					const ty = Math.max(
						0,
						Math.min(size - 1, Math.floor(sv * (size - 1)))
					);
					const idx = ty * size + tx;
					const shadowDepth = buffer[idx];

					validSampleCount++;

					const isOccluded = currentDepth - bias > shadowDepth;
					if (isOccluded) {
						visibilityR += 1.0 - strength;
						visibilityG += 1.0 - strength;
						visibilityB += 1.0 - strength;
						continue;
					}

					let transSampleR = 1.0;
					let transSampleG = 1.0;
					let transSampleB = 1.0;
					if (transmissionBuffer) {
						const cIdx = idx * 3;
						transSampleR = transmissionBuffer[cIdx];
						transSampleG = transmissionBuffer[cIdx + 1];
						transSampleB = transmissionBuffer[cIdx + 2];
					}

					visibilityR += 1.0 - strength + strength * transSampleR;
					visibilityG += 1.0 - strength + strength * transSampleG;
					visibilityB += 1.0 - strength + strength * transSampleB;
				}
			} else {
				// No blockers found, unshadowed
				return { r: 1.0, g: 1.0, b: 1.0 };
			}
		} else {
			// Fallback: Vogel Disk PCF for smoother fixed-size soft shadows
			const theta =
				(worldPoint.x * 12.9898 +
					worldPoint.y * 78.233 +
					worldPoint.z * 37.719) %
				(Math.PI * 2);

			const pcfRadius = params.shadowPCF ?? 1.5; // Fixed radius in texels
			const numSamples = Math.floor(params.shadowSamples ?? 16);
			const radiusUV = pcfRadius * texelSize;

			for (let i = 0; i < numSamples; i++) {
				const offset = getVogelSample(i, numSamples, theta);
				const su = u + offset.x * radiusUV;
				const sv = v + offset.y * radiusUV;
				if (su < 0 || su > 1 || sv < 0 || sv > 1) continue;

				const tx = Math.max(0, Math.min(size - 1, Math.floor(su * (size - 1))));
				const ty = Math.max(0, Math.min(size - 1, Math.floor(sv * (size - 1))));
				const idx = ty * size + tx;
				const shadowDepth = buffer[idx];

				validSampleCount++;

				const isOccluded = currentDepth - bias > shadowDepth;
				if (isOccluded) {
					visibilityR += 1.0 - strength;
					visibilityG += 1.0 - strength;
					visibilityB += 1.0 - strength;
					continue;
				}

				let transSampleR = 1.0;
				let transSampleG = 1.0;
				let transSampleB = 1.0;
				if (transmissionBuffer) {
					const cIdx = idx * 3;
					transSampleR = transmissionBuffer[cIdx];
					transSampleG = transmissionBuffer[cIdx + 1];
					transSampleB = transmissionBuffer[cIdx + 2];
				}

				visibilityR += 1.0 - strength + strength * transSampleR;
				visibilityG += 1.0 - strength + strength * transSampleG;
				visibilityB += 1.0 - strength + strength * transSampleB;
			}
		}

		if (validSampleCount === 0) return { r: 1.0, g: 1.0, b: 1.0 };

		const invCount = 1.0 / validSampleCount;

		return {
			r: Math.max(0, Math.min(1, visibilityR * invCount)),
			g: Math.max(0, Math.min(1, visibilityG * invCount)),
			b: Math.max(0, Math.min(1, visibilityB * invCount)),
		};
	}
}
