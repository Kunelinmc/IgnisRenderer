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
	[key: string]: unknown;
}

export type ShadowCameraMatrix = Matrix4;

export interface ShadowMapContext {
	worldPoint: IVector3;
	normal?: IVector3 | null;
	viewProjectionMatrix: Matrix4 | null;
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

		// Note: Slope bias is only effective with a surface normal
		const bias =
			normal ?
				Math.min(
					maxBias,
					constantBias +
						slopeBias * (1.0 - Vector3.dot(Vector3.normalize(normal), L)) +
						texelBias
				)
			:	Math.min(maxBias, constantBias + texelBias);

		const samples = Math.max(1, Math.floor(params.shadowPCF ?? 1));
		const texelSize = 1.0 / size;
		const strength = Math.max(0, Math.min(1, params.shadowStrength ?? 1.0));

		let visibilityR = 0;
		let visibilityG = 0;
		let visibilityB = 0;
		let validSampleCount = 0;

		for (let y = -samples; y <= samples; y++) {
			for (let x = -samples; x <= samples; x++) {
				const su = u + x * texelSize;
				const sv = v + y * texelSize;
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
