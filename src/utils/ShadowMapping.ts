/**
 * Shadow Mapping utilities
 */

import { ShadowConstants } from "../core/Constants";

import { Matrix4 } from "../maths/Matrix4";
import { Vector3 } from "../maths/Vector3";
import type { IVector3 } from "../maths/types";
import type { ShadowCastingLight } from "../lights";

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
	normal: IVector3;
	viewProjectionMatrix: Matrix4 | null;
	latestLightDir: IVector3;
	buffer: Float32Array;
	size: number;
	params: ShadowParams;
}

export class ShadowMap {
	public size: number;
	public buffer: Float32Array;
	public viewMatrix: Matrix4 | null = null;
	public projectionMatrix: Matrix4 | null = null;
	public viewProjectionMatrix: Matrix4 | null = null;
	public params: ShadowParams;
	public latestLightDir: IVector3 = { x: 0, y: -1, z: 0 };

	constructor(size = 1024, params: ShadowParams = {}) {
		this.size = size;
		this.buffer = new Float32Array(size * size);
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
	 * Get shadow factor for a world point
	 */
	public getShadowFactor(worldPoint: IVector3, normal: IVector3): number {
		return ShadowMap._calculateShadowFactor({
			worldPoint,
			normal,
			viewProjectionMatrix: this.viewProjectionMatrix,
			latestLightDir: this.latestLightDir,
			buffer: this.buffer,
			size: this.size,
			params: this.params,
		});
	}

	private static _calculateShadowFactor(ctx: ShadowMapContext): number {
		const {
			worldPoint,
			normal,
			viewProjectionMatrix,
			latestLightDir,
			buffer,
			size,
			params,
		} = ctx;

		if (!viewProjectionMatrix) return 1.0;

		const N = Vector3.normalize(normal || { x: 0, y: 1, z: 0 });
		const L = Vector3.normalize({
			x: -latestLightDir.x,
			y: -latestLightDir.y,
			z: -latestLightDir.z,
		});
		const cosTheta = Math.max(0, Vector3.dot(N, L));

		const normalBias = params.shadowNormalBias ?? 1.0;
		const normalBiasMin = params.shadowNormalBiasMin ?? 0.05;
		const normalOffset =
			normalBiasMin + (normalBias - normalBiasMin) * (1.0 - cosTheta);
		const offsetPoint = {
			x: worldPoint.x + N.x * normalOffset,
			y: worldPoint.y + N.y * normalOffset,
			z: worldPoint.z + N.z * normalOffset,
		};

		const lightSpacePos = Matrix4.transformPoint(
			viewProjectionMatrix,
			offsetPoint
		);
		const w = lightSpacePos.w;
		if (w <= ShadowConstants.MIN_CLIP_W) return 1.0;
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
			return 1.0;
		}

		const constantBias = params.shadowBias ?? 0.008;
		const slopeBias = params.shadowSlopeBias ?? 0.03;
		const texelBias = (params.shadowTexelBias ?? 1.0) * (2.0 / size);
		const maxBias = params.shadowMaxBias ?? 0.05;
		const bias = Math.min(
			maxBias,
			constantBias + slopeBias * (1.0 - cosTheta) + texelBias
		);

		if (params.shadowPCF === undefined || params.shadowPCF <= 0) {
			return ShadowMap._sampleShadowStatic(
				u,
				v,
				currentDepth,
				bias,
				buffer,
				size,
				params
			);
		}

		return ShadowMap._samplePCFStatic(
			u,
			v,
			currentDepth,
			bias,
			buffer,
			size,
			params
		);
	}

	private static _sampleShadowStatic(
		u: number,
		v: number,
		currentDepth: number,
		bias: number,
		buffer: Float32Array,
		size: number,
		params: ShadowParams
	): number {
		// Revert to (size - 1) mapping to match original engine behavior
		const tx = Math.max(0, Math.min(size - 1, Math.floor(u * (size - 1))));
		const ty = Math.max(0, Math.min(size - 1, Math.floor(v * (size - 1))));
		const shadowDepth = buffer[ty * size + tx];

		return currentDepth - bias > shadowDepth ?
				1.0 - (params.shadowStrength ?? 1.0)
			:	1.0;
	}

	private static _samplePCFStatic(
		u: number,
		v: number,
		currentDepth: number,
		bias: number,
		buffer: Float32Array,
		size: number,
		params: ShadowParams
	): number {
		let shadow = 0;
		const samples = Math.max(1, Math.floor(params.shadowPCF ?? 1));
		const texelSize = 1.0 / size;
		const strength = params.shadowStrength ?? 1.0;
		let validSampleCount = 0;

		for (let y = -samples; y <= samples; y++) {
			for (let x = -samples; x <= samples; x++) {
				const su = u + x * texelSize;
				const sv = v + y * texelSize;
				if (su < 0 || su > 1 || sv < 0 || sv > 1) continue;

				const tx = Math.max(0, Math.min(size - 1, Math.floor(su * (size - 1))));
				const ty = Math.max(0, Math.min(size - 1, Math.floor(sv * (size - 1))));
				const shadowDepth = buffer[ty * size + tx];
				validSampleCount++;

				if (currentDepth - bias > shadowDepth) {
					shadow += 1.0;
				}
			}
		}

		if (validSampleCount === 0) return 1.0;
		return 1.0 - (shadow / validSampleCount) * strength;
	}
}
