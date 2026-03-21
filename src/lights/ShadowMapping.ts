/**
 * Shared shadow metadata definitions.
 *
 * Runtime shadow buffers and sampling logic are implemented inside each backend.
 */

import type { Matrix4 } from "../maths/Matrix4";
import type { IVector3 } from "../maths/types";

export interface ShadowParams {
	shadowBias?: number;
	shadowSlopeBias?: number;
	shadowNormalBias?: number;
	shadowNormalBiasMin?: number;
	shadowTexelBias?: number;
	shadowMaxBias?: number;
	shadowPCF?: number;
	shadowStrength?: number;
	shadowRadius?: number;
	shadowSamples?: number;
	shadowSearchSamples?: number;
	[key: string]: unknown;
}

const DEFAULT_SHADOW_PARAMS: ShadowParams = {
	shadowBias: 0.008,
	shadowSlopeBias: 0.03,
	shadowNormalBias: 1.0,
	shadowNormalBiasMin: 0.05,
	shadowTexelBias: 1.0,
	shadowMaxBias: 0.05,
	shadowPCF: 1,
	shadowStrength: 1,
};

export class ShadowMap {
	public size: number;
	public params: ShadowParams;
	public viewMatrix: Matrix4 | null = null;
	public projectionMatrix: Matrix4 | null = null;
	public viewProjectionMatrix: Matrix4 | null = null;
	public latestLightDir: IVector3 = { x: 0, y: -1, z: 0 };
	public stabilizedBoundsRadius: number | null = null;

	constructor(size = 1024, params: ShadowParams = {}) {
		this.size = size;
		this.params = {
			...DEFAULT_SHADOW_PARAMS,
			...params,
		};
	}
}
