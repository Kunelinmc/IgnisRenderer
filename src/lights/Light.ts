import { Matrix4 } from "../maths/Matrix4";
import type { IVector3, SHCoefficients } from "../maths/types";
import type { RGB } from "../utils/Color";

export enum LightType {
	Ambient = "ambient",
	Directional = "directional",
	Point = "point",
	Spot = "spot",
	LightProbe = "lightProbe",
}

export interface ShadowCameraResult {
	view: Matrix4;
	projection: Matrix4;
	lightDir: IVector3;
}

export interface ShadowCaster {
	setupShadowCamera(ctx: {
		sceneBounds: { center: IVector3; radius: number };
		worldMatrix: Matrix4;
	}): ShadowCameraResult | null;
}

export interface LightParams {
	color?: RGB;
	intensity?: number;
	castShadow?: boolean;
}

/**
 * Result of a light's contribution to a specific point in the scene
 */
export interface LightContribution {
	type: "ambient" | "direct" | "irradiance";
	// Base light color in display (sRGB-like) domain.
	// Linear intensity/attenuation is carried separately by `intensity`.
	color: RGB;
	// Scalar intensity in linear domain. Includes light intensity and any
	// distance/cone attenuation terms. Defaults to 1 when omitted.
	intensity?: number;
	direction?: IVector3; // Direction towards the light (L vector)
}

export interface LightProbeLike {
	type: LightType.LightProbe;
	sh: SHCoefficients;
	intensity: number;
}

export interface SurfacePoint {
	position: IVector3;
	normal?: IVector3;
}

export abstract class Light<TType extends LightType = LightType> {
	public readonly type: TType;
	public color: RGB;
	public intensity: number;
	public castShadow: boolean;

	/**
	 * World matrix of the light, updated once per frame if needed.
	 */
	private _worldMatrix: Matrix4 = Matrix4.identity();

	public get worldMatrix(): Matrix4 {
		return this._worldMatrix;
	}

	protected constructor(type: TType, params: LightParams = {}) {
		this.type = type;
		this.color = params.color ?? { r: 255, g: 255, b: 255 };
		this.intensity = params.intensity ?? 1.0;
		this.castShadow = params.castShadow ?? false;
	}

	/**
	 * Update the world matrix for this light.
	 * This should be called once per frame if the light or its parent transforms.
	 */
	public updateWorldMatrix(matrix: Matrix4): void {
		this._worldMatrix = matrix;
	}

	/**
	 * Compute the light's contribution to a specific point.
	 * Uses the internal worldMatrix for shared world-space calculations.
	 * Returns null if the light has no effect (e.g., out of range or outside cone).
	 */
	abstract computeContribution(surface: SurfacePoint): LightContribution | null;

	/**
	 * Validate and return the required world-space sample position.
	 */
	protected _requireSurfacePosition(surface: SurfacePoint): IVector3 {
		return surface?.position || { x: 0, y: 0, z: 0 };
	}

	/**
	 * Reference to shadow caster logic if this light supports it.
	 */
	public shadow?: ShadowCaster;
}
