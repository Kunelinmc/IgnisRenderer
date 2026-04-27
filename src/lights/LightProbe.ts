import { Matrix4 } from "../maths/Matrix4";
import { SH } from "../maths/SH";
import type { IVector3, Matrix3Arr, SHCoefficients } from "../maths/types";
import { Vector3 } from "../maths/Vector3";
import { Light, LightType, type LightParams } from "./Light";

export type LightProbeShape = "global" | "sphere" | "box";

export interface LightProbeRuntimeCache {
	probeToWorldMatrix: Matrix4;
	worldToProbeMatrix: Matrix4;
	worldToProbe3x3: Matrix3Arr;
	probeWorldPosition: IVector3;
	invHalfExtents: IVector3;
	radiusInv: number;
	effectiveBlendDistance: number;
	priority: number;
}

export interface LightProbeParams extends LightParams {
	sh?: SHCoefficients | null;
	shape?: LightProbeShape;
	radius?: number;
	halfExtents?: IVector3;
	blendDistance?: number;
	priority?: number;
}

const LIGHT_PROBE_NUMERIC_EPSILON = 1e-6;
const LIGHT_PROBE_MIN_BLEND_DISTANCE = 0.01;
const LIGHT_PROBE_BLEND_FLOOR_RATIO = 0.1;

/**
 * LightProbe stores spherical harmonics coefficients for diffuse irradiance.
 * Baking/projection from environment maps lives in pipeline helpers.
 */
export class LightProbe extends Light<LightType.LightProbe> {
	public sh: SHCoefficients;
	public shape: LightProbeShape;
	public radius: number;
	public halfExtents: Vector3;
	public blendDistance: number;
	public priority: number;

	private _runtimeCache: LightProbeRuntimeCache;
	private _runtimeDirty = true;
	private _matrixSignature = new Float32Array(16);
	private _lastShape: LightProbeShape;
	private _lastRadius: number;
	private _lastHalfExtents = new Float32Array(3);
	private _lastBlendDistance: number;
	private _lastPriority: number;

	constructor();
	constructor(params: LightProbeParams);
	constructor(sh: SHCoefficients | null, intensity?: number);
	constructor(
		shOrParams: LightProbeParams | SHCoefficients | null = null,
		intensity = 1.0
	) {
		const params =
			isLightProbeParams(shOrParams) ?
				shOrParams
			:	{
					sh: shOrParams,
					intensity,
				};
		super(LightType.LightProbe, params);
		this.sh = cloneSHCoefficients(params.sh);
		this.shape = sanitizeLightProbeShape(params.shape ?? "global");
		this.radius = sanitizeLightProbeRadius(params.radius ?? 5);
		this.halfExtents = new Vector3();
		this.halfExtents.copy(sanitizeHalfExtents(params.halfExtents));
		this.blendDistance = sanitizeLightProbeBlendDistance(
			params.blendDistance ?? 0.15
		);
		this.priority = sanitizeLightProbePriority(params.priority ?? 0);

		this._runtimeCache = {
			probeToWorldMatrix: Matrix4.identity(),
			worldToProbeMatrix: Matrix4.identity(),
			worldToProbe3x3: [
				[1, 0, 0],
				[0, 1, 0],
				[0, 0, 1],
			],
			probeWorldPosition: { x: 0, y: 0, z: 0 },
			invHalfExtents: { x: 1, y: 1, z: 1 },
			radiusInv: 1,
			effectiveBlendDistance: this.blendDistance,
			priority: this.priority,
		};

		this._lastShape = this.shape;
		this._lastRadius = this.radius;
		this._lastHalfExtents[0] = this.halfExtents.x;
		this._lastHalfExtents[1] = this.halfExtents.y;
		this._lastHalfExtents[2] = this.halfExtents.z;
		this._lastBlendDistance = this.blendDistance;
		this._lastPriority = this.priority;
		for (let i = 0; i < this._matrixSignature.length; i++) {
			this._matrixSignature[i] = Number.NaN;
		}
	}

	public copy(source: LightProbe | SHCoefficients): LightProbe {
		const sourceSH = source instanceof LightProbe ? source.sh : source;
		for (let i = 0; i < this.sh.length; i++) {
			const coefficient = sourceSH[i];
			this.sh[i].r = coefficient?.r ?? 0;
			this.sh[i].g = coefficient?.g ?? 0;
			this.sh[i].b = coefficient?.b ?? 0;
		}

		if (source instanceof LightProbe) {
			this.intensity = source.intensity;
			this.shape = sanitizeLightProbeShape(source.shape);
			this.radius = sanitizeLightProbeRadius(source.radius);
			this.halfExtents.copy(sanitizeHalfExtents(source.halfExtents));
			this.blendDistance = sanitizeLightProbeBlendDistance(
				source.blendDistance
			);
			this.priority = sanitizeLightProbePriority(source.priority);
			this.markRuntimeDirty();
		}

		return this;
	}

	public markRuntimeDirty(): void {
		this._runtimeDirty = true;
	}

	public refreshRuntimeCache(): void {
		this._updateRuntimeCache();
	}

	public getRuntimeCache(): LightProbeRuntimeCache {
		if (this._runtimeDirty || this._runtimeStateChanged()) {
			this._updateRuntimeCache();
		}
		return this._runtimeCache;
	}

	protected override _copyClonePropertiesTo(target: this): void {
		super._copyClonePropertiesTo(target);
		target.sh = cloneSHCoefficients(this.sh);
		target.shape = this.shape;
		target.radius = this.radius;
		target.halfExtents.copy(this.halfExtents);
		target.blendDistance = this.blendDistance;
		target.priority = this.priority;
		target.markRuntimeDirty();
	}

	private _runtimeStateChanged(): boolean {
		if (this.shape !== this._lastShape) return true;
		if (this.radius !== this._lastRadius) return true;
		if (this.halfExtents.x !== this._lastHalfExtents[0]) return true;
		if (this.halfExtents.y !== this._lastHalfExtents[1]) return true;
		if (this.halfExtents.z !== this._lastHalfExtents[2]) return true;
		if (this.blendDistance !== this._lastBlendDistance) return true;
		if (this.priority !== this._lastPriority) return true;

		const elements = this.worldMatrix.elements;
		let cursor = 0;
		for (let row = 0; row < 4; row++) {
			for (let col = 0; col < 4; col++) {
				if (elements[row][col] !== this._matrixSignature[cursor]) {
					return true;
				}
				cursor++;
			}
		}
		return false;
	}

	private _updateRuntimeCache(): void {
		const world = this.worldMatrix;
		world.copyTo(this._runtimeCache.probeToWorldMatrix);

		const inverse3x3 = Matrix4.inverse3x3(world) ?? [
			[1, 0, 0],
			[0, 1, 0],
			[0, 0, 1],
		];
		copyMatrix3(this._runtimeCache.worldToProbe3x3, inverse3x3);

		const worldToProbe = this._runtimeCache.worldToProbeMatrix.elements;
		worldToProbe[0][0] = inverse3x3[0][0];
		worldToProbe[0][1] = inverse3x3[0][1];
		worldToProbe[0][2] = inverse3x3[0][2];
		worldToProbe[1][0] = inverse3x3[1][0];
		worldToProbe[1][1] = inverse3x3[1][1];
		worldToProbe[1][2] = inverse3x3[1][2];
		worldToProbe[2][0] = inverse3x3[2][0];
		worldToProbe[2][1] = inverse3x3[2][1];
		worldToProbe[2][2] = inverse3x3[2][2];
		worldToProbe[3][0] = 0;
		worldToProbe[3][1] = 0;
		worldToProbe[3][2] = 0;
		worldToProbe[3][3] = 1;

		const worldElements = world.elements;
		const tx = worldElements[0][3];
		const ty = worldElements[1][3];
		const tz = worldElements[2][3];
		worldToProbe[0][3] =
			-(inverse3x3[0][0] * tx +
				inverse3x3[0][1] * ty +
				inverse3x3[0][2] * tz);
		worldToProbe[1][3] =
			-(inverse3x3[1][0] * tx +
				inverse3x3[1][1] * ty +
				inverse3x3[1][2] * tz);
		worldToProbe[2][3] =
			-(inverse3x3[2][0] * tx +
				inverse3x3[2][1] * ty +
				inverse3x3[2][2] * tz);

		const probePosition = Matrix4.transformPoint(world, { x: 0, y: 0, z: 0 });
		this._runtimeCache.probeWorldPosition.x = probePosition.x;
		this._runtimeCache.probeWorldPosition.y = probePosition.y;
		this._runtimeCache.probeWorldPosition.z = probePosition.z;

		const safeHalfExtents = sanitizeHalfExtents(this.halfExtents);
		this._runtimeCache.invHalfExtents.x = 1 / safeHalfExtents.x;
		this._runtimeCache.invHalfExtents.y = 1 / safeHalfExtents.y;
		this._runtimeCache.invHalfExtents.z = 1 / safeHalfExtents.z;

		const safeRadius = sanitizeLightProbeRadius(this.radius);
		this._runtimeCache.radiusInv = 1 / safeRadius;
		this._runtimeCache.priority = sanitizeLightProbePriority(this.priority);

		const resolvedShape = sanitizeLightProbeShape(this.shape);
		const probeSizeMetric =
			resolvedShape === "box" ?
				Math.max(
					safeHalfExtents.x,
					safeHalfExtents.y,
					safeHalfExtents.z
				)
			: safeRadius;
		const blendFloor = Math.max(
			LIGHT_PROBE_MIN_BLEND_DISTANCE,
			probeSizeMetric * LIGHT_PROBE_BLEND_FLOOR_RATIO
		);
		this._runtimeCache.effectiveBlendDistance = Math.max(
			0,
			sanitizeLightProbeBlendDistance(this.blendDistance),
			blendFloor
		);

		this._lastShape = this.shape;
		this._lastRadius = this.radius;
		this._lastHalfExtents[0] = this.halfExtents.x;
		this._lastHalfExtents[1] = this.halfExtents.y;
		this._lastHalfExtents[2] = this.halfExtents.z;
		this._lastBlendDistance = this.blendDistance;
		this._lastPriority = this.priority;
		copyMatrixSignature(this._matrixSignature, this.worldMatrix);
		this._runtimeDirty = false;
	}
}

function isLightProbeParams(
	value: LightProbeParams | SHCoefficients | null | undefined
): value is LightProbeParams {
	return value !== null && value !== undefined && !Array.isArray(value);
}

function cloneSHCoefficients(coefficients?: SHCoefficients | null): SHCoefficients {
	if (!coefficients) {
		return SH.empty();
	}

	return coefficients.map((coefficient) => ({
		r: coefficient.r,
		g: coefficient.g,
		b: coefficient.b,
	}));
}

function sanitizeLightProbeShape(value: LightProbeShape): LightProbeShape {
	if (value === "global" || value === "sphere" || value === "box") {
		return value;
	}
	return "global";
}

function sanitizeLightProbeRadius(value: number): number {
	if (!Number.isFinite(value)) return 5;
	return Math.max(LIGHT_PROBE_NUMERIC_EPSILON, Math.abs(value));
}

function sanitizeHalfExtents(value: IVector3 | undefined): IVector3 {
	const x = Number.isFinite(value?.x) ? Math.abs(value.x) : 5;
	const y = Number.isFinite(value?.y) ? Math.abs(value.y) : 5;
	const z = Number.isFinite(value?.z) ? Math.abs(value.z) : 5;
	return {
		x: Math.max(LIGHT_PROBE_NUMERIC_EPSILON, x),
		y: Math.max(LIGHT_PROBE_NUMERIC_EPSILON, y),
		z: Math.max(LIGHT_PROBE_NUMERIC_EPSILON, z),
	};
}

function sanitizeLightProbeBlendDistance(value: number): number {
	if (!Number.isFinite(value)) return 0.15;
	return Math.max(0, value);
}

function sanitizeLightProbePriority(value: number): number {
	if (!Number.isFinite(value)) return 0;
	return Math.max(-2147483647, Math.min(2147483647, Math.trunc(value)));
}

function copyMatrix3(target: Matrix3Arr, source: Matrix3Arr): void {
	target[0][0] = source[0][0];
	target[0][1] = source[0][1];
	target[0][2] = source[0][2];
	target[1][0] = source[1][0];
	target[1][1] = source[1][1];
	target[1][2] = source[1][2];
	target[2][0] = source[2][0];
	target[2][1] = source[2][1];
	target[2][2] = source[2][2];
}

function copyMatrixSignature(target: Float32Array, matrix: Matrix4): void {
	const elements = matrix.elements;
	let cursor = 0;
	for (let row = 0; row < 4; row++) {
		for (let col = 0; col < 4; col++) {
			target[cursor++] = elements[row][col];
		}
	}
}
