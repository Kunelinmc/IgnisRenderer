import type { Texture } from "../core/Texture";
import { Matrix4 } from "../maths/Matrix4";
import type { Matrix3Arr, IVector3 } from "../maths/types";
import { Vector3 } from "../maths/Vector3";
import { Light, LightType, type LightParams } from "./Light";

export type ReflectionProbeShape = "sphere" | "box";
export type ReflectionProbeParallaxMode = "off" | "box" | "sphere";

export interface ReflectionProbeRuntimeCache {
	probeToWorldMatrix: Matrix4;
	worldToProbeMatrix: Matrix4;
	worldToProbe3x3: Matrix3Arr;
	probeWorldPosition: IVector3;
	invHalfExtents: IVector3;
	radiusInv: number;
	effectiveBlendDistance: number;
	blendExponent: number;
}

export interface ReflectionProbeParams extends LightParams {
	shape?: ReflectionProbeShape;
	radius?: number;
	halfExtents?: IVector3;
	blendDistance?: number;
	blendExponent?: number;
	parallaxMode?: ReflectionProbeParallaxMode;
	prefilteredMap?: Texture | null;
}

const REFLECTION_PROBE_NUMERIC_EPSILON = 1e-6;
const REFLECTION_PROBE_MIN_BLEND_DISTANCE = 0.01;
const REFLECTION_PROBE_BLEND_FLOOR_RATIO = 0.1;

export class ReflectionProbe extends Light<LightType.ReflectionProbe> {
	public shape: ReflectionProbeShape;
	public radius: number;
	public halfExtents: Vector3;
	public blendDistance: number;
	public blendExponent: number;
	public parallaxMode: ReflectionProbeParallaxMode;
	public prefilteredMap: Texture | null;

	private _runtimeCache: ReflectionProbeRuntimeCache;
	private _runtimeDirty = true;
	private _matrixSignature = new Float32Array(16);
	private _lastShape: ReflectionProbeShape;
	private _lastRadius: number;
	private _lastHalfExtents = new Float32Array(3);
	private _lastBlendDistance: number;
	private _lastBlendExponent: number;
	private _lastParallaxMode: ReflectionProbeParallaxMode;

	constructor(params: ReflectionProbeParams = {}) {
		super(LightType.ReflectionProbe, params);
		this.shape = params.shape ?? "sphere";
		this.radius = Math.max(
			REFLECTION_PROBE_NUMERIC_EPSILON,
			params.radius ?? 5
		);
		this.halfExtents = new Vector3();
		this.halfExtents.copy(params.halfExtents ?? { x: 5, y: 5, z: 5 });
		this.blendDistance = Math.max(0, params.blendDistance ?? 0.15);
		this.blendExponent = sanitizeBlendExponent(params.blendExponent ?? 1);
		this.parallaxMode =
			params.parallaxMode ?? (this.shape === "box" ? "box" : "off");
		this.prefilteredMap = params.prefilteredMap ?? null;

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
			blendExponent: this.blendExponent,
		};

		this._lastShape = this.shape;
		this._lastRadius = this.radius;
		this._lastHalfExtents[0] = this.halfExtents.x;
		this._lastHalfExtents[1] = this.halfExtents.y;
		this._lastHalfExtents[2] = this.halfExtents.z;
		this._lastBlendDistance = this.blendDistance;
		this._lastBlendExponent = this.blendExponent;
		this._lastParallaxMode = this.parallaxMode;
		for (let i = 0; i < this._matrixSignature.length; i++) {
			this._matrixSignature[i] = Number.NaN;
		}
	}

	public markRuntimeDirty(): void {
		this._runtimeDirty = true;
	}

	public refreshRuntimeCache(): void {
		this._updateRuntimeCache();
	}

	public getRuntimeCache(): ReflectionProbeRuntimeCache {
		if (this._runtimeDirty || this._runtimeStateChanged()) {
			this._updateRuntimeCache();
		}
		return this._runtimeCache;
	}

	protected override _copyClonePropertiesTo(target: this): void {
		super._copyClonePropertiesTo(target);
		target.shape = this.shape;
		target.radius = this.radius;
		target.halfExtents.copy(this.halfExtents);
		target.blendDistance = this.blendDistance;
		target.blendExponent = this.blendExponent;
		target.parallaxMode = this.parallaxMode;
		target.prefilteredMap = this.prefilteredMap;
		target.markRuntimeDirty();
	}

	private _runtimeStateChanged(): boolean {
		if (this.shape !== this._lastShape) return true;
		if (this.radius !== this._lastRadius) return true;
		if (this.halfExtents.x !== this._lastHalfExtents[0]) return true;
		if (this.halfExtents.y !== this._lastHalfExtents[1]) return true;
		if (this.halfExtents.z !== this._lastHalfExtents[2]) return true;
		if (this.blendDistance !== this._lastBlendDistance) return true;
		if (this.blendExponent !== this._lastBlendExponent) return true;
		if (this.parallaxMode !== this._lastParallaxMode) return true;

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
		copyMatrix(this._runtimeCache.probeToWorldMatrix, world);

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

		const probePosition = this._runtimeCache.probeWorldPosition;
		const worldPosition = Matrix4.transformPoint(world, { x: 0, y: 0, z: 0 });
		probePosition.x = worldPosition.x;
		probePosition.y = worldPosition.y;
		probePosition.z = worldPosition.z;

		const safeHalfX = Math.max(
			Math.abs(this.halfExtents.x),
			REFLECTION_PROBE_NUMERIC_EPSILON
		);
		const safeHalfY = Math.max(
			Math.abs(this.halfExtents.y),
			REFLECTION_PROBE_NUMERIC_EPSILON
		);
		const safeHalfZ = Math.max(
			Math.abs(this.halfExtents.z),
			REFLECTION_PROBE_NUMERIC_EPSILON
		);
		this._runtimeCache.invHalfExtents.x = 1 / safeHalfX;
		this._runtimeCache.invHalfExtents.y = 1 / safeHalfY;
		this._runtimeCache.invHalfExtents.z = 1 / safeHalfZ;

		const safeRadius = Math.max(
			Math.abs(this.radius),
			REFLECTION_PROBE_NUMERIC_EPSILON
		);
		this._runtimeCache.radiusInv = 1 / safeRadius;
		this._runtimeCache.blendExponent = sanitizeBlendExponent(this.blendExponent);

		const probeSizeMetric =
			this.shape === "box" ? Math.max(safeHalfX, safeHalfY, safeHalfZ) : safeRadius;
		const blendFloor = Math.max(
			REFLECTION_PROBE_MIN_BLEND_DISTANCE,
			probeSizeMetric * REFLECTION_PROBE_BLEND_FLOOR_RATIO
		);
		this._runtimeCache.effectiveBlendDistance = Math.max(
			0,
			this.blendDistance,
			blendFloor
		);

		this._lastShape = this.shape;
		this._lastRadius = this.radius;
		this._lastHalfExtents[0] = this.halfExtents.x;
		this._lastHalfExtents[1] = this.halfExtents.y;
		this._lastHalfExtents[2] = this.halfExtents.z;
		this._lastBlendDistance = this.blendDistance;
		this._lastBlendExponent = this.blendExponent;
		this._lastParallaxMode = this.parallaxMode;
		copyMatrixSignature(this._matrixSignature, this.worldMatrix);
		this._runtimeDirty = false;
	}
}

function sanitizeBlendExponent(value: number): number {
	if (!Number.isFinite(value)) return 1;
	return Math.max(0.01, value);
}

function copyMatrix(target: Matrix4, source: Matrix4): void {
	const targetElements = target.elements;
	const sourceElements = source.elements;
	for (let row = 0; row < 4; row++) {
		targetElements[row][0] = sourceElements[row][0];
		targetElements[row][1] = sourceElements[row][1];
		targetElements[row][2] = sourceElements[row][2];
		targetElements[row][3] = sourceElements[row][3];
	}
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
