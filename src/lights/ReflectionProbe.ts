import type { Texture } from "../core/Texture";
import { Matrix3 } from "../maths/Matrix3";
import { Matrix4 } from "../maths/Matrix4";
import type { IVector3 } from "../maths/types";
import { Vector3 } from "../maths/Vector3";
import { Light, LightType, type LightParams } from "./Light";
import { ReflectionProbeCaptureController } from "./ProbeCaptureController";

export type ReflectionProbeShape = "sphere" | "box";
export type ReflectionProbeParallaxMode = "off" | "box" | "sphere";
export type ReflectionProbeSource =
	| "environment"
	| "capturedScene"
	| "manual";
export type ReflectionProbeCaptureUpdateMode =
	| "manual"
	| "onSceneDirty"
	| "interval";

export interface ReflectionProbeCaptureResolution {
	width: number;
	height: number;
}

export interface ReflectionProbeRuntimeCache {
	probeToWorldMatrix: Matrix4;
	worldToProbeMatrix: Matrix4;
	worldToProbe3x3: Matrix3;
	probeWorldPosition: IVector3;
	captureWorldPosition: IVector3;
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
	source?: ReflectionProbeSource;
	captureUpdateMode?: ReflectionProbeCaptureUpdateMode;
	captureIntervalSeconds?: number;
	captureResolution?: Partial<ReflectionProbeCaptureResolution>;
	captureFar?: number;
	includeEnvironment?: boolean;
	includeMeshes?: boolean;
	includeTransparent?: boolean;
	includeParticles?: boolean;
	includeShadows?: boolean;
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
	public source: ReflectionProbeSource;
	public captureUpdateMode: ReflectionProbeCaptureUpdateMode;
	public captureIntervalSeconds: number;
	public captureResolution: ReflectionProbeCaptureResolution;
	public captureFar: number;
	public includeEnvironment: boolean;
	public includeMeshes: boolean;
	public includeTransparent: boolean;
	public includeParticles: boolean;
	public includeShadows: boolean;
	public readonly capture: ReflectionProbeCaptureController;

	private _runtimeCache: ReflectionProbeRuntimeCache;
	private _runtimeDirty = true;
	private _matrixSignature = new Float32Array(16);
	private _lastShape: ReflectionProbeShape;
	private _lastRadius: number;
	private _lastHalfExtents = new Float32Array(3);
	private _lastBlendDistance: number;
	private _lastBlendExponent: number;
	private _lastParallaxMode: ReflectionProbeParallaxMode;
	private _captureRequestToken = 0;
	private _captureRevision = 0;

	constructor(params: ReflectionProbeParams = {}) {
		super(LightType.ReflectionProbe, params);
		this.shape = params.shape ?? "sphere";
		this.radius = Math.max(REFLECTION_PROBE_NUMERIC_EPSILON, params.radius ?? 5);
		this.halfExtents = new Vector3();
		this.halfExtents.copy(params.halfExtents ?? { x: 5, y: 5, z: 5 });
		this.blendDistance = Math.max(0, params.blendDistance ?? 0.15);
		this.blendExponent = sanitizeBlendExponent(params.blendExponent ?? 1);
		this.parallaxMode = params.parallaxMode ?? (this.shape === "box" ? "box" : "off");
		this.prefilteredMap = params.prefilteredMap ?? null;
		this.source = sanitizeReflectionProbeSource(params.source ?? "environment");
		this.captureUpdateMode = sanitizeCaptureUpdateMode(
			params.captureUpdateMode ?? "onSceneDirty",
		);
		this.captureIntervalSeconds = sanitizeCaptureIntervalSeconds(
			params.captureIntervalSeconds ?? 1,
		);
		this.captureResolution = sanitizeCaptureResolution(params.captureResolution);
		this.captureFar = sanitizeCaptureFar(params.captureFar ?? 200);
		this.includeEnvironment = params.includeEnvironment ?? true;
		this.includeMeshes = params.includeMeshes ?? true;
		this.includeTransparent = params.includeTransparent ?? true;
		this.includeParticles = params.includeParticles ?? true;
		this.includeShadows = params.includeShadows ?? true;
		this.capture = new ReflectionProbeCaptureController();

		this._runtimeCache = {
			probeToWorldMatrix: Matrix4.identity(),
			worldToProbeMatrix: Matrix4.identity(),
			worldToProbe3x3: Matrix3.identity(),
			probeWorldPosition: { x: 0, y: 0, z: 0 },
			captureWorldPosition: { x: 0, y: 0, z: 0 },
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

	public requestCapture(): void {
		this._captureRequestToken++;
		this._captureRevision++;
	}

	public get captureRequestToken(): number {
		return this._captureRequestToken;
	}

	public get captureRevision(): number {
		return this._captureRevision;
	}

	public markCaptureUpdated(): void {
		this._captureRevision++;
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
		target.source = this.source;
		target.captureUpdateMode = this.captureUpdateMode;
		target.captureIntervalSeconds = this.captureIntervalSeconds;
		target.captureResolution.width = this.captureResolution.width;
		target.captureResolution.height = this.captureResolution.height;
		target.captureFar = this.captureFar;
		target.includeEnvironment = this.includeEnvironment;
		target.includeMeshes = this.includeMeshes;
		target.includeTransparent = this.includeTransparent;
		target.includeParticles = this.includeParticles;
		target.includeShadows = this.includeShadows;
		target._captureRequestToken = this._captureRequestToken;
		target._captureRevision = this._captureRevision;
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
		world.copyTo(this._runtimeCache.probeToWorldMatrix);

		const inverse3x3 = Matrix4.inverse3x3(world, this._runtimeCache.worldToProbe3x3);
		if (inverse3x3) {
			this._runtimeCache.worldToProbe3x3.copy(inverse3x3);
		} else {
			Matrix3.identity().copyTo(this._runtimeCache.worldToProbe3x3);
		}
		const inverseElements = this._runtimeCache.worldToProbe3x3.elements;

		const worldToProbe = this._runtimeCache.worldToProbeMatrix.elements;
		worldToProbe[0][0] = inverseElements[0][0];
		worldToProbe[0][1] = inverseElements[0][1];
		worldToProbe[0][2] = inverseElements[0][2];
		worldToProbe[1][0] = inverseElements[1][0];
		worldToProbe[1][1] = inverseElements[1][1];
		worldToProbe[1][2] = inverseElements[1][2];
		worldToProbe[2][0] = inverseElements[2][0];
		worldToProbe[2][1] = inverseElements[2][1];
		worldToProbe[2][2] = inverseElements[2][2];
		worldToProbe[3][0] = 0;
		worldToProbe[3][1] = 0;
		worldToProbe[3][2] = 0;
		worldToProbe[3][3] = 1;

		const worldElements = world.elements;
		const tx = worldElements[0][3];
		const ty = worldElements[1][3];
		const tz = worldElements[2][3];
		worldToProbe[0][3] = -(
			inverseElements[0][0] * tx +
			inverseElements[0][1] * ty +
			inverseElements[0][2] * tz
		);
		worldToProbe[1][3] = -(
			inverseElements[1][0] * tx +
			inverseElements[1][1] * ty +
			inverseElements[1][2] * tz
		);
		worldToProbe[2][3] = -(
			inverseElements[2][0] * tx +
			inverseElements[2][1] * ty +
			inverseElements[2][2] * tz
		);

		const probePosition = this._runtimeCache.probeWorldPosition;
		const worldPosition = Matrix4.transformPoint(world, { x: 0, y: 0, z: 0 });
		probePosition.x = worldPosition.x;
		probePosition.y = worldPosition.y;
		probePosition.z = worldPosition.z;
		resolveCaptureWorldPosition(this, this._runtimeCache.captureWorldPosition);

		const safeHalfX = Math.max(Math.abs(this.halfExtents.x), REFLECTION_PROBE_NUMERIC_EPSILON);
		const safeHalfY = Math.max(Math.abs(this.halfExtents.y), REFLECTION_PROBE_NUMERIC_EPSILON);
		const safeHalfZ = Math.max(Math.abs(this.halfExtents.z), REFLECTION_PROBE_NUMERIC_EPSILON);
		this._runtimeCache.invHalfExtents.x = 1 / safeHalfX;
		this._runtimeCache.invHalfExtents.y = 1 / safeHalfY;
		this._runtimeCache.invHalfExtents.z = 1 / safeHalfZ;

		const safeRadius = Math.max(Math.abs(this.radius), REFLECTION_PROBE_NUMERIC_EPSILON);
		this._runtimeCache.radiusInv = 1 / safeRadius;
		this._runtimeCache.blendExponent = sanitizeBlendExponent(this.blendExponent);

		const probeSizeMetric =
			this.shape === "box" ? Math.max(safeHalfX, safeHalfY, safeHalfZ) : safeRadius;
		const blendFloor = Math.max(
			REFLECTION_PROBE_MIN_BLEND_DISTANCE,
			probeSizeMetric * REFLECTION_PROBE_BLEND_FLOOR_RATIO,
		);
		this._runtimeCache.effectiveBlendDistance = Math.max(0, this.blendDistance, blendFloor);

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

function sanitizeReflectionProbeSource(
	value: ReflectionProbeSource
): ReflectionProbeSource {
	if (
		value === "environment" ||
		value === "capturedScene" ||
		value === "manual"
	) {
		return value;
	}
	return "environment";
}

function sanitizeCaptureUpdateMode(
	value: ReflectionProbeCaptureUpdateMode
): ReflectionProbeCaptureUpdateMode {
	if (value === "manual" || value === "onSceneDirty" || value === "interval") {
		return value;
	}
	return "onSceneDirty";
}

function sanitizeCaptureIntervalSeconds(value: number): number {
	if (!Number.isFinite(value)) return 1;
	return Math.max(0.01, value);
}

function sanitizeCaptureFar(value: number): number {
	if (!Number.isFinite(value)) return 200;
	return Math.max(1, value);
}

function resolveCaptureWorldPosition(
	probe: ReflectionProbe,
	out: IVector3
): void {
	const parent = probe.parent;
	const sceneRoot = probe.scene?.root ?? null;
	const captureNode =
		parent && sceneRoot && parent === sceneRoot ? probe : (parent ?? probe);
	const elements = captureNode.worldMatrix.elements;
	out.x = elements[0][3];
	out.y = elements[1][3];
	out.z = elements[2][3];
}

function sanitizeCaptureResolution(
	value: Partial<ReflectionProbeCaptureResolution> | undefined
): ReflectionProbeCaptureResolution {
	const width = Math.max(8, Math.floor(value?.width ?? 512));
	const height = Math.max(4, Math.floor(value?.height ?? 256));
	return { width, height };
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
