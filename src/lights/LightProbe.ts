import { Matrix3 } from "../maths/Matrix3";
import { Matrix4 } from "../maths/Matrix4";
import { SH } from "../maths/SH";
import type { IVector3, SHCoefficients } from "../maths/types";
import { Vector3 } from "../maths/Vector3";
import { Light, LightType, type LightParams } from "./Light";
import { ProbeCaptureController } from "./runtime/ProbeCaptureController";

export type LightProbeShape = "global" | "sphere" | "box";
export type LightProbeSource =
	| "environment"
	| "capturedScene"
	| "manual";

export interface LightProbeCaptureResolution {
	width: number;
	height: number;
}

export interface LightProbeRuntimeCache {
	probeToWorldMatrix: Matrix4;
	worldToProbeMatrix: Matrix4;
	worldToProbe3x3: Matrix3;
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
	source?: LightProbeSource;
	captureResolution?: Partial<LightProbeCaptureResolution>;
	captureFar?: number;
	includeEnvironment?: boolean;
	includeMeshes?: boolean;
	includeTransparent?: boolean;
	includeParticles?: boolean;
	includeShadows?: boolean;
}

const LIGHT_PROBE_NUMERIC_EPSILON = 1e-6;
const LIGHT_PROBE_MIN_BLEND_DISTANCE = 0.01;
const LIGHT_PROBE_BLEND_FLOOR_RATIO = 0.1;
const SH_COEFFICIENT_COMPONENTS = new Set<PropertyKey>(["r", "g", "b"]);

/**
 * LightProbe stores spherical harmonics coefficients for diffuse irradiance.
 * Baking/projection from environment maps lives in pipeline helpers.
 */
export class LightProbe extends Light<LightType.LightProbe> {
	public shape: LightProbeShape;
	public radius: number;
	public halfExtents: Vector3;
	public blendDistance: number;
	public priority: number;
	/** Source that owns the probe coefficients and runtime updates. */
	public source: LightProbeSource;
	/** Equirectangular capture resolution used before SH projection. */
	public captureResolution: LightProbeCaptureResolution;
	/** Far clipping distance for scene captures. */
	public captureFar: number;
	/** Whether capture includes the active environment background. */
	public includeEnvironment: boolean;
	/** Whether capture includes prepared scene meshes. */
	public includeMeshes: boolean;
	/** Whether capture includes transparent draw packets. */
	public includeTransparent: boolean;
	/** Whether capture includes particle render batches. */
	public includeParticles: boolean;
	/** Whether capture includes shadowed lighting where supported. */
	public includeShadows: boolean;
	/** Runtime texture output bindings for captured-scene updates. */
	public readonly capture: ProbeCaptureController;

	private _runtimeCache: LightProbeRuntimeCache;
	private _runtimeDirty = true;
	private _matrixSignature = new Float32Array(16);
	private _lastShape: LightProbeShape;
	private _lastRadius: number;
	private _lastHalfExtents = new Float32Array(3);
	private _lastBlendDistance: number;
	private _lastPriority: number;
	private _captureRequestToken = 0;
	private _captureRevision = 0;
	private _sh: SHCoefficients = SH.empty();
	private _shProxy: SHCoefficients | null = null;
	private _shStorageInitialized = false;
	private _suppressSHMutationTracking = 0;

	/** Mutable probe-owned spherical harmonics coefficients. */
	public get sh(): SHCoefficients {
		if (!this._shProxy) {
			this._shProxy = this._createSHProxy();
		}
		return this._shProxy;
	}

	public set sh(value: SHCoefficients) {
		this._replaceSHStorage(value);
		this._markTrackedSHMutation("lighting");
	}

	/**
	 * Creates a light probe from a single parameter object.
	 *
	 * @param params - Probe initialization parameters. Pass `{}` for defaults
	 * and `{ sh }` for authored spherical harmonics coefficients.
	 * @throws TypeError when `params` is not an object.
	 * @sideEffects Allocates probe SH data and runtime cache state.
	 */
	constructor(params: LightProbeParams) {
		const resolvedParams = validateLightProbeParams(params);
		super(LightType.LightProbe, resolvedParams);
		this._replaceSHStorage(resolvedParams.sh);
		this.shape = sanitizeLightProbeShape(resolvedParams.shape ?? "global");
		this.radius = sanitizeLightProbeRadius(resolvedParams.radius ?? 5);
		this.halfExtents = new Vector3();
		this.halfExtents.copy(sanitizeHalfExtents(resolvedParams.halfExtents));
		this.blendDistance = sanitizeLightProbeBlendDistance(resolvedParams.blendDistance ?? 0.15);
		this.priority = sanitizeLightProbePriority(resolvedParams.priority ?? 0);
		this.source = sanitizeLightProbeSource(resolvedParams.source ?? "environment");
		this.captureResolution = sanitizeLightProbeCaptureResolution(
			resolvedParams.captureResolution,
		);
		this.captureFar = sanitizeLightProbeCaptureFar(resolvedParams.captureFar ?? 200);
		this.includeEnvironment = resolvedParams.includeEnvironment ?? true;
		this.includeMeshes = resolvedParams.includeMeshes ?? true;
		this.includeTransparent = resolvedParams.includeTransparent ?? true;
		this.includeParticles = resolvedParams.includeParticles ?? true;
		this.includeShadows = resolvedParams.includeShadows ?? true;
		this.capture = new ProbeCaptureController();

		this._runtimeCache = {
			probeToWorldMatrix: Matrix4.identity(),
			worldToProbeMatrix: Matrix4.identity(),
			worldToProbe3x3: Matrix3.identity(),
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
		this._shStorageInitialized = true;
	}

	protected override _createCloneInstance(): this {
		return new LightProbe({}) as this;
	}

	public copy(source: LightProbe | SHCoefficients): LightProbe {
		const sourceSH = source instanceof LightProbe ? source.sh : source;
		this._copySHCoefficients(sourceSH, "lighting");

		if (source instanceof LightProbe) {
			this.shape = sanitizeLightProbeShape(source.shape);
			this.radius = sanitizeLightProbeRadius(source.radius);
			this.halfExtents.copy(sanitizeHalfExtents(source.halfExtents));
			this.blendDistance = sanitizeLightProbeBlendDistance(source.blendDistance);
			this.priority = sanitizeLightProbePriority(source.priority);
			this.source = sanitizeLightProbeSource(source.source);
			this.captureResolution = sanitizeLightProbeCaptureResolution(source.captureResolution);
			this.captureFar = sanitizeLightProbeCaptureFar(source.captureFar);
			this.includeEnvironment = source.includeEnvironment;
			this.includeMeshes = source.includeMeshes;
			this.includeTransparent = source.includeTransparent;
			this.includeParticles = source.includeParticles;
			this.includeShadows = source.includeShadows;
			this._captureRevision = source._captureRevision;
			this.markRuntimeDirty();
		}

		return this;
	}

	/**
	 * Requests a fresh captured-scene probe update.
	 *
	 * @returns Nothing.
	 * @sideEffects Increments `captureRequestToken` and `captureRevision`, and
	 * invalidates scene lighting when attached.
	 */
	public requestCapture(): void {
		this._captureRequestToken++;
		this._captureRevision++;
		this.scene?.invalidate("lighting");
	}

	/**
	 * Writes captured spherical harmonics and invalidates the probe-capture path.
	 *
	 * @internal Owned by `ProbeCaptureRuntime`; applications should assign `sh`.
	 * @param coefficients Captured radiance SH coefficients to copy.
	 * @returns Nothing.
	 * @sideEffects Increments `captureRevision` and invalidates the attached scene
	 * with the non-capture-relevant `probe-capture` reason.
	 */
	public writeCapturedSH(coefficients: SHCoefficients): void {
		this._copySHCoefficients(coefficients, "probe-capture");
		this._captureRevision++;
	}

	/**
	 * Monotonic token used by capture runtimes to detect explicit requests.
	 *
	 * @returns The current capture request token.
	 */
	public get captureRequestToken(): number {
		return this._captureRequestToken;
	}

	/**
	 * Monotonic revision incremented when capture data changes or is requested.
	 *
	 * @returns The current capture data revision.
	 */
	public get captureRevision(): number {
		return this._captureRevision;
	}

	/**
	 * Marks the stored capture data as updated by runtime.
	 *
	 * @returns Nothing.
	 * @sideEffects Increments `captureRevision`.
	 */
	public markCaptureUpdated(): void {
		this._captureRevision++;
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

	/**
	 * Computes the normalized probe-volume metric at a world-space position.
	 *
	 * @param worldPosition - Position to evaluate in world space.
	 * @returns The normalized box or sphere distance, or positive infinity for
	 * a global probe.
	 */
	public getMetric(worldPosition: IVector3): number {
		const cache = this.getRuntimeCache();
		const localPosition = Matrix4.transformPoint(
			cache.worldToProbeMatrix,
			worldPosition,
		);

		if (this.shape === "box") {
			return Math.max(
				Math.abs(localPosition.x) * cache.invHalfExtents.x,
				Math.abs(localPosition.y) * cache.invHalfExtents.y,
				Math.abs(localPosition.z) * cache.invHalfExtents.z,
			);
		}

		if (this.shape === "sphere") {
			return (
				Math.hypot(localPosition.x, localPosition.y, localPosition.z) *
				cache.radiusInv
			);
		}

		return Number.POSITIVE_INFINITY;
	}

	protected override _copyClonePropertiesTo(target: this): void {
		super._copyClonePropertiesTo(target);
		target.sh = cloneSHCoefficients(this.sh);
		target.shape = this.shape;
		target.radius = this.radius;
		target.halfExtents.copy(this.halfExtents);
		target.blendDistance = this.blendDistance;
		target.priority = this.priority;
		target.source = this.source;
		target.captureResolution.width = this.captureResolution.width;
		target.captureResolution.height = this.captureResolution.height;
		target.captureFar = this.captureFar;
		target.includeEnvironment = this.includeEnvironment;
		target.includeMeshes = this.includeMeshes;
		target.includeTransparent = this.includeTransparent;
		target.includeParticles = this.includeParticles;
		target.includeShadows = this.includeShadows;
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

	private _replaceSHStorage(coefficients?: SHCoefficients | null): void {
		const cloned = cloneSHCoefficients(coefficients);
		for (let i = 0; i < cloned.length; i++) {
			cloned[i] = this._trackSHCoefficient(cloned[i]);
		}
		this._sh = cloned;
		this._shProxy = null;
	}

	private _createSHProxy(): SHCoefficients {
		return new Proxy(this._sh, {
			set: (target, property, value) => {
				const coefficientIndex = resolveArrayIndexProperty(property);
				if (coefficientIndex === null) {
					const previous = Reflect.get(target, property);
					const applied = Reflect.set(target, property, value);
					if (applied && previous !== value) {
						this._markTrackedSHMutation("lighting");
					}
					return applied;
				}
				target[coefficientIndex] = this._trackSHCoefficient(
					value as SHCoefficients[number]
				);
				this._markTrackedSHMutation("lighting");
				return true;
			},
		});
	}

	private _trackSHCoefficient(
		coefficient: SHCoefficients[number] | undefined
	): SHCoefficients[number] {
		const values = {
			r: coefficient?.r ?? 0,
			g: coefficient?.g ?? 0,
			b: coefficient?.b ?? 0,
		};
		const target = {};
		for (const component of SH_COEFFICIENT_COMPONENTS) {
			const key = component as "r" | "g" | "b";
			Object.defineProperty(target, key, {
				enumerable: true,
				configurable: true,
				get: () => values[key],
				set: (value: number) => {
					if (values[key] === value) return;
					values[key] = value;
					this._markTrackedSHMutation("lighting");
				},
			});
		}
		return target as SHCoefficients[number];
	}

	private _copySHCoefficients(
		coefficients: SHCoefficients,
		dirtyReason: "lighting" | "probe-capture"
	): void {
		this._suppressSHMutationTracking++;
		try {
			copySHCoefficients(this._sh, coefficients);
		} finally {
			this._suppressSHMutationTracking--;
		}
		this._markTrackedSHMutation(dirtyReason);
	}

	private _markTrackedSHMutation(
		dirtyReason: "lighting" | "probe-capture"
	): void {
		if (!this._shStorageInitialized || this._suppressSHMutationTracking > 0) {
			return;
		}
		this.scene?.invalidate(dirtyReason);
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
			resolvedShape === "box"
				? Math.max(safeHalfExtents.x, safeHalfExtents.y, safeHalfExtents.z)
				: safeRadius;
		const blendFloor = Math.max(
			LIGHT_PROBE_MIN_BLEND_DISTANCE,
			probeSizeMetric * LIGHT_PROBE_BLEND_FLOOR_RATIO,
		);
		this._runtimeCache.effectiveBlendDistance = Math.max(
			0,
			sanitizeLightProbeBlendDistance(this.blendDistance),
			blendFloor,
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

function validateLightProbeParams(value: unknown): LightProbeParams {
	if (
		value === null ||
		value === undefined ||
		typeof value !== "object" ||
		Array.isArray(value)
	) {
		throw new TypeError(
			"LightProbe constructor expects a LightProbeParams object. " +
				"Use new LightProbe({}) for defaults or new LightProbe({ sh }) " +
				"for authored coefficients."
		);
	}
	return value as LightProbeParams;
}

function cloneSHCoefficients(coefficients?: SHCoefficients | null): SHCoefficients {
	if (!coefficients) {
		return SH.empty();
	}

	const result = SH.empty();
	copySHCoefficients(result, coefficients);
	return result;
}

function copySHCoefficients(
	target: SHCoefficients,
	source: SHCoefficients
): void {
	for (let i = 0; i < target.length; i++) {
		const coefficient = source[i];
		target[i].r = coefficient?.r ?? 0;
		target[i].g = coefficient?.g ?? 0;
		target[i].b = coefficient?.b ?? 0;
	}
}

function resolveArrayIndexProperty(property: PropertyKey): number | null {
	if (typeof property !== "string") return null;
	if (property === "" || `${Number(property)}` !== property) return null;
	const index = Number(property);
	return Number.isInteger(index) && index >= 0 ? index : null;
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

function sanitizeLightProbeSource(value: LightProbeSource): LightProbeSource {
	if (
		value === "environment" ||
		value === "capturedScene" ||
		value === "manual"
	) {
		return value;
	}
	return "environment";
}

function sanitizeLightProbeCaptureFar(value: number): number {
	if (!Number.isFinite(value)) return 200;
	return Math.max(1, value);
}

function sanitizeLightProbeCaptureResolution(
	value: Partial<LightProbeCaptureResolution> | undefined
): LightProbeCaptureResolution {
	const width = Math.max(8, Math.floor(value?.width ?? 64));
	const height = Math.max(4, Math.floor(value?.height ?? 32));
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
