import { Matrix3 } from "../maths/Matrix3";
import { Matrix4 } from "../maths/Matrix4";
import { SH } from "../maths/SH";
import type { IVector3, SHCoefficients } from "../maths/types";
import { Vector3 } from "../maths/Vector3";
import { Light, LightType, type LightParams } from "./Light";
import type {
	LightProbeCaptureResolution,
	LightProbeCaptureUpdateMode,
} from "./LightProbe";

export type IrradianceProbeGridSource = "manual" | "capturedScene";

export interface IrradianceProbeGridDimensions {
	x: number;
	y: number;
	z: number;
}

export interface IrradianceProbeGridCellCoord {
	x: number;
	y: number;
	z: number;
}

export type IrradianceProbeGridCellRef =
	| number
	| IrradianceProbeGridCellCoord;

export interface IrradianceProbeGridRuntimeCache {
	gridToWorldMatrix: Matrix4;
	worldToGridMatrix: Matrix4;
	worldToGrid3x3: Matrix3;
	dimensions: IrradianceProbeGridDimensions;
	cellCount: number;
	halfExtents: IVector3;
	invHalfExtents: IVector3;
	effectiveBlendDistance: number;
	priority: number;
	cellWorldPositions: IVector3[];
	validMask: Uint8Array;
	cellRevisions: Uint32Array;
	textureRevision: number;
}

export interface IrradianceProbeGridParams extends LightParams {
	dimensions: IrradianceProbeGridDimensions;
	halfExtents?: IVector3;
	blendDistance?: number;
	priority?: number;
	source?: IrradianceProbeGridSource;
	captureUpdateMode?: LightProbeCaptureUpdateMode;
	captureIntervalSeconds?: number;
	captureResolution?: Partial<LightProbeCaptureResolution>;
	captureFar?: number;
	includeEnvironment?: boolean;
	includeMeshes?: boolean;
	includeTransparent?: boolean;
	includeParticles?: boolean;
	includeShadows?: boolean;
	sh?: SHCoefficients[] | null;
	validMask?: ArrayLike<number> | null;
}

const IRRADIANCE_PROBE_GRID_MAX_CELL_COUNT = 256;
const IRRADIANCE_PROBE_GRID_NUMERIC_EPSILON = 1e-6;
const SH_COEFFICIENT_COMPONENTS = new Set<PropertyKey>(["r", "g", "b"]);

/**
 * Regular local-box grid of spherical harmonics irradiance probes.
 *
 * Grid cells are indexed with X as the fastest axis, followed by Y and then Z.
 * Captured-scene grids are updated by `ProbeCaptureRuntime`; authored grids
 * should use `setCellSH` so validity and renderer revisions remain coherent.
 */
export class IrradianceProbeGrid extends Light<LightType.IrradianceProbeGrid> {
	/** Integer grid dimensions in X/Y/Z order. Total cell count must be <= 256. */
	public dimensions: IrradianceProbeGridDimensions;
	/** Local-space box half extents before applying the owning node transform. */
	public halfExtents: Vector3;
	/** Box-edge fade distance in normalized box metric units. */
	public blendDistance: number;
	/** Selection priority. Higher priority wins before camera distance and id. */
	public priority: number;
	/** SH data source. `"capturedScene"` enables `ProbeCaptureRuntime` updates. */
	public source: IrradianceProbeGridSource;
	/** Capture scheduling mode used when `source` is `"capturedScene"`. */
	public captureUpdateMode: LightProbeCaptureUpdateMode;
	/** Minimum seconds between interval captures. */
	public captureIntervalSeconds: number;
	/** Equirectangular capture resolution used for SH projection. */
	public captureResolution: LightProbeCaptureResolution;
	/** Far plane distance for scene captures. */
	public captureFar: number;
	/** Includes scene environment lighting in captures when true. */
	public includeEnvironment: boolean;
	/** Includes mesh rendering in captures when true. */
	public includeMeshes: boolean;
	/** Includes transparent mesh rendering in captures when true. */
	public includeTransparent: boolean;
	/** Includes particle rendering in captures when true. */
	public includeParticles: boolean;
	/** Includes shadowing in captures when true. */
	public includeShadows: boolean;
	private _sh: SHCoefficients[] = [];
	private _shProxy: SHCoefficients[] | null = null;

	private _validMask: Uint8Array;
	private _cellRevisions: Uint32Array;
	private _cellCaptureRequestTokens: Uint32Array;
	private _captureRequestToken = 0;
	private _captureRevision = 0;
	private _textureRevision = 0;
	private _runtimeCache: IrradianceProbeGridRuntimeCache;
	private _runtimeDirty = true;
	private _matrixSignature = new Float32Array(16);
	private _lastDimensions: IrradianceProbeGridDimensions;
	private _lastHalfExtents = new Float32Array(3);
	private _lastBlendDistance: number;
	private _lastPriority: number;
	private _suppressSHMutationTracking = 0;
	private _storageInitialized = false;

	/**
	 * Creates an irradiance probe grid.
	 *
	 * @param params - Grid parameters. `dimensions` is required and the product
	 * must not exceed 256 cells.
	 * @throws TypeError when `params` is not an object.
	 * @throws RangeError when `dimensions` produces more than 256 cells.
	 * @sideEffects Allocates per-cell SH, validity, capture, and runtime cache
	 * storage.
	 */
	constructor(params: IrradianceProbeGridParams) {
		const resolvedParams = validateIrradianceProbeGridParams(params);
		super(LightType.IrradianceProbeGrid, resolvedParams);
		this.dimensions = sanitizeDimensions(resolvedParams.dimensions);
		const cellCount = getCellCount(this.dimensions);
		this.halfExtents = new Vector3();
		this.halfExtents.copy(sanitizeHalfExtents(resolvedParams.halfExtents));
		this.blendDistance = sanitizeBlendDistance(
			resolvedParams.blendDistance ?? 0.15
		);
		this.priority = sanitizePriority(resolvedParams.priority ?? 0);
		this.source = sanitizeSource(resolvedParams.source ?? "manual");
		this.captureUpdateMode = sanitizeCaptureUpdateMode(
			resolvedParams.captureUpdateMode ?? "manual"
		);
		this.captureIntervalSeconds = sanitizeCaptureIntervalSeconds(
			resolvedParams.captureIntervalSeconds ?? 1
		);
		this.captureResolution = sanitizeCaptureResolution(
			resolvedParams.captureResolution
		);
		this.captureFar = sanitizeCaptureFar(resolvedParams.captureFar ?? 200);
		this.includeEnvironment = resolvedParams.includeEnvironment ?? true;
		this.includeMeshes = resolvedParams.includeMeshes ?? true;
		this.includeTransparent = resolvedParams.includeTransparent ?? true;
		this.includeParticles = resolvedParams.includeParticles ?? true;
		this.includeShadows = resolvedParams.includeShadows ?? true;
		this._replaceSHStorage(resolvedParams.sh, cellCount);
		this._validMask = createValidMask(resolvedParams.validMask, cellCount);
		this._cellRevisions = new Uint32Array(cellCount);
		this._cellCaptureRequestTokens = new Uint32Array(cellCount);
		this._runtimeCache = {
			gridToWorldMatrix: Matrix4.identity(),
			worldToGridMatrix: Matrix4.identity(),
			worldToGrid3x3: Matrix3.identity(),
			dimensions: { ...this.dimensions },
			cellCount,
			halfExtents: {
				x: this.halfExtents.x,
				y: this.halfExtents.y,
				z: this.halfExtents.z,
			},
			invHalfExtents: {
				x: 1 / this.halfExtents.x,
				y: 1 / this.halfExtents.y,
				z: 1 / this.halfExtents.z,
			},
			effectiveBlendDistance: this.blendDistance,
			priority: this.priority,
			cellWorldPositions: createCellWorldPositionArray(cellCount),
			validMask: this._validMask,
			cellRevisions: this._cellRevisions,
			textureRevision: this._textureRevision,
		};
		this._lastDimensions = { ...this.dimensions };
		this._lastHalfExtents[0] = this.halfExtents.x;
		this._lastHalfExtents[1] = this.halfExtents.y;
		this._lastHalfExtents[2] = this.halfExtents.z;
		this._lastBlendDistance = this.blendDistance;
		this._lastPriority = this.priority;
		for (let i = 0; i < this._matrixSignature.length; i++) {
			this._matrixSignature[i] = Number.NaN;
		}
		this._storageInitialized = true;
	}

	/** Per-cell SH coefficients, indexed with X fastest, then Y, then Z. */
	public get sh(): SHCoefficients[] {
		if (!this._shProxy) {
			this._shProxy = this._createGridSHProxy();
		}
		return this._shProxy;
	}

	public set sh(value: SHCoefficients[]) {
		const cellCount =
			this.dimensions ? getCellCount(sanitizeDimensions(this.dimensions))
			: Array.isArray(value) ? value.length
			: 0;
		this._replaceSHStorage(value, cellCount);
		if (this._storageInitialized && this._suppressSHMutationTracking === 0) {
			this._markAllCellDataChanged(true, "lighting");
		}
	}

	protected override _createCloneInstance(): this {
		return new IrradianceProbeGrid({
			dimensions: this.dimensions,
		}) as this;
	}

	/**
	 * Resolves a cell coordinate into the flat grid index.
	 *
	 * @param x - X coordinate in `[0, dimensions.x - 1]`.
	 * @param y - Y coordinate in `[0, dimensions.y - 1]`.
	 * @param z - Z coordinate in `[0, dimensions.z - 1]`.
	 * @returns Flat index with X as the fastest axis.
	 * @throws RangeError when any coordinate is outside the grid.
	 * @sideEffects None.
	 */
	public getCellIndex(x: number, y: number, z: number): number {
		const ix = sanitizeCellCoord(x, this.dimensions.x, "x");
		const iy = sanitizeCellCoord(y, this.dimensions.y, "y");
		const iz = sanitizeCellCoord(z, this.dimensions.z, "z");
		return ix + iy * this.dimensions.x + iz * this.dimensions.x * this.dimensions.y;
	}

	/**
	 * Returns the SH coefficients stored for a cell.
	 *
	 * @param cell - Flat cell index or `{ x, y, z }` coordinate.
	 * @returns The cell-owned SH coefficient array.
	 * @sideEffects None.
	 */
	public getCellSH(cell: IrradianceProbeGridCellRef): SHCoefficients {
		return this.sh[this._resolveCellIndex(cell)];
	}

	/**
	 * Writes authored SH coefficients for a cell and marks it valid.
	 *
	 * @param cell - Flat cell index or `{ x, y, z }` coordinate.
	 * @param coeffs - Source SH coefficients copied into the cell.
	 * @returns Nothing.
	 * @sideEffects Updates validity, cell revision, texture revision, capture
	 * revision, and invalidates scene lighting when attached.
	 */
	public setCellSH(
		cell: IrradianceProbeGridCellRef,
		coeffs: SHCoefficients
	): void {
		const index = this._resolveCellIndex(cell);
		this._writeCellSH(index, coeffs, true, "lighting");
	}

	/**
	 * Clears a cell and marks it invalid.
	 *
	 * @param cell - Flat cell index or `{ x, y, z }` coordinate.
	 * @returns Nothing.
	 * @sideEffects Updates validity, cell revision, texture revision, capture
	 * revision, and invalidates scene lighting when attached.
	 */
	public clearCell(cell: IrradianceProbeGridCellRef): void {
		const index = this._resolveCellIndex(cell);
		this._writeCellSH(index, SH.empty(), false, "lighting");
	}

	/**
	 * Requests captured-scene updates.
	 *
	 * @param cell - Optional flat cell index or `{ x, y, z }` coordinate. When
	 * omitted, all cells are requested.
	 * @returns Nothing.
	 * @sideEffects Increments capture request tokens and invalidates lighting.
	 */
	public requestCapture(cell?: IrradianceProbeGridCellRef): void {
		this._captureRequestToken++;
		this._captureRevision++;
		if (cell === undefined) {
			this._cellCaptureRequestTokens.fill(this._captureRequestToken);
		} else {
			this._cellCaptureRequestTokens[this._resolveCellIndex(cell)] =
				this._captureRequestToken;
		}
		this.scene?.invalidate("lighting");
	}

	/**
	 * Returns the monotonic token for all explicit capture requests.
	 *
	 * @returns The latest capture request token.
	 */
	public get captureRequestToken(): number {
		return this._captureRequestToken;
	}

	/**
	 * Returns the monotonic capture/data revision for the grid.
	 *
	 * @returns Current capture revision.
	 */
	public get captureRevision(): number {
		return this._captureRevision;
	}

	/**
	 * Returns the renderer texture revision for SH upload caches.
	 *
	 * @returns Current texture revision.
	 */
	public get textureRevision(): number {
		return this._textureRevision;
	}

	/**
	 * Returns the explicit request token for a cell.
	 *
	 * @param cellIndex - Flat cell index.
	 * @returns Current request token for the cell.
	 * @sideEffects None.
	 */
	public getCellCaptureRequestToken(cellIndex: number): number {
		return this._cellCaptureRequestTokens[this._resolveCellIndex(cellIndex)];
	}

	/**
	 * Marks a runtime capture result as written for one cell.
	 *
	 * @param cellIndex - Flat cell index.
	 * @returns Nothing.
	 * @sideEffects Marks the cell valid and advances revisions.
	 */
	public markCellCaptureUpdated(cellIndex: number): void {
		this._markCellDataChanged(cellIndex, true, "lighting");
	}

	/**
	 * Writes captured-scene SH coefficients for one cell.
	 *
	 * @internal Owned by `ProbeCaptureRuntime`; application-authored data should
	 * use `setCellSH` so intent remains explicit.
	 * @param cellIndex - Flat cell index.
	 * @param coeffs - Captured SH coefficients copied into the cell.
	 * @returns Nothing.
	 * @sideEffects Marks the cell valid, advances revisions, and invalidates the
	 * scene with the non-capture-relevant `probe-capture` reason.
	 */
	public writeCapturedCellSH(
		cellIndex: number,
		coeffs: SHCoefficients
	): void {
		const index = this._resolveCellIndex(cellIndex);
		this._writeCellSH(index, coeffs, true, "probe-capture");
	}

	/**
	 * Returns whether a cell is valid for trilinear sampling.
	 *
	 * @param cell - Flat cell index or `{ x, y, z }` coordinate.
	 * @returns `true` when the cell has usable SH data.
	 * @sideEffects None.
	 */
	public isCellValid(cell: IrradianceProbeGridCellRef): boolean {
		return this._validMask[this._resolveCellIndex(cell)] > 0;
	}

	/**
	 * Marks cached transform and cell-position data dirty.
	 *
	 * @returns Nothing.
	 * @sideEffects Forces the next `getRuntimeCache` call to rebuild matrices.
	 */
	public markRuntimeDirty(): void {
		this._runtimeDirty = true;
	}

	/**
	 * Refreshes cached runtime data immediately.
	 *
	 * @returns Nothing.
	 * @sideEffects Recomputes matrices and cell world positions when dirty.
	 */
	public refreshRuntimeCache(): void {
		this._updateRuntimeCache();
	}

	/**
	 * Returns renderer-facing cached grid data.
	 *
	 * @returns Runtime cache containing transform, dimensions, validity, and
	 * revision data.
	 * @sideEffects Recomputes cache when transform or grid parameters changed.
	 */
	public getRuntimeCache(): IrradianceProbeGridRuntimeCache {
		if (this._runtimeDirty || this._runtimeStateChanged()) {
			this._updateRuntimeCache();
		}
		return this._runtimeCache;
	}

	protected override _copyClonePropertiesTo(target: this): void {
		super._copyClonePropertiesTo(target);
		target.dimensions = { ...this.dimensions };
		target.halfExtents.copy(this.halfExtents);
		target.blendDistance = this.blendDistance;
		target.priority = this.priority;
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
		target._resizeStorage(getCellCount(this.dimensions));
		target._withSHMutationTrackingSuppressed(() => {
			for (let i = 0; i < this.sh.length; i++) {
				copySHCoefficients(target.sh[i], this.sh[i]);
			}
		});
		target._validMask.set(this._validMask);
		target._cellRevisions.set(this._cellRevisions);
		target._cellCaptureRequestTokens.set(this._cellCaptureRequestTokens);
		target._captureRequestToken = this._captureRequestToken;
		target._captureRevision = this._captureRevision;
		target._textureRevision = this._textureRevision;
		target.markRuntimeDirty();
	}

	private _resolveCellIndex(cell: IrradianceProbeGridCellRef): number {
		if (typeof cell === "number") {
			const index = Math.floor(cell);
			if (index < 0 || index >= this.sh.length) {
				throw new RangeError(
					`IrradianceProbeGrid cell index ${cell} is outside [0, ${this.sh.length - 1}].`
				);
			}
			return index;
		}
		return this.getCellIndex(cell.x, cell.y, cell.z);
	}

	private _writeCellSH(
		cellIndex: number,
		coeffs: SHCoefficients,
		valid: boolean,
		dirtyReason: "lighting" | "probe-capture"
	): void {
		this._withSHMutationTrackingSuppressed(() => {
			copySHCoefficients(this._sh[cellIndex], coeffs);
		});
		this._markCellDataChanged(cellIndex, valid, dirtyReason);
	}

	private _markCellDataChanged(
		cellIndex: number,
		valid: boolean,
		dirtyReason: "lighting" | "probe-capture"
	): void {
		const index = this._resolveCellIndex(cellIndex);
		this._validMask[index] = valid ? 1 : 0;
		this._cellRevisions[index]++;
		this._textureRevision++;
		this._captureRevision++;
		this._runtimeCache.textureRevision = this._textureRevision;
		this.scene?.invalidate(dirtyReason);
	}

	private _markAllCellDataChanged(
		valid: boolean,
		dirtyReason: "lighting" | "probe-capture"
	): void {
		for (let index = 0; index < this._sh.length; index++) {
			this._validMask[index] = valid ? 1 : 0;
			this._cellRevisions[index]++;
		}
		this._textureRevision++;
		this._captureRevision++;
		this._runtimeCache.textureRevision = this._textureRevision;
		this.scene?.invalidate(dirtyReason);
	}

	private _resizeStorage(cellCount: number): void {
		this._replaceSHStorage(this._sh, cellCount);
		this._validMask = createValidMask(this._validMask, cellCount);
		this._cellRevisions = resizeUint32Array(this._cellRevisions, cellCount);
		this._cellCaptureRequestTokens = resizeUint32Array(
			this._cellCaptureRequestTokens,
			cellCount
		);
		this._runtimeCache.cellWorldPositions =
			createCellWorldPositionArray(cellCount);
		this._runtimeCache.validMask = this._validMask;
		this._runtimeCache.cellRevisions = this._cellRevisions;
	}

	private _replaceSHStorage(
		source: SHCoefficients[] | null | undefined,
		cellCount: number
	): void {
		const cloned = cloneGridSH(source, cellCount);
		this._sh = cloned.map((cell, index) => this._trackCellSH(index, cell));
		this._shProxy = null;
	}

	private _createGridSHProxy(): SHCoefficients[] {
		return new Proxy(this._sh, {
			set: (target, property, value) => {
				const index = resolveArrayIndexProperty(property);
				if (index === null) {
					return Reflect.set(target, property, value);
				}
				const cell = cloneSHCoefficients(value as SHCoefficients);
				target[index] = this._trackCellSH(index, cell);
				this._markTrackedSHMutation(index);
				return true;
			},
		});
	}

	private _trackCellSH(
		cellIndex: number,
		cell: SHCoefficients
	): SHCoefficients {
		for (let coeffIndex = 0; coeffIndex < cell.length; coeffIndex++) {
			cell[coeffIndex] = this._trackSHCoefficient(cellIndex, cell[coeffIndex]);
		}
		return new Proxy(cell, {
			set: (target, property, value) => {
				const coeffIndex = resolveArrayIndexProperty(property);
				if (coeffIndex === null) {
					return Reflect.set(target, property, value);
				}
				target[coeffIndex] = this._trackSHCoefficient(
					cellIndex,
					value as SHCoefficients[number]
				);
				this._markTrackedSHMutation(cellIndex);
				return true;
			},
		});
	}

	private _trackSHCoefficient(
		cellIndex: number,
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
					if (values[key] === value) {
						return;
					}
					values[key] = value;
					this._markTrackedSHMutation(cellIndex);
				},
			});
		}
		return target as SHCoefficients[number];
	}

	private _markTrackedSHMutation(cellIndex: number): void {
		if (!this._storageInitialized || this._suppressSHMutationTracking > 0) {
			return;
		}
		this._markCellDataChanged(cellIndex, true, "lighting");
	}

	private _withSHMutationTrackingSuppressed(callback: () => void): void {
		this._suppressSHMutationTracking++;
		try {
			callback();
		} finally {
			this._suppressSHMutationTracking--;
		}
	}

	private _runtimeStateChanged(): boolean {
		if (!dimensionsEqual(this.dimensions, this._lastDimensions)) return true;
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
		this.dimensions = sanitizeDimensions(this.dimensions);
		const cellCount = getCellCount(this.dimensions);
		if (cellCount !== this.sh.length) {
			this._resizeStorage(cellCount);
		}
		const safeHalfExtents = sanitizeHalfExtents(this.halfExtents);
		this.halfExtents.copy(safeHalfExtents);
		const cache = this._runtimeCache;
		this.worldMatrix.copyTo(cache.gridToWorldMatrix);
		Matrix4.inverse(this.worldMatrix, cache.worldToGridMatrix) ??
			Matrix4.identity().copyTo(cache.worldToGridMatrix);
		const inverse3x3 = Matrix4.inverse3x3(
			this.worldMatrix,
			cache.worldToGrid3x3
		);
		if (inverse3x3) {
			cache.worldToGrid3x3.copy(inverse3x3);
		} else {
			Matrix3.identity().copyTo(cache.worldToGrid3x3);
		}
		cache.dimensions.x = this.dimensions.x;
		cache.dimensions.y = this.dimensions.y;
		cache.dimensions.z = this.dimensions.z;
		cache.cellCount = cellCount;
		cache.halfExtents.x = this.halfExtents.x;
		cache.halfExtents.y = this.halfExtents.y;
		cache.halfExtents.z = this.halfExtents.z;
		cache.invHalfExtents.x = 1 / this.halfExtents.x;
		cache.invHalfExtents.y = 1 / this.halfExtents.y;
		cache.invHalfExtents.z = 1 / this.halfExtents.z;
		cache.effectiveBlendDistance = resolveEffectiveBlendDistance(this.blendDistance);
		cache.priority = sanitizePriority(this.priority);
		cache.textureRevision = this._textureRevision;
		updateCellWorldPositions(cache.cellWorldPositions, this);
		this._lastDimensions = { ...this.dimensions };
		this._lastHalfExtents[0] = this.halfExtents.x;
		this._lastHalfExtents[1] = this.halfExtents.y;
		this._lastHalfExtents[2] = this.halfExtents.z;
		this._lastBlendDistance = this.blendDistance;
		this._lastPriority = this.priority;
		copyMatrixSignature(this._matrixSignature, this.worldMatrix);
		this._runtimeDirty = false;
	}
}

export const IRRADIANCE_PROBE_GRID_MAX_CELLS =
	IRRADIANCE_PROBE_GRID_MAX_CELL_COUNT;

function validateIrradianceProbeGridParams(
	value: unknown
): IrradianceProbeGridParams {
	if (
		value === null ||
		value === undefined ||
		typeof value !== "object" ||
		Array.isArray(value)
	) {
		throw new TypeError(
			"IrradianceProbeGrid constructor expects an IrradianceProbeGridParams object."
		);
	}
	return value as IrradianceProbeGridParams;
}

function sanitizeDimensions(
	value: IrradianceProbeGridDimensions
): IrradianceProbeGridDimensions {
	const dimensions = {
		x: sanitizeDimension(value?.x),
		y: sanitizeDimension(value?.y),
		z: sanitizeDimension(value?.z),
	};
	const cellCount = getCellCount(dimensions);
	if (cellCount > IRRADIANCE_PROBE_GRID_MAX_CELL_COUNT) {
		throw new RangeError(
			`IrradianceProbeGrid supports at most ${IRRADIANCE_PROBE_GRID_MAX_CELL_COUNT} cells; received ${cellCount}.`
		);
	}
	return dimensions;
}

function sanitizeDimension(value: number): number {
	if (!Number.isFinite(value)) return 1;
	return Math.max(1, Math.floor(value));
}

function getCellCount(dimensions: IrradianceProbeGridDimensions): number {
	return dimensions.x * dimensions.y * dimensions.z;
}

function sanitizeHalfExtents(value: IVector3 | undefined): IVector3 {
	const x = Number.isFinite(value?.x) ? Math.abs(value.x) : 5;
	const y = Number.isFinite(value?.y) ? Math.abs(value.y) : 5;
	const z = Number.isFinite(value?.z) ? Math.abs(value.z) : 5;
	return {
		x: Math.max(IRRADIANCE_PROBE_GRID_NUMERIC_EPSILON, x),
		y: Math.max(IRRADIANCE_PROBE_GRID_NUMERIC_EPSILON, y),
		z: Math.max(IRRADIANCE_PROBE_GRID_NUMERIC_EPSILON, z),
	};
}

function sanitizeBlendDistance(value: number): number {
	if (!Number.isFinite(value)) return 0.15;
	return Math.max(0, value);
}

function sanitizePriority(value: number): number {
	if (!Number.isFinite(value)) return 0;
	return Math.max(-2147483647, Math.min(2147483647, Math.trunc(value)));
}

function sanitizeSource(value: IrradianceProbeGridSource): IrradianceProbeGridSource {
	if (value === "manual" || value === "capturedScene") {
		return value;
	}
	return "manual";
}

function sanitizeCaptureUpdateMode(
	value: LightProbeCaptureUpdateMode
): LightProbeCaptureUpdateMode {
	if (value === "manual" || value === "onSceneDirty" || value === "interval") {
		return value;
	}
	return "manual";
}

function sanitizeCaptureIntervalSeconds(value: number): number {
	if (!Number.isFinite(value)) return 1;
	return Math.max(0.01, value);
}

function sanitizeCaptureFar(value: number): number {
	if (!Number.isFinite(value)) return 200;
	return Math.max(1, value);
}

function sanitizeCaptureResolution(
	value: Partial<LightProbeCaptureResolution> | undefined
): LightProbeCaptureResolution {
	const width = Math.max(8, Math.floor(value?.width ?? 64));
	const height = Math.max(4, Math.floor(value?.height ?? 32));
	return { width, height };
}

function sanitizeCellCoord(value: number, limit: number, axis: string): number {
	const coord = Math.floor(value);
	if (coord < 0 || coord >= limit) {
		throw new RangeError(
			`IrradianceProbeGrid ${axis} coordinate ${value} is outside [0, ${limit - 1}].`
		);
	}
	return coord;
}

function cloneGridSH(
	source: SHCoefficients[] | null | undefined,
	cellCount: number
): SHCoefficients[] {
	const result: SHCoefficients[] = [];
	for (let index = 0; index < cellCount; index++) {
		result.push(cloneSHCoefficients(source?.[index]));
	}
	return result;
}

function cloneSHCoefficients(coefficients?: SHCoefficients | null): SHCoefficients {
	const result = SH.empty();
	if (!coefficients) {
		return result;
	}
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

function createValidMask(
	source: ArrayLike<number> | null | undefined,
	cellCount: number
): Uint8Array {
	const result = new Uint8Array(cellCount);
	if (!source) {
		return result;
	}
	const count = Math.min(source.length, cellCount);
	for (let i = 0; i < count; i++) {
		result[i] = source[i] ? 1 : 0;
	}
	return result;
}

function resizeUint32Array(source: Uint32Array, cellCount: number): Uint32Array {
	const result = new Uint32Array(cellCount);
	result.set(source.subarray(0, Math.min(source.length, cellCount)));
	return result;
}

function resolveArrayIndexProperty(property: PropertyKey): number | null {
	if (typeof property !== "string") {
		return null;
	}
	if (property === "" || `${Number(property)}` !== property) {
		return null;
	}
	const index = Number(property);
	if (!Number.isInteger(index) || index < 0) {
		return null;
	}
	return index;
}

function createCellWorldPositionArray(cellCount: number): IVector3[] {
	return Array.from({ length: cellCount }, () => ({ x: 0, y: 0, z: 0 }));
}

function updateCellWorldPositions(
	positions: IVector3[],
	grid: IrradianceProbeGrid
): void {
	const dimensions = grid.dimensions;
	for (let z = 0; z < dimensions.z; z++) {
		for (let y = 0; y < dimensions.y; y++) {
			for (let x = 0; x < dimensions.x; x++) {
				const index = grid.getCellIndex(x, y, z);
				const local = {
					x: resolveAxisCellPosition(x, dimensions.x, grid.halfExtents.x),
					y: resolveAxisCellPosition(y, dimensions.y, grid.halfExtents.y),
					z: resolveAxisCellPosition(z, dimensions.z, grid.halfExtents.z),
				};
				Matrix4.transformPoint(grid.worldMatrix, local, positions[index]);
			}
		}
	}
}

function resolveAxisCellPosition(
	index: number,
	count: number,
	halfExtent: number
): number {
	if (count <= 1) {
		return 0;
	}
	return -halfExtent + (index / Math.max(1, count - 1)) * halfExtent * 2;
}

function resolveEffectiveBlendDistance(blendDistance: number): number {
	return sanitizeBlendDistance(blendDistance);
}

function dimensionsEqual(
	left: IrradianceProbeGridDimensions,
	right: IrradianceProbeGridDimensions
): boolean {
	return left.x === right.x && left.y === right.y && left.z === right.z;
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
