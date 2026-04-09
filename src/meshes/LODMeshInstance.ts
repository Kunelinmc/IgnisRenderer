import type { IVector3 } from "../maths/types";
import { MeshAsset } from "./MeshAsset";
import { MeshInstance, type MeshInstanceParams } from "./MeshInstance";

export interface LODMeshLevel {
	mesh: MeshAsset;
	distance: number;
}

export interface LODMeshInstanceParams
	extends Omit<MeshInstanceParams, "mesh"> {
	levels: LODMeshLevel[];
	activeLevelIndex?: number;
	hysteresis?: number;
}

export interface LODMeshUpdateOptions {
	notifyScene?: boolean;
}

const _tmpLODWorldPosition = { x: 0, y: 0, z: 0 };

export class LODMeshInstance extends MeshInstance {
	private _levels: LODMeshLevel[];
	private _activeLevelIndex: number;
	private _hysteresis: number;

	constructor(params: LODMeshInstanceParams) {
		const levels = normalizeLevels(params.levels);
		const activeLevelIndex = clampLevelIndex(
			params.activeLevelIndex ?? 0,
			levels.length
		);
		super({
			...params,
			mesh: levels[activeLevelIndex].mesh,
		});
		this._levels = levels;
		this._activeLevelIndex = activeLevelIndex;
		this._hysteresis = sanitizeHysteresis(params.hysteresis);
	}

	public get levels(): ReadonlyArray<LODMeshLevel> {
		return this.getLevels();
	}

	public get activeLevelIndex(): number {
		return this._activeLevelIndex;
	}

	public get hysteresis(): number {
		return this._hysteresis;
	}

	public getLevels(): LODMeshLevel[] {
		return this._levels.map((level) => ({
			mesh: level.mesh,
			distance: level.distance,
		}));
	}

	public setLevels(levels: LODMeshLevel[], activeLevelIndex?: number): this {
		const normalized = normalizeLevels(levels);
		const nextIndex = clampLevelIndex(
			activeLevelIndex ?? this._activeLevelIndex,
			normalized.length
		);
		this._levels = normalized;
		this._applyLevel(nextIndex, true);
		return this;
	}

	public setActiveLevelIndex(index: number): this {
		const nextIndex = clampLevelIndex(index, this._levels.length);
		this._applyLevel(nextIndex, true);
		return this;
	}

	public setHysteresis(value: number): this {
		this._hysteresis = sanitizeHysteresis(value);
		return this;
	}

	public resolveLevelIndex(distanceToCamera: number): number {
		const safeDistance = sanitizeDistance(distanceToCamera);
		const desired = resolveLevelIndexForDistance(this._levels, safeDistance);
		return applyDistanceHysteresis(
			this._levels,
			safeDistance,
			this._activeLevelIndex,
			desired,
			this._hysteresis
		);
	}

	public updateLODByDistance(
		distanceToCamera: number,
		options: LODMeshUpdateOptions = {}
	): boolean {
		const nextLevelIndex = this.resolveLevelIndex(distanceToCamera);
		return this._applyLevel(nextLevelIndex, options.notifyScene ?? true);
	}

	public updateLODForCamera(
		cameraWorldPosition: IVector3,
		options: LODMeshUpdateOptions = {}
	): boolean {
		const worldPosition = this.getWorldPosition(_tmpLODWorldPosition);
		const dx = worldPosition.x - cameraWorldPosition.x;
		const dy = worldPosition.y - cameraWorldPosition.y;
		const dz = worldPosition.z - cameraWorldPosition.z;
		const distanceToCamera = Math.hypot(dx, dy, dz);
		return this.updateLODByDistance(distanceToCamera, options);
	}

	private _applyLevel(levelIndex: number, notifyScene: boolean): boolean {
		const level = this._levels[levelIndex];
		const nextMesh = level.mesh;
		const meshChanged =
			this._activeLevelIndex !== levelIndex || this.mesh !== nextMesh;

		this._activeLevelIndex = levelIndex;
		this.mesh = nextMesh;

		if (meshChanged && notifyScene) {
			this._syncSceneState();
		}
		return meshChanged;
	}

	private _syncSceneState(): void {
		if (!this._scene) return;
		this._scene.invalidate("transform");
		this._scene.spatial?.markDirty(this);
	}

	protected override _createCloneInstance(): this {
		return new LODMeshInstance({
			levels: this.getLevels(),
			activeLevelIndex: this._activeLevelIndex,
			hysteresis: this._hysteresis,
			skeleton: this.skeleton,
			morphWeights: this.morphWeights.map(
				(weights) => new Float32Array(weights)
			),
		}) as this;
	}

	protected override _copyClonePropertiesTo(target: this): void {
		super._copyClonePropertiesTo(target);
		if (!(target instanceof LODMeshInstance)) return;
		target._levels = this.getLevels();
		target._activeLevelIndex = this._activeLevelIndex;
		target._hysteresis = this._hysteresis;
		target.mesh = target._levels[target._activeLevelIndex].mesh;
	}
}

function normalizeLevels(levels: LODMeshLevel[]): LODMeshLevel[] {
	if (!Array.isArray(levels) || levels.length === 0) {
		throw new Error("LODMeshInstance requires at least one LOD level");
	}

	const normalized = levels.map((level, index) => {
		if (!(level.mesh instanceof MeshAsset)) {
			throw new Error(
				`LODMeshInstance level ${index} must provide a MeshAsset`
			);
		}
		const distance = sanitizeDistance(level.distance);
		return {
			mesh: level.mesh,
			distance,
			index,
		};
	});

	normalized.sort((left, right) => {
		if (left.distance !== right.distance) {
			return left.distance - right.distance;
		}
		return left.index - right.index;
	});

	return normalized.map((level) => ({
		mesh: level.mesh,
		distance: level.distance,
	}));
}

function clampLevelIndex(value: number, levelCount: number): number {
	if (levelCount <= 0) return 0;
	if (!Number.isFinite(value)) return 0;
	return Math.min(levelCount - 1, Math.max(0, Math.floor(value)));
}

function sanitizeDistance(value: number): number {
	if (!Number.isFinite(value)) return Number.POSITIVE_INFINITY;
	return Math.max(0, value);
}

function sanitizeHysteresis(value: number | undefined): number {
	if (value === undefined || !Number.isFinite(value)) return 0;
	return Math.max(0, value);
}

function resolveLevelIndexForDistance(
	levels: ReadonlyArray<LODMeshLevel>,
	distanceToCamera: number
): number {
	for (let i = 0; i < levels.length; i++) {
		if (distanceToCamera <= levels[i].distance) {
			return i;
		}
	}
	return levels.length - 1;
}

function applyDistanceHysteresis(
	levels: ReadonlyArray<LODMeshLevel>,
	distanceToCamera: number,
	currentIndex: number,
	desiredIndex: number,
	hysteresis: number
): number {
	if (
		hysteresis <= 0 ||
		desiredIndex === currentIndex ||
		currentIndex < 0 ||
		currentIndex >= levels.length
	) {
		return desiredIndex;
	}

	if (desiredIndex > currentIndex) {
		const farBoundary = levels[currentIndex].distance + hysteresis;
		return distanceToCamera > farBoundary ? desiredIndex : currentIndex;
	}

	const nearBoundary = levels[desiredIndex].distance - hysteresis;
	return distanceToCamera <= nearBoundary ? desiredIndex : currentIndex;
}
