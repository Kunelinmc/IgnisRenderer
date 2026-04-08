import { Platform } from "../foundation/Platform";
import { MeshAsset } from "./MeshAsset";
import { MeshInstance, type MeshInstanceParams } from "./MeshInstance";
import type { PhysicsSystem } from "../physics";
import { CSGBuilder, normalizeGraphInput, type CSGGraphInput } from "../csg/CSGBuilder";
import { buildCSGMeshAsset } from "../csg/solvers";
import type {
	CSGBuildOptions,
	CSGExecutionMode,
	CSGGraph,
	CSGPhysicsSyncMode,
	CSGRebuildResult,
	CSGSolverPreference,
} from "../csg/types";

export interface CSGMeshInstanceParams extends Omit<MeshInstanceParams, "mesh"> {
	graph?: CSGGraphInput;
	mesh?: MeshAsset;
	buildOptions?: CSGBuildOptions;
	solverPreference?: CSGSolverPreference;
	executionMode?: CSGExecutionMode;
	physicsSync?: CSGPhysicsSyncMode;
	physicsSystem?: PhysicsSystem | null;
}

export class CSGMeshInstance extends MeshInstance {
	public physicsSync: CSGPhysicsSyncMode;
	public physicsSystem: PhysicsSystem | null;

	private _graph: CSGGraph | null;
	private _buildOptions: CSGBuildOptions;
	private _solverPreference: CSGSolverPreference;
	private _executionMode: CSGExecutionMode;
	private _dirty: boolean;
	private _flushToken = 0;
	private _latestRequestedToken = 0;
	private _lastResult: CSGRebuildResult | null = null;

	constructor(params: CSGMeshInstanceParams = {}) {
		super({
			...params,
			mesh: params.mesh ?? MeshAsset.fromFaces([]),
		});
		this._graph = params.graph ? normalizeGraphInput(params.graph) : null;
		this._buildOptions = { ...(params.buildOptions ?? {}) };
		this._solverPreference = params.solverPreference ?? "auto";
		this._executionMode = params.executionMode ?? "sync";
		this.physicsSync = params.physicsSync ?? "off";
		this.physicsSystem = params.physicsSystem ?? null;
		this._dirty = !!params.graph;
	}

	public get isCSGDirty(): boolean {
		return this._dirty;
	}

	public get solverPreference(): CSGSolverPreference {
		return this._solverPreference;
	}

	public get executionMode(): CSGExecutionMode {
		return this._executionMode;
	}

	public getGraph(): CSGGraph | null {
		if (!this._graph) return null;
		return new CSGBuilder(this._graph).getGraph();
	}

	public setGraph(input: CSGGraphInput): this {
		this._graph = normalizeGraphInput(input);
		this.markCSGDirty();
		return this;
	}

	public setSolverPreference(preference: CSGSolverPreference): this {
		this._solverPreference = preference;
		this.markCSGDirty();
		return this;
	}

	public setExecutionMode(mode: CSGExecutionMode): this {
		this._executionMode = mode;
		return this;
	}

	public setBuildOptions(options: CSGBuildOptions): this {
		this._buildOptions = { ...this._buildOptions, ...options };
		this.markCSGDirty();
		return this;
	}

	public markCSGDirty(): void {
		this._dirty = true;
		this._scene?.invalidate("transform");
	}

	public flushCSG(options: CSGBuildOptions = {}): CSGRebuildResult | Promise<CSGRebuildResult> {
		if (!this._graph) {
			return {
				ok: false,
				meshAsset: null,
				triangleCount: 0,
				solverId: this._solverPreference,
				fallbackUsed: false,
				stale: false,
				diagnostics: [
					{
						code: "csg-graph-missing",
						message: "CSGMeshInstance.flushCSG requires a valid CSG graph",
						severity: "error",
					},
				],
			};
		}

		if (!this._dirty && this._lastResult) {
			return {
				...this._lastResult,
				diagnostics: this._lastResult.diagnostics.slice(),
			};
		}

		const token = ++this._flushToken;
		this._latestRequestedToken = token;
		const buildOptions = {
			...this._buildOptions,
			...options,
			solverPreference: options.solverPreference ?? this._solverPreference,
		};

		if (this._executionMode === "worker") {
			return Promise.resolve().then(() => this._flushNow(token, buildOptions, true));
		}
		return this._flushNow(token, buildOptions, false);
	}

	private _flushNow(
		token: number,
		options: CSGBuildOptions,
		asynchronous: boolean,
	): CSGRebuildResult {
		if (!this._graph) {
			return {
				ok: false,
				meshAsset: null,
				triangleCount: 0,
				solverId: this._solverPreference,
				fallbackUsed: false,
				stale: false,
				diagnostics: [
					{
						code: "csg-graph-missing",
						message: "CSGMeshInstance.flushCSG requires a valid CSG graph",
						severity: "error",
					},
				],
			};
		}

		const result = buildCSGMeshAsset(this._graph, options);
		if (asynchronous) {
			result.diagnostics.push({
				code: "csg-worker-fallback-sync",
				message: Platform.hasWorker()
					? "CSG worker execution requested but no worker executor is configured; fell back to sync execution"
					: "CSG worker execution requested but Worker is unavailable; fell back to sync execution",
				severity: "warning",
			});
		}
		if (token !== this._latestRequestedToken) {
			return {
				...result,
				stale: true,
			};
		}

		if (result.ok && result.meshAsset) {
			const nextVersion = resolveNextGeometryVersion(this.mesh);
			for (const primitive of result.meshAsset.primitives) {
				primitive.geometryVersion = nextVersion;
			}
			this.mesh = result.meshAsset;
			this._dirty = false;
			this._syncSceneState();
			if (this.physicsSync === "auto" && this.physicsSystem) {
				try {
					this.physicsSystem.rebuildColliders(this);
				} catch (error) {
					result.diagnostics.push({
						code: "csg-physics-sync-failed",
						message:
							error instanceof Error ? error.message : "CSG physics auto-sync failed",
						severity: "warning",
					});
				}
			}
		}

		this._lastResult = {
			...result,
			diagnostics: result.diagnostics.slice(),
		};
		return result;
	}

	private _syncSceneState(): void {
		if (!this._scene) return;
		this._scene.invalidate("transform");
		this._scene.spatial?.markDirty(this);
	}

	protected override _createCloneInstance(): this {
		return new CSGMeshInstance({
			mesh: this.mesh,
			graph: this._graph ? new CSGBuilder(this._graph).getGraph() : undefined,
			buildOptions: { ...this._buildOptions },
			solverPreference: this._solverPreference,
			executionMode: this._executionMode,
			physicsSync: this.physicsSync,
			physicsSystem: this.physicsSystem,
		}) as this;
	}

	protected override _copyClonePropertiesTo(target: this): void {
		super._copyClonePropertiesTo(target);
		if (!(target instanceof CSGMeshInstance)) return;
		target._graph = this._graph ? new CSGBuilder(this._graph).getGraph() : null;
		target._buildOptions = { ...this._buildOptions };
		target._solverPreference = this._solverPreference;
		target._executionMode = this._executionMode;
		target.physicsSync = this.physicsSync;
		target.physicsSystem = this.physicsSystem;
		target._dirty = this._dirty;
		target._lastResult = this._lastResult
			? {
					...this._lastResult,
					diagnostics: this._lastResult.diagnostics.slice(),
				}
			: null;
	}
}

function resolveNextGeometryVersion(mesh: MeshAsset): number {
	let maxVersion = 0;
	for (const primitive of mesh.primitives) {
		maxVersion = Math.max(maxVersion, primitive.geometryVersion ?? 0);
	}
	return maxVersion + 1;
}
