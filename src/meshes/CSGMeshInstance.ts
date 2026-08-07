import { Platform } from "../foundation/Platform";
import { MeshAsset } from "./MeshAsset";
import { MeshInstance, type MeshInstanceParams } from "./MeshInstance";
import type { PhysicsSystem } from "../physics";
import { CSGBuilder, normalizeGraphInput, type CSGGraphInput } from "../csg/CSGBuilder";
import { buildCSGMeshAsset } from "../csg/solvers";
import type {
	CSGBuildOptions,
	CSGExecutor,
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
	executor?: CSGExecutor | null;
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
	private _executor: CSGExecutor | null;
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
		this._executor = params.executor ?? null;
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

	public get executor(): CSGExecutor | null {
		return this._executor;
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

	public setExecutor(executor: CSGExecutor | null): this {
		this._executor = executor;
		return this;
	}

	public setBuildOptions(options: CSGBuildOptions): this {
		this._buildOptions = { ...this._buildOptions, ...options };
		this.markCSGDirty();
		return this;
	}

	public markCSGDirty(): void {
		this._dirty = true;
		this.scene?.invalidate("transform");
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

		if (this._executionMode === "worker" && this._executor) {
			return this._executor
				.execute(this._graph, buildOptions)
				.then((result) => this._applyAsyncResult(token, result));
		}

		if (this._executionMode === "worker") {
			return Promise.resolve().then(() =>
				this._applyAsyncResult(token, this._buildSyncResult(buildOptions, true))
			);
		}

		return this._applyResult(token, this._buildSyncResult(buildOptions, false));
	}

	private _buildSyncResult(
		options: CSGBuildOptions,
		fromWorkerFallback: boolean,
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
		if (fromWorkerFallback) {
			result.diagnostics.push({
				code: "csg-worker-fallback-sync",
				message: Platform.hasWorker()
					? "CSG worker execution requested but no worker executor is configured; fell back to sync execution"
					: "CSG worker execution requested but Worker is unavailable; fell back to sync execution",
				severity: "warning",
			});
		}
		return result;
	}

	private _applyAsyncResult(
		token: number,
		result: CSGRebuildResult
	): CSGRebuildResult {
		return this._applyResult(token, result);
	}

	private _applyResult(token: number, result: CSGRebuildResult): CSGRebuildResult {
		if (token !== this._latestRequestedToken) {
			return {
				...result,
				stale: true,
			};
		}

		if (result.ok && result.meshAsset) {
			for (const primitive of result.meshAsset.primitives) {
				result.meshAsset.markPrimitiveGeometryDirty(primitive);
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
		const scene = this.scene;
		if (!scene) return;
		scene.invalidate("transform");
		scene.spatial?.markDirty(this);
	}

	protected override _createCloneInstance(): this {
		return new CSGMeshInstance({
			mesh: this.mesh,
			graph: this._graph ? new CSGBuilder(this._graph).getGraph() : undefined,
			buildOptions: { ...this._buildOptions },
			executor: this._executor,
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
		target._executor = this._executor;
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
