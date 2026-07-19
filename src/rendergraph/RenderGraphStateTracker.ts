import { RenderGraphAnalyzer, type RenderGraphAnalyzerOptions } from "./RenderGraphAnalyzer";
import type {
	RenderGraphAnalysisCompleteness,
	RenderGraphAnalysisSnapshot,
	RenderGraphAnalyzedStage,
	RenderGraphDiagnostic,
	RenderGraphLiveRange,
	RenderGraphNode,
	RenderGraphResourceDebugState,
	RenderGraphResourceDescriptor,
	RenderGraphTrackerDebugState,
	RenderGraphTransition,
} from "./types";

/** @internal Reusable streaming tracker for stage-by-stage backend execution. */
export class RenderGraphStateTracker<TPayload = unknown, TKind extends string = string> {
	private readonly _analyzer: RenderGraphAnalyzer<TPayload, TKind>;
	private _state: RenderGraphTrackerDebugState["state"] = "idle";
	private _lastAttempt: RenderGraphAnalysisSnapshot | null = null;
	private _lastSuccessful: RenderGraphAnalysisSnapshot | null = null;

	constructor(options: RenderGraphAnalyzerOptions<TPayload, TKind> = {}) {
		this._analyzer = new RenderGraphAnalyzer({
			...options,
			validateStreamingDependencies: options.validateStreamingDependencies ?? true,
		});
	}

	/** @internal Resets analysis state for one backend frame. */
	public beginFrame(resources: readonly RenderGraphResourceDescriptor[]): void {
		if (this._state === "active" || this._state === "sealed") {
			throw new Error("Render graph tracker already has an active frame.");
		}
		this._analyzer.reset(resources);
		this._state = "active";
	}

	/** @internal Appends one ordered stage without global reordering. */
	public appendStage(request: {
		readonly nodes: readonly RenderGraphNode<TPayload, TKind>[];
	}): RenderGraphAnalyzedStage<TPayload, TKind> {
		this._assertActive("append a stage");
		return this._analyzer.analyzeNodes(request.nodes);
	}

	/** @internal Marks graph recording complete but not yet committed. */
	public seal(): void {
		this._assertActive("seal the frame");
		this._state = "sealed";
	}

	/** @internal Retains a successful snapshot after the complete backend commit. */
	public commit(): void {
		if (this._state !== "active" && this._state !== "sealed") return;
		const snapshot = this._analyzer.createSnapshot("committed");
		this._lastAttempt = snapshot;
		this._lastSuccessful = snapshot;
		this._state = "committed";
	}

	/** @internal Retains a failed attempt without replacing the last success. */
	public abort(_error?: unknown): void {
		if (this._state !== "active" && this._state !== "sealed") return;
		this._lastAttempt = this._analyzer.createSnapshot("aborted");
		this._state = "aborted";
	}

	public markCompleteness(completeness: RenderGraphAnalysisCompleteness): void {
		this._assertActive("mark graph completeness");
		this._analyzer.markCompleteness(completeness);
	}

	public recordOpaqueStage(stage: string, message: string): void {
		this._assertActive("record an opaque stage");
		this._analyzer.recordOpaqueStage(stage, message);
	}

	public getDiagnostics(): readonly RenderGraphDiagnostic[] {
		return this._analyzer.getDiagnostics();
	}

	public getShadowDiagnostics(): readonly RenderGraphDiagnostic[] {
		return this._analyzer.getShadowDiagnostics();
	}

	public getTransitions(): readonly RenderGraphTransition[] {
		return this._analyzer.getTransitions();
	}

	public getLiveRanges(): readonly RenderGraphLiveRange[] {
		return this._analyzer.getLiveRanges();
	}

	public getResourceDebugState(): readonly RenderGraphResourceDebugState[] {
		return this._analyzer.getResourceDebugState();
	}

	public getDebugState(): RenderGraphTrackerDebugState {
		const current =
			this._state === "active" || this._state === "sealed"
				? this._analyzer.createSnapshot(this._state)
				: null;
		return Object.freeze({
			state: this._state,
			current,
			lastAttempt: this._lastAttempt,
			lastSuccessful: this._lastSuccessful,
		});
	}

	private _assertActive(action: string): void {
		if (this._state !== "active") {
			throw new Error(`Render graph tracker cannot ${action} in state "${this._state}".`);
		}
	}
}
