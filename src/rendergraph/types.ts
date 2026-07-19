import type { FramePassStage } from "../pipeline/types";

/** @internal Backend-private logical render graph resource identifier. */
export type RenderGraphResourceId = string;

/** @internal Backend-private logical render graph node identifier. */
export type RenderGraphNodeId = string;

/** @internal Logical resource allocation origin. */
export type RenderGraphResourceOrigin = "imported" | "graph";

/** @internal Logical resource kind without a backend-native handle. */
export type RenderGraphResourceKind = "texture" | "buffer" | "external";

/** @internal Logical residency policy; it does not transfer native ownership. */
export type RenderGraphResourceResidency =
	| "external"
	| "frame"
	| "transient"
	| "history";

/** @internal Whether the logical contents can be consumed before a graph write. */
export type RenderGraphContentState = "valid" | "undefined" | "unknown";

/** @internal Logical resource access mode. */
export type RenderGraphAccess = "read" | "write" | "read-write";

/** @internal Backend-agnostic resource usage intent. */
export type RenderGraphUsage =
	| "sampled"
	| "color-attachment"
	| "depth-attachment"
	| "storage"
	| "copy-source"
	| "copy-target"
	| "present"
	| "cpu-read"
	| "cpu-write";

/** @internal Execution domain metadata reserved for whole-frame scheduling. */
export type RenderGraphExecutionDomain =
	| "graphics"
	| "compute"
	| "copy"
	| "cpu";

/** @internal Optional logical subresource range retained for future analysis. */
export interface RenderGraphSubresourceRange {
	readonly mipStart?: number;
	readonly mipCount?: number;
	readonly layerStart?: number;
	readonly layerCount?: number;
	readonly aspect?: "all" | "color" | "depth" | "stencil";
}

/** @internal Logical resource descriptor consumed by Render Graph analysis. */
export interface RenderGraphResourceDescriptor {
	readonly id: RenderGraphResourceId;
	readonly origin: RenderGraphResourceOrigin;
	readonly kind?: RenderGraphResourceKind;
	readonly residency?: RenderGraphResourceResidency;
	readonly initialContent?: RenderGraphContentState;
	readonly optional?: boolean;
	readonly format?: string;
	readonly width?: number;
	readonly height?: number;
	readonly mipMode?: "single" | "full-chain";
}

/** @internal One logical resource access declared by a graph node. */
export interface RenderGraphResourceRef {
	readonly resource: RenderGraphResourceId;
	readonly access: RenderGraphAccess;
	readonly usage: RenderGraphUsage;
	readonly optional?: boolean;
	readonly subresource?: RenderGraphSubresourceRange;
}

/** @internal A lifetime mutation accepted in legacy or descriptor form. */
export type RenderGraphResourceMutation =
	| RenderGraphResourceId
	| {
			readonly resource: RenderGraphResourceId;
			readonly usage?: RenderGraphUsage;
			readonly optional?: boolean;
	  };

/** @internal An existence requirement that does not imply a resource access. */
export interface RenderGraphResourceRequirement {
	readonly resource: RenderGraphResourceId;
	readonly optional?: boolean;
}

/** @internal A backend-private logical render graph node. */
export interface RenderGraphNode<
	TPayload = unknown,
	TKind extends string = string,
> {
	readonly id: RenderGraphNodeId;
	readonly stage: FramePassStage;
	readonly kind: TKind;
	readonly label: string;
	readonly domain?: RenderGraphExecutionDomain;
	readonly dependsOn?: readonly RenderGraphNodeId[];
	readonly requires?: readonly RenderGraphResourceRequirement[];
	readonly creates?: readonly RenderGraphResourceMutation[];
	readonly destroys?: readonly RenderGraphResourceMutation[];
	readonly resources?: readonly RenderGraphResourceRef[];
	readonly payload?: TPayload;
}

/** @internal Compiler diagnostic classification. */
export type RenderGraphDiagnosticCode =
	| "duplicate-node"
	| "duplicate-resource"
	| "missing-dependency"
	| "cycle"
	| "read-before-create"
	| "duplicate-create"
	| "destroy-before-create"
	| "missing-resource"
	| "undeclared-resource"
	| "implicit-resource-declaration"
	| "implicit-create"
	| "read-content-unknown"
	| "read-before-initialize"
	| "use-after-destroy"
	| "opaque-stage-effects"
	| string;

/** @internal Compiler or backend-rule diagnostic for a logical graph. */
export interface RenderGraphDiagnostic {
	readonly phase: "build" | "compile" | "lower" | "execute" | "commit";
	readonly enforcement: "enforced" | "shadow";
	readonly severity: "error" | "warning";
	readonly code: RenderGraphDiagnosticCode;
	readonly backend?: string;
	readonly stage?: FramePassStage;
	readonly nodeId?: RenderGraphNodeId;
	readonly resourceId?: RenderGraphResourceId;
	readonly message: string;
	readonly cause?: unknown;
}

/** @internal One logical access transition retained for lowering and diagnostics. */
export interface RenderGraphTransition {
	readonly nodeId: RenderGraphNodeId;
	readonly fromNodeId?: RenderGraphNodeId;
	readonly resourceId: RenderGraphResourceId;
	readonly generation: number;
	readonly previousAccess?: RenderGraphAccess;
	readonly previousUsage?: RenderGraphUsage;
	readonly access: RenderGraphAccess;
	readonly usage: RenderGraphUsage;
	readonly subresource?: RenderGraphSubresourceRange;
	readonly scope: "initial" | "inter-node" | "intra-node";
	readonly hazard?: "read-after-write" | "write-after-read" | "write-after-write";
	readonly reason?:
		| "read-after-write"
		| "write-after-read"
		| "write-after-write"
		| "usage-transition";
}

/** @internal One generation-aware logical resource live range. */
export interface RenderGraphResourceLifetime {
	readonly resourceId: RenderGraphResourceId;
	readonly generation: number;
	readonly firstNodeId: RenderGraphNodeId;
	readonly lastNodeId: RenderGraphNodeId;
	readonly createdByNodeId?: RenderGraphNodeId;
	readonly firstUseNodeId?: RenderGraphNodeId;
	readonly lastUseNodeId?: RenderGraphNodeId;
	readonly destroyedByNodeId?: RenderGraphNodeId;
}

/** @internal Preferred name for generation-aware resource lifetime output. */
export type RenderGraphLiveRange = RenderGraphResourceLifetime;

/** @internal Sanitized state for one logical resource generation. */
export interface RenderGraphResourceDebugState {
	readonly id: RenderGraphResourceId;
	readonly origin: RenderGraphResourceOrigin;
	readonly active: boolean;
	readonly generation: number;
	readonly content: RenderGraphContentState;
	readonly lastNodeId: RenderGraphNodeId | null;
	readonly lastAccess:
		| "create"
		| "read"
		| "write"
		| "read-write"
		| "destroy"
		| null;
	readonly lastUsage: RenderGraphUsage | null;
}

/** @internal Immutable logical graph compiler result. */
export interface CompiledRenderGraph<
	TPayload = unknown,
	TKind extends string = string,
> {
	readonly nodes: readonly RenderGraphNode<TPayload, TKind>[];
	readonly resources: readonly RenderGraphResourceDescriptor[];
	readonly diagnostics: readonly RenderGraphDiagnostic[];
	readonly shadowDiagnostics: readonly RenderGraphDiagnostic[];
	readonly transitions: readonly RenderGraphTransition[];
	readonly lifetimes: readonly RenderGraphResourceLifetime[];
	readonly liveRanges: readonly RenderGraphLiveRange[];
}

/** @internal One streaming stage result produced without reordering nodes. */
export interface RenderGraphAnalyzedStage<
	TPayload = unknown,
	TKind extends string = string,
> {
	readonly nodes: readonly RenderGraphNode<TPayload, TKind>[];
	readonly diagnostics: readonly RenderGraphDiagnostic[];
	readonly shadowDiagnostics: readonly RenderGraphDiagnostic[];
	readonly transitions: readonly RenderGraphTransition[];
}

/** @internal Analysis completeness for graph-visible frame work. */
export type RenderGraphAnalysisCompleteness = "complete" | "coarse" | "opaque";

/** @internal Immutable snapshot of one streaming graph attempt. */
export interface RenderGraphAnalysisSnapshot {
	readonly state: "active" | "sealed" | "committed" | "aborted";
	readonly completeness: RenderGraphAnalysisCompleteness;
	readonly nodeIds: readonly RenderGraphNodeId[];
	readonly resources: readonly RenderGraphResourceDebugState[];
	readonly transitions: readonly RenderGraphTransition[];
	readonly liveRanges: readonly RenderGraphLiveRange[];
	readonly diagnostics: readonly RenderGraphDiagnostic[];
	readonly shadowDiagnostics: readonly RenderGraphDiagnostic[];
}

/** @internal Current and retained outcomes for one reusable graph tracker. */
export interface RenderGraphTrackerDebugState {
	readonly state: "idle" | "active" | "sealed" | "committed" | "aborted";
	readonly current: RenderGraphAnalysisSnapshot | null;
	readonly lastAttempt: RenderGraphAnalysisSnapshot | null;
	readonly lastSuccessful: RenderGraphAnalysisSnapshot | null;
}

/** @internal Pure validation extension composed by backend-private facades. */
export interface RenderGraphValidationRule<
	TPayload = unknown,
	TKind extends string = string,
> {
	validateNode(
		node: RenderGraphNode<TPayload, TKind>,
		context: {
			readonly isResourceActive: (resourceId: RenderGraphResourceId) => boolean;
		},
	): readonly RenderGraphDiagnostic[];
}
