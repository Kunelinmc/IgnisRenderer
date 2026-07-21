import type { FramePassStage } from "../pipeline/types";

/** @internal Backend-private logical render graph resource identifier. */
export type RenderGraphResourceId = string;

/** @internal Backend-private logical render graph node identifier. */
export type RenderGraphNodeId = string;

/** @internal Backend-private stable physical resource identifier. */
export type RenderGraphPhysicalResourceId = string;

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

/** @internal Whether logical contents can be consumed before a graph write. */
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

/** @internal Execution domain metadata reserved for backend scheduling. */
export type RenderGraphExecutionDomain = "graphics" | "compute" | "copy" | "cpu";

/** @internal Texture aspect selected by one logical access. */
export type RenderGraphTextureAspect = "color" | "depth" | "stencil";

/** @internal Optional logical texture range. Omission means the full texture. */
export interface RenderGraphTextureSubresourceRange {
	readonly kind: "texture";
	readonly mipStart?: number;
	readonly mipCount?: number;
	readonly layerStart?: number;
	readonly layerCount?: number;
	readonly aspects?: readonly RenderGraphTextureAspect[];
}

/** @internal Optional logical buffer range. Omission means the full buffer. */
export interface RenderGraphBufferSubresourceRange {
	readonly kind: "buffer";
	readonly offset?: number;
	readonly size?: number;
}

/** @internal Logical texture or buffer range retained by analysis. */
export type RenderGraphSubresourceRange =
	| RenderGraphTextureSubresourceRange
	| RenderGraphBufferSubresourceRange;

/** @internal Normalized texture range with concrete bounds. */
export interface RenderGraphNormalizedTextureRange {
	readonly kind: "texture";
	readonly mipStart: number;
	readonly mipCount: number;
	readonly layerStart: number;
	readonly layerCount: number;
	readonly aspects: readonly RenderGraphTextureAspect[];
}

/** @internal Normalized buffer range with concrete bounds. */
export interface RenderGraphNormalizedBufferRange {
	readonly kind: "buffer";
	readonly offset: number;
	readonly size: number;
}

/** @internal Concrete range used for overlap analysis. */
export type RenderGraphNormalizedSubresourceRange =
	| RenderGraphNormalizedTextureRange
	| RenderGraphNormalizedBufferRange;

interface RenderGraphResourceDescriptorBase {
	readonly id: RenderGraphResourceId;
	readonly origin: RenderGraphResourceOrigin;
	readonly residency?: RenderGraphResourceResidency;
	readonly initialContent?: RenderGraphContentState;
	readonly optional?: boolean;
}

/** @internal Logical texture descriptor without a native texture handle. */
export interface RenderGraphTextureDescriptor extends RenderGraphResourceDescriptorBase {
	readonly kind: "texture";
	readonly format?: string;
	readonly width?: number;
	readonly height?: number;
	readonly depthOrArrayLayers?: number;
	readonly dimension?: "1d" | "2d" | "3d";
	readonly sampleCount?: number;
	readonly mipLevelCount?: number;
	readonly allowedUsages?: readonly RenderGraphUsage[];
	/** @deprecated Use `mipLevelCount`. */
	readonly mipMode?: "single" | "full-chain";
}

/** @internal Logical buffer descriptor without a native buffer handle. */
export interface RenderGraphBufferDescriptor extends RenderGraphResourceDescriptorBase {
	readonly kind: "buffer";
	readonly size?: number;
	readonly allowedUsages?: readonly RenderGraphUsage[];
}

/** @internal Opaque backend or host resource descriptor. */
export interface RenderGraphExternalDescriptor extends RenderGraphResourceDescriptorBase {
	readonly kind: "external";
	readonly allowedUsages?: readonly RenderGraphUsage[];
}

/** @internal Logical resource descriptor consumed by Render Graph analysis. */
export type RenderGraphResourceDescriptor =
	| RenderGraphTextureDescriptor
	| RenderGraphBufferDescriptor
	| RenderGraphExternalDescriptor;

/** @internal Stable logical-to-physical identity without a native handle. */
export interface RenderGraphPhysicalBinding {
	readonly resourceId: RenderGraphResourceId;
	readonly generation?: number;
	readonly physicalId: RenderGraphPhysicalResourceId;
	readonly kind: RenderGraphResourceKind;
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
export interface RenderGraphNode<TPayload = unknown, TKind extends string = string> {
	readonly id: RenderGraphNodeId;
	readonly stage: FramePassStage;
	readonly kind: TKind;
	readonly label: string;
	readonly domain?: RenderGraphExecutionDomain;
	readonly retention?: "always" | "if-reachable";
	readonly opaque?: boolean;
	readonly dependsOn?: readonly RenderGraphNodeId[];
	readonly requires?: readonly RenderGraphResourceRequirement[];
	readonly creates?: readonly RenderGraphResourceMutation[];
	readonly destroys?: readonly RenderGraphResourceMutation[];
	readonly resources?: readonly RenderGraphResourceRef[];
	readonly payload?: TPayload;
}

/** @internal One final logical graph output used as a DCE root. */
export interface RenderGraphExport {
	readonly name?: string;
	readonly resource: RenderGraphResourceId;
	readonly subresource?: RenderGraphSubresourceRange;
}

/** @internal A named subgraph resource port. */
export interface RenderGraphSubgraphPort {
	readonly name: string;
	readonly resource: RenderGraphResourceId;
	readonly optional?: boolean;
}

/** @internal One resolved named port retained for graph tooling. */
export interface RenderGraphPortResolution {
	readonly namespace?: string;
	readonly direction: "import" | "export";
	readonly port: string;
	readonly resource: RenderGraphResourceId;
}

/** @internal Immutable input accepted by the whole-frame compiler. */
export interface RenderGraphDefinition<TPayload = unknown, TKind extends string = string> {
	readonly resources: readonly RenderGraphResourceDescriptor[];
	readonly bindings?: readonly RenderGraphPhysicalBinding[];
	readonly nodes: readonly RenderGraphNode<TPayload, TKind>[];
	readonly exports?: readonly RenderGraphExport[];
	readonly imports?: readonly RenderGraphSubgraphPort[];
	readonly outputPorts?: readonly RenderGraphSubgraphPort[];
	readonly portResolutions?: readonly RenderGraphPortResolution[];
	readonly completeness?: RenderGraphAnalysisCompleteness;
	readonly shadowDiagnostics?: readonly RenderGraphDiagnostic[];
	readonly buildDiagnostics?: readonly RenderGraphDiagnostic[];
}

/** @internal Compiler diagnostic classification. */
export type RenderGraphDiagnosticCode =
	| "duplicate-node"
	| "duplicate-resource"
	| "duplicate-binding"
	| "missing-dependency"
	| "cycle"
	| "invalid-resource-descriptor"
	| "invalid-subresource-range"
	| "binding-kind-mismatch"
	| "missing-binding"
	| "missing-subgraph-port"
	| "duplicate-subgraph-port"
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
	| "physical-feedback-loop"
	| "opaque-stage-effects"
	| (string & {});

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

/** @internal One explicit or inferred ordering edge. */
export interface RenderGraphDependency {
	readonly fromNodeId: RenderGraphNodeId;
	readonly toNodeId: RenderGraphNodeId;
	readonly kind:
		| "explicit"
		| "read-after-write"
		| "write-after-read"
		| "write-after-write"
		| "usage-transition";
	readonly resourceId?: RenderGraphResourceId;
	readonly physicalId?: RenderGraphPhysicalResourceId;
	readonly subresource?: RenderGraphNormalizedSubresourceRange;
}

/** @internal One logical access transition retained for lowering and diagnostics. */
export interface RenderGraphTransition {
	readonly nodeId: RenderGraphNodeId;
	readonly fromNodeId?: RenderGraphNodeId;
	readonly resourceId: RenderGraphResourceId;
	readonly physicalId?: RenderGraphPhysicalResourceId;
	readonly generation: number;
	readonly previousAccess?: RenderGraphAccess;
	readonly previousUsage?: RenderGraphUsage;
	readonly access: RenderGraphAccess;
	readonly usage: RenderGraphUsage;
	readonly subresource?: RenderGraphNormalizedSubresourceRange;
	readonly scope: "initial" | "inter-node" | "intra-node";
	readonly hazard?: "read-after-write" | "write-after-read" | "write-after-write";
	readonly reason?:
		| "read-after-write"
		| "write-after-read"
		| "write-after-write"
		| "usage-transition";
}

/** @internal One generation-aware logical resource live range. */
export interface RenderGraphLiveRange {
	readonly resourceId: RenderGraphResourceId;
	readonly generation: number;
	readonly firstNodeId: RenderGraphNodeId;
	readonly lastNodeId: RenderGraphNodeId;
	readonly createdByNodeId?: RenderGraphNodeId;
	readonly firstUseNodeId?: RenderGraphNodeId;
	readonly lastUseNodeId?: RenderGraphNodeId;
	readonly destroyedByNodeId?: RenderGraphNodeId;
}

/** @internal One normalized subresource lifetime. */
export interface RenderGraphSubresourceLiveRange extends RenderGraphLiveRange {
	readonly subresource: RenderGraphNormalizedSubresourceRange;
}

/** @internal Logical allocation intent; it never owns a native allocation. */
export interface RenderGraphAllocationRequest {
	readonly resourceId: RenderGraphResourceId;
	readonly generation: number;
	readonly compatibilityKey: string;
	readonly allocateBeforeNodeId: RenderGraphNodeId;
	readonly releaseAfterNodeId: RenderGraphNodeId;
}

/** @internal Sanitized state for one logical resource generation. */
export interface RenderGraphResourceDebugState {
	readonly id: RenderGraphResourceId;
	readonly origin: RenderGraphResourceOrigin;
	readonly active: boolean;
	readonly generation: number;
	readonly content: RenderGraphContentState;
	readonly physicalId?: RenderGraphPhysicalResourceId;
	readonly lastNodeId: RenderGraphNodeId | null;
	readonly lastAccess: "create" | "read" | "write" | "read-write" | "destroy" | null;
	readonly lastUsage: RenderGraphUsage | null;
}

/** @internal One retained execution slice grouped by renderer pass stage. */
export interface CompiledRenderGraphStage<TPayload = unknown, TKind extends string = string> {
	readonly stage: FramePassStage;
	readonly nodes: readonly RenderGraphNode<TPayload, TKind>[];
}

/** @internal Immutable logical graph compiler result. */
export interface CompiledRenderGraph<TPayload = unknown, TKind extends string = string> {
	readonly declaredNodes: readonly RenderGraphNode<TPayload, TKind>[];
	readonly nodes: readonly RenderGraphNode<TPayload, TKind>[];
	readonly culledNodeIds: readonly RenderGraphNodeId[];
	readonly stages: readonly CompiledRenderGraphStage<TPayload, TKind>[];
	readonly resources: readonly RenderGraphResourceDescriptor[];
	readonly bindings: readonly RenderGraphPhysicalBinding[];
	readonly exports: readonly RenderGraphExport[];
	readonly portResolutions: readonly RenderGraphPortResolution[];
	readonly dependencies: readonly RenderGraphDependency[];
	readonly diagnostics: readonly RenderGraphDiagnostic[];
	readonly shadowDiagnostics: readonly RenderGraphDiagnostic[];
	readonly transitions: readonly RenderGraphTransition[];
	readonly liveRanges: readonly RenderGraphLiveRange[];
	readonly subresourceLiveRanges: readonly RenderGraphSubresourceLiveRange[];
	readonly allocationRequests: readonly RenderGraphAllocationRequest[];
	readonly completeness: RenderGraphAnalysisCompleteness;
}

/** @internal One streaming stage result retained for legacy analysis clients. */
export interface RenderGraphAnalyzedStage<TPayload = unknown, TKind extends string = string> {
	readonly nodes: readonly RenderGraphNode<TPayload, TKind>[];
	readonly diagnostics: readonly RenderGraphDiagnostic[];
	readonly shadowDiagnostics: readonly RenderGraphDiagnostic[];
	readonly transitions: readonly RenderGraphTransition[];
}

/** @internal Analysis completeness for graph-visible frame work. */
export type RenderGraphAnalysisCompleteness = "complete" | "coarse" | "opaque";

/** @internal Immutable snapshot of one graph attempt. */
export interface RenderGraphAnalysisSnapshot {
	readonly state: "active" | "sealed" | "committed" | "aborted";
	readonly completeness: RenderGraphAnalysisCompleteness;
	readonly nodeIds: readonly RenderGraphNodeId[];
	readonly declaredNodeIds?: readonly RenderGraphNodeId[];
	readonly culledNodeIds?: readonly RenderGraphNodeId[];
	readonly resources: readonly RenderGraphResourceDebugState[];
	readonly transitions: readonly RenderGraphTransition[];
	readonly dependencies?: readonly RenderGraphDependency[];
	readonly liveRanges: readonly RenderGraphLiveRange[];
	readonly subresourceLiveRanges?: readonly RenderGraphSubresourceLiveRange[];
	readonly allocationRequests?: readonly RenderGraphAllocationRequest[];
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
export interface RenderGraphValidationRule<TPayload = unknown, TKind extends string = string> {
	validateNode(
		node: RenderGraphNode<TPayload, TKind>,
		context: {
			readonly isResourceActive: (resourceId: RenderGraphResourceId) => boolean;
		},
	): readonly RenderGraphDiagnostic[];
}
