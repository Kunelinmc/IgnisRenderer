import type { FramePassStage } from "../pipeline/types";

/** @internal Backend-private logical render graph resource identifier. */
export type RenderGraphResourceId = string;

/** @internal Backend-private logical render graph node identifier. */
export type RenderGraphNodeId = string;

/** @internal Logical resource allocation origin. */
export type RenderGraphResourceOrigin = "imported" | "graph";

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
	| "cpu-read"
	| "cpu-write";

/** @internal Logical resource descriptor consumed by `RenderGraphCompiler`. */
export interface RenderGraphResourceDescriptor {
	readonly id: RenderGraphResourceId;
	readonly origin: RenderGraphResourceOrigin;
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
}

/** @internal A backend-private logical render graph node. */
export interface RenderGraphNode<TPayload = unknown> {
	readonly id: RenderGraphNodeId;
	readonly stage: FramePassStage;
	readonly kind: string;
	readonly label: string;
	readonly dependsOn?: readonly RenderGraphNodeId[];
	readonly creates?: readonly RenderGraphResourceId[];
	readonly resources?: readonly RenderGraphResourceRef[];
	readonly payload?: TPayload;
}

/** @internal Compiler diagnostic for a logical graph error. */
export interface RenderGraphDiagnostic {
	readonly severity: "error" | "warning";
	readonly code:
		| "duplicate-node"
		| "duplicate-resource"
		| "missing-dependency"
		| "cycle"
		| "read-before-create"
		| "duplicate-create";
	readonly nodeId?: RenderGraphNodeId;
	readonly resourceId?: RenderGraphResourceId;
	readonly message: string;
}

/** @internal One logical access transition retained for diagnostics. */
export interface RenderGraphTransition {
	readonly nodeId: RenderGraphNodeId;
	readonly resourceId: RenderGraphResourceId;
	readonly previousAccess?: RenderGraphAccess;
	readonly access: RenderGraphAccess;
	readonly usage: RenderGraphUsage;
}

/** @internal First and last planned node use for one logical resource. */
export interface RenderGraphResourceLifetime {
	readonly resourceId: RenderGraphResourceId;
	readonly firstNodeId: RenderGraphNodeId;
	readonly lastNodeId: RenderGraphNodeId;
}

/** @internal Immutable logical graph compiler result. */
export interface CompiledRenderGraph<TPayload = unknown> {
	readonly nodes: readonly RenderGraphNode<TPayload>[];
	readonly resources: readonly RenderGraphResourceDescriptor[];
	readonly diagnostics: readonly RenderGraphDiagnostic[];
	readonly transitions: readonly RenderGraphTransition[];
	readonly lifetimes: readonly RenderGraphResourceLifetime[];
}
