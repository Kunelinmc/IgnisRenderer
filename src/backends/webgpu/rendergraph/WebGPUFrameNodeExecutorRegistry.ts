import type {
	WebGPUFrameGraphNode,
	WebGPUFrameGraphNodeKind,
} from "./types";
import type { WebGPUFrameSession } from "./WebGPUFrameSession";
import { WEBGPU_FRAME_GRAPH_NODE_KINDS } from "./types";
import type { WebGPUFrameGraphModule } from "./WebGPUFrameGraphModule";

export type WebGPUFrameNodeExecutor = (
	node: WebGPUFrameGraphNode,
	session: WebGPUFrameSession,
) => Promise<void>;

export type WebGPUFrameNodeExecutorTable = {
	readonly [Kind in WebGPUFrameGraphNodeKind]: WebGPUFrameNodeExecutor;
};

/**
 * Resolves WebGPU frame graph node kinds to their runtime executors.
 *
 * @internal Owned by the WebGPU frame graph runtime. Applications should use
 * `Renderer.renderFrame()` instead.
 */
export class WebGPUFrameNodeExecutorRegistry {
	private readonly _executors: ReadonlyMap<string, WebGPUFrameNodeExecutor>;
	private readonly _ownersByKind: ReadonlyMap<
		WebGPUFrameGraphNodeKind,
		readonly string[]
	>;

	constructor(
		executors: ReadonlyMap<string, WebGPUFrameNodeExecutor>,
		ownersByKind: ReadonlyMap<WebGPUFrameGraphNodeKind, readonly string[]>,
	) {
		this._executors = executors;
		this._ownersByKind = ownersByKind;
	}

	public static fromModules(
		modules: readonly WebGPUFrameGraphModule[],
	): WebGPUFrameNodeExecutorRegistry {
		const executors = new Map<string, WebGPUFrameNodeExecutor>();
		const ownersByKind = new Map<WebGPUFrameGraphNodeKind, string[]>();
		for (const module of modules) {
			for (const [kind, executor] of Object.entries(module.executors)) {
				const nodeKind = kind as WebGPUFrameGraphNodeKind;
				const key = executorKey(module.id, nodeKind);
				if (executors.has(key)) {
					throw new Error(
						`WebGPU node executor "${key}" is registered more than once.`,
					);
				}
				executors.set(key, executor);
				const owners = ownersByKind.get(nodeKind) ?? [];
				owners.push(module.id);
				ownersByKind.set(nodeKind, owners);
			}
		}
		const missing = WEBGPU_FRAME_GRAPH_NODE_KINDS.filter(
			(kind) => (ownersByKind.get(kind)?.length ?? 0) === 0,
		);
		if (missing.length > 0) {
			throw new Error(
				`WebGPU frame modules are missing executors: ${missing.join(", ")}.`,
			);
		}
		return new WebGPUFrameNodeExecutorRegistry(executors, ownersByKind);
	}

	public async execute(
		node: WebGPUFrameGraphNode,
		session: WebGPUFrameSession,
	): Promise<void> {
		const owners = this._ownersByKind.get(node.kind) ?? [];
		const ownerId = node.ownerId ?? (owners.length === 1 ? owners[0] : null);
		const executor = ownerId
			? this._executors.get(executorKey(ownerId, node.kind))
			: undefined;
		if (typeof executor !== "function") {
			throw new Error(
				`WebGPU frame graph node "${node.id}" has no owner-aware executor.`,
			);
		}
		await executor(node, session);
	}
}

function executorKey(ownerId: string, kind: WebGPUFrameGraphNodeKind): string {
	return `${ownerId}:${kind}`;
}
