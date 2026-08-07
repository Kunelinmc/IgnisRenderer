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
	private readonly _executors: WebGPUFrameNodeExecutorTable;

	constructor(executors: WebGPUFrameNodeExecutorTable) {
		this._executors = executors;
	}

	public static fromModules(
		modules: readonly WebGPUFrameGraphModule[],
	): WebGPUFrameNodeExecutorRegistry {
		const executors: Partial<Record<WebGPUFrameGraphNodeKind, WebGPUFrameNodeExecutor>> = {};
		for (const module of modules) {
			for (const [kind, executor] of Object.entries(module.executors)) {
				const nodeKind = kind as WebGPUFrameGraphNodeKind;
				if (executors[nodeKind]) {
					throw new Error(
						`WebGPU node kind "${nodeKind}" has duplicate module owners.`,
					);
				}
				executors[nodeKind] = executor;
			}
		}
		const missing = WEBGPU_FRAME_GRAPH_NODE_KINDS.filter(
			(kind) => typeof executors[kind] !== "function",
		);
		if (missing.length > 0) {
			throw new Error(
				`WebGPU frame modules are missing executors: ${missing.join(", ")}.`,
			);
		}
		return new WebGPUFrameNodeExecutorRegistry(
			executors as WebGPUFrameNodeExecutorTable,
		);
	}

	public async execute(
		node: WebGPUFrameGraphNode,
		session: WebGPUFrameSession,
	): Promise<void> {
		const executor = this._executors[node.kind];
		if (typeof executor !== "function") {
			throw new Error(
				`WebGPU frame graph node kind "${node.kind}" has no executor.`,
			);
		}
		await executor(node, session);
	}
}
