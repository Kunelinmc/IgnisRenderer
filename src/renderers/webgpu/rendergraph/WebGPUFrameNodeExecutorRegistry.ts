import type {
	WebGPUFrameGraphNode,
	WebGPUFrameGraphNodeKind,
} from "./types";
import type { WebGPUFrameSession } from "./WebGPUFrameSession";

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
