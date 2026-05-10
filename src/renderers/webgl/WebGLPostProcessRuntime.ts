import type { FrameContext } from "../../pipeline/types";
import type { ResolvedPostProcessState } from "../../pipeline/PostProcess";
import {
	WebGLPostProcessGraph,
	type WebGLPostProcessPassContext,
	type WebGLPostProcessPassPlugin,
} from "./WebGLPostProcessGraph";

export class WebGLPostProcessRuntime {
	private _graph: WebGLPostProcessGraph;

	constructor(passes: readonly WebGLPostProcessPassPlugin[] = []) {
		this._graph = new WebGLPostProcessGraph(passes);
	}

	public registerPass(pass: WebGLPostProcessPassPlugin): void {
		this._graph.registerPass(pass);
	}

	public unregisterPass(id: string): void {
		this._graph.unregisterPass(id);
	}

	public hasPass(id: string): boolean {
		return this._graph.hasPass(id);
	}

	public collectWarmupHints(
		postProcess: ResolvedPostProcessState,
		warn: (key: string, message: string) => void,
		allowedPassIds?: ReadonlySet<string>
	): string[] {
		const hints = new Set<string>();
		const orderedPasses = this._graph.getExecutionOrder(postProcess, warn);
		for (const pass of orderedPasses) {
			if (allowedPassIds && !allowedPassIds.has(pass.id)) {
				continue;
			}
			for (const hint of pass.precompileHints ?? [`postprocess:${pass.id}`]) {
				hints.add(hint);
			}
		}
		return Array.from(hints);
	}

	public execute(
		frameContext: FrameContext,
		postProcess: ResolvedPostProcessState,
		warn: (key: string, message: string) => void
	): string[] {
		const context: WebGLPostProcessPassContext = {
			frameContext,
			postProcess,
		};
		return this._graph.execute(context, postProcess, warn);
	}
}

export type {
	WebGLPostProcessPassContext,
	WebGLPostProcessPassPlugin,
} from "./WebGLPostProcessGraph";
