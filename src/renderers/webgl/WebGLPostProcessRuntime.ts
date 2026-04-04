import type { FrameContext, ResolvedFeatureState } from "../../pipeline/types";
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
		features: ResolvedFeatureState,
		warn: (key: string, message: string) => void,
		allowedPassIds?: ReadonlySet<string>
	): string[] {
		const hints = new Set<string>();
		const orderedPasses = this._graph.getExecutionOrder(features, warn);
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
		features: ResolvedFeatureState,
		warn: (key: string, message: string) => void
	): string[] {
		const context: WebGLPostProcessPassContext = {
			frameContext,
		};
		return this._graph.execute(context, features, warn);
	}
}

export type {
	WebGLPostProcessPassContext,
	WebGLPostProcessPassPlugin,
} from "./WebGLPostProcessGraph";
