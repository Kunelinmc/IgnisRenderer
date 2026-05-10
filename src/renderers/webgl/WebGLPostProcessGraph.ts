import type { FrameContext } from "../../pipeline/types";
import type { ResolvedPostProcessState } from "../../pipeline/PostProcessController";

export interface WebGLPostProcessPassContext {
	frameContext: FrameContext;
	postProcess: ResolvedPostProcessState;
}

export interface WebGLPostProcessPassPlugin {
	id: string;
	dependsOn: string[];
	precompileHints?: string[];
	isEnabled(postProcess: ResolvedPostProcessState): boolean;
	execute(context: WebGLPostProcessPassContext): void;
}

export class WebGLPostProcessGraph {
	private _passes = new Map<string, WebGLPostProcessPassPlugin>();

	constructor(passes: readonly WebGLPostProcessPassPlugin[] = []) {
		for (const pass of passes) {
			this.registerPass(pass);
		}
	}

	public registerPass(pass: WebGLPostProcessPassPlugin): void {
		if (!pass.id) {
			throw new Error("WebGL post-process pass id is required.");
		}
		this._passes.set(pass.id, pass);
	}

	public unregisterPass(id: string): void {
		this._passes.delete(id);
	}

	public hasPass(id: string): boolean {
		return this._passes.has(id);
	}

	public getPass(id: string): WebGLPostProcessPassPlugin | null {
		return this._passes.get(id) ?? null;
	}

	public getExecutionOrder(
		postProcess: ResolvedPostProcessState,
		warn: (key: string, message: string) => void
	): WebGLPostProcessPassPlugin[] {
		const enabled = new Map<string, WebGLPostProcessPassPlugin>();
		for (const [id, pass] of this._passes.entries()) {
			if (pass.isEnabled(postProcess)) {
				enabled.set(id, pass);
			}
		}

		const order: WebGLPostProcessPassPlugin[] = [];
		const state = new Map<string, number>();
		const invalid = new Set<string>();

		const visit = (id: string): boolean => {
			if (invalid.has(id)) return false;
			const current = state.get(id) ?? 0;
			if (current === 2) return true;
			if (current === 1) {
				warn(
					`webgl-post-cycle-${id}`,
					`WebGL post-process dependency cycle detected at pass "${id}", skipping cycle branch`
				);
				invalid.add(id);
				return false;
			}

			const pass = enabled.get(id);
			if (!pass) return false;
			state.set(id, 1);

			for (const dependencyId of pass.dependsOn) {
				if (!this._passes.has(dependencyId)) {
					warn(
						`webgl-post-dependency-missing-${id}-${dependencyId}`,
						`WebGL post-process pass "${id}" depends on unknown pass "${dependencyId}"; skipping "${id}"`
					);
					invalid.add(id);
					state.set(id, 2);
					return false;
				}
				if (!enabled.has(dependencyId)) {
					continue;
				}
				if (!visit(dependencyId)) {
					invalid.add(id);
					state.set(id, 2);
					return false;
				}
			}

			state.set(id, 2);
			order.push(pass);
			return true;
		};

		for (const id of enabled.keys()) {
			visit(id);
		}

		return order;
	}

	public execute(
		context: WebGLPostProcessPassContext,
		postProcess: ResolvedPostProcessState,
		warn: (key: string, message: string) => void
	): string[] {
		const order = this.getExecutionOrder(postProcess, warn);
		for (const pass of order) {
			pass.execute(context);
		}
		return order.map((pass) => pass.id);
	}
}
