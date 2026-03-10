import type { FrameContext, ResolvedFeatureState } from "../../pipeline/types";
import type { ICommandEncoder } from "../ICommandEncoder";
import type { IRenderTexture } from "../types";
import type { WebGPUBackend } from "../WebGPUBackend";

export type WebGPUPostProcessPassKind = "compute" | "render";

export interface WebGPUFrameTargets {
	sceneColor: IRenderTexture;
	sceneColorMain: IRenderTexture;
	postPing: IRenderTexture;
	postPong: IRenderTexture;
	gAlbedoAlpha: IRenderTexture;
	gNormalRoughMetal: IRenderTexture;
	gEmissiveOcclusion: IRenderTexture;
	gMotionDepth: IRenderTexture;
	depth: IRenderTexture;
	aoRaw: IRenderTexture;
	aoBlur: IRenderTexture;
	ssrRaw: IRenderTexture;
	hiZ: IRenderTexture;
	historyRead: IRenderTexture;
	historyWrite: IRenderTexture;
	ssrHistoryRead: IRenderTexture;
	ssrHistoryWrite: IRenderTexture;
	volumetricHistoryRead: IRenderTexture;
	volumetricHistoryWrite: IRenderTexture;
	volumetricReservoirHistoryRead: IRenderTexture;
	volumetricReservoirHistoryWrite: IRenderTexture;
	motionHistoryRead: IRenderTexture;
	motionHistoryWrite: IRenderTexture;
}

export interface WebGPUPostProcessPassContext {
	backend: WebGPUBackend;
	encoder: ICommandEncoder;
	frameContext: FrameContext;
	targets: WebGPUFrameTargets;
}

export interface WebGPUPostProcessPassPlugin {
	id: string;
	dependsOn: string[];
	kind?: WebGPUPostProcessPassKind;
	isEnabled(features: ResolvedFeatureState): boolean;
	execute(context: WebGPUPostProcessPassContext): Promise<void> | void;
}

export class WebGPUPostProcessGraph {
	private _passes = new Map<string, WebGPUPostProcessPassPlugin>();

	constructor(passes: WebGPUPostProcessPassPlugin[] = []) {
		for (const pass of passes) {
			this.registerPass(pass);
		}
	}

	public registerPass(pass: WebGPUPostProcessPassPlugin): void {
		if (!pass.id) {
			throw new Error("Post-process pass id is required.");
		}
		this._passes.set(pass.id, pass);
	}

	public unregisterPass(id: string): void {
		this._passes.delete(id);
	}

	public hasPass(id: string): boolean {
		return this._passes.has(id);
	}

	public getPass(id: string): WebGPUPostProcessPassPlugin | null {
		return this._passes.get(id) ?? null;
	}

	public getExecutionOrder(
		features: ResolvedFeatureState,
		warn: (key: string, message: string) => void
	): WebGPUPostProcessPassPlugin[] {
		const enabled = new Map<string, WebGPUPostProcessPassPlugin>();
		for (const [id, pass] of this._passes.entries()) {
			if (pass.isEnabled(features)) {
				enabled.set(id, pass);
			}
		}

		const order: WebGPUPostProcessPassPlugin[] = [];
		const state = new Map<string, number>();
		const invalid = new Set<string>();

		const visit = (id: string): boolean => {
			if (invalid.has(id)) return false;
			const current = state.get(id) ?? 0;
			if (current === 2) return true;
			if (current === 1) {
				warn(
					`webgpu-post-cycle-${id}`,
					`WebGPU post-process dependency cycle detected at pass "${id}", skipping cycle branch`
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
						`webgpu-post-dependency-missing-${id}-${dependencyId}`,
						`WebGPU post-process pass "${id}" depends on unknown pass "${dependencyId}"; skipping "${id}"`
					);
					invalid.add(id);
					state.set(id, 2);
					return false;
				}

				if (!enabled.has(dependencyId)) {
					// Enabled subset auto-shrink: ignore disabled known dependency.
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

	public async execute(
		context: WebGPUPostProcessPassContext,
		features: ResolvedFeatureState,
		warn: (key: string, message: string) => void
	): Promise<string[]> {
		const order = this.getExecutionOrder(features, warn);
		for (const pass of order) {
			await pass.execute(context);
		}
		return order.map((pass) => pass.id);
	}
}
