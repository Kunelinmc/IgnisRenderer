import type { FrameContext } from "../../pipeline/types";
import type { ResolvedPostProcessState } from "../../pipeline/PostProcessController";
import type { ICommandEncoder } from "../ICommandEncoder";
import type { IRenderTexture } from "../types";
import type { WebGPUBackend } from "../WebGPUBackend";
import {
	WEBGPU_POST_PROCESS_PASS_IDS,
	type WebGPUPostProcessExecuteResult,
	type WebGPUPostProcessRuntimeExecuteRequest,
	type WebGPUPostProcessRuntimePass,
} from "./postprocess/types";

export type WebGPUPostProcessPassKind = "compute" | "render";

export const WEBGPU_BUILTIN_POST_PROCESS_PASS_IDS = [
	...WEBGPU_POST_PROCESS_PASS_IDS,
	"gamma",
] as const;

export type WebGPUBuiltinPostProcessPassId =
	(typeof WEBGPU_BUILTIN_POST_PROCESS_PASS_IDS)[number];

const WEBGPU_BUILTIN_POST_PROCESS_PASS_ID_SET = new Set<string>(
	WEBGPU_BUILTIN_POST_PROCESS_PASS_IDS
);

/**
 * Returns whether `id` is reserved by a built-in WebGPU post-process pass.
 *
 * @param id Candidate graph or runtime pass id.
 * @returns `true` when the id cannot be used by custom post-process passes.
 */
export function isWebGPUBuiltinPostProcessPassId(id: string): boolean {
	return WEBGPU_BUILTIN_POST_PROCESS_PASS_ID_SET.has(id);
}

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
	oitAccum: IRenderTexture;
	oitReveal: IRenderTexture;
	oitSceneColorCopy: IRenderTexture;
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
	postProcess: ResolvedPostProcessState;
	targets: WebGPUFrameTargets;
	executeRuntimePass(
		request: WebGPUPostProcessRuntimeExecuteRequest
	): Promise<WebGPUPostProcessExecuteResult>;
}

export interface WebGPUPostProcessPassPlugin {
	id: string;
	dependsOn: string[];
	kind?: WebGPUPostProcessPassKind;
	precompileHints?: string[];
	runtime?: WebGPUPostProcessRuntimePass;
	isEnabled(postProcess: ResolvedPostProcessState): boolean;
	execute(context: WebGPUPostProcessPassContext): Promise<void> | void;
}

export class WebGPUPostProcessGraph {
	private _passes = new Map<string, WebGPUPostProcessPassPlugin>();

	constructor(passes: WebGPUPostProcessPassPlugin[] = []) {
		for (const pass of passes) {
			this._registerPass(pass, true);
		}
	}

	/**
	 * Throws when a custom pass cannot be registered in this graph.
	 *
	 * @param pass Pass descriptor to validate.
	 * @throws If the id is empty, reserved by a built-in pass, or duplicated.
	 */
	public assertCanRegisterPass(pass: WebGPUPostProcessPassPlugin): void {
		this._assertPassCanRegister(pass, false);
	}

	public registerPass(pass: WebGPUPostProcessPassPlugin): void {
		this._registerPass(pass, false);
	}

	public unregisterPass(id: string): void {
		if (isWebGPUBuiltinPostProcessPassId(id)) {
			throw new Error(
				`Cannot unregister built-in WebGPU post-process pass "${id}".`
			);
		}
		this._passes.delete(id);
	}

	public hasPass(id: string): boolean {
		return this._passes.has(id);
	}

	public getPass(id: string): WebGPUPostProcessPassPlugin | null {
		return this._passes.get(id) ?? null;
	}

	public getExecutionOrder(
		postProcess: ResolvedPostProcessState,
		warn: (key: string, message: string) => void
	): WebGPUPostProcessPassPlugin[] {
		const enabled = new Map<string, WebGPUPostProcessPassPlugin>();
		for (const [id, pass] of this._passes.entries()) {
			if (pass.isEnabled(postProcess)) {
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
		postProcess: ResolvedPostProcessState,
		warn: (key: string, message: string) => void
	): Promise<string[]> {
		const order = this.getExecutionOrder(postProcess, warn);
		for (const pass of order) {
			await pass.execute(context);
		}
		return order.map((pass) => pass.id);
	}

	private _registerPass(
		pass: WebGPUPostProcessPassPlugin,
		allowBuiltIn: boolean
	): void {
		this._assertPassCanRegister(pass, allowBuiltIn);
		this._passes.set(pass.id, pass);
	}

	private _assertPassCanRegister(
		pass: WebGPUPostProcessPassPlugin,
		allowBuiltIn: boolean
	): void {
		if (!pass.id) {
			throw new Error("Post-process pass id is required.");
		}
		if (!allowBuiltIn && isWebGPUBuiltinPostProcessPassId(pass.id)) {
			throw new Error(
				`Cannot register built-in WebGPU post-process pass "${pass.id}".`
			);
		}
		if (this._passes.has(pass.id)) {
			throw new Error(
				`WebGPU post-process pass "${pass.id}" is already registered.`
			);
		}
		if (!pass.runtime) {
			return;
		}
		if (!pass.runtime.id) {
			throw new Error("WebGPU post-process runtime pass id is required.");
		}
		if (!allowBuiltIn && isWebGPUBuiltinPostProcessPassId(pass.runtime.id)) {
			throw new Error(
				`Cannot register built-in WebGPU post-process runtime pass "${pass.runtime.id}".`
			);
		}
		for (const registered of this._passes.values()) {
			if (registered.runtime?.id === pass.runtime.id) {
				throw new Error(
					`WebGPU post-process runtime pass "${pass.runtime.id}" is already registered.`
				);
			}
		}
	}
}
