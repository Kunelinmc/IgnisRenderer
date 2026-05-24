import type { ShaderCompileError } from "../../shaders/runtime";
import { toShaderCompileError } from "../../pipeline/WarmupPlanner";
import type { IWebGPUComputeFacade } from "./ComputeFacade";
import { PostProcessSharedContext } from "./postprocess/PostProcessSharedContext";
import { SpatialPostProcessDelegate } from "./postprocess/SpatialPostProcessDelegate";
import { TemporalPostProcessDelegate } from "./postprocess/TemporalPostProcessDelegate";
import { isWebGPUBuiltinPostProcessPassId } from "./WebGPUPostProcessContracts";
import type {
	WebGPUPostProcessExecuteResult,
	WebGPUPostProcessRuntimeExecuteRequest,
	WebGPUPostProcessRuntimePass,
	WebGPUPostProcessRuntimePassRegistry,
} from "./postprocess/types";

export type {
	WebGPUCustomPostProcessExecuteRequest,
	WebGPUPostProcessExecuteRequest,
	WebGPUPostProcessExecuteResult,
	WebGPUPostProcessPassId,
	WebGPUPostProcessRuntimeContext,
	WebGPUPostProcessRuntimeExecuteRequest,
	WebGPUPostProcessRuntimePass,
} from "./postprocess/types";

interface RegisteredRuntimePass {
	pass: WebGPUPostProcessRuntimePass;
	builtIn: boolean;
}

interface BuiltInPostProcessDelegate {
	registerPasses(registry: WebGPUPostProcessRuntimePassRegistry): void;
	destroy(): void;
}

export class WebGPUPostProcessRuntime {
	private _shared: PostProcessSharedContext;
	private _runtimePassById = new Map<string, RegisteredRuntimePass>();
	private _warmupPassesByHint = new Map<string, RegisteredRuntimePass[]>();
	private _builtInDelegates: BuiltInPostProcessDelegate[];

	constructor(
		computeFacade: IWebGPUComputeFacade,
		warn: (key: string, message: string) => void,
		frameBindGroupLayout?: GPUBindGroupLayout
	) {
		this._shared = new PostProcessSharedContext(
			computeFacade,
			warn,
			frameBindGroupLayout
		);
		this._builtInDelegates = [
			new SpatialPostProcessDelegate(this._shared),
			new TemporalPostProcessDelegate(this._shared),
		];
		const builtInRegistry: WebGPUPostProcessRuntimePassRegistry = {
			registerRuntimePass: (pass) =>
				this._registerRuntimePass(pass, { builtIn: true }),
		};
		for (const delegate of this._builtInDelegates) {
			delegate.registerPasses(builtInRegistry);
		}
	}

	public get sharedContext(): PostProcessSharedContext {
		return this._shared;
	}

	/**
	 * Throws when a custom runtime pass cannot be registered.
	 *
	 * @param pass Runtime pass descriptor to validate.
	 * @throws If the id is empty, reserved by a built-in pass, or duplicated.
	 */
	public assertCanRegisterRuntimePass(
		pass: WebGPUPostProcessRuntimePass
	): void {
		this._assertRuntimePassCanRegister(pass, false);
	}

	/**
	 * Registers a custom runtime pass used by post-process graph plugins.
	 *
	 * @param pass Runtime pass descriptor. The `id` must be a unique custom id.
	 * @throws If the pass id is empty, reserved, or already registered.
	 */
	public registerRuntimePass(pass: WebGPUPostProcessRuntimePass): void {
		this._registerRuntimePass(pass, { builtIn: false });
	}

	/**
	 * Unregisters a custom runtime pass by id.
	 *
	 * @param id Runtime pass id to remove. Unknown custom ids are ignored.
	 * @throws If `id` belongs to a built-in runtime pass.
	 */
	public unregisterRuntimePass(id: string): void {
		const entry = this._runtimePassById.get(id);
		if (entry?.builtIn || isWebGPUBuiltinPostProcessPassId(id)) {
			throw new Error(
				`Cannot unregister built-in WebGPU post-process runtime pass "${id}".`
			);
		}
		if (!entry) {
			return;
		}
		this._runtimePassById.delete(id);
		this._removeWarmupPass(entry);
	}

	/**
	 * Invalidate all cached bind groups. Call when frame targets are
	 * destroyed/rebuilt (e.g. on resize) so stale texture references are
	 * not reused.
	 */
	public invalidateBindings(): void {
		this._shared.invalidateBindings();
		for (const entry of this._runtimePassById.values()) {
			entry.pass.invalidateBindings?.(this._shared);
		}
	}

	public onShaderRuntimeChanged(): void {
		this._shared.onShaderRuntimeChanged();
		for (const entry of this._runtimePassById.values()) {
			entry.pass.onShaderRuntimeChanged?.(this._shared);
		}
	}

	public destroy(): void {
		this._shared.destroy();
		for (const delegate of this._builtInDelegates) {
			delegate.destroy();
		}
		this._runtimePassById.clear();
		this._warmupPassesByHint.clear();
	}

	public async warmupHints(hints: readonly string[]): Promise<{
		compiled: number;
		failed: number;
		errors: ShaderCompileError[];
	}> {
		let compiled = 0;
		let failed = 0;
		const errors: ShaderCompileError[] = [];
		const seen = new Set<string>();
		for (const hint of hints) {
			if (seen.has(hint)) {
				continue;
			}
			seen.add(hint);
			try {
				const warmed = await this._warmupHint(hint);
				if (warmed) {
					compiled++;
				}
			} catch (error) {
				failed++;
				errors.push(
					toShaderCompileError(error, "webgpu", `WebGPUPostWarmup:${hint}`)
				);
			}
		}
		return {
			compiled,
			failed,
			errors,
		};
	}

	public async executePass(
		request: WebGPUPostProcessRuntimeExecuteRequest
	): Promise<WebGPUPostProcessExecuteResult> {
		const entry = this._runtimePassById.get(request.passId);
		if (!entry) {
			return { ran: false };
		}
		const result = await entry.pass.execute(request, this._shared);
		if (result && typeof result === "object") {
			return result;
		}
		return { ran: true };
	}

	private async _warmupHint(hint: string): Promise<boolean> {
		const entries = this._warmupPassesByHint.get(hint);
		if (!entries) {
			return false;
		}
		let warmed = false;
		for (const entry of entries) {
			if (!entry.pass.warmup) {
				continue;
			}
			const result = await entry.pass.warmup(hint, this._shared);
			if (result !== false) {
				warmed = true;
			}
		}
		return warmed;
	}

	private _registerRuntimePass(
		pass: WebGPUPostProcessRuntimePass,
		options: { builtIn: boolean }
	): void {
		this._assertRuntimePassCanRegister(pass, options.builtIn);
		const entry: RegisteredRuntimePass = {
			pass,
			builtIn: options.builtIn,
		};
		this._runtimePassById.set(pass.id, entry);
		for (const hint of pass.warmupHints ?? []) {
			const entries = this._warmupPassesByHint.get(hint) ?? [];
			entries.push(entry);
			this._warmupPassesByHint.set(hint, entries);
		}
	}

	private _assertRuntimePassCanRegister(
		pass: WebGPUPostProcessRuntimePass,
		allowBuiltIn: boolean
	): void {
		if (!pass.id) {
			throw new Error("WebGPU post-process runtime pass id is required.");
		}
		if (!allowBuiltIn && isWebGPUBuiltinPostProcessPassId(pass.id)) {
			throw new Error(
				`Cannot register built-in WebGPU post-process runtime pass "${pass.id}".`
			);
		}
		if (this._runtimePassById.has(pass.id)) {
			throw new Error(
				`WebGPU post-process runtime pass "${pass.id}" is already registered.`
			);
		}
	}

	private _removeWarmupPass(entry: RegisteredRuntimePass): void {
		for (const [hint, entries] of this._warmupPassesByHint.entries()) {
			const filtered = entries.filter((candidate) => candidate !== entry);
			if (filtered.length === 0) {
				this._warmupPassesByHint.delete(hint);
			} else if (filtered.length !== entries.length) {
				this._warmupPassesByHint.set(hint, filtered);
			}
		}
	}
}
