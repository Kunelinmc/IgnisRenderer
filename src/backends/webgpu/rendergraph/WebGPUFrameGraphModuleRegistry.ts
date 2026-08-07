import type { FrameContext, FramePass } from "../../../pipeline/types";

import {
	WebGPUFrameNodeExecutorRegistry,
} from "./WebGPUFrameNodeExecutorRegistry";
import type { WebGPUFrameSession } from "./WebGPUFrameSession";
import {
	WebGPUFrameModuleStateStore,
	type WebGPUFrameGraphContribution,
	type WebGPUFrameGraphModule,
	type WebGPUFrameModuleAnalysisInput,
	type WebGPUFrameModuleConfigurationInput,
	type WebGPUFrameModulePlanningInput,
} from "./WebGPUFrameGraphModule";
import type {
	WebGPUFrameModuleConfigurationContribution,
} from "./WebGPUFrameConfigurationContribution";
import type { WebGPUFrameGraphStagePlan } from "./types";

/**
 * Initialization-time registry for backend-private WebGPU frame modules.
 *
 * @internal Owned by the WebGPU runtime composition root. Applications must
 * not register frame graph modules.
 */
export class WebGPUFrameGraphModuleRegistry {
	private readonly _modules: WebGPUFrameGraphModule[] = [];
	private readonly _ids = new Set<string>();
	private _executors: WebGPUFrameNodeExecutorRegistry | null = null;
	private _sealed = false;

	public register(module: WebGPUFrameGraphModule): void {
		if (this._sealed) {
			throw new Error("WebGPU frame module registry is sealed.");
		}
		if (this._ids.has(module.id)) {
			throw new Error(`WebGPU frame module "${module.id}" is already registered.`);
		}
		this._ids.add(module.id);
		this._modules.push(module);
	}

	public seal(): void {
		if (this._sealed) return;
		this._executors = WebGPUFrameNodeExecutorRegistry.fromModules(this._modules);
		Object.freeze(this._modules);
		this._sealed = true;
	}

	public get modules(): readonly WebGPUFrameGraphModule[] {
		return this._modules;
	}

	public analyze(input: WebGPUFrameModuleAnalysisInput): WebGPUFrameModuleStateStore {
		this._assertSealed();
		const state = new WebGPUFrameModuleStateStore();
		for (const module of this._modules) {
			this._runAnalysisHook(state, module, () => module.analyze?.(input, state));
		}
		state.seal();
		return state;
	}

	public collectConfigurationContributions(
		input: WebGPUFrameModuleConfigurationInput,
	): readonly WebGPUFrameModuleConfigurationContribution[] {
		this._assertSealed();
		return this._modules.flatMap((module) => {
			const contribution = module.contributeConfiguration?.(input);
			return contribution ? [contribution] : [];
		});
	}

	public planStage(input: WebGPUFrameModulePlanningInput): WebGPUFrameGraphStagePlan {
		this._assertSealed();
		const modules = input.exclusiveModuleId
			? this._modules.filter((module) => module.id === input.exclusiveModuleId)
			: this._modules;
		const contributions = modules.flatMap((module) =>
			(module.planStage?.(input) ?? []).map((contribution) => ({
				moduleId: module.id,
				contribution,
			})),
		);
		if (input.exclusiveModuleId && modules.length !== 1) {
			throw new Error(
				`WebGPU frame module "${input.exclusiveModuleId}" is unavailable.`,
			);
		}
		this._validateContributions(input.pass, contributions);
		contributions.sort((a, b) => a.contribution.order - b.contribution.order);
		return {
			pass: input.pass,
			nodes: contributions.flatMap(({ contribution }) => contribution.nodes ?? []),
			composition: contributions.find(({ contribution }) => contribution.composition)
				?.contribution.composition,
		};
	}

	public async execute(
		node: Parameters<WebGPUFrameNodeExecutorRegistry["execute"]>[0],
		session: WebGPUFrameSession,
	): Promise<void> {
		this._assertSealed();
		await this._executors!.execute(node, session);
	}

	public beginFrame(context: FrameContext): void {
		this._assertSealed();
		for (const module of this._modules) module.beginFrame?.(context);
	}

	public async finalizeRecording(session: WebGPUFrameSession): Promise<void> {
		for (const module of this._modules) await module.finalizeRecording?.(session);
	}

	public async afterSubmit(session: WebGPUFrameSession): Promise<void> {
		for (const module of this._modules) await module.afterSubmit?.(session);
	}

	public commitFrameState(): void {
		for (const module of this._modules) module.commitFrameState?.();
	}

	public abortFrameState(error?: unknown): void {
		for (const module of this._modules) module.abortFrameState?.(error);
	}

	public invalidateFrameResources(): void {
		for (const module of this._modules) module.invalidateFrameResources?.();
	}

	public onDisplayOutputChanged(): void {
		for (const module of this._modules) module.onDisplayOutputChanged?.();
	}

	public onShaderRuntimeChanged(): void {
		for (const module of this._modules) module.onShaderRuntimeChanged?.();
	}

	public destroy(): void {
		for (const module of this._modules) module.destroy();
	}

	private _assertSealed(): void {
		if (!this._sealed || !this._executors) {
			throw new Error("WebGPU frame module registry is not sealed.");
		}
	}

	private _runAnalysisHook(
		state: WebGPUFrameModuleStateStore,
		module: WebGPUFrameGraphModule,
		hook: () => void,
	): void {
		state.beginAnalysis(module.id);
		try {
			hook();
		} finally {
			state.endAnalysis(module.id);
		}
	}

	private _validateContributions(
		pass: FramePass,
		entries: readonly {
			readonly moduleId: string;
			readonly contribution: WebGPUFrameGraphContribution;
		}[],
	): void {
		const orders = new Map<number, string>();
		const nodeIds = new Map<string, string>();
		let compositionOwner: string | null = null;
		for (const { moduleId, contribution } of entries) {
			const priorOrder = orders.get(contribution.order);
			if (priorOrder) {
				throw new Error(
					`WebGPU stage "${pass.stage}" order ${contribution.order} is owned by ` +
					`both "${priorOrder}" and "${moduleId}".`,
				);
			}
			orders.set(contribution.order, moduleId);
			if (contribution.composition) {
				if (compositionOwner) {
					throw new Error(
						`WebGPU stage "${pass.stage}" has composed subgraphs from ` +
						`both "${compositionOwner}" and "${moduleId}".`,
					);
				}
				compositionOwner = moduleId;
			}
			for (const node of contribution.nodes ?? []) {
				const priorNodeOwner = nodeIds.get(node.id);
				if (priorNodeOwner) {
					throw new Error(
						`WebGPU frame graph node "${node.id}" is contributed by ` +
						`both "${priorNodeOwner}" and "${moduleId}".`,
					);
				}
				nodeIds.set(node.id, moduleId);
			}
		}
	}
}
