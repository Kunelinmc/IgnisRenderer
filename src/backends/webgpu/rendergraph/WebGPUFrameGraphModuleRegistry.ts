import type { FrameContext, FramePass } from "../../../pipeline/types";
import type { FramePreparationRequirements } from "../../../pipeline/FrameRequirements";

import {
	WebGPUFrameNodeExecutorRegistry,
} from "./WebGPUFrameNodeExecutorRegistry";
import type {
	WebGPUCommittingFrameSession,
	WebGPURecordingFrameSession,
} from "./WebGPUFrameSession";
import type { WebGPUFrameExecutionContext } from "./WebGPUFrameExecutionContext";
import type { WebGPUFrameGraphCompiler } from "./WebGPUFrameGraphCompiler";
import type { WebGPUFrameTargetView } from "./WebGPUFrameTargetManager";
import type {
	WebGPUFrameMessageHandler,
	WebGPUFrameMessagePhase,
	WebGPUFrameMessageSnapshot,
} from "./WebGPUFrameMessage";
import { WebGPUFrameMessageRegistry } from "./WebGPUFrameMessageRegistry";
import {
	WEBGPU_FRAME_GRAPH_FRAGMENT_MESSAGE,
	WEBGPU_FRAME_FINAL_OUTPUT_MESSAGE,
	WEBGPU_FRAME_PLANNING_REQUEST_MESSAGE,
	type WebGPUFrameFinalOutputMessage,
} from "./WebGPUFrameMessages";
import { WEBGPU_FRAME_GRAPH_RESOURCES } from "./WebGPUFrameGraphResourceCatalog";
import {
	WEBGPU_FRAME_STAGE_LANES,
	type WebGPUFrameGraphContribution,
	type WebGPUFrameGraphModule,
	type WebGPUFrameModulePlanningInput,
} from "./WebGPUFrameGraphModule";
import type {
	WebGPUCompiledFrameGraphStage,
	WebGPUFrameGraphStagePlan,
} from "./types";

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
	private readonly _messages = new WebGPUFrameMessageRegistry();
	private _sealed = false;
	private _finalOutput: WebGPUFrameFinalOutputMessage = {
		resource: WEBGPU_FRAME_GRAPH_RESOURCES.frameColor,
		colorDomain: "scene-linear-hdr",
	};

	public register(module: WebGPUFrameGraphModule): void {
		if (this._sealed) {
			throw new Error("WebGPU frame module registry is sealed.");
		}
		if (this._ids.has(module.id)) {
			throw new Error(`WebGPU frame module "${module.id}" is already registered.`);
		}
		this._ids.add(module.id);
		this._modules.push(module);
		for (const handler of module.messageHandlers ?? []) {
			this._messages.register(handler);
		}
		if (module.planStage) {
			this._messages.register({
				id: "plan-stage",
				moduleId: module.id,
				phase: "planning",
				inputs: [
					{ descriptor: WEBGPU_FRAME_PLANNING_REQUEST_MESSAGE },
					...(module.planningInputs ?? []),
				],
				outputs: [
					WEBGPU_FRAME_GRAPH_FRAGMENT_MESSAGE,
					WEBGPU_FRAME_FINAL_OUTPUT_MESSAGE,
				],
				run: (messages, publisher) => {
					const request = messages.get(WEBGPU_FRAME_PLANNING_REQUEST_MESSAGE);
					for (const contribution of module.planStage!({ ...request, messages })) {
						publisher.publish(WEBGPU_FRAME_GRAPH_FRAGMENT_MESSAGE, {
							moduleId: module.id,
							contribution: this._ownContribution(module.id, contribution),
						});
						if (contribution.finalOutput) {
							publisher.publish(
								WEBGPU_FRAME_FINAL_OUTPUT_MESSAGE,
								contribution.finalOutput,
							);
						}
					}
				},
			});
		}
	}

	public registerMessageHandler(handler: WebGPUFrameMessageHandler): void {
		if (this._sealed) {
			throw new Error("WebGPU frame module registry is sealed.");
		}
		this._messages.register(handler);
	}

	public seal(): void {
		if (this._sealed) return;
		this._executors = WebGPUFrameNodeExecutorRegistry.fromModules(this._modules);
		this._messages.seal();
		Object.freeze(this._modules);
		this._sealed = true;
	}

	public dispatchMessages(
		phase: WebGPUFrameMessagePhase,
		options?: Parameters<WebGPUFrameMessageRegistry["dispatch"]>[1],
	): Promise<WebGPUFrameMessageSnapshot> {
		this._assertSealed();
		return this._messages.dispatch(phase, options);
	}

	public get modules(): readonly WebGPUFrameGraphModule[] {
		return this._modules;
	}

	public async planStage(
		input: Omit<WebGPUFrameModulePlanningInput, "messages">,
		base: WebGPUFrameMessageSnapshot,
	): Promise<WebGPUFrameGraphStagePlan> {
		this._assertSealed();
		const request = input.finalization === true && !input.finalColorResource
			? { ...input, finalColorResource: this._finalOutput.resource }
			: input;
		const snapshot = await this._messages.dispatch("planning", {
			prior: base,
			seeds: [{ descriptor: WEBGPU_FRAME_PLANNING_REQUEST_MESSAGE, value: request }],
		});
		const contributions = snapshot.getAll(WEBGPU_FRAME_GRAPH_FRAGMENT_MESSAGE);
		const finalOutputs = snapshot.getAll(WEBGPU_FRAME_FINAL_OUTPUT_MESSAGE);
		if (finalOutputs.length > 1) {
			throw new Error(
				`WebGPU stage "${input.pass.stage}" published multiple final outputs.`,
			);
		}
		if (finalOutputs[0]) this._finalOutput = finalOutputs[0];
		const selected = this._selectExclusiveContribution(input.pass, contributions);
		this._validateContributions(input.pass, selected);
		const ordered = this._orderContributions(input.pass, selected);
		return {
			pass: input.pass,
			nodes: ordered.flatMap(({ contribution }) => contribution.nodes ?? []),
			imports: ordered.flatMap(({ contribution }) => contribution.imports ?? []),
			composition: ordered.find(({ contribution }) => contribution.composition)
				?.contribution.composition,
		};
	}

	private _ownContribution(
		moduleId: string,
		contribution: WebGPUFrameGraphContribution,
	): WebGPUFrameGraphContribution {
		return {
			...contribution,
			nodes: contribution.nodes?.map((node) => ({
				...node,
				id: `${node.stage}:${moduleId}:${node.localId ?? node.kind}`,
				ownerId: moduleId,
			})),
		};
	}

	public async execute(
		node: Parameters<WebGPUFrameNodeExecutorRegistry["execute"]>[0],
		context: WebGPUFrameExecutionContext,
	): Promise<void> {
		this._assertSealed();
		await this._executors!.execute(node, context);
	}

	public beginFrame(context: FrameContext): void {
		this._assertSealed();
		this._finalOutput = {
			resource: WEBGPU_FRAME_GRAPH_RESOURCES.frameColor,
			colorDomain: "scene-linear-hdr",
		};
		for (const module of this._modules) module.beginFrame?.(context);
	}

	public syncFrame(context: FrameContext): void {
		for (const module of this._modules) module.syncFrame?.(context);
	}

	public createAnalysisSeeds(context: FrameContext) {
		return this._modules.flatMap(
			(module) => module.createAnalysisSeeds?.(context) ?? [],
		);
	}

	public sealFrame(
		context: FrameContext,
		targets: WebGPUFrameTargetView,
	): FramePreparationRequirements {
		const requirements = this._modules
			.map((module) => module.sealFrame?.(context, targets) ?? null)
			.filter((value): value is FramePreparationRequirements => value !== null);
		if (requirements.length > 1) {
			throw new Error("Multiple WebGPU frame modules published preparation requirements.");
		}
		return requirements[0] ?? {};
	}

	public async executeComposedStage(
		stage: WebGPUCompiledFrameGraphStage | undefined,
		compiler: Pick<WebGPUFrameGraphCompiler, "recordSkippedNode">,
		recordExecutedNode: (nodeId: string) => void,
	): Promise<boolean> {
		for (const module of this._modules) {
			if (await module.executeComposedStage?.(stage, compiler, recordExecutedNode)) {
				return true;
			}
		}
		return false;
	}

	public activateFrame(context: WebGPUFrameExecutionContext): void {
		for (const module of this._modules) module.activateFrame?.(context);
	}

	public closeFrame(): void {
		for (const module of this._modules) module.closeFrame?.();
	}

	public get finalOutput() {
		return this._finalOutput;
	}

	public async finalizeRecording(session: WebGPURecordingFrameSession): Promise<void> {
		for (const module of this._modules) await module.finalizeRecording?.(session);
	}

	public async afterSubmit(session: WebGPUCommittingFrameSession): Promise<void> {
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

	private _validateContributions(
		pass: FramePass,
		entries: readonly {
			readonly moduleId: string;
			readonly contribution: WebGPUFrameGraphContribution;
		}[],
	): void {
		const nodeIds = new Map<string, string>();
		let compositionOwner: string | null = null;
		for (const { moduleId, contribution } of entries) {
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

	private _selectExclusiveContribution(
		pass: FramePass,
		entries: readonly ContributionEntry[],
	): readonly ContributionEntry[] {
		const exclusive = entries.filter(({ contribution }) => contribution.exclusive === true);
		if (exclusive.length > 1) {
			throw new Error(
				`WebGPU stage "${pass.stage}" has multiple exclusive graph contributors: ` +
				exclusive.map(({ moduleId }) => `"${moduleId}"`).join(", ") + ".",
			);
		}
		return exclusive.length === 1 ? exclusive : entries;
	}

	private _orderContributions(
		pass: FramePass,
		entries: readonly ContributionEntry[],
	): readonly ContributionEntry[] {
		const laneRank = new Map(WEBGPU_FRAME_STAGE_LANES.map((lane, index) => [lane, index]));
		const result: ContributionEntry[] = [];
		for (const lane of WEBGPU_FRAME_STAGE_LANES) {
			const laneEntries = entries.filter(({ contribution }) => contribution.lane === lane);
			if (laneEntries.length <= 1) {
				result.push(...laneEntries);
				continue;
			}
			const byId = new Map(laneEntries.map((entry) => [entry.moduleId, entry]));
			const outgoing = new Map(laneEntries.map(({ moduleId }) => [moduleId, new Set<string>()]));
			const indegree = new Map(laneEntries.map(({ moduleId }) => [moduleId, 0]));
			for (const entry of laneEntries) {
				for (const target of entry.contribution.before ?? []) {
					if (byId.has(target)) outgoing.get(entry.moduleId)!.add(target);
				}
				for (const source of entry.contribution.after ?? []) {
					if (byId.has(source)) outgoing.get(source)!.add(entry.moduleId);
				}
			}
			for (const targets of outgoing.values()) {
				for (const target of targets) indegree.set(target, indegree.get(target)! + 1);
			}
			for (let i = 0; i < laneEntries.length; i++) {
				for (let j = i + 1; j < laneEntries.length; j++) {
					const left = laneEntries[i].moduleId;
					const right = laneEntries[j].moduleId;
					if (!hasPath(left, right, outgoing) && !hasPath(right, left, outgoing)) {
						throw new Error(
							`WebGPU stage "${pass.stage}" lane "${lane}" contributors ` +
							`"${left}" and "${right}" require a static before/after edge.`,
						);
					}
				}
			}
			const ready = laneEntries
				.filter(({ moduleId }) => indegree.get(moduleId) === 0)
				.sort((a, b) => a.moduleId.localeCompare(b.moduleId));
			while (ready.length > 0) {
				const entry = ready.shift()!;
				result.push(entry);
				for (const target of outgoing.get(entry.moduleId)!) {
					indegree.set(target, indegree.get(target)! - 1);
					if (indegree.get(target) === 0) {
						ready.push(byId.get(target)!);
						ready.sort((a, b) => a.moduleId.localeCompare(b.moduleId));
					}
				}
			}
			if (result.filter(({ contribution }) => contribution.lane === lane).length !== laneEntries.length) {
				throw new Error(`WebGPU stage "${pass.stage}" lane "${lane}" contains a cycle.`);
			}
		}
		for (const entry of entries) {
			if (!laneRank.has(entry.contribution.lane)) {
				throw new Error(`Unknown WebGPU frame stage lane "${entry.contribution.lane}".`);
			}
		}
		return result;
	}
}

interface ContributionEntry {
	readonly moduleId: string;
	readonly contribution: WebGPUFrameGraphContribution;
}

function hasPath(
	from: string,
	to: string,
	outgoing: ReadonlyMap<string, ReadonlySet<string>>,
): boolean {
	const pending = [from];
	const visited = new Set<string>();
	while (pending.length > 0) {
		const current = pending.pop()!;
		if (!visited.add(current)) continue;
		for (const target of outgoing.get(current) ?? []) {
			if (target === to) return true;
			pending.push(target);
		}
	}
	return false;
}
