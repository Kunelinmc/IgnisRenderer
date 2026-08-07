import type { FrameContext, FramePass } from "../../../pipeline/types";
import type { PreparedFramePacketSet } from "../../../pipeline/FramePacketContributorRegistry";
import type { PlannedPostProcessPass } from "../../../postprocess";

import type {
	WebGPUFrameModuleConfigurationContribution,
} from "./WebGPUFrameConfigurationContribution";
import type { WebGPUFrameNodeExecutor } from "./WebGPUFrameNodeExecutorRegistry";
import type { WebGPUFrameSession } from "./WebGPUFrameSession";
import type {
	WebGPUComposedFrameGraphStage,
	WebGPUFrameGraphNode,
	WebGPUFrameGraphNodeKind,
	WebGPUFrameGraphPlannerState,
	WebGPUFrameGraphResourceId,
} from "./types";

/** @internal Typed key owned by one backend-private frame module. */
export interface WebGPUFrameModuleStateKey<TValue> {
	readonly id: string;
	readonly ownerId: string;
	readonly __valueType?: TValue;
}

/** @internal Creates a typed state key for one backend-private frame module. */
export function defineWebGPUFrameModuleStateKey<TValue>(
	ownerId: string,
	id: string,
): WebGPUFrameModuleStateKey<TValue> {
	return Object.freeze({ id, ownerId });
}

/** Type-safe frame-local state shared after every module finishes analysis. */
export class WebGPUFrameModuleStateStore {
	private readonly _values = new Map<
		WebGPUFrameModuleStateKey<unknown>,
		unknown
	>();
	private readonly _keysById = new Map<
		string,
		WebGPUFrameModuleStateKey<unknown>
	>();
	private _analysisOwnerId: string | null = null;
	private _sealed = false;

	public set<TValue>(key: WebGPUFrameModuleStateKey<TValue>, value: TValue): void {
		if (this._sealed) {
			throw new Error("WebGPU frame module state is sealed.");
		}
		if (!this._analysisOwnerId) {
			throw new Error("WebGPU frame module state may only be written during analysis.");
		}
		if (key.ownerId !== this._analysisOwnerId) {
			throw new Error(
				`WebGPU frame module "${this._analysisOwnerId}" cannot write state ` +
				`owned by "${key.ownerId}".`,
			);
		}
		const existingKey = this._keysById.get(key.id);
		if (existingKey && existingKey !== key) {
			throw new Error(
				`WebGPU frame module state key "${key.id}" has conflicting identities.`,
			);
		}
		if (this._values.has(key)) {
			throw new Error(`WebGPU frame module state "${key.id}" is already defined.`);
		}
		this._keysById.set(key.id, key);
		this._values.set(key, value);
	}

	public get<TValue>(key: WebGPUFrameModuleStateKey<TValue>): TValue | undefined {
		if (this._analysisOwnerId && key.ownerId !== this._analysisOwnerId) {
			throw new Error(
				`WebGPU frame module "${this._analysisOwnerId}" cannot read state ` +
				`owned by "${key.ownerId}" during analysis.`,
			);
		}
		const existingKey = this._keysById.get(key.id);
		if (existingKey && existingKey !== key) {
			throw new Error(
				`WebGPU frame module state key "${key.id}" has conflicting identities.`,
			);
		}
		return this._values.get(key) as TValue | undefined;
	}

	public require<TValue>(key: WebGPUFrameModuleStateKey<TValue>): TValue {
		const value = this.get(key);
		if (value === undefined) {
			throw new Error(`WebGPU frame module state "${key.id}" is unavailable.`);
		}
		return value;
	}

	/** @internal Owned by `WebGPUFrameGraphModuleRegistry`. */
	public beginAnalysis(ownerId: string): void {
		if (this._sealed || this._analysisOwnerId) {
			throw new Error("WebGPU frame module state analysis scope is unavailable.");
		}
		this._analysisOwnerId = ownerId;
	}

	/** @internal Owned by `WebGPUFrameGraphModuleRegistry`. */
	public endAnalysis(ownerId: string): void {
		if (this._analysisOwnerId !== ownerId) {
			throw new Error(`WebGPU frame module state owner mismatch for "${ownerId}".`);
		}
		this._analysisOwnerId = null;
	}

	/** @internal Owned by `WebGPUFrameGraphModuleRegistry`. */
	public seal(): void {
		if (this._analysisOwnerId) {
			throw new Error("WebGPU frame module state cannot seal during analysis.");
		}
		this._sealed = true;
	}
}

/** @internal Input shared with frame modules during the analysis phase. */
export interface WebGPUFrameModuleAnalysisInput {
	readonly context: FrameContext;
	readonly framePackets: PreparedFramePacketSet;
	readonly postProcessPasses: readonly PlannedPostProcessPass[];
}

/** @internal Configuration input supplied after module analysis is sealed. */
export interface WebGPUFrameModuleConfigurationInput {
	readonly context: FrameContext;
	readonly state: WebGPUFrameModuleStateStore;
}

/** @internal Input shared with frame modules after analysis is sealed. */
export interface WebGPUFrameModulePlanningInput {
	readonly pass: FramePass;
	readonly context: FrameContext;
	readonly state: WebGPUFrameGraphPlannerState;
	readonly moduleState: WebGPUFrameModuleStateStore;
	readonly finalization?: boolean;
	readonly finalColorResource?: WebGPUFrameGraphResourceId;
	readonly exclusiveModuleId?: string;
}

/** @internal Ordered graph contribution from one backend-private module. */
export interface WebGPUFrameGraphContribution {
	readonly order: number;
	readonly nodes?: readonly WebGPUFrameGraphNode[];
	readonly composition?: WebGPUComposedFrameGraphStage;
}

/**
 * Backend-private unit of WebGPU frame analysis, planning, execution, and
 * feature-local lifecycle.
 *
 * @internal Registered by the WebGPU runtime composition root. Applications
 * must use `Renderer.renderFrame()` instead.
 */
export interface WebGPUFrameGraphModule {
	readonly id: string;
	readonly executors: Readonly<
		Partial<Record<WebGPUFrameGraphNodeKind, WebGPUFrameNodeExecutor>>
	>;
	analyze?(
		input: WebGPUFrameModuleAnalysisInput,
		state: WebGPUFrameModuleStateStore,
	): void;
	contributeConfiguration?(
		input: WebGPUFrameModuleConfigurationInput,
	): WebGPUFrameModuleConfigurationContribution;
	planStage?(
		input: WebGPUFrameModulePlanningInput,
	): readonly WebGPUFrameGraphContribution[];
	beginFrame?(context: FrameContext): void;
	finalizeRecording?(session: WebGPUFrameSession): void | Promise<void>;
	afterSubmit?(session: WebGPUFrameSession): void | Promise<void>;
	commitFrameState?(): void;
	abortFrameState?(error?: unknown): void;
	invalidateFrameResources?(): void;
	onDisplayOutputChanged?(): void;
	onShaderRuntimeChanged?(): void;
	destroy(): void;
}
