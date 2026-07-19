import type { FrameContext, PreparedScene } from "../../pipeline/types";
import type {
	WebGPUFeatureState,
	WebGPULightingCatalog,
	WebGPULightingState,
} from "./types";

export interface WebGPUFrameFeatureKey<TValue> {
	readonly id: string;
	readonly __valueType?: TValue;
}

export function defineWebGPUFrameFeatureKey<TValue>(
	id: string
): WebGPUFrameFeatureKey<TValue> {
	return { id };
}

export class WebGPUFrameFeatureDataStore {
	private readonly _values = new Map<string, unknown>();

	public set<TValue>(
		key: WebGPUFrameFeatureKey<TValue>,
		value: TValue
	): void {
		this._values.set(key.id, value);
	}

	public get<TValue>(key: WebGPUFrameFeatureKey<TValue>): TValue | undefined {
		return this._values.get(key.id) as TValue | undefined;
	}

	public has<TValue>(key: WebGPUFrameFeatureKey<TValue>): boolean {
		return this._values.has(key.id);
	}
}

export interface WebGPUFrameFeatureContext {
	readonly frameContext: FrameContext;
	readonly scene: PreparedScene;
	readonly featureState: WebGPUFeatureState;
	readonly lightingCatalog: WebGPULightingCatalog;
	readonly lightingState: WebGPULightingState;
	readonly dataStore: WebGPUFrameFeatureDataStore;
	readonly renderWidth: number;
	readonly renderHeight: number;
}

export interface WebGPUFrameFeatureModule<TValue> {
	readonly id: string;
	readonly key: WebGPUFrameFeatureKey<TValue>;
	isEnabled(context: WebGPUFrameFeatureContext): boolean;
	prepare(context: WebGPUFrameFeatureContext): TValue;
	destroy?(): void;
}

export class WebGPUFrameFeatureRegistry {
	private readonly _modules: WebGPUFrameFeatureModule<unknown>[];

	public constructor(modules: readonly WebGPUFrameFeatureModule<unknown>[]) {
		this._modules = modules.slice();
	}

	public prepareFrame(
		context: Omit<WebGPUFrameFeatureContext, "dataStore">
	): WebGPUFrameFeatureDataStore {
		const store = new WebGPUFrameFeatureDataStore();
		const moduleContext: WebGPUFrameFeatureContext = {
			...context,
			dataStore: store,
		};
		for (const module of this._modules) {
			if (!module.isEnabled(moduleContext)) {
				continue;
			}
			store.set(module.key, module.prepare(moduleContext));
		}
		return store;
	}

	public destroy(): void {
		for (const module of this._modules) {
			module.destroy?.();
		}
	}
}
