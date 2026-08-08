import { IdGenerator } from "../../foundation/IdGenerator";
import { LightType } from "..";
import type {
	CascadedShadowConfig,
	ShadowConfig,
	ShadowParams,
	SingleMapShadowConfig,
} from "./ShadowMapping";
import type {
	ShadowBiasSettings,
	ShadowBoundLightType,
	ShadowDefinitionListener,
	ShadowDefinitionSnapshot,
	ShadowFilterMode,
	ShadowMapKind,
	ShadowMapBaseOptions,
	ShadowProjectionSnapshot,
	ShadowSamplingSettings,
	ShadowStoragePreference,
} from "./types";

const DEFAULT_SHADOW_SIZE = 1024;

export abstract class ShadowMapBase {
	public readonly id: string;
	private _enabled: boolean;
	private _priority: number;
	private _size: number;
	private readonly _bias: ShadowBiasSettings;
	private readonly _sampling: ShadowSamplingSettings;
	private readonly _listeners = new Set<ShadowDefinitionListener>();
	private _revision = 0;
	private _batchDepth = 0;
	private _batchChanged = false;

	public abstract readonly kind: ShadowMapKind;

	protected constructor(options: ShadowMapBaseOptions = {}) {
		this.id = options.id ?? IdGenerator.nextId("shadow");
		this._enabled = options.enabled !== false;
		this._priority = toFiniteNumber(options.priority, 0);
		this._size = Math.max(
			1,
			Math.floor(toFiniteNumber(options.size, DEFAULT_SHADOW_SIZE))
		);
		this._bias = createObservableSettings({
			constant: toFiniteNumber(options.bias?.constant, 0),
			slope: toFiniteNumber(options.bias?.slope, 0.01),
			normal: toFiniteNumber(options.bias?.normal, 0.01),
			normalMin: toFiniteNumber(options.bias?.normalMin, 0.01),
			texel: toFiniteNumber(options.bias?.texel, 1.0),
			max: toFiniteNumber(options.bias?.max, 0.1),
		}, BIAS_NORMALIZERS, () => this._markDefinitionChanged());
		this._sampling = createObservableSettings({
			filterMode: options.sampling?.filterMode ?? "pcf",
			pcfRadius: toFiniteNumber(options.sampling?.pcfRadius, 1),
			strength: toFiniteNumber(options.sampling?.strength, 1),
			radius: toFiniteNumber(options.sampling?.radius, 0),
			samples: Math.max(1, Math.floor(toFiniteNumber(options.sampling?.samples, 16))),
			searchSamples: Math.max(
				1,
				Math.floor(toFiniteNumber(options.sampling?.searchSamples, 16))
			),
		}, SAMPLING_NORMALIZERS, () => this._markDefinitionChanged());
	}

	public get revision(): number {
		return this._revision;
	}

	public get enabled(): boolean {
		return this._enabled;
	}

	public set enabled(value: boolean) {
		const normalized = value !== false;
		if (this._enabled === normalized) return;
		this._enabled = normalized;
		this._markDefinitionChanged();
	}

	public get priority(): number {
		return this._priority;
	}

	public set priority(value: number) {
		const normalized = toFiniteNumber(value, 0);
		if (Object.is(this._priority, normalized)) return;
		this._priority = normalized;
		this._markDefinitionChanged();
	}

	public get size(): number {
		return this._size;
	}

	public set size(value: number) {
		const normalized = Math.max(
			1,
			Math.floor(toFiniteNumber(value, DEFAULT_SHADOW_SIZE))
		);
		if (this._size === normalized) return;
		this._size = normalized;
		this._markDefinitionChanged();
	}

	public get bias(): ShadowBiasSettings {
		return this._bias;
	}

	public set bias(value: ShadowBiasSettings) {
		assignObservableSettings(this._bias, value);
	}

	public get sampling(): ShadowSamplingSettings {
		return this._sampling;
	}

	public set sampling(value: ShadowSamplingSettings) {
		assignObservableSettings(this._sampling, value);
	}

	/**
	 * Applies multiple definition changes as one observable revision.
	 *
	 * @param options Settings to normalize and merge into this definition.
	 * @returns This definition for chained configuration.
	 * @sideEffects At most one definition notification is emitted.
	 */
	public update(options: Partial<ShadowMapBaseOptions>): this {
		return this._runDefinitionUpdate(() => {
			if (options.enabled !== undefined) this.enabled = options.enabled;
			if (options.priority !== undefined) this.priority = options.priority;
			if (options.size !== undefined) this.size = options.size;
			if (options.bias !== undefined) this.bias = options.bias;
			if (options.sampling !== undefined) this.sampling = options.sampling;
		});
	}

	/** @internal Subscribes a shadow manager to definition changes. */
	public subscribe(listener: ShadowDefinitionListener): () => void {
		this._listeners.add(listener);
		return () => this._listeners.delete(listener);
	}

	/** @internal Captures immutable input for `ShadowPlanner`. */
	public snapshot(): ShadowDefinitionSnapshot {
		return Object.freeze({
			id: this.id,
			kind: this.kind,
			enabled: this.enabled,
			projection: Object.freeze(this.createProjectionSnapshot()),
			storagePreference: this.storagePreference,
			resolution: this.size,
			bias: Object.freeze({ ...this.bias }) as Readonly<
				Required<ShadowBiasSettings>
			>,
			sampling: Object.freeze({ ...this.sampling }) as Readonly<
				Required<ShadowSamplingSettings>
			>,
			priority: this.priority,
			revision: this.revision,
		});
	}

	public get filterMode(): ShadowFilterMode {
		return this.sampling.filterMode ?? "pcf";
	}

	protected get storagePreference(): ShadowStoragePreference {
		return "atlas";
	}

	protected createProjectionSnapshot(): ShadowProjectionSnapshot {
		return { technique: "single" };
	}

	protected _runDefinitionUpdate(apply: () => void): this {
		this._batchDepth++;
		try {
			apply();
		} finally {
			this._batchDepth--;
			if (this._batchDepth === 0 && this._batchChanged) {
				this._batchChanged = false;
				this._emitDefinitionChanged();
			}
		}
		return this;
	}

	protected _markDefinitionChanged(): void {
		if (this._batchDepth > 0) {
			this._batchChanged = true;
			return;
		}
		this._emitDefinitionChanged();
	}

	protected _createObservableSettings<T extends object>(
		initial: T,
		normalizers: SettingNormalizers<T>
	): T {
		return createObservableSettings(
			initial,
			normalizers,
			() => this._markDefinitionChanged()
		);
	}

	protected _assignObservableSettings<T extends object>(
		target: T,
		source: Partial<T>
	): void {
		assignObservableSettings(target, source);
	}

	private _emitDefinitionChanged(): void {
		this._revision++;
		for (const listener of this._listeners) listener(this);
	}

	public resolveBoundLightType(lightType: LightType): ShadowBoundLightType {
		switch (lightType) {
			case LightType.Directional:
				return "directional";
			case LightType.Point:
				return "point";
			case LightType.Spot:
				return "spot";
			default:
				return "rectArea";
		}
	}

	public estimateCost(
		lightType: LightType,
		size: number = this.size,
		cascadeCount: number = 1
	): number {
		const normalizedSize = Math.max(1, size) / 1024;
		const perSliceCost = normalizedSize * normalizedSize;
		const boundType = this.resolveBoundLightType(lightType);
		const sliceMultiplier =
			boundType === "point" ?
				Math.max(1, cascadeCount) * 6
			:	Math.max(1, cascadeCount);
		return perSliceCost * sliceMultiplier;
	}

	/**
	 * @internal Shadow scheduling hook used by `ShadowManager`.
	 *
	 * Resolves the number of logical shadow cascades this map requests for the
	 * provided light type. Custom shadow maps should override this when their
	 * `toLegacyShadowConfig` output depends on multiple cascades.
	 */
	public resolveCascadeCount(_lightType: LightType): number {
		return 1;
	}

	public abstract toLegacyShadowConfig(
		lightType: LightType,
		overrides?: {
			size?: number;
			cascadeCount?: number;
		}
	): ShadowConfig;

	protected resolveShadowParams(): ShadowParams {
		return {
			shadowBias: this.bias.constant,
			shadowSlopeBias: this.bias.slope,
			shadowNormalBias: this.bias.normal,
			shadowNormalBiasMin: this.bias.normalMin,
			shadowTexelBias: this.bias.texel,
			shadowMaxBias: this.bias.max,
			shadowPCF: this.sampling.pcfRadius,
			shadowStrength: this.sampling.strength,
			shadowRadius: this.sampling.radius,
			shadowSamples: this.sampling.samples,
			shadowSearchSamples: this.sampling.searchSamples,
		};
	}

	protected createSingleMapLegacyConfig(sizeOverride?: number): SingleMapShadowConfig {
		const size = Math.max(1, Math.floor(sizeOverride ?? this.size));
		return {
			strategy: "single-map",
			size,
			priority: this.priority,
			params: this.resolveShadowParams(),
		};
	}

	protected createCSMLegacyConfig(
		cascadeCount: number,
		options: {
			size?: number;
			lambda?: number;
			maxDistance?: number;
			blendRatio?: number;
			stabilize?: boolean;
		}
	): CascadedShadowConfig {
		const size = Math.max(1, Math.floor(options.size ?? this.size));
		return {
			strategy: "csm",
			size,
			priority: this.priority,
			params: this.resolveShadowParams(),
			cascadeCount: clampCascadeCount(cascadeCount),
			splitMode: "practical",
			lambda: toFiniteNumber(options.lambda, 0.65),
			maxDistance: options.maxDistance,
			blendRatio: toFiniteNumber(options.blendRatio, 0.1),
			stabilize: options.stabilize !== false,
		};
	}
}

type SettingNormalizers<T extends object> = {
	[K in keyof T]-?: (value: unknown, previous: T[K]) => T[K];
};

function createObservableSettings<T extends object>(
	initial: T,
	normalizers: SettingNormalizers<T>,
	onChange: () => void
): T {
	const values = { ...initial };
	const target = {} as T;
	for (const key of Object.keys(initial) as Array<keyof T>) {
		Object.defineProperty(target, key, {
			enumerable: true,
			get: () => values[key],
			set: (value: unknown) => {
				const normalized = normalizers[key](value, values[key]);
				if (Object.is(values[key], normalized)) return;
				values[key] = normalized;
				onChange();
			},
		});
	}
	return target;
}

function assignObservableSettings<T extends object>(target: T, source: Partial<T>): void {
	for (const key of Object.keys(target) as Array<keyof T>) {
		if (source[key] !== undefined) target[key] = source[key] as T[keyof T];
	}
}

const BIAS_NORMALIZERS: SettingNormalizers<Required<ShadowBiasSettings>> = {
	constant: (value) => toFiniteNumber(value, 0),
	slope: (value) => toFiniteNumber(value, 0.01),
	normal: (value) => toFiniteNumber(value, 0.01),
	normalMin: (value) => toFiniteNumber(value, 0.01),
	texel: (value) => toFiniteNumber(value, 1),
	max: (value) => toFiniteNumber(value, 0.1),
};

const SAMPLING_NORMALIZERS: SettingNormalizers<Required<ShadowSamplingSettings>> = {
	filterMode: (value) => value === "vsm" ? "vsm" : "pcf",
	pcfRadius: (value) => toFiniteNumber(value, 1),
	strength: (value) => toFiniteNumber(value, 1),
	radius: (value) => toFiniteNumber(value, 0),
	samples: (value) => Math.max(1, Math.floor(toFiniteNumber(value, 16))),
	searchSamples: (value) => Math.max(1, Math.floor(toFiniteNumber(value, 16))),
};

function toFiniteNumber(value: unknown, fallback: number): number {
	if (typeof value !== "number" || !Number.isFinite(value)) {
		return fallback;
	}
	return value;
}

function clampCascadeCount(value: number): 1 | 2 | 3 | 4 {
	if (value <= 1) return 1;
	if (value <= 2) return 2;
	if (value >= 4) return 4;
	return 3;
}
