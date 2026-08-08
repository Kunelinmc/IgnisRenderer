import type { ShadowCastingLight } from "..";
import { CascadedShadowMap } from "./CascadedShadowMap";
import { PagedShadowMap } from "./PagedShadowMap";
import { ShadowMapBase } from "./ShadowMapBase";
import { SingleShadowMap } from "./SingleShadowMap";
import { VarianceShadowMap, type VarianceShadowMapOptions } from "./VarianceShadowMap";
import type {
	CascadedShadowMapOptions,
	PagedShadowMapOptions,
	ShadowMapBaseOptions,
} from "./types";

/** @internal Scene owns this invalidation bridge. */
export interface ShadowManagerOptions {
	onChange?: () => void;
}

/** Owns built-in shadow definitions and persistent light bindings. */
export class ShadowManager {
	private readonly _mapsById = new Map<string, ShadowMapBase>();
	private readonly _shadowByLight = new Map<ShadowCastingLight, ShadowMapBase>();
	private readonly _lightsByShadowId = new Map<string, Set<ShadowCastingLight>>();
	private readonly _definitionUnsubscribers = new Map<string, () => void>();
	private readonly _onChange?: () => void;
	private _version = 0;

	constructor(options: ShadowManagerOptions = {}) {
		this._onChange = options.onChange;
	}

	public createSingle(options: ShadowMapBaseOptions = {}): SingleShadowMap {
		return this._track(new SingleShadowMap(options));
	}

	public createVariance(options: VarianceShadowMapOptions = {}): VarianceShadowMap {
		return this._track(new VarianceShadowMap(options));
	}

	public createCascaded(options: CascadedShadowMapOptions = {}): CascadedShadowMap {
		return this._track(new CascadedShadowMap(options));
	}

	/** Creates an unbound built-in paged shadow definition. */
	public createPaged(options: PagedShadowMapOptions = {}): PagedShadowMap {
		return this._track(new PagedShadowMap(options));
	}

	public bind(light: ShadowCastingLight, shadowMap: ShadowMapBase): void {
		const existing = this._shadowByLight.get(light);
		if (existing?.id === shadowMap.id) return;
		if (existing) this._detachBinding(light, existing);
		this._mapsById.set(shadowMap.id, shadowMap);
		this._observe(shadowMap);
		this._shadowByLight.set(light, shadowMap);
		let lights = this._lightsByShadowId.get(shadowMap.id);
		if (!lights) {
			lights = new Set();
			this._lightsByShadowId.set(shadowMap.id, lights);
		}
		lights.add(light);
		this._markChanged();
	}

	public rebind(light: ShadowCastingLight, shadowMap: ShadowMapBase): void {
		this.bind(light, shadowMap);
	}

	public unbindLight(light: ShadowCastingLight): void {
		const existing = this._shadowByLight.get(light);
		if (!existing) return;
		this._detachBinding(light, existing);
		this._markChanged();
	}

	public destroy(shadowMap: ShadowMapBase): void {
		const lights = this._lightsByShadowId.get(shadowMap.id);
		for (const light of lights ?? []) this._shadowByLight.delete(light);
		this._lightsByShadowId.delete(shadowMap.id);
		this._mapsById.delete(shadowMap.id);
		this._definitionUnsubscribers.get(shadowMap.id)?.();
		this._definitionUnsubscribers.delete(shadowMap.id);
		this._markChanged();
	}

	public clear(): void {
		for (const unsubscribe of this._definitionUnsubscribers.values()) unsubscribe();
		this._definitionUnsubscribers.clear();
		this._mapsById.clear();
		this._shadowByLight.clear();
		this._lightsByShadowId.clear();
		this._markChanged();
	}

	public get version(): number {
		return this._version;
	}

	public getBoundShadowMap(light: ShadowCastingLight): ShadowMapBase | undefined {
		return this._shadowByLight.get(light);
	}

	private _track<TShadowMap extends ShadowMapBase>(shadowMap: TShadowMap): TShadowMap {
		this._mapsById.set(shadowMap.id, shadowMap);
		this._observe(shadowMap);
		this._markChanged();
		return shadowMap;
	}

	private _observe(shadowMap: ShadowMapBase): void {
		if (this._definitionUnsubscribers.has(shadowMap.id)) return;
		this._definitionUnsubscribers.set(
			shadowMap.id,
			shadowMap.subscribe(() => this._markChanged())
		);
	}

	private _detachBinding(light: ShadowCastingLight, shadowMap: ShadowMapBase): void {
		this._shadowByLight.delete(light);
		const lights = this._lightsByShadowId.get(shadowMap.id);
		if (!lights) return;
		lights.delete(light);
		if (lights.size === 0) this._lightsByShadowId.delete(shadowMap.id);
	}

	private _markChanged(): void {
		this._version++;
		this._onChange?.();
	}
}
