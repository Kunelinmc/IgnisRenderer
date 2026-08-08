import type { ShadowCastingLight } from "..";
import type { ShadowRenderSet } from "./ShadowMapping";
import type { ShadowBindingRecord } from "./types";

/**
 * @deprecated Compatibility snapshot for custom shadow definitions. Renderer
 * frames use `ShadowFramePlan`; this type will be removed with the legacy
 * custom-map surface in the next breaking release.
 */
export class ShadowFrameState {
	public readonly version: number;
	private readonly _records: ShadowBindingRecord[];
	private readonly _shadowMaps: Map<ShadowCastingLight, ShadowRenderSet>;

	constructor(
		version: number,
		records: ShadowBindingRecord[],
		shadowMaps: Map<ShadowCastingLight, ShadowRenderSet>
	) {
		this.version = version;
		this._records = records;
		this._shadowMaps = shadowMaps;
	}

	public get records(): ShadowBindingRecord[] {
		return this._records;
	}

	public get shadowMaps(): Map<ShadowCastingLight, ShadowRenderSet> {
		return this._shadowMaps;
	}

	public get(light: ShadowCastingLight): ShadowRenderSet | undefined {
		return this._shadowMaps.get(light);
	}

	public has(light: ShadowCastingLight): boolean {
		return this._shadowMaps.has(light);
	}

	public entries(): IterableIterator<[ShadowCastingLight, ShadowRenderSet]> {
		return this._shadowMaps.entries();
	}

	public values(): IterableIterator<ShadowRenderSet> {
		return this._shadowMaps.values();
	}

	public keys(): IterableIterator<ShadowCastingLight> {
		return this._shadowMaps.keys();
	}

	public getLights(): ShadowCastingLight[] {
		return this._records.map((record) => record.light);
	}
}
