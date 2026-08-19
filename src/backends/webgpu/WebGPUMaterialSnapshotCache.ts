import type { Material } from "../../materials/Material";
import { ShaderMaterial } from "../../materials/ShaderMaterial";
import { createWebGPUMaterialUniformData } from "./material";
import type { WebGPUTextureRegistry } from "./WebGPUTextureRegistry";
import type {
	IRenderTexture,
	ISampler,
} from "../types";
import type { WebGPUMaterialUniformData } from "./types";

/** @internal Fully resolved material inputs reusable by compatible draws. */
export interface WebGPUResolvedMaterialSnapshot {
	readonly revision: number;
	readonly data: WebGPUMaterialUniformData;
	readonly textures: IRenderTexture[];
	readonly samplers: ISampler[];
	readonly anisotropyTexture: IRenderTexture;
}

interface CachedMaterialSnapshot {
	revision: number;
	promise: Promise<WebGPUResolvedMaterialSnapshot>;
}

/** @internal Resolves material and texture state once per render revision. */
export class WebGPUMaterialSnapshotCache {
	private _entries = new WeakMap<
		Material,
		[CachedMaterialSnapshot | null, CachedMaterialSnapshot | null]
	>();
	private _refreshedMaterials = new WeakSet<Material>();
	private _frameHits = 0;
	private _frameResolves = 0;

	constructor(private readonly _textures: WebGPUTextureRegistry) {}

	public beginFrame(): void {
		this._refreshedMaterials = new WeakSet();
		this._frameHits = 0;
		this._frameResolves = 0;
	}

	public resolve(
		material: Material,
		wireframe: boolean,
	): Promise<WebGPUResolvedMaterialSnapshot> {
		if (material instanceof ShaderMaterial || !this._refreshedMaterials.has(material)) {
			material.refreshRevision();
			this._refreshedMaterials.add(material);
		}
		const revision = material._getRevisionInternal();
		let variants = this._entries.get(material);
		if (!variants) {
			variants = [null, null];
			this._entries.set(material, variants);
		}
		const variantIndex = wireframe ? 1 : 0;
		const cached = variants[variantIndex];
		if (cached?.revision === revision) {
			this._frameHits++;
			return cached.promise;
		}
		this._frameResolves++;

		const data = createWebGPUMaterialUniformData(material, wireframe);
		const promise = this._resolveResources(revision, data);
		const entry: CachedMaterialSnapshot = { revision, promise };
		variants[variantIndex] = entry;
		void promise.catch(() => {
			const current = this._entries.get(material);
			if (current?.[variantIndex] === entry) current[variantIndex] = null;
		});
		return promise;
	}

	public clear(): void {
		this._entries = new WeakMap();
		this._refreshedMaterials = new WeakSet();
	}

	public getDebugStats(): {
		readonly frameHits: number;
		readonly frameResolves: number;
	} {
		return {
			frameHits: this._frameHits,
			frameResolves: this._frameResolves,
		};
	}

	private async _resolveResources(
		revision: number,
		data: WebGPUMaterialUniformData,
	): Promise<WebGPUResolvedMaterialSnapshot> {
		const textures = await Promise.all(
			data.textureSlots.map((slot, index) =>
				this._textures.getTextureForSlotAsync(slot.map, index),
			),
		);
		const samplers = data.textureSlots.map((slot) =>
			this._textures.getSamplerForTexture(slot.map),
		);
		const anisotropyTexture = await this._textures.getTextureForSlotAsync(
			data.anisotropyTexture.map,
			-1,
		);
		return { revision, data, textures, samplers, anisotropyTexture };
	}
}
