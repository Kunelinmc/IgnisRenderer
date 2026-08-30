import type { Material } from "../../materials/Material";
import { ShaderMaterial } from "../../materials/ShaderMaterial";
import {
	resolveWebGLMaterialState,
	type WebGLResolvedMaterialState,
} from "./WebGLMaterialState";
import {
	resolveWebGLSceneMaterialVariant,
	type WebGLSceneVariantDescriptor,
} from "./WebGLSceneProgramVariants";

export interface WebGLResolvedMaterialSnapshot {
	readonly revision: number;
	readonly data: WebGLResolvedMaterialState;
	readonly materialVariant: WebGLSceneVariantDescriptor["material"];
}

interface CachedWebGLMaterialSnapshot extends WebGLResolvedMaterialSnapshot {}

/** @internal Resolves built-in WebGL material state once per render revision. */
export class WebGLMaterialSnapshotCache {
	private _entries = new WeakMap<Material, CachedWebGLMaterialSnapshot>();
	private _refreshedMaterials = new WeakSet<Material>();

	public beginFrame(): void {
		this._refreshedMaterials = new WeakSet();
	}

	public resolve(material: Material): WebGLResolvedMaterialSnapshot {
		if (material instanceof ShaderMaterial) {
			throw new Error("ShaderMaterial does not use built-in WebGL material snapshots.");
		}
		if (
			typeof material.refreshRevision !== "function" ||
			typeof material._getRevisionInternal !== "function"
		) {
			const data = resolveWebGLMaterialState(material);
			return {
				revision: 0,
				data,
				materialVariant: resolveWebGLSceneMaterialVariant(material, data),
			};
		}
		if (!this._refreshedMaterials.has(material)) {
			material.refreshRevision();
			this._refreshedMaterials.add(material);
		}
		const revision = material._getRevisionInternal();
		const cached = this._entries.get(material);
		if (cached?.revision === revision) return cached;
		const data = resolveWebGLMaterialState(material);
		const snapshot = {
			revision,
			data,
			materialVariant: resolveWebGLSceneMaterialVariant(material, data),
		} satisfies CachedWebGLMaterialSnapshot;
		this._entries.set(material, snapshot);
		return snapshot;
	}

	public clear(): void {
		this._entries = new WeakMap();
		this._refreshedMaterials = new WeakSet();
	}
}
