import type { ShadowMap } from "../../lights/ShadowMapping";
import type { WebGPUBackend } from "../WebGPUBackend";
import { TextureFormat, TextureUsage, type IRenderTexture } from "../types";
import type { WebGPULightingState } from "./";

interface ShadowAtlas {
	tileSize: number;
	texture: IRenderTexture;
}

interface ShadowSlice {
	enabled: boolean;
	shadowMap: ShadowMap | null;
	atlasTileSize: number;
}

const SHADOW_ATLAS_COLUMNS = 4;
const SHADOW_ATLAS_ROWS = 2;

export class WebGPUShadowAtlasAllocator {
	private _backend: WebGPUBackend;
	private _atlas: ShadowAtlas | null = null;

	constructor(backend: WebGPUBackend) {
		this._backend = backend;
	}

	public prepare(lightingState: WebGPULightingState): void {
		const directionalShadows = lightingState.directionalShadows;
		const spotShadows = lightingState.spotShadows;
		const maxShadowSize = Math.max(
			getMaxShadowSize(directionalShadows),
			getMaxShadowSize(spotShadows)
		);

		for (const shadow of directionalShadows) {
			shadow.atlasTileSize = maxShadowSize;
		}
		for (const shadow of spotShadows) {
			shadow.atlasTileSize = maxShadowSize;
		}

		this.ensureAtlasForTileSize(Math.max(1, maxShadowSize));
	}

	public ensureAtlasForTileSize(tileSize: number): IRenderTexture {
		const safeTileSize = Math.max(1, tileSize | 0);

		if (!this._atlas || this._atlas.tileSize !== safeTileSize) {
			this._atlas?.texture.destroy();

			this._atlas = {
				tileSize: safeTileSize,
				texture: this._backend.createTexture({
					width: safeTileSize * SHADOW_ATLAS_COLUMNS,
					height: safeTileSize * SHADOW_ATLAS_ROWS,
					format: TextureFormat.Depth32Float,
					usage: TextureUsage.RenderAttachment | TextureUsage.TextureBinding,
					label: "WebGPUShadowDepthAtlas",
				}),
			};
		}

		return this._atlas.texture;
	}

	public get atlas(): IRenderTexture | null {
		return this._atlas?.texture ?? null;
	}

	public get tileSize(): number {
		return this._atlas?.tileSize ?? 0;
	}

	public get columns(): number {
		return SHADOW_ATLAS_COLUMNS;
	}

	public get rows(): number {
		return SHADOW_ATLAS_ROWS;
	}

	public get directionalAtlas(): IRenderTexture | null {
		return this.atlas;
	}

	public get spotAtlas(): IRenderTexture | null {
		return this.atlas;
	}

	public destroy(): void {
		this._atlas?.texture.destroy();
		this._atlas = null;
	}
}

function getMaxShadowSize(shadows: ShadowSlice[]): number {
	let tileSize = 0;
	for (const shadow of shadows) {
		if (!shadow?.enabled || !shadow.shadowMap) continue;
		tileSize = Math.max(tileSize, shadow.shadowMap.size | 0);
	}

	return tileSize;
}
