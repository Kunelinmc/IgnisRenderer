import type { WebGPUDeviceResourceHost } from "./WebGPUDeviceResourceHost";
import { TextureFormat } from "../../core/TextureFormat";
import { TextureUsage, type IRenderTexture } from "../types";
import {
	WEBGPU_SHADOW_ATLAS_COLUMNS,
	WEBGPU_SHADOW_ATLAS_ROWS,
} from "./constants";
import type { WebGPULightingState } from "./";

interface ShadowAtlas {
	tileSize: number;
	texture: IRenderTexture;
	transmittanceTexture: IRenderTexture;
}

interface ShadowSlice {
	enabled: boolean;
	shadowMapSize: number;
	shadowMapBaseSize: number;
	atlasTileSize: number;
}

export class WebGPUShadowAtlasAllocator {
	private _backend: WebGPUDeviceResourceHost;
	private _atlas: ShadowAtlas | null = null;

	constructor(backend: WebGPUDeviceResourceHost) {
		this._backend = backend;
	}

	public prepare(
		lightingState: WebGPULightingState,
		minTileSize: number = 0
	): void {
		const directionalShadows = lightingState.directionalShadows;
		const spotShadows = lightingState.spotShadows;
		const maxShadowSize = Math.max(
			getMaxShadowSize(directionalShadows),
			getMaxShadowSize(spotShadows)
		);
		const requestedTileSize = Math.max(maxShadowSize, minTileSize | 0);
		const resolvedTileSize = this._resolveAtlasTileSize(requestedTileSize);

		for (const shadow of directionalShadows) {
			shadow.atlasTileSize = resolvedTileSize;
		}
		for (const shadow of spotShadows) {
			shadow.atlasTileSize = resolvedTileSize;
		}

		this.ensureAtlasForTileSize(Math.max(1, resolvedTileSize));
	}

	/**
	 * Ensures that a shadow atlas texture at least as large as the requested tile
	 * size exists.
	 *
	 * @param tileSize The requested tile size.
	 * @returns The shadow depth atlas texture.
	 * @internal WebGPU shadow rendering and frame binding infrastructure only.
	 */
	public ensureAtlasForTileSize(tileSize: number): IRenderTexture {
		const requestedTileSize = Math.max(1, tileSize | 0);
		const safeTileSize = this._resolveAtlasTileSize(requestedTileSize);

		if (!this._atlas || this._atlas.tileSize < safeTileSize) {
			this._atlas?.texture.destroy();
			this._atlas?.transmittanceTexture.destroy();

			this._atlas = {
				tileSize: safeTileSize,
				texture: this._backend.createTexture({
					width: safeTileSize * WEBGPU_SHADOW_ATLAS_COLUMNS,
					height: safeTileSize * WEBGPU_SHADOW_ATLAS_ROWS,
					format: TextureFormat.Depth32Float,
					usage: TextureUsage.RenderAttachment | TextureUsage.TextureBinding,
					label: "WebGPUShadowDepthAtlas",
				}),
				transmittanceTexture: this._backend.createTexture({
					width: safeTileSize * WEBGPU_SHADOW_ATLAS_COLUMNS,
					height: safeTileSize * WEBGPU_SHADOW_ATLAS_ROWS,
					format: TextureFormat.RGBA16Float,
					usage: TextureUsage.RenderAttachment | TextureUsage.TextureBinding,
					label: "WebGPUShadowTransmittanceAtlas",
				}),
			};
		}

		return this._atlas.texture;
	}

	public get atlas(): IRenderTexture | null {
		return this._atlas?.texture ?? null;
	}

	public get transmittanceAtlas(): IRenderTexture | null {
		return this._atlas?.transmittanceTexture ?? null;
	}

	public get tileSize(): number {
		return this._atlas?.tileSize ?? 0;
	}

	public get columns(): number {
		return WEBGPU_SHADOW_ATLAS_COLUMNS;
	}

	public get rows(): number {
		return WEBGPU_SHADOW_ATLAS_ROWS;
	}

	public get directionalAtlas(): IRenderTexture | null {
		return this.atlas;
	}

	public get spotAtlas(): IRenderTexture | null {
		return this.atlas;
	}

	public destroy(): void {
		this._atlas?.texture.destroy();
		this._atlas?.transmittanceTexture.destroy();
		this._atlas = null;
	}

	private _resolveAtlasTileSize(tileSize: number): number {
		const safeTileSize = Math.max(1, tileSize | 0);
		const maxTextureDimension2D = this._resolveMaxTextureDimension2D();
		const maxTileSizeByWidth = Math.floor(
			maxTextureDimension2D / Math.max(1, WEBGPU_SHADOW_ATLAS_COLUMNS)
		);
		const maxTileSizeByHeight = Math.floor(
			maxTextureDimension2D / Math.max(1, WEBGPU_SHADOW_ATLAS_ROWS)
		);
		const maxTileSize = Math.max(
			1,
			Math.min(maxTileSizeByWidth, maxTileSizeByHeight)
		);
		return Math.min(safeTileSize, maxTileSize);
	}

	private _resolveMaxTextureDimension2D(): number {
		const fallback = 8192;
		const limit = this._backend.device?.limits?.maxTextureDimension2D;
		if (!Number.isFinite(limit)) {
			return fallback;
		}
		return Math.max(1, Math.floor(limit));
	}
}

function getMaxShadowSize(shadows: ShadowSlice[]): number {
	let tileSize = 0;
	for (const shadow of shadows) {
		if (!shadow?.enabled) continue;
		tileSize = Math.max(tileSize, shadow.shadowMapBaseSize | 0);
	}

	return tileSize;
}
