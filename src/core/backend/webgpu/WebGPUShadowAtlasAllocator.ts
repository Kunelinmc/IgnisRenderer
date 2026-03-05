import type { WebGPULightingState } from "./";
import { alignTo } from "./";
import { clamp } from "../../../maths/Common";
import type { ShadowMap } from "../../../utils/ShadowMapping";
import type { WebGPUBackend } from "../WebGPUBackend";
import { TextureFormat, TextureUsage, type IRenderTexture } from "../types";

interface ShadowAtlas {
	tileSize: number;
	texture: IRenderTexture;
	uploadBuffer: Uint8Array;
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
		const tileSize = Math.max(
			getMaxShadowSize(directionalShadows),
			getMaxShadowSize(spotShadows)
		);

		for (const shadow of directionalShadows) {
			shadow.atlasTileSize = tileSize;
		}
		for (const shadow of spotShadows) {
			shadow.atlasTileSize = tileSize;
		}

		if (tileSize <= 0) {
			this._atlas?.texture.destroy();
			this._atlas = null;
			return;
		}

		if (!this._atlas || this._atlas.tileSize !== tileSize) {
			this._atlas?.texture.destroy();

			const atlasWidth = tileSize * SHADOW_ATLAS_COLUMNS;
			const atlasHeight = tileSize * SHADOW_ATLAS_ROWS;
			const bytesPerRow = alignTo(atlasWidth * 4, 256);

			this._atlas = {
				tileSize,
				texture: this._backend.createTexture({
					width: atlasWidth,
					height: atlasHeight,
					format: TextureFormat.RGBA8Unorm,
					usage: TextureUsage.TextureBinding | TextureUsage.CopyDst,
					label: "WebGPUShadowAtlas",
				}),
				uploadBuffer: new Uint8Array(bytesPerRow * atlasHeight),
			};
		}

		const upload = createShadowAtlasUploadData(
			directionalShadows,
			spotShadows,
			tileSize,
			this._atlas.uploadBuffer
		);
		this._backend.writeTexture(
			this._atlas.texture,
			upload.data as any,
			{
				bytesPerRow: upload.bytesPerRow,
				rowsPerImage: upload.height,
			},
			{
				width: upload.width,
				height: upload.height,
				depthOrArrayLayers: 1,
			}
		);
	}

	public get atlas(): IRenderTexture | null {
		return this._atlas?.texture ?? null;
	}

	public get directionalAtlas(): IRenderTexture | null {
		return this.atlas;
	}

	public get spotAtlas(): IRenderTexture | null {
		return this.atlas;
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

function createShadowAtlasUploadData(
	directionalShadows: ShadowSlice[],
	spotShadows: ShadowSlice[],
	tileSize: number,
	data: Uint8Array
): {
	data: Uint8Array;
	bytesPerRow: number;
	width: number;
	height: number;
} {
	const atlasWidth = tileSize * SHADOW_ATLAS_COLUMNS;
	const atlasHeight = tileSize * SHADOW_ATLAS_ROWS;
	const bytesPerRow = alignTo(atlasWidth * 4, 256);
	data.fill(255);

	for (let shadowIndex = 0; shadowIndex < directionalShadows.length; shadowIndex++) {
		const shadow = directionalShadows[shadowIndex];
		if (!shadow?.enabled || !shadow.shadowMap) continue;
		copyShadowMapToTile(
			shadow.shadowMap,
			shadowIndex,
			0,
			tileSize,
			bytesPerRow,
			data
		);
	}

	for (let shadowIndex = 0; shadowIndex < spotShadows.length; shadowIndex++) {
		const shadow = spotShadows[shadowIndex];
		if (!shadow?.enabled || !shadow.shadowMap) continue;
		copyShadowMapToTile(
			shadow.shadowMap,
			shadowIndex,
			1,
			tileSize,
			bytesPerRow,
			data
		);
	}

	return {
		data,
		bytesPerRow,
		width: atlasWidth,
		height: atlasHeight,
	};
}

function copyShadowMapToTile(
	shadowMap: ShadowMap,
	tileX: number,
	tileY: number,
	tileSize: number,
	bytesPerRow: number,
	data: Uint8Array
): void {
	const { size, buffer } = shadowMap;
	const originX = tileX * tileSize;
	const originY = tileY * tileSize;

	for (let y = 0; y < size; y++) {
		const shadowRow = y * size;
		const atlasRowOffset = (originY + y) * bytesPerRow;
		for (let x = 0; x < size; x++) {
			const depth = buffer[shadowRow + x];
			const normalized = Number.isFinite(depth)
				? clamp(depth * 0.5 + 0.5, 0, 1)
				: 1;
			const encoded = Math.round(normalized * 0xffffffff);
			const pixelOffset = atlasRowOffset + (originX + x) * 4;
			data[pixelOffset] = encoded >>> 24;
			data[pixelOffset + 1] = (encoded >>> 16) & 0xff;
			data[pixelOffset + 2] = (encoded >>> 8) & 0xff;
			data[pixelOffset + 3] = encoded & 0xff;
		}
	}
}
