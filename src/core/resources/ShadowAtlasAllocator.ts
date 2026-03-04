import type { WebGPULightingState } from "../bridge/webgpu";
import { alignTo } from "../bridge/webgpu";
import type { ShadowMap } from "../../utils/ShadowMapping";
import type { WebGPUBackend } from "../backend/WebGPUBackend";
import { TextureFormat, TextureUsage, type IRenderTexture } from "../ral/types";

interface ShadowAtlas {
	tileSize: number;
	texture: IRenderTexture;
	uploadBuffer: Uint8Array;
}

export class ShadowAtlasAllocator {
	private _backend: WebGPUBackend;
	private _directionalAtlas: ShadowAtlas | null = null;
	private _spotAtlas: ShadowAtlas | null = null;

	constructor(backend: WebGPUBackend) {
		this._backend = backend;
	}

	public prepare(lightingState: WebGPULightingState): void {
		this._directionalAtlas = this._prepareShadowAtlas(
			lightingState.directionalShadows,
			this._directionalAtlas,
			"WebGPUDirectionalShadowAtlas"
		);
		this._spotAtlas = this._prepareShadowAtlas(
			lightingState.spotShadows,
			this._spotAtlas,
			"WebGPUSpotShadowAtlas"
		);
	}

	public get directionalAtlas(): IRenderTexture | null {
		return this._directionalAtlas?.texture ?? null;
	}

	public get spotAtlas(): IRenderTexture | null {
		return this._spotAtlas?.texture ?? null;
	}

	private _prepareShadowAtlas(
		shadows: Array<{
			enabled: boolean;
			shadowMap: ShadowMap | null;
			atlasTileSize: number;
		}>,
		current: ShadowAtlas | null,
		label: string
	): ShadowAtlas | null {
		let tileSize = 0;
		for (const shadow of shadows) {
			if (!shadow?.enabled || !shadow.shadowMap) continue;
			tileSize = Math.max(tileSize, shadow.shadowMap.size | 0);
		}

		for (const shadow of shadows) {
			if (!shadow) continue;
			shadow.atlasTileSize = tileSize;
		}

		if (tileSize <= 0) return null;

		let atlas = current;
		if (!atlas || atlas.tileSize !== tileSize) {
			atlas?.texture.destroy();

			const atlasWidth = tileSize * 2;
			const atlasHeight = tileSize * 2;
			const bytesPerRow = alignTo(atlasWidth * 4, 256);

			atlas = {
				tileSize,
				texture: this._backend.createTexture({
					width: tileSize * 2,
					height: tileSize * 2,
					format: TextureFormat.RGBA8Unorm,
					usage: TextureUsage.TextureBinding | TextureUsage.CopyDst,
					label,
				}),
				uploadBuffer: new Uint8Array(bytesPerRow * atlasHeight),
			};
		}

		const upload = createShadowAtlasUploadData(
			shadows,
			tileSize,
			atlas.uploadBuffer
		);
		this._backend.writeTexture(
			atlas.texture,
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

		return atlas;
	}
}

function createShadowAtlasUploadData(
	shadows: Array<{ enabled: boolean; shadowMap: ShadowMap | null }>,
	tileSize: number,
	data: Uint8Array
): {
	data: Uint8Array;
	bytesPerRow: number;
	width: number;
	height: number;
} {
	const atlasWidth = tileSize * 2;
	const atlasHeight = tileSize * 2;
	const bytesPerRow = alignTo(atlasWidth * 4, 256);
	data.fill(255);

	for (let shadowIndex = 0; shadowIndex < shadows.length; shadowIndex++) {
		const shadow = shadows[shadowIndex];
		if (!shadow?.enabled || !shadow.shadowMap) continue;

		const { size, buffer } = shadow.shadowMap;
		const tileX = shadowIndex % 2;
		const tileY = (shadowIndex / 2) | 0;
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

	return {
		data,
		bytesPerRow,
		width: atlasWidth,
		height: atlasHeight,
	};
}

function clamp(value: number, min: number, max: number): number {
	return Math.max(min, Math.min(max, value));
}
