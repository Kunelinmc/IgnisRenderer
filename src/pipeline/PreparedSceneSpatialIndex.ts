import type { DirtyRect } from "./incremental";
import type {
	DrawPacket,
	PreparedSceneSpatialIndex as IPreparedSceneSpatialIndex,
} from "./types";

interface PreparedSceneSpatialIndexBuildInput {
	viewportWidth: number;
	viewportHeight: number;
	tileSize: number;
	packetRects: ReadonlyMap<string, DirtyRect>;
	opaquePackets: DrawPacket[];
	transparentPackets: DrawPacket[];
}

export class PreparedSceneTileSpatialIndex
	implements IPreparedSceneSpatialIndex
{
	private _viewportWidth: number;
	private _viewportHeight: number;
	private _tileSize: number;
	private _tileColumns: number;
	private _tileRows: number;
	private _opaqueBuckets = new Map<number, number[]>();
	private _transparentBuckets = new Map<number, number[]>();
	private _opaqueFallbackIndices: number[] = [];
	private _transparentFallbackIndices: number[] = [];
	private _opaquePackets: DrawPacket[];
	private _transparentPackets: DrawPacket[];

	public constructor(input: PreparedSceneSpatialIndexBuildInput) {
		this._viewportWidth = Math.max(1, Math.floor(input.viewportWidth));
		this._viewportHeight = Math.max(1, Math.floor(input.viewportHeight));
		this._tileSize = Math.max(4, Math.floor(input.tileSize || 32));
		this._tileColumns = Math.max(
			1,
			Math.ceil(this._viewportWidth / this._tileSize)
		);
		this._tileRows = Math.max(
			1,
			Math.ceil(this._viewportHeight / this._tileSize)
		);
		this._opaquePackets = input.opaquePackets.slice();
		this._transparentPackets = input.transparentPackets.slice();
		this._indexPackets(
			this._opaquePackets,
			input.packetRects,
			this._opaqueBuckets,
			this._opaqueFallbackIndices
		);
		this._indexPackets(
			this._transparentPackets,
			input.packetRects,
			this._transparentBuckets,
			this._transparentFallbackIndices
		);
	}

	public queryOpaquePackets(rect: DirtyRect): DrawPacket[] {
		return this._queryPackets(
			rect,
			this._opaquePackets,
			this._opaqueBuckets,
			this._opaqueFallbackIndices
		);
	}

	public queryTransparentPackets(rect: DirtyRect): DrawPacket[] {
		return this._queryPackets(
			rect,
			this._transparentPackets,
			this._transparentBuckets,
			this._transparentFallbackIndices
		);
	}

	public queryOpaquePacketsInRects(rects: DirtyRect[]): DrawPacket[] {
		return this._queryPacketsInRects(
			rects,
			this._opaquePackets,
			this._opaqueBuckets,
			this._opaqueFallbackIndices
		);
	}

	public queryTransparentPacketsInRects(rects: DirtyRect[]): DrawPacket[] {
		return this._queryPacketsInRects(
			rects,
			this._transparentPackets,
			this._transparentBuckets,
			this._transparentFallbackIndices
		);
	}

	private _queryPacketsInRects(
		rects: DirtyRect[],
		packets: DrawPacket[],
		buckets: Map<number, number[]>,
		fallbackIndices: number[]
	): DrawPacket[] {
		if (packets.length === 0) {
			return [];
		}
		const indices = new Set<number>(fallbackIndices);
		for (const rect of rects) {
			this._collectIndicesForRect(rect, buckets, indices);
		}
		if (indices.size === packets.length) {
			return packets.slice();
		}
		return this._packetsFromIndexSet(indices, packets);
	}

	private _queryPackets(
		rect: DirtyRect,
		packets: DrawPacket[],
		buckets: Map<number, number[]>,
		fallbackIndices: number[]
	): DrawPacket[] {
		if (packets.length === 0) {
			return [];
		}
		const indices = new Set<number>(fallbackIndices);
		this._collectIndicesForRect(rect, buckets, indices);
		if (indices.size === packets.length) {
			return packets.slice();
		}
		return this._packetsFromIndexSet(indices, packets);
	}

	private _collectIndicesForRect(
		rect: DirtyRect,
		buckets: Map<number, number[]>,
		result: Set<number>
	): void {
		const clamped = this._clampRect(rect);
		if (!clamped) {
			return;
		}
		const minTileX = Math.floor(clamped.x / this._tileSize);
		const minTileY = Math.floor(clamped.y / this._tileSize);
		const maxTileX = Math.floor((clamped.x + clamped.width - 1) / this._tileSize);
		const maxTileY = Math.floor(
			(clamped.y + clamped.height - 1) / this._tileSize
		);
		for (let tileY = minTileY; tileY <= maxTileY; tileY++) {
			for (let tileX = minTileX; tileX <= maxTileX; tileX++) {
				if (
					tileX < 0 ||
					tileY < 0 ||
					tileX >= this._tileColumns ||
					tileY >= this._tileRows
				) {
					continue;
				}
				const tileIndex = tileY * this._tileColumns + tileX;
				const bucket = buckets.get(tileIndex);
				if (!bucket) {
					continue;
				}
				for (const packetIndex of bucket) {
					result.add(packetIndex);
				}
			}
		}
	}

	private _packetsFromIndexSet(
		indices: Set<number>,
		packets: DrawPacket[]
	): DrawPacket[] {
		if (indices.size === 0) {
			return [];
		}
		const ordered = Array.from(indices)
			.filter((index) => index >= 0 && index < packets.length)
			.sort((left, right) => left - right);
		const result = new Array<DrawPacket>(ordered.length);
		for (let i = 0; i < ordered.length; i++) {
			result[i] = packets[ordered[i]];
		}
		return result;
	}

	private _indexPackets(
		packets: DrawPacket[],
		packetRects: ReadonlyMap<string, DirtyRect>,
		buckets: Map<number, number[]>,
		fallbackIndices: number[]
	): void {
		for (let packetIndex = 0; packetIndex < packets.length; packetIndex++) {
			const packet = packets[packetIndex];
			const rect = packetRects.get(packet.id);
			const clamped = rect ? this._clampRect(rect) : null;
			if (!clamped) {
				fallbackIndices.push(packetIndex);
				continue;
			}
			const minTileX = Math.floor(clamped.x / this._tileSize);
			const minTileY = Math.floor(clamped.y / this._tileSize);
			const maxTileX = Math.floor(
				(clamped.x + clamped.width - 1) / this._tileSize
			);
			const maxTileY = Math.floor(
				(clamped.y + clamped.height - 1) / this._tileSize
			);
			for (let tileY = minTileY; tileY <= maxTileY; tileY++) {
				for (let tileX = minTileX; tileX <= maxTileX; tileX++) {
					if (
						tileX < 0 ||
						tileY < 0 ||
						tileX >= this._tileColumns ||
						tileY >= this._tileRows
					) {
						continue;
					}
					const tileIndex = tileY * this._tileColumns + tileX;
					let bucket = buckets.get(tileIndex);
					if (!bucket) {
						bucket = [];
						buckets.set(tileIndex, bucket);
					}
					bucket.push(packetIndex);
				}
			}
		}
	}

	private _clampRect(rect: DirtyRect): DirtyRect | null {
		const minX = Math.max(0, Math.floor(rect.x));
		const minY = Math.max(0, Math.floor(rect.y));
		const maxX = Math.min(
			this._viewportWidth,
			Math.ceil(rect.x + Math.max(0, rect.width))
		);
		const maxY = Math.min(
			this._viewportHeight,
			Math.ceil(rect.y + Math.max(0, rect.height))
		);
		const width = maxX - minX;
		const height = maxY - minY;
		if (width <= 0 || height <= 0) {
			return null;
		}
		return {
			x: minX,
			y: minY,
			width,
			height,
		};
	}
}
