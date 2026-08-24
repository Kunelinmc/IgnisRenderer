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

interface PacketRectBVHEntry {
	packetIndex: number;
	minX: number;
	minY: number;
	maxX: number;
	maxY: number;
	centroidX: number;
	centroidY: number;
}

interface PacketRectBVHNode {
	minX: number;
	minY: number;
	maxX: number;
	maxY: number;
	left?: PacketRectBVHNode;
	right?: PacketRectBVHNode;
	entries?: PacketRectBVHEntry[];
}

interface PacketRectBVHStats {
	minX: number;
	minY: number;
	maxX: number;
	maxY: number;
	centroidExtentX: number;
	centroidExtentY: number;
}

interface RectQueryBounds {
	minX: number;
	minY: number;
	maxX: number;
	maxY: number;
}

const PACKET_RECT_BVH_LEAF_SIZE = 8;
const PACKET_RECT_BVH_AXIS_EPSILON = 1e-4;

export class PreparedSceneTileSpatialIndex
	implements IPreparedSceneSpatialIndex
{
	private _viewportWidth: number;
	private _viewportHeight: number;
	private _leafSize: number;
	private _opaqueFallbackIndices: number[] = [];
	private _transparentFallbackIndices: number[] = [];
	private _opaquePackets: DrawPacket[];
	private _transparentPackets: DrawPacket[];
	private _opaqueTree: PacketRectBVHNode | null = null;
	private _transparentTree: PacketRectBVHNode | null = null;

	public constructor(input: PreparedSceneSpatialIndexBuildInput) {
		this._viewportWidth = Math.max(1, Math.floor(input.viewportWidth));
		this._viewportHeight = Math.max(1, Math.floor(input.viewportHeight));
		const resolvedTileSize = Math.max(4, Math.floor(input.tileSize || 32));
		this._leafSize = Math.max(
			4,
			Math.min(32, Math.floor(resolvedTileSize / 4) || PACKET_RECT_BVH_LEAF_SIZE)
		);
		this._opaquePackets = input.opaquePackets.slice();
		this._transparentPackets = input.transparentPackets.slice();
		this._opaqueTree = this._buildPacketTree(
			this._opaquePackets,
			input.packetRects,
			this._opaqueFallbackIndices
		);
		this._transparentTree = this._buildPacketTree(
			this._transparentPackets,
			input.packetRects,
			this._transparentFallbackIndices
		);
	}

	public queryOpaquePackets(rect: DirtyRect): DrawPacket[] {
		return this._queryPackets(
			rect,
			this._opaquePackets,
			this._opaqueTree,
			this._opaqueFallbackIndices
		);
	}

	public queryTransparentPackets(rect: DirtyRect): DrawPacket[] {
		return this._queryPackets(
			rect,
			this._transparentPackets,
			this._transparentTree,
			this._transparentFallbackIndices
		);
	}

	public queryOpaquePacketsInRects(rects: DirtyRect[]): DrawPacket[] {
		return this._queryPacketsInRects(
			rects,
			this._opaquePackets,
			this._opaqueTree,
			this._opaqueFallbackIndices
		);
	}

	public queryTransparentPacketsInRects(rects: DirtyRect[]): DrawPacket[] {
		return this._queryPacketsInRects(
			rects,
			this._transparentPackets,
			this._transparentTree,
			this._transparentFallbackIndices
		);
	}

	private _queryPacketsInRects(
		rects: DirtyRect[],
		packets: DrawPacket[],
		tree: PacketRectBVHNode | null,
		fallbackIndices: number[]
	): DrawPacket[] {
		if (packets.length === 0) {
			return [];
		}
		const indices = new Set<number>(fallbackIndices);
		for (const rect of rects) {
			this._collectIndicesForRect(rect, tree, indices);
		}
		if (indices.size === packets.length) {
			return packets.slice();
		}
		return this._packetsFromIndexSet(indices, packets);
	}

	private _queryPackets(
		rect: DirtyRect,
		packets: DrawPacket[],
		tree: PacketRectBVHNode | null,
		fallbackIndices: number[]
	): DrawPacket[] {
		if (packets.length === 0) {
			return [];
		}
		const indices = new Set<number>(fallbackIndices);
		this._collectIndicesForRect(rect, tree, indices);
		if (indices.size === packets.length) {
			return packets.slice();
		}
		return this._packetsFromIndexSet(indices, packets);
	}

	private _collectIndicesForRect(
		rect: DirtyRect,
		tree: PacketRectBVHNode | null,
		result: Set<number>
	): void {
		if (!tree) {
			return;
		}
		const clamped = this._clampRect(rect);
		if (!clamped) {
			return;
		}
		queryPacketRectBVH(tree, toRectQueryBounds(clamped), result);
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

	private _buildPacketTree(
		packets: DrawPacket[],
		packetRects: ReadonlyMap<string, DirtyRect>,
		fallbackIndices: number[]
	): PacketRectBVHNode | null {
		const entries: PacketRectBVHEntry[] = [];
		for (let packetIndex = 0; packetIndex < packets.length; packetIndex++) {
			const packet = packets[packetIndex];
			const rect = packetRects.get(packet.submission.id);
			const clamped = rect ? this._clampRect(rect) : null;
			if (!clamped) {
				fallbackIndices.push(packetIndex);
				continue;
			}
			const minX = clamped.x;
			const minY = clamped.y;
			const maxX = clamped.x + clamped.width;
			const maxY = clamped.y + clamped.height;
			entries.push({
				packetIndex,
				minX,
				minY,
				maxX,
				maxY,
				centroidX: (minX + maxX) * 0.5,
				centroidY: (minY + maxY) * 0.5,
			});
		}
		if (entries.length === 0) {
			return null;
		}
		return buildPacketRectBVH(entries, 0, entries.length, this._leafSize);
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

function buildPacketRectBVH(
	entries: PacketRectBVHEntry[],
	start: number,
	end: number,
	leafSize: number
): PacketRectBVHNode | null {
	const count = end - start;
	if (count <= 0) {
		return null;
	}

	const stats = computePacketRectBVHStats(entries, start, end);
	if (
		count <= Math.max(1, leafSize) ||
		Math.max(stats.centroidExtentX, stats.centroidExtentY) <=
			PACKET_RECT_BVH_AXIS_EPSILON
	) {
		return {
			minX: stats.minX,
			minY: stats.minY,
			maxX: stats.maxX,
			maxY: stats.maxY,
			entries: entries.slice(start, end),
		};
	}

	const splitAxis = stats.centroidExtentX >= stats.centroidExtentY ? "x" : "y";
	const middle = start + (count >> 1);
	quickSelectPacketRectEntries(entries, start, end, middle, splitAxis);
	const left = buildPacketRectBVH(entries, start, middle, leafSize);
	const right = buildPacketRectBVH(entries, middle, end, leafSize);
	if (!left || !right) {
		return {
			minX: stats.minX,
			minY: stats.minY,
			maxX: stats.maxX,
			maxY: stats.maxY,
			entries: entries.slice(start, end),
		};
	}
	return {
		minX: stats.minX,
		minY: stats.minY,
		maxX: stats.maxX,
		maxY: stats.maxY,
		left,
		right,
	};
}

function computePacketRectBVHStats(
	entries: PacketRectBVHEntry[],
	start: number,
	end: number
): PacketRectBVHStats {
	let minX = Infinity;
	let minY = Infinity;
	let maxX = -Infinity;
	let maxY = -Infinity;
	let centroidMinX = Infinity;
	let centroidMinY = Infinity;
	let centroidMaxX = -Infinity;
	let centroidMaxY = -Infinity;
	for (let index = start; index < end; index++) {
		const entry = entries[index];
		if (entry.minX < minX) minX = entry.minX;
		if (entry.minY < minY) minY = entry.minY;
		if (entry.maxX > maxX) maxX = entry.maxX;
		if (entry.maxY > maxY) maxY = entry.maxY;
		if (entry.centroidX < centroidMinX) centroidMinX = entry.centroidX;
		if (entry.centroidY < centroidMinY) centroidMinY = entry.centroidY;
		if (entry.centroidX > centroidMaxX) centroidMaxX = entry.centroidX;
		if (entry.centroidY > centroidMaxY) centroidMaxY = entry.centroidY;
	}
	return {
		minX,
		minY,
		maxX,
		maxY,
		centroidExtentX: centroidMaxX - centroidMinX,
		centroidExtentY: centroidMaxY - centroidMinY,
	};
}

function queryPacketRectBVH(
	node: PacketRectBVHNode,
	rect: RectQueryBounds,
	result: Set<number>
): void {
	if (!rectQueryBoundsIntersect(node, rect)) {
		return;
	}
	if (node.entries) {
		for (const entry of node.entries) {
			if (!rectQueryBoundsIntersect(entry, rect)) {
				continue;
			}
			result.add(entry.packetIndex);
		}
		return;
	}
	if (node.left) {
		queryPacketRectBVH(node.left, rect, result);
	}
	if (node.right) {
		queryPacketRectBVH(node.right, rect, result);
	}
}

function toRectQueryBounds(rect: DirtyRect): RectQueryBounds {
	return {
		minX: rect.x,
		minY: rect.y,
		maxX: rect.x + rect.width,
		maxY: rect.y + rect.height,
	};
}

function rectQueryBoundsIntersect(
	left: { minX: number; minY: number; maxX: number; maxY: number },
	right: { minX: number; minY: number; maxX: number; maxY: number }
): boolean {
	return (
		left.minX <= right.maxX &&
		left.maxX >= right.minX &&
		left.minY <= right.maxY &&
		left.maxY >= right.minY
	);
}

function quickSelectPacketRectEntries(
	entries: PacketRectBVHEntry[],
	start: number,
	end: number,
	target: number,
	axis: "x" | "y"
): void {
	let left = start;
	let right = end - 1;
	while (left < right) {
		const pivotIndex = left + ((right - left) >> 1);
		const pivotValue = getPacketRectEntryCentroid(entries[pivotIndex], axis);
		let lt = left;
		let gt = right;
		let index = left;
		while (index <= gt) {
			const value = getPacketRectEntryCentroid(entries[index], axis);
			if (value < pivotValue) {
				swapPacketRectEntries(entries, lt, index);
				lt++;
				index++;
			} else if (value > pivotValue) {
				swapPacketRectEntries(entries, index, gt);
				gt--;
			} else {
				index++;
			}
		}
		if (target < lt) {
			right = lt - 1;
		} else if (target > gt) {
			left = gt + 1;
		} else {
			return;
		}
	}
}

function getPacketRectEntryCentroid(
	entry: PacketRectBVHEntry,
	axis: "x" | "y"
): number {
	return axis === "x" ? entry.centroidX : entry.centroidY;
}

function swapPacketRectEntries(
	entries: PacketRectBVHEntry[],
	leftIndex: number,
	rightIndex: number
): void {
	if (leftIndex === rightIndex) {
		return;
	}
	const tmp = entries[leftIndex];
	entries[leftIndex] = entries[rightIndex];
	entries[rightIndex] = tmp;
}
