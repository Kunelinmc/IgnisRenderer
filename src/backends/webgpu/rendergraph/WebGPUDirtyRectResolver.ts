import type {
	DirtyRect,
} from "../../../pipeline/incremental";
import type {
	DrawPacket,
	FrameContext,
} from "../../../pipeline/types";

/**
 * Resolves incremental dirty regions and per-rect packet subsets for WebGPU
 * scene recording.
 *
 * @internal WebGPU frame-recording utility.
 */
export class WebGPUDirtyRectResolver {
	/**
	 * Returns whether the frame can reuse non-dirty regions from previous GPU
	 * attachments.
	 *
	 * @param context Current frame context, or `null` before frame setup.
	 * @returns `true` when incremental rendering is active for dirty rectangles.
	 * @sideEffects None.
	 */
	public static isIncrementalPartial(context: FrameContext | null): boolean {
		if (!context?.incremental) {
			return false;
		}
		return (
			context.incremental.enabled &&
			!context.incremental.forceFullFrame &&
			context.incremental.dirtyRects.length > 0
		);
	}

	/**
	 * Resolves dirty rectangles in the coordinate space of a target attachment.
	 *
	 * @param context Current frame context, or `null` before frame setup.
	 * @param targetWidth Target attachment width.
	 * @param targetHeight Target attachment height.
	 * @returns Full-frame or scaled dirty rectangles clipped to the target.
	 * @sideEffects None.
	 */
	public static resolveDirtyRects(
		context: FrameContext | null,
		targetWidth: number,
		targetHeight: number
	): DirtyRect[] {
		const width = Math.max(1, Math.floor(targetWidth));
		const height = Math.max(1, Math.floor(targetHeight));
		if (!context) {
			return [{ x: 0, y: 0, width, height }];
		}
		if (!WebGPUDirtyRectResolver.isIncrementalPartial(context)) {
			return [{ x: 0, y: 0, width, height }];
		}
		const sourceWidth = Math.max(1, Math.floor(context.attachments.width));
		const sourceHeight = Math.max(1, Math.floor(context.attachments.height));
		const scaleX = width / sourceWidth;
		const scaleY = height / sourceHeight;
		const resolved: DirtyRect[] = [];
		for (const rect of context.incremental.dirtyRects) {
			const minX = Math.max(0, Math.floor(rect.x * scaleX));
			const minY = Math.max(0, Math.floor(rect.y * scaleY));
			const maxX = Math.min(
				width,
				Math.ceil((rect.x + rect.width) * scaleX)
			);
			const maxY = Math.min(
				height,
				Math.ceil((rect.y + rect.height) * scaleY)
			);
			const rectWidth = maxX - minX;
			const rectHeight = maxY - minY;
			if (rectWidth <= 0 || rectHeight <= 0) {
				continue;
			}
			resolved.push({
				x: minX,
				y: minY,
				width: rectWidth,
				height: rectHeight,
			});
		}
		return resolved;
	}

	/**
	 * Selects draw packets intersecting a dirty rectangle when a prepared scene
	 * spatial index is available.
	 *
	 * @param context Current frame context.
	 * @param packets Candidate draw packets.
	 * @param rect Dirty rectangle in target coordinates.
	 * @returns Candidate packets filtered for the rectangle.
	 * @sideEffects None.
	 */
	public static selectPacketsForRect(
		context: FrameContext,
		packets: DrawPacket[],
		rect: DirtyRect
	): DrawPacket[] {
		const spatialIndex = context.scene.spatialIndex;
		if (!spatialIndex) {
			return packets;
		}
		if (packets === context.scene.opaquePackets) {
			return spatialIndex.queryOpaquePackets(rect);
		}
		if (packets === context.scene.transparentPackets) {
			return spatialIndex.queryTransparentPackets(rect);
		}
		return packets;
	}

	/**
	 * Selects transparent packet subsets while preserving explicit caller-owned
	 * packet filters.
	 *
	 * @param context Current frame context.
	 * @param packets Candidate transparent packets.
	 * @param rect Dirty rectangle in target coordinates.
	 * @returns Transparent packets intersecting the rectangle and candidate set.
	 * @sideEffects None.
	 */
	public static selectTransparentSubsetForRect(
		context: FrameContext,
		packets: DrawPacket[],
		rect: DirtyRect
	): DrawPacket[] {
		const spatialIndex = context.scene.spatialIndex;
		if (!spatialIndex) {
			return packets;
		}
		const rectPackets = spatialIndex.queryTransparentPackets(rect);
		if (packets === context.scene.transparentPackets) {
			return rectPackets;
		}
		if (packets.length <= 0 || rectPackets.length <= 0) {
			return [];
		}
		const packetSet = new Set(packets);
		return rectPackets.filter((packet) => packetSet.has(packet));
	}
}
