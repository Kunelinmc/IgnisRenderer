import type { DrawPacket } from "../../pipeline/types";
import type { ICommandEncoder } from "../ICommandEncoder";
import type { IBindingGroup } from "../types";
import type {
	WebGPUDrawResourceOptions,
	WebGPUDrawResources,
	WebGPUPreparedFrameResources,
} from "./WebGPURenderResources";

export interface WebGPUDrawSubmissionRect {
	x: number;
	y: number;
	width: number;
	height: number;
}

export interface WebGPUDrawSubmissionBinding {
	slot: number;
	group: IBindingGroup;
}

export interface WebGPUDrawResourceProvider {
	getDrawResources(
		packet: DrawPacket,
		frameResources: WebGPUPreparedFrameResources,
		options?: WebGPUDrawResourceOptions
	): Promise<WebGPUDrawResources[] | null>;
}

export interface WebGPUDrawSubmissionRequest {
	encoder: ICommandEncoder;
	resources: WebGPUDrawResourceProvider;
	frameResources: WebGPUPreparedFrameResources;
	packets: DrawPacket[];
	dirtyRects?: readonly WebGPUDrawSubmissionRect[] | null;
	selectPacketsForRect?: (
		packets: DrawPacket[],
		rect: WebGPUDrawSubmissionRect
	) => DrawPacket[];
	resolveDrawOptions?: (
		packet: DrawPacket,
		rect: WebGPUDrawSubmissionRect | null
	) => WebGPUDrawResourceOptions;
	resolveBindings?: (
		draw: WebGPUDrawResources,
		packet: DrawPacket,
		rect: WebGPUDrawSubmissionRect | null
	) => readonly WebGPUDrawSubmissionBinding[];
}

export interface WebGPUDrawSubmissionResult {
	drawCount: number;
	submittedPacketIds: Set<string>;
}

export function getDefaultWebGPUDrawBindings(
	draw: WebGPUDrawResources
): WebGPUDrawSubmissionBinding[] {
	return [
		{ slot: 0, group: draw.frameBinding },
		{ slot: 1, group: draw.modelBinding },
		{ slot: 2, group: draw.clusteredBinding },
	];
}

export async function submitWebGPUDraws(
	request: WebGPUDrawSubmissionRequest
): Promise<WebGPUDrawSubmissionResult> {
	const submittedPacketIds = new Set<string>();
	let drawCount = 0;
	const rects = request.dirtyRects;

	const submitPackets = async (
		packets: DrawPacket[],
		rect: WebGPUDrawSubmissionRect | null
	): Promise<void> => {
		for (const packet of packets) {
			const resourcesList = await request.resources.getDrawResources(
				packet,
				request.frameResources,
				request.resolveDrawOptions?.(packet, rect) ?? {}
			);
			if (!resourcesList || resourcesList.length <= 0) {
				continue;
			}
			submittedPacketIds.add(packet.id);
			for (const draw of resourcesList) {
				request.encoder.setPipeline(draw.pipeline);
				const bindings =
					request.resolveBindings?.(draw, packet, rect) ??
					getDefaultWebGPUDrawBindings(draw);
				for (const binding of bindings) {
					request.encoder.setBindingGroup(binding.slot, binding.group);
				}
				request.encoder.setVertexBuffer(0, draw.vertexBuffer);
				request.encoder.setIndexBuffer(draw.indexBuffer, "uint32");
				request.encoder.drawIndexed(draw.indexCount);
				drawCount++;
			}
		}
	};

	if (!rects || rects.length <= 0) {
		await submitPackets(request.packets, null);
		return {
			drawCount,
			submittedPacketIds,
		};
	}

	for (const rect of rects) {
		const packetsInRect =
			request.selectPacketsForRect?.(request.packets, rect) ?? request.packets;
		if (packetsInRect.length <= 0) {
			continue;
		}
		request.encoder.setScissorRect?.(rect.x, rect.y, rect.width, rect.height);
		await submitPackets(packetsInRect, rect);
	}

	return {
		drawCount,
		submittedPacketIds,
	};
}
