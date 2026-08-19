import type { DrawPacket } from "../../pipeline/types";
import type { ICommandEncoder } from "../ICommandEncoder";
import type {
	IBindingGroup,
	IRenderBuffer,
	IRenderPipeline,
	IndexFormat,
} from "../types";
import type {
	WebGPUDrawResourceOptions,
	WebGPUDrawResources,
	WebGPUPreparedFrameResources,
} from "./WebGPUResourceContracts";

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
		options: WebGPUDrawResourceOptions
	): Promise<WebGPUDrawResources[] | null>;
}

export interface WebGPUDrawSubmissionRequest {
	encoder: ICommandEncoder;
	resources: WebGPUDrawResourceProvider;
	frameResources: WebGPUPreparedFrameResources;
	packets: DrawPacket[];
	preparedResources?: ReadonlyMap<DrawPacket, WebGPUDrawResources[] | null>;
	dirtyRects?: readonly WebGPUDrawSubmissionRect[] | null;
	selectPacketsForRect?: (
		packets: DrawPacket[],
		rect: WebGPUDrawSubmissionRect
	) => DrawPacket[];
	resolveDrawOptions: (
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
	const selectedByRect = rects && rects.length > 0 ? rects.map((rect) =>
		request.selectPacketsForRect?.(request.packets, rect) ?? request.packets
	) : null;
	const packetsToPrepare = selectedByRect
		? Array.from(new Set(selectedByRect.flat()))
		: request.packets;
	const prepared = request.preparedResources;
	const preparedVariants = new WeakMap<
		DrawPacket,
		Map<string, WebGPUDrawResources[] | null>
	>();
	if (!request.preparedResources) {
		await prepareDrawResources(
			request,
			selectedByRect,
			packetsToPrepare,
			preparedVariants,
		);
	}

	let lastPipeline: IRenderPipeline | null = null;
	const lastGroups = new Map<number, IBindingGroup>();
	const lastVertexBuffers = new Map<number, IRenderBuffer>();
	let lastIndexBuffer: IRenderBuffer | null = null;
	let lastIndexFormat: IndexFormat | null = null;
	const bindGroup = (slot: number, group: IBindingGroup): void => {
		if (lastGroups.get(slot) === group) return;
		request.encoder.setBindingGroup(slot, group);
		lastGroups.set(slot, group);
	};

	const submitPackets = async (
		packets: DrawPacket[],
		rect: WebGPUDrawSubmissionRect | null
	): Promise<void> => {
		const commands: Array<{
			packet: DrawPacket;
			draw: WebGPUDrawResources;
			firstInstance: number;
			instanceCount: number;
			packetIds: string[];
		}> = [];
		for (const packet of packets) {
			const resourcesList = prepared
				? prepared.get(packet) ?? null
				: preparedVariants.get(packet)?.get(createDrawOptionsKey(
					request.resolveDrawOptions(packet, rect),
				)) ?? null;
			if (!resourcesList || resourcesList.length <= 0) {
				continue;
			}
			for (const draw of resourcesList) {
				const firstInstance = draw.firstInstance ?? 0;
				const previous = commands[commands.length - 1];
				if (
					draw.staticBatchKey &&
					previous?.draw.staticBatchKey === draw.staticBatchKey &&
					previous.firstInstance + previous.instanceCount === firstInstance
				) {
					previous.instanceCount++;
					previous.packetIds.push(packet.id);
					continue;
				}
				commands.push({
					packet,
					draw,
					firstInstance,
					instanceCount: 1,
					packetIds: [packet.id],
				});
			}
		}
		for (const command of commands) {
			const { draw, packet } = command;
			for (const packetId of command.packetIds) submittedPacketIds.add(packetId);
				if (lastPipeline !== draw.pipeline) {
					request.encoder.setPipeline(draw.pipeline);
					lastPipeline = draw.pipeline;
				}
				if (request.resolveBindings) {
					for (const binding of request.resolveBindings(draw, packet, rect)) {
						bindGroup(binding.slot, binding.group);
					}
				} else {
					bindGroup(0, draw.frameBinding);
					bindGroup(1, draw.modelBinding);
					bindGroup(2, draw.clusteredBinding);
				}
				const vertexBindings = draw.vertexBindings;
				if (!vertexBindings) {
					const buffer = (draw as WebGPUDrawResources & {
						vertexBuffer: IRenderBuffer;
					}).vertexBuffer;
					if (lastVertexBuffers.get(0) !== buffer) {
						request.encoder.setVertexBuffer(0, buffer);
						lastVertexBuffers.set(0, buffer);
					}
				}
				for (const binding of vertexBindings ?? []) {
					if (lastVertexBuffers.get(binding.slot) === binding.buffer) continue;
					request.encoder.setVertexBuffer(binding.slot, binding.buffer);
					lastVertexBuffers.set(binding.slot, binding.buffer);
				}
				const indexFormat = draw.indexFormat ?? "uint32";
				if (lastIndexBuffer !== draw.indexBuffer || lastIndexFormat !== indexFormat) {
					request.encoder.setIndexBuffer(draw.indexBuffer, indexFormat);
					lastIndexBuffer = draw.indexBuffer;
					lastIndexFormat = indexFormat;
				}
				request.encoder.drawIndexed(
					draw.indexCount,
					command.instanceCount,
					0,
					0,
					command.firstInstance,
				);
				drawCount++;
		}
	};

	if (!rects || rects.length <= 0) {
		await submitPackets(request.packets, null);
		return {
			drawCount,
			submittedPacketIds,
		};
	}

	for (let rectIndex = 0; rectIndex < rects.length; rectIndex++) {
		const rect = rects[rectIndex];
		const packetsInRect = selectedByRect?.[rectIndex] ?? request.packets;
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

async function prepareDrawResources(
	request: WebGPUDrawSubmissionRequest,
	selectedByRect: readonly (readonly DrawPacket[])[] | null,
	packets: readonly DrawPacket[],
	target: WeakMap<DrawPacket, Map<string, WebGPUDrawResources[] | null>>,
): Promise<void> {
	const tasks: Array<{
		packet: DrawPacket;
		options: WebGPUDrawResourceOptions;
		key: string;
	}> = [];
	const enqueue = (packet: DrawPacket, rect: WebGPUDrawSubmissionRect | null): void => {
		const options = request.resolveDrawOptions(packet, rect);
		const key = createDrawOptionsKey(options);
		let variants = target.get(packet);
		if (!variants) {
			variants = new Map();
			target.set(packet, variants);
		}
		if (variants.has(key)) return;
		variants.set(key, null);
		tasks.push({ packet, options, key });
	};
	if (selectedByRect && request.dirtyRects) {
		for (let index = 0; index < selectedByRect.length; index++) {
			for (const packet of selectedByRect[index]) {
				enqueue(packet, request.dirtyRects[index]);
			}
		}
	} else {
		for (const packet of packets) enqueue(packet, null);
	}
	let cursor = 0;
	const workerCount = Math.min(16, tasks.length);
	await Promise.all(Array.from({ length: workerCount }, async () => {
		while (cursor < tasks.length) {
			const task = tasks[cursor++];
			const resources = await request.resources.getDrawResources(
				task.packet,
				request.frameResources,
				task.options,
			);
			target.get(task.packet)!.set(task.key, resources);
		}
	}));
}

function createDrawOptionsKey(options: WebGPUDrawResourceOptions): string {
	return [
		options.sceneTargetMode ?? "default",
		options.transparentPipelineMode ?? "default",
		options.drawMode ?? "default",
		options.sampleCount,
		options.deferredGBufferLayout ?? "extended",
	].join("|");
}
