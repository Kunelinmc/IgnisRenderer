import type { DrawPacket } from "../../pipeline/types";
import type { ICommandEncoder } from "../ICommandEncoder";
import type {
	IBindingGroup,
	IRenderBuffer,
	IRenderPipeline,
	IndexFormat,
} from "../types";
import type {
	WebGPUDrawBinding,
	WebGPUDrawResourceOptions,
	WebGPUDrawResourceResult,
	WebGPUDrawResources,
	WebGPUPreparedFrameResources,
	WebGPUStaticBatchingStats,
} from "./WebGPUResourceContracts";

export interface WebGPUDrawSubmissionRect {
	x: number;
	y: number;
	width: number;
	height: number;
}

export interface WebGPUDrawResourceProvider {
	getDrawResources(
		packet: DrawPacket,
		frameResources: WebGPUPreparedFrameResources,
		options: WebGPUDrawResourceOptions
	): WebGPUDrawResourceResult;
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
	) => readonly WebGPUDrawBinding[];
}

export interface WebGPUDrawSubmissionResult {
	drawCount: number;
	submittedPacketIds: Set<string>;
	/** @internal WebGPU submission-local split counts; public rendering uses Renderer.renderFrame(). */
	readonly staticBatching: Readonly<WebGPUStaticBatchingStats>;
}

export function getDefaultWebGPUDrawBindings(
	draw: WebGPUDrawResources
): readonly WebGPUDrawBinding[] {
	return draw.bindings;
}

export async function submitWebGPUDraws(
	request: WebGPUDrawSubmissionRequest
): Promise<WebGPUDrawSubmissionResult> {
	const submittedPacketIds = new Set<string>();
	const staticBatching: WebGPUStaticBatchingStats = {
		nonAdjacentKeySplits: 0,
		nonContiguousInstanceSplits: 0,
	};
	let seenBatchKeys: Set<string> | null = null;
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
	const lastGroups = new Map<number, {
		group: IBindingGroup;
		offsets: readonly number[] | undefined;
	}>();
	const lastVertexBuffers = new Map<number, IRenderBuffer>();
	let lastIndexBuffer: IRenderBuffer | null = null;
	let lastIndexFormat: IndexFormat | null = null;
	const bindGroup = (
		slot: number,
		group: IBindingGroup,
		offsets?: readonly number[],
	): void => {
		const last = lastGroups.get(slot);
		if (last?.group === group && sameDynamicOffsets(last.offsets, offsets)) return;
		request.encoder.setBindingGroup(slot, group, offsets);
		lastGroups.set(slot, { group, offsets });
	};

	const submitPackets = async (
		packets: DrawPacket[],
		rect: WebGPUDrawSubmissionRect | null
	): Promise<void> => {
		seenBatchKeys?.clear();
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
					previous.packetIds.push(packet.submission.id);
					continue;
				}
				if (draw.staticBatchKey) {
					// Each extra run has one cause; non-adjacency takes precedence over a gap.
					if (previous?.draw.staticBatchKey === draw.staticBatchKey) {
						staticBatching.nonContiguousInstanceSplits++;
					} else if (seenBatchKeys?.has(draw.staticBatchKey)) {
						staticBatching.nonAdjacentKeySplits++;
					}
					(seenBatchKeys ??= new Set<string>()).add(draw.staticBatchKey);
				}
				commands.push({
					packet,
					draw,
					firstInstance,
					instanceCount: 1,
					packetIds: [packet.submission.id],
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
						bindGroup(binding.slot, binding.group, binding.dynamicOffsets);
					}
				} else {
					for (const binding of draw.bindings) {
						bindGroup(binding.slot, binding.group, binding.dynamicOffsets);
					}
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
	} else {
		for (let rectIndex = 0; rectIndex < rects.length; rectIndex++) {
			const rect = rects[rectIndex];
			const packetsInRect = selectedByRect?.[rectIndex] ?? request.packets;
			if (packetsInRect.length <= 0) {
				continue;
			}
			request.encoder.setScissorRect?.(rect.x, rect.y, rect.width, rect.height);
			await submitPackets(packetsInRect, rect);
		}
	}

	const frameStats = request.frameResources.staticBatchingStats;
	if (frameStats) {
		frameStats.nonAdjacentKeySplits += staticBatching.nonAdjacentKeySplits;
		frameStats.nonContiguousInstanceSplits += staticBatching.nonContiguousInstanceSplits;
	}
	return {
		drawCount,
		submittedPacketIds,
		staticBatching,
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
			const preparation = request.resources.getDrawResources(
				task.packet,
				request.frameResources,
				task.options,
			);
			const resources = preparation instanceof Promise ? await preparation : preparation;
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

function sameDynamicOffsets(
	a: readonly number[] | undefined,
	b: readonly number[] | undefined,
): boolean {
	if (a === b) return true;
	if (!a || !b || a.length !== b.length) return false;
	for (let index = 0; index < a.length; index++) {
		if (a[index] !== b[index]) return false;
	}
	return true;
}
