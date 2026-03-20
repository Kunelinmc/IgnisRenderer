import {
	postMessageWorkerTransportPlugin,
	sharedArrayBufferWorkerTransportPlugin,
} from "../../../workers/transports";
import type { WorkerTransportPlugin } from "../../../workers/types";
import type {
	SoftwareRasterTileBinEntry,
	SoftwareRasterWorkerTaskPayload,
	SoftwareRasterWorkerTaskResult,
} from "./softwareRasterWorkerProtocol";

interface WorkerEnvelope<TPayload> {
	id: number;
	payload: TPayload;
}

interface WorkerResultEnvelope<TResult> {
	id: number;
	result?: TResult;
	error?: string;
}

type WorkerScopeLike = typeof globalThis & {
	postMessage(message: unknown, transfer?: Transferable[]): void;
	onmessage: ((event: { data: unknown }) => void) | null;
};

const workerScope = globalThis as WorkerScopeLike;

const transportPlugins: WorkerTransportPlugin[] = [
	sharedArrayBufferWorkerTransportPlugin,
	postMessageWorkerTransportPlugin,
];

function decodeEnvelope(
	data: unknown
): {
	plugin: WorkerTransportPlugin;
	envelope: WorkerEnvelope<SoftwareRasterWorkerTaskPayload>;
} | null {
	for (const plugin of transportPlugins) {
		const envelope = plugin.decodeTask(data);
		if (!envelope) continue;
		return {
			plugin,
			envelope: envelope as WorkerEnvelope<SoftwareRasterWorkerTaskPayload>,
		};
	}
	return null;
}

function encodeResult(
	plugin: WorkerTransportPlugin,
	envelope: WorkerResultEnvelope<SoftwareRasterWorkerTaskResult>
): {
	message: unknown;
	transfer?: Transferable[];
} {
	return plugin.encodeResult(envelope);
}

function executeTask(
	payload: SoftwareRasterWorkerTaskPayload
): SoftwareRasterWorkerTaskResult {
	if (payload.type !== "bin-main-pass") {
		throw new Error(`Unknown software raster worker task "${payload}"`);
	}

	const tileSize = Math.max(1, Math.floor(payload.tileSize));
	const tileColumns = Math.max(1, Math.ceil(payload.width / tileSize));
	const tileRows = Math.max(1, Math.ceil(payload.height / tileSize));
	const maxTileX = tileColumns - 1;
	const maxTileY = tileRows - 1;

	const bins = new Map<number, number[]>();

	const start = Math.max(0, Math.floor(payload.startIndex));
	const end = Math.min(
		payload.triangleBounds.length,
		Math.max(start, Math.floor(payload.endIndex))
	);

	for (let triangleIndex = start; triangleIndex < end; triangleIndex++) {
		const bounds = payload.triangleBounds[triangleIndex];
		if (!bounds) continue;
		if (bounds.maxTileX < bounds.minTileX || bounds.maxTileY < bounds.minTileY) {
			continue;
		}

		const minTileX = Math.max(0, Math.min(maxTileX, bounds.minTileX));
		const minTileY = Math.max(0, Math.min(maxTileY, bounds.minTileY));
		const maxTileXClamped = Math.max(0, Math.min(maxTileX, bounds.maxTileX));
		const maxTileYClamped = Math.max(0, Math.min(maxTileY, bounds.maxTileY));

		for (let tileY = minTileY; tileY <= maxTileYClamped; tileY++) {
			const rowOffset = tileY * tileColumns;
			for (let tileX = minTileX; tileX <= maxTileXClamped; tileX++) {
				const tileIndex = rowOffset + tileX;
				let bucket = bins.get(tileIndex);
				if (!bucket) {
					bucket = [];
					bins.set(tileIndex, bucket);
				}
				bucket.push(triangleIndex);
			}
		}
	}

	const entries: SoftwareRasterTileBinEntry[] = [...bins.entries()]
		.sort((left, right) => left[0] - right[0])
		.map(([tileIndex, triangleIndices]) => ({
			tileIndex,
			triangleIndices,
		}));

	return {
		type: "bin-main-pass",
		bins: entries,
	};
}

workerScope.onmessage = (event) => {
	void (async () => {
		const decoded = decodeEnvelope(event.data);
		if (!decoded) return;
		const { plugin, envelope } = decoded;

		try {
			const result = executeTask(envelope.payload);
			const encoded = encodeResult(plugin, {
				id: envelope.id,
				result,
			});
			workerScope.postMessage(
				encoded.message,
				encoded.transfer && encoded.transfer.length > 0 ?
					encoded.transfer
				:	undefined
			);
		} catch (error) {
			const encoded = encodeResult(plugin, {
				id: envelope.id,
				error: error instanceof Error ? error.message : String(error),
			});
			workerScope.postMessage(
				encoded.message,
				encoded.transfer && encoded.transfer.length > 0 ?
					encoded.transfer
				:	undefined
			);
		}
	})();
};
