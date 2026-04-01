import { Texture } from "../../core/Texture";
import { prefilterEnvMapMipLevel } from "../lightProbeBakeCore";
import type {
	LightProbeBakeWorkerTaskPayload,
	LightProbeBakeWorkerTaskResult,
} from "./lightProbeBakeWorkerProtocol";

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

function isWorkerEnvelope(
	value: unknown
): value is WorkerEnvelope<LightProbeBakeWorkerTaskPayload> {
	if (!value || typeof value !== "object") return false;
	const candidate = value as WorkerEnvelope<LightProbeBakeWorkerTaskPayload>;
	return (
		typeof candidate.id === "number" &&
		!!candidate.payload &&
		typeof candidate.payload === "object"
	);
}

function executeTask(
	payload: LightProbeBakeWorkerTaskPayload
): LightProbeBakeWorkerTaskResult {
	if (payload.type !== "prefilter-mip") {
		throw new Error(`Unknown light probe bake task "${payload}"`);
	}

	const env = payload.envMap;
	const envMap = new Texture(env.data, env.width, env.height, env.colorSpace);
	const mip = prefilterEnvMapMipLevel(
		envMap,
		payload.level,
		payload.baseWidth,
		payload.baseHeight,
		payload.maxMipLevels
	);

	return {
		type: "prefilter-mip",
		level: mip.level,
		width: mip.width,
		height: mip.height,
		data: mip.data,
	};
}

workerScope.onmessage = (event) => {
	const envelope = event.data;
	if (!isWorkerEnvelope(envelope)) return;

	try {
		const result = executeTask(envelope.payload);
		const response: WorkerResultEnvelope<LightProbeBakeWorkerTaskResult> = {
			id: envelope.id,
			result,
		};
		workerScope.postMessage(response);
	} catch (error) {
		const response: WorkerResultEnvelope<LightProbeBakeWorkerTaskResult> = {
			id: envelope.id,
			error: error instanceof Error ? error.message : String(error),
		};
		workerScope.postMessage(response);
	}
};
