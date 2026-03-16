import type {
	WorkerRuntimeCapabilities,
	WorkerTaskEnvelope,
	WorkerTaskResultEnvelope,
	WorkerTransportEncodedMessage,
	WorkerTransportPlugin,
} from "./types";
import { Platform } from "../foundation/Platform";

interface SharedArrayBufferWirePacket {
	transport: "shared-array-buffer";
	byteLength: number;
	buffer: SharedArrayBuffer;
}

const SHARED_ARRAY_BUFFER_TRANSPORT_TAG = "shared-array-buffer";

let _textEncoder: TextEncoder | null = null;
let _textDecoder: TextDecoder | null = null;

function getTextEncoder(): TextEncoder {
	if (_textEncoder) return _textEncoder;
	_textEncoder = new TextEncoder();
	return _textEncoder;
}

function getTextDecoder(): TextDecoder {
	if (_textDecoder) return _textDecoder;
	_textDecoder = new TextDecoder();
	return _textDecoder;
}

function isObjectRecord(value: unknown): value is Record<string, unknown> {
	return !!value && typeof value === "object";
}

function isSharedArrayBufferWirePacket(
	value: unknown
): value is SharedArrayBufferWirePacket {
	if (!isObjectRecord(value)) return false;
	if (value.transport !== SHARED_ARRAY_BUFFER_TRANSPORT_TAG) return false;
	if (!Number.isFinite(value.byteLength)) return false;
	if (!Platform.hasSharedArrayBuffer()) return false;
	return value.buffer instanceof SharedArrayBuffer;
}

function encodeJsonToSharedPacket(value: unknown): SharedArrayBufferWirePacket {
	if (!Platform.hasSharedArrayBuffer()) {
		throw new Error(
			"SharedArrayBuffer transport is unavailable in this runtime"
		);
	}
	let serialized: string;
	try {
		serialized = JSON.stringify(value) ?? "null";
	} catch (error) {
		throw new Error(
			`SharedArrayBuffer transport failed to serialize payload: ${String(error)}`
		);
	}
	const bytes = getTextEncoder().encode(serialized);
	const buffer = new SharedArrayBuffer(bytes.byteLength);
	new Uint8Array(buffer).set(bytes);
	return {
		transport: SHARED_ARRAY_BUFFER_TRANSPORT_TAG,
		byteLength: bytes.byteLength,
		buffer,
	};
}

function decodeJsonFromSharedPacket(
	packet: SharedArrayBufferWirePacket
): unknown {
	const declaredLength = Math.max(0, Math.floor(packet.byteLength));
	const clampedLength = Math.min(declaredLength, packet.buffer.byteLength);
	const bytes = new Uint8Array(packet.buffer, 0, clampedLength);
	const serialized = getTextDecoder().decode(bytes);
	try {
		return JSON.parse(serialized);
	} catch (error) {
		throw new Error(
			`SharedArrayBuffer transport failed to parse payload: ${String(error)}`
		);
	}
}

function toTaskEnvelope(data: unknown): WorkerTaskEnvelope<unknown> | null {
	if (!isObjectRecord(data)) return null;
	if (!Number.isFinite(data.id)) return null;
	return {
		id: Math.floor(data.id as number),
		payload: data.payload,
	};
}

function toResultEnvelope(
	data: unknown
): WorkerTaskResultEnvelope<unknown> | null {
	if (!isObjectRecord(data)) return null;
	if (!Number.isFinite(data.id)) return null;
	const error =
		typeof data.error === "string" ? data.error
		: data.error === undefined ? undefined
		: String(data.error);
	return {
		id: Math.floor(data.id as number),
		result: data.result,
		error,
	};
}

export function resolveWorkerRuntimeCapabilities(
	overrides: Partial<WorkerRuntimeCapabilities> = {}
): WorkerRuntimeCapabilities {
	const crossOriginIsolated =
		overrides.crossOriginIsolated ??
		Platform.isCrossOriginIsolated(globalThis, true);
	const hasSharedArrayBuffer = Platform.hasSharedArrayBuffer();
	const sharedArrayBufferRequested =
		overrides.sharedArrayBuffer ??
		(hasSharedArrayBuffer && crossOriginIsolated);
	return {
		sharedArrayBuffer: sharedArrayBufferRequested && hasSharedArrayBuffer,
		crossOriginIsolated,
	};
}

export const postMessageWorkerTransportPlugin: WorkerTransportPlugin = {
	id: "post-message",
	mode: "post-message",
	isSupported: () => true,
	encodeTask: (envelope): WorkerTransportEncodedMessage => ({
		message: envelope,
	}),
	decodeTask: (data) => toTaskEnvelope(data),
	encodeResult: (envelope): WorkerTransportEncodedMessage => ({
		message: envelope,
	}),
	decodeResult: (data) => toResultEnvelope(data),
};

export const sharedArrayBufferWorkerTransportPlugin: WorkerTransportPlugin = {
	id: "shared-array-buffer",
	mode: "shared-array-buffer",
	isSupported: (capabilities) => capabilities.sharedArrayBuffer,
	encodeTask: (envelope): WorkerTransportEncodedMessage => ({
		message: encodeJsonToSharedPacket(envelope),
	}),
	decodeTask: (data) => {
		if (!isSharedArrayBufferWirePacket(data)) return null;
		return toTaskEnvelope(decodeJsonFromSharedPacket(data));
	},
	encodeResult: (envelope): WorkerTransportEncodedMessage => ({
		message: encodeJsonToSharedPacket(envelope),
	}),
	decodeResult: (data) => {
		if (!isSharedArrayBufferWirePacket(data)) return null;
		return toResultEnvelope(decodeJsonFromSharedPacket(data));
	},
};

export const DEFAULT_WORKER_TRANSPORT_PLUGINS: WorkerTransportPlugin[] = [
	sharedArrayBufferWorkerTransportPlugin,
	postMessageWorkerTransportPlugin,
];

export function resolveWorkerTransportPlugin(
	candidates: WorkerTransportPlugin[] | undefined,
	overrides: Partial<WorkerRuntimeCapabilities> = {}
): {
	plugin: WorkerTransportPlugin;
	capabilities: WorkerRuntimeCapabilities;
} {
	const capabilities = resolveWorkerRuntimeCapabilities(overrides);
	const plugins =
		candidates && candidates.length > 0 ?
			candidates
		:	DEFAULT_WORKER_TRANSPORT_PLUGINS;
	for (const plugin of plugins) {
		const supported = plugin.isSupported?.(capabilities) ?? true;
		if (!supported) continue;
		return {
			plugin,
			capabilities,
		};
	}
	return {
		plugin: postMessageWorkerTransportPlugin,
		capabilities,
	};
}

export function decodeWorkerTaskEnvelope(
	data: unknown,
	pluginCandidates: WorkerTransportPlugin[] = DEFAULT_WORKER_TRANSPORT_PLUGINS
): WorkerTaskEnvelope<unknown> | null {
	for (const plugin of pluginCandidates) {
		const decoded = plugin.decodeTask(data);
		if (decoded) return decoded;
	}
	return null;
}

export function encodeWorkerTaskResult(
	envelope: WorkerTaskResultEnvelope<unknown>,
	plugin: WorkerTransportPlugin
): WorkerTransportEncodedMessage {
	return plugin.encodeResult(envelope);
}
