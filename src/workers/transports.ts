import type {
	WorkerRuntimeCapabilities,
	WorkerTaskEnvelope,
	WorkerTaskResultEnvelope,
	WorkerTransportEncodedMessage,
	WorkerTransportPlugin,
} from "./types";
import { Platform } from "../foundation/Platform";

type SharedArrayBufferPacketKind = "task" | "result";

interface SharedArrayBufferWirePacket {
	transport: "shared-array-buffer";
	kind: SharedArrayBufferPacketKind;
	byteLength: number;
	buffer: SharedArrayBuffer;
}

const SHARED_ARRAY_BUFFER_TRANSPORT_TAG = "shared-array-buffer";
const SHARED_PACKET_MAGIC = 0x53414250;
const SHARED_PACKET_VERSION = 1;
const SHARED_PACKET_KIND_TASK = 1;
const SHARED_PACKET_KIND_RESULT = 2;
const RESULT_FLAG_HAS_RESULT = 1 << 0;
const RESULT_FLAG_HAS_ERROR = 1 << 1;

const VALUE_TAG_NULL = 0;
const VALUE_TAG_UNDEFINED = 1;
const VALUE_TAG_FALSE = 2;
const VALUE_TAG_TRUE = 3;
const VALUE_TAG_NUMBER = 4;
const VALUE_TAG_STRING = 5;
const VALUE_TAG_ARRAY = 6;
const VALUE_TAG_OBJECT = 7;

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

class BinaryBufferWriter {
	private _bytes: Uint8Array;
	private _view: DataView;
	private _offset: number;

	public constructor(initialCapacity: number = 128) {
		const capacity = Math.max(16, Math.floor(initialCapacity));
		const buffer = new ArrayBuffer(capacity);
		this._bytes = new Uint8Array(buffer);
		this._view = new DataView(buffer);
		this._offset = 0;
	}

	public writeUint8(value: number): void {
		this._ensureCapacity(1);
		this._view.setUint8(this._offset, value & 0xff);
		this._offset += 1;
	}

	public writeUint16(value: number): void {
		this._ensureCapacity(2);
		this._view.setUint16(this._offset, value & 0xffff, true);
		this._offset += 2;
	}

	public writeUint32(value: number): void {
		this._ensureCapacity(4);
		this._view.setUint32(this._offset, value >>> 0, true);
		this._offset += 4;
	}

	public writeFloat64(value: number): void {
		this._ensureCapacity(8);
		this._view.setFloat64(this._offset, value, true);
		this._offset += 8;
	}

	public writeBytes(bytes: Uint8Array): void {
		this._ensureCapacity(bytes.byteLength);
		this._bytes.set(bytes, this._offset);
		this._offset += bytes.byteLength;
	}

	public toSharedArrayBuffer(): {
		buffer: SharedArrayBuffer;
		byteLength: number;
	} {
		const buffer = new SharedArrayBuffer(this._offset);
		const bytes = new Uint8Array(buffer);
		bytes.set(this._bytes.subarray(0, this._offset));
		return {
			buffer,
			byteLength: this._offset,
		};
	}

	private _ensureCapacity(additionalByteLength: number): void {
		const required = this._offset + additionalByteLength;
		if (required <= this._bytes.byteLength) return;
		let nextCapacity = this._bytes.byteLength;
		while (nextCapacity < required) {
			nextCapacity *= 2;
		}
		const nextBuffer = new ArrayBuffer(nextCapacity);
		const nextBytes = new Uint8Array(nextBuffer);
		nextBytes.set(this._bytes, 0);
		this._bytes = nextBytes;
		this._view = new DataView(nextBuffer);
	}
}

class BinaryBufferReader {
	private _view: DataView;
	private _bytes: Uint8Array;
	private _offset: number;
	private _byteLength: number;

	public constructor(buffer: SharedArrayBuffer, byteLength: number) {
		this._byteLength = normalizeByteLength(byteLength, buffer.byteLength);
		this._view = new DataView(buffer);
		this._bytes = new Uint8Array(buffer);
		this._offset = 0;
	}

	public readUint8(): number {
		this._ensureReadable(1);
		const value = this._view.getUint8(this._offset);
		this._offset += 1;
		return value;
	}

	public readUint16(): number {
		this._ensureReadable(2);
		const value = this._view.getUint16(this._offset, true);
		this._offset += 2;
		return value;
	}

	public readUint32(): number {
		this._ensureReadable(4);
		const value = this._view.getUint32(this._offset, true);
		this._offset += 4;
		return value;
	}

	public readFloat64(): number {
		this._ensureReadable(8);
		const value = this._view.getFloat64(this._offset, true);
		this._offset += 8;
		return value;
	}

	public readBytes(length: number): Uint8Array {
		const byteLength = Math.max(0, Math.floor(length));
		this._ensureReadable(byteLength);
		const start = this._offset;
		const end = this._offset + byteLength;
		this._offset = end;
		return this._bytes.subarray(start, end);
	}

	public getRemainingByteLength(): number {
		return this._byteLength - this._offset;
	}

	private _ensureReadable(byteLength: number): void {
		if (this._offset + byteLength <= this._byteLength) return;
		throw new Error(
			`SharedArrayBuffer packet truncated at byte offset ${this._offset}`
		);
	}
}

function normalizeByteLength(byteLength: number, maxByteLength: number): number {
	if (!Number.isFinite(byteLength)) return 0;
	const normalized = Math.max(0, Math.floor(byteLength));
	return Math.min(normalized, Math.max(0, Math.floor(maxByteLength)));
}

function isSharedArrayBufferPacketKind(
	value: unknown
): value is SharedArrayBufferPacketKind {
	return value === "task" || value === "result";
}

function isSharedArrayBufferWirePacket(
	value: unknown
): value is SharedArrayBufferWirePacket {
	if (!isObjectRecord(value)) return false;
	if (value.transport !== SHARED_ARRAY_BUFFER_TRANSPORT_TAG) return false;
	if (!isSharedArrayBufferPacketKind(value.kind)) return false;
	if (!Number.isFinite(value.byteLength)) return false;
	if (!Platform.hasSharedArrayBuffer()) return false;
	return value.buffer instanceof SharedArrayBuffer;
}

function assertSharedArrayBufferTransportAvailable(): void {
	if (!Platform.hasSharedArrayBuffer()) {
		throw new Error(
			"SharedArrayBuffer transport is unavailable in this runtime"
		);
	}
}

function normalizeEnvelopeId(id: number): number {
	if (!Number.isFinite(id)) {
		throw new Error(
			`SharedArrayBuffer transport received an invalid envelope id: ${String(id)}`
		);
	}
	const normalized = Math.floor(id);
	if (normalized < 0) {
		throw new Error(
			`SharedArrayBuffer transport requires non-negative envelope ids, got ${normalized}`
		);
	}
	return normalized >>> 0;
}

function isPlainObjectRecord(value: unknown): value is Record<string, unknown> {
	if (!isObjectRecord(value)) return false;
	const prototype = Object.getPrototypeOf(value);
	return prototype === Object.prototype || prototype === null;
}

function writeString(writer: BinaryBufferWriter, value: string): void {
	const bytes = getTextEncoder().encode(value);
	writer.writeUint32(bytes.byteLength);
	writer.writeBytes(bytes);
}

function toTextDecodableBytes(bytes: Uint8Array): Uint8Array {
	if (!Platform.hasSharedArrayBuffer()) return bytes;
	if (!(bytes.buffer instanceof SharedArrayBuffer)) return bytes;
	const copy = new Uint8Array(bytes.byteLength);
	copy.set(bytes);
	return copy;
}

function readString(reader: BinaryBufferReader): string {
	const byteLength = reader.readUint32();
	const bytes = reader.readBytes(byteLength);
	return getTextDecoder().decode(toTextDecodableBytes(bytes));
}

function writeStructuredValue(
	writer: BinaryBufferWriter,
	value: unknown,
	activeValues: Set<object>
): void {
	if (value === null) {
		writer.writeUint8(VALUE_TAG_NULL);
		return;
	}

	switch (typeof value) {
		case "undefined":
			writer.writeUint8(VALUE_TAG_UNDEFINED);
			return;
		case "boolean":
			writer.writeUint8(value ? VALUE_TAG_TRUE : VALUE_TAG_FALSE);
			return;
		case "number":
			writer.writeUint8(VALUE_TAG_NUMBER);
			writer.writeFloat64(value);
			return;
		case "string":
			writer.writeUint8(VALUE_TAG_STRING);
			writeString(writer, value);
			return;
		case "object":
			break;
		default:
			throw new Error(
				`Unsupported structured value type: ${typeof value}`
			);
	}

	if (Array.isArray(value)) {
		if (activeValues.has(value)) {
			throw new Error("Circular references are not supported in SAB payloads");
		}
		activeValues.add(value);
		writer.writeUint8(VALUE_TAG_ARRAY);
		writer.writeUint32(value.length);
		for (const entry of value) {
			writeStructuredValue(writer, entry, activeValues);
		}
		activeValues.delete(value);
		return;
	}

	if (!isPlainObjectRecord(value)) {
		const descriptor = Object.prototype.toString.call(value);
		throw new Error(
			`Unsupported structured object type in SAB payload: ${descriptor}`
		);
	}
	if (activeValues.has(value)) {
		throw new Error("Circular references are not supported in SAB payloads");
	}
	activeValues.add(value);
	writer.writeUint8(VALUE_TAG_OBJECT);
	const entries = Object.entries(value);
	writer.writeUint32(entries.length);
	for (const [key, entryValue] of entries) {
		writeString(writer, key);
		writeStructuredValue(writer, entryValue, activeValues);
	}
	activeValues.delete(value);
}

function readStructuredValue(reader: BinaryBufferReader): unknown {
	const tag = reader.readUint8();
	switch (tag) {
		case VALUE_TAG_NULL:
			return null;
		case VALUE_TAG_UNDEFINED:
			return undefined;
		case VALUE_TAG_FALSE:
			return false;
		case VALUE_TAG_TRUE:
			return true;
		case VALUE_TAG_NUMBER:
			return reader.readFloat64();
		case VALUE_TAG_STRING:
			return readString(reader);
		case VALUE_TAG_ARRAY: {
			const length = reader.readUint32();
			const array: unknown[] = new Array(length);
			for (let i = 0; i < length; i++) {
				array[i] = readStructuredValue(reader);
			}
			return array;
		}
		case VALUE_TAG_OBJECT: {
			const entryCount = reader.readUint32();
			const objectValue: Record<string, unknown> = {};
			for (let i = 0; i < entryCount; i++) {
				const key = readString(reader);
				objectValue[key] = readStructuredValue(reader);
			}
			return objectValue;
		}
		default:
			throw new Error(`Unsupported SAB value tag: ${tag}`);
	}
}

function encodeSharedArrayBufferTaskEnvelope(
	envelope: WorkerTaskEnvelope<unknown>
): SharedArrayBufferWirePacket {
	assertSharedArrayBufferTransportAvailable();
	try {
		const writer = new BinaryBufferWriter();
		writer.writeUint32(SHARED_PACKET_MAGIC);
		writer.writeUint16(SHARED_PACKET_VERSION);
		writer.writeUint8(SHARED_PACKET_KIND_TASK);
		writer.writeUint8(0);
		writer.writeUint32(normalizeEnvelopeId(envelope.id));
		writeStructuredValue(writer, envelope.payload, new Set());
		const packet = writer.toSharedArrayBuffer();
		return {
			transport: SHARED_ARRAY_BUFFER_TRANSPORT_TAG,
			kind: "task",
			byteLength: packet.byteLength,
			buffer: packet.buffer,
		};
	} catch (error) {
		throw new Error(
			`SharedArrayBuffer transport failed to encode task payload: ${String(error)}`
		);
	}
}

function encodeSharedArrayBufferResultEnvelope(
	envelope: WorkerTaskResultEnvelope<unknown>
): SharedArrayBufferWirePacket {
	assertSharedArrayBufferTransportAvailable();
	try {
		const writer = new BinaryBufferWriter();
		const hasResult = Object.prototype.hasOwnProperty.call(envelope, "result");
		const rawError = (envelope as { error?: unknown }).error;
		const hasError = rawError !== undefined;
		const normalizedError =
			!hasError ? undefined
			: typeof rawError === "string" ? rawError
			: String(rawError);
		let flags = 0;
		if (hasResult) flags |= RESULT_FLAG_HAS_RESULT;
		if (hasError) flags |= RESULT_FLAG_HAS_ERROR;

		writer.writeUint32(SHARED_PACKET_MAGIC);
		writer.writeUint16(SHARED_PACKET_VERSION);
		writer.writeUint8(SHARED_PACKET_KIND_RESULT);
		writer.writeUint8(flags);
		writer.writeUint32(normalizeEnvelopeId(envelope.id));
		if (hasResult) {
			writeStructuredValue(
				writer,
				(envelope as { result?: unknown }).result,
				new Set()
			);
		}
		if (hasError) {
			writeString(writer, normalizedError!);
		}

		const packet = writer.toSharedArrayBuffer();
		return {
			transport: SHARED_ARRAY_BUFFER_TRANSPORT_TAG,
			kind: "result",
			byteLength: packet.byteLength,
			buffer: packet.buffer,
		};
	} catch (error) {
		throw new Error(
			`SharedArrayBuffer transport failed to encode result payload: ${String(error)}`
		);
	}
}

function decodeSharedArrayBufferTaskEnvelope(
	packet: SharedArrayBufferWirePacket
): WorkerTaskEnvelope<unknown> | null {
	if (packet.kind !== "task") return null;
	try {
		const reader = new BinaryBufferReader(packet.buffer, packet.byteLength);
		const magic = reader.readUint32();
		if (magic !== SHARED_PACKET_MAGIC) {
			throw new Error("Invalid packet magic");
		}
		const version = reader.readUint16();
		if (version !== SHARED_PACKET_VERSION) {
			throw new Error(`Unsupported packet version: ${version}`);
		}
		const kind = reader.readUint8();
		if (kind !== SHARED_PACKET_KIND_TASK) return null;
		reader.readUint8(); // flags (reserved for future use)
		const id = reader.readUint32();
		const payload = readStructuredValue(reader);
		if (reader.getRemainingByteLength() !== 0) {
			throw new Error(
				`Unexpected trailing bytes: ${reader.getRemainingByteLength()}`
			);
		}
		return {
			id,
			payload,
		};
	} catch (error) {
		throw new Error(
			`SharedArrayBuffer transport failed to decode task payload: ${String(error)}`
		);
	}
}

function decodeSharedArrayBufferResultEnvelope(
	packet: SharedArrayBufferWirePacket
): WorkerTaskResultEnvelope<unknown> | null {
	if (packet.kind !== "result") return null;
	try {
		const reader = new BinaryBufferReader(packet.buffer, packet.byteLength);
		const magic = reader.readUint32();
		if (magic !== SHARED_PACKET_MAGIC) {
			throw new Error("Invalid packet magic");
		}
		const version = reader.readUint16();
		if (version !== SHARED_PACKET_VERSION) {
			throw new Error(`Unsupported packet version: ${version}`);
		}
		const kind = reader.readUint8();
		if (kind !== SHARED_PACKET_KIND_RESULT) return null;
		const flags = reader.readUint8();
		const id = reader.readUint32();

		const hasResult = (flags & RESULT_FLAG_HAS_RESULT) !== 0;
		const hasError = (flags & RESULT_FLAG_HAS_ERROR) !== 0;
		const result = hasResult ? readStructuredValue(reader) : undefined;
		const error = hasError ? readString(reader) : undefined;

		if (reader.getRemainingByteLength() !== 0) {
			throw new Error(
				`Unexpected trailing bytes: ${reader.getRemainingByteLength()}`
			);
		}
		return {
			id,
			result,
			error,
		};
	} catch (error) {
		throw new Error(
			`SharedArrayBuffer transport failed to decode result payload: ${String(error)}`
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
		message: encodeSharedArrayBufferTaskEnvelope(envelope),
	}),
	decodeTask: (data) => {
		if (!isSharedArrayBufferWirePacket(data)) return null;
		return decodeSharedArrayBufferTaskEnvelope(data);
	},
	encodeResult: (envelope): WorkerTransportEncodedMessage => ({
		message: encodeSharedArrayBufferResultEnvelope(envelope),
	}),
	decodeResult: (data) => {
		if (!isSharedArrayBufferWirePacket(data)) return null;
		return decodeSharedArrayBufferResultEnvelope(data);
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
