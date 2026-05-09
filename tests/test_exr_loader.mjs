import assert from "node:assert/strict";
import { deflateSync } from "node:zlib";

import { Scene } from "../src/core/Scene.ts";
import { Loader } from "../src/loaders/Loader.ts";
import { EXRLoader } from "../src/loaders/EXRLoader.ts";

const COMPRESSION_NONE = 0;
const COMPRESSION_ZIP = 3;
const PIXEL_HALF = 1;
const PIXEL_FLOAT = 2;

function nearlyEqual(left, right, epsilon = 1e-4) {
	return Math.abs(left - right) <= epsilon;
}

function assertNearly(left, right, message) {
	assert.ok(nearlyEqual(left, right), `${message}: ${left} !== ${right}`);
}

function testParseUncompressedHalfRGB() {
	const bytes = createScanlineEXR({
		width: 2,
		height: 2,
		compression: COMPRESSION_NONE,
		channels: [
			{ name: "B", pixelType: PIXEL_HALF, sample: (x, y) => 0.25 + x + y },
			{ name: "G", pixelType: PIXEL_HALF, sample: (x, y) => 0.5 + x + y },
			{ name: "R", pixelType: PIXEL_HALF, sample: (x, y) => 1 + x + y },
		],
	});
	const texture = new EXRLoader().parse(toArrayBuffer(bytes));

	assert.equal(texture.width, 2);
	assert.equal(texture.height, 2);
	assert.equal(texture.colorSpace, "HDR");
	assert.ok(texture.data instanceof Float32Array);
	assert.equal(texture.wrapS, "Repeat");
	assert.equal(texture.wrapT, "Clamp");

	assertNearly(texture.data[0], 1, "pixel 0 R");
	assertNearly(texture.data[1], 0.5, "pixel 0 G");
	assertNearly(texture.data[2], 0.25, "pixel 0 B");
	assertNearly(texture.data[3], 1, "pixel 0 A default");

	const pixel3 = (1 * 2 + 1) * 4;
	assertNearly(texture.data[pixel3], 3, "pixel 3 R");
	assertNearly(texture.data[pixel3 + 1], 2.5, "pixel 3 G");
	assertNearly(texture.data[pixel3 + 2], 2.25, "pixel 3 B");
	assertNearly(texture.data[pixel3 + 3], 1, "pixel 3 A default");
}

function testParseLayeredFloatChannelsWithDefaultAlpha() {
	const bytes = createScanlineEXR({
		width: 1,
		height: 1,
		compression: COMPRESSION_NONE,
		channels: [
			{ name: "beauty.B", pixelType: PIXEL_FLOAT, sample: () => 0.125 },
			{ name: "beauty.G", pixelType: PIXEL_FLOAT, sample: () => 0.25 },
			{ name: "beauty.R", pixelType: PIXEL_FLOAT, sample: () => 0.5 },
		],
	});
	const texture = new EXRLoader().parse(toArrayBuffer(bytes), {
		defaultAlpha: 0.75,
	});

	assertNearly(texture.data[0], 0.5, "layered R");
	assertNearly(texture.data[1], 0.25, "layered G");
	assertNearly(texture.data[2], 0.125, "layered B");
	assertNearly(texture.data[3], 0.75, "layered default alpha");
}

async function testZipParseAndEnvironmentAssignment() {
	Loader.clearSharedCache();

	const bytes = createScanlineEXR({
		width: 2,
		height: 2,
		compression: COMPRESSION_ZIP,
		channels: [
			{ name: "A", pixelType: PIXEL_HALF, sample: () => 0.875 },
			{ name: "B", pixelType: PIXEL_HALF, sample: (x, y) => 0.125 + x + y },
			{ name: "G", pixelType: PIXEL_HALF, sample: (x, y) => 0.25 + x + y },
			{ name: "R", pixelType: PIXEL_HALF, sample: (x, y) => 0.5 + x + y },
		],
	});
	const loader = new EXRLoader();
	const scene = new Scene();
	const originalFetch = globalThis.fetch;

	try {
		globalThis.fetch = async () => ({
			ok: true,
			statusText: "OK",
			headers: {
				get() {
					return null;
				},
			},
			body: null,
			async arrayBuffer() {
				return toArrayBuffer(bytes);
			},
		});

		const texture = await loader.loadEnvironment(
			"https://example.com/env.exr",
			scene
		);

		assert.equal(scene.environment.backgroundTexture, texture);
		assert.equal(scene.environment.iblTexture, texture);
		assert.equal(texture.isLoadErrorFallback, false);
		assertNearly(texture.data[0], 0.5, "zip R");
		assertNearly(texture.data[1], 0.25, "zip G");
		assertNearly(texture.data[2], 0.125, "zip B");
		assertNearly(texture.data[3], 0.875, "zip A");
	} finally {
		globalThis.fetch = originalFetch;
	}
}

function testSyncParseRejectsZipCompression() {
	const bytes = createScanlineEXR({
		width: 1,
		height: 1,
		compression: COMPRESSION_ZIP,
		channels: [
			{ name: "B", pixelType: PIXEL_HALF, sample: () => 0 },
			{ name: "G", pixelType: PIXEL_HALF, sample: () => 0 },
			{ name: "R", pixelType: PIXEL_HALF, sample: () => 0 },
		],
	});

	assert.throws(
		() => new EXRLoader().parse(toArrayBuffer(bytes)),
		/parseAsync\(\) or load\(\)/
	);
}

function createScanlineEXR({ width, height, compression, channels }) {
	const sortedChannels = [...channels].sort((left, right) =>
		left.name.localeCompare(right.name)
	);
	const header = [];
	pushU32(header, 20000630);
	pushU32(header, 2);
	pushAttribute(header, "channels", "chlist", createChannelList(sortedChannels));
	pushAttribute(header, "compression", "compression", [compression]);
	pushAttribute(header, "dataWindow", "box2i", createBox2i(0, 0, width - 1, height - 1));
	pushAttribute(header, "displayWindow", "box2i", createBox2i(0, 0, width - 1, height - 1));
	pushAttribute(header, "lineOrder", "lineOrder", [0]);
	pushAttribute(header, "pixelAspectRatio", "float", createFloat32Payload([1]));
	pushAttribute(header, "screenWindowCenter", "v2f", createFloat32Payload([0, 0]));
	pushAttribute(header, "screenWindowWidth", "float", createFloat32Payload([1]));
	header.push(0);

	const rowsPerChunk = compression === COMPRESSION_ZIP ? 16 : 1;
	const chunks = [];
	for (let y = 0; y < height; y += rowsPerChunk) {
		const rowCount = Math.min(rowsPerChunk, height - y);
		const raw = createPixelBlock(width, y, rowCount, sortedChannels);
		const payload = compression === COMPRESSION_ZIP ? zipCompressEXR(raw) : raw;
		const chunk = [];
		pushI32(chunk, y);
		pushU32(chunk, payload.length);
		pushBytes(chunk, payload);
		chunks.push(new Uint8Array(chunk));
	}

	const offsetTable = [];
	let chunkOffset = header.length + chunks.length * 8;
	for (const chunk of chunks) {
		pushU64(offsetTable, chunkOffset);
		chunkOffset += chunk.length;
	}

	return concatBytes(new Uint8Array(header), new Uint8Array(offsetTable), ...chunks);
}

function createChannelList(channels) {
	const bytes = [];
	for (const channel of channels) {
		pushCString(bytes, channel.name);
		pushI32(bytes, channel.pixelType);
		bytes.push(0, 0, 0, 0);
		pushI32(bytes, 1);
		pushI32(bytes, 1);
	}
	bytes.push(0);
	return bytes;
}

function createBox2i(xMin, yMin, xMax, yMax) {
	const bytes = [];
	pushI32(bytes, xMin);
	pushI32(bytes, yMin);
	pushI32(bytes, xMax);
	pushI32(bytes, yMax);
	return bytes;
}

function createFloat32Payload(values) {
	const bytes = [];
	for (const value of values) {
		pushF32(bytes, value);
	}
	return bytes;
}

function createPixelBlock(width, yStart, rowCount, channels) {
	const bytes = [];
	for (let localY = 0; localY < rowCount; localY++) {
		const y = yStart + localY;
		for (const channel of channels) {
			for (let x = 0; x < width; x++) {
				const value = channel.sample(x, y);
				if (channel.pixelType === PIXEL_HALF) {
					pushU16(bytes, float32ToFloat16Bits(value));
				} else {
					pushF32(bytes, value);
				}
			}
		}
	}
	return new Uint8Array(bytes);
}

function zipCompressEXR(raw) {
	const predicted = new Uint8Array(raw);
	for (let i = predicted.length - 1; i > 0; i--) {
		predicted[i] = (predicted[i] - predicted[i - 1] + 128) & 0xff;
	}
	const interleaved = new Uint8Array(predicted.length);
	let evenOffset = 0;
	let oddOffset = (predicted.length + 1) >> 1;
	for (let i = 0; i < predicted.length; i++) {
		if ((i & 1) === 0) {
			interleaved[evenOffset++] = predicted[i];
		} else {
			interleaved[oddOffset++] = predicted[i];
		}
	}
	return new Uint8Array(deflateSync(interleaved));
}

function pushAttribute(bytes, name, type, payload) {
	pushCString(bytes, name);
	pushCString(bytes, type);
	pushI32(bytes, payload.length);
	pushBytes(bytes, payload);
}

function pushCString(bytes, value) {
	for (let i = 0; i < value.length; i++) {
		bytes.push(value.charCodeAt(i));
	}
	bytes.push(0);
}

function pushBytes(bytes, value) {
	for (const byte of value) {
		bytes.push(byte);
	}
}

function pushU16(bytes, value) {
	bytes.push(value & 0xff, (value >> 8) & 0xff);
}

function pushU32(bytes, value) {
	bytes.push(
		value & 0xff,
		(value >> 8) & 0xff,
		(value >> 16) & 0xff,
		(value >> 24) & 0xff
	);
}

function pushI32(bytes, value) {
	pushU32(bytes, value >>> 0);
}

function pushU64(bytes, value) {
	pushU32(bytes, value >>> 0);
	pushU32(bytes, Math.floor(value / 2 ** 32));
}

function pushF32(bytes, value) {
	const buffer = new ArrayBuffer(4);
	new DataView(buffer).setFloat32(0, value, true);
	pushBytes(bytes, new Uint8Array(buffer));
}

function concatBytes(...chunks) {
	const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
	const result = new Uint8Array(total);
	let offset = 0;
	for (const chunk of chunks) {
		result.set(chunk, offset);
		offset += chunk.length;
	}
	return result;
}

function toArrayBuffer(bytes) {
	return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

const FLOAT32_TO_FLOAT16_FLOAT = new Float32Array(1);
const FLOAT32_TO_FLOAT16_INT = new Int32Array(FLOAT32_TO_FLOAT16_FLOAT.buffer);

function float32ToFloat16Bits(value) {
	FLOAT32_TO_FLOAT16_FLOAT[0] = value;
	const bits = FLOAT32_TO_FLOAT16_INT[0];
	const sign = (bits >> 16) & 0x8000;
	let exponent = ((bits >> 23) & 0xff) - 127 + 15;
	let mantissa = bits & 0x7fffff;

	if (exponent <= 0) {
		if (exponent < -10) {
			return sign;
		}
		mantissa = (mantissa | 0x800000) >> (1 - exponent);
		return sign | ((mantissa + 0x1000) >> 13);
	}

	if (exponent >= 31) {
		return sign | 0x7bff;
	}

	let halfMantissa = (mantissa + 0x1000) >> 13;
	if (halfMantissa === 0x400) {
		halfMantissa = 0;
		exponent++;
		if (exponent >= 31) {
			return sign | 0x7bff;
		}
	}

	return sign | (exponent << 10) | halfMantissa;
}

async function run() {
	testParseUncompressedHalfRGB();
	testParseLayeredFloatChannelsWithDefaultAlpha();
	await testZipParseAndEnvironmentAssignment();
	testSyncParseRejectsZipCompression();
	console.log("EXR loader tests passed");
}

await run();
