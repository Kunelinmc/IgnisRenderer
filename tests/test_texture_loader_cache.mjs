import assert from "node:assert/strict";
import { Loader } from "../src/loaders/Loader.ts";
import { TextureLoader } from "../src/loaders/TextureLoader.ts";
import { Texture } from "../src/core/Texture.ts";

const TEST_URL = "https://example.com/shared-texture.png";

function createResponse(bytes) {
	return {
		ok: true,
		statusText: "OK",
		headers: {
			get() {
				return null;
			},
		},
		body: null,
		async arrayBuffer() {
			return bytes.buffer.slice(
				bytes.byteOffset,
				bytes.byteOffset + bytes.byteLength
			);
		},
	};
}

function createTexture(marker) {
	return new Texture(new Uint8ClampedArray([marker, 0, 0, 255]), 1, 1);
}

async function testSharedCacheAcrossInstances() {
	Loader.clearSharedCache();

	let fetchCount = 0;
	const bytes = new Uint8Array([1, 2, 3, 4]);

	const originalFetch = globalThis.fetch;
	const originalCreateObjectURL = URL.createObjectURL;
	const originalRevokeObjectURL = URL.revokeObjectURL;

	try {
		globalThis.fetch = async () => {
			fetchCount++;
			return createResponse(bytes);
		};
		URL.createObjectURL = () => "blob:ignis-cache-test";
		URL.revokeObjectURL = () => {};

		let decodeCountA = 0;
		let decodeCountB = 0;
		const loaderA = new TextureLoader();
		const loaderB = new TextureLoader();

		loaderA._loadImage = async () => {
			decodeCountA++;
			return createTexture(64);
		};
		loaderB._loadImage = async () => {
			decodeCountB++;
			return createTexture(128);
		};

		const first = await loaderA.load(TEST_URL);
		const second = await loaderB.load(TEST_URL);

		assert.equal(fetchCount, 1);
		assert.equal(decodeCountA, 1);
		assert.equal(decodeCountB, 0);
		assert.equal(first, second);
	} finally {
		globalThis.fetch = originalFetch;
		URL.createObjectURL = originalCreateObjectURL;
		URL.revokeObjectURL = originalRevokeObjectURL;
	}
}

async function testInFlightDeduplication() {
	Loader.clearSharedCache();

	let fetchCount = 0;
	let decodeCount = 0;
	const bytes = new Uint8Array([8, 6, 7, 5]);

	const originalFetch = globalThis.fetch;
	const originalCreateObjectURL = URL.createObjectURL;
	const originalRevokeObjectURL = URL.revokeObjectURL;

	try {
		globalThis.fetch = async () => {
			fetchCount++;
			await new Promise((resolve) => setTimeout(resolve, 20));
			return createResponse(bytes);
		};
		URL.createObjectURL = () => "blob:ignis-cache-test-flight";
		URL.revokeObjectURL = () => {};

		const loaderA = new TextureLoader();
		const loaderB = new TextureLoader();

		const decode = async () => {
			decodeCount++;
			return createTexture(255);
		};
		loaderA._loadImage = decode;
		loaderB._loadImage = decode;

		const [first, second] = await Promise.all([
			loaderA.load(TEST_URL),
			loaderB.load(TEST_URL),
		]);

		assert.equal(fetchCount, 1);
		assert.equal(decodeCount, 1);
		assert.equal(first, second);
	} finally {
		globalThis.fetch = originalFetch;
		URL.createObjectURL = originalCreateObjectURL;
		URL.revokeObjectURL = originalRevokeObjectURL;
	}
}

async function testLoadErrorFallbackIsTagged() {
	Loader.clearSharedCache();

	const loader = new TextureLoader();
	loader._fetchWithProgress = async () => {
		throw new Error("network failed");
	};

	const originalConsoleError = console.error;
	try {
		console.error = () => {};
		const texture = await loader.load("https://example.com/fail.png");
		assert.equal(texture.isLoadErrorFallback, true);
		assert.deepEqual(Array.from(texture.data ?? []), [255, 0, 255, 255]);
	} finally {
		console.error = originalConsoleError;
	}
}

async function run() {
	await testSharedCacheAcrossInstances();
	await testInFlightDeduplication();
	await testLoadErrorFallbackIsTagged();
	console.log("Texture loader cache tests passed");
}

await run();
