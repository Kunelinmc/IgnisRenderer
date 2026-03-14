import assert from "node:assert/strict";
import { GLTFLoader } from "../src/loaders/GLTFLoader.ts";

const COMPONENT_TYPE_FLOAT = 5126;

async function testRejectsCyclicNodeHierarchy() {
	const loader = new GLTFLoader();
	const json = {
		asset: { version: "2.0" },
		scene: 0,
		scenes: [{ nodes: [0] }],
		nodes: [{ children: [1] }, { children: [0] }],
	};
	const encoded = new TextEncoder().encode(JSON.stringify(json));
	await assert.rejects(
		loader.parse(encoded.buffer, ""),
		/cyclic node hierarchy/i
	);
}

function testRejectsOutOfBoundsAccessorRange() {
	const loader = new GLTFLoader();
	const json = {
		accessors: [
			{
				bufferView: 0,
				componentType: COMPONENT_TYPE_FLOAT,
				count: 4,
				type: "VEC3",
			},
		],
		bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: 16 }],
	};
	const buffer = new Uint8Array(16);
	assert.throws(
		() => loader.getAccessorData(json, [buffer], 0),
		/exceeds its bufferView byteLength/i
	);
}

function testRejectsOversizedAccessorAllocation() {
	const loader = new GLTFLoader();
	const json = {
		accessors: [
			{
				componentType: COMPONENT_TYPE_FLOAT,
				count: 100_000_000,
				type: "VEC4",
			},
		],
	};
	assert.throws(
		() => loader.getAccessorData(json, [], 0),
		/exceeds safe allocation limit/i
	);
}

async function testRejectsUnsupportedBufferScheme() {
	const loader = new GLTFLoader();
	const json = {
		asset: { version: "2.0" },
		scene: 0,
		scenes: [{ nodes: [] }],
		nodes: [],
		buffers: [{ uri: "file:///etc/passwd" }],
	};
	const encoded = new TextEncoder().encode(JSON.stringify(json));
	await assert.rejects(
		loader.parse(encoded.buffer, "https://example.com/assets/"),
		/unsupported uri scheme/i
	);
}

async function run() {
	await testRejectsCyclicNodeHierarchy();
	testRejectsOutOfBoundsAccessorRange();
	testRejectsOversizedAccessorAllocation();
	await testRejectsUnsupportedBufferScheme();
	console.log("GLTF loader security tests passed");
}

await run();
