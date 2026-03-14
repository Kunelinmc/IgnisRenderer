import assert from "node:assert/strict";
import { GLTFLoader } from "../src/loaders/GLTFLoader.ts";
import { Scene } from "../src/core/Scene.ts";

async function run() {
	const loader = new GLTFLoader();
	const json = {
		asset: { version: "2.0" },
		scene: 0,
		scenes: [{ nodes: [] }],
		nodes: [],
	};
	const encoded = new TextEncoder().encode(JSON.stringify(json));
	const prefab = await loader.parsePrefab(encoded.buffer, "");
	assert.ok(prefab);
	assert.equal(typeof prefab.instantiate, "function");

	const scene = new Scene();
	const { root, rootEntity } = prefab.instantiate(scene);
	assert.ok(root);
	assert.ok(rootEntity !== null);

	console.log("GLTF prefab contract tests passed");
}

await run();
