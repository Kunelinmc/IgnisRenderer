import assert from "node:assert/strict";
import { Buffer } from "node:buffer";

import { GLTFLoader } from "../../../src/loaders/GLTFLoader.ts";
import { Scene } from "../../../src/core/Scene.ts";
import { MeshInstance } from "../../../src/meshes/MeshInstance.ts";

async function testEmptyPrefab() {
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
	const root = prefab.instantiate(scene);
	assert.ok(root);
	assert.equal(scene.contains(root), true);
}

function createMorphSource() {
	const data = new Float32Array([
		0, 0, 0, 1, 0, 0, 0, 1, 0,
		1, 0, 0, 1, 0, 0, 1, 0, 0,
	]);
	const primitive = {
		attributes: { POSITION: 0 },
		targets: [{ POSITION: 1 }, { POSITION: 1 }, { POSITION: 1 }],
	};
	const json = {
		asset: { version: "2.0" },
		scene: 0,
		scenes: [{ nodes: [0, 1, 2, 3, 4, 5] }],
		nodes: [
			{ mesh: 0 },
			{ mesh: 0, weights: [0.75] },
			{ mesh: 0, weights: [0, 1, 0.5, 9] },
			{ mesh: 1 },
			{ mesh: 1, weights: [0.5] },
			{ mesh: 2 },
		],
		meshes: [
			{ primitives: [primitive, primitive], weights: [0.25, 0.5] },
			{ primitives: [primitive] },
			{ primitives: [{ attributes: { POSITION: 0 } }] },
		],
		buffers: [{
			byteLength: data.byteLength,
			uri: `data:application/octet-stream;base64,${Buffer.from(data.buffer).toString("base64")}`,
		}],
		bufferViews: [
			{ buffer: 0, byteOffset: 0, byteLength: 36 },
			{ buffer: 0, byteOffset: 36, byteLength: 36 },
		],
		accessors: [
			{ bufferView: 0, componentType: 5126, count: 3, type: "VEC3" },
			{ bufferView: 1, componentType: 5126, count: 3, type: "VEC3" },
		],
	};
	return new TextEncoder().encode(JSON.stringify(json)).buffer;
}

async function testLoadedMorphWeightsAndPrefabIsolation() {
	const loader = new GLTFLoader();
	const prefab = await loader.parsePrefab(createMorphSource());
	const instances = prefab.root.children.map((node) => node.children[0]);
	const [base, partial, override, zero, nodeOnly, staticMesh] = instances;
	const values = instances.map((instance) =>
		instance.morphWeights.map((weights) => Array.from(weights))
	);
	assert.deepEqual(values, [
		[[0.25, 0.5, 0], [0.25, 0.5, 0]],
		[[0.75, 0.5, 0], [0.75, 0.5, 0]],
		[[0, 1, 0.5], [0, 1, 0.5]],
		[[0, 0, 0]],
		[[0.5, 0, 0]],
		[[]],
	]);
	assert.equal(base.mesh, partial.mesh);
	assert.equal(base.mesh, override.mesh);
	assert.equal(zero.mesh, nodeOnly.mesh);
	const bindings = loader.getLastAnimationBundle().morphBindings;
	assert.equal(bindings.find((binding) => binding.instance === zero).targetCount, 3);
	assert.equal(bindings.some((binding) => binding.instance === staticMesh), false);

	const scene = new Scene();
	const firstRoot = prefab.instantiate(scene);
	const secondRoot = prefab.instantiate(scene);
	const first = firstRoot.children[0].children[0];
	const second = secondRoot.children[0].children[0];
	assert.equal(first.mesh, base.mesh);
	assert.equal(second.mesh, base.mesh);
	assert.deepEqual(first.morphWeights.map((weights) => Array.from(weights)), [
		[0.25, 0.5, 0], [0.25, 0.5, 0],
	]);
	first.morphWeights[0][0] = 1;
	assert.equal(first.morphWeights[1][0], 0.25);
	assert.equal(second.morphWeights[0][0], 0.25);
	assert.equal(base.morphWeights[0][0], 0.25);
	assert.equal(partial.morphWeights[0][0], 0.75);
	assert.equal(override.morphWeights[0][0], 0);

	const fresh = new MeshInstance({ mesh: base.mesh });
	assert.deepEqual(fresh.morphWeights.map((weights) => Array.from(weights)), [
		[0, 0, 0], [0, 0, 0],
	]);
}

async function run() {
	await testEmptyPrefab();
	await testLoadedMorphWeightsAndPrefabIsolation();
	console.log("GLTF prefab contract tests passed");
}

await run();
