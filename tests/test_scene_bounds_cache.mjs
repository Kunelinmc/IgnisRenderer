import assert from "node:assert/strict";
import { Scene } from "../src/core/Scene.ts";
import { Material } from "../src/materials/Material.ts";
import { SimpleModel } from "../src/models/SimpleModel.ts";

function createModel(x) {
	const model = SimpleModel.fromFaces([
		{
			material: new Material(),
			vertices: [
				{ x: -1, y: 0, z: 0, normal: { x: 0, y: 0, z: 1 } },
				{ x: 1, y: 0, z: 0, normal: { x: 0, y: 0, z: 1 } },
				{ x: 0, y: 1, z: 0, normal: { x: 0, y: 0, z: 1 } },
			],
		},
	]);
	model.transform.position.x = x;
	return model;
}

function run() {
	const scene = new Scene();
	const initialVersion = scene.version;
	const firstBounds = scene.getBounds();
	assert.equal(firstBounds.radius, 100);

	const left = createModel(-10);
	const right = createModel(10);
	scene.addModel(left);
	scene.addModel(right);
	assert.ok(scene.version > initialVersion);

	const expanded = scene.getBounds();
	const cached = scene.getBounds();
	assert.deepEqual(expanded, cached);
	assert.ok(expanded.radius > 0);

	scene.removeModel(right);
	const reduced = scene.getBounds();
	assert.ok(reduced.radius < expanded.radius);

	scene.clear();
	const cleared = scene.getBounds();
	assert.equal(cleared.radius, 100);

	console.log("Scene bounds cache tests passed");
}

run();
