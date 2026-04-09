import assert from "node:assert/strict";
import { CSG } from "../src/csg/CSGBuilder.ts";
import { CSGMeshInstance } from "../src/meshes/CSGMeshInstance.ts";
import { MeshFactory } from "../src/meshes/MeshFactory.ts";
import { Material } from "../src/materials/Material.ts";

function createOperands() {
	const left = MeshFactory.createBox(
		{ x: -0.2, y: 0, z: 0 },
		2,
		2,
		2,
		new Material({ name: "PhysicsCSGLeft" })
	);
	const right = MeshFactory.createBox(
		{ x: 0.25, y: 0, z: 0 },
		2,
		2,
		2,
		new Material({ name: "PhysicsCSGRight" })
	);
	return { left, right };
}

function run() {
	const { left, right } = createOperands();
	let rebuildCount = 0;
	const physics = {
		rebuildColliders() {
			rebuildCount++;
			return [];
		},
	};

	const csgMesh = new CSGMeshInstance({
		graph: CSG.from(left).union(right),
		physicsSync: "auto",
		physicsSystem: physics,
		name: "physics-csg-auto",
	});

	const first = csgMesh.flushCSG();
	assert.equal(first.ok, true);
	assert.equal(rebuildCount, 1);

	csgMesh.physicsSync = "off";
	csgMesh.setGraph(CSG.from(left).subtract(right));
	const second = csgMesh.flushCSG();
	assert.equal(second.ok, true);
	assert.equal(rebuildCount, 1);

	console.log("Physics CSG sync tests passed");
}

run();
