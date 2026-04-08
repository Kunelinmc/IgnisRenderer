import assert from "node:assert/strict";
import { CSG } from "../src/csg/CSGBuilder.ts";
import { MeshFactory } from "../src/meshes/MeshFactory.ts";
import { MeshAsset } from "../src/meshes/MeshAsset.ts";
import { Material } from "../src/materials/Material.ts";

function createOverlapBoxes() {
	const left = MeshFactory.createBox(
		{ x: -0.25, y: 0, z: 0 },
		2,
		2,
		2,
		new Material({ name: "LeftMaterial" })
	);
	const right = MeshFactory.createBox(
		{ x: 0.35, y: 0, z: 0 },
		2,
		2,
		2,
		new Material({ name: "RightMaterial" })
	);
	return { left, right };
}

function testBooleanOperations() {
	const { left, right } = createOverlapBoxes();

	const union = CSG.from(left).union(right).toMeshAsset();
	const subtract = CSG.from(left).subtract(right).toMeshAsset();
	const intersect = CSG.from(left).intersect(right).toMeshAsset();
	const xor = CSG.from(left).xor(right).toMeshAsset();

	assert.ok(union.primitives.length > 0);
	assert.ok(subtract.primitives.length > 0);
	assert.ok(intersect.primitives.length > 0);
	assert.ok(xor.primitives.length > 0);
	assert.ok(
		union.boundingSphere.radius >= intersect.boundingSphere.radius,
		"union radius should be >= intersect radius"
	);
}

function testClosedManifoldValidation() {
	const openPlane = MeshFactory.createPlane(
		{ x: 0, y: 0, z: 0 },
		2,
		2,
		new Material({ name: "PlaneMaterial" })
	);
	const box = MeshFactory.createBox(
		{ x: 0, y: 0, z: 0 },
		2,
		2,
		2,
		new Material({ name: "BoxMaterial" })
	);

	const result = CSG.from(openPlane).union(box).solve();
	assert.equal(result.ok, false);
	assert.ok(
		result.diagnostics.some((diagnostic) =>
			diagnostic.code === "csg-input-non-manifold"
		)
	);
}

function testTopologyValidation() {
	const box = MeshFactory.createBox(
		{ x: 0, y: 0, z: 0 },
		2,
		2,
		2,
		new Material({ name: "TopologyBox" })
	);
	const primitive = box.mesh.primitives[0];
	const invalidMesh = new MeshAsset([
		{
			...primitive,
			topology: "line-list",
		},
	]);
	const result = CSG.from(invalidMesh).solve();
	assert.equal(result.ok, false);
	assert.ok(
		result.diagnostics.some((diagnostic) =>
			diagnostic.code === "csg-input-non-triangle-topology"
		)
	);
}

function testOutputTriangleLimit() {
	const { left, right } = createOverlapBoxes();
	const result = CSG.from(left).union(right).solve({
		maxOutputTriangles: 1,
	});
	assert.equal(result.ok, false);
	assert.ok(
		result.diagnostics.some((diagnostic) =>
			diagnostic.code === "csg-output-triangle-limit"
		)
	);
}

function testAttributeDropDiagnostics() {
	const box = MeshFactory.createBox(
		{ x: 0, y: 0, z: 0 },
		2,
		2,
		2,
		new Material({ name: "AttributeBox" })
	);
	for (const primitive of box.mesh.primitives) {
		const uv0 = primitive.geometry.uv0;
		primitive.geometry.uv1 =
			uv0 ? new Float32Array(uv0) : new Float32Array(primitive.geometry.positions.length / 3 * 2);
	}

	const result = CSG.from(box).solve();
	assert.equal(result.ok, true);
	assert.ok(
		result.diagnostics.some((diagnostic) =>
			diagnostic.code === "csg-attribute-dropped"
		)
	);
}

function run() {
	testBooleanOperations();
	testClosedManifoldValidation();
	testTopologyValidation();
	testOutputTriangleLimit();
	testAttributeDropDiagnostics();
	console.log("CSG core tests passed");
}

run();
