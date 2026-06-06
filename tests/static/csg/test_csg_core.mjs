import assert from "node:assert/strict";
import { CSG } from "../../../src/csg/CSGBuilder.ts";
import { CSGSolver } from "../../../src/csg/solvers.ts";
import { MeshFactory } from "../../../src/meshes/MeshFactory.ts";
import { MeshAsset } from "../../../src/meshes/MeshAsset.ts";
import { Material } from "../../../src/materials/Material.ts";

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

function testSolverClassAutoFallback() {
	const { left, right } = createOverlapBoxes();
	const graph = CSG.from(left).union(right).getGraph();
	const solver = new CSGSolver();

	solver.registerWasmSolver({
		id: "failing-wasm",
		solve() {
			throw new Error("expected wasm failure");
		},
	});

	assert.deepEqual(solver.listWasmSolvers(), ["failing-wasm"]);

	const result = solver.buildMeshAsset(graph);
	assert.equal(result.ok, true);
	assert.equal(result.solverId, "builtin");
	assert.equal(result.fallbackUsed, true);
	assert.ok(
		result.diagnostics.some((diagnostic) =>
			diagnostic.code === "csg-solver-auto-fallback"
		)
	);

	solver.unregisterWasmSolver("failing-wasm");
	assert.deepEqual(solver.listWasmSolvers(), []);
	assert.deepEqual(new CSGSolver().listWasmSolvers(), []);
}

function testSolverClassWasmSelection() {
	const { left, right } = createOverlapBoxes();
	const graph = CSG.from(left).union(right).getGraph();
	const solver = new CSGSolver([
		{
			id: "mock-wasm",
			solve(request) {
				assert.equal(request.options.solverPreference, "auto");
				return {
					meshAsset: MeshAsset.fromFaces([]),
					triangleCount: 0,
					diagnostics: [
						{
							code: "mock-wasm-used",
							message: "Mock wasm solver was selected",
							severity: "info",
						},
					],
				};
			},
		},
	]);

	const wasmResult = solver.buildMeshAsset(graph);
	assert.equal(wasmResult.ok, true);
	assert.equal(wasmResult.solverId, "mock-wasm");
	assert.ok(
		wasmResult.diagnostics.some((diagnostic) =>
			diagnostic.code === "mock-wasm-used"
		)
	);

	const builtinResult = solver.buildMeshAsset(graph, {
		solverPreference: "builtin",
	});
	assert.equal(builtinResult.ok, true);
	assert.equal(builtinResult.solverId, "builtin");
	assert.ok(builtinResult.triangleCount > 0);
}

function testSolverClassMissingWasmDiagnostic() {
	const { left, right } = createOverlapBoxes();
	const graph = CSG.from(left).union(right).getGraph();
	const solver = new CSGSolver();
	const result = solver.buildMeshAsset(graph, {
		solverPreference: "wasm",
	});

	assert.equal(result.ok, false);
	assert.equal(result.solverId, "wasm");
	assert.ok(
		result.diagnostics.some((diagnostic) =>
			diagnostic.code === "csg-solver-missing"
		)
	);
}

function run() {
	testBooleanOperations();
	testClosedManifoldValidation();
	testTopologyValidation();
	testOutputTriangleLimit();
	testAttributeDropDiagnostics();
	testSolverClassAutoFallback();
	testSolverClassWasmSelection();
	testSolverClassMissingWasmDiagnostic();
	console.log("CSG core tests passed");
}

run();
