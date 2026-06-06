import assert from "node:assert/strict";
import { Platform } from "../../../src/foundation/Platform.ts";
import { CSG } from "../../../src/csg/CSGBuilder.ts";
import { buildCSGMeshAsset } from "../../../src/csg/solvers.ts";
import { CSGMeshInstance } from "../../../src/meshes/CSGMeshInstance.ts";
import { MeshFactory } from "../../../src/meshes/MeshFactory.ts";
import { Material } from "../../../src/materials/Material.ts";

function createOperands() {
	const left = MeshFactory.createBox(
		{ x: -0.2, y: 0, z: 0 },
		2,
		2,
		2,
		new Material({ name: "LeftCSG" })
	);
	const right = MeshFactory.createBox(
		{ x: 0.35, y: 0, z: 0 },
		2,
		2,
		2,
		new Material({ name: "RightCSG" })
	);
	return { left, right };
}

async function testFlushLifecycleAndStaleToken() {
	const { left, right } = createOperands();
	const instance = new CSGMeshInstance({
		graph: CSG.from(left).subtract(right),
		name: "csgLifecycle",
	});

	const first = instance.flushCSG();
	assert.equal(first.ok, true);
	assert.equal(instance.isCSGDirty, false);
	assert.ok(instance.mesh.primitives.length > 0);

	instance.setExecutionMode("worker");
	instance.setGraph(CSG.from(left).union(right));
	const flushA = instance.flushCSG();
	instance.setGraph(CSG.from(left).intersect(right));
	const flushB = instance.flushCSG();

	const resultA = await flushA;
	const resultB = await flushB;
	assert.equal(resultA.stale, true);
	assert.equal(resultB.stale, false);
	assert.equal(resultB.ok, true);

	if (!Platform.hasWorker()) {
		assert.ok(
			resultB.diagnostics.some((diagnostic) =>
				diagnostic.code === "csg-worker-fallback-sync"
			)
		);
	}
}

function testPhysicsAutoSyncToggle() {
	const { left, right } = createOperands();
	let rebuildCalls = 0;
	const fakePhysicsSystem = {
		rebuildColliders() {
			rebuildCalls++;
			return [];
		},
	};

	const instance = new CSGMeshInstance({
		graph: CSG.from(left).union(right),
		physicsSync: "auto",
		physicsSystem: fakePhysicsSystem,
		name: "csgPhysicsSync",
	});

	const first = instance.flushCSG();
	assert.equal(first.ok, true);
	assert.equal(rebuildCalls, 1);

	instance.physicsSync = "off";
	instance.setGraph(CSG.from(left).subtract(right));
	const second = instance.flushCSG();
	assert.equal(second.ok, true);
	assert.equal(rebuildCalls, 1);
}

async function testWorkerExecutorPath() {
	const { left, right } = createOperands();
	const operations = [];
	const executor = {
		execute(graph, options) {
			const operation = graph.kind === "op" ? graph.operation : "leaf";
			operations.push(operation);
			const delayMs = operation === "union" ? 10 : 0;
			return new Promise((resolve) => {
				setTimeout(() => {
					resolve(buildCSGMeshAsset(graph, options));
				}, delayMs);
			});
		},
	};

	const instance = new CSGMeshInstance({
		graph: CSG.from(left).subtract(right),
		executionMode: "worker",
		executor,
		name: "csgCustomExecutor",
	});

	instance.setGraph(CSG.from(left).union(right));
	const flushA = instance.flushCSG();
	instance.setGraph(CSG.from(left).intersect(right));
	const flushB = instance.flushCSG();

	const resultA = await flushA;
	const resultB = await flushB;
	assert.equal(operations.length, 2);
	assert.equal(resultA.stale, true);
	assert.equal(resultB.stale, false);
	assert.equal(resultB.ok, true);
	assert.equal(
		resultB.diagnostics.some(
			(diagnostic) => diagnostic.code === "csg-worker-fallback-sync"
		),
		false
	);
}

async function run() {
	await testFlushLifecycleAndStaleToken();
	testPhysicsAutoSyncToggle();
	await testWorkerExecutorPath();
	console.log("CSG mesh instance tests passed");
}

await run();
