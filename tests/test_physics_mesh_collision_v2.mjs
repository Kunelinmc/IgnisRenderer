import assert from "node:assert/strict";
import { Node } from "../src/core/Node.ts";
import { Scene } from "../src/core/Scene.ts";
import { Material } from "../src/materials/Material.ts";
import { MeshAsset } from "../src/meshes/MeshAsset.ts";
import { MeshInstance } from "../src/meshes/MeshInstance.ts";
import { PhysicsSystem } from "../src/physics/PhysicsSystem.ts";
import { SimplePhysicsAdapter } from "../src/physics/adapters/SimplePhysicsAdapter.ts";
import { RapierPhysicsAdapter } from "../src/physics/adapters/RapierPhysicsAdapter.ts";
import { createFakeRapierModule } from "./helpers/test_fakes.mjs";

class CaptureAdapter extends SimplePhysicsAdapter {
	constructor(id = "capture") {
		super(id);
		this.addColliderCalls = [];
		this.destroyColliderCalls = [];
		this.lastRaycastFilter = null;
	}

	addCollider(worldId, bodyId, colliderId, descriptor, shape) {
		this.addColliderCalls.push({
			worldId,
			bodyId,
			colliderId,
			descriptor: { ...descriptor },
			shape,
		});
		super.addCollider(worldId, bodyId, colliderId, descriptor, shape);
	}

	destroyCollider(worldId, colliderId) {
		this.destroyColliderCalls.push({ worldId, colliderId });
		super.destroyCollider(worldId, colliderId);
	}

	raycast(worldId, query) {
		this.lastRaycastFilter =
			query.filter ?
				{
					...query.filter,
					includeColliderIds: query.filter.includeColliderIds ?
						[...query.filter.includeColliderIds]
					:	undefined,
				}
			:	null;
		return super.raycast(worldId, query);
	}
}

function createTriangleMeshInstance(vertices) {
	const mesh = MeshAsset.fromFaces([
		{
			material: new Material({ name: "MeshColliderV2" }),
			vertices,
		},
	]);
	const meshInstance = new MeshInstance({ mesh });
	meshInstance.updateWorldMatrix();
	return meshInstance;
}

function createCenteredMesh(z = 5) {
	return createTriangleMeshInstance([
		{
			x: -0.5,
			y: -0.5,
			z,
			u: 0,
			v: 0,
			normal: { x: 0, y: 0, z: 1 },
		},
		{
			x: 0.5,
			y: -0.5,
			z,
			u: 1,
			v: 0,
			normal: { x: 0, y: 0, z: 1 },
		},
		{
			x: 0,
			y: 0.5,
			z,
			u: 0.5,
			v: 1,
			normal: { x: 0, y: 0, z: 1 },
		},
	]);
}

function createVerticalParallelMesh() {
	return createTriangleMeshInstance([
		{
			x: 1,
			y: -0.5,
			z: 5,
			u: 0,
			v: 0,
			normal: { x: 1, y: 0, z: 0 },
		},
		{
			x: 1,
			y: 0.5,
			z: 5,
			u: 1,
			v: 0,
			normal: { x: 1, y: 0, z: 0 },
		},
		{
			x: 1,
			y: 0,
			z: 7,
			u: 0.5,
			v: 1,
			normal: { x: 1, y: 0, z: 0 },
		},
	]);
}

function testTrimeshCookCompatibilityLayer() {
	const adapter = new CaptureAdapter("mesh-v2-compat");
	const physics = new PhysicsSystem({ adapter });
	physics.initSync();
	physics.createWorld({ worldId: "main" });

	const sourceMesh = createCenteredMesh();
	const body = physics.attachBody(new Node(), {
		worldId: "main",
		body: { type: "fixed" },
		authority: "physics",
	});

	const warnings = [];
	const originalWarn = console.warn;
	console.warn = (...args) => {
		warnings.push(args.map((value) => String(value)).join(" "));
	};
	try {
		physics.addCollider(body, {
			mode: "trimesh-cook",
			sourceNode: sourceMesh,
		});
		physics.addCollider(body, {
			mode: "trimesh-cook",
			sourceNode: sourceMesh,
		});
	} finally {
		console.warn = originalWarn;
	}

	assert.equal(adapter.addColliderCalls.length, 2);
	const translated = adapter.addColliderCalls[0].descriptor;
	assert.equal(translated.mode, "mesh");
	assert.equal(translated.meshPolicy, "fixed");
	assert.equal(translated.narrowphase, "face-bvh");
	assert.equal(translated.backendPreference, "approx");
	assert.equal(warnings.length, 1);
	assert.ok(
		warnings[0].includes("deprecated"),
		"Expected a deprecation warning for trimesh-cook"
	);

	physics.destroyWorld("main");
}

function testRebuildCollidersGeometryVersionGate() {
	const adapter = new CaptureAdapter("mesh-v2-rebuild");
	const physics = new PhysicsSystem({ adapter });
	physics.initSync();
	physics.createWorld({ worldId: "main" });

	const sourceMesh = createCenteredMesh();
	const body = physics.attachBody(new Node(), {
		worldId: "main",
		body: { type: "fixed" },
		authority: "physics",
	});
	const collider = physics.addCollider(body, {
		mode: "mesh",
		sourceNode: sourceMesh,
		meshPolicy: "fixed",
		narrowphase: "face-bvh",
		backendPreference: "approx",
	});

	assert.equal(adapter.addColliderCalls.length, 1);
	assert.equal(adapter.destroyColliderCalls.length, 0);

	sourceMesh.position.x = 3;
	sourceMesh.updateWorldMatrix();
	const transformOnlyRebuild = physics.rebuildColliders(body);
	assert.equal(transformOnlyRebuild.length, 1);
	assert.equal(transformOnlyRebuild[0].id, collider.id);
	assert.equal(
		adapter.destroyColliderCalls.length,
		0,
		"Transform-only updates must not trigger collider recook"
	);
	assert.equal(
		adapter.addColliderCalls.length,
		1,
		"Transform-only updates must not recreate collider"
	);

	sourceMesh.mesh.primitives[0].geometryVersion += 1;
	const geometryRebuild = physics.rebuildColliders(body);
	assert.equal(geometryRebuild.length, 1);
	assert.notEqual(
		geometryRebuild[0].id,
		collider.id,
		"Geometry version changes must recreate collider"
	);
	assert.equal(adapter.destroyColliderCalls.length, 1);
	assert.equal(adapter.addColliderCalls.length, 2);

	physics.destroyWorld("main");
}

function testBindSceneSpatialCandidateFiltering() {
	const adapter = new CaptureAdapter("mesh-v2-spatial");
	const physics = new PhysicsSystem({ adapter });
	physics.initSync();
	physics.createWorld({ worldId: "main" });

	const scene = new Scene();
	const nearMesh = createCenteredMesh(6);
	const farMesh = createCenteredMesh(6);
	farMesh.position.x = 40;
	farMesh.updateWorldMatrix();
	scene.add(nearMesh);
	scene.add(farMesh);
	scene.updateWorldMatrices();
	physics.bindSceneSpatial(scene);

	const nearBody = physics.attachBody(nearMesh, {
		worldId: "main",
		body: { type: "fixed" },
		authority: "physics",
	});
	const farBody = physics.attachBody(farMesh, {
		worldId: "main",
		body: { type: "fixed" },
		authority: "physics",
	});
	const nonMeshBody = physics.attachBody(
		new Node({ position: { x: 0, y: 0, z: 2 } }),
		{
			worldId: "main",
			body: { type: "fixed" },
			authority: "physics",
		}
	);

	const nearCollider = physics.addCollider(nearBody, {
		mode: "mesh",
		sourceNode: nearMesh,
		meshPolicy: "fixed",
		narrowphase: "face-bvh",
		backendPreference: "approx",
	});
	const farCollider = physics.addCollider(farBody, {
		mode: "mesh",
		sourceNode: farMesh,
		meshPolicy: "fixed",
		narrowphase: "face-bvh",
		backendPreference: "approx",
	});
	const nonMeshCollider = physics.addCollider(nonMeshBody, {
		mode: "explicit",
		shape: { kind: "sphere", radius: 0.5 },
	});

	physics.raycast({
		worldId: "main",
		origin: { x: 0, y: 0, z: 0 },
		direction: { x: 0, y: 0, z: 1 },
		maxDistance: 20,
	});

	const includeColliderIds = adapter.lastRaycastFilter?.includeColliderIds ?? [];
	const includeSet = new Set(includeColliderIds);
	assert.equal(includeSet.has(nearCollider.id), true);
	assert.equal(includeSet.has(nonMeshCollider.id), true);
	assert.equal(includeSet.has(farCollider.id), false);

	physics.destroyWorld("main");
}

async function testRapierExactMeshRaycastAgainstApprox() {
	const fakeRapier = createFakeRapierModule();
	const physics = new PhysicsSystem({
		adapter: new RapierPhysicsAdapter({
			moduleLoader: async () => fakeRapier.module,
			strict: true,
		}),
	});
	await physics.init();
	physics.createWorld({
		worldId: "main",
		gravity: { x: 0, y: 0, z: 0 },
		mode: "variable",
	});

	const sourceMesh = createVerticalParallelMesh();
	const exactBody = physics.attachBody(new Node(), {
		worldId: "main",
		body: { type: "fixed" },
		authority: "physics",
	});
	const approxBody = physics.attachBody(new Node(), {
		worldId: "main",
		body: { type: "fixed" },
		authority: "physics",
	});

	const exactCollider = physics.addCollider(exactBody, {
		mode: "mesh",
		sourceNode: sourceMesh,
		meshPolicy: "fixed",
		narrowphase: "face-bvh",
		backendPreference: "exact",
	});
	const approxCollider = physics.addCollider(approxBody, {
		mode: "mesh",
		sourceNode: sourceMesh,
		meshPolicy: "fixed",
		narrowphase: "proxy",
		backendPreference: "approx",
	});

	const query = {
		worldId: "main",
		origin: { x: 0, y: 0, z: 0 },
		direction: { x: 0, y: 0, z: 1 },
		maxDistance: 20,
	};
	const exactHit = physics.raycast({
		...query,
		filter: { includeColliderIds: [exactCollider.id] },
	});
	assert.equal(
		exactHit,
		null,
		"Exact mesh raycast should not hit when the ray is parallel to all triangles"
	);

	const approxHit = physics.raycast({
		...query,
		filter: { includeColliderIds: [approxCollider.id] },
	});
	assert.ok(
		approxHit,
		"Approx mesh raycast may return conservative proxy hits"
	);
	assert.equal(approxHit?.colliderId, approxCollider.id);

	physics.destroyWorld("main");
}

async function run() {
	testTrimeshCookCompatibilityLayer();
	testRebuildCollidersGeometryVersionGate();
	testBindSceneSpatialCandidateFiltering();
	await testRapierExactMeshRaycastAgainstApprox();
	console.log("Physics mesh collision V2 tests passed");
}

await run();
