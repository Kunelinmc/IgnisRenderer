import assert from "node:assert/strict";
import { Node } from "../src/core/Node.ts";
import { PhysicsSystem } from "../src/physics/PhysicsSystem.ts";
import { SimplePhysicsAdapter } from "../src/physics/adapters/SimplePhysicsAdapter.ts";

class TrackingPhysicsAdapter extends SimplePhysicsAdapter {
	constructor() {
		super("tracking");
		this.stepWorldCalls = 0;
		this.setBodyTransformCalls = 0;
	}

	stepWorld(worldId, deltaSeconds) {
		this.stepWorldCalls++;
		return super.stepWorld(worldId, deltaSeconds);
	}

	setBodyTransform(worldId, bodyId, transform) {
		this.setBodyTransformCalls++;
		super.setBodyTransform(worldId, bodyId, transform);
	}
}

function testAnimationAuthorityDirtySync() {
	const adapter = new TrackingPhysicsAdapter();
	const physics = new PhysicsSystem({ adapter });
	physics.initSync();
	physics.createWorld({
		worldId: "main",
		gravity: { x: 0, y: 0, z: 0 },
		mode: "variable",
	});

	const node = new Node();
	const body = physics.attachBody(node, {
		worldId: "main",
		body: { type: "kinematic" },
		authority: "animation",
	});
	physics.addCollider(body, {
		mode: "explicit",
		shape: { kind: "sphere", radius: 1 },
	});

	physics.step(0.016);
	assert.equal(
		adapter.setBodyTransformCalls,
		0,
		"Expected no animation transform upload when node is unchanged"
	);

	node.position.x = 2;
	physics.step(0.016);
	assert.equal(
		adapter.setBodyTransformCalls,
		1,
		"Expected one transform upload when animation body moved"
	);

	physics.step(0.016);
	assert.equal(
		adapter.setBodyTransformCalls,
		1,
		"Expected unchanged animation body to skip transform upload"
	);
}

function testSleepingIslandSkipsWorldStep() {
	const adapter = new TrackingPhysicsAdapter();
	const physics = new PhysicsSystem({ adapter });
	physics.initSync();
	physics.createWorld({
		worldId: "main",
		gravity: { x: 0, y: 0, z: 0 },
		mode: "variable",
	});

	const node = new Node();
	const body = physics.attachBody(node, {
		worldId: "main",
		body: { type: "dynamic" },
		authority: "physics",
	});
	physics.addCollider(body, {
		mode: "explicit",
		shape: { kind: "sphere", radius: 1 },
	});

	physics.step(0.016);
	const firstStepCalls = adapter.stepWorldCalls;
	assert.ok(firstStepCalls > 0, "Expected first frame to step physics world");

	physics.step(0.016);
	assert.equal(
		adapter.stepWorldCalls,
		firstStepCalls,
		"Expected sleeping island to skip redundant world step"
	);
}

function run() {
	testAnimationAuthorityDirtySync();
	testSleepingIslandSkipsWorldStep();
	console.log("Physics system optimization tests passed");
}

run();
