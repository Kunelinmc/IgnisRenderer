import assert from "node:assert/strict";
import { Node } from "../src/core/Node.ts";
import { PhysicsSystem } from "../src/physics/PhysicsSystem.ts";
import { SimplePhysicsAdapter } from "../src/physics/adapters/SimplePhysicsAdapter.ts";
import { AmmoWorkerPhysicsAdapter } from "../src/physics/adapters/AmmoWorkerPhysicsAdapter.ts";
import { RapierWorkerPhysicsAdapter } from "../src/physics/adapters/RapierWorkerPhysicsAdapter.ts";
import {
	postMessageWorkerTransportPlugin,
	sharedArrayBufferWorkerTransportPlugin,
} from "../src/workers/transports.ts";

import { FakeWorker } from "./helpers/test_fakes.mjs";

class AsyncOnlyStepAdapter extends SimplePhysicsAdapter {
	constructor() {
		super("async-only-step");
		this.syncStepCalls = 0;
		this.asyncStepCalls = 0;
	}

	stepWorld(worldId, deltaSeconds) {
		this.syncStepCalls++;
		throw new Error(
			`Synchronous stepWorld should not be used for "${worldId}" (${deltaSeconds})`
		);
	}

	async stepWorldAsync(worldId, deltaSeconds) {
		this.asyncStepCalls++;
		return super.stepWorld(worldId, deltaSeconds);
	}
}

function decodeTaskEnvelopeWithPlugin(data) {
	const sharedDecoded = sharedArrayBufferWorkerTransportPlugin.decodeTask(data);
	if (sharedDecoded) {
		return {
			plugin: sharedArrayBufferWorkerTransportPlugin,
			envelope: sharedDecoded,
		};
	}
	const postDecoded = postMessageWorkerTransportPlugin.decodeTask(data);
	if (postDecoded) {
		return {
			plugin: postMessageWorkerTransportPlugin,
			envelope: postDecoded,
		};
	}
	return null;
}

function createWorkerHandler(observedModes, dispatchPayloads) {
	return (message, worker) => {
		const decoded = decodeTaskEnvelopeWithPlugin(message);
		assert.ok(decoded, "Expected worker payload to decode");
		observedModes.push(decoded.plugin.mode);

		const payload = decoded.envelope.payload;
		let result = null;
		if (payload?.type === "dispatch") {
			dispatchPayloads.push(payload);
			if (payload.request?.type === "stepWorld") {
				result = {
					bodyStates: [],
					events: [],
					activeBodies: 1,
					sleepingBodies: 0,
					ccdBodies: 0,
				};
			}
			if (
				payload.request?.type === "raycast" ||
				payload.request?.type === "sphereCast" ||
				payload.request?.type === "boxCast"
			) {
				result = {
					worldId: payload.request.worldId,
					bodyId: "body-1",
					colliderId: "collider-1",
					point: { x: 0, y: 0, z: 0 },
					normal: { x: 0, y: 1, z: 0 },
					distance: 1,
					fraction: 0.25,
					isTrigger: false,
				};
			}
			if (payload.request?.type === "raycastAll") {
				result = [
					{
						worldId: payload.request.worldId,
						bodyId: "body-1",
						colliderId: "collider-1",
						point: { x: 0, y: 0, z: 0 },
						normal: { x: 0, y: 1, z: 0 },
						distance: 1,
						fraction: 0.25,
						isTrigger: false,
					},
				];
			}
			if (
				payload.request?.type === "overlapSphere" ||
				payload.request?.type === "overlapBox"
			) {
				result = [
					{
						worldId: payload.request.worldId,
						bodyId: "body-1",
						colliderId: "collider-1",
						isTrigger: false,
					},
				];
			}
		}

		const encoded = decoded.plugin.encodeResult({
			id: decoded.envelope.id,
			result,
		});
		worker.emitMessage(encoded.message);
	};
}

function createCharacterControllerWorkerHandler(dispatchPayloads) {
	const groundedByController = new Map();
	return (message, worker) => {
		const decoded = decodeTaskEnvelopeWithPlugin(message);
		assert.ok(decoded, "Expected worker payload to decode");

		const payload = decoded.envelope.payload;
		let result = null;
		if (payload?.type === "dispatch") {
			dispatchPayloads.push(payload);
			for (const command of payload.commands) {
				if (command.type === "createCharacterController") {
					groundedByController.set(
						`${command.worldId}:${command.controllerId}`,
						false
					);
				}
				if (command.type === "destroyCharacterController") {
					groundedByController.delete(
						`${command.worldId}:${command.controllerId}`
					);
				}
				if (command.type === "moveCharacterController") {
					groundedByController.set(
						`${command.worldId}:${command.controllerId}`,
						true
					);
				}
			}

			if (payload.request?.type === "stepWorld") {
				result = {
					bodyStates: [],
					events: [],
					activeBodies: 1,
					sleepingBodies: 0,
					ccdBodies: 0,
				};
			}
			if (payload.request?.type === "isCharacterControllerGrounded") {
				result =
					groundedByController.get(
						`${payload.request.worldId}:${payload.request.controllerId}`
					) ?? false;
			}
		}

		const encoded = decoded.plugin.encodeResult({
			id: decoded.envelope.id,
			result,
		});
		worker.emitMessage(encoded.message);
	};
}

function createWorkerAdapter(adapterType, options) {
	if (adapterType === "rapier") {
		return new RapierWorkerPhysicsAdapter(options);
	}
	if (adapterType === "ammo") {
		return new AmmoWorkerPhysicsAdapter(options);
	}
	throw new Error(`Unknown worker adapter type "${adapterType}"`);
}

async function testPhysicsSystemStepAsyncUsesAdapterAsyncPath() {
	const adapter = new AsyncOnlyStepAdapter();
	const physics = new PhysicsSystem({ adapter });
	physics.initSync();
	physics.createWorld({
		worldId: "main",
		mode: "variable",
		gravity: { x: 0, y: 0, z: 0 },
	});

	const node = new Node();
	physics.attachBody(node, {
		worldId: "main",
		body: {
			type: "dynamic",
			linearVelocity: { x: 2, y: 0, z: 0 },
		},
		authority: "physics",
	});

	const report = await physics.stepAsync(0.1);
	assert.equal(adapter.syncStepCalls, 0);
	assert.equal(adapter.asyncStepCalls, 1);
	assert.ok(report.worldReports.length > 0);
	assert.ok(node.position.x > 0);
}

async function testWorkerAdapterFallsBackToPostMessageTransport(adapterType) {
	const observedModes = [];
	const dispatchPayloads = [];
	const adapter = createWorkerAdapter(adapterType, {
		strict: false,
		fallbackOnWorkerFailure: false,
		runtimeCapabilities: {
			sharedArrayBuffer: false,
			crossOriginIsolated: false,
		},
		createWorker: () =>
			new FakeWorker(createWorkerHandler(observedModes, dispatchPayloads)),
	});

	await adapter.init();
	adapter.createWorld({ worldId: "main", mode: "variable" });
	adapter.createBody(
		"main",
		"body-1",
		{ type: "dynamic" },
		{
			position: { x: 0, y: 0, z: 0 },
			rotation: [0, 0, 0, 1],
		}
	);
	adapter.addCollider(
		"main",
		"body-1",
		"collider-1",
		{
			mode: "explicit",
			shape: { kind: "sphere", radius: 1 },
		},
		{
			kind: "sphere",
			radius: 1,
		}
	);

	const stepResult = await adapter.stepWorldAsync("main", 0.016);
	assert.equal(stepResult.activeBodies, 1);
	assert.equal(dispatchPayloads.length, 1);
	assert.equal(dispatchPayloads[0].commands.length, 3);

	const stats = adapter.getWorkerPoolStats();
	assert.equal(stats?.transportMode, "post-message");
	assert.equal(stats?.transportPluginId, "post-message");
	assert.ok(observedModes.every((mode) => mode === "post-message"));
}

async function testWorkerAdapterPrefersSharedArrayBufferTransport(adapterType) {
	if (typeof SharedArrayBuffer !== "function") return;

	const observedModes = [];
	const dispatchPayloads = [];
	const adapter = createWorkerAdapter(adapterType, {
		strict: false,
		fallbackOnWorkerFailure: false,
		runtimeCapabilities: {
			sharedArrayBuffer: true,
			crossOriginIsolated: true,
		},
		createWorker: () =>
			new FakeWorker(createWorkerHandler(observedModes, dispatchPayloads)),
	});

	await adapter.init();
	adapter.createWorld({ worldId: "main", mode: "variable" });
	await adapter.stepWorldAsync("main", 0.016);
	assert.equal(dispatchPayloads.length, 1);

	const stats = adapter.getWorkerPoolStats();
	assert.equal(stats?.transportMode, "shared-array-buffer");
	assert.equal(stats?.transportPluginId, "shared-array-buffer");
	assert.ok(observedModes.every((mode) => mode === "shared-array-buffer"));
}

async function testWorkerAdapterRaycastAsync(adapterType) {
	const observedModes = [];
	const dispatchPayloads = [];
	const adapter = createWorkerAdapter(adapterType, {
		strict: false,
		fallbackOnWorkerFailure: false,
		runtimeCapabilities: {
			sharedArrayBuffer: false,
			crossOriginIsolated: false,
		},
		createWorker: () =>
			new FakeWorker(createWorkerHandler(observedModes, dispatchPayloads)),
	});

	await adapter.init();
	adapter.createWorld({ worldId: "main", mode: "variable" });
	adapter.createBody(
		"main",
		"body-1",
		{ type: "dynamic" },
		{
			position: { x: 0, y: 0, z: 0 },
			rotation: [0, 0, 0, 1],
		}
	);

	const hit = await adapter.raycastAsync("main", {
		origin: { x: 0, y: 0, z: 5 },
		direction: { x: 0, y: 0, z: -1 },
		maxDistance: 10,
	});
	assert.equal(hit?.bodyId, "body-1");
	assert.equal(dispatchPayloads.length, 1);
	assert.equal(dispatchPayloads[0].commands.length, 2);
	assert.equal(dispatchPayloads[0].request?.type, "raycast");
	assert.equal(dispatchPayloads[0].request?.worldId, "main");

	const secondHit = await adapter.raycastAsync("main", {
		origin: { x: 0, y: 1, z: 5 },
		direction: { x: 0, y: -1, z: -1 },
		maxDistance: 10,
	});
	assert.equal(secondHit?.bodyId, "body-1");
	assert.equal(dispatchPayloads.length, 2);
	assert.equal(dispatchPayloads[1].commands.length, 0);
	assert.equal(dispatchPayloads[1].request?.type, "raycast");
	assert.ok(observedModes.every((mode) => mode === "post-message"));
}

async function testPhysicsSystemRaycastAsyncUsesWorkerAdapter(adapterType) {
	const observedModes = [];
	const dispatchPayloads = [];
	const adapter = createWorkerAdapter(adapterType, {
		strict: false,
		fallbackOnWorkerFailure: false,
		runtimeCapabilities: {
			sharedArrayBuffer: false,
			crossOriginIsolated: false,
		},
		createWorker: () =>
			new FakeWorker(createWorkerHandler(observedModes, dispatchPayloads)),
	});

	const physics = new PhysicsSystem({ adapter });
	await physics.init();
	physics.createWorld({ worldId: "main", mode: "variable" });

	const hit = await physics.raycastAsync({
		worldId: "main",
		origin: { x: 0, y: 0, z: 5 },
		direction: { x: 0, y: 0, z: -1 },
		maxDistance: 10,
	});

	assert.equal(hit?.bodyId, "body-1");
	assert.equal(dispatchPayloads.length, 1);
	assert.equal(dispatchPayloads[0].commands.length, 1);
	assert.equal(dispatchPayloads[0].request?.type, "raycast");
	assert.ok(observedModes.every((mode) => mode === "post-message"));
}

async function testWorkerAdapterFallbackUpdatesCapabilities(adapterType) {
	const adapter = createWorkerAdapter(adapterType, {
		enabled: false,
		fallbackAdapter: new SimplePhysicsAdapter(`${adapterType}-fallback`),
	});
	const physics = new PhysicsSystem({ adapter });

	await physics.init();
	assert.equal(adapter.capabilities.characterController, true);
	assert.equal(adapter.capabilities.query, true);
	assert.equal(adapter.capabilities.shapeCast, true);
	assert.equal(adapter.capabilities.syncInit, false);

	physics.createWorld({
		worldId: "main",
		mode: "variable",
	});
	const node = new Node({ position: { x: 0, y: 1, z: 0 } });
	const body = physics.attachBody(node, {
		worldId: "main",
		body: { type: "kinematic" },
		authority: "animation",
	});
	const controller = physics.createCharacterController({
		worldId: "main",
		body,
		radius: 0.3,
		height: 1.7,
	});
	const move = controller.moveAndSlide({ x: 0, y: 0, z: 1 }, 0.016);
	assert.equal(move.moved.z, 1);
	assert.equal(typeof controller.isGrounded(), "boolean");
}

async function testRapierWorkerAdapterCharacterControllerSyncContract() {
	const dispatchPayloads = [];
	const adapter = createWorkerAdapter("rapier", {
		strict: false,
		fallbackOnWorkerFailure: false,
		runtimeCapabilities: {
			sharedArrayBuffer: false,
			crossOriginIsolated: false,
		},
		createWorker: () =>
			new FakeWorker(createCharacterControllerWorkerHandler(dispatchPayloads)),
	});
	const physics = new PhysicsSystem({ adapter });

	await physics.init();
	assert.equal(adapter.capabilities.characterController, true);

	physics.createWorld({
		worldId: "main",
		mode: "variable",
	});
	const node = new Node({ position: { x: 0, y: 1, z: 0 } });
	const body = physics.attachBody(node, {
		worldId: "main",
		body: { type: "kinematic" },
		authority: "animation",
	});

	const controller = physics.createCharacterController({
		worldId: "main",
		body,
		radius: 0.3,
		height: 1.7,
	});
	const move = controller.moveAndSlide({ x: 0, y: 0, z: 1 }, 0.016);
	assert.equal(move.moved.z, 1);
	assert.equal(move.grounded, false);

	await physics.stepAsync(0.016);
	assert.equal(controller.isGrounded(), true);

	assert.ok(
		dispatchPayloads.some(
			(payload) =>
				payload.commands.some(
					(command) => command.type === "createCharacterController"
				)
		),
		"Expected createCharacterController command to be dispatched to worker"
	);
	assert.ok(
		dispatchPayloads.some(
			(payload) => payload.request?.type === "isCharacterControllerGrounded"
		),
		"Expected grounded-state request to be dispatched after stepping"
	);
}

async function run() {
	await testPhysicsSystemStepAsyncUsesAdapterAsyncPath();
	await testWorkerAdapterFallsBackToPostMessageTransport("rapier");
	await testWorkerAdapterFallsBackToPostMessageTransport("ammo");
	await testWorkerAdapterPrefersSharedArrayBufferTransport("rapier");
	await testWorkerAdapterPrefersSharedArrayBufferTransport("ammo");
	await testWorkerAdapterRaycastAsync("rapier");
	await testWorkerAdapterRaycastAsync("ammo");
	await testPhysicsSystemRaycastAsyncUsesWorkerAdapter("rapier");
	await testPhysicsSystemRaycastAsyncUsesWorkerAdapter("ammo");
	await testWorkerAdapterFallbackUpdatesCapabilities("rapier");
	await testWorkerAdapterFallbackUpdatesCapabilities("ammo");
	await testRapierWorkerAdapterCharacterControllerSyncContract();
	console.log("Physics worker adapter tests passed");
}

run().catch((error) => {
	console.error(error);
	process.exit(1);
});
