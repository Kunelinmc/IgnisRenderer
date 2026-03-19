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

class FakeWorker {
	constructor(handler) {
		this._handler = handler;
		this._terminated = false;
		this._listeners = {
			message: new Set(),
			error: new Set(),
		};
		this.onmessage = null;
		this.onerror = null;
	}

	addEventListener(type, listener) {
		const set = this._listeners[type];
		if (!set) return;
		set.add(listener);
	}

	removeEventListener(type, listener) {
		const set = this._listeners[type];
		if (!set) return;
		set.delete(listener);
	}

	postMessage(message) {
		if (this._terminated) {
			throw new Error("Cannot postMessage on a terminated FakeWorker");
		}
		queueMicrotask(() => {
			if (this._terminated) return;
			this._handler(message, this);
		});
	}

	emitMessage(data) {
		if (this._terminated) return;
		const event = { data };
		if (typeof this.onmessage === "function") {
			this.onmessage(event);
		}
		for (const listener of this._listeners.message) {
			listener(event);
		}
	}

	terminate() {
		this._terminated = true;
	}
}

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

async function run() {
	await testPhysicsSystemStepAsyncUsesAdapterAsyncPath();
	await testWorkerAdapterFallsBackToPostMessageTransport("rapier");
	await testWorkerAdapterFallsBackToPostMessageTransport("ammo");
	await testWorkerAdapterPrefersSharedArrayBufferTransport("rapier");
	await testWorkerAdapterPrefersSharedArrayBufferTransport("ammo");
	console.log("Physics worker adapter tests passed");
}

run().catch((error) => {
	console.error(error);
	process.exit(1);
});
