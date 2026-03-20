import assert from "node:assert/strict";
import { WorkerScheduler } from "../src/workers/WorkerScheduler.ts";
import {
	DEFAULT_WORKER_TRANSPORT_PLUGINS,
	decodeWorkerTaskEnvelope,
	encodeWorkerTaskResult,
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

async function testAutoFallbackToPostMessage() {
	const scheduler = new WorkerScheduler();
	scheduler.registerPool({
		id: "fallback",
		size: 1,
		runtimeCapabilities: {
			sharedArrayBuffer: false,
		},
		createWorker: () =>
			new FakeWorker((message, worker) => {
				const request = decodeWorkerTaskEnvelope(message, [
					postMessageWorkerTransportPlugin,
				]);
				assert.ok(request);
				worker.emitMessage({
					id: request.id,
					result: request.payload,
				});
			}),
	});

	const result = await scheduler.schedule("fallback", { value: 42 });
	assert.deepEqual(result, { value: 42 });

	const stats = scheduler.getPoolStats("fallback");
	assert.equal(stats?.transportMode, "post-message");
	assert.equal(stats?.transportPluginId, "post-message");
	assert.equal(stats?.sharedArrayBufferEnabled, false);
	scheduler.shutdownAll();
}

async function testSharedArrayBufferTransportRoundtrip() {
	if (typeof SharedArrayBuffer !== "function") {
		return;
	}
	const payload = {
		value: 99,
		meta: {
			label: "sab-structured",
			optional: undefined,
		},
		items: [1, undefined, { nested: true }],
	};
	const scheduler = new WorkerScheduler();
	scheduler.registerPool({
		id: "shared",
		size: 1,
		runtimeCapabilities: {
			sharedArrayBuffer: true,
			crossOriginIsolated: true,
		},
		transportPlugins: DEFAULT_WORKER_TRANSPORT_PLUGINS,
		createWorker: () =>
			new FakeWorker((message, worker) => {
				const request = decodeWorkerTaskEnvelope(message, [
					sharedArrayBufferWorkerTransportPlugin,
				]);
				assert.ok(request);
				const encoded = encodeWorkerTaskResult(
					{
						id: request.id,
						result: {
							ok: true,
							value: request.payload,
						},
					},
					sharedArrayBufferWorkerTransportPlugin
				);
				worker.emitMessage(encoded.message);
			}),
	});

	const result = await scheduler.schedule("shared", payload);
	assert.deepEqual(result, {
		ok: true,
		value: payload,
	});

	const stats = scheduler.getPoolStats("shared");
	assert.equal(stats?.transportMode, "shared-array-buffer");
	assert.equal(stats?.transportPluginId, "shared-array-buffer");
	assert.equal(stats?.sharedArrayBufferEnabled, true);
	scheduler.shutdownAll();
}

async function testCustomTransportPlugin() {
	const customTransportPlugin = {
		id: "custom-wrap",
		mode: "post-message",
		isSupported: () => true,
		encodeTask: (envelope) => ({
			message: {
				customTask: envelope,
			},
		}),
		decodeTask: (data) => {
			if (!data || typeof data !== "object") return null;
			const payload = data.customTask;
			if (!payload || typeof payload !== "object") return null;
			if (!Number.isFinite(payload.id)) return null;
			return {
				id: payload.id,
				payload: payload.payload,
			};
		},
		encodeResult: (envelope) => ({
			message: {
				customResult: envelope,
			},
		}),
		decodeResult: (data) => {
			if (!data || typeof data !== "object") return null;
			const payload = data.customResult;
			if (!payload || typeof payload !== "object") return null;
			if (!Number.isFinite(payload.id)) return null;
			return {
				id: payload.id,
				result: payload.result,
				error: typeof payload.error === "string" ? payload.error : undefined,
			};
		},
	};

	const scheduler = new WorkerScheduler();
	scheduler.registerPool({
		id: "custom",
		size: 1,
		transportPlugin: customTransportPlugin,
		createWorker: () =>
			new FakeWorker((message, worker) => {
				const request = customTransportPlugin.decodeTask(message);
				assert.ok(request);
				const response = customTransportPlugin.encodeResult({
					id: request.id,
					result: `custom:${request.payload.value}`,
				});
				worker.emitMessage(response.message);
			}),
	});

	const result = await scheduler.schedule("custom", { value: 7 });
	assert.equal(result, "custom:7");
	const stats = scheduler.getPoolStats("custom");
	assert.equal(stats?.transportPluginId, "custom-wrap");
	scheduler.shutdownAll();
}

async function run() {
	await testAutoFallbackToPostMessage();
	await testSharedArrayBufferTransportRoundtrip();
	await testCustomTransportPlugin();
	console.log("Worker transport plugin tests passed");
}

run().catch((error) => {
	console.error(error);
	process.exit(1);
});
