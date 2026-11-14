import assert from "node:assert/strict";

import * as postprocess from "../../../src/postprocess/index.ts";
import { BackendPostProcessRuntime } from "../../../src/postprocess/BackendPostProcessRuntime.ts";
import { createPostProcessResourceAccessor } from "../../../src/postprocess/PostProcessResourceAccessor.ts";
import { createPostProcessTestImplementations } from "../../helpers/postprocess.mjs";

const READ = { access: "read", usage: "sampled" };
const WRITE = { access: "write", usage: "cpu-write" };

function testBreakingPublicSurface() {
	for (const removed of [
		"PostProcessGraphCompiler",
		"PostProcessGraphMetadata",
		"PostProcessPassImplementationMetadata",
		"PostProcessPassMetadata",
	]) {
		assert.equal(removed in postprocess, false);
	}
	for (const retained of [
		"PostProcessPlanner",
		"PostProcessSubgraphBuilder",
		"PostProcessPass",
		"PostProcessPassRegistry",
	]) {
		assert.equal(retained in postprocess, true);
	}
}

class LabelPass extends postprocess.PostProcessPass {
	constructor(config = {}) {
		super({ id: "label-pass", ...config });
	}
}

function testPassLabelContract() {
	assert.equal(new LabelPass().label, "label-pass");
	assert.equal(new LabelPass({ label: "Readable pass" }).label, "Readable pass");
}

function testImplementationOwnershipAndLifecycleNotifications() {
	const created = [];
	const invalidated = [];
	const released = [];
	const pass = new LabelPass({
		implementations: {
			software: (backend) => {
				const implementation = {
					describeExecution: () => ({
						color: { access: "read-write", output: "preserve" },
					}),
					execute: () => ({ ran: true }),
					invalidate: () => invalidated.push(implementation),
					destroy: () => released.push(implementation),
				};
				created.push({ backend, implementation });
				return implementation;
			},
		},
	});
	const registry = new postprocess.PostProcessPassRegistry();
	registry.registerPass(pass);
	const passChanges = [];
	const registryChanges = [];
	pass.on("change", (change) => passChanges.push(change));
	registry.on("change", (change) => registryChanges.push(change));
	const firstOwner = createPostProcessTestImplementations();
	const secondOwner = createPostProcessTestImplementations();
	assert.equal(created.length, 0);
	const first = firstOwner.get(pass, "software");
	assert.strictEqual(firstOwner.get(pass, "software"), first);
	assert.equal(firstOwner.get(pass, "webgpu"), null);
	const second = secondOwner.get(pass, "software");
	assert.notStrictEqual(first, second);
	assert.notStrictEqual(created[0].backend, created[1].backend);
	assert.equal(created[0].backend.profile.id, "software");
	assert.equal(created.length, 2);

	pass.invalidate();
	pass.destroy();
	registry.invalidatePasses();
	registry.destroyPasses();
	const lifecycleChange = {
		passId: pass.id,
		builtIn: false,
		reason: "lifecycle",
	};
	assert.deepEqual(passChanges, Array(4).fill(lifecycleChange));
	assert.deepEqual(registryChanges, passChanges);
	assert.deepEqual(invalidated, [], "logical notifications must not invalidate owned instances");
	assert.deepEqual(released, [], "logical notifications must not release owned instances");

	firstOwner.destroy(pass);
	assert.deepEqual(released, [first]);
	assert.deepEqual(second.execute(), { ran: true });
	const replacement = firstOwner.get(pass, "software");
	assert.notStrictEqual(replacement, first);
	firstOwner.destroy();
	secondOwner.destroy();
	assert.deepEqual(released, [first, replacement, second]);
	assert.equal(passChanges.length, 4, "test cleanup must not emit logical lifecycle changes");
	registry.unregisterPass(pass.id);
	assert.deepEqual(registryChanges.at(-1), {
		passId: pass.id,
		builtIn: false,
		reason: "unregister",
	});
}

function testResourceAccessorContract() {
	const copied = [];
	const declaration = {
		color: { access: "read", output: "new-version" },
		gBuffer: [
			{ semantic: "depth", ...READ },
			{ semantic: "normal", ...READ, optional: true },
		],
		histories: [{
			descriptor: { id: "taa" },
			read: [READ],
			write: [WRITE],
		}],
		transients: [{
			descriptor: { id: "scratch" },
			uses: [WRITE],
		}],
		shared: [
			{ id: "required", ...READ },
			{ id: "optional", ...READ, optional: true },
		],
	};
	const accessor = createPostProcessResourceAccessor({
		passId: "accessor-test",
		declaration,
		colorInput: "input",
		colorOutput: "output",
		getGBuffer: (id) => id === "depth" ? "depth" : null,
		getHistory: (id) => id === "taa" ? {
			read: "history-read",
			write: "history-write",
			valid: true,
		} : null,
		getTransient: (id) => id === "scratch" ? "scratch" : null,
		getShared: (id) => id === "required" ? "shared" : null,
		copyGBufferToHistory: (semantic, id) => copied.push([semantic, id]),
	});

	assert.deepEqual(accessor.color, { input: "input", output: "output" });
	assert.equal(accessor.getGBuffer("depth"), "depth");
	assert.equal(accessor.getGBuffer("normal"), null);
	assert.deepEqual(accessor.getHistory("taa"), {
		read: "history-read",
		write: "history-write",
		valid: true,
	});
	assert.equal(accessor.getTransient("scratch"), "scratch");
	assert.equal(accessor.getShared("required"), "shared");
	assert.equal(accessor.getShared("optional"), null);
	accessor.copyGBufferToHistory("depth", "taa");
	assert.deepEqual(copied, [["depth", "taa"]]);
	assert.throws(() => accessor.getTransient("undeclared"), /undeclared transient/);
	assert.throws(() => accessor.getShared("undeclared"), /undeclared shared resource/);
}

class LifecyclePass extends postprocess.PostProcessPass {
	constructor(state) {
		super({
			id: "lifecycle",
			enabled: true,
			schedule: { placement: "temporal" },
			implementations: {
				software: (backend) => ({
					describeExecution: () => ({
						color: { access: "read", output: "new-version" },
						histories: [{
							descriptor: { id: "temporal", format: "rgba16float" },
							read: [READ],
							write: [WRITE],
						}],
					}),
					execute: (request) => {
						state.historyValidity.push(request.histories.temporal.valid);
						if (state.mode === "throw") throw new Error("execution failed");
						if (state.mode === "skip") return { ran: false };
						if (state.mode === "invalid-skip") {
							return { ran: false, updatedHistoryIds: ["temporal"] };
						}
						return { ran: true, updatedHistoryIds: ["temporal"] };
					},
					invalidate: () => state.implementationEvents.push(["invalidate", backend]),
					destroy: () => state.implementationEvents.push(["destroy", backend]),
				}),
			},
		});
	}
}

function createLifecycleHarness() {
	const state = {
		mode: "run",
		historyValidity: [],
		completions: 0,
		gBufferResourceModes: [],
		implementationEvents: [],
	};
	const registry = new postprocess.PostProcessPassRegistry();
	registry.registerPass(new LifecyclePass(state));
	const snapshot = registry.createSnapshot("software");
	const executor = {
		backend: "software",
		createGBufferBridge: (context, options = {}) => {
			state.gBufferResourceModes.push(options.resourceMode ?? "physical");
			return {
				width: context.attachments.width,
				height: context.attachments.height,
				normalSpace: "view",
				depthEncoding: "linear-view-z",
				channels: {
					color: {
						semantic: "color",
						width: context.attachments.width,
						height: context.attachments.height,
						handle: {
							backend: "software",
							data: context.attachments.pixels,
						},
					},
				},
				worldPosition: { source: "derived", available: false },
			};
		},
		createResource: (descriptor) => ({
			...descriptor,
			backend: "software",
			resource: new Float32Array(descriptor.width * descriptor.height * 4),
		}),
		destroyResource: () => {},
		createPassExecutionContext: () => Object.freeze({ resources: {} }),
		completePass: (_request, result) => {
			state.completions++;
			return { committed: result.ran !== false };
		},
	};
	const backend = { type: "software" };
	const runtime = new BackendPostProcessRuntime({ executor, backend });
	const context = {
		postProcess: snapshot,
		attachments: {
			width: 4,
			height: 4,
			pixels: new Uint8Array(4 * 4 * 4),
		},
		incremental: {
			enabled: false,
			forceFullFrame: false,
			dirtyRects: [],
			firstPass: null,
			temporalHistoryReset: false,
		},
	};
	return { state, runtime, context, executor, backend, registry };
}

async function testBackendOwnedImplementationLifecycle() {
	const { state, runtime, context, executor, backend, registry } = createLifecycleHarness();
	const otherBackend = { type: "software" };
	const otherRuntime = new BackendPostProcessRuntime({ executor, backend: otherBackend });
	const firstImplementation = runtime.planFrame(context).passes[0].implementation;
	const otherImplementation = otherRuntime.planFrame(context).passes[0].implementation;
	assert.notStrictEqual(firstImplementation, otherImplementation);
	const changes = [];
	registry.on("change", (change) => changes.push(change.reason));

	runtime.invalidateFrameSized();
	assert.deepEqual(state.implementationEvents, [["invalidate", backend]]);
	assert.deepEqual(changes, ["lifecycle"]);
	runtime.destroy();
	assert.deepEqual(state.implementationEvents, [
		["invalidate", backend],
		["destroy", backend],
	]);
	assert.deepEqual(changes, ["lifecycle", "lifecycle"]);
	assert.strictEqual(otherRuntime.planFrame(context).passes[0].implementation, otherImplementation);
	await otherRuntime.execute(context);
	otherRuntime.commitFrame();
	assert.deepEqual(state.historyValidity, [false]);
	otherRuntime.destroy();
	assert.deepEqual(state.implementationEvents.at(-1), ["destroy", otherBackend]);
	assert.deepEqual(changes, ["lifecycle", "lifecycle", "lifecycle"]);
}

async function testLifecycleCommitSkipAbortAndHistory() {
	const { state, runtime, context } = createLifecycleHarness();
	await runtime.execute(context);
	runtime.commitFrame();
	assert.deepEqual(state.gBufferResourceModes.slice(0, 2), [
		"synthetic",
		"physical",
	]);
	assert.deepEqual(state.historyValidity, [false]);
	assert.deepEqual(runtime.getDebugState().lastSuccessful.executedPassIds, ["lifecycle"]);

	await runtime.execute(context);
	runtime.commitFrame();
	assert.deepEqual(state.historyValidity, [false, true]);

	state.mode = "skip";
	await runtime.execute(context);
	runtime.commitFrame();
	assert.deepEqual(runtime.getDebugState().lastSuccessful.skippedPassIds, ["lifecycle"]);
	assert.equal(runtime.getDebugState().lastSuccessful.resolvedOutputColor, "scene-color");

	state.mode = "invalid-skip";
	await assert.rejects(() => runtime.execute(context), /cannot update history when ran is false/);
	assert.deepEqual(runtime.getDebugState().lastSuccessful.skippedPassIds, ["lifecycle"]);

	state.mode = "throw";
	await assert.rejects(() => runtime.execute(context), /execution failed/);
	assert.deepEqual(runtime.getDebugState().lastSuccessful.skippedPassIds, ["lifecycle"]);
}

testBreakingPublicSurface();
testPassLabelContract();
testImplementationOwnershipAndLifecycleNotifications();
testResourceAccessorContract();
await testBackendOwnedImplementationLifecycle();
await testLifecycleCommitSkipAbortAndHistory();

console.log("Post-process public contract and lifecycle tests passed");
