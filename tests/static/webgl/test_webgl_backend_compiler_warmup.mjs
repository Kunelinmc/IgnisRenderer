import assert from "node:assert/strict";import { WebGLProgramCompiler } from "../../../src/backends/webgl/WebGLProgramCompiler.ts";import { WebGLProgramWarmupQueue } from "../../../src/backends/webgl/WebGLProgramWarmupQueue.ts";import { createCompilerSlot, createProgramWarmupTrackingGL, CUSTOM_WEBGL_VERTEX, CUSTOM_WEBGL_FRAGMENT, runWebGLBackendFile } from "../../helpers/webgl-backend.mjs";

import { Logger } from "../../../src/foundation/Logger.ts";
import { Material, AlphaMode } from "../../../src/materials/Material.ts";
import { ShaderSource } from "../../../src/shaders/ShaderSource.ts";
import { WebGLSceneProgramRepository } from "../../../src/backends/webgl/WebGLSceneProgramRepository.ts";
import { WebGLSceneProgramWarmupContributor } from "../../../src/backends/webgl/WebGLSceneProgramPlanner.ts";
import { WebGLWarmupCoordinator } from "../../../src/backends/webgl/WebGLWarmupCoordinator.ts";
import { createTestDrawPacket } from "../helpers/drawPacket.mjs";

function testProgramCompilerParallelWarmupDefersStatusQueries() {
	const gl = createProgramWarmupTrackingGL({
		parallel: true,
		completeAfterPolls: 2,
	});
	const compiler = new WebGLProgramCompiler(gl);
	const slot = createCompilerSlot(compiler, "WebGLFXAAProgram", [
		"uSourceMap",
		"uTexelSize",
	]);
	const handle = slot.warmup();

	assert.equal(gl.calls.linkProgram, 1);
	assert.equal(gl.calls.getShaderParameter.length, 0);
	assert.equal(
		gl.calls.getProgramParameter.includes(gl.LINK_STATUS),
		false
	);
	assert.equal(gl.calls.getUniformLocation.length, 0);
	assert.equal(handle.isComplete(), false);
	assert.equal(handle.isComplete(), false);
	assert.equal(gl.calls.getShaderParameter.length, 0);
	assert.equal(
		gl.calls.getProgramParameter.includes(gl.LINK_STATUS),
		false
	);

	assert.equal(handle.isComplete(), true);
	handle.finalize();

	assert.deepEqual(gl.calls.getShaderParameter, [
		gl.COMPILE_STATUS,
		gl.COMPILE_STATUS,
	]);
	assert.ok(gl.calls.getProgramParameter.includes(gl.LINK_STATUS));
	assert.ok(gl.calls.getUniformLocation.includes("uSourceMap"));
	assert.ok(gl.calls.getUniformLocation.includes("uTexelSize"));
}

function testProgramCompilerFallbackWarmupBatchesBeforeFinalize() {
	const gl = createProgramWarmupTrackingGL();
	const compiler = new WebGLProgramCompiler(gl);

	const fxaa = createCompilerSlot(compiler, "WebGLFXAAProgram").warmup();
	const present = createCompilerSlot(compiler, "WebGLPresentProgram").warmup();

	assert.equal(gl.calls.linkProgram, 2);
	assert.equal(gl.calls.getShaderParameter.length, 0);
	assert.equal(
		gl.calls.getProgramParameter.includes(gl.LINK_STATUS),
		false
	);

	fxaa.finalize();
	present.finalize();

	assert.equal(gl.calls.getShaderParameter.length, 4);
	assert.equal(
		gl.calls.getProgramParameter.filter((parameter) => parameter === gl.LINK_STATUS)
			.length,
		2
	);
}

function testProgramCompilerTryGetDefersFallbackFinalization() {
	const gl = createProgramWarmupTrackingGL();
	let pendingNotifications = 0;
	const compiler = new WebGLProgramCompiler(
		gl,
		undefined,
		undefined,
		{
			onProgramCompilePending: () => {
				pendingNotifications++;
			},
		},
	);
	const fxaaSlot = createCompilerSlot(compiler, "WebGLFXAAProgram", [
		"uSourceMap",
		"uTexelSize",
	]);
	const bloomSlot = createCompilerSlot(compiler, "WebGLBloomProgram", [
		"uBloomParams",
	]);

	compiler.beginFrame();
	assert.equal(fxaaSlot.tryGet(), null);
	assert.equal(bloomSlot.tryGet(), null);
	assert.equal(pendingNotifications, 1);
	assert.equal(gl.calls.linkProgram, 2);
	assert.equal(gl.calls.getShaderParameter.length, 0);
	assert.equal(
		gl.calls.getProgramParameter.includes(gl.LINK_STATUS),
		false
	);
	assert.equal(gl.calls.getUniformLocation.length, 0);

	compiler.beginFrame();
	assert.equal(fxaaSlot.tryGet(), null);
	assert.equal(bloomSlot.tryGet(), null);
	assert.equal(pendingNotifications, 2);
	assert.equal(gl.calls.getShaderParameter.length, 0);
	assert.equal(
		gl.calls.getProgramParameter.includes(gl.LINK_STATUS),
		false
	);

	compiler.beginFrame();
	const fxaa = fxaaSlot.tryGet();
	const bloomPending = bloomSlot.tryGet();
	assert.ok(fxaa);
	assert.equal(bloomPending, null);
	assert.equal(pendingNotifications, 3);
	assert.equal(gl.calls.getShaderParameter.length, 2);
	assert.equal(
		gl.calls.getProgramParameter.filter((parameter) => parameter === gl.LINK_STATUS)
			.length,
		1
	);
	assert.ok(gl.calls.getUniformLocation.includes("uSourceMap"));
	assert.ok(gl.calls.getUniformLocation.includes("uTexelSize"));
	assert.equal(
		gl.calls.getUniformLocation.includes("uBloomParams"),
		false
	);

	compiler.beginFrame();
	const bloom = bloomSlot.tryGet();
	assert.ok(bloom);
	assert.equal(pendingNotifications, 3);
	assert.equal(gl.calls.getShaderParameter.length, 4);
	assert.equal(
		gl.calls.getProgramParameter.filter((parameter) => parameter === gl.LINK_STATUS)
			.length,
		2
	);
	assert.ok(gl.calls.getUniformLocation.includes("uBloomParams"));
}

function testProgramCompilerValidationIsOptIn() {
	const gl = createProgramWarmupTrackingGL({
		validateStatus: false,
	});
	const compiler = new WebGLProgramCompiler(gl);

	createCompilerSlot(compiler, "WebGLFXAAProgram").get();

	assert.equal(gl.calls.validateProgram, 0);
	assert.equal(
		gl.calls.getProgramParameter.includes(gl.VALIDATE_STATUS),
		false
	);
}

function testProgramCompilerValidationWarnsWhenEnabled() {
	const warnings = [];
	const gl = createProgramWarmupTrackingGL({
		validateStatus: false,
	});
	const compiler = new WebGLProgramCompiler(
		gl,
		undefined,
		undefined,
		{
			validatePrograms: true,
			warn: (key, message) => warnings.push({ key, message }),
		},
	);

	createCompilerSlot(compiler, "WebGLFXAAProgram").get();

	assert.equal(gl.calls.validateProgram, 1);
	assert.ok(gl.calls.getProgramParameter.includes(gl.VALIDATE_STATUS));
	assert.ok(
		warnings.some((warning) =>
			warning.key.startsWith("webgl-program-validate-WebGLFXAAProgram")
		)
	);
}

function testProgramCompilerSlotLifecycleAndStaleWarmup() {
	const gl = createProgramWarmupTrackingGL({
		parallel: true,
		completeAfterPolls: 100,
	});
	let deletedPrograms = 0;
	gl.deleteProgram = () => {
		deletedPrograms++;
	};
	const compiler = new WebGLProgramCompiler(gl);
	let sourceRevision = 0;
	let sourceResolutions = 0;
	const createSlot = () => compiler.createSlot({
		label: "WebGLLifecycleProgram",
		vertex: () => {
			sourceResolutions++;
			return `${CUSTOM_WEBGL_VERTEX}\n// revision ${sourceRevision}`;
		},
		fragment: () => {
			sourceResolutions++;
			return `${CUSTOM_WEBGL_FRAGMENT}\n// revision ${sourceRevision}`;
		},
		reflect: (_gl, program) => ({ program }),
	});
	const slot = createSlot();

	assert.throws(createSlot, /already registered/);
	const staleHandle = slot.warmup();
	assert.equal(compiler.getCompileState(slot.label), "pending");
	sourceRevision++;
	slot.invalidate();
	assert.equal(compiler.getCompileState(slot.label), "idle");
	assert.throws(() => staleHandle.isComplete(), /became stale/);
	assert.throws(() => staleHandle.finalize(), /became stale/);
	assert.equal(deletedPrograms, 1);

	const first = slot.get();
	assert.ok(first.program);
	assert.equal(sourceResolutions, 4);
	sourceRevision++;
	slot.invalidate();
	const second = slot.get();
	assert.ok(second.program);
	assert.notEqual(second.program, first.program);
	assert.equal(sourceResolutions, 6);
	assert.equal(deletedPrograms, 2);

	slot.destroy();
	slot.destroy();
	assert.equal(deletedPrograms, 3);
	const replacement = createSlot();
	replacement.get();
	compiler.destroy();
	compiler.destroy();
	assert.equal(deletedPrograms, 4);
	assert.throws(() => replacement.get(), /destroyed/);
}

async function testProgramWarmupQueuePrioritizesCoreWork() {
	const events = [];
	const queue = new WebGLProgramWarmupQueue({
		waitForSlice: async () => {},
	});
	const yieldController = { yieldIfNeeded: async () => {} };
	const createHandle = (label) => ({
		label,
		isComplete: () => true,
		finalize: () => {
			events.push(`finalize:${label}`);
		},
	});

	queue.enqueue({
		label: "post",
		priority: "postprocess",
		action: () => {
			events.push("action:post");
			return [createHandle("post")];
		},
	});
	queue.enqueue({
		label: "core",
		priority: "core",
		action: () => {
			events.push("action:core");
			return [createHandle("core")];
		},
	});
	queue.enqueue({
		label: "optional",
		priority: "optional",
		action: () => {
			events.push("action:optional");
			return [createHandle("optional")];
		},
	});

	const result = await queue.run(yieldController);

	assert.deepEqual(events, [
		"action:core",
		"finalize:core",
		"action:optional",
		"finalize:optional",
		"action:post",
		"finalize:post",
	]);
	assert.equal(result.compiled, 3);
	assert.equal(result.failed, 0);
}

async function testProgramWarmupQueueFinalizesOneProgramPerSlice() {
	let slice = 0;
	const finalizedAt = [];
	const queue = new WebGLProgramWarmupQueue({
		maxFinalizesPerSlice: 1,
		timeBudgetMs: Number.POSITIVE_INFINITY,
		waitForSlice: async () => {
			slice++;
		},
	});
	const yieldController = { yieldIfNeeded: async () => {} };

	queue.enqueue({
		label: "batch",
		priority: "core",
		action: () => ["a", "b", "c"].map((label) => ({
			label,
			isComplete: () => true,
			finalize: () => {
				finalizedAt.push(slice);
			},
		})),
	});

	const result = await queue.run(yieldController);

	assert.deepEqual(finalizedAt, [0, 1, 2]);
	assert.equal(result.compiled, 3);
	assert.equal(result.failed, 0);
}

function testProgramWarmupQueueTimeBoxesSliceWithInjectedClock() {
	let nowMs = 0;
	let slice = 0;
	const finalizedAt = [];
	const queue = new WebGLProgramWarmupQueue({
		timeBudgetMs: 10,
		now: () => (nowMs += 4),
		waitForSlice: async () => {
			slice++;
		},
	});
	const yieldController = { yieldIfNeeded: async () => {} };

	queue.enqueue({
		label: "batch",
		priority: "core",
		action: () => ["a", "b", "c", "d"].map((label) => ({
			label,
			isComplete: () => true,
			finalize: () => {
				finalizedAt.push(slice);
			},
		})),
	});

	return queue.run(yieldController).then((result) => {
		// The budget cuts the first slice after three finalizations; the
		// fourth still runs as the guaranteed first attempt of the next slice.
		assert.deepEqual(finalizedAt, [0, 0, 0, 1]);
		assert.equal(result.compiled, 4);
		assert.equal(result.failed, 0);
	});
}

async function testProgramWarmupQueueReportsStaleHandles() {
	const queue = new WebGLProgramWarmupQueue({
		waitForSlice: async () => {},
	});
	const yieldController = { yieldIfNeeded: async () => {} };

	queue.enqueue({
		label: "stale",
		priority: "core",
		action: () => [{
			label: "stale",
			isComplete: () => {
				throw new Error("stale handle");
			},
			finalize: () => {
				throw new Error("should not finalize");
			},
		}],
	});

	const result = await queue.run(yieldController);

	assert.equal(result.compiled, 0);
	assert.equal(result.failed, 1);
	assert.equal(result.errors[0].label, "stale");
	assert.match(String(result.errors[0].error), /stale handle/);
}

async function testProgramWarmupQueueObservesAbortSignal() {
	const controller = new AbortController();
	let sliceCount = 0;
	const queue = new WebGLProgramWarmupQueue({
		waitForSlice: async () => {
			sliceCount++;
			controller.abort(new Error("context lost"));
		},
	});
	queue.enqueue({
		label: "pending",
		priority: "core",
		action: () => [{
			label: "pending",
			isComplete: () => false,
			finalize: () => {
				throw new Error("should not finalize");
			},
		}],
	});

	await assert.rejects(
		queue.run({ yieldIfNeeded: async () => {} }, {}, controller.signal),
		(error) => error?.name === "AbortError",
	);
	assert.equal(sliceCount, 1);
}

function testProgramCompilerTracksForcedFinalizeStats() {
	const gl = createProgramWarmupTrackingGL();
	const compiler = new WebGLProgramCompiler(gl);
	const fxaaSlot = createCompilerSlot(compiler, "WebGLFXAAProgram", [
		"uSourceMap",
	]);
	compiler.beginFrame();

	assert.deepEqual(compiler.getStats(), {
		pendingCompiles: 0,
		forcedFinalizesThisFrame: 0,
		issuedCompilesThisFrame: 0,
	});

	assert.ok(fxaaSlot.get());
	assert.equal(compiler.getStats().forcedFinalizesThisFrame, 1);
	assert.equal(compiler.getStats().pendingCompiles, 0);
	fxaaSlot.get();
	assert.equal(compiler.getStats().forcedFinalizesThisFrame, 1);

	compiler.beginFrame();
	const bloomHandle = createCompilerSlot(
		compiler,
		"WebGLBloomProgram",
		["uBloomParams"],
	).warmup();
	assert.equal(compiler.getStats().pendingCompiles, 1);
	assert.equal(compiler.getStats().forcedFinalizesThisFrame, 0);
	bloomHandle.finalize();
	assert.equal(compiler.getStats().pendingCompiles, 0);

	compiler.beginFrame();
	assert.deepEqual(compiler.getStats(), {
		pendingCompiles: 0,
		forcedFinalizesThisFrame: 0,
		issuedCompilesThisFrame: 0,
	});
}

function testProgramCompilerIssueProgramCompileStartsWithoutHandles() {
	const gl = createProgramWarmupTrackingGL();
	const compiler = new WebGLProgramCompiler(gl);
	const slot = createCompilerSlot(compiler, "WebGLFXAAProgram", [
		"uSourceMap",
	]);
	compiler.beginFrame();

	assert.equal(
		compiler.issueProgramCompile(CUSTOM_WEBGL_VERTEX, CUSTOM_WEBGL_FRAGMENT, "WebGLFXAAProgram"),
		true
	);
	assert.equal(gl.calls.linkProgram, 1);
	assert.equal(compiler.getStats().pendingCompiles, 1);
	assert.equal(compiler.getStats().issuedCompilesThisFrame, 1);

	const mark = compiler.markWarmupHandles();
	assert.equal(
		compiler.issueProgramCompile(CUSTOM_WEBGL_VERTEX, CUSTOM_WEBGL_FRAGMENT, "WebGLFXAAProgram"),
		false
	);
	assert.equal(compiler.collectWarmupHandlesSince(mark).length, 0);
	assert.equal(gl.calls.linkProgram, 1);

	assert.ok(slot.get());
	assert.equal(gl.calls.linkProgram, 1);
	assert.equal(compiler.getStats().pendingCompiles, 0);
}

async function testProgramCompilerWaitPreservesSynchronousFallback() {
	const gl = createProgramWarmupTrackingGL();
	const compiler = new WebGLProgramCompiler(gl);
	const slot = createCompilerSlot(compiler, "fallback");
	compiler.issueProgramCompile(CUSTOM_WEBGL_VERTEX, CUSTOM_WEBGL_FRAGMENT, slot.label);
	await compiler.waitForPendingCompiles([slot.label]);
	assert.equal(gl.calls.getShaderParameter.length, 0);
	assert.equal(gl.calls.getProgramParameter.length, 0);
	assert.ok(slot.get());
	assert.ok(gl.calls.getProgramParameter.includes(gl.LINK_STATUS));
	compiler.destroy();
}

async function testProgramCompilerWaitRejectsCancelledOrStaleWork() {
	for (const interruption of ["abort", "invalidate", "destroy", "slot"]) {
		const gl = createProgramWarmupTrackingGL({ parallel: true, completeAfterPolls: Infinity });
		const compiler = new WebGLProgramCompiler(gl);
		const slot = createCompilerSlot(compiler, "interrupted");
		compiler.issueProgramCompile(CUSTOM_WEBGL_VERTEX, CUSTOM_WEBGL_FRAGMENT, slot.label);
		const controller = new AbortController();
		const waiting = compiler.waitForPendingCompiles([slot.label], controller.signal);
		const rejected = assert.rejects(waiting, /cancelled|invalidated|destroyed/);
		if (interruption === "abort") controller.abort(new Error("cancelled"));
		else if (interruption === "slot") slot.invalidate();
		else compiler[interruption]();
		await rejected;
		assert.equal(gl.calls.getShaderParameter.length, 0);
		assert.equal(gl.calls.getProgramParameter.includes(gl.LINK_STATUS), false);
		compiler.destroy();
	}
}

async function testProgramCompilerTimingsAreOptInAndSeparateWait() {
	const records = [];
	Logger.configure({ level: "info", sink: { debug: (...args) => records.push(args) } });
	const gl = createProgramWarmupTrackingGL({ parallel: true, completeAfterPolls: 1 });
	const compiler = new WebGLProgramCompiler(gl);
	try {
		createCompilerSlot(compiler, "quiet").get();
		assert.equal(records.length, 0);
		Logger.setLevel("debug");
		const slot = createCompilerSlot(compiler, "measured");
		compiler.issueProgramCompile(CUSTOM_WEBGL_VERTEX, CUSTOM_WEBGL_FRAGMENT, slot.label);
		await compiler.waitForPendingCompiles([slot.label]);
		const completedPolls = gl.calls.getProgramParameter.length;
		await compiler.waitForPendingCompiles([slot.label]);
		assert.equal(gl.calls.getProgramParameter.length, completedPolls,
			"completed alternatives must not be polled again before first use");
		assert.equal(gl.calls.getUniformLocation.length, 0);
		assert.ok(slot.get());
		const timings = records.map((entry) => entry.at(-1));
		assert.deepEqual(timings.map((entry) => entry.phase), ["issue", "wait", "finalize"]);
		for (const entry of timings) {
			assert.equal(entry.parallel, true);
			assert.equal(entry.programCount, 1);
			assert.ok(Number.isFinite(entry.durationMs) && entry.durationMs >= 0);
		}
		assert.equal(timings[0].label, "measured");
		assert.equal(timings[2].label, "measured");
		await compiler.waitForPendingCompiles([slot.label]);
		slot.get();
		assert.equal(records.length, 3, "warm-cache frames must not emit compile timings");
	} finally {
		compiler.destroy();
		Logger.reset();
	}
}

async function testSceneWarmupHonorsSchedulingAndCancellation() {
	const results = [];
	for (const spec of [
		{ name: "disabled", options: { scheduling: "immediate", yieldIntervalMs: 0 } },
		{ name: "budget", options: { scheduling: "immediate", yieldIntervalMs: 15 } },
		{ name: "idle", options: { scheduling: "idle", yieldIntervalMs: 4 } },
		{ name: "cancel-yield", options: { yieldIntervalMs: 4 }, cancelYield: true },
		{ name: "cancel-source", options: { yieldIntervalMs: 4 }, cancelSource: true },
	]) {
		ShaderSource.clearCache("webgl");
		const compiler = new WebGLProgramCompiler(createProgramWarmupTrackingGL());
		const repository = new WebGLSceneProgramRepository({ compiler });
		const contributor = new WebGLSceneProgramWarmupContributor(repository, false, undefined);
		// Isolate source scheduling from the queue's independent finalization slices.
		const coordinator = new WebGLWarmupCoordinator({ compiler, contributors: [{
			collectWarmupTasks: (request) => contributor.collectWarmupTasks(request)
				.filter((task) => task.label === "WebGLSceneSource:builtin"),
		}] });
		const materials = [new Material(), new Material({ alphaMode: AlphaMode.Mask })];
		const context = { scene: {
			opaquePackets: materials.map((material) => createTestDrawPacket({ material })),
			transparentPackets: [], lights: [],
		}, features: {} };
		const plan = { sceneTargetMode: "single", materials };
		const controller = new AbortController();
		const originalPrepare = ShaderSource.prepare;
		const originalNow = performance.now;
		const originalTimeout = globalThis.setTimeout;
		const originalIdle = globalThis.requestIdleCallback;
		let now = 0;
		let sources = 0;
		let timers = 0;
		let idleCalls = 0;
		let afterAbort = 0;
		try {
			performance.now = () => now;
			ShaderSource.prepare = async (...args) => {
				sources++;
				if (controller.signal.aborted) afterAbort++;
				const result = await originalPrepare.apply(ShaderSource, args);
				now += 5;
				if (spec.cancelSource) controller.abort();
				return result;
			};
			globalThis.setTimeout = (callback) => {
				timers++;
				queueMicrotask(() => {
					if (spec.cancelYield) controller.abort();
					callback();
				});
				return 0;
			};
			globalThis.requestIdleCallback = (callback) => {
				idleCalls++;
				queueMicrotask(callback);
				return 0;
			};
			let errorName = null;
			try {
				await coordinator.warmup(context, plan, spec.options, undefined, controller.signal);
				// Cached sources must skip both preparation and yield checks.
				await coordinator.warmup(context, plan, spec.options, undefined, controller.signal);
			} catch (error) {
				errorName = error.name;
			}
			results.push({ name: spec.name, sources, timers, idleCalls, afterAbort, errorName });
		} finally {
			ShaderSource.prepare = originalPrepare;
			performance.now = originalNow;
			globalThis.setTimeout = originalTimeout;
			if (originalIdle === undefined) delete globalThis.requestIdleCallback;
			else globalThis.requestIdleCallback = originalIdle;
			repository.destroy();
			compiler.destroy();
		}
	}
	assert.deepEqual(results, [
		{ name: "disabled", sources: 4, timers: 0, idleCalls: 0, afterAbort: 0, errorName: null },
		{ name: "budget", sources: 4, timers: 1, idleCalls: 0, afterAbort: 0, errorName: null },
		{ name: "idle", sources: 4, timers: 0, idleCalls: 2, afterAbort: 0, errorName: null },
		{ name: "cancel-yield", sources: 2, timers: 1, idleCalls: 0, afterAbort: 0,
			errorName: "AbortError" },
		{ name: "cancel-source", sources: 1, timers: 0, idleCalls: 0, afterAbort: 0,
			errorName: "AbortError" },
	]);
}

await runWebGLBackendFile([
	testProgramCompilerParallelWarmupDefersStatusQueries,
	testProgramCompilerFallbackWarmupBatchesBeforeFinalize,
	testProgramCompilerTryGetDefersFallbackFinalization,
	testProgramCompilerValidationIsOptIn,
	testProgramCompilerValidationWarnsWhenEnabled,
	testProgramCompilerSlotLifecycleAndStaleWarmup,
	testProgramWarmupQueuePrioritizesCoreWork,
	testProgramWarmupQueueFinalizesOneProgramPerSlice,
	testProgramWarmupQueueTimeBoxesSliceWithInjectedClock,
	testProgramWarmupQueueReportsStaleHandles,
	testProgramWarmupQueueObservesAbortSignal,
	testProgramCompilerTracksForcedFinalizeStats,
	testProgramCompilerIssueProgramCompileStartsWithoutHandles,
	testProgramCompilerWaitPreservesSynchronousFallback,
	testProgramCompilerWaitRejectsCancelledOrStaleWork,
	testProgramCompilerTimingsAreOptInAndSeparateWait,
	testSceneWarmupHonorsSchedulingAndCancellation,
], "WebGL compiler and warmup tests");
