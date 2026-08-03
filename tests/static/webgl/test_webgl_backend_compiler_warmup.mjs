import assert from "node:assert/strict";import { WebGLProgramCompiler } from "../../../src/backends/webgl/WebGLProgramCompiler.ts";import { WebGLProgramWarmupQueue } from "../../../src/backends/webgl/WebGLProgramWarmupQueue.ts";import { createCompilerSlot, createProgramWarmupTrackingGL, CUSTOM_WEBGL_VERTEX, CUSTOM_WEBGL_FRAGMENT, runWebGLBackendFile } from "../../helpers/webgl-backend.mjs";

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

await runWebGLBackendFile([
	testProgramCompilerParallelWarmupDefersStatusQueries,
	testProgramCompilerFallbackWarmupBatchesBeforeFinalize,
	testProgramCompilerTryGetDefersFallbackFinalization,
	testProgramCompilerValidationIsOptIn,
	testProgramCompilerValidationWarnsWhenEnabled,
	testProgramCompilerSlotLifecycleAndStaleWarmup,
	testProgramWarmupQueuePrioritizesCoreWork,
	testProgramWarmupQueueFinalizesOneProgramPerSlice,
	testProgramWarmupQueueReportsStaleHandles,
	testProgramWarmupQueueObservesAbortSignal,
], "WebGL compiler and warmup tests");
