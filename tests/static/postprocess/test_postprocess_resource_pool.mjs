import assert from "node:assert/strict";

import { PostProcessResourcePool } from "../../../src/postprocess/index.ts";

function createExecutor() {
	return {
		backend: "test",
		created: [],
		destroyed: [],
		createResource(desc) {
			const handle = {
				id: desc.id,
				backend: this.backend,
				width: desc.width,
				height: desc.height,
				format: desc.format,
				mipMode: desc.mipMode,
				resource: {
					id: `${desc.id}:${this.created.length}`,
				},
			};
			this.created.push(handle);
			return handle;
		},
		destroyResource(handle) {
			this.destroyed.push(handle);
		},
		executePass() {
			return { ran: true };
		},
	};
}

function createGraph(overrides = {}) {
	return {
		backend: "test",
		postProcess: {},
		frameContext: {},
		gBuffer: {},
		width: overrides.width ?? 64,
		height: overrides.height ?? 32,
		orderedPasses: [],
		passes: [],
		startPassId: null,
		historyDescriptors:
			overrides.historyDescriptors ?? [
				{ id: "history", format: "rgba16float" },
			],
		transientDescriptors:
			overrides.transientDescriptors ?? [
				{ id: "scratch", format: "rgba16float" },
			],
		signature: overrides.signature ?? "camera-a|history",
	};
}

function testResourcePoolCommitsAndAbortsHistory() {
	const executor = createExecutor();
	const pool = new PostProcessResourcePool();
	const first = pool.prepare({
		executor,
		graph: createGraph(),
		reset: false,
	});

	assert.equal(first.transientsChanged, true);
	assert.deepEqual(
		executor.created.map((handle) => handle.id),
		["history:read", "history:write", "scratch"]
	);
	assert.equal(first.histories.history.valid, false);
	const firstScratch = first.transients.scratch.handle;

	pool.markUpdatedMany(["history"]);
	pool.commitFrame();
	const committed = pool.prepare({
		executor,
		graph: createGraph(),
		reset: false,
	});
	assert.equal(committed.transientsChanged, false);
	assert.strictEqual(committed.transients.scratch.handle, firstScratch);
	assert.equal(committed.histories.history.valid, true);
	assert.equal(committed.histories.history.read.id, "history:write");
	assert.equal(committed.histories.history.write.id, "history:read");

	pool.markUpdatedMany(["history"]);
	pool.abortFrame();
	const aborted = pool.prepare({
		executor,
		graph: createGraph(),
		reset: false,
	});
	assert.equal(aborted.histories.history.read.id, "history:write");
	assert.equal(aborted.histories.history.write.id, "history:read");
}

function testResourcePoolRecreatesTransientsAndInvalidatesFrameSized() {
	const executor = createExecutor();
	const pool = new PostProcessResourcePool();
	const first = pool.prepare({
		executor,
		graph: createGraph(),
		reset: false,
	});
	const firstScratch = first.transients.scratch.handle;

	const recreated = pool.prepare({
		executor,
		graph: createGraph({
			transientDescriptors: [{ id: "scratch", format: "rgba8unorm" }],
		}),
		reset: false,
	});
	assert.equal(recreated.transientsChanged, true);
	assert.notStrictEqual(recreated.transients.scratch.handle, firstScratch);
	assert.ok(executor.destroyed.includes(firstScratch));

	pool.markUpdatedMany(["history"]);
	pool.commitFrame();
	pool.invalidateFrameSized(executor);
	assert.equal(executor.destroyed.filter((handle) => handle.id === "history:read").length, 1);
	assert.equal(executor.destroyed.filter((handle) => handle.id === "history:write").length, 1);
	assert.ok(
		executor.destroyed.includes(recreated.transients.scratch.handle)
	);

	const afterInvalidate = pool.prepare({
		executor,
		graph: createGraph(),
		reset: false,
	});
	assert.equal(afterInvalidate.histories.history.valid, false);
	assert.equal(afterInvalidate.transientsChanged, true);

	pool.destroy(executor);
	assert.equal(
		executor.destroyed.includes(afterInvalidate.histories.history.read),
		true
	);
	assert.equal(
		executor.destroyed.includes(afterInvalidate.histories.history.write),
		true
	);
	assert.equal(
		executor.destroyed.includes(afterInvalidate.transients.scratch.handle),
		true
	);
}

testResourcePoolCommitsAndAbortsHistory();
testResourcePoolRecreatesTransientsAndInvalidatesFrameSized();

console.log("Postprocess resource pool tests passed");
