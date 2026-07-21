import assert from "node:assert/strict";

import {
	PostProcessPass,
	PostProcessPassRegistry,
	PostProcessPlanner,
	hasPostProcessExecutionPasses,
} from "../../../src/postprocess/index.ts";

const COLOR = { access: "read", output: "new-version" };
const READ = { access: "read", usage: "sampled" };
const WRITE = { access: "write", usage: "storage" };

class TestPass extends PostProcessPass {
	constructor({ id, schedule, declaration, describe, implementation = true }) {
		super({
			id,
			schedule,
			enabled: true,
			implementations: implementation ? {
				software: () => ({
					id: `${id}:software`,
					describeExecution: describe ?? (() => declaration),
					execute: () => ({ ran: true }),
				}),
			} : {},
		});
	}
}

function frame(incremental = {}) {
	return {
		attachments: { width: 128, height: 64 },
		incremental: {
			enabled: false,
			forceFullFrame: false,
			firstPass: null,
			postProcessStartPass: null,
			...incremental,
		},
	};
}

function gBuffer(channels = {}) {
	return {
		width: 128,
		height: 64,
		normalSpace: "world",
		depthEncoding: "linear-view-z",
		channels: {
			color: {
				semantic: "color",
				width: 128,
				height: 64,
				handle: { backend: "software", data: new Uint8Array(4) },
			},
			depth: {
				semantic: "depth",
				width: 128,
				height: 64,
				handle: { backend: "software", data: new Float32Array(1) },
			},
			...channels,
		},
		worldPosition: { source: "derived", available: false },
	};
}

function plan(registry, overrides = {}) {
	const warnings = [];
	const context = overrides.frameContext ?? frame();
	const snapshot = registry.createSnapshot("software");
	const result = new PostProcessPlanner().plan({
		postProcess: snapshot,
		backend: "software",
		frameContext: context,
		gBuffer: overrides.gBuffer ?? gBuffer(),
		startPassId: overrides.startPassId,
		resolveImplementation: (pass) => overrides.missingIds?.has(pass.id) ?
			null : pass.getImplementation("software"),
		isSharedResourceAvailable: overrides.isSharedResourceAvailable,
		warn: (key, message) => warnings.push({ key, message }),
	});
	return { result, warnings, snapshot };
}

function testOrderingSlicingAvailabilityAndSingleDescribe() {
	let descriptions = 0;
	const registry = new PostProcessPassRegistry();
	registry.registerPass(new TestPass({
		id: "spatial",
		schedule: { placement: "spatial", order: 2 },
		describe: () => {
			descriptions++;
			return { color: COLOR };
		},
	}));
	registry.registerPass(new TestPass({
		id: "required-motion",
		schedule: { placement: "temporal" },
		declaration: {
			color: COLOR,
			gBuffer: [{ semantic: "motion", ...READ }],
		},
	}));
	registry.registerPass(new TestPass({
		id: "optional-motion",
		schedule: { placement: "temporal", order: 1 },
		declaration: {
			color: COLOR,
			gBuffer: [{ semantic: "motion", ...READ, optional: true }],
		},
	}));
	registry.registerPass(new TestPass({
		id: "missing",
		schedule: { placement: "hdr" },
		declaration: { color: COLOR },
	}));

	const { result, warnings, snapshot } = plan(registry, {
		missingIds: new Set(["missing"]),
	});
	assert.equal(hasPostProcessExecutionPasses(snapshot, {
		backend: "software",
		frameContext: frame(),
	}), true);
	assert.deepEqual(result.orderedPasses.map((entry) => entry.id), [
		"spatial",
		"required-motion",
		"optional-motion",
		"missing",
	]);
	assert.deepEqual(result.passes.map((entry) => entry.id), [
		"spatial",
		"optional-motion",
	]);
	assert.equal(descriptions, 1);
	assert.deepEqual(warnings.map((entry) => entry.key), [
		"postprocess-requirement-missing-required-motion",
		"postprocess-implementation-missing-missing",
	]);

	const sliced = plan(registry, {
		missingIds: new Set(["missing"]),
		frameContext: frame({
			enabled: true,
			firstPass: "postprocess",
			postProcessStartPass: "optional-motion",
		}),
	});
	assert.equal(sliced.result.startPassId, "optional-motion");
	assert.deepEqual(sliced.result.passes.map((entry) => entry.id), ["optional-motion"]);
}

function testDescriptorsAndDeterministicSignature() {
	const registry = new PostProcessPassRegistry();
	registry.registerPass(new TestPass({
		id: "temporal",
		schedule: { placement: "temporal" },
		declaration: {
			color: COLOR,
			histories: [{
				descriptor: { id: "taa", format: "rgba16float" },
				read: [READ],
				write: [WRITE],
			}],
			transients: [{
				descriptor: { id: "scratch", format: "rgba16float" },
				uses: [WRITE],
			}],
			shared: [{ id: "backend:ready", ...READ }],
		},
	}));
	const first = plan(registry, {
		isSharedResourceAvailable: (id) => id === "backend:ready",
	}).result;
	const second = plan(registry, {
		isSharedResourceAvailable: (id) => id === "backend:ready",
	}).result;
	assert.deepEqual(first.historyDescriptors, [{ id: "taa", format: "rgba16float" }]);
	assert.deepEqual(first.transientDescriptors, [{ id: "scratch", format: "rgba16float" }]);
	assert.equal(first.signature, second.signature);
}

testOrderingSlicingAvailabilityAndSingleDescribe();
testDescriptorsAndDeterministicSignature();

// Keep the conflict assertion separate so its message checks backend, pass and resource.
{
	const registry = new PostProcessPassRegistry();
	for (const [id, format] of [["first", "rgba16float"], ["second", "rgba8unorm"]]) {
		registry.registerPass(new TestPass({
			id,
			declaration: {
				color: COLOR,
				histories: [{
					descriptor: { id: "conflict", format },
					read: [READ],
					write: [WRITE],
				}],
			},
		}));
	}
	assert.throws(
		() => plan(registry),
		(error) => /software/.test(error.message) &&
			/second/.test(error.message) &&
			/conflict/.test(error.message),
	);
}

// Validate malformed declarations after the conflict behavior.
for (const [id, declaration] of [
	["illegal-color", { color: { access: "read", output: "preserve" } }],
	["empty-uses", {
		color: COLOR,
		transients: [{ descriptor: { id: "empty" }, uses: [] }],
	}],
	["duplicate-id", {
		color: COLOR,
		shared: [{ id: "same", ...READ }, { id: "same", ...READ }],
	}],
]) {
	const registry = new PostProcessPassRegistry();
	registry.registerPass(new TestPass({ id, declaration }));
	assert.throws(
		() => plan(registry),
		(error) => /software execution declaration/.test(error.message),
	);
}

console.log("Post-process planner tests passed");
