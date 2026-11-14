import assert from "node:assert/strict";
import { Logger } from "../../../src/foundation/Logger.ts";
import { BackendPostProcessRuntime } from "../../../src/postprocess/BackendPostProcessRuntime.ts";

import {
	PostProcessPass,
	PostProcessPassRegistry,
	PostProcessPlanner,
	hasPostProcessExecutionPasses,
} from "../../../src/postprocess/index.ts";
import { createPostProcessTestImplementations } from "../../helpers/postprocess.mjs";

const testImplementations = createPostProcessTestImplementations();

const COLOR = { access: "read", output: "new-version" };
const READ = { access: "read", usage: "sampled" };
const WRITE = { access: "write", usage: "storage" };

class TestPass extends PostProcessPass {
	constructor({
		id,
		schedule,
		declaration,
		describe,
		implementation = true,
		alphaContract,
	}) {
		super({
			id,
			schedule,
			alphaContract,
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
		alphaMode: "opaque",
		attachments: { width: 128, height: 64 },
		incremental: {
			enabled: false,
			forceFullFrame: false,
			firstPass: null,
			...incremental,
		},
	};
}

function testTransparentOutputAlphaCompatibility() {
	const incompatible = new PostProcessPassRegistry();
	incompatible.registerPass(new TestPass({
		id: "legacy-alpha",
		declaration: { color: COLOR },
	}));
	const transparentFrame = frame();
	transparentFrame.alphaMode = "premultiplied";
	assert.throws(
		() => plan(incompatible, { frameContext: transparentFrame }),
		(error) => /legacy-alpha/.test(error.message) &&
			/alphaContract/.test(error.message),
	);

	const compatible = new PostProcessPassRegistry();
	compatible.registerPass(new TestPass({
		id: "alpha-aware",
		alphaContract: "premultiplied",
		declaration: { color: COLOR },
	}));
	assert.deepEqual(
		plan(compatible, { frameContext: transparentFrame }).result.passes
			.map((entry) => entry.id),
		["alpha-aware"],
	);
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
	Logger.configure({
		resetOnceKeys: true,
		sink: { warn: (_scope, message) => warnings.push({
			key: message.match(/^\[([^\]]+)\]/)?.[1], message,
		}) },
	});
	const context = overrides.frameContext ?? frame();
	const snapshot = registry.createSnapshot("software");
	let result;
	try {
		result = new PostProcessPlanner().plan({
			postProcess: snapshot,
			backend: "software",
			frameContext: context,
			gBuffer: overrides.gBuffer ?? gBuffer(),
			resolveImplementation: (pass) => overrides.missingIds?.has(pass.id) ?
				null : testImplementations.get(pass, "software"),
			isSharedResourceAvailable: overrides.isSharedResourceAvailable,
		});
	} finally {
		Logger.reset();
	}
	return { result, warnings, snapshot };
}

function testOrderingAvailabilityAndCompleteIncrementalSequence() {
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
		"postprocess-implementation-missing-missing",
		"postprocess-requirement-missing-required-motion",
	]);

	const incremental = plan(registry, {
		missingIds: new Set(["missing"]),
		frameContext: frame({
			enabled: true,
			firstPass: "postprocess",
		}),
	});
	assert.deepEqual(incremental.result.passes.map((entry) => entry.id), [
		"spatial",
		"optional-motion",
	]);
	assert.equal(descriptions, 2);
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

function testDeclarationPlanFinalizesAvailabilityWithoutRedescribing() {
	let descriptions = 0;
	const registry = new PostProcessPassRegistry();
	registry.registerPass(new TestPass({
		id: "shared-reader",
		declaration: null,
		describe: () => {
			descriptions++;
			return {
				color: COLOR,
				shared: [{ id: "backend:frame-hiz", ...READ }],
			};
		},
	}));
	const context = frame();
	const snapshot = registry.createSnapshot("software");
	const planner = new PostProcessPlanner();
	const declarations = planner.describe({
		postProcess: snapshot,
		backend: "software",
		frameContext: context,
		gBuffer: gBuffer(),
		resolveImplementation: (pass) => testImplementations.get(pass, "software"),
	});
	assert.equal(descriptions, 1);
	assert.deepEqual(declarations.passes.map((entry) => entry.id), ["shared-reader"]);

	const warnings = [];
	Logger.configure({
		resetOnceKeys: true,
		sink: { warn: (_scope, message) => warnings.push({
			key: message.match(/^\[([^\]]+)\]/)?.[1], message,
		}) },
	});
	const unavailable = planner.finalize(declarations, {
		gBuffer: gBuffer(),
		isSharedResourceAvailable: () => false,
	});
	assert.equal(descriptions, 1);
	assert.deepEqual(unavailable.passes, []);
	assert.deepEqual(warnings.map((entry) => entry.key), [
		"postprocess-backend-shared-unavailable-shared-reader",
	]);

	const available = planner.finalize(declarations, {
		gBuffer: gBuffer(),
		isSharedResourceAvailable: () => true,
	});
	assert.equal(descriptions, 1);
	assert.deepEqual(available.passes.map((entry) => entry.id), ["shared-reader"]);
	Logger.reset();
}

function testWarmupSilencePreservesRenderWarnings() {
	const registry = new PostProcessPassRegistry();
	const missing = new TestPass({ id: "missing-runtime", declaration: { color: COLOR } });
	registry.registerPass(missing);
	registry.registerPass(new TestPass({
		id: "missing-motion",
		declaration: { color: COLOR, gBuffer: [{ semantic: "motion", ...READ }] },
	}));
	registry.registerPass(new TestPass({
		id: "missing-shared",
		declaration: { color: COLOR, shared: [{ id: "missing", ...READ }] },
	}));
	const context = { ...frame(), postProcess: registry.createSnapshot("software") };
	// Retain snapshot eligibility while simulating an unavailable runtime implementation.
	missing.getImplementationFactory = () => null;
	const executor = {
		backend: "software",
		createGBufferBridge: () => gBuffer(),
		isGraphResourceAvailable: () => false,
	};
	const warnings = [];
	Logger.configure({
		resetOnceKeys: true,
		sink: { warn: (...args) => warnings.push(args.join(" ")) },
	});
	try {
		const runtime = new BackendPostProcessRuntime({ executor, backend: {} });
		assert.deepEqual(runtime.planWarmup(context).passes.map((pass) => pass.id), [
			"missing-shared",
		]);
		assert.deepEqual(warnings, [], "warmup must remain silent");
		Logger.warn("unrelated warning");
		assert.equal(warnings.length, 1, "warmup must not silence the global logger");
		assert.deepEqual(runtime.planFrame(context).passes, []);
		assert.equal(warnings.length, 4);
		for (const key of [
			"postprocess-implementation-missing-missing-runtime",
			"postprocess-requirement-missing-missing-motion",
			"postprocess-backend-shared-unavailable-missing-shared",
		]) assert.ok(warnings.some((message) => message.includes(key)), key);
		new BackendPostProcessRuntime({ executor, backend: {} }).planFrame(context);
		assert.equal(warnings.length, 4, "render warnings retain global deduplication");
	} finally {
		Logger.reset();
	}
}

testOrderingAvailabilityAndCompleteIncrementalSequence();
testDescriptorsAndDeterministicSignature();
testDeclarationPlanFinalizesAvailabilityWithoutRedescribing();
testTransparentOutputAlphaCompatibility();
testWarmupSilencePreservesRenderWarnings();

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

{
	const registry = new PostProcessPassRegistry();
	for (const id of ["jitter-a", "jitter-b"]) {
		registry.registerPass(new TestPass({
			id,
			declaration: {
				color: COLOR,
				frameRequirements: {
					cameraJitter: { sequence: "halton-2-3", scale: 1 },
				},
			},
		}));
	}
	assert.deepEqual(plan(registry).result.frameRequirements, {
		cameraJitter: { sequence: "halton-2-3", scale: 1 },
	});
}

{
	const registry = new PostProcessPassRegistry();
	for (const [id, scale] of [["jitter-a", 1], ["jitter-b", 0.5]]) {
		registry.registerPass(new TestPass({
			id,
			declaration: {
				color: COLOR,
				frameRequirements: {
					cameraJitter: { sequence: "halton-2-3", scale },
				},
			},
		}));
	}
	assert.throws(
		() => plan(registry),
		(error) => /jitter-a/.test(error.message) && /jitter-b/.test(error.message),
	);
}

{
	const registry = new PostProcessPassRegistry();
	registry.registerPass(new TestPass({
		id: "ineligible-jitter",
		declaration: {
			color: COLOR,
			gBuffer: [{ semantic: "motion", ...READ }],
			frameRequirements: {
				cameraJitter: { sequence: "halton-2-3", scale: 1 },
			},
		},
	}));
	assert.deepEqual(plan(registry).result.frameRequirements, {});
}

testImplementations.destroy();
console.log("Post-process planner tests passed");
