import assert from "node:assert/strict";

import {
	PostProcessGraphCompiler,
	PostProcessPass,
	PostProcessPassRegistry,
	hasPostProcessExecutionPasses,
} from "../../../src/postprocess/index.ts";

class TestPostProcessPass extends PostProcessPass {
	constructor(config) {
		super({
			id: config.id,
			placement: config.placement,
			order: config.order,
			enabled: config.enabled ?? true,
			options: config.options ?? {},
			implementations: {
				test: {},
			},
		});
		this._requirements = config.requirements ?? {};
		this._histories = config.histories ?? [];
		this._transients = config.transients ?? [];
		this._shouldExecute = config.shouldExecute ?? true;
		this._signature = config.signature ?? config.id;
	}

	getRequirements() {
		return this._requirements;
	}

	getHistoryDescriptors() {
		return this._histories;
	}

	getTransientResourceDescriptors() {
		return this._transients;
	}

	getHistorySignature() {
		return this._signature;
	}

	shouldExecute() {
		return this._shouldExecute;
	}
}

function createFrameContext(overrides = {}) {
	return {
		viewCamera: {
			type: "perspective",
			fov: 60,
			aspectRatio: 16 / 9,
			near: 0.1,
			far: 1000,
		},
		attachments: {
			width: 128,
			height: 64,
		},
		incremental: {
			enabled: false,
			forceFullFrame: false,
			firstPass: null,
			postProcessStartPass: null,
			...(overrides.incremental ?? {}),
		},
		...(overrides.frameContext ?? {}),
	};
}

function createGBuffer(overrides = {}) {
	return {
		width: 128,
		height: 64,
		normalSpace: "world",
		depthEncoding: "linear-view-z",
		motionEncoding: "ndc-delta",
		channels: {
			color: { semantic: "color", width: 128, height: 64, handle: {} },
			depth: { semantic: "depth", width: 128, height: 64, handle: {} },
			...(overrides.channels ?? {}),
		},
		worldPosition: {
			source: "derived",
			available: false,
		},
		...overrides,
	};
}

function createRegistry() {
	const registry = new PostProcessPassRegistry();
	registry.registerPass(
		new TestPostProcessPass({
			id: "spatial-a",
			placement: "spatial",
			order: 3,
		})
	);
	registry.registerPass(
		new TestPostProcessPass({
			id: "need-motion",
			placement: "temporal",
			requirements: { gBuffer: ["motion"] },
		})
	);
	registry.registerPass(
		new TestPostProcessPass({
			id: "history-a",
			placement: "hdr",
			histories: [{ id: "history-shared", format: "rgba16float" }],
		})
	);
	registry.registerPass(
		new TestPostProcessPass({
			id: "history-conflict",
			placement: "hdr",
			order: 1,
			histories: [{ id: "history-shared", format: "rgba8unorm" }],
		})
	);
	registry.registerPass(
		new TestPostProcessPass({
			id: "transient-a",
			placement: "overlay",
			transients: [{ id: "scratch", format: "rgba16float" }],
		})
	);
	registry.registerPass(
		new TestPostProcessPass({
			id: "transient-conflict",
			placement: "overlay",
			order: 1,
			transients: [{ id: "scratch", format: "rgba8unorm" }],
		})
	);
	registry.registerPass(
		new TestPostProcessPass({
			id: "present-a",
			placement: "present",
		})
	);
	registry.registerPass(
		new TestPostProcessPass({
			id: "predicate-skip",
			placement: "present",
			shouldExecute: false,
		})
	);
	return registry;
}

function compile(registry, overrides = {}) {
	const warnings = [];
	const context = overrides.context ?? createFrameContext();
	const compiler = new PostProcessGraphCompiler();
	const graph = compiler.compile({
		postProcess: registry.createSnapshot("test"),
		backend: "test",
		frameContext: context,
		gBuffer: overrides.gBuffer ?? createGBuffer(),
		startPassId: overrides.startPassId,
		resolveImplementation: overrides.resolveImplementation,
		isGraphResourceAvailable: overrides.isGraphResourceAvailable,
		warn: (key, message) => warnings.push({ key, message }),
	});
	return { graph, warnings };
}

function testCompilerOrdersFiltersAndCollectsDescriptors() {
	const registry = createRegistry();
	const { graph, warnings } = compile(registry);

	assert.equal(
		hasPostProcessExecutionPasses(registry.createSnapshot("test"), {
			backend: "test",
			frameContext: createFrameContext(),
		}),
		true
	);
	assert.deepEqual(
		graph.orderedPasses.map((pass) => pass.id),
		[
			"spatial-a",
			"need-motion",
			"history-a",
			"history-conflict",
			"transient-a",
			"transient-conflict",
			"present-a",
		]
	);
	assert.deepEqual(
		graph.passes.map((pass) => pass.id),
		[
			"spatial-a",
			"history-a",
			"history-conflict",
			"transient-a",
			"transient-conflict",
			"present-a",
		]
	);
	assert.equal(graph.startPassId, null);
	assert.deepEqual(graph.historyDescriptors, [
		{ id: "history-shared", format: "rgba16float" },
	]);
	assert.deepEqual(graph.transientDescriptors, [
		{ id: "scratch", format: "rgba16float" },
	]);
	assert.deepEqual(
		warnings.map((warning) => warning.key),
		[
			"postprocess-requirement-missing-need-motion",
			"postprocess-history-conflict-history-shared",
			"postprocess-transient-conflict-scratch",
		]
	);
	assert.ok(graph.signature.includes("history-a:history-a"));
}

function testCompilerUsesIncrementalStartPass() {
	const registry = createRegistry();
	const { graph } = compile(registry, {
		context: createFrameContext({
			incremental: {
				enabled: true,
				forceFullFrame: false,
				firstPass: "postprocess",
				postProcessStartPass: "transient-a",
			},
		}),
	});

	assert.equal(graph.startPassId, "transient-a");
	assert.deepEqual(
		graph.passes.map((pass) => pass.id),
		["transient-a", "transient-conflict", "present-a"]
	);
	assert.deepEqual(graph.historyDescriptors, []);
}

function testCompilerHonorsExplicitStartPassAndSatisfiedRequirements() {
	const registry = createRegistry();
	const { graph, warnings } = compile(registry, {
		startPassId: "need-motion",
		gBuffer: createGBuffer({
			channels: {
				color: { semantic: "color", width: 128, height: 64, handle: {} },
				depth: { semantic: "depth", width: 128, height: 64, handle: {} },
				motion: { semantic: "motion", width: 128, height: 64, handle: {} },
			},
		}),
	});

	assert.equal(graph.startPassId, "need-motion");
	assert.equal(graph.passes[0].id, "need-motion");
	assert.equal(
		warnings.some(
			(warning) => warning.key === "postprocess-requirement-missing-need-motion"
		),
		false
	);
}

function testCompilerFiltersByBackendSharedResourceAvailability() {
	const registry = new PostProcessPassRegistry();
	registry.registerPass(new TestPostProcessPass({
		id: "required-unavailable",
		placement: "hdr",
		histories: [{ id: "unused-history", format: "rgba16float" }],
	}));
	registry.registerPass(new TestPostProcessPass({
		id: "optional-unavailable",
		placement: "hdr",
		order: 1,
	}));
	registry.registerPass(new TestPostProcessPass({
		id: "required-available",
		placement: "hdr",
		order: 2,
	}));

	const sharedUse = (id, optional = false) => ({
		color: { access: "read", output: "new-version" },
		backendShared: [{
			id,
			access: "read",
			usage: "sampled",
			...(optional ? { optional: true } : {}),
		}],
	});
	const metadata = new Map([
		["required-unavailable", sharedUse("backend:missing")],
		["optional-unavailable", sharedUse("backend:optional", true)],
		["required-available", sharedUse("backend:ready")],
	]);
	const availabilityChecks = [];
	const { graph, warnings } = compile(registry, {
		resolveImplementation: (pass) => ({
			metadata: { graph: metadata.get(pass.id) },
		}),
		isGraphResourceAvailable: (resourceId) => {
			availabilityChecks.push(resourceId);
			return resourceId === "backend:ready";
		},
	});

	assert.deepEqual(
		graph.passes.map((pass) => pass.id),
		["optional-unavailable", "required-available"]
	);
	assert.deepEqual(graph.historyDescriptors, []);
	assert.deepEqual(availabilityChecks, ["backend:missing", "backend:ready"]);
	assert.deepEqual(
		warnings.map((warning) => warning.key),
		["postprocess-backend-shared-unavailable-required-unavailable"]
	);
	assert.match(warnings[0].message, /backend:missing/);
}

testCompilerOrdersFiltersAndCollectsDescriptors();
testCompilerUsesIncrementalStartPass();
testCompilerHonorsExplicitStartPassAndSatisfiedRequirements();
testCompilerFiltersByBackendSharedResourceAvailability();

console.log("Postprocess graph compiler tests passed");
