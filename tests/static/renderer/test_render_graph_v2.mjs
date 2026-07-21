import assert from "node:assert/strict";

import { RenderGraphBuilder } from "../../../src/rendergraph/RenderGraphBuilder.ts";
import { RenderGraphAttemptTracker } from "../../../src/rendergraph/RenderGraphAttemptTracker.ts";
import { RenderGraphCompiler } from "../../../src/rendergraph/RenderGraphCompiler.ts";

const compiler = new RenderGraphCompiler();

function texture(id, origin = "imported", overrides = {}) {
	return {
		id,
		origin,
		kind: "texture",
		residency: origin === "graph" ? "transient" : "frame",
		initialContent: origin === "graph" ? "undefined" : "valid",
		format: "rgba8unorm",
		width: 16,
		height: 16,
		depthOrArrayLayers: 2,
		dimension: "2d",
		sampleCount: 1,
		mipLevelCount: 4,
		...overrides,
	};
}

function node(id, resources, overrides = {}) {
	return {
		id,
		stage: "test",
		kind: "test",
		label: id,
		resources,
		...overrides,
	};
}

function testSubresourceAndPhysicalHazards() {
	const graph = compiler.compile({
		resources: [
			texture("texture-a"),
			texture("texture-b"),
			texture("texture-c"),
			{
				id: "buffer-a",
				origin: "imported",
				kind: "buffer",
				residency: "frame",
				initialContent: "valid",
				size: 1024,
			},
		],
		bindings: [
			{ resourceId: "texture-a", physicalId: "physical:shared", kind: "texture" },
			{ resourceId: "texture-b", physicalId: "physical:shared", kind: "texture" },
			{ resourceId: "texture-c", physicalId: "physical:other", kind: "texture" },
			{ resourceId: "buffer-a", physicalId: "physical:buffer", kind: "buffer" },
		],
		nodes: [
			node("write-mip-0", [{
				resource: "texture-a",
				access: "write",
				usage: "color-attachment",
				subresource: { kind: "texture", mipStart: 0, mipCount: 1, layerStart: 0, layerCount: 1 },
			}]),
			node("read-other-mip", [{
				resource: "texture-b",
				access: "read",
				usage: "sampled",
				subresource: { kind: "texture", mipStart: 1, mipCount: 1, layerStart: 0, layerCount: 1 },
			}]),
			node("read-alias-mip-0", [{
				resource: "texture-b",
				access: "read",
				usage: "sampled",
				subresource: { kind: "texture", mipStart: 0, mipCount: 1, layerStart: 0, layerCount: 1 },
			}]),
			node("read-other-physical", [{
				resource: "texture-c",
				access: "read",
				usage: "sampled",
			}]),
			node("write-buffer-head", [{
				resource: "buffer-a",
				access: "write",
				usage: "storage",
				subresource: { kind: "buffer", offset: 0, size: 128 },
			}]),
			node("read-buffer-tail", [{
				resource: "buffer-a",
				access: "read",
				usage: "storage",
				subresource: { kind: "buffer", offset: 256, size: 128 },
			}]),
			node("read-buffer-head", [{
				resource: "buffer-a",
				access: "read",
				usage: "storage",
				subresource: { kind: "buffer", offset: 64, size: 64 },
			}]),
		],
	});

	assert.deepEqual(graph.nodes.map((entry) => entry.id), [
		"write-mip-0",
		"read-other-mip",
		"read-alias-mip-0",
		"read-other-physical",
		"write-buffer-head",
		"read-buffer-tail",
		"read-buffer-head",
	]);
	assert.ok(graph.dependencies.some((edge) =>
		edge.fromNodeId === "write-mip-0" && edge.toNodeId === "read-alias-mip-0" &&
		edge.physicalId === "physical:shared"));
	assert.ok(!graph.dependencies.some((edge) => edge.toNodeId === "read-other-mip"));
	assert.ok(!graph.dependencies.some((edge) => edge.toNodeId === "read-other-physical"));
	assert.ok(graph.dependencies.some((edge) =>
		edge.fromNodeId === "write-buffer-head" && edge.toNodeId === "read-buffer-head"));
	assert.ok(!graph.dependencies.some((edge) => edge.toNodeId === "read-buffer-tail"));

	const generationAliasGraph = compiler.compile({
		resources: [
			texture("alias-a"),
			texture("alias-b", "graph"),
			texture("alias-output", "graph"),
		],
		bindings: [
			{ resourceId: "alias-a", physicalId: "physical:generation-shared", kind: "texture" },
			{ resourceId: "alias-b", physicalId: "physical:generation-shared", kind: "texture" },
		],
		nodes: [
			{ ...node("create-alias-b", []), creates: ["alias-b"] },
			node("write-alias-a", [{
				resource: "alias-a",
				access: "write",
				usage: "color-attachment",
			}], { retention: "if-reachable" }),
			node("consume-alias-b", [
				{ resource: "alias-b", access: "read", usage: "sampled" },
				{ resource: "alias-output", access: "write", usage: "color-attachment" },
			], { retention: "if-reachable" }),
		],
		exports: [{ resource: "alias-output" }],
	});
	assert.deepEqual(generationAliasGraph.nodes.map((entry) => entry.id), [
		"create-alias-b",
		"write-alias-a",
		"consume-alias-b",
	]);
	assert.ok(generationAliasGraph.dependencies.some((edge) =>
		edge.fromNodeId === "write-alias-a" && edge.toNodeId === "consume-alias-b" &&
		edge.physicalId === "physical:generation-shared"));
}

function testDCEAllocationAndGenerationReset() {
	const graph = compiler.compile({
		resources: [
			texture("intermediate", "graph"),
			texture("output", "graph"),
			texture("dead", "graph"),
		],
		nodes: [
			node("producer", [{
				resource: "intermediate",
				access: "write",
				usage: "color-attachment",
			}], { retention: "if-reachable" }),
			node("consumer", [
				{ resource: "intermediate", access: "read", usage: "sampled" },
				{ resource: "output", access: "write", usage: "color-attachment" },
			], { retention: "if-reachable" }),
			node("dead", [{
				resource: "dead",
				access: "write",
				usage: "color-attachment",
			}], { retention: "if-reachable" }),
			node("cpu-observable", [], { retention: "if-reachable", domain: "cpu" }),
		],
		exports: [{ name: "color", resource: "output" }],
	});

	assert.deepEqual(graph.nodes.map((entry) => entry.id), [
		"producer",
		"consumer",
		"cpu-observable",
	]);
	assert.deepEqual(graph.culledNodeIds, ["dead"]);
	assert.equal(graph.allocationRequests.length, 2);
	assert.deepEqual(
		graph.allocationRequests.map((request) => request.resourceId).sort(),
		["intermediate", "output"],
	);
	assert.ok(Object.isFrozen(graph));
	assert.ok(Object.isFrozen(graph.nodes));
	assert.ok(Object.isFrozen(graph.liveRanges));
	assert.equal("lifetimes" in graph, false);

	const generationGraph = compiler.compile({
		resources: [{
			id: "scratch",
			origin: "graph",
			kind: "buffer",
			residency: "transient",
			initialContent: "undefined",
			size: 256,
		}],
		bindings: [{ resourceId: "scratch", physicalId: "slot:scratch", kind: "buffer" }],
		nodes: [
			{ ...node("create-1", []), creates: ["scratch"] },
			node("write-1", [{ resource: "scratch", access: "write", usage: "storage" }]),
			{ ...node("destroy-1", []), destroys: ["scratch"] },
			{ ...node("create-2", []), creates: ["scratch"] },
			node("read-2", [{ resource: "scratch", access: "read", usage: "storage" }]),
		],
	});
	const recreatedRead = generationGraph.transitions.find(
		(transition) => transition.nodeId === "read-2",
	);
	assert.equal(recreatedRead.generation, 2);
	assert.equal(recreatedRead.fromNodeId, undefined);
	const attempts = new RenderGraphAttemptTracker();
	attempts.begin(generationGraph);
	attempts.recordSkippedNode("read-2", [{
		resourceId: "scratch",
		resolvedResourceId: "fallback",
	}]);
	const scratchState = attempts.getDebugState().current.resources.find(
		(resource) => resource.id === "scratch",
	);
	assert.equal(scratchState.active, true);
	assert.equal(scratchState.generation, 2);
	assert.equal(scratchState.lastNodeId, "read-2");
	assert.deepEqual(attempts.getDebugState().current.executionOverlay, {
		skippedNodeIds: ["read-2"],
		resourceAliases: [{ resourceId: "scratch", resolvedResourceId: "fallback" }],
	});
	assert.equal(Object.isFrozen(attempts.getDebugState().current.executionOverlay), true);
	attempts.seal();
	assert.throws(() => attempts.recordSkippedNode("read-2"), /state "sealed"/);
}

function testDCEExportRangeAndValidation() {
	const graph = compiler.compile({
		resources: [texture("mipped", "graph")],
		nodes: [
			node("write-mip-0", [{
				resource: "mipped",
				access: "write",
				usage: "color-attachment",
				subresource: { kind: "texture", mipStart: 0, mipCount: 1 },
			}], { retention: "if-reachable" }),
			node("write-mip-1", [{
				resource: "mipped",
				access: "write",
				usage: "color-attachment",
				subresource: { kind: "texture", mipStart: 1, mipCount: 1 },
			}], { retention: "if-reachable" }),
		],
		exports: [{
			resource: "mipped",
			subresource: { kind: "texture", mipStart: 1, mipCount: 1 },
		}],
	});
	assert.deepEqual(graph.nodes.map((entry) => entry.id), ["write-mip-1"]);
	assert.deepEqual(graph.culledNodeIds, ["write-mip-0"]);

	const invalid = compiler.compile({
		resources: [
			texture("small"),
			texture("different", "imported", { width: 32 }),
		],
		bindings: [
			{ resourceId: "small", physicalId: "physical:conflict", kind: "texture" },
			{ resourceId: "different", physicalId: "physical:conflict", kind: "texture" },
		],
		nodes: [node("bad-range", [{
			resource: "small",
			access: "read",
			usage: "sampled",
			subresource: { kind: "texture", mipStart: 99, mipCount: 1 },
		}])],
	});
	assert.ok(invalid.diagnostics.some((entry) => entry.code === "invalid-subresource-range"));
	assert.ok(invalid.diagnostics.some((entry) => entry.code === "physical-descriptor-conflict"));
	assert.equal(invalid.culledNodeIds.length, 0);
}

function testSubgraphCompositionAndStructuralErrors() {
	const subgraph = {
		resources: [texture("input"), texture("output", "graph")],
		imports: [{ name: "source", resource: "input" }],
		outputPorts: [{ name: "result", resource: "output" }],
		exports: [{ name: "result", resource: "output" }],
		nodes: [node("copy", [
			{ resource: "input", access: "read", usage: "sampled" },
			{ resource: "output", access: "write", usage: "color-attachment" },
		], { retention: "if-reachable" })],
	};
	const builder = new RenderGraphBuilder();
	builder.addResource(texture("parent-source"));
	builder.addResource(texture("parent-result", "imported"));
	builder.addNode(node("anchor", []));
	const composed = builder.addSubgraph(subgraph, {
		namespace: "post",
		inputs: { source: "parent-source" },
		outputs: { result: "parent-result" },
		dependsOn: ["anchor"],
	});
	assert.equal(composed.outputs.result, "parent-result");
	assert.equal(composed.resources.input, "parent-source");
	assert.equal(composed.resources.output, "parent-result");
	assert.equal(composed.nodes.copy, "post:copy");
	const compiled = compiler.compile(builder.build());
	assert.deepEqual(compiled.nodes.map((entry) => entry.id), ["anchor", "post:copy"]);
	assert.ok(compiled.dependencies.some((entry) =>
		entry.fromNodeId === "anchor" && entry.toNodeId === "post:copy"));
	assert.deepEqual(compiled.portResolutions.map((entry) => entry.port), ["source", "result"]);
	assert.equal(compiled.exports[0].name, "post:result");

	const optional = new RenderGraphBuilder();
	optional.addSubgraph({
		resources: [texture("optional-input", "imported", { optional: true })],
		imports: [{ name: "optional", resource: "optional-input", optional: true }],
		nodes: [node("optional-reader", [{
			resource: "optional-input",
			access: "read",
			usage: "sampled",
			optional: true,
		}])],
	}, { namespace: "optional-child" });
	const optionalGraph = compiler.compile(optional.build());
	assert.ok(!optionalGraph.diagnostics.some((entry) => entry.code === "missing-resource"));
	assert.ok(!optionalGraph.resources.some((entry) => entry.id === "optional-child:optional-input"));

	const incompatible = new RenderGraphBuilder();
	incompatible.addResource(texture("parent-source"));
	incompatible.addResource({
		id: "bad-output",
		origin: "imported",
		kind: "buffer",
		residency: "frame",
		size: 64,
	});
	incompatible.addSubgraph(subgraph, {
		namespace: "bad",
		inputs: { source: "parent-source" },
		outputs: { result: "bad-output" },
	});
	const invalid = compiler.compile(incompatible.build());
	assert.ok(invalid.diagnostics.some((entry) => entry.code === "incompatible-subgraph-port"));

	const structural = compiler.compile({
		resources: [],
		nodes: [
			node("a", [], { dependsOn: ["b"] }),
			node("b", [], { dependsOn: ["a", "missing"] }),
		],
	});
	assert.ok(structural.diagnostics.some((entry) => entry.code === "cycle"));
	assert.ok(structural.diagnostics.some((entry) => entry.code === "missing-dependency"));
}

testSubresourceAndPhysicalHazards();
testDCEAllocationAndGenerationReset();
testDCEExportRangeAndValidation();
testSubgraphCompositionAndStructuralErrors();
console.log("Render Graph V2 tests passed");
