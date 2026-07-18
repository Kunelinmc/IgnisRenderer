import assert from "node:assert/strict";
import { RenderGraphCompiler } from "../../../src/rendergraph/RenderGraphCompiler.ts";

const compiler = new RenderGraphCompiler();

{
	const compiled = compiler.compile({
		resources: [
			{ id: "frame:color", origin: "imported" },
			{ id: "post:color", origin: "graph" },
		],
		nodes: [
			{
				id: "post:b",
				stage: "postprocess",
				kind: "postprocess-pass",
				label: "B",
				dependsOn: ["post:a"],
				resources: [
					{ resource: "post:color", access: "read", usage: "sampled" },
				],
			},
			{
				id: "post:a",
				stage: "postprocess",
				kind: "postprocess-pass",
				label: "A",
				creates: ["post:color"],
				resources: [
					{ resource: "frame:color", access: "read", usage: "sampled" },
					{ resource: "post:color", access: "write", usage: "storage" },
				],
			},
		],
	});
	assert.deepEqual(compiled.nodes.map((node) => node.id), ["post:a", "post:b"]);
	assert.equal(compiled.diagnostics.length, 0);
}

{
	const compiled = compiler.compile({
		resources: [],
		nodes: [{
			id: "broken",
			stage: "postprocess",
			kind: "postprocess-pass",
			label: "Broken",
			resources: [{ resource: "missing", access: "read", usage: "sampled" }],
		}],
	});
	assert.equal(compiled.diagnostics[0]?.code, "read-before-create");
}

{
	const compiled = compiler.compile({
		resources: [{ id: "imported", origin: "imported" }],
		nodes: [
			{ id: "late", stage: "postprocess", kind: "test", label: "Late", dependsOn: ["first"] },
			{ id: "independent", stage: "postprocess", kind: "test", label: "Independent" },
			{ id: "first", stage: "postprocess", kind: "test", label: "First" },
		],
	});
	assert.deepEqual(compiled.nodes.map((node) => node.id), ["independent", "first", "late"]);
	assert.equal(Object.isFrozen(compiled), true);
	assert.equal(Object.isFrozen(compiled.nodes), true);
	assert.equal(Object.isFrozen(compiled.nodes[0]), true);
}

{
	const compiled = compiler.compile({
		resources: [{ id: "created", origin: "graph" }],
		nodes: [
			{ id: "a", stage: "postprocess", kind: "test", label: "A", dependsOn: ["missing"] },
			{ id: "b", stage: "postprocess", kind: "test", label: "B", dependsOn: ["c"] },
			{ id: "c", stage: "postprocess", kind: "test", label: "C", dependsOn: ["b"] },
			{ id: "d", stage: "postprocess", kind: "test", label: "D", destroys: ["created"] },
		],
	});
	assert.ok(compiled.diagnostics.some((diagnostic) => diagnostic.code === "missing-dependency"));
	assert.ok(compiled.diagnostics.some((diagnostic) => diagnostic.code === "cycle"));
	assert.ok(compiled.diagnostics.some((diagnostic) => diagnostic.code === "destroy-before-create"));
}

{
	const compiled = compiler.compile({
		resources: [{ id: "frame:color", origin: "imported" }],
		nodes: [
			{
				id: "write",
				stage: "postprocess",
				kind: "test",
				label: "Write",
				resources: [{ resource: "frame:color", access: "write", usage: "storage" }],
			},
			{
				id: "read",
				stage: "postprocess",
				kind: "test",
				label: "Read",
				resources: [{ resource: "frame:color", access: "read", usage: "sampled" }],
			},
		],
	});
	assert.equal(compiled.transitions[1]?.hazard, "read-after-write");
}

console.log("Post-process render graph tests passed");
