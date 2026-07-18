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

console.log("Post-process render graph tests passed");
