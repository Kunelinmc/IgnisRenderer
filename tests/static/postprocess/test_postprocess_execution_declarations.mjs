import assert from "node:assert/strict";

import * as declarations from "../../../src/postprocess/executionDeclarations.ts";

function testExecutionPoliciesAreExplicitData() {
	assert.equal("createPostProcessExecutionDeclaration" in declarations, false);
	assert.deepEqual(declarations.SOFTWARE_IN_PLACE_EXECUTION, {
		color: { access: "read-write", output: "preserve" },
	});
	assert.deepEqual(declarations.WEBGPU_VERSIONED_EXECUTION, {
		color: { access: "read", output: "new-version" },
	});
	assert.deepEqual(declarations.WEBGL_VERSIONED_EXECUTION, {
		color: { access: "read", output: "new-version" },
	});
	assert.deepEqual(declarations.POST_PROCESS_CPU_READ, {
		access: "read",
		usage: "cpu-read",
	});
	assert.deepEqual(declarations.POST_PROCESS_STORAGE_WRITE, {
		access: "write",
		usage: "storage",
	});
}

testExecutionPoliciesAreExplicitData();
console.log("Post-process execution declaration tests passed");
