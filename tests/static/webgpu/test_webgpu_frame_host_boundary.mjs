import assert from "node:assert/strict";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";

const root = join(import.meta.dir, "../../../src/backends/webgpu/rendergraph");
const violations = [];

function visit(directory) {
	for (const entry of readdirSync(directory)) {
		const path = join(directory, entry);
		if (statSync(path).isDirectory()) {
			visit(path);
			continue;
		}
		if (!path.endsWith(".ts")) continue;
		const source = readFileSync(path, "utf8");
		if (/from\s+["'][^"']*WebGPUBackend["']/.test(source)) {
			violations.push(path);
		}
	}
}

visit(root);
assert.deepEqual(violations, []);
console.log("WebGPU frame host boundary tests passed");
