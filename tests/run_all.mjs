import { spawnSync } from "node:child_process";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { dirname } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));

const tests = [
	"test_lighting.mjs",
	"test_point_spot_lighting.mjs",
	"test_sh_lighting_regression.mjs",
	"test_lighting_shader_semantics.mjs",
	"test_model_factory_winding.mjs",
	"test_sparse_accessor.mjs",
	"test_pbr_textures.mjs",
];

let failed = false;

console.log("🚀 Running all tests...\n");

for (const test of tests) {
	console.log(`----------------------------------------`);
	console.log(`Running ${test}...`);
	const result = spawnSync("npx", ["tsx", join(__dirname, test)], {
		stdio: "inherit",
		shell: true,
	});

	if (result.status !== 0) {
		console.error(`❌ ${test} FAILED`);
		failed = true;
	} else {
		console.log(`✅ ${test} PASSED`);
	}
}

console.log(`\n----------------------------------------`);
if (failed) {
	console.log("❌ Some tests failed!");
	process.exit(1);
} else {
	console.log("✨ All tests passed!");
}
