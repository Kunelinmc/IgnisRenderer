import { join } from "node:path";

import { TESTS_DIR, runDiscoveredTests } from "./runners/parallel_runner.mjs";

await runDiscoveredTests({
	testsRoot: join(TESTS_DIR, "browser"),
	label: "browser tests",
	argv: process.argv.slice(2),
	allowEmpty: true,
});
