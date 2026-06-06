import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = path.resolve(__dirname, "../../..");
const CURRENT_TEST_FILE = path.resolve(
	PROJECT_ROOT,
	"tests/static/shaders/test_shader_directive_migration_guard.mjs"
);

const BANNED_PATTERNS = [
	/\bpreprocessEngineShaderDirectives\b/,
	/\bENGINE_DIRECTIVE_RUNTIME\b/,
];

function collectFiles(rootDirectory) {
	const results = [];
	const queue = [rootDirectory];
	while (queue.length > 0) {
		const current = queue.pop();
		const entries = fs.readdirSync(current, { withFileTypes: true });
		for (const entry of entries) {
			const absolutePath = path.join(current, entry.name);
			if (entry.isDirectory()) {
				queue.push(absolutePath);
				continue;
			}
			if (!/\.(ts|mjs|js)$/i.test(entry.name)) {
				continue;
			}
			results.push(absolutePath);
		}
	}
	return results;
}

function testNoLegacyDirectivePreprocessReferences() {
	const scannedRoots = [
		path.resolve(PROJECT_ROOT, "src"),
		path.resolve(PROJECT_ROOT, "tests"),
	];
	const findings = [];
	for (const root of scannedRoots) {
		const files = collectFiles(root);
		for (const filePath of files) {
			if (filePath === CURRENT_TEST_FILE) {
				continue;
			}
			const content = fs.readFileSync(filePath, "utf8");
			for (const pattern of BANNED_PATTERNS) {
				if (pattern.test(content)) {
					findings.push(
						`${path.relative(PROJECT_ROOT, filePath)} matched ${pattern}`
					);
				}
			}
		}
	}
	assert.deepEqual(
		findings,
		[],
		"Legacy directive preprocess APIs are still referenced."
	);
}

function run() {
	testNoLegacyDirectivePreprocessReferences();
	console.log("Shader directive migration guard passed");
}

run();
