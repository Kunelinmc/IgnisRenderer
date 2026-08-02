import assert from "node:assert/strict";
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { dirname, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const TEST_DIR = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = resolve(TEST_DIR, "../../..");
const DOCS_ROOT = resolve(PROJECT_ROOT, "docs");
const DOCS_INDEX = resolve(DOCS_ROOT, "README.md");
const ALLOWED_CATEGORIES = new Set([
	"architecture",
	"contracts",
	"contributing",
	"migrations",
	"public",
	"reference",
]);

function normalizePath(path) {
	return path.split(sep).join("/");
}

function stripFencedCode(content) {
	return content.replace(/```[\s\S]*?```/g, "");
}

function collectMarkdownFiles(root) {
	const files = [];
	const queue = [root];
	while (queue.length > 0) {
		const current = queue.pop();
		for (const entry of readdirSync(current, { withFileTypes: true })) {
			if (entry.name.startsWith(".")) {
				continue;
			}
			const path = resolve(current, entry.name);
			if (entry.isDirectory()) {
				queue.push(path);
			} else if (entry.name.endsWith(".md")) {
				files.push(path);
			}
		}
	}
	return files.sort();
}

function localMarkdownTargets(file, content) {
	const targets = [];
	const pattern = /\[[^\]]+\]\(([^)]+)\)/g;
	for (const match of stripFencedCode(content).matchAll(pattern)) {
		const rawTarget = match[1].trim().replace(/^<|>$/g, "");
		if (
			rawTarget.startsWith("#")
			|| /^[a-z][a-z0-9+.-]*:/i.test(rawTarget)
		) {
			continue;
		}
		const fileTarget = rawTarget.split("#", 1)[0];
		if (!fileTarget.endsWith(".md")) {
			continue;
		}
		targets.push(resolve(dirname(file), decodeURIComponent(fileTarget)));
	}
	return targets;
}

function testStructure() {
	const files = collectMarkdownFiles(DOCS_ROOT);
	assert.ok(files.length > 0, "Expected documentation files.");

	for (const file of files) {
		const path = normalizePath(relative(DOCS_ROOT, file));
		const parts = path.split("/");
		if (parts.length === 1) {
			assert.equal(path, "README.md", `Unexpected root document: ${path}`);
		} else {
			assert.equal(parts.length, 2, `Documentation path is too deep: ${path}`);
			assert.ok(ALLOWED_CATEGORIES.has(parts[0]), `Unknown category: ${path}`);
		}
		assert.doesNotMatch(
			parts.at(-1),
			/(?:^|[-_])(?:v\d+|phase[-_]?\d+)(?:[-_.]|$)/i,
			`Editorial version marker in filename: ${path}`
		);
		assert.doesNotMatch(
			parts.at(-1),
			/-(?:contract|architecture|migration)\.md$/i,
			`Redundant document-type suffix: ${path}`
		);
	}
}

function testLinksAndTitles() {
	const files = collectMarkdownFiles(DOCS_ROOT);
	for (const file of files) {
		const content = readFileSync(file, "utf8");
		const path = normalizePath(relative(DOCS_ROOT, file));
		const titles = stripFencedCode(content).match(/^# .+$/gm) ?? [];
		assert.equal(titles.length, 1, `Expected one document title: ${path}`);
		assert.doesNotMatch(
			titles[0],
			/\b(?:v\d+|phase\s+\d+)\b/i,
			`Editorial version marker in title: ${path}`
		);

		const targets = localMarkdownTargets(file, content);
		assert.ok(targets.length > 0, `Expected related documentation links: ${path}`);
		for (const target of targets) {
			assert.ok(target.startsWith(DOCS_ROOT), `Link escapes docs tree: ${path}`);
			assert.ok(existsSync(target), `Broken documentation link from ${path}: ${target}`);
			assert.ok(statSync(target).isFile(), `Documentation link is not a file: ${target}`);
		}
	}
}

function testIndexCoverage() {
	const files = collectMarkdownFiles(DOCS_ROOT).filter((file) => file !== DOCS_INDEX);
	const indexContent = readFileSync(DOCS_INDEX, "utf8");
	const indexed = new Set(localMarkdownTargets(DOCS_INDEX, indexContent));
	for (const file of files) {
		const path = normalizePath(relative(DOCS_ROOT, file));
		assert.ok(indexed.has(file), `Documentation index does not link to ${path}`);
	}
}

function testLegacyPaths() {
	const legacy = [
		"docs/internal/",
		"docs/public/apis/",
		"docs/public/guides/",
		"docs/contracts/rendering/backends/",
	];
	const files = [
		resolve(PROJECT_ROOT, "AGENTS.md"),
		resolve(PROJECT_ROOT, "README.md"),
		...collectMarkdownFiles(DOCS_ROOT),
	];
	for (const file of files) {
		const content = readFileSync(file, "utf8");
		for (const value of legacy) {
			assert.ok(!content.includes(value), `Legacy docs path ${value} remains in ${file}`);
		}
	}
}

testStructure();
testLinksAndTitles();
testIndexCoverage();
testLegacyPaths();

console.log("Documentation structure tests passed");
