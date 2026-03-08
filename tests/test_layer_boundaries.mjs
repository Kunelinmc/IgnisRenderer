import assert from "node:assert/strict";
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { dirname, join, relative, sep } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const rootDir = join(__dirname, "..");

function walkFiles(dirPath) {
	const entries = readdirSync(dirPath);
	const files = [];

	for (const entry of entries) {
		const fullPath = join(dirPath, entry);
		const stat = statSync(fullPath);
		if (stat.isDirectory()) {
			files.push(...walkFiles(fullPath));
			continue;
		}
		files.push(fullPath);
	}

	return files;
}

function normalizePath(pathValue) {
	return pathValue.split(sep).join("/");
}

function collectImportSpecifiers(sourceCode) {
	const specifiers = [];
	const importExportRegex =
		/(?:import|export)\s+[\s\S]*?\sfrom\s+['"]([^'"]+)['"]/g;
	const dynamicImportRegex = /import\s*\(\s*['"]([^'"]+)['"]\s*\)/g;

	for (const regex of [importExportRegex, dynamicImportRegex]) {
		let match = regex.exec(sourceCode);
		while (match) {
			specifiers.push(match[1]);
			match = regex.exec(sourceCode);
		}
	}

	return specifiers;
}

function run() {
	const scanFiles = [
		...walkFiles(join(rootDir, "src")),
		...walkFiles(join(rootDir, "tests")),
	]
		.filter((filePath) => /\.(ts|mjs)$/i.test(filePath))
		.map((filePath) => normalizePath(filePath));

	const violations = [];

	for (const fullPath of scanFiles) {
		const relPath = normalizePath(relative(rootDir, fullPath));
		const sourceCode = readFileSync(fullPath, "utf8");
		const specifiers = collectImportSpecifiers(sourceCode);

		for (const specifier of specifiers) {
			const disallowedLegacyPath =
				specifier.includes("core/bridge") ||
				specifier.includes("core/resources") ||
				specifier.includes("core/ral") ||
				specifier.includes("core/backend") ||
				specifier.includes("core/pipeline");
			if (disallowedLegacyPath) {
				violations.push(`${relPath} imports legacy path "${specifier}"`);
			}

			const inDefinitionLayer =
				relPath.startsWith("src/lights/") ||
				relPath.startsWith("src/materials/") ||
				relPath.startsWith("src/particles/") ||
				relPath.startsWith("src/animation/");
			if (inDefinitionLayer) {
				const touchesPipelineImpl =
					specifier.includes("renderers/") || specifier.includes("simulation/");
				if (touchesPipelineImpl) {
					violations.push(
						`${relPath} imports runtime pipeline logic "${specifier}"`
					);
				}
			}

			const inWebGPUBackendLayer = relPath.startsWith("src/renderers/webgpu/");
			const touchesSoftwareLighting =
				specifier.includes("renderers/software/LightEvaluator") ||
				specifier.includes("software/LightEvaluator");
			if (inWebGPUBackendLayer && touchesSoftwareLighting) {
				violations.push(
					`${relPath} imports software lighting implementation "${specifier}"`
				);
			}
		}
	}

	for (const oldDir of [
		"src/core/bridge",
		"src/core/resources",
		"src/core/ral",
		"src/core/geometry",
		"src/core/software",
		"src/core/backend",
		"src/core/pipeline",
	]) {
		if (existsSync(join(rootDir, oldDir))) {
			violations.push(`legacy directory still exists: ${oldDir}`);
		}
	}

	if (violations.length > 0) {
		assert.fail(`Layer boundary violations:\n${violations.join("\n")}`);
	}

	console.log("Layer boundary tests passed");
}

run();
