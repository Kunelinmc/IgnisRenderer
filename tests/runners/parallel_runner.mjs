import { spawn } from "node:child_process";
import { existsSync, readdirSync, statSync } from "node:fs";
import { availableParallelism } from "node:os";
import { dirname, isAbsolute, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export const TESTS_DIR = resolve(dirname(fileURLToPath(import.meta.url)), "..");
export const PROJECT_ROOT = resolve(TESTS_DIR, "..");

const BUN_EXECUTABLE = process.versions?.bun ? process.execPath : "bun";

function normalizePath(pathValue) {
	return pathValue.replace(/\\/g, "/");
}

function collectTestFiles(rootDirectory) {
	if (!existsSync(rootDirectory)) {
		return [];
	}

	const results = [];
	const queue = [rootDirectory];
	while (queue.length > 0) {
		const current = queue.pop();
		const entries = readdirSync(current, { withFileTypes: true });
		for (const entry of entries) {
			const absolutePath = join(current, entry.name);
			if (entry.isDirectory()) {
				queue.push(absolutePath);
				continue;
			}
			if (entry.name.startsWith("test_") && entry.name.endsWith(".mjs")) {
				results.push(absolutePath);
			}
		}
	}

	return results.sort((left, right) => {
		const leftPath = normalizePath(relative(rootDirectory, left));
		const rightPath = normalizePath(relative(rootDirectory, right));
		return leftPath.localeCompare(rightPath);
	});
}

function resolveRequestedTestPath(value, testsRoot) {
	const candidates = [
		isAbsolute(value) ? value : resolve(PROJECT_ROOT, value),
		isAbsolute(value) ? value : resolve(testsRoot, value),
	];

	for (const candidate of candidates) {
		if (existsSync(candidate)) {
			return candidate;
		}
	}

	throw new Error(`Requested test path does not exist: ${value}`);
}

function parseOptions(argv, testsRoot) {
	let jobsFromCli = null;
	let failFast = false;
	let listOnly = false;
	const requestedPaths = [];

	for (let i = 0; i < argv.length; i++) {
		const arg = argv[i];
		if (arg === "--fail-fast") {
			failFast = true;
			continue;
		}
		if (arg === "--serial") {
			jobsFromCli = 1;
			continue;
		}
		if (arg === "--list") {
			listOnly = true;
			continue;
		}
		if (arg === "-j" || arg === "--jobs") {
			const value = argv[i + 1];
			if (value) {
				jobsFromCli = Number.parseInt(value, 10);
				i++;
			}
			continue;
		}
		if (arg.startsWith("--jobs=")) {
			jobsFromCli = Number.parseInt(arg.slice("--jobs=".length), 10);
			continue;
		}
		requestedPaths.push(arg);
	}

	const tests = requestedPaths.length > 0
		? requestedPaths.flatMap((requestedPath) => {
			const absolutePath = resolveRequestedTestPath(requestedPath, testsRoot);
			const stat = statSync(absolutePath);
			if (stat.isDirectory()) {
				return collectTestFiles(absolutePath);
			}
			return [absolutePath];
		})
		: collectTestFiles(testsRoot);

	const envJobs = Number.parseInt(process.env.TEST_JOBS ?? "", 10);
	const parsedJobs = Number.isInteger(jobsFromCli) ? jobsFromCli : envJobs;
	const jobs = Number.isInteger(parsedJobs) && parsedJobs > 0
		? parsedJobs
		: getDefaultJobs(tests.length);

	return {
		tests,
		jobs: Math.max(1, Math.min(Math.max(1, tests.length), jobs)),
		failFast,
		listOnly,
	};
}

function getDefaultJobs(testCount) {
	const detected = (() => {
		try {
			return availableParallelism();
		} catch {
			return 4;
		}
	})();
	return Math.max(1, Math.min(Math.max(1, testCount), detected));
}

function formatDuration(durationMs) {
	if (durationMs < 1000) {
		return `${durationMs}ms`;
	}
	return `${(durationMs / 1000).toFixed(2)}s`;
}

function runSingleTest(testPath, index, total) {
	return new Promise((resolveTest) => {
		const startTime = Date.now();
		const testName = normalizePath(relative(PROJECT_ROOT, testPath));
		console.log(`[start ${index + 1}/${total}] ${testName}`);

		const child = spawn(BUN_EXECUTABLE, [testPath], {
			cwd: PROJECT_ROOT,
			stdio: ["ignore", "pipe", "pipe"],
			env: process.env,
		});

		let stdout = "";
		let stderr = "";

		child.stdout.on("data", (chunk) => {
			stdout += String(chunk);
		});
		child.stderr.on("data", (chunk) => {
			stderr += String(chunk);
		});

		child.on("error", (error) => {
			resolveTest({
				testName,
				ok: false,
				code: 1,
				output: `${stdout}${stderr}\n${String(error)}`,
				durationMs: Date.now() - startTime,
			});
		});

		child.on("close", (code, signal) => {
			const finalCode = typeof code === "number" ? code : 1;
			const signalText = signal ? `\nterminated by signal: ${signal}` : "";
			resolveTest({
				testName,
				ok: finalCode === 0,
				code: finalCode,
				output: `${stdout}${stderr}${signalText}`,
				durationMs: Date.now() - startTime,
			});
		});
	});
}

export async function runDiscoveredTests(options) {
	const { testsRoot, label, argv, allowEmpty = false } = options;
	const { tests, jobs, failFast, listOnly } = parseOptions(argv, testsRoot);

	if (listOnly) {
		for (const testPath of tests) {
			console.log(normalizePath(relative(PROJECT_ROOT, testPath)));
		}
		return;
	}

	if (tests.length === 0) {
		if (allowEmpty) {
			console.log(`No ${label} found in ${normalizePath(relative(PROJECT_ROOT, testsRoot))}.`);
			return;
		}
		console.error(`No ${label} found in ${normalizePath(relative(PROJECT_ROOT, testsRoot))}.`);
		process.exit(1);
	}

	const total = tests.length;
	const startedAt = Date.now();

	console.log(
		`Running ${total} ${label} with concurrency=${jobs} (failFast=${failFast})\n`
	);

	const queue = tests.map((testPath, index) => ({ testPath, index }));
	const failures = [];
	let stopped = false;

	const workers = Array.from({ length: jobs }, async () => {
		while (queue.length > 0 && !stopped) {
			const next = queue.shift();
			if (!next) {
				return;
			}

			const result = await runSingleTest(next.testPath, next.index, total);
			const title = `${result.ok ? "PASS" : "FAIL"} ${result.testName}`;
			const duration = formatDuration(result.durationMs);

			console.log("----------------------------------------");
			console.log(`${title} (${duration})`);
			if (result.output.trim().length > 0) {
				console.log(result.output.trimEnd());
			}

			if (!result.ok) {
				failures.push(result);
				if (failFast) {
					stopped = true;
				}
			}
		}
	});

	await Promise.all(workers);

	const totalDuration = formatDuration(Date.now() - startedAt);
	console.log("\n----------------------------------------");
	if (failures.length > 0) {
		console.log(
			`Some ${label} failed (${failures.length}/${total}) in ${totalDuration}.`
		);
		process.exit(1);
		return;
	}
	console.log(`All ${label} passed in ${totalDuration}.`);
}
