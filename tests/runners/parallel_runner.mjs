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
	let timeoutMsFromCli = null;
	let timeoutActionFromCli = null;
	let noTimeoutSkip = false;
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
		if (arg === "--no-timeout-skip") {
			noTimeoutSkip = true;
			continue;
		}
		if (arg === "--timeout-ms") {
			const value = argv[i + 1];
			if (value) {
				timeoutMsFromCli = Number.parseInt(value, 10);
				i++;
			}
			continue;
		}
		if (arg.startsWith("--timeout-ms=")) {
			timeoutMsFromCli = Number.parseInt(arg.slice("--timeout-ms=".length), 10);
			continue;
		}
		if (arg === "--timeout-action") {
			const value = argv[i + 1];
			if (value) {
				timeoutActionFromCli = value;
				i++;
			}
			continue;
		}
		if (arg.startsWith("--timeout-action=")) {
			timeoutActionFromCli = arg.slice("--timeout-action=".length);
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
	const envTimeoutMs = Number.parseInt(process.env.TEST_TIMEOUT_MS ?? "", 10);
	const parsedTimeoutMs = Number.isInteger(timeoutMsFromCli)
		? timeoutMsFromCli
		: envTimeoutMs;
	const timeoutMs = Number.isInteger(parsedTimeoutMs) && parsedTimeoutMs > 0
		? parsedTimeoutMs
		: 0;
	const timeoutAction = normalizeTimeoutAction(
		timeoutActionFromCli ?? process.env.TEST_TIMEOUT_ACTION
	);
	const shouldDisableTimeoutSkip = noTimeoutSkip
		|| process.env.TEST_NO_TIMEOUT_SKIP === "1"
		|| process.env.TEST_NO_TIMEOUT_SKIP === "true";

	return {
		tests,
		jobs: Math.max(1, Math.min(Math.max(1, tests.length), jobs)),
		failFast,
		listOnly,
		timeoutMs: shouldDisableTimeoutSkip && timeoutAction === "skip" ? 0 : timeoutMs,
		timeoutAction,
		noTimeoutSkip: shouldDisableTimeoutSkip,
	};
}

function normalizeTimeoutAction(value) {
	if (value === "skip" || value === "fail") {
		return value;
	}
	return "fail";
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

function runSingleTest(testPath, index, total, options = {}) {
	return new Promise((resolveTest) => {
		const startTime = Date.now();
		const testName = normalizePath(relative(PROJECT_ROOT, testPath));
		console.log(`[start ${index + 1}/${total}] ${testName}`);
		let completed = false;
		let timedOut = false;
		let forceKillId = null;
		const timeoutMs = options.timeoutMs ?? 0;
		const timeoutAction = options.timeoutAction ?? "fail";

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

		const timeoutId = timeoutMs > 0
			? setTimeout(() => {
				timedOut = true;
				child.kill();
				forceKillId = setTimeout(() => {
					if (!completed) {
						child.kill("SIGKILL");
					}
				}, 1000);
			}, timeoutMs)
			: null;

		const resolveOnce = (result) => {
			if (completed) {
				return;
			}
			completed = true;
			if (timeoutId) {
				clearTimeout(timeoutId);
			}
			if (forceKillId) {
				clearTimeout(forceKillId);
			}
			resolveTest(result);
		};

		child.on("error", (error) => {
			resolveOnce({
				testName,
				ok: false,
				skipped: false,
				code: 1,
				output: `${stdout}${stderr}\n${String(error)}`,
				durationMs: Date.now() - startTime,
			});
		});

		child.on("close", (code, signal) => {
			const finalCode = typeof code === "number" ? code : 1;
			const signalText = signal ? `\nterminated by signal: ${signal}` : "";
			const timeoutText = timedOut
				? `\ntimed out after ${timeoutMs}ms`
				: "";
			const skipped = timedOut && timeoutAction === "skip";
			resolveOnce({
				testName,
				ok: skipped || finalCode === 0,
				skipped,
				code: finalCode,
				output: `${stdout}${stderr}${signalText}${timeoutText}`,
				durationMs: Date.now() - startTime,
			});
		});
	});
}

export async function runDiscoveredTests(options) {
	const { testsRoot, label, argv, allowEmpty = false } = options;
	const {
		tests,
		jobs,
		failFast,
		listOnly,
		timeoutMs,
		timeoutAction,
		noTimeoutSkip,
	} = parseOptions(argv, testsRoot);

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
		`Running ${total} ${label} with concurrency=${jobs} `
			+ `(failFast=${failFast}, timeoutMs=${timeoutMs}, `
			+ `timeoutAction=${timeoutAction}, noTimeoutSkip=${noTimeoutSkip})\n`
	);

	const queue = tests.map((testPath, index) => ({ testPath, index }));
	const failures = [];
	const skipped = [];
	let stopped = false;

	const workers = Array.from({ length: jobs }, async () => {
		while (queue.length > 0 && !stopped) {
			const next = queue.shift();
			if (!next) {
				return;
			}

			const result = await runSingleTest(next.testPath, next.index, total, {
				timeoutMs,
				timeoutAction,
			});
			const status = result.skipped ? "SKIP" : result.ok ? "PASS" : "FAIL";
			const title = `${status} ${result.testName}`;
			const duration = formatDuration(result.durationMs);

			console.log("----------------------------------------");
			console.log(`${title} (${duration})`);
			if (result.output.trim().length > 0) {
				console.log(result.output.trimEnd());
			}

			if (result.skipped) {
				skipped.push(result);
				continue;
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
			`Some ${label} failed (${failures.length}/${total}, `
				+ `skipped ${skipped.length}) in ${totalDuration}.`
		);
		process.exit(1);
		return;
	}
	if (skipped.length > 0) {
		console.log(
			`${label} passed with skips (${skipped.length}/${total}) in ${totalDuration}.`
		);
		return;
	}
	console.log(`All ${label} passed in ${totalDuration}.`);
}
