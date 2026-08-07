import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

import {
	WEBGPU_DEFERRED_BASE_COLOR_BYTES_PER_SAMPLE,
	WEBGPU_DEFERRED_COLOR_BYTES_PER_SAMPLE,
} from "../../src/backends/webgpu/constants.ts";

const WARMUP_FRAMES = 20;
const MEASURED_FRAMES = 100;
const REQUIRED_FRAME_COUNT = WARMUP_FRAMES + MEASURED_FRAMES;
const BASE_4K_MINIMUM_IMPROVEMENT = 0.15;
const EXTENDED_MAXIMUM_REGRESSION = 0.05;
const STORAGE_BYTES_PER_PIXEL = 16;

function percentile(sorted, fraction) {
	const index = Math.min(
		sorted.length - 1,
		Math.max(0, Math.ceil(sorted.length * fraction) - 1)
	);
	return sorted[index];
}

function summarizeSamples(samples) {
	assert.ok(
		Array.isArray(samples) && samples.length >= REQUIRED_FRAME_COUNT,
		`each run requires at least ${REQUIRED_FRAME_COUNT} GPU timestamp samples`
	);
	const measured = samples
		.slice(WARMUP_FRAMES)
		.map(Number)
		.filter(Number.isFinite)
		.sort((left, right) => left - right);
	assert.ok(
		measured.length >= MEASURED_FRAMES,
		`each run requires ${MEASURED_FRAMES} finite post-warmup samples`
	);
	return {
		medianMs: percentile(measured, 0.5),
		p95Ms: percentile(measured, 0.95),
		sampleCount: measured.length,
	};
}

function runKey(run) {
	return [run.workload, run.width, run.height, run.layout, run.revision].join(":");
}

function comparableSettings(run) {
	return JSON.stringify({
		adapter: run.adapter,
		width: run.width,
		height: run.height,
		lightCount: run.lightCount,
		postProcess: run.postProcess,
		workload: run.workload,
		layout: run.layout,
	});
}

function findPair(runs, workload, width, height, layout) {
	const matches = runs.filter(
		(run) =>
			run.workload === workload &&
			run.width === width &&
			run.height === height &&
			run.layout === layout
	);
	const before = matches.find((run) => run.revision === "before");
	const after = matches.find((run) => run.revision === "after");
	assert.ok(before && after, `missing before/after pair for ${workload} ${width}x${height}`);
	assert.equal(
		comparableSettings(before),
		comparableSettings(after),
		`benchmark settings changed for ${workload} ${width}x${height}`
	);
	return { before, after };
}

function evaluatePair(pair) {
	const before = summarizeSamples(pair.before.samplesMs);
	const after = summarizeSamples(pair.after.samplesMs);
	return {
		before,
		after,
		medianChange:
			(before.medianMs - after.medianMs) / Math.max(before.medianMs, 1e-9),
	};
}

function printUsage() {
	console.log(
		"Usage: bun run bench:webgpu-deferred -- --input <capture.json>\n" +
		"Capture GPU timestamp-query durations for G-buffer + deferred-lighting. " +
		"Each run needs 120+ frames; the first 20 are discarded.\n" +
		"Required pairs: base-memory-bound and full-extended at 1920x1080 and " +
		"3840x2160, each with revision=before/after."
	);
}

const inputFlag = process.argv.findIndex((argument) => argument === "--input");
const inputPath = inputFlag >= 0 ? process.argv[inputFlag + 1] : null;
if (!inputPath) {
	printUsage();
	process.exit(0);
}

assert.equal(WEBGPU_DEFERRED_BASE_COLOR_BYTES_PER_SAMPLE, 24);
assert.equal(
	WEBGPU_DEFERRED_COLOR_BYTES_PER_SAMPLE + STORAGE_BYTES_PER_PIXEL,
	60
);

const capture = JSON.parse(await readFile(inputPath, "utf8"));
assert.ok(Array.isArray(capture.runs), "capture.runs must be an array");
const duplicateKeys = capture.runs.map(runKey);
assert.equal(new Set(duplicateKeys).size, duplicateKeys.length, "duplicate benchmark run");

const reports = [];
for (const [workload, layout] of [
	["base-memory-bound", "base"],
	["full-extended", "extended"],
]) {
	for (const [width, height] of [[1920, 1080], [3840, 2160]]) {
		const result = evaluatePair(findPair(capture.runs, workload, width, height, layout));
		reports.push({ workload, layout, width, height, ...result });
	}
}

const base4K = reports.find(
	(report) => report.workload === "base-memory-bound" && report.width === 3840
);
const extendedReports = reports.filter((report) => report.layout === "extended");
assert.ok(
	base4K.medianChange >= BASE_4K_MINIMUM_IMPROVEMENT,
	`4K base median improved ${(base4K.medianChange * 100).toFixed(2)}%; expected at least 15%`
);
for (const report of extendedReports) {
	assert.ok(
		report.medianChange >= -EXTENDED_MAXIMUM_REGRESSION,
		`${report.width}x${report.height} extended median regressed ` +
			`${(-report.medianChange * 100).toFixed(2)}%; maximum is 5%`
	);
}

console.table(
	reports.map((report) => ({
		workload: report.workload,
		resolution: `${report.width}x${report.height}`,
		beforeMedianMs: report.before.medianMs,
		afterMedianMs: report.after.medianMs,
		afterP95Ms: report.after.p95Ms,
		improvementPercent: Number((report.medianChange * 100).toFixed(2)),
	}))
);
console.log("WebGPU deferred benchmark acceptance passed");
