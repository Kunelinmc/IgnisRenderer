import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import { compileWGSLMath } from "../../helpers/wgsl-math.mjs";
import {
	createSSRTraceParams,
	resolveSSROptions,
} from "../../../src/postprocess/passes/ScreenSpaceReflectionsPass.ts";

assert.equal(resolveSSROptions().raysPerPixel, 4);
assert.equal(resolveSSROptions({ raysPerPixel: 0 }).raysPerPixel, 1);
assert.equal(resolveSSROptions({ raysPerPixel: 99 }).raysPerPixel, 8);
assert.equal(resolveSSROptions({ raysPerPixel: 3.9 }).raysPerPixel, 3);
assert.equal(resolveSSROptions({ raysPerPixel: NaN }).raysPerPixel, 4);
assert.equal(createSSRTraceParams(8, 8,
	resolveSSROptions({ raysPerPixel: 6 }), 3, true, 1)[15], 6);

const source = readFileSync(new URL(
	"../../../src/shaders/webgpu/postprocess/ssr.wgsl", import.meta.url,
), "utf8");
const names = ["ssrDepthCrossing", "ssrAcceptRefinedHit", "ssrBlendTemporal", "ssrBlendMoments"];
const functions = names.map((name) => {
	const start = source.indexOf(`fn ${name}(`);
	assert.ok(start >= 0, `SSR shader must provide executable ${name} math.`);
	let end = source.indexOf("{", start);
	let depth = 1;
	while (depth > 0 && ++end < source.length) {
		if (source[end] === "{") depth++;
		if (source[end] === "}") depth--;
	}
	assert.equal(depth, 0);
	return source.slice(start, end + 1);
});
const shader = compileWGSLMath(functions.join("\n"), names);

const firstMoments = shader.ssrBlendMoments([0.25, 0.125], [5, 25, 1, 1], 0.85, false);
assert.deepEqual(firstMoments, [0.25, 0.125, 1, 1]);
let moments = [0.25, 0.0625, 1, 1];
for (let frame = 0; frame < 120; frame++) {
	moments = shader.ssrBlendMoments([0.25, 0.0625], moments, 0.85, true);
}
assert.ok(Math.abs(moments[0] - 0.25) < 1e-6);
assert.ok(Math.abs(moments[1] - moments[0] ** 2) < 1e-6);
assert.ok(moments[2] > 12 && moments[2] < 13,
	"Effective temporal samples must converge without inventing spatial blur.");

// A step may jump across a thin depth interval without landing in it. Every
// jitter phase must still produce a bracket for the known planar intersection.
for (let index = 0; index < 1000; index++) {
	const jitter = (index + 0.5) / 1000;
	const frontT = 2 + jitter;
	const backT = 3 + jitter;
	assert.equal(shader.ssrDepthCrossing(frontT - 3, backT - 3), true);
}
assert.equal(shader.ssrDepthCrossing(-1, -0.1), false);
assert.equal(shader.ssrDepthCrossing(1, 2), false);
assert.equal(shader.ssrDepthCrossing(0.1, -0.1), false);

// A sign change at an occlusion edge must not turn an unrelated foreground or
// background depth into a reflection hit after refinement.
assert.equal(shader.ssrAcceptRefinedHit(5.01, 5, 0.02), true);
assert.equal(shader.ssrAcceptRefinedHit(5.3, 5, 0.02), false);
assert.equal(shader.ssrAcceptRefinedHit(5, 12, 0.02), false);
assert.equal(shader.ssrAcceptRefinedHit(5, 0, 0.02), false);

const current = [0, 0, 0, 0];
const history = [2, 1, 0.5, 0.8];
const miss = shader.ssrBlendTemporal(current, history, 0.85, true);
assert.ok(miss[0] > 1.69 && miss[0] < 1.71,
	"A miss must retain decaying valid history rather than replacing it with zero.");
assert.ok(miss[3] > 0 && miss[3] < history[3]);
assert.deepEqual(shader.ssrBlendTemporal(current, history, 0.85, false), current);
assert.deepEqual(shader.ssrBlendTemporal(current, history, 0, true), current);
assert.deepEqual(shader.ssrBlendTemporal([1, 0.5, 0.25, 0.4], history, 0.85, false),
	[1, 0.5, 0.25, 0.4].map(Math.fround));
let decayed = history;
for (let frame = 0; frame < 120; frame++) {
	decayed = shader.ssrBlendTemporal(current, decayed, 0.85, true);
}
assert.ok(decayed.every((value) => value < 1e-6),
	"Repeated misses must eventually remove old reflections.");

console.log("SSR depth crossing, refined hit rejection, and temporal miss math passed.");
