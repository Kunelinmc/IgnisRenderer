import { expect, test } from "@playwright/test";

import { withBrowserFixture } from "../helpers/browserFixture.ts";

test("SSR refines crossed surfaces and decays valid history on misses", async ({ page }) => {
	await withBrowserFixture(page, async () => {
		const { setup } = await import("../fixtures/webgpu_ssr.ts");
		return setup();
	}, (result) => {
		test.skip(!result.supported, "Chromium did not expose a WebGPU adapter.");
		if (!result.supported) return;
		expect(result.validation).toBeNull();
		expect(result.shaderCompilationErrors).toEqual([]);
		for (const crossing of result.crossings) {
			expect(crossing.hits).toBe(1000);
			expect(crossing.maximumError).toBeLessThan(0.02);
		}
		expect(result.depthEdgeResult.hits).toBe(0);
		expect(result.pixels.validHistoryMiss[0]).toBeCloseTo(1.7, 2);
		expect(result.pixels.validHistoryMiss[3]).toBeCloseTo(0.68, 2);
		for (const key of ["invalidHistoryMiss", "zeroIntensity", "background", "roughSurface"] as const) {
			expect(result.pixels[key]).toEqual([0, 0, 0, 0]);
		}
		expect(result.pixels.decayedHistory.every((value) => value < 1e-6)).toBe(true);
		expect(result.sampling.oneRay.mean).toBeGreaterThan(0);
		expect(result.sampling.fourRays.mean).toBeGreaterThan(result.sampling.oneRay.mean * 0.5);
		expect(result.sampling.fourRays.mean).toBeLessThan(result.sampling.oneRay.mean * 1.5);
		expect(result.sampling.fourRays.variance).toBeLessThan(result.sampling.oneRay.variance * 0.75);
		for (const color of result.colors) {
			const bodyPeak = Math.max(...color.body);
			const reflectionPeak = Math.max(...color.reflection);
			for (let channel = 0; channel < 3; channel++) {
				expect(color.reflection[channel] / reflectionPeak)
					.toBeCloseTo(color.body[channel] / bodyPeak, 2);
			}
		}
		expect(result.reconstruction.mirrorMaximumError).toBeLessThan(0.001);
		expect(result.reconstruction.cleanEdgeMaximumError).toBeLessThan(0.001);
		expect(result.reconstruction.noisyEdgePeak).toBeGreaterThan(0.5);
		expect(result.reconstruction.noisyEdgeNeighbor).toBeGreaterThan(0);
		expect(result.reconstruction.roughNoisyEdgePeak).toBeLessThan(result.reconstruction.noisyEdgePeak);
		expect(result.reconstruction.convergedEdgePeak).toBeGreaterThan(0.99);
		expect(result.reconstruction.checkerVariance).toBeLessThan(0.0225 * 0.5);
		expect(result.reconstruction.checkerMean).toBeCloseTo(0.4, 2);
	});
});
