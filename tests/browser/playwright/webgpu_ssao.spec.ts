import { expect, test } from "@playwright/test";

import { withBrowserFixture } from "../helpers/browserFixture.ts";

test("SSAO keeps a physical contact radius across projections and resolutions", async ({ page }) => {
	await withBrowserFixture(page, async () => {
		const { setup } = await import("../fixtures/webgpu_ssao.ts");
		return setup();
	}, (result) => {
		test.skip(!result.supported, "Chromium did not expose a WebGPU adapter.");
		if (!result.supported) return;
		expect(result.validation).toBeNull();
		expect(result.shaderCompilationErrors).toEqual([]);
		for (const entry of result.cases) {
			expect(entry.flat.maximumError).toBeLessThan(0.001);
			expect(entry.zeroIntensity.maximumError).toBeLessThan(0.001);
			expect(entry.excessiveBias.maximumError).toBeLessThan(0.001);
			expect(entry.contact.minimum).toBeLessThan(0.96);
			expect(entry.contact.coverage).toBeGreaterThan(0.01);
		}
		for (const orthographic of [false, true]) {
			const coverage = result.cases.filter((entry) => entry.orthographic === orthographic)
				.map((entry) => entry.contact.coverage);
			expect(Math.max(...coverage) - Math.min(...coverage)).toBeLessThan(0.02);
		}
	});
});
