import { expect, test } from "@playwright/test";

import { withBrowserFixture } from "../helpers/browserFixture.ts";

test("WebGPU Phong responds to the direction of incident light", async ({ page }) => {
	await withBrowserFixture(page, async () => {
		const { setup } = await import("../fixtures/webgpu_phong_lighting.ts");
		return setup();
	}, (result) => {
		test.skip(!result.supported, !result.supported ? result.reason : undefined);
		if (!result.supported) return;
		expect(result.errors).toEqual([]);
		expect(Number.isFinite(result.lit)).toBe(true);
		expect(result.lit).toBeGreaterThan(0);
		expect(result.backlit).toBeLessThan(result.lit * 0.01);
	});
});
