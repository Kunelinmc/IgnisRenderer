import { expect, test } from "@playwright/test";

test("WebGL auxiliary raster draws geometry and accelerates IBL", async ({ page }) => {
	await page.goto("/tests/browser/fixtures/webgl_auxiliary_raster.html");
	const result = await page.evaluate(() => window.webglAuxiliaryRasterResult);
	expect(result.center[0]).toBeGreaterThan(0.8);
	expect(result.center[1]).toBeGreaterThan(0.15);
	expect(result.corner[0]).toBeLessThan(0.1);
	expect(result.mipCount).toBe(1);
	expect(result.mipDataIsFloat32).toBe(true);
});
