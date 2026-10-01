import { expect, test } from "@playwright/test";

import { withBrowserFixture } from "../helpers/browserFixture.ts";

test("WebGL auxiliary raster draws geometry and accelerates IBL", async ({ page }) => {
	await withBrowserFixture(page, async () => {
		const { setup } = await import("../fixtures/webgl_auxiliary_raster.ts");
		return setup();
	}, (result) => {
		expect(result.center[0]).toBeGreaterThan(0.8);
		expect(result.center[1]).toBeGreaterThan(0.15);
		expect(result.corner[0]).toBeLessThan(0.1);
		expect(result.mipCount).toBe(1);
		expect(result.mipDataIsFloat32).toBe(true);
	});
});
