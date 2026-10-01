import { expect, test } from "@playwright/test";

import { withBrowserFixture } from "../helpers/browserFixture.ts";

test("WebGPU accepts packed semantic geometry layouts", async ({ page }) => {
	await withBrowserFixture(page, async () => {
		const { setup } = await import("../fixtures/webgpu_geometry_packing.ts");
		return setup();
	}, (result) => {
		test.skip(!result.supported, "Chromium did not expose a WebGPU adapter.");
		if (!result.supported) return;
		expect(result.validation).toBeNull();
		expect(result.shaderCompilationErrors).toEqual([]);
		expect(result.coloredPixels).toBeGreaterThan(0);
		expect(result.vertexByteLength).toBe(84);
		expect(result.positionStride).toBe(12);
		expect(result.surfaceStride).toBe(16);
		expect(result.defaultStride).toBe(0);
	});
});
