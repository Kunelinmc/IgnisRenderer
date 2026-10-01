import { expect, test } from "@playwright/test";

import { withBrowserFixture } from "../helpers/browserFixture.ts";

test("WebGPU records a scene view target in the frame transaction", async ({ page }) => {
	await withBrowserFixture(page, async () => {
		const { setup } = await import("../fixtures/webgpu_render_target_view.ts");
		return setup();
	}, (result) => {
		test.skip(!result.supported, "WebGPU is unavailable in this browser environment");
		if (!result.supported) return;
		expect(result.generation).toBe(1);
		expect(result.origin).toBe("top-left");
		expect(result.center[0]).toBeCloseTo(0, 3);
		expect(result.center[3]).toBeCloseTo(1, 3);
	});
});
