import { expect, test } from "@playwright/test";

import { withBrowserFixture } from "../helpers/browserFixture.ts";

test("WebGPU batches compatible static meshes", async ({ page }) => {
	await withBrowserFixture(page, async () => {
		const { setup } = await import("../fixtures/webgpu_static_mesh_batching.ts");
		return setup();
	}, (result) => {
		test.skip(!result.supported, "Chromium did not expose a WebGPU adapter.");
		if (!result.supported) return;
		expect(result.instanceCounts).toContain(2);
	});
});
