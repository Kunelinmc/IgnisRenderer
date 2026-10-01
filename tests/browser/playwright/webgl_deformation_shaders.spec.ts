import { expect, test } from "@playwright/test";

import { withBrowserFixture } from "../helpers/browserFixture.ts";

test("WebGL compiles built-in and ShaderMaterial deformation ABI", async ({ page }) => {
	await withBrowserFixture(page, async () => {
		const { setup } = await import("../fixtures/webgl_deformation_shaders.ts");
		return setup();
	}, (result) => {
		test.skip(!result.supported, "WebGL2 is unavailable.");
		if (!result.supported) return;
		expect(result.errors).toEqual([]);
	});
});
