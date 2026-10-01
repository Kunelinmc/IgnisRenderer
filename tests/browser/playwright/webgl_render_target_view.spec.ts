import { expect, test } from "@playwright/test";

import { withBrowserFixture } from "../helpers/browserFixture.ts";

test("WebGL renders a committed scene view into an HDR target", async ({ page }) => {
	await withBrowserFixture(page, async () => {
		const { setup } = await import("../fixtures/webgl_render_target_view.ts");
		return setup();
	}, (result) => {
		expect(result.generation).toBe(1);
		expect(result.origin).toBe("top-left");
		expect(result.width).toBe(4);
		expect(result.height).toBe(4);
		expect(result.center[0]).toBeCloseTo(0, 3);
		expect(result.center[3]).toBeCloseTo(1, 3);
		expect(result.probeCompleted).toBe(true);
	});
});
