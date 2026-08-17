import { expect, test } from "@playwright/test";

test("WebGL presents verified float16 Display-P3 HDR and restores SDR", async ({ page }) => {
	await page.goto("/tests/browser/fixtures/webgl_display_hdr.html");
	const result = await page.evaluate(() => window.webglDisplayHDRResult);

	expect(result.raw.hdrFormat).toBe(0x881a);
	expect(result.raw.hdrColorSpace).toBe("display-p3");
	expect(result.raw.pixel[0]).toBeGreaterThan(1);
	expect(result.raw.pixel[1]).toBeCloseTo(0.25, 2);
	expect(result.raw.pixel[2]).toBeCloseTo(0.5, 2);
	expect(result.raw.error).toBe(0);
	expect(result.raw.sdrFormat).toBe(0x8058);
	expect(result.raw.sdrColorSpace).toBe("srgb");

	expect(result.renderer.hdrDynamicRange).toBe("hdr");
	expect(result.renderer.hdrColorSpace).toBe("display-p3");
	expect(result.renderer.hdrFormat).toBe(0x881a);
	expect(result.renderer.sdrDynamicRange).toBe("sdr");
	expect(result.renderer.sdrColorSpace).toBe("srgb");
	expect(result.renderer.sdrFormat).toBe(0x8058);
	expect(result.renderer.alpha).toBe(true);
	expect(result.renderer.antialias).toBe(false);
	expect(result.renderer.premultipliedAlpha).toBe(true);
});
