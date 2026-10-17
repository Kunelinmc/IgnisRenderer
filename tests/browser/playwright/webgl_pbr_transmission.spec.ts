import { expect, test } from "@playwright/test";

import { withBrowserFixture } from "../helpers/browserFixture.ts";

test("WebGL glass retains directional transmission without diffuse back-light haze", async ({ page }) => {
	test.setTimeout(120_000);
	await withBrowserFixture(page, async () => {
		const { setup } = await import("../fixtures/pbr_transmission.ts");
		return setup("webgl");
	}, async (result) => {
		test.skip(!result.supported, !result.supported ? result.reason : undefined);
		if (!result.supported) return;
		await test.info().attach("transmission-radiance", {
			body: JSON.stringify(result, null, 2), contentType: "application/json",
		});
		expect(result.errors).toEqual([]);
		for (const row of result.rows) {
			const label = `${row.kind}, clustered=${row.clustered}`;
			expect(Number.isFinite(row.aligned), label).toBe(true);
			expect(row.aligned, label).toBeGreaterThan(0);
			expect(row.offAxis, label).toBeLessThan(row.aligned * 0.01);
			expect(row.roughOffAxis, label).toBeGreaterThan(row.offAxis * 10);
			expect(row.metal, label).toBe(0);
			expect(row.noTransmission, label).toBe(0);
		}
		const { bare, clearcoated, absorbed, unshadowed, shadowed } = result.attenuation;
		expect(clearcoated / bare).toBeCloseTo(0.96, 2);
		expect(absorbed / bare).toBeCloseTo(64 / 255, 2);
		expect(unshadowed).toBeGreaterThan(0);
		expect(shadowed).toBeLessThan(unshadowed * 0.01);
		for (const row of result.shadowRows) {
			const label = `${row.kind} shadow, clustered=${row.clustered}`;
			expect(row.unshadowed, label).toBeGreaterThan(0);
			expect(row.shadowed, label).toBeLessThan(row.unshadowed * 0.01);
		}
		expect(result.iorRatios[1]).toBeGreaterThan(0);
		expect(result.iorRatios[1]).toBeLessThan(result.iorRatios[0] * 0.1);
		expect(Number.isFinite(result.grazing)).toBe(true);
		expect(result.grazing).toBeGreaterThan(0);
		expect(result.rejectedMicrofacet).toBe(0);
	});
});
