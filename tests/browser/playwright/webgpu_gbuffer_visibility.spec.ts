import { expect, test } from "@playwright/test";

import { withBrowserFixture } from "../helpers/browserFixture.ts";

test("WebGPU extension payloads follow the visible surface for all PBR lobes", async ({ page }) => {
	test.setTimeout(120_000);
	await withBrowserFixture(page, async () => {
		const { setup } = await import("../fixtures/webgpu_gbuffer_visibility.ts");
		return setup();
	}, (result) => {
		test.skip(!result.supported, !result.supported ? result.reason : undefined);
		if (!result.supported) return;
		expect(result.errors).toEqual([]);
		expect(result.customResults).toHaveLength(2);
		for (const custom of result.customResults) {
			expect(custom.sourcePreserved).toBe(true);
			expect(custom.rows).toHaveLength(12);
			expect(custom.rows.filter((row) => row.case === "coplanar")
				.map((row) => row.coplanarWinner).sort()).toEqual(["front", "rear"]);
			for (const row of custom.rows) {
				const label = `custom earlyZ=${custom.earlyZ}: ${JSON.stringify(row)}`;
				expect(row.extensions && row.lighting, label).toBe(true);
				expect(row.error, label).toBe(0);
				expect(row.baselineDifference, label).toBeGreaterThan(0);
				expect(row.earlyZDraws, label).toBe(0);
			}
			expect(custom.negative.map((row) => row.entry).sort()).toEqual(["base", "extensions"]);
			for (const row of custom.negative) {
				expect(row.rejected, row.message).toBe(true);
				expect(row.recordedPasses, row.message).toEqual([]);
				expect(row.recovered, row.message).toBe(true);
			}
		}
		expect(result.rows).toHaveLength(24);
		for (const earlyZ of [false, true]) {
			for (const lobe of ["clearcoat", "iridescence", "anisotropy"]) {
				expect(result.rows.filter((row) => row.coplanar && row.earlyZ === earlyZ &&
					row.lobe === lobe).map((row) => row.winner).sort()).toEqual(["front", "rear"]);
			}
		}
		expect(result.samples.every((sample) => sample.extensions && sample.lighting)).toBe(true);
		for (const row of result.rows) {
			const label = JSON.stringify(row);
			expect(row.baselineDifference, label).toBeGreaterThan(5);
			expect(row.error, label).toBeLessThanOrEqual(2);
		}
	});
});
