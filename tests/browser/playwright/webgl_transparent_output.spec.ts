import { expect, test } from "@playwright/test";

test("WebGL transparent output composites with the DOM background", async ({
	page,
}) => {
	await page.setViewportSize({ width: 96, height: 48 });
	await page.goto("/tests/browser/fixtures/webgl_transparent_output.html");
	await page.waitForFunction(() =>
		window.webglTransparentOutputReady === true ||
		window.webglTransparentOutputError !== undefined
	);
	const error = await page.evaluate(() => window.webglTransparentOutputError);
	expect(error).toBeUndefined();

	const screenshot = await page.screenshot({ type: "png" });
	const pixels = await page.evaluate(async (base64) => {
		const image = new Image();
		image.src = `data:image/png;base64,${base64}`;
		await image.decode();
		const sampleCanvas = document.createElement("canvas");
		sampleCanvas.width = image.width;
		sampleCanvas.height = image.height;
		const context = sampleCanvas.getContext("2d");
		if (!context) throw new Error("Failed to create screenshot sampling context.");
		context.drawImage(image, 0, 0);
		return {
			transparent: [...context.getImageData(16, 16, 1, 1).data],
		};
	}, screenshot.toString("base64"));

	expect(pixels.transparent.slice(0, 3)).toEqual([20, 120, 220]);
});
