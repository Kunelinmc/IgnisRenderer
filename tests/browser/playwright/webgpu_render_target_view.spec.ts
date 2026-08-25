import { expect, test } from "@playwright/test";

test("WebGPU records a scene view target in the frame transaction", async ({ page }) => {
	await page.goto("/");
	const result = await page.evaluate(async () => {
		if (!navigator.gpu) return { supported: false as const };
		const canvas = document.createElement("canvas");
		canvas.width = 8;
		canvas.height = 8;
		document.body.appendChild(canvas);
		const { Renderer } = await import("/src/rendering/Renderer.ts");
		const { WebGPUBackend } = await import("/src/backends/webgpu/WebGPUBackend.ts");
		const { Camera } = await import("/src/cameras/Camera.ts");
		const { TextureFormat } = await import("/src/core/TextureFormat.ts");
		const renderer = new Renderer(canvas, new WebGPUBackend(), new Camera());
		renderer.features.enableEnvironment = false;
		renderer.features.enableShadows = false;
		try {
			await renderer.initialize();
		} catch {
			return { supported: false as const };
		}
		const target = renderer.renderTargets.create({
			size: { mode: "fixed", width: 4, height: 4 },
			color: [{ format: TextureFormat.RGBA16Float }],
			depth: { format: TextureFormat.Depth32Float },
		});
		const ticket = target.enqueueJob({
			kind: "scene-view",
			camera: renderer.camera,
			content: { environment: false, particles: false, shadows: "disabled" },
			readback: { attachmentIndex: 0 },
		});
		await renderer.renderFrame(performance.now());
		const completion = await ticket.done;
		const pixels = completion.readback?.toRGBAFloat32() ?? new Float32Array();
		const output = {
			supported: true as const,
			generation: completion.generation,
			origin: completion.readback?.origin,
			center: Array.from(pixels.slice(0, 4)),
		};
		await renderer.destroy();
		return output;
	});
	test.skip(!result.supported, "WebGPU is unavailable in this browser environment");
	if (!result.supported) return;
	expect(result.generation).toBe(1);
	expect(result.origin).toBe("top-left");
	expect(result.center[0]).toBeCloseTo(0, 3);
	expect(result.center[3]).toBeCloseTo(1, 3);
});
