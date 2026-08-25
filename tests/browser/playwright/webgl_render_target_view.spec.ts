import { expect, test } from "@playwright/test";

test("WebGL renders a committed scene view into an HDR target", async ({ page }) => {
	await page.goto("/");
	const result = await page.evaluate(async () => {
		const canvas = document.createElement("canvas");
		canvas.width = 8;
		canvas.height = 8;
		document.body.appendChild(canvas);
		const { Renderer } = await import("/src/rendering/Renderer.ts");
		const { WebGLBackend } = await import("/src/backends/webgl/WebGLBackend.ts");
		const { Camera } = await import("/src/cameras/Camera.ts");
		const { TextureFormat } = await import("/src/core/TextureFormat.ts");
		const renderer = new Renderer(canvas, new WebGLBackend(), new Camera());
		renderer.features.enableEnvironment = false;
		renderer.features.enableShadows = false;
		await renderer.initialize();
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
			generation: completion.generation,
			origin: completion.readback?.origin,
			width: completion.readback?.width,
			height: completion.readback?.height,
			center: Array.from(pixels.slice(0, 4)),
			probeCompleted: false,
		};
		const { ReflectionProbe } = await import("/src/lights/ReflectionProbe.ts");
		const probe = new ReflectionProbe({
			source: "capturedScene",
			captureResolution: { width: 8, height: 4 },
			includeEnvironment: false,
			includeMeshes: true,
			includeTransparent: false,
			includeParticles: false,
			includeShadows: false,
		});
		renderer.scene.add(probe);
		probe.requestCapture();
		for (let frame = 0; frame < 16; frame++) {
			renderer.requestRender("probe-capture");
			await renderer.renderFrame(performance.now() + frame + 1);
		}
		output.probeCompleted = probe.prefilteredMap !== null;
		await renderer.destroy();
		return output;
	});
	expect(result.generation).toBe(1);
	expect(result.origin).toBe("top-left");
	expect(result.width).toBe(4);
	expect(result.height).toBe(4);
	expect(result.center[0]).toBeCloseTo(0, 3);
	expect(result.center[3]).toBeCloseTo(1, 3);
	expect(result.probeCompleted).toBe(true);
});
