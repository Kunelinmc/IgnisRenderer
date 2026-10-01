import { createBrowserTestSession, createTestCanvas } from "../helpers/browserTestSession.ts";

/** @internal Browser test fixture; invoke through withBrowserFixture(). */
export function setup() {
	return createBrowserTestSession(async (defer) => {
		if (!navigator.gpu) return { supported: false as const };
		const canvas = createTestCanvas(defer, 8, 8);
		const { Renderer } = await import("../../../src/rendering/Renderer.ts");
		const { WebGPUBackend } = await import("../../../src/backends/webgpu/WebGPUBackend.ts");
		const { Camera } = await import("../../../src/cameras/Camera.ts");
		const { TextureFormat } = await import("../../../src/core/TextureFormat.ts");
		const renderer = new Renderer(canvas, new WebGPUBackend(), new Camera());
		defer(() => renderer.destroy());
		renderer.features.enableEnvironment = false;
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
		return output;
	});
}
