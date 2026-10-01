import { WebGLBackend } from "../../../src/backends/webgl/WebGLBackend.ts";
import { Renderer } from "../../../src/rendering/Renderer.ts";

import { createBrowserTestSession, createTestCanvas } from "../helpers/browserTestSession.ts";

/** @internal Browser test fixture; invoke through withBrowserFixture(). */
export function setup() {
	return createBrowserTestSession(async (defer) => {
		for (const element of [document.documentElement, document.body]) {
			const background = element.style.background;
			defer(() => { element.style.background = background; });
			element.style.background = "rgb(20, 120, 220)";
		}
		const canvas = createTestCanvas(defer, 32, 32);
		canvas.style.background = "transparent";
		const renderer = new Renderer(canvas, new WebGLBackend(), null, {
			transparentOutput: true,
		});
		defer(() => renderer.destroy());
		await renderer.initialize();
		await renderer.renderFrame(performance.now());
		for (let attempt = 0; attempt < 2; attempt++) {
			await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
			renderer.requestRender();
			await renderer.renderFrame(performance.now() + attempt + 1);
		}
		return null;
	});
}
