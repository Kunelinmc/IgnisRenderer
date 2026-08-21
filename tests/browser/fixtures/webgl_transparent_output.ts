import { WebGLBackend } from "../../../src/backends/webgl/WebGLBackend";
import { Renderer } from "../../../src/rendering/Renderer";

declare global {
	interface Window {
		webglTransparentOutputReady?: boolean;
		webglTransparentOutputError?: string;
	}
}

async function renderCanvas(id: string, transparentOutput: boolean): Promise<void> {
	const canvas = document.getElementById(id) as HTMLCanvasElement | null;
	if (!canvas) throw new Error(`Missing canvas "${id}".`);
	const renderer = new Renderer(canvas, new WebGLBackend(), null, {
		transparentOutput,
	});
	await renderer.initialize();
	await renderer.renderFrame(performance.now());
	for (let attempt = 0; attempt < 2; attempt++) {
		await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
		renderer.requestRender();
		await renderer.renderFrame(performance.now() + attempt + 1);
	}
}

try {
	await renderCanvas("transparent", true);
	window.webglTransparentOutputReady = true;
} catch (error) {
	window.webglTransparentOutputError = String(error);
	throw error;
}
