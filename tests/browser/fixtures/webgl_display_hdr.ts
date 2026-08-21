import { Renderer, WebGLBackend } from "../../../src/index";

interface ExtendedWebGL2RenderingContext extends WebGL2RenderingContext {
	readonly drawingBufferFormat: number;
	drawingBufferStorage(format: number, width: number, height: number): void;
}

interface WebGLDisplayHDRBrowserResult {
	readonly raw: {
		readonly hdrFormat: number;
		readonly hdrColorSpace: PredefinedColorSpace;
		readonly pixel: number[];
		readonly error: number;
		readonly sdrFormat: number;
		readonly sdrColorSpace: PredefinedColorSpace;
	};
	readonly renderer: {
		readonly hdrDynamicRange: string;
		readonly hdrColorSpace: string;
		readonly hdrFormat: number;
		readonly sdrDynamicRange: string;
		readonly sdrColorSpace: string;
		readonly sdrFormat: number;
		readonly alpha: boolean;
		readonly antialias: boolean;
		readonly premultipliedAlpha: boolean;
	};
}

declare global {
	interface Window {
		webglDisplayHDRResult: Promise<WebGLDisplayHDRBrowserResult>;
	}
}

window.webglDisplayHDRResult = run();

async function run(): Promise<WebGLDisplayHDRBrowserResult> {
	const rawCanvas = document.querySelector<HTMLCanvasElement>("#raw-surface");
	const rendererCanvas = document.querySelector<HTMLCanvasElement>("#renderer-surface");
	if (!rawCanvas || !rendererCanvas) {
		throw new Error("Browser test canvases are unavailable.");
	}

	const rawGL = rawCanvas.getContext("webgl2", {
		alpha: true,
		antialias: false,
		premultipliedAlpha: true,
	}) as ExtendedWebGL2RenderingContext | null;
	if (!rawGL || typeof rawGL.drawingBufferStorage !== "function") {
		throw new Error("Chromium does not expose WebGL drawingBufferStorage().");
	}
	if (!rawGL.getExtension("EXT_color_buffer_float")) {
		throw new Error("Chromium does not expose EXT_color_buffer_float.");
	}
	rawGL.drawingBufferColorSpace = "display-p3";
	rawGL.drawingBufferStorage(rawGL.RGBA16F, 8, 8);
	drawExtendedColor(rawGL);
	const pixel = new Float32Array(4);
	rawGL.readPixels(4, 4, 1, 1, rawGL.RGBA, rawGL.FLOAT, pixel);
	const raw = {
		hdrFormat: rawGL.drawingBufferFormat,
		hdrColorSpace: rawGL.drawingBufferColorSpace,
		pixel: Array.from(pixel),
		error: rawGL.getError(),
		sdrFormat: 0,
		sdrColorSpace: "srgb" as PredefinedColorSpace,
	};
	rawGL.drawingBufferColorSpace = "srgb";
	rawGL.drawingBufferStorage(rawGL.RGBA8, 8, 8);
	raw.sdrFormat = rawGL.drawingBufferFormat;
	raw.sdrColorSpace = rawGL.drawingBufferColorSpace;

	const nativeMatchMedia = window.matchMedia.bind(window);
	window.matchMedia = ((query: string) => {
		if (query === "(dynamic-range: high)") {
			return {
				matches: true,
				media: query,
				onchange: null,
				addEventListener() {},
				removeEventListener() {},
				addListener() {},
				removeListener() {},
				dispatchEvent: () => true,
			};
		}
		return nativeMatchMedia(query);
	}) as typeof window.matchMedia;

	const backend = new WebGLBackend({ shaderMode: "strict" });
	const renderer = new Renderer(rendererCanvas, backend, null, {
		displayOutput: { mode: "hdr", hdrHeadroom: 4 },
	});
	await renderer.initialize();
	try {
		await renderer.renderFrame(0);
		const rendererGL = rendererCanvas.getContext(
			"webgl2",
		) as ExtendedWebGL2RenderingContext | null;
		if (!rendererGL) throw new Error("Renderer WebGL context is unavailable.");
		const hdrState = renderer.getDisplayOutputState();
		const hdrFormat = rendererGL.drawingBufferFormat;
		const attributes = rendererGL.getContextAttributes();
		const sdrState = await renderer.setDisplayOutput({ mode: "sdr" });
		return {
			raw,
			renderer: {
				hdrDynamicRange: hdrState?.activeDynamicRange ?? "missing",
				hdrColorSpace: hdrState?.colorSpace ?? "missing",
				hdrFormat,
				sdrDynamicRange: sdrState.activeDynamicRange,
				sdrColorSpace: sdrState.colorSpace,
				sdrFormat: rendererGL.drawingBufferFormat,
				alpha: attributes?.alpha ?? false,
				antialias: attributes?.antialias ?? true,
				premultipliedAlpha: attributes?.premultipliedAlpha ?? false,
			},
		};
	} finally {
		await renderer.destroy();
		window.matchMedia = nativeMatchMedia;
	}
}

function drawExtendedColor(gl: WebGL2RenderingContext): void {
	const vertex = compileShader(
		gl,
		gl.VERTEX_SHADER,
		`#version 300 es
void main() {
	vec2 positions[3] = vec2[3](
		vec2(-1.0, -1.0),
		vec2(3.0, -1.0),
		vec2(-1.0, 3.0)
	);
	gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);
}`,
	);
	const fragment = compileShader(
		gl,
		gl.FRAGMENT_SHADER,
		`#version 300 es
precision highp float;
out vec4 color;
void main() { color = vec4(2.0, 0.25, 0.5, 1.0); }`,
	);
	const program = gl.createProgram();
	if (!program) throw new Error("Failed to create HDR browser-test program.");
	gl.attachShader(program, vertex);
	gl.attachShader(program, fragment);
	gl.linkProgram(program);
	if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
		throw new Error(gl.getProgramInfoLog(program) ?? "Failed to link HDR test.");
	}
	gl.useProgram(program);
	gl.viewport(0, 0, 8, 8);
	gl.drawArrays(gl.TRIANGLES, 0, 3);
	gl.deleteProgram(program);
	gl.deleteShader(vertex);
	gl.deleteShader(fragment);
}

function compileShader(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
	const shader = gl.createShader(type);
	if (!shader) throw new Error("Failed to create HDR browser-test shader.");
	gl.shaderSource(shader, source);
	gl.compileShader(shader);
	if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
		throw new Error(gl.getShaderInfoLog(shader) ?? "Failed to compile HDR test.");
	}
	return shader;
}
