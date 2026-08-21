import {
	BufferUsage,
	IBLPrefilter,
	PrimitiveTopology,
	Renderer,
	Texture,
	TextureFormat,
	TextureUsage,
	WEBGL_AUXILIARY_RASTER_EXTENSION,
	WebGLBackend,
} from "../../../src/index";

declare global {
	interface Window {
		webglAuxiliaryRasterResult: Promise<{
			center: number[];
			corner: number[];
			mipCount: number;
			mipDataIsFloat32: boolean;
		}>;
	}
}

window.webglAuxiliaryRasterResult = run();

async function run() {
	const canvas = document.querySelector<HTMLCanvasElement>("#surface");
	if (!canvas) throw new Error("Browser test canvas is unavailable.");
	const backend = new WebGLBackend({ shaderMode: "strict" });
	const renderer = new Renderer(canvas, backend);
	await renderer.initialize();
	try {
		const raster = renderer.requireBackendExtension(
			WEBGL_AUXILIARY_RASTER_EXTENSION,
		);
		const geometry = await raster.execute({
			label: "browser-indexed-triangle",
			task: async ({ encoder, resources }) => {
				const target = resources.createTexture({
					width: 8,
					height: 8,
					format: TextureFormat.RGBA8Unorm,
					usage: TextureUsage.RenderAttachment | TextureUsage.CopySrc,
				});
				const vertexModule = await resources.createShaderModule({
					stage: "vertex",
					language: "glsl",
					sourceKind: "unknown",
					code: `#version 300 es
layout(location=0) in vec2 position;
void main() { gl_Position = vec4(position, 0.0, 1.0); }`,
				});
				const fragmentModule = await resources.createShaderModule({
					stage: "fragment",
					language: "glsl",
					sourceKind: "unknown",
					code: `#version 300 es
precision highp float;
out vec4 color;
void main() { color = vec4(1.0, 0.25, 0.0, 1.0); }`,
				});
				const pipeline = await resources.createRenderPipeline({
					vertex: {
						module: vertexModule,
						entryPoint: "main",
						buffers: [{
							arrayStride: 8,
							attributes: [{
								format: "float32x2",
								offset: 0,
								shaderLocation: 0,
							}],
						}],
					},
					fragment: {
						module: fragmentModule,
						entryPoint: "main",
						targets: [{ format: TextureFormat.RGBA8Unorm }],
					},
					primitive: { topology: PrimitiveTopology.TriangleList },
				});
				const vertices = resources.createBuffer({
					size: 24,
					usage: BufferUsage.Vertex,
					initialData: new Float32Array([
						-0.9, -0.9,
						0.9, -0.9,
						0, 0.9,
					]),
				});
				const indices = resources.createBuffer({
					size: 6,
					usage: BufferUsage.Index,
					initialData: new Uint16Array([0, 1, 2]),
				});
				encoder.beginRenderPass({
					colorAttachments: [{
						view: target,
						loadOp: "clear",
						storeOp: "store",
						clearValue: { r: 0, g: 0, b: 0, a: 1 },
					}],
				});
				encoder.setPipeline(pipeline);
				encoder.setVertexBuffer(0, vertices);
				encoder.setIndexBuffer(indices, "uint16");
				encoder.drawIndexed(3);
				encoder.endRenderPass();
				const pixels = (await resources.readTexture({
					texture: target,
					width: 8,
					height: 8,
					format: TextureFormat.RGBA8Unorm,
				})).toRGBAFloat32();
				return {
					center: Array.from(pixels.slice((4 * 8 + 4) * 4, (4 * 8 + 4) * 4 + 4)),
					corner: Array.from(pixels.slice((7 * 8) * 4, (7 * 8) * 4 + 4)),
				};
			},
		});

		const prefiltered = await new IBLPrefilter({ backend }).prefilter(
			new Texture({
				data: new Float32Array([2, 1, 0.5, 1]),
				width: 1,
				height: 1,
				colorSpace: "HDR",
			}),
			{ acceleration: "webgl", maxMipLevels: 1 },
		);
		return {
			...geometry,
			mipCount: prefiltered.mipmaps.length,
			mipDataIsFloat32: prefiltered.mipmaps[0] instanceof Float32Array,
		};
	} finally {
		await renderer.destroy();
	}
}
