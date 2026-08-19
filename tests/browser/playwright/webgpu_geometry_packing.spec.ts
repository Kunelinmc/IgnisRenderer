import { expect, test } from "@playwright/test";

test("WebGPU accepts packed semantic geometry layouts", async ({ page }) => {
	await page.goto("/");
	const result = await page.evaluate(async () => {
		if (!navigator.gpu) return { supported: false as const };
		const adapter = await navigator.gpu.requestAdapter();
		if (!adapter) return { supported: false as const };
		const device = await adapter.requestDevice();
		const { packWebGPUVertexGeometry } = await import(
			"/src/backends/webgpu/WebGPUGeometryPacking.ts"
		);
		const { ShaderSource } = await import("/src/shaders/ShaderSource.ts");
		const {
			DEFAULT_SHADER_DIRECTIVE_PROFILE_REGISTRY,
			ShaderBackendCompileStage,
			ShaderRuntime,
		} = await import("/src/shaders/runtime/index.ts");
		const compileStage = new ShaderBackendCompileStage({
			backend: "webgpu",
			runtime: new ShaderRuntime({ mode: "strict" }),
			profiles: DEFAULT_SHADER_DIRECTIVE_PROFILE_REGISTRY,
			mode: "strict",
		});
		const shaderCompilationErrors: string[] = [];
		for (const id of ["webgpu.scene.composite", "webgpu.shadow.depth.composite"]) {
			const composite = await ShaderSource.load(id);
			const processed = await compileStage.compileAsync({
				code: composite.code,
				language: "wgsl",
				stage: "vertex",
				entryPoint: "vsMain",
				label: id,
				sourceKind: id.includes("shadow") ? "shadow" : "scene",
				sourceMap: composite.sourceMap,
				directiveSourcePath:
					composite.sourceMap.segments[0]?.sourcePath ?? id,
			});
			const module = device.createShaderModule({ code: processed.code });
			const info = await module.getCompilationInfo();
			for (const message of info.messages) {
				if (message.type === "error") {
					shaderCompilationErrors.push(`${id}: ${message.message}`);
				}
			}
		}
		const geometry = {
			positions: new Float32Array([
				-1, -1, 0,
				3, -1, 0,
				-1, 3, 0,
			]),
			normals: new Float32Array([
				0, 0, 1,
				0, 0, 1,
				0, 0, 1,
			]),
			uv0: new Float32Array([0, 0, 1, 0, 0, 1]),
			indices: new Uint32Array([0, 1, 2]),
		};
		const packed = packWebGPUVertexGeometry(geometry, 3);
		const createBuffer = (data: ArrayBufferView, usage: GPUBufferUsageFlags) => {
			const byteLength = Math.max(4, (data.byteLength + 3) & ~3);
			const buffer = device.createBuffer({
				size: byteLength,
				usage: usage | GPUBufferUsage.COPY_DST,
			});
			if ((data.byteLength & 3) === 0) {
				device.queue.writeBuffer(buffer, 0, data as GPUAllowSharedBufferSource);
			} else {
				const padded = new Uint8Array(byteLength);
				padded.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
				device.queue.writeBuffer(buffer, 0, padded);
			}
			return buffer;
		};
		const position = createBuffer(packed.position.data, GPUBufferUsage.VERTEX);
		const surface = createBuffer(packed.surface!.data, GPUBufferUsage.VERTEX);
		const defaults = createBuffer(packed.defaultData, GPUBufferUsage.VERTEX);
		const indices = createBuffer(new Uint16Array([0, 1, 2]), GPUBufferUsage.INDEX);
		const target = device.createTexture({
			size: [4, 4],
			format: "rgba8unorm",
			usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
		});
		const readback = device.createBuffer({
			size: 256 * 4,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
		});
		const shader = device.createShaderModule({ code: `
			struct VertexInput {
				@location(0) position: vec3<f32>,
				@location(1) uv0: vec2<f32>,
				@location(2) normal: vec3<f32>,
				@location(3) tangent: vec4<f32>,
				@location(4) uv1: vec2<f32>,
				@location(5) joints0: vec4<f32>,
				@location(6) weights0: vec4<f32>,
				@location(7) joints1: vec4<f32>,
				@location(8) weights1: vec4<f32>,
				@location(9) uv2: vec2<f32>,
				@location(10) uv3: vec2<f32>,
			}
			struct VertexOutput {
				@builtin(position) position: vec4<f32>,
				@location(0) color: vec4<f32>,
			}
			@vertex fn vsMain(input: VertexInput) -> VertexOutput {
				var output: VertexOutput;
				output.position = vec4<f32>(input.position, 1.0);
				let defaults = input.tangent.x + input.uv1.x + input.uv2.x +
					input.uv3.x + input.joints0.x + input.weights0.x +
					input.joints1.x + input.weights1.x;
				output.color = vec4<f32>(input.uv0, input.normal.z, 1.0) +
					vec4<f32>(defaults * 0.01);
				return output;
			}
			@fragment fn fsMain(input: VertexOutput) -> @location(0) vec4<f32> {
				return input.color;
			}
		` });
		device.pushErrorScope("validation");
		const pipeline = device.createRenderPipeline({
			layout: "auto",
			vertex: {
				module: shader,
				entryPoint: "vsMain",
				buffers: packed.sceneLayouts as GPUVertexBufferLayout[],
			},
			fragment: {
				module: shader,
				entryPoint: "fsMain",
				targets: [{ format: "rgba8unorm" }],
			},
			primitive: { topology: "triangle-list" },
		});
		const encoder = device.createCommandEncoder();
		const pass = encoder.beginRenderPass({
			colorAttachments: [{
				view: target.createView(),
				clearValue: { r: 0, g: 0, b: 0, a: 0 },
				loadOp: "clear",
				storeOp: "store",
			}],
		});
		pass.setPipeline(pipeline);
		pass.setVertexBuffer(0, position);
		pass.setVertexBuffer(1, surface);
		pass.setVertexBuffer(2, defaults);
		pass.setVertexBuffer(3, defaults);
		pass.setIndexBuffer(indices, "uint16");
		pass.drawIndexed(3);
		pass.end();
		encoder.copyTextureToBuffer(
			{ texture: target },
			{ buffer: readback, bytesPerRow: 256 },
			[4, 4]
		);
		device.queue.submit([encoder.finish()]);
		await readback.mapAsync(GPUMapMode.READ);
		const bytes = new Uint8Array(readback.getMappedRange()).slice();
		readback.unmap();
		const validation = await device.popErrorScope();
		let coloredPixels = 0;
		for (let y = 0; y < 4; y++) {
			for (let x = 0; x < 4; x++) {
				if (bytes[y * 256 + x * 4 + 2] > 0) coloredPixels++;
			}
		}
		device.destroy();
		return {
			supported: true as const,
			validation: validation?.message ?? null,
			shaderCompilationErrors,
			coloredPixels,
			vertexByteLength: packed.vertexByteLength,
			positionStride: packed.sceneLayouts[0].arrayStride,
			surfaceStride: packed.sceneLayouts[1].arrayStride,
			defaultStride: packed.sceneLayouts[3].arrayStride,
		};
	});

	test.skip(!result.supported, "Chromium did not expose a WebGPU adapter.");
	if (!result.supported) return;
	expect(result.validation).toBeNull();
	expect(result.shaderCompilationErrors).toEqual([]);
	expect(result.coloredPixels).toBeGreaterThan(0);
	expect(result.vertexByteLength).toBe(84);
	expect(result.positionStride).toBe(12);
	expect(result.surfaceStride).toBe(16);
	expect(result.defaultStride).toBe(0);
});
