import constants from "../../../src/shaders/webgpu/directives/constants.wgsl?raw";
import ssao from "../../../src/shaders/webgpu/postprocess/ssao.wgsl?raw";

import { Camera } from "../../../src/cameras/Camera.ts";
import { OrthographicCamera } from "../../../src/cameras/OrthographicCamera.ts";
import {
	createSSAOKernelParams,
	resolveSSAOOptions,
} from "../../../src/postprocess/passes/ScreenSpaceAmbientOcclusionPass.ts";
import { createBrowserTestSession } from "../helpers/browserTestSession.ts";

function toHalf(value: number): number {
	if (value === 0) return 0;
	const exponent = Math.floor(Math.log2(value));
	return ((exponent + 15) << 10) + Math.round((value / 2 ** exponent - 1) * 1024);
}

function fromHalf(value: number): number {
	const exponent = (value >> 10) & 31;
	return exponent === 0 ? (value & 1023) * 2 ** -24 :
		(1 + (value & 1023) / 1024) * 2 ** (exponent - 15);
}

/** @internal Exercises actual SSAO GPU coverage; invoke through withBrowserFixture(). */
export function setup(sourceOverride?: string) {
	return createBrowserTestSession(async (defer) => {
		if (!navigator.gpu) return { supported: false as const };
		const adapter = await navigator.gpu.requestAdapter();
		if (!adapter) return { supported: false as const };
		const device = await adapter.requestDevice();
		defer(() => { device.destroy(); });
		device.pushErrorScope("validation");
		const module = device.createShaderModule({
			code: constants + "\n" + (sourceOverride ?? ssao).replace(/^#import.*$/gm, ""),
		});
		const errors = (await module.getCompilationInfo()).messages
			.filter((entry) => entry.type === "error").map((entry) => entry.message);
		if (errors.length) throw new Error(errors.join("\n"));
		const pipeline = await device.createComputePipelineAsync({
			layout: "auto", compute: { module, entryPoint: "csRaw" },
		});
		const sampler = device.createSampler({ minFilter: "linear", magFilter: "linear" });
		const cases = [];
		for (const orthographic of [false, true]) {
			const camera = orthographic ? new OrthographicCamera(4) :
				new Camera({ fov: 60, aspectRatio: 1 });
			for (const [size, downsample] of [[64, 1], [64, 2], [128, 2]]) {
				const aoSize = size / downsample;
				const texture = (width: number, storage = false) => {
					const result = device.createTexture({
						size: [width, width], format: "rgba16float",
						usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST |
							GPUTextureUsage.COPY_SRC | (storage ? GPUTextureUsage.STORAGE_BINDING : 0),
					});
					defer(() => { result.destroy(); });
					return result;
				};
				const normal = texture(size), depth = texture(size), output = texture(aoSize, true);
				const normalData = new Uint16Array(size * size * 4);
				for (let i = 0; i < size * size; i++) {
					normalData.set([toHalf(0.5), toHalf(0.5), 0, 0], i * 4);
				}
				device.queue.writeTexture({ texture: normal }, normalData,
					{ bytesPerRow: size * 8 }, [size, size]);
				const uniform = device.createBuffer({ size: 80,
					usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
				defer(() => { uniform.destroy(); });
				const group = device.createBindGroup({
					layout: pipeline.getBindGroupLayout(0), entries: [
						{ binding: 0, resource: normal.createView() },
						{ binding: 1, resource: depth.createView() },
						{ binding: 2, resource: sampler },
						{ binding: 3, resource: { buffer: uniform } },
						{ binding: 4, resource: output.createView() },
					],
				});
				const run = async (contact: boolean, intensity = 1, bias = 0.025) => {
					const data = new Uint16Array(size * size * 4);
					for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
						data[(y * size + x) * 4 + 2] = toHalf(contact && x < size / 2 ? 2.8 : 3);
					}
					device.queue.writeTexture({ texture: depth }, data,
						{ bytesPerRow: size * 8 }, [size, size]);
					const params = createSSAOKernelParams(size, size, aoSize, aoSize,
						resolveSSAOOptions({ radius: 0.65, intensity, bias, samples: 32 }), camera, 0, 0, 0);
					device.queue.writeBuffer(uniform, 0, params as GPUAllowSharedBufferSource);
					const bytesPerRow = aoSize * 8;
					const readback = device.createBuffer({ size: bytesPerRow * aoSize,
						usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
					try {
						const encoder = device.createCommandEncoder();
						const pass = encoder.beginComputePass();
						pass.setPipeline(pipeline);
						pass.setBindGroup(0, group);
						pass.dispatchWorkgroups(Math.ceil(aoSize / 8), Math.ceil(aoSize / 8));
						pass.end();
						encoder.copyTextureToBuffer({ texture: output },
							{ buffer: readback, bytesPerRow }, [aoSize, aoSize]);
						device.queue.submit([encoder.finish()]);
						await readback.mapAsync(GPUMapMode.READ);
						const values = new Uint16Array(readback.getMappedRange());
						let minimum = 1, occluded = 0, maximumError = 0;
						for (let i = 0; i < aoSize * aoSize; i++) {
							const value = fromHalf(values[i * 4]);
							minimum = Math.min(minimum, value);
							maximumError = Math.max(maximumError, Math.abs(1 - value));
							if (value < 0.99) occluded++;
						}
						return { minimum, coverage: occluded / (aoSize * aoSize), maximumError };
					} finally {
						readback.unmap();
						readback.destroy();
					}
				};
				cases.push({ orthographic, size, downsample,
					flat: await run(false), contact: await run(true),
					zeroIntensity: await run(true, 0), excessiveBias: await run(true, 1, 1) });
			}
		}
		return { supported: true as const, cases,
			validation: (await device.popErrorScope())?.message ?? null, shaderCompilationErrors: errors };
	});
}
