import constants from "../../../src/shaders/webgpu/directives/constants.wgsl?raw";
import frameCamera from "../../../src/shaders/webgpu/common/frameCamera.wgsl?raw";
import ssr from "../../../src/shaders/webgpu/postprocess/ssr.wgsl?raw";
import toneMapping from "../../../src/shaders/webgpu/postprocess/toneMapping.wgsl?raw";
import srgb from "../../../src/shaders/webgpu/directives/srgb.wgsl?raw";
import denoise from "../../../src/shaders/webgpu/postprocess/denoise.wgsl?raw";
import probe from "./webgpu_ssr_probe.wgsl?raw";

import { createBrowserTestSession } from "../helpers/browserTestSession.ts";

const PROBE_COUNT = 1000;
const PROBE_BYTE_LENGTH = PROBE_COUNT * 16;

function toHalf(value: number): number {
	if (value === 0) return 0;
	const exponent = Math.floor(Math.log2(value));
	return ((exponent + 15) << 10) + Math.round((value / 2 ** exponent - 1) * 1024);
}

function fromHalf(value: number): number {
	const exponent = (value >> 10) & 31;
	const fraction = value & 1023;
	return exponent === 0 ? fraction * 2 ** -24 : (1 + fraction / 1024) * 2 ** (exponent - 15);
}

/** @internal SSR GPU regression fixture; invoke through withBrowserFixture(). */
export function setup(sourceOverride?: string) {
	return createBrowserTestSession(async (defer) => {
		if (!navigator.gpu) return { supported: false as const };
		const adapter = await navigator.gpu.requestAdapter();
		if (!adapter) return { supported: false as const };
		const device = await adapter.requestDevice();
		defer(() => { device.destroy(); });
		device.pushErrorScope("validation");
		const code = [constants, frameCamera, (sourceOverride ?? ssr).replace(/^#import.*$/gm, ""), probe]
			.join("\n");
		const module = device.createShaderModule({ code });
		const shaderCompilationErrors = (await module.getCompilationInfo()).messages
			.filter((message) => message.type === "error").map((message) => message.message);
		if (shaderCompilationErrors.length) {
			throw new Error(shaderCompilationErrors.join("\n"));
		}
		const createBuffer = (size: number, usage: GPUBufferUsageFlags, data?: Float32Array) => {
			const buffer = device.createBuffer({ size, usage });
			defer(() => { buffer.destroy(); });
			if (data) device.queue.writeBuffer(buffer, 0, data as GPUAllowSharedBufferSource);
			return buffer;
		};
		const writeTexture = (texture: GPUTexture, values: number[]) => {
			device.queue.writeTexture({ texture }, new Uint16Array(values.map(toHalf)),
				{ bytesPerRow: texture.width * 8 }, [texture.width, 1]);
		};
		const createTexture = (values: number[], storage = false) => {
			const texture = device.createTexture({
				size: [values.length / 4, 1], format: "rgba16float",
				usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST |
					GPUTextureUsage.COPY_SRC | (storage ? GPUTextureUsage.STORAGE_BINDING : 0),
			});
			defer(() => { texture.destroy(); });
			writeTexture(texture, values);
			return texture;
		};
		const depth = createTexture([0, 0, 2, 0]);
		const normal = createTexture([0.5, 0.5, 0.01, 0]);
		const hiZ = createTexture([5, 5, 0, 1]);
		const depthEdge = createTexture([12, 12, 0, 1, 1, 1, 0, 1]);
		const history = createTexture([2, 1, 0.5, 0.8]);
		const momentHistory = createTexture([0, 0, 1, 1]);
		const momentOutput = createTexture([0, 0, 0, 0], true);
		const motionHistory = createTexture([0, 0, 2, 0]);
		const scene = createTexture([0, 0, 0, 1]);
		const output = createTexture([0, 0, 0, 0], true);
		const sampler = device.createSampler();
		const data = new Float32Array([
			1, 1, 30, 0.2, 1, 1, 0.85, 0.12, 64, 6, 0, 0.85, 1, 0.02, 0, 4,
		]);
		const params = createBuffer(64, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, data);
		const cameraData = new Float32Array(72);
		for (let matrix = 0; matrix < 4; matrix++) {
			for (let lane = 0; lane < 4; lane++) cameraData[matrix * 16 + lane * 5] = 1;
		}
		const camera = createBuffer(288,
			GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, cameraData);
		const results = createBuffer(PROBE_BYTE_LENGTH,
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
		const readback = createBuffer(PROBE_BYTE_LENGTH,
			GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
		const probeLayout = device.createPipelineLayout({ bindGroupLayouts: [
			device.createBindGroupLayout({ entries: [
				{ binding: 3, visibility: GPUShaderStage.COMPUTE, texture: {} },
				{ binding: 6, visibility: GPUShaderStage.COMPUTE, sampler: {} },
				{ binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: {} },
			] }),
			device.createBindGroupLayout({ entries: [
				{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: {} },
			] }),
			device.createBindGroupLayout({ entries: [
				{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
			] }),
		] });

		async function tracePlane(entryPoint: string, steps: number, texture: GPUTexture) {
			data[8] = steps;
			device.queue.writeBuffer(params, 0, data);
			const pipeline = await device.createComputePipelineAsync({
				layout: probeLayout, compute: { module, entryPoint },
			});
			const groups = [
				device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
					{ binding: 3, resource: texture.createView() },
					{ binding: 6, resource: sampler },
					{ binding: 7, resource: { buffer: params } },
				] }),
				device.createBindGroup({ layout: pipeline.getBindGroupLayout(1), entries: [
					{ binding: 0, resource: { buffer: camera } },
				] }),
				device.createBindGroup({ layout: pipeline.getBindGroupLayout(2), entries: [
					{ binding: 0, resource: { buffer: results } },
				] }),
			];
			const encoder = device.createCommandEncoder();
			const pass = encoder.beginComputePass();
			pass.setPipeline(pipeline);
			groups.forEach((group, index) => pass.setBindGroup(index, group));
			pass.dispatchWorkgroups(Math.ceil(PROBE_COUNT / 64));
			pass.end();
			encoder.copyBufferToBuffer(results, 0, readback, 0, PROBE_BYTE_LENGTH);
			device.queue.submit([encoder.finish()]);
			await readback.mapAsync(GPUMapMode.READ);
			const values = new Float32Array(readback.getMappedRange());
			let hits = 0;
			let maximumError = 0;
			for (let index = 0; index < PROBE_COUNT; index++) {
				hits += values[index * 4];
				if (values[index * 4] > 0.5) {
					maximumError = Math.max(maximumError, Math.abs(values[index * 4 + 1] - 3));
				}
			}
			readback.unmap();
			return { steps, hits, maximumError };
		}
		const crossings = [
			await tracePlane("probeCrossing", 64, hiZ),
			await tracePlane("probeCrossing", 128, hiZ),
		];
		const depthEdgeResult = await tracePlane("probeDepthEdge", 64, depthEdge);
		const trace = await device.createComputePipelineAsync({
			layout: "auto", compute: { module, entryPoint: "csTrace" },
		});
		const group = device.createBindGroup({ layout: trace.getBindGroupLayout(0), entries: [
			{ binding: 0, resource: scene.createView() },
			{ binding: 1, resource: normal.createView() },
			{ binding: 2, resource: depth.createView() },
			{ binding: 3, resource: hiZ.createView() },
			{ binding: 4, resource: history.createView() },
			{ binding: 5, resource: motionHistory.createView() },
			{ binding: 6, resource: sampler },
			{ binding: 7, resource: { buffer: params } },
			{ binding: 8, resource: output.createView() },
			{ binding: 9, resource: momentHistory.createView() },
			{ binding: 10, resource: momentOutput.createView() },
		] });
		const cameraGroup = device.createBindGroup({ layout: trace.getBindGroupLayout(1), entries: [
			{ binding: 0, resource: { buffer: camera } },
		] });
		async function missPixel(
			valid: number, intensity = 1, depthValue = 2, roughness = 0.01, frames = 1,
		) {
			data[12] = valid;
			data[5] = intensity;
			device.queue.writeBuffer(params, 0, data);
			writeTexture(depth, [0, 0, depthValue, 0]);
			writeTexture(normal, [0.5, 0.5, roughness, 0]);
			writeTexture(history, [2, 1, 0.5, 0.8]);
			writeTexture(momentHistory, [0, 0, 1, 1]);
			const encoder = device.createCommandEncoder();
			for (let frame = 0; frame < frames; frame++) {
				const pass = encoder.beginComputePass();
				pass.setPipeline(trace);
				pass.setBindGroup(0, group);
				pass.setBindGroup(1, cameraGroup);
				pass.dispatchWorkgroups(1);
				pass.end();
				if (frames > 1) {
					encoder.copyTextureToTexture({ texture: output }, { texture: history }, [1, 1]);
					encoder.copyTextureToTexture({ texture: momentOutput }, { texture: momentHistory }, [1, 1]);
				}
			}
			encoder.copyTextureToBuffer({ texture: output }, { buffer: readback, bytesPerRow: 256 }, [1, 1]);
			device.queue.submit([encoder.finish()]);
			await readback.mapAsync(GPUMapMode.READ);
			const pixel = Array.from(new Uint16Array(readback.getMappedRange()).slice(0, 4)).map(fromHalf);
			readback.unmap();
			return pixel;
		}
		const pixels = {
			validHistoryMiss: await missPixel(1),
			invalidHistoryMiss: await missPixel(0),
			zeroIntensity: await missPixel(1, 0),
			background: await missPixel(1, 1, 0),
			roughSurface: await missPixel(1, 1, 2, 1),
			decayedHistory: await missPixel(1, 1, 2, 0.01, 120),
		};

		// Sparse emissive radiance makes the one-ray signal noisy. Compare the
		// actual csTrace output over the same frame sequence with one/four rays.
		const stripeValues = Array.from({ length: 16 }, (_, index) =>
			index === 10 || index === 11 ? [6, 3, 1, 1] : [0, 0, 0, 1]).flat();
		const stripes = createTexture(stripeValues);
		const varianceReadback = createBuffer(64 * 256,
			GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
		const stripeGroup = device.createBindGroup({
			layout: trace.getBindGroupLayout(0), entries: [
				{ binding: 0, resource: stripes.createView() },
				{ binding: 1, resource: normal.createView() },
				{ binding: 2, resource: depth.createView() },
				{ binding: 3, resource: hiZ.createView() },
				{ binding: 4, resource: history.createView() },
				{ binding: 5, resource: motionHistory.createView() },
				{ binding: 6, resource: sampler },
				{ binding: 7, resource: { buffer: params } },
				{ binding: 8, resource: output.createView() },
				{ binding: 9, resource: momentHistory.createView() },
				{ binding: 10, resource: momentOutput.createView() },
			],
		});
		cameraData[0] = 0.5;
		device.queue.writeBuffer(camera, 0, cameraData);
		writeTexture(depth, [0, 0, 2, 0]);
		writeTexture(normal, [0.5 + 0.5 / 1.1, 0.5, 0.22, 0.3]);
		async function traceVariance(raysPerPixel: number) {
			data[5] = 1;
			data[12] = 0;
			data[15] = raysPerPixel;
			for (let frame = 0; frame < 64; frame++) {
				data[14] = frame;
				device.queue.writeBuffer(params, 0, data);
				const encoder = device.createCommandEncoder();
				const pass = encoder.beginComputePass();
				pass.setPipeline(trace);
				pass.setBindGroup(0, stripeGroup);
				pass.setBindGroup(1, cameraGroup);
				pass.dispatchWorkgroups(1);
				pass.end();
				encoder.copyTextureToBuffer({ texture: output },
					{ buffer: varianceReadback, offset: frame * 256, bytesPerRow: 256 }, [1, 1]);
				device.queue.submit([encoder.finish()]);
			}
			await varianceReadback.mapAsync(GPUMapMode.READ);
			const values = new Uint16Array(varianceReadback.getMappedRange());
			const signal = Array.from({ length: 64 }, (_, frame) => fromHalf(values[frame * 128]));
			varianceReadback.unmap();
			const mean = signal.reduce((sum, value) => sum + value, 0) / signal.length;
			const variance = signal.reduce((sum, value) => sum + (value - mean) ** 2, 0) /
				signal.length;
			return { mean, variance };
		}
		const sampling = { oneRay: await traceVariance(1), fourRays: await traceVariance(4) };

		// Read back the actual display-linear mapping of an emitter and the same
		// radiance attenuated by reflection. Their normalized RGB must agree.
		const toneModule = device.createShaderModule({ code: [constants, srgb,
			toneMapping.replace(/^#import.*$/gm, "")].join("\n") });
		const tonePipeline = await device.createComputePipelineAsync({
			layout: "auto", compute: { module: toneModule, entryPoint: "csMain" },
		});
		const toneSource = createTexture([0, 0, 0, 1]);
		const toneParams = createBuffer(16,
			GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, new Float32Array([1, 1, 0, 0]));
		const toneGroup = device.createBindGroup({ layout: tonePipeline.getBindGroupLayout(0), entries: [
			{ binding: 0, resource: toneSource.createView() },
			{ binding: 1, resource: { buffer: toneParams } },
			{ binding: 2, resource: output.createView() },
		] });
		async function mappedColor(radiance: number[]) {
			writeTexture(toneSource, [...radiance, 1]);
			const encoder = device.createCommandEncoder();
			const pass = encoder.beginComputePass();
			pass.setPipeline(tonePipeline);
			pass.setBindGroup(0, toneGroup);
			pass.dispatchWorkgroups(1);
			pass.end();
			encoder.copyTextureToBuffer({ texture: output },
				{ buffer: readback, bytesPerRow: 256 }, [1, 1]);
			device.queue.submit([encoder.finish()]);
			await readback.mapAsync(GPUMapMode.READ);
			const rgb = Array.from(new Uint16Array(readback.getMappedRange()).slice(0, 3)).map(fromHalf);
			readback.unmap();
			return rgb;
		}
		const colors = [];
		for (const radiance of [[45 / 255 * 6, 205 / 255 * 6, 6], [5, 115 / 255 * 5, 45 / 255 * 5]]) {
			colors.push({ body: await mappedColor(radiance),
				reflection: await mappedColor(radiance.map((value) => value * 0.1)) });
		}

		// Flat mirror receivers have no geometric edge at a reflected image edge.
		// Roughness-aware reconstruction must retain that edge exactly while
		// smoothing the same sparse radiance signal on a rough receiver.
		const denoiseModule = device.createShaderModule({ code: denoise });
		const horizontal = await device.createComputePipelineAsync({ layout: "auto",
			compute: { module: denoiseModule, entryPoint: "csDenoiseHorizontal" } });
		const vertical = await device.createComputePipelineAsync({ layout: "auto",
			compute: { module: denoiseModule, entryPoint: "csDenoiseVertical" } });
		const edgeValues = Array.from({ length: 16 }, (_, index) =>
			index === 8 ? [1, 1, 1, 1] : [0, 0, 0, 1]).flat();
		const edgeSource = createTexture(edgeValues);
		const edgeScratch = createTexture(Array(64).fill(0), true);
		const edgeOutput = createTexture(Array(64).fill(0), true);
		const edgeNormal = createTexture([0.5, 0.5, 0.01, 0]);
		const edgeMoments = createTexture(Array.from({ length: 16 }, () => [0.2, 0.04, 1, 1]).flat());
		async function reconstruct(
			roughness: number, noiseVariance = 0, frames = 1, inputSignal = edgeSource,
		) {
			writeTexture(edgeNormal, [0.5, 0.5, roughness, 0]);
			writeTexture(edgeMoments, Array.from({ length: 16 }, () =>
				[0.2, 0.04 + noiseVariance, frames, 1]).flat());
			const encoder = device.createCommandEncoder();
			let source = inputSignal;
			for (const step of [1, 2, 4]) {
				const filterParams = createBuffer(64,
					GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
					new Float32Array([1 / 16, 1, 2, step, 48, 32, 0, 0.05, 0, 1, 1, 1, 0, 0, 0, 0]));
				for (const [pipeline, input, target] of [
					[horizontal, source, edgeScratch], [vertical, edgeScratch, edgeOutput],
				] as const) {
					const binding = device.createBindGroup({
						layout: pipeline.getBindGroupLayout(0), entries: [
							{ binding: 0, resource: input.createView() },
							{ binding: 1, resource: edgeNormal.createView() },
							{ binding: 2, resource: depth.createView() },
							{ binding: 3, resource: sampler },
							{ binding: 4, resource: { buffer: filterParams } },
							{ binding: 5, resource: target.createView() },
							{ binding: 6, resource: edgeMoments.createView() },
						],
					});
					const pass = encoder.beginComputePass();
					pass.setPipeline(pipeline);
					pass.setBindGroup(0, binding);
					pass.dispatchWorkgroups(2, 1);
					pass.end();
				}
				source = edgeOutput;
			}
			encoder.copyTextureToBuffer({ texture: edgeOutput },
				{ buffer: readback, bytesPerRow: 256 }, [16, 1]);
			device.queue.submit([encoder.finish()]);
			await readback.mapAsync(GPUMapMode.READ);
			const signal = Array.from(new Uint16Array(readback.getMappedRange()).slice(0, 64))
				.map(fromHalf);
			readback.unmap();
			return signal;
		}
		const mirror = await reconstruct(0.01);
		const rough = await reconstruct(0.22);
		const noisyEdge = await reconstruct(0.22, 0.16);
		const roughNoisyEdge = await reconstruct(0.65, 0.16);
		const convergedEdge = await reconstruct(0.22, 0.16, 128);
		const checkerValues = Array.from({ length: 16 }, (_, index) => {
			const value = index % 2 ? 0.55 : 0.25;
			return [value, value, value, 1];
		}).flat();
		const checkerSource = createTexture(checkerValues);
		const smoothedChecker = await reconstruct(0.22, 0.16, 1, checkerSource);
		const innerValues = Array.from({ length: 12 }, (_, index) => smoothedChecker[(index + 2) * 4]);
		const mean = innerValues.reduce((sum, value) => sum + value, 0) / innerValues.length;
		const reconstruction = {
			mirrorMaximumError: Math.max(...mirror.map((value, index) => Math.abs(value - edgeValues[index]))),
			cleanEdgeMaximumError: Math.max(...rough.map((value, index) => Math.abs(value - edgeValues[index]))),
			noisyEdgePeak: noisyEdge[8 * 4],
			noisyEdgeNeighbor: noisyEdge[7 * 4],
			roughNoisyEdgePeak: roughNoisyEdge[8 * 4],
			convergedEdgePeak: convergedEdge[8 * 4],
			checkerVariance: innerValues.reduce((sum, value) => sum + (value - mean) ** 2, 0) /
				innerValues.length,
			checkerMean: mean,
		};
		const validation = await device.popErrorScope();
		return {
			supported: true as const, crossings, depthEdgeResult, pixels,
			sampling, colors, reconstruction, shaderCompilationErrors,
			validation: validation?.message ?? null,
		};
	});
}
