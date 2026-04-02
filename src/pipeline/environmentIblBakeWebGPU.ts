import { Texture } from "../core/Texture";
import { lerp } from "../maths/Common";
import {
	AddressMode,
	BufferUsage,
	FilterMode,
	TextureFormat,
	TextureUsage,
	type IBindingGroup,
	type IComputePipeline,
	type IRenderBuffer,
	type IRenderTexture,
	type ISampler,
	type IShaderModule,
} from "../renderers/types";
import {
	resolveWebGPUComputeFacade,
	type WebGPUComputeFacadeSource,
} from "../renderers/webgpu/computeFacade";
import {
	getWebGPUBindGroup,
	getWebGPUComputePipeline,
	getWebGPUTexture,
} from "../renderers/webgpu/WebGPUResourceAccess";
import { alignTo, createTextureUploadData } from "../renderers/webgpu/texture";
import { destroyResource } from "../renderers/webgpu/computeUtils";
import { loadEnvironmentIBLPrefilterShaderSource } from "../shaders/webgpu/environmentIblPrefilterShaderSource";

import {
	ENVIRONMENT_IBL_MAX_MIP_LEVELS,
	buildPrefilteredTexture,
	resolvePrefilterBaseDimensions,
	type EnvironmentIBLPrefilterMipData,
} from "./environmentIblBakeCore";

const WORKGROUP_SIZE = 8;
const PREFILTER_PARAMS_SIZE = 32;
const GPU_MAX_SAMPLE_COUNT = 256;
const GPU_MIN_SAMPLE_COUNT = 48;

interface EnvironmentIBLWebGPUContext {
	device: GPUDevice;
	queue: GPUQueue;
}

interface EnvironmentIBLWebGPUResources {
	sampler: ISampler;
	module: IShaderModule;
	pipeline: IComputePipeline;
	inputTexture: IRenderTexture;
}

interface EnvironmentIBLMipResources {
	outputTexture: IRenderTexture;
	paramsBuffer: IRenderBuffer;
	bindGroup: IBindingGroup;
}

function createAbortError(): Error {
	const error = new Error("Environment IBL bake was aborted");
	error.name = "AbortError";
	return error;
}

function assertNotAborted(signal?: AbortSignal | null): void {
	if (!signal?.aborted) return;
	throw createAbortError();
}

function resolveWebGPUContext(
	source: WebGPUComputeFacadeSource
): EnvironmentIBLWebGPUContext {
	if (!source || typeof source !== "object") {
		throw new Error(
			"WebGPU acceleration requires a webgpuSource that exposes an initialized GPU device and queue."
		);
	}

	const visited = new WeakSet<object>();
	let current: unknown = source;
	let depth = 0;

	while (current && typeof current === "object") {
		if (depth++ > 32) {
			break;
		}
		const currentObject = current as object;
		if (visited.has(currentObject)) {
			break;
		}
		visited.add(currentObject);

		const candidate = current as {
			device?: GPUDevice;
			queue?: GPUQueue;
			backend?: unknown;
			getComputeFacade?: () => unknown;
		};
		const device = candidate.device;
		const queue = candidate.queue ?? candidate.device?.queue;
		if (
			device &&
			queue &&
			typeof device.createCommandEncoder === "function" &&
			typeof queue.submit === "function"
		) {
			return { device, queue };
		}

		if (typeof candidate.getComputeFacade === "function") {
			const resolved = candidate.getComputeFacade();
			if (resolved && resolved !== currentObject) {
				current = resolved;
				continue;
			}
		}

		if (candidate.backend && candidate.backend !== currentObject) {
			current = candidate.backend;
			continue;
		}

		break;
	}

	throw new Error(
		"WebGPU acceleration requires a webgpuSource that exposes an initialized GPU device and queue."
	);
}

function resolveGpuSampleCount(level: number, maxMipLevels: number): number {
	if (maxMipLevels <= 1) {
		return GPU_MAX_SAMPLE_COUNT;
	}
	const roughness = level / (maxMipLevels - 1);
	const sampleCount = Math.floor(
		lerp(GPU_MAX_SAMPLE_COUNT, GPU_MIN_SAMPLE_COUNT, roughness)
	);
	return Math.max(GPU_MIN_SAMPLE_COUNT, sampleCount);
}

function createPrefilterParamsBuffer(
	width: number,
	height: number,
	sourceWidth: number,
	sourceHeight: number,
	roughness: number,
	sampleCount: number,
	sourceIsLinear: boolean
): ArrayBuffer {
	const buffer = new ArrayBuffer(PREFILTER_PARAMS_SIZE);
	const view = new DataView(buffer);
	view.setUint32(0, width, true);
	view.setUint32(4, height, true);
	view.setUint32(8, sourceWidth, true);
	view.setUint32(12, sourceHeight, true);
	view.setFloat32(16, roughness, true);
	view.setUint32(20, sampleCount, true);
	view.setUint32(24, sourceIsLinear ? 1 : 0, true);
	view.setUint32(28, 0, true);
	return buffer;
}

function unpackRGBA8Readback(
	readbackBytes: Uint8Array,
	width: number,
	height: number,
	bytesPerRow: number
): Float32Array {
	const output = new Float32Array(width * height * 4);
	for (let y = 0; y < height; y++) {
		const srcRowOffset = y * bytesPerRow;
		const dstRowOffset = y * width * 4;
		for (let x = 0; x < width; x++) {
			const srcOffset = srcRowOffset + x * 4;
			const dstOffset = dstRowOffset + x * 4;
			output[dstOffset] = readbackBytes[srcOffset] / 255;
			output[dstOffset + 1] = readbackBytes[srcOffset + 1] / 255;
			output[dstOffset + 2] = readbackBytes[srcOffset + 2] / 255;
			output[dstOffset + 3] = 1;
		}
	}
	return output;
}

function createPrefilterMipResources(
	pipeline: IComputePipeline,
	sampler: ISampler,
	inputTexture: IRenderTexture,
	outputTexture: IRenderTexture,
	paramsBuffer: IRenderBuffer,
	computeFacade: ReturnType<typeof resolveWebGPUComputeFacade>,
	level: number
): EnvironmentIBLMipResources {
	const bindGroup = computeFacade.createBindingGroup({
		pipeline,
		layoutIndex: 0,
		entries: [
			{ binding: 0, resource: sampler },
			{ binding: 1, resource: inputTexture },
			{ binding: 2, resource: outputTexture },
			{ binding: 3, resource: paramsBuffer },
		],
		label: `EnvironmentIBLBakePrefilterBindGroup_mip${level}`,
	});

	return {
		outputTexture,
		paramsBuffer,
		bindGroup,
	};
}

async function createWebGPUResources(
	envMap: Texture,
	computeFacade: ReturnType<typeof resolveWebGPUComputeFacade>
): Promise<EnvironmentIBLWebGPUResources> {
	const shaderCode = await loadEnvironmentIBLPrefilterShaderSource();
	const module = await computeFacade.createShaderModule({
		label: "EnvironmentIBLBakePrefilterModule",
		code: shaderCode,
		language: "wgsl",
		stage: "compute",
		sourceKind: "unknown",
	});

	const pipeline = computeFacade.createComputePipeline({
		label: "EnvironmentIBLBakePrefilterPipeline",
		compute: {
			module,
			entryPoint: "csMain",
		},
	});

	const sampler = computeFacade.createSampler({
		label: "EnvironmentIBLBakePrefilterSampler",
		addressModeU: AddressMode.Repeat,
		addressModeV: AddressMode.ClampToEdge,
		magFilter: FilterMode.Linear,
		minFilter: FilterMode.Linear,
		mipmapFilter: FilterMode.Linear,
	});

	const inputTexture = computeFacade.createTexture({
		width: Math.max(1, envMap.width),
		height: Math.max(1, envMap.height),
		format: TextureFormat.RGBA8Unorm,
		usage: TextureUsage.TextureBinding | TextureUsage.CopyDst,
		label: "EnvironmentIBLBakeInputTexture",
	});

	return {
		sampler,
		module,
		pipeline,
		inputTexture,
	};
}

function uploadSourceTexture(
	context: EnvironmentIBLWebGPUContext,
	inputTexture: IRenderTexture,
	envMap: Texture
): void {
	const upload = createTextureUploadData(envMap);
	const uploadData =
		upload.data.buffer instanceof ArrayBuffer ?
			new Uint8Array(
				upload.data.buffer,
				upload.data.byteOffset,
				upload.data.byteLength
			)
		:	new Uint8Array(upload.data);
	context.queue.writeTexture(
		{
			texture: getWebGPUTexture(inputTexture).texture,
			mipLevel: 0,
		},
		uploadData,
		{
			offset: 0,
			bytesPerRow: upload.bytesPerRow,
			rowsPerImage: upload.height,
		},
		{
			width: upload.width,
			height: upload.height,
			depthOrArrayLayers: 1,
		}
	);
}

async function bakeMipLevelWithWebGPU(
	context: EnvironmentIBLWebGPUContext,
	pipeline: IComputePipeline,
	bindGroup: IBindingGroup,
	outputTexture: IRenderTexture,
	width: number,
	height: number,
	level: number,
	signal?: AbortSignal | null
): Promise<Float32Array> {
	assertNotAborted(signal);

	const bytesPerPixel = 4;
	const bytesPerRow = alignTo(width * bytesPerPixel, 256);
	const readbackSize = bytesPerRow * height;
	const readbackBuffer = context.device.createBuffer({
		label: `EnvironmentIBLBakeReadback_mip${level}`,
		size: Math.max(readbackSize, 4),
		usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
	});

	try {
		const commandEncoder = context.device.createCommandEncoder({
			label: `EnvironmentIBLBakePrefilterEncoder_mip${level}`,
		});
		const computePass = commandEncoder.beginComputePass({
			label: `EnvironmentIBLBakePrefilterPass_mip${level}`,
		});
		computePass.setPipeline(getWebGPUComputePipeline(pipeline));
		computePass.setBindGroup(0, getWebGPUBindGroup(bindGroup));
		computePass.dispatchWorkgroups(
			Math.ceil(width / WORKGROUP_SIZE),
			Math.ceil(height / WORKGROUP_SIZE),
			1
		);
		computePass.end();

		commandEncoder.copyTextureToBuffer(
			{
				texture: getWebGPUTexture(outputTexture).texture,
				mipLevel: 0,
			},
			{
				buffer: readbackBuffer,
				offset: 0,
				bytesPerRow,
				rowsPerImage: height,
			},
			{
				width,
				height,
				depthOrArrayLayers: 1,
			}
		);

		context.queue.submit([commandEncoder.finish()]);
		await readbackBuffer.mapAsync(GPUMapMode.READ, 0, readbackSize);
		assertNotAborted(signal);

		const mappedRange = readbackBuffer.getMappedRange(0, readbackSize);
		const copied = new Uint8Array(mappedRange.slice(0));
		readbackBuffer.unmap();
		return unpackRGBA8Readback(copied, width, height, bytesPerRow);
	} finally {
		try {
			readbackBuffer.destroy();
		} catch {
			// ignore
		}
	}
}

function destroyMipResources(resources: EnvironmentIBLMipResources): void {
	destroyResource(resources.bindGroup);
	destroyResource(resources.paramsBuffer);
	destroyResource(resources.outputTexture);
}

function destroyWebGPUResources(resources: EnvironmentIBLWebGPUResources): void {
	destroyResource(resources.inputTexture);
	destroyResource(resources.sampler);
	destroyResource(resources.pipeline);
	destroyResource(resources.module);
}

export async function prefilterEnvMapWithWebGPU(
	envMap: Texture,
	source: WebGPUComputeFacadeSource,
	signal?: AbortSignal | null,
	onMipComplete?: (level: number) => void
): Promise<Texture> {
	assertNotAborted(signal);

	const computeFacade = resolveWebGPUComputeFacade(source);
	const context = resolveWebGPUContext(source);
	const sourceIsLinear =
		envMap.colorSpace === "HDR" || envMap.colorSpace === "Linear";
	const { baseWidth, baseHeight } = resolvePrefilterBaseDimensions(envMap);
	const resources = await createWebGPUResources(envMap, computeFacade);
	const mipmaps: EnvironmentIBLPrefilterMipData[] = [];

	try {
		uploadSourceTexture(context, resources.inputTexture, envMap);

		for (let level = 0; level < ENVIRONMENT_IBL_MAX_MIP_LEVELS; level++) {
			assertNotAborted(signal);
			const roughness =
				ENVIRONMENT_IBL_MAX_MIP_LEVELS <= 1 ?
					0
				:	level / (ENVIRONMENT_IBL_MAX_MIP_LEVELS - 1);
			const sampleCount = resolveGpuSampleCount(
				level,
				ENVIRONMENT_IBL_MAX_MIP_LEVELS
			);
			const width = Math.max(1, baseWidth >> level);
			const height = Math.max(1, baseHeight >> level);

			const outputTexture = computeFacade.createTexture({
				width,
				height,
				format: TextureFormat.RGBA8Unorm,
				usage:
					TextureUsage.StorageBinding |
					TextureUsage.CopySrc |
					TextureUsage.TextureBinding,
				label: `EnvironmentIBLBakePrefilterOutput_mip${level}`,
			});

			const paramsBuffer = computeFacade.createBuffer({
				size: PREFILTER_PARAMS_SIZE,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
				label: `EnvironmentIBLBakePrefilterParams_mip${level}`,
			});

			const params = createPrefilterParamsBuffer(
				width,
				height,
				envMap.width,
				envMap.height,
				roughness,
				sampleCount,
				sourceIsLinear
			);
			computeFacade.writeBuffer(paramsBuffer, params, 0);

			const mipResources = createPrefilterMipResources(
				resources.pipeline,
				resources.sampler,
				resources.inputTexture,
				outputTexture,
				paramsBuffer,
				computeFacade,
				level
			);

			try {
				const mipData = await bakeMipLevelWithWebGPU(
					context,
					resources.pipeline,
					mipResources.bindGroup,
					mipResources.outputTexture,
					width,
					height,
					level,
					signal
				);
				mipmaps.push({
					level,
					width,
					height,
					data: mipData,
				});
			} finally {
				destroyMipResources(mipResources);
			}

			onMipComplete?.(level);
		}
	} finally {
		destroyWebGPUResources(resources);
	}

	return buildPrefilteredTexture(baseWidth, baseHeight, mipmaps);
}
