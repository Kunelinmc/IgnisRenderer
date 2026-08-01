import type { Texture } from "../../core/Texture";
import {
	AddressMode,
	BufferUsage,
	FilterMode,
	TextureFormat,
	TextureUsage,
	type IRenderBuffer,
	type IRenderTexture,
	type ISampler,
} from "../types";
import type { IComputeKernel, IComputeRuntime } from "../IComputeRuntime";
import {
	assertIBLPrefilterNotAborted,
	assertIBLPrefilterSourceRevision,
	type IBLPrefilterExecutionRequest,
	type IBLPrefilterExecutorAvailability,
	type IBLPrefilterExecutorLike,
	type IBLPrefilterMipData,
} from "../../lights/ibl/IBLPrefilterExecutor";
import { ShaderSource } from "../../shaders/ShaderSource";

import type { IWebGPUComputeFacade } from "./ComputeFacade";
import { ComputeRuntime } from "./ComputeRuntime";
import {
	createTextureMipUploadLevels,
	resolveWebGPUTextureUploadFormat,
} from "./texture";
import { WEBGPU_2D_COMPUTE_WORKGROUP_SIZE as WORKGROUP_SIZE } from "./constants";

const PREFILTER_PARAMS_SIZE = 32;
const GPU_MAX_SAMPLE_COUNT = 256;
const GPU_MIN_SAMPLE_COUNT = 48;

interface IBLPrefilterWebGPUResources {
	runtime: IComputeRuntime;
	sampler: ISampler;
	kernel: IComputeKernel;
	inputTexture: IRenderTexture;
	inputTextureFormat: TextureFormat;
}

/** @internal Backend-owned WebGPU compute executor for IBL prefiltering. */
export class WebGPUIBLPrefilterExecutor implements IBLPrefilterExecutorLike {
	public readonly id = "webgpu" as const;
	private readonly _computeFacade: IWebGPUComputeFacade;

	public constructor(computeFacade: IWebGPUComputeFacade) {
		this._computeFacade = computeFacade;
	}

	public getAvailability(): IBLPrefilterExecutorAvailability {
		const ready = !!this._computeFacade.device && !!this._computeFacade.queue;
		return {
			state: ready ? "ready" : "temporarily-unavailable",
			acceptsRequests: ready,
			reason: ready ? null :
				"WebGPU IBL prefilter executor requires an initialized device and queue.",
		};
	}

	public async execute(
		request: IBLPrefilterExecutionRequest,
	): Promise<IBLPrefilterMipData[]> {
		const availability = this.getAvailability();
		if (!availability.acceptsRequests) {
			throw new Error(
				availability.reason ?? "WebGPU IBL prefilter executor is unavailable.",
			);
		}
		assertIBLPrefilterSourceRevision(
			request.envMap,
			request.sourceRevision,
		);
		assertIBLPrefilterNotAborted(request.signal);
		const resources = await createWebGPUResources(
			request.envMap,
			this._computeFacade,
		);
		const mipmaps: IBLPrefilterMipData[] = [];
		try {
			uploadSourceTexture(
				resources.runtime,
				resources.inputTexture,
				request.envMap,
				resources.inputTextureFormat,
			);
			for (const mipPlan of request.plan.mipLevels) {
				assertIBLPrefilterNotAborted(request.signal);
				const outputTexture = resources.runtime.createTexture({
					width: mipPlan.width,
					height: mipPlan.height,
					format: TextureFormat.RGBA16Float,
					usage:
						TextureUsage.StorageBinding |
						TextureUsage.CopySrc |
						TextureUsage.TextureBinding,
					label: `IBLPrefilterOutput_mip${mipPlan.level}`,
				});
				const paramsBuffer = resources.runtime.createBuffer({
					size: PREFILTER_PARAMS_SIZE,
					usage: BufferUsage.Uniform | BufferUsage.CopyDst,
					label: `IBLPrefilterParams_mip${mipPlan.level}`,
				});
				try {
					resources.runtime.writeBuffer(
						paramsBuffer,
						createPrefilterParamsBuffer(
							mipPlan.width,
							mipPlan.height,
							request.envMap.width,
							request.envMap.height,
							mipPlan.roughness,
							resolveSampleCountByRoughness(mipPlan.roughness),
							resolveTextureIsLinear(request.envMap),
							Math.max(1, request.envMap.mipmaps.length || 1),
						),
						0,
					);
					const data = await prefilterMipLevel(
						resources,
						paramsBuffer,
						outputTexture,
						mipPlan.width,
						mipPlan.height,
						request.signal,
					);
					mipmaps.push({ ...mipPlan, data });
				} finally {
					paramsBuffer.destroy();
					outputTexture.destroy();
				}
				request.onMipComplete?.(mipPlan.level);
			}
			return mipmaps;
		} finally {
			destroyWebGPUResources(resources);
		}
	}
}

async function createWebGPUResources(
	envMap: Texture,
	computeFacade: IWebGPUComputeFacade,
): Promise<IBLPrefilterWebGPUResources> {
	const runtime = new ComputeRuntime(computeFacade);
	try {
		const kernel = await runtime.createKernel({
			label: "IBLPrefilter",
			code: await ShaderSource.load("webgpu.iblPrefilter.raw"),
			language: "wgsl",
			sourceKind: "unknown",
			bindings: [
				{ key: "envSampler", binding: 0, type: "sampler" },
				{ key: "envTexture", binding: 1, type: "texture" },
				{ key: "outputTexture", binding: 2, type: "texture" },
				{ key: "params", binding: 3, type: "buffer" },
			],
			workgroupSize: { x: WORKGROUP_SIZE, y: WORKGROUP_SIZE, z: 1 },
		});
		const sampler = runtime.createSampler({
			label: "IBLPrefilterSampler",
			addressModeU: AddressMode.Repeat,
			addressModeV: AddressMode.ClampToEdge,
			magFilter: FilterMode.Linear,
			minFilter: FilterMode.Linear,
			mipmapFilter: FilterMode.Linear,
		});
		const inputTextureFormat = resolveWebGPUTextureUploadFormat(envMap);
		const inputTexture = runtime.createTexture({
			width: Math.max(1, envMap.width),
			height: Math.max(1, envMap.height),
			format: inputTextureFormat,
			usage: TextureUsage.TextureBinding | TextureUsage.CopyDst,
			mipLevelCount: Math.max(1, envMap.mipmaps.length || 1),
			label: "IBLPrefilterInputTexture",
		});
		return { runtime, sampler, kernel, inputTexture, inputTextureFormat };
	} catch (error) {
		runtime.destroy();
		throw error;
	}
}

function uploadSourceTexture(
	runtime: IComputeRuntime,
	inputTexture: IRenderTexture,
	envMap: Texture,
	format: TextureFormat,
): void {
	for (const upload of createTextureMipUploadLevels(envMap, format)) {
		const uploadData = upload.data.buffer instanceof ArrayBuffer ?
			new Uint8Array(
				upload.data.buffer,
				upload.data.byteOffset,
				upload.data.byteLength,
			) :
			new Uint8Array(upload.data);
		runtime.writeTexture(
			inputTexture,
			uploadData,
			{
				offset: 0,
				bytesPerRow: upload.bytesPerRow,
				rowsPerImage: upload.height,
				mipLevel: upload.mipLevel,
			},
			{
				width: upload.width,
				height: upload.height,
				depthOrArrayLayers: 1,
			},
		);
	}
}

async function prefilterMipLevel(
	resources: IBLPrefilterWebGPUResources,
	paramsBuffer: IRenderBuffer,
	outputTexture: IRenderTexture,
	width: number,
	height: number,
	signal?: AbortSignal | null,
): Promise<Float32Array> {
	assertIBLPrefilterNotAborted(signal);
	const ticket = resources.kernel.dispatch({
		label: "IBLPrefilterDispatch",
		resources: {
			envSampler: resources.sampler,
			envTexture: resources.inputTexture,
			outputTexture,
			params: paramsBuffer,
		},
		dispatch2D: { width, height, depth: 1 },
	});
	await ticket.done;
	assertIBLPrefilterNotAborted(signal);
	const readback = await resources.runtime.readTexture({
		texture: outputTexture,
		width,
		height,
		format: TextureFormat.RGBA16Float,
	});
	assertIBLPrefilterNotAborted(signal);
	const result = readback.toRGBAFloat32();
	for (let index = 3; index < result.length; index += 4) result[index] = 1;
	return result;
}

function createPrefilterParamsBuffer(
	width: number,
	height: number,
	sourceWidth: number,
	sourceHeight: number,
	roughness: number,
	sampleCount: number,
	sourceIsLinear: boolean,
	sourceMipLevelCount: number,
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
	view.setUint32(28, Math.max(1, Math.floor(sourceMipLevelCount)), true);
	return buffer;
}

function resolveSampleCountByRoughness(roughness: number): number {
	const sampleCount = Math.floor(
		GPU_MAX_SAMPLE_COUNT +
			(GPU_MIN_SAMPLE_COUNT - GPU_MAX_SAMPLE_COUNT) * roughness,
	);
	return Math.max(
		GPU_MIN_SAMPLE_COUNT,
		Math.min(GPU_MAX_SAMPLE_COUNT, sampleCount),
	);
}

function resolveTextureIsLinear(texture: Texture): boolean {
	return texture.colorSpace === "HDR" || texture.colorSpace === "Linear";
}

function destroyWebGPUResources(
	resources: IBLPrefilterWebGPUResources,
): void {
	resources.inputTexture.destroy();
	const destroySampler = (
		resources.sampler as { destroy?: () => void } | null
	)?.destroy;
	if (typeof destroySampler === "function") {
		destroySampler.call(resources.sampler);
	}
	resources.kernel.destroy();
	resources.runtime.destroy();
}
