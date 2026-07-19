import { Texture } from "../../core/Texture";
import { Platform } from "../../foundation/Platform";
import { lerp } from "../../maths/Common";
import { hammersley, importanceSampleGGX_VNDF } from "../../maths/Sampling";
import type { IVector3 } from "../../maths/types";
import { Vector3 } from "../../maths/Vector3";
import {
	AddressMode,
	BufferUsage,
	FilterMode,
	TextureFormat,
	TextureUsage,
	type IRenderBuffer,
	type IRenderTexture,
	type ISampler,
} from "../../backends/types";
import type {
	IComputeKernel,
	IComputeRuntime,
} from "../../backends/IComputeRuntime";
import type { WebGPUComputeFacadeSource } from "../../backends/webgpu/ComputeFacade";
import { ComputeRuntime } from "../../backends/webgpu/ComputeRuntime";
import {
	createTextureMipUploadLevels,
	resolveWebGPUTextureUploadFormat,
} from "../../backends/webgpu/texture";
import {
	WEBGPU_2D_COMPUTE_WORKGROUP_SIZE as WORKGROUP_SIZE,
} from "../../backends/webgpu/constants";
import { ShaderSource } from "../../shaders/ShaderSource";
import { globalWorkerScheduler } from "../../workers/WorkerScheduler";
import { postMessageWorkerTransportPlugin } from "../../workers/transports";
import type { WorkerLike } from "../../workers/types";
import type {
	IBLPrefilterWorkerEnvMapPayload,
	IBLPrefilterWorkerTaskPayload,
	IBLPrefilterWorkerTaskResult,
} from "./workers/iblPrefilterWorkerProtocol";
import {
	ensureEnvironmentTextureEquirect,
	isTextureReadyForEnvironment,
	sampleEnvironmentTextureLevelLinear,
} from "../runtime/environmentMapRuntime";

export const IBL_PREFILTER_MAX_SAMPLE_WIDTH = 128;
export const IBL_PREFILTER_MAX_SAMPLE_HEIGHT = 64;
export const IBL_PREFILTER_MAX_MIP_LEVELS = 5;

const PREFILTER_PARAMS_SIZE = 32;
const CPU_MAX_SAMPLE_COUNT = 1024;
const CPU_MIN_SAMPLE_COUNT = 64;
const GPU_MAX_SAMPLE_COUNT = 256;
const GPU_MIN_SAMPLE_COUNT = 48;
const DEFAULT_PREFILTER_POOL_PREFIX = "ibl-prefilter";
const PREFILTER_EPSILON = 1e-6;
const EQUIRECT_DISTORTION_EPSILON = 1e-4;

interface MutableRGB {
	r: number;
	g: number;
	b: number;
}

interface IBLPrefilterWebGPUResources {
	runtime: IComputeRuntime;
	sampler: ISampler;
	kernel: IComputeKernel;
	inputTexture: IRenderTexture;
	inputTextureFormat: TextureFormat;
}

interface IBLPrefilterMipResources {
	outputTexture: IRenderTexture;
	paramsBuffer: IRenderBuffer;
}

export interface IBLPrefilterMipData {
	level: number;
	width: number;
	height: number;
	data: Float32Array;
}

export type IBLPrefilterAcceleration =
	| "auto"
	| "worker"
	| "cpu"
	| "webgpu";

export interface IBLPrefilterProgress {
	phase: "prefilter";
	completed: number;
	total: number;
	detail?: string;
}

export type IBLPrefilterBackendSource = WebGPUComputeFacadeSource;

export interface IBLPrefilterOptions {
	signal?: AbortSignal | null;
	onProgress?: (progress: IBLPrefilterProgress) => void;
	acceleration?: IBLPrefilterAcceleration;
	workerCount?: number;
	backend?: IBLPrefilterBackendSource | null;
	computeSource?: WebGPUComputeFacadeSource | null;
	maxSampleWidth?: number;
	maxSampleHeight?: number;
	maxMipLevels?: number;
}

export interface IBLPrefilterConstructorOptions {
	backend?: IBLPrefilterBackendSource | null;
	computeSource?: WebGPUComputeFacadeSource | null;
}

export interface ResolvedIBLPrefilterOptions {
	maxSampleWidth: number;
	maxSampleHeight: number;
	maxMipLevels: number;
}

interface IBLPrefilterRuntimeOptions {
	signal?: AbortSignal | null;
	acceleration?: IBLPrefilterAcceleration;
	workerCount?: number;
	computeSource: WebGPUComputeFacadeSource | null;
}

function createIBLPrefilterAbortError(): Error {
	const error = new Error("IBL prefilter was aborted");
	error.name = "AbortError";
	return error;
}

function assertPrefilterNotAborted(signal?: AbortSignal | null): void {
	if (!signal?.aborted) return;
	throw createIBLPrefilterAbortError();
}

function resolveTextureIsLinear(texture: Texture): boolean {
	return texture.colorSpace === "HDR" || texture.colorSpace === "Linear";
}

function sanitizePrefilterDimension(
	value: number | undefined,
	fallback: number
): number {
	if (!Number.isFinite(value)) {
		return Math.max(1, Math.floor(fallback));
	}
	return Math.max(1, Math.floor(value as number));
}

function sanitizePrefilterMipLevelCount(
	value: number | undefined,
	fallback: number
): number {
	if (!Number.isFinite(value)) {
		return Math.max(1, Math.floor(fallback));
	}
	return Math.max(1, Math.floor(value as number));
}

export function resolveIBLPrefilterOptions(
	options: IBLPrefilterOptions = {}
): ResolvedIBLPrefilterOptions {
	return {
		maxSampleWidth: sanitizePrefilterDimension(
			options.maxSampleWidth,
			IBL_PREFILTER_MAX_SAMPLE_WIDTH
		),
		maxSampleHeight: sanitizePrefilterDimension(
			options.maxSampleHeight,
			IBL_PREFILTER_MAX_SAMPLE_HEIGHT
		),
		maxMipLevels: sanitizePrefilterMipLevelCount(
			options.maxMipLevels,
			IBL_PREFILTER_MAX_MIP_LEVELS
		),
	};
}

function resolveRoughnessFromMipLevel(level: number, maxMipLevels: number): number {
	if (maxMipLevels <= 1) {
		return 0;
	}
	return level / (maxMipLevels - 1);
}

function resolveSampleCountByRoughness(
	roughness: number,
	maxSamples: number,
	minSamples: number
): number {
	const sampleCount = Math.floor(lerp(maxSamples, minSamples, roughness));
	return Math.max(minSamples, Math.min(maxSamples, sampleCount));
}

function distributionGGX(nDotH: number, roughness: number): number {
	const alpha = Math.max(roughness * roughness, 1e-4);
	const alpha2 = alpha * alpha;
	const denom = nDotH * nDotH * (alpha2 - 1.0) + 1.0;
	return alpha2 / Math.max(Math.PI * denom * denom, PREFILTER_EPSILON);
}

function computeGGXSamplePDF(
	nDotH: number,
	vDotH: number,
	roughness: number
): number {
	if (nDotH <= 0 || vDotH <= 0) {
		return PREFILTER_EPSILON;
	}
	const d = distributionGGX(nDotH, roughness);
	return Math.max(
		(d * nDotH) / Math.max(4.0 * vDotH, PREFILTER_EPSILON),
		PREFILTER_EPSILON
	);
}

function computeEquirectTexelSolidAngle(
	sourceWidth: number,
	sourceHeight: number,
	directionY: number
): number {
	const safeWidth = Math.max(1, sourceWidth);
	const safeHeight = Math.max(1, sourceHeight);
	const sinTheta = Math.sqrt(Math.max(0, 1.0 - directionY * directionY));
	return (
		(2.0 *
			Math.PI *
			Math.PI *
			Math.max(sinTheta, EQUIRECT_DISTORTION_EPSILON)) /
		(safeWidth * safeHeight)
	);
}

function resolvePrefilterSampleMipLevel(
	envMap: Texture,
	roughness: number,
	sampleCount: number,
	pdf: number,
	directionY: number
): number {
	const mipCount = Math.max(1, envMap.mipmaps.length || 1);
	if (mipCount <= 1 || roughness <= PREFILTER_EPSILON) {
		return 0;
	}
	const texelSolidAngle = computeEquirectTexelSolidAngle(
		envMap.width,
		envMap.height,
		directionY
	);
	const sampleSolidAngle = 1.0 / Math.max(sampleCount * pdf, PREFILTER_EPSILON);
	const lod = 0.5 * Math.log2(sampleSolidAngle / texelSolidAngle);
	return Math.max(0, Math.min(mipCount - 1, lod));
}

export function resolvePrefilterBaseDimensions(
	envMap: Texture,
	options: {
		maxSampleWidth?: number;
		maxSampleHeight?: number;
	} = {}
): {
	baseWidth: number;
	baseHeight: number;
} {
	const maxSampleWidth = sanitizePrefilterDimension(
		options.maxSampleWidth,
		IBL_PREFILTER_MAX_SAMPLE_WIDTH
	);
	const maxSampleHeight = sanitizePrefilterDimension(
		options.maxSampleHeight,
		IBL_PREFILTER_MAX_SAMPLE_HEIGHT
	);
	return {
		baseWidth: Math.min(envMap.width, maxSampleWidth),
		baseHeight: Math.min(envMap.height, maxSampleHeight),
	};
}

export function prefilterEnvMapMipLevel(
	envMap: Texture,
	level: number,
	baseWidth: number,
	baseHeight: number,
	maxMipLevels: number = IBL_PREFILTER_MAX_MIP_LEVELS,
	signal?: AbortSignal | null
): IBLPrefilterMipData {
	assertPrefilterNotAborted(signal);
	const roughness = resolveRoughnessFromMipLevel(level, maxMipLevels);
	const sampleCount = resolveSampleCountByRoughness(
		roughness,
		CPU_MAX_SAMPLE_COUNT,
		CPU_MIN_SAMPLE_COUNT
	);
	const width = Math.max(1, baseWidth >> level);
	const height = Math.max(1, baseHeight >> level);
	const data = new Float32Array(width * height * 4);
	const normal: IVector3 = { x: 0, y: 0, z: 0 };
	const radiance: MutableRGB = { r: 0, g: 0, b: 0 };

	for (let j = 0; j < height; j++) {
		assertPrefilterNotAborted(signal);
		const theta = ((j + 0.5) / height) * Math.PI;
		for (let i = 0; i < width; i++) {
			const phi = ((i + 0.5) / width) * 2 * Math.PI;
			normal.x = Math.sin(theta) * Math.sin(phi);
			normal.y = Math.cos(theta);
			normal.z = Math.sin(theta) * Math.cos(phi);

			prefilterSpecular(
				envMap,
				normal,
				roughness,
				sampleCount,
				radiance
			);
			const idx = (j * width + i) * 4;
			data[idx] = radiance.r;
			data[idx + 1] = radiance.g;
			data[idx + 2] = radiance.b;
			data[idx + 3] = 1;
		}
	}

	return {
		level,
		width,
		height,
		data,
	};
}

export function buildPrefilteredTexture(
	baseWidth: number,
	baseHeight: number,
	mipData: IBLPrefilterMipData[]
): Texture {
	const sorted = [...mipData].sort((left, right) => left.level - right.level);
	return new Texture({
		data: sorted[0]?.data ?? null,
		width: baseWidth,
		height: baseHeight,
		format: TextureFormat.RGBA16Float,
		colorSpace: "HDR",
		levels: sorted.map((mip) => ({
			data: mip.data,
			width: mip.width,
			height: mip.height,
		})),
		usageHint: "color",
	});
}

export function prefilterEnvMapCPU(
	envMap: Texture,
	prefilterOptions: ResolvedIBLPrefilterOptions,
	signal?: AbortSignal | null,
	onMipComplete?: (level: number, total: number) => void
): Texture {
	const { baseWidth, baseHeight } = resolvePrefilterBaseDimensions(envMap, {
		maxSampleWidth: prefilterOptions.maxSampleWidth,
		maxSampleHeight: prefilterOptions.maxSampleHeight,
	});
	const totalMipLevels = Math.max(1, prefilterOptions.maxMipLevels);
	const mipmaps: IBLPrefilterMipData[] = [];
	for (let level = 0; level < totalMipLevels; level++) {
		const mip = prefilterEnvMapMipLevel(
			envMap,
			level,
			baseWidth,
			baseHeight,
			totalMipLevels,
			signal
		);
		mipmaps.push(mip);
		onMipComplete?.(level, totalMipLevels);
	}
	return buildPrefilteredTexture(baseWidth, baseHeight, mipmaps);
}

function prefilterSpecular(
	envMap: Texture,
	normal: IVector3,
	roughness: number,
	sampleCount: number,
	outColor: MutableRGB
): void {
	let totalWeight = 0;
	outColor.r = 0;
	outColor.g = 0;
	outColor.b = 0;

	for (let i = 0; i < sampleCount; i++) {
		const xi = hammersley(i, sampleCount);
		const view = normal;
		const half = importanceSampleGGX_VNDF(xi, view, normal, roughness);
		const nDotH = Math.max(Vector3.dot(normal, half), 0);
		const vDotH = Math.max(Vector3.dot(view, half), 0);
		if (vDotH <= 0) continue;
		const lightDir = Vector3.normalize({
			x: 2.0 * nDotH * half.x - view.x,
			y: 2.0 * nDotH * half.y - view.y,
			z: 2.0 * nDotH * half.z - view.z,
		});

		const nDotL = Math.max(Vector3.dot(normal, lightDir), 0);
		if (nDotL <= 0) continue;

		const pdf = computeGGXSamplePDF(nDotH, vDotH, roughness);
		const sampleMipLevel = resolvePrefilterSampleMipLevel(
			envMap,
			roughness,
			sampleCount,
			pdf,
			lightDir.y
		);
		const sample = sampleEnvironmentTextureLevelLinear(
			envMap,
			lightDir,
			sampleMipLevel
		);

		outColor.r += sample.r * nDotL;
		outColor.g += sample.g * nDotL;
		outColor.b += sample.b * nDotL;
		totalWeight += nDotL;
	}

	if (totalWeight <= 0) {
		return;
	}

	outColor.r /= totalWeight;
	outColor.g /= totalWeight;
	outColor.b /= totalWeight;
}

function createPrefilterParamsBuffer(
	width: number,
	height: number,
	sourceWidth: number,
	sourceHeight: number,
	roughness: number,
	sampleCount: number,
	sourceIsLinear: boolean,
	sourceMipLevelCount: number
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

function createPrefilterMipResources(
	outputTexture: IRenderTexture,
	paramsBuffer: IRenderBuffer
): IBLPrefilterMipResources {
	return {
		outputTexture,
		paramsBuffer,
	};
}

async function createWebGPUResources(
	envMap: Texture,
	source: WebGPUComputeFacadeSource
): Promise<IBLPrefilterWebGPUResources> {
	const runtime = new ComputeRuntime(source);
	try {
		const shaderCode = await ShaderSource.load(
			"webgpu.iblPrefilter.raw"
		);
		const kernel = await runtime.createKernel({
			label: "IBLPrefilter",
			code: shaderCode,
			language: "wgsl",
			sourceKind: "unknown",
			bindings: [
				{
					key: "envSampler",
					binding: 0,
					type: "sampler",
				},
				{
					key: "envTexture",
					binding: 1,
					type: "texture",
				},
				{
					key: "outputTexture",
					binding: 2,
					type: "texture",
				},
				{
					key: "params",
					binding: 3,
					type: "buffer",
				},
			],
			workgroupSize: {
				x: WORKGROUP_SIZE,
				y: WORKGROUP_SIZE,
				z: 1,
			},
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

		return {
			runtime,
			sampler,
			kernel,
			inputTexture,
			inputTextureFormat,
		};
	} catch (error) {
		runtime.destroy();
		throw error;
	}
}

function uploadSourceTexture(
	runtime: IComputeRuntime,
	inputTexture: IRenderTexture,
	envMap: Texture,
	format: TextureFormat
): void {
	const uploads = createTextureMipUploadLevels(envMap, format);
	for (const upload of uploads) {
		const uploadData =
			upload.data.buffer instanceof ArrayBuffer ?
				new Uint8Array(
					upload.data.buffer,
					upload.data.byteOffset,
					upload.data.byteLength
				)
			: 	new Uint8Array(upload.data);
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
			}
		);
	}
}

async function prefilterMipLevelWithWebGPU(
	runtime: IComputeRuntime,
	kernel: IComputeKernel,
	sampler: ISampler,
	inputTexture: IRenderTexture,
	paramsBuffer: IRenderBuffer,
	outputTexture: IRenderTexture,
	width: number,
	height: number,
	signal?: AbortSignal | null
): Promise<Float32Array> {
	assertPrefilterNotAborted(signal);
	const ticket = kernel.dispatch({
		label: "IBLPrefilterDispatch",
		resources: {
			envSampler: sampler,
			envTexture: inputTexture,
			outputTexture: outputTexture,
			params: paramsBuffer,
		},
		dispatch2D: {
			width,
			height,
			depth: 1,
		},
	});
	await ticket.done;
	assertPrefilterNotAborted(signal);

	const readback = await runtime.readTexture({
		texture: outputTexture,
		width,
		height,
		format: TextureFormat.RGBA16Float,
	});
	assertPrefilterNotAborted(signal);
	const result = readback.toRGBAFloat32();
	for (let i = 3; i < result.length; i += 4) {
		result[i] = 1;
	}
	return result;
}

function destroyMipResources(resources: IBLPrefilterMipResources): void {
	resources.paramsBuffer.destroy();
	resources.outputTexture.destroy();
}

function destroyWebGPUResources(resources: IBLPrefilterWebGPUResources): void {
	resources.inputTexture.destroy();
	const destroySampler = (
		resources.sampler as { destroy?: () => void } | null
	)?.destroy;
	if (typeof destroySampler === "function") {
		try {
			destroySampler.call(resources.sampler);
		} catch (error) {
			const detail =
				error instanceof Error ? error.message : String(error);
			throw new Error(`Failed to destroy IBL prefilter sampler: ${detail}`);
		}
	}
	resources.kernel.destroy();
	resources.runtime.destroy();
}

export async function prefilterEnvMapWithWebGPU(
	envMap: Texture,
	source: WebGPUComputeFacadeSource,
	prefilterOptions: ResolvedIBLPrefilterOptions,
	signal?: AbortSignal | null,
	onMipComplete?: (level: number) => void
): Promise<Texture> {
	assertPrefilterNotAborted(signal);

	const sourceIsLinear = resolveTextureIsLinear(envMap);
	const { baseWidth, baseHeight } = resolvePrefilterBaseDimensions(envMap, {
		maxSampleWidth: prefilterOptions.maxSampleWidth,
		maxSampleHeight: prefilterOptions.maxSampleHeight,
	});
	const resources = await createWebGPUResources(envMap, source);
	const mipmaps: IBLPrefilterMipData[] = [];
	const totalMipLevels = Math.max(1, prefilterOptions.maxMipLevels);

	try {
		uploadSourceTexture(
			resources.runtime,
			resources.inputTexture,
			envMap,
			resources.inputTextureFormat
		);

		for (let level = 0; level < totalMipLevels; level++) {
			assertPrefilterNotAborted(signal);
			const roughness = resolveRoughnessFromMipLevel(level, totalMipLevels);
			const sampleCount = resolveSampleCountByRoughness(
				roughness,
				GPU_MAX_SAMPLE_COUNT,
				GPU_MIN_SAMPLE_COUNT
			);
			const width = Math.max(1, baseWidth >> level);
			const height = Math.max(1, baseHeight >> level);

			const outputTexture = resources.runtime.createTexture({
				width,
				height,
				format: TextureFormat.RGBA16Float,
				usage:
					TextureUsage.StorageBinding |
					TextureUsage.CopySrc |
					TextureUsage.TextureBinding,
				label: `IBLPrefilterOutput_mip${level}`,
			});

			const paramsBuffer = resources.runtime.createBuffer({
				size: PREFILTER_PARAMS_SIZE,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
				label: `IBLPrefilterParams_mip${level}`,
			});

			const params = createPrefilterParamsBuffer(
				width,
				height,
				envMap.width,
				envMap.height,
				roughness,
				sampleCount,
				sourceIsLinear,
				Math.max(1, envMap.mipmaps.length || 1)
			);
			resources.runtime.writeBuffer(paramsBuffer, params, 0);

			const mipResources = createPrefilterMipResources(
				outputTexture,
				paramsBuffer
			);

			try {
				const mipData = await prefilterMipLevelWithWebGPU(
					resources.runtime,
					resources.kernel,
					resources.sampler,
					resources.inputTexture,
					mipResources.paramsBuffer,
					mipResources.outputTexture,
					width,
					height,
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

function resolveWorkerCount(requestedCount?: number): number {
	const fallback = Platform.getHardwareConcurrency(4);
	if (!Number.isFinite(requestedCount)) {
		return Math.max(1, fallback);
	}
	return Math.max(1, Math.floor(requestedCount as number));
}

function createPrefilterWorker(workerIndex: number, poolId: string): WorkerLike {
	if (typeof Worker !== "function") {
		throw new Error(
			`Worker constructor is unavailable for pool "${poolId}" (worker #${workerIndex})`
		);
	}

	return new Worker(
		new URL("./workers/iblPrefilter.worker.ts", import.meta.url),
		{
			type: "module",
		}
	) as unknown as WorkerLike;
}

function resolveWorkerPoolId(): string {
	return `${DEFAULT_PREFILTER_POOL_PREFIX}-${Math.random().toString(36).slice(2)}`;
}

function toWorkerEnvMapPayload(envMap: Texture): IBLPrefilterWorkerEnvMapPayload {
	return {
		width: envMap.width,
		height: envMap.height,
		colorSpace: envMap.colorSpace,
		data: envMap.data,
	};
}

async function prefilterEnvMapWithWorkers(
	envMap: Texture,
	options: IBLPrefilterRuntimeOptions,
	prefilterOptions: ResolvedIBLPrefilterOptions,
	onMipComplete: (level: number) => void
): Promise<Texture> {
	const poolId = resolveWorkerPoolId();
	const totalMipLevels = Math.max(1, prefilterOptions.maxMipLevels);
	const workerCount = Math.min(
		resolveWorkerCount(options.workerCount),
		totalMipLevels
	);
	const { baseWidth, baseHeight } = resolvePrefilterBaseDimensions(envMap, {
		maxSampleWidth: prefilterOptions.maxSampleWidth,
		maxSampleHeight: prefilterOptions.maxSampleHeight,
	});
	const envPayload = toWorkerEnvMapPayload(envMap);

	globalWorkerScheduler.registerPool({
		id: poolId,
		size: workerCount,
		createWorker: (workerIndex, id) => createPrefilterWorker(workerIndex, id),
		transportPlugins: [postMessageWorkerTransportPlugin],
		defaultTimeoutMs: 0,
	});

	try {
		const tasks: Promise<IBLPrefilterMipData>[] = [];
		for (let level = 0; level < totalMipLevels; level++) {
			assertPrefilterNotAborted(options.signal);
			const payload: IBLPrefilterWorkerTaskPayload = {
				type: "prefilter-mip",
				envMap: envPayload,
				baseWidth,
				baseHeight,
				maxMipLevels: totalMipLevels,
				level,
			};

			const task = globalWorkerScheduler
				.schedule<
					IBLPrefilterWorkerTaskResult,
					IBLPrefilterWorkerTaskPayload
				>(
					poolId,
					payload,
					{
						signal: options.signal ?? null,
					}
				)
				.then((result) => {
					if (!result || result.type !== "prefilter-mip") {
						throw new Error(
							"IBL prefilter worker returned an invalid response"
						);
					}
					onMipComplete(result.level);
					return {
						level: result.level,
						width: result.width,
						height: result.height,
						data: result.data,
					};
				});

			tasks.push(task);
		}

		const mipData = await Promise.all(tasks);
		return buildPrefilteredTexture(baseWidth, baseHeight, mipData);
	} finally {
		globalWorkerScheduler.unregisterPool(poolId);
	}
}

function prefilterEnvMapOnCPU(
	envMap: Texture,
	options: IBLPrefilterRuntimeOptions,
	prefilterOptions: ResolvedIBLPrefilterOptions,
	onMipComplete: (level: number) => void
): Texture {
	return prefilterEnvMapCPU(
		envMap,
		prefilterOptions,
		options.signal ?? null,
		(level) => {
			onMipComplete(level);
		}
	);
}

function canUseWorkerAcceleration(options: IBLPrefilterRuntimeOptions): boolean {
	return (
		options.acceleration === "worker" ||
		(options.acceleration !== "cpu" &&
			options.acceleration !== "webgpu" &&
			Platform.hasWorker())
	);
}

function canUseWebGPUAcceleration(options: IBLPrefilterRuntimeOptions): boolean {
	if (options.acceleration === "webgpu") {
		return true;
	}
	if (options.acceleration === "cpu" || options.acceleration === "worker") {
		return false;
	}
	return !!options.computeSource;
}

async function prefilterEnvMapOnWebGPU(
	envMap: Texture,
	options: IBLPrefilterRuntimeOptions,
	prefilterOptions: ResolvedIBLPrefilterOptions,
	onMipComplete: (level: number) => void
): Promise<Texture> {
	if (!options.computeSource) {
		throw new Error(
			"WebGPU acceleration was requested for IBL prefiltering, but no WebGPU backend or compute source was provided."
		);
	}
	return prefilterEnvMapWithWebGPU(
		envMap,
		options.computeSource,
		prefilterOptions,
		options.signal ?? null,
		onMipComplete
	);
}

async function prefilterEnvMap(
	envMap: Texture,
	options: IBLPrefilterRuntimeOptions,
	prefilterOptions: ResolvedIBLPrefilterOptions,
	onMipComplete: (level: number) => void
): Promise<Texture> {
	if (canUseWebGPUAcceleration(options)) {
		try {
			return await prefilterEnvMapOnWebGPU(
				envMap,
				options,
				prefilterOptions,
				onMipComplete
			);
		} catch (error) {
			if (options.acceleration === "webgpu") {
				throw error;
			}
		}
	}

	if (!canUseWorkerAcceleration(options)) {
		if (options.acceleration === "worker") {
			throw new Error(
				"Worker acceleration was requested for IBL prefiltering, but Worker API is unavailable."
			);
		}
		return prefilterEnvMapOnCPU(
			envMap,
			options,
			prefilterOptions,
			onMipComplete
		);
	}

	try {
		return await prefilterEnvMapWithWorkers(
			envMap,
			options,
			prefilterOptions,
			onMipComplete
		);
	} catch (error) {
		if (options.acceleration === "worker") {
			throw error;
		}
		return prefilterEnvMapOnCPU(
			envMap,
			options,
			prefilterOptions,
			onMipComplete
		);
	}
}

function isConstructorOptions(
	source: IBLPrefilterBackendSource | IBLPrefilterConstructorOptions | null
): source is IBLPrefilterConstructorOptions {
	if (!source || typeof source !== "object") {
		return false;
	}
	return "backend" in source || "computeSource" in source;
}

function isPotentialWebGPUComputeSource(
	source: unknown
): source is WebGPUComputeFacadeSource {
	if (!source || typeof source !== "object") {
		return false;
	}
	const type = (source as { type?: unknown }).type;
	return typeof type === "string" ? type === "webgpu" : true;
}

function resolveComputeSource(
	source: WebGPUComputeFacadeSource | IBLPrefilterBackendSource | null
): WebGPUComputeFacadeSource | null {
	if (!isPotentialWebGPUComputeSource(source)) {
		return null;
	}
	return source;
}

/**
 * Prefilters equirectangular environment maps into specular IBL mip chains.
 *
 * @remarks The class may run independently from `Renderer`. Passing a WebGPU
 * backend or compute facade enables GPU acceleration; all
 * other sources use worker/CPU fallback according to `acceleration`.
 */
export class IBLPrefilter {
	private readonly _backend: IBLPrefilterBackendSource | null;
	private readonly _computeSource: WebGPUComputeFacadeSource | null;

	/**
	 * Creates a standalone environment IBL prefilter service.
	 *
	 * @param source Optional WebGPU backend, WebGPU compute source, or
	 * constructor options. Passing a WebGPU-capable source enables GPU
	 * acceleration when requested or selected by `auto`.
	 * @constraints The source must outlive calls to `prefilter()`.
	 * @sideEffects None.
	 */
	public constructor(
		source: IBLPrefilterBackendSource | IBLPrefilterConstructorOptions | null =
			null
	) {
		if (isConstructorOptions(source)) {
			this._backend = source.backend ?? null;
			this._computeSource = source.computeSource ?? null;
		} else {
			this._backend = source;
			this._computeSource = null;
		}
	}

	/**
	 * Generates a prefiltered HDR environment texture.
	 *
	 * @param envMap Source 2D equirectangular or cubemap environment texture.
	 * @param options Acceleration, sizing, cancellation, and progress options.
	 * @returns Prefiltered HDR `Texture` with roughness mip levels.
	 * @constraints Source texture must contain ready pixel data.
	 * @sideEffects May allocate transient worker or WebGPU resources and destroy
	 * them before resolving.
	 */
	public async prefilter(
		envMap: Texture,
		options: IBLPrefilterOptions = {}
	): Promise<Texture> {
		assertPrefilterNotAborted(options.signal);
		const sampledEnvironment = ensureEnvironmentTextureEquirect(envMap);
		if (
			!sampledEnvironment ||
			!isTextureReadyForEnvironment(sampledEnvironment)
		) {
			throw new Error(
				"IBL prefilter requires a valid environment texture (2D equirect or cubemap)."
			);
		}

		const prefilterOptions = resolveIBLPrefilterOptions(options);
		const totalMipLevels = prefilterOptions.maxMipLevels;
		let completed = 0;
		const computeSource = resolveComputeSource(
			options.computeSource ??
				this._computeSource ??
				options.backend ??
				this._backend ??
				null
		);

		return prefilterEnvMap(
			sampledEnvironment,
			{
				signal: options.signal ?? null,
				acceleration: options.acceleration ?? "auto",
				workerCount: options.workerCount,
				computeSource,
			},
			prefilterOptions,
			(level) => {
				completed++;
				options.onProgress?.({
					phase: "prefilter",
					completed,
					total: totalMipLevels,
					detail: `mip ${level + 1}/${totalMipLevels}`,
				});
			}
		);
	}
}

/**
 * Prefilters an environment texture with a one-shot `IBLPrefilter` instance.
 *
 * @param envMap Source 2D equirectangular or cubemap environment texture.
 * @param options Acceleration, backend, sizing, cancellation, and progress
 * options.
 * @returns Prefiltered HDR `Texture` with roughness mip levels.
 * @constraints Source texture must contain ready pixel data.
 * @sideEffects May allocate transient worker or WebGPU resources and destroy
 * them before resolving.
 */
export async function prefilterEnvironmentIBL(
	envMap: Texture,
	options: IBLPrefilterOptions = {}
): Promise<Texture> {
	const prefilter = new IBLPrefilter({
		backend: options.backend ?? null,
		computeSource: options.computeSource ?? null,
	});
	return prefilter.prefilter(envMap, options);
}
