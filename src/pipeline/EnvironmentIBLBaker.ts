import { Texture } from "../core/Texture";
import { Platform } from "../foundation/Platform";
import { lerp, sRGBToLinear } from "../maths/Common";
import { hammersley, importanceSampleGGX_VNDF } from "../maths/Sampling";
import { SH } from "../maths/SH";
import type { IVector3, SHCoefficients } from "../maths/types";
import { Vector3 } from "../maths/Vector3";
import {
	AddressMode,
	BufferUsage,
	FilterMode,
	TextureFormat,
	TextureUsage,
	type IRenderBuffer,
	type IRenderTexture,
	type ISampler,
} from "../renderers/types";
import type {
	IComputeKernel,
	IComputeRuntime,
} from "../renderers/IComputeRuntime";
import type { WebGPUComputeFacadeSource } from "../renderers/webgpu/ComputeFacade";
import { ComputeRuntime } from "../renderers/webgpu/ComputeRuntime";
import { destroyResource } from "../renderers/webgpu/computeUtils";
import { createTextureUploadData } from "../renderers/webgpu/texture";
import { loadEnvironmentIBLPrefilterShaderSource } from "../shaders/webgpu/environmentIblPrefilterShaderSource";
import { globalWorkerScheduler } from "../workers/WorkerScheduler";
import { postMessageWorkerTransportPlugin } from "../workers/transports";
import type { WorkerLike } from "../workers/types";
import type {
	EnvironmentIBLBakeWorkerEnvMapPayload,
	EnvironmentIBLBakeWorkerTaskPayload,
	EnvironmentIBLBakeWorkerTaskResult,
} from "./workers/environmentIblBakeWorkerProtocol";

export const ENVIRONMENT_IBL_MAX_SAMPLE_WIDTH = 128;
export const ENVIRONMENT_IBL_MAX_SAMPLE_HEIGHT = 64;
export const ENVIRONMENT_IBL_MAX_MIP_LEVELS = 5;

const WORKGROUP_SIZE = 8;
const PREFILTER_PARAMS_SIZE = 32;
const CPU_MAX_SAMPLE_COUNT = 1024;
const CPU_MIN_SAMPLE_COUNT = 64;
const GPU_MAX_SAMPLE_COUNT = 256;
const GPU_MIN_SAMPLE_COUNT = 48;
const DEFAULT_BAKE_POOL_PREFIX = "environment-ibl-bake";

const SRGB_TO_LINEAR_LUT = createSRGBToLinearLUT();

interface MutableRGB {
	r: number;
	g: number;
	b: number;
}

interface EnvironmentIBLWebGPUResources {
	runtime: IComputeRuntime;
	sampler: ISampler;
	kernel: IComputeKernel;
	inputTexture: IRenderTexture;
}

interface EnvironmentIBLMipResources {
	outputTexture: IRenderTexture;
	paramsBuffer: IRenderBuffer;
}

export interface EnvironmentIBLPrefilterMipData {
	level: number;
	width: number;
	height: number;
	data: Float32Array;
}

export type EnvironmentIBLBakeAcceleration =
	| "auto"
	| "worker"
	| "cpu"
	| "webgpu";

export interface EnvironmentIBLBakeProgress {
	phase: "project-sh" | "prefilter" | "finalize";
	completed: number;
	total: number;
	detail?: string;
}

export interface EnvironmentIBLBakeOptions {
	signal?: AbortSignal | null;
	onProgress?: (progress: EnvironmentIBLBakeProgress) => void;
	acceleration?: EnvironmentIBLBakeAcceleration;
	workerCount?: number;
	webgpuSource?: WebGPUComputeFacadeSource | null;
}

export interface BakedEnvironmentIBL {
	sh: SHCoefficients;
	prefilteredMap: Texture;
}

function createEnvironmentIBLBakeAbortError(): Error {
	const error = new Error("Environment IBL bake was aborted");
	error.name = "AbortError";
	return error;
}

function assertBakeNotAborted(signal?: AbortSignal | null): void {
	if (!signal?.aborted) return;
	throw createEnvironmentIBLBakeAbortError();
}

function resolveTextureIsLinear(texture: Texture): boolean {
	return texture.colorSpace === "HDR" || texture.colorSpace === "Linear";
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

function createSRGBToLinearLUT(): Float32Array {
	const lut = new Float32Array(256);
	for (let i = 0; i < lut.length; i++) {
		lut[i] = sRGBToLinear(i / 255);
	}
	return lut;
}

function decodeSRGBToLinear01(value255: number): number {
	if (value255 >= 0 && value255 <= 255 && Number.isInteger(value255)) {
		return SRGB_TO_LINEAR_LUT[value255];
	}
	return sRGBToLinear(value255 / 255);
}

export function projectEquirectTextureToSH(
	envMap: Texture,
	signal?: AbortSignal | null
): SHCoefficients {
	if (!envMap || !envMap.data) {
		return SH.empty();
	}

	const { width, height, data } = envMap;
	const sh = SH.empty();
	const sourceIsLinear = resolveTextureIsLinear(envMap);

	const sampleWidth = Math.min(width, ENVIRONMENT_IBL_MAX_SAMPLE_WIDTH);
	const sampleHeight = Math.min(height, ENVIRONMENT_IBL_MAX_SAMPLE_HEIGHT);

	const stepX = width / sampleWidth;
	const stepY = height / sampleHeight;

	const dTheta = Math.PI / sampleHeight;
	const dPhi = (2 * Math.PI) / sampleWidth;

	let totalWeight = 0;
	for (let sj = 0; sj < sampleHeight; sj++) {
		assertBakeNotAborted(signal);
		const theta = (sj + 0.5) * dTheta;
		const sinTheta = Math.sin(theta);
		const cosTheta = Math.cos(theta);
		const weight = sinTheta * dTheta * dPhi;
		const j = Math.floor((sj + 0.5) * stepY);

		for (let si = 0; si < sampleWidth; si++) {
			const phi = (si + 0.5) * dPhi;
			const x = sinTheta * Math.sin(phi);
			const y = cosTheta;
			const z = sinTheta * Math.cos(phi);
			const basis = SH.evalBasis({ x, y, z });
			const i = Math.floor((si + 0.5) * stepX);
			const idx = (j * width + i) * 4;

			const r =
				sourceIsLinear ?
					data[idx] * 255
				: 	decodeSRGBToLinear01(data[idx]) * 255;
			const g =
				sourceIsLinear ?
					data[idx + 1] * 255
				: 	decodeSRGBToLinear01(data[idx + 1]) * 255;
			const b =
				sourceIsLinear ?
					data[idx + 2] * 255
				: 	decodeSRGBToLinear01(data[idx + 2]) * 255;

			for (let k = 0; k < sh.length; k++) {
				const basisWeight = basis[k] * weight;
				sh[k].r += r * basisWeight;
				sh[k].g += g * basisWeight;
				sh[k].b += b * basisWeight;
			}

			totalWeight += weight;
		}
	}

	const normFactor = (4 * Math.PI) / totalWeight;
	for (let k = 0; k < sh.length; k++) {
		sh[k].r *= normFactor;
		sh[k].g *= normFactor;
		sh[k].b *= normFactor;
	}

	return sh;
}

export function resolvePrefilterBaseDimensions(envMap: Texture): {
	baseWidth: number;
	baseHeight: number;
} {
	return {
		baseWidth: Math.min(envMap.width, ENVIRONMENT_IBL_MAX_SAMPLE_WIDTH),
		baseHeight: Math.min(envMap.height, ENVIRONMENT_IBL_MAX_SAMPLE_HEIGHT),
	};
}

export function prefilterEnvMapMipLevel(
	envMap: Texture,
	level: number,
	baseWidth: number,
	baseHeight: number,
	maxMipLevels: number = ENVIRONMENT_IBL_MAX_MIP_LEVELS,
	signal?: AbortSignal | null
): EnvironmentIBLPrefilterMipData {
	assertBakeNotAborted(signal);
	const roughness = resolveRoughnessFromMipLevel(level, maxMipLevels);
	const sampleCount = resolveSampleCountByRoughness(
		roughness,
		CPU_MAX_SAMPLE_COUNT,
		CPU_MIN_SAMPLE_COUNT
	);
	const width = Math.max(1, baseWidth >> level);
	const height = Math.max(1, baseHeight >> level);
	const data = new Float32Array(width * height * 4);
	const sourceIsLinear = resolveTextureIsLinear(envMap);

	const normal: IVector3 = { x: 0, y: 0, z: 0 };
	const radiance: MutableRGB = { r: 0, g: 0, b: 0 };

	for (let j = 0; j < height; j++) {
		assertBakeNotAborted(signal);
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
				sourceIsLinear,
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
	mipData: EnvironmentIBLPrefilterMipData[]
): Texture {
	const prefiltered = new Texture(null, baseWidth, baseHeight, "HDR");
	const sorted = [...mipData].sort((left, right) => left.level - right.level);
	prefiltered.mipmaps = sorted.map((mip) => mip.data);
	prefiltered.data = prefiltered.mipmaps[0] ?? null;
	return prefiltered;
}

export function prefilterEnvMapCPU(
	envMap: Texture,
	signal?: AbortSignal | null,
	onMipComplete?: (level: number, total: number) => void
): Texture {
	const { baseWidth, baseHeight } = resolvePrefilterBaseDimensions(envMap);
	const totalMipLevels = ENVIRONMENT_IBL_MAX_MIP_LEVELS;
	const mipmaps: EnvironmentIBLPrefilterMipData[] = [];
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
	sourceIsLinear: boolean,
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
		const lightDir = Vector3.normalize({
			x: 2.0 * nDotH * half.x - view.x,
			y: 2.0 * nDotH * half.y - view.y,
			z: 2.0 * nDotH * half.z - view.z,
		});

		const nDotL = Math.max(Vector3.dot(normal, lightDir), 0);
		if (nDotL <= 0) continue;

		const phi = Math.atan2(lightDir.x, lightDir.z);
		const theta = Math.acos(Math.max(-1, Math.min(1, lightDir.y)));
		const u = (phi + Math.PI) / (2 * Math.PI);
		const v = theta / Math.PI;
		const sample = envMap.sample(u, v);

		const r = sourceIsLinear ? sample.r / 255 : decodeSRGBToLinear01(sample.r);
		const g = sourceIsLinear ? sample.g / 255 : decodeSRGBToLinear01(sample.g);
		const b = sourceIsLinear ? sample.b / 255 : decodeSRGBToLinear01(sample.b);

		outColor.r += r * nDotL;
		outColor.g += g * nDotL;
		outColor.b += b * nDotL;
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

function createPrefilterMipResources(
	outputTexture: IRenderTexture,
	paramsBuffer: IRenderBuffer
): EnvironmentIBLMipResources {
	return {
		outputTexture,
		paramsBuffer,
	};
}

async function createWebGPUResources(
	envMap: Texture,
	source: WebGPUComputeFacadeSource
): Promise<EnvironmentIBLWebGPUResources> {
	const runtime = new ComputeRuntime(source);
	try {
		const shaderCode = await loadEnvironmentIBLPrefilterShaderSource();
		const kernel = await runtime.createKernel({
			label: "EnvironmentIBLBakePrefilter",
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
			label: "EnvironmentIBLBakePrefilterSampler",
			addressModeU: AddressMode.Repeat,
			addressModeV: AddressMode.ClampToEdge,
			magFilter: FilterMode.Linear,
			minFilter: FilterMode.Linear,
			mipmapFilter: FilterMode.Linear,
		});

		const inputTexture = runtime.createTexture({
			width: Math.max(1, envMap.width),
			height: Math.max(1, envMap.height),
			format: TextureFormat.RGBA8Unorm,
			usage: TextureUsage.TextureBinding | TextureUsage.CopyDst,
			label: "EnvironmentIBLBakeInputTexture",
		});

		return {
			runtime,
			sampler,
			kernel,
			inputTexture,
		};
	} catch (error) {
		runtime.destroy();
		throw error;
	}
}

function uploadSourceTexture(
	runtime: IComputeRuntime,
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
		: 	new Uint8Array(upload.data);
	runtime.writeTexture(
		inputTexture,
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
	assertBakeNotAborted(signal);
	const ticket = kernel.dispatch({
		label: "EnvironmentIBLBakePrefilterDispatch",
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
	assertBakeNotAborted(signal);

	const readback = await runtime.readTexture({
		texture: outputTexture,
		width,
		height,
		format: TextureFormat.RGBA8Unorm,
	});
	assertBakeNotAborted(signal);
	const result = readback.toNormalizedRGBA8Float32();
	for (let i = 3; i < result.length; i += 4) {
		result[i] = 1;
	}
	return result;
}

function destroyMipResources(resources: EnvironmentIBLMipResources): void {
	destroyResource(resources.paramsBuffer);
	destroyResource(resources.outputTexture);
}

function destroyWebGPUResources(resources: EnvironmentIBLWebGPUResources): void {
	destroyResource(resources.inputTexture);
	destroyResource(resources.sampler);
	resources.kernel.destroy();
	resources.runtime.destroy();
}

export async function prefilterEnvMapWithWebGPU(
	envMap: Texture,
	source: WebGPUComputeFacadeSource,
	signal?: AbortSignal | null,
	onMipComplete?: (level: number) => void
): Promise<Texture> {
	assertBakeNotAborted(signal);

	const sourceIsLinear = resolveTextureIsLinear(envMap);
	const { baseWidth, baseHeight } = resolvePrefilterBaseDimensions(envMap);
	const resources = await createWebGPUResources(envMap, source);
	const mipmaps: EnvironmentIBLPrefilterMipData[] = [];
	const totalMipLevels = ENVIRONMENT_IBL_MAX_MIP_LEVELS;

	try {
		uploadSourceTexture(resources.runtime, resources.inputTexture, envMap);

		for (let level = 0; level < totalMipLevels; level++) {
			assertBakeNotAborted(signal);
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
				format: TextureFormat.RGBA8Unorm,
				usage:
					TextureUsage.StorageBinding |
					TextureUsage.CopySrc |
					TextureUsage.TextureBinding,
				label: `EnvironmentIBLBakePrefilterOutput_mip${level}`,
			});

			const paramsBuffer = resources.runtime.createBuffer({
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
			resources.runtime.writeBuffer(paramsBuffer, params, 0);

			const mipResources = createPrefilterMipResources(
				outputTexture,
				paramsBuffer
			);

			try {
				const mipData = await bakeMipLevelWithWebGPU(
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

function emitProgress(
	options: EnvironmentIBLBakeOptions,
	progress: EnvironmentIBLBakeProgress
): void {
	options.onProgress?.(progress);
}

function resolveWorkerCount(requestedCount?: number): number {
	const fallback = Platform.getHardwareConcurrency(4);
	if (!Number.isFinite(requestedCount)) {
		return Math.max(1, fallback);
	}
	return Math.max(1, Math.floor(requestedCount as number));
}

function createBakeWorker(workerIndex: number, poolId: string): WorkerLike {
	const workerCtor =
		(globalThis as typeof globalThis & {
			Worker?: new (...args: any[]) => Worker;
		}).Worker;

	if (typeof workerCtor !== "function") {
		throw new Error(
			`Worker constructor is unavailable for pool "${poolId}" (worker #${workerIndex})`
		);
	}

	return new workerCtor(
		new URL("./workers/environmentIblBake.worker.ts", import.meta.url),
		{
			type: "module",
		}
	) as unknown as WorkerLike;
}

function resolveWorkerPoolId(): string {
	return `${DEFAULT_BAKE_POOL_PREFIX}-${Math.random().toString(36).slice(2)}`;
}

function toWorkerEnvMapPayload(envMap: Texture): EnvironmentIBLBakeWorkerEnvMapPayload {
	return {
		width: envMap.width,
		height: envMap.height,
		colorSpace: envMap.colorSpace,
		data: envMap.data,
	};
}

async function prefilterEnvMapWithWorkers(
	envMap: Texture,
	options: EnvironmentIBLBakeOptions,
	onMipComplete: (level: number) => void
): Promise<Texture> {
	const poolId = resolveWorkerPoolId();
	const totalMipLevels = ENVIRONMENT_IBL_MAX_MIP_LEVELS;
	const workerCount = Math.min(
		resolveWorkerCount(options.workerCount),
		totalMipLevels
	);
	const { baseWidth, baseHeight } = resolvePrefilterBaseDimensions(envMap);
	const envPayload = toWorkerEnvMapPayload(envMap);

	globalWorkerScheduler.registerPool({
		id: poolId,
		size: workerCount,
		createWorker: (workerIndex, id) => createBakeWorker(workerIndex, id),
		transportPlugins: [postMessageWorkerTransportPlugin],
		defaultTimeoutMs: 0,
	});

	try {
		const tasks: Promise<EnvironmentIBLPrefilterMipData>[] = [];
		for (let level = 0; level < totalMipLevels; level++) {
			assertBakeNotAborted(options.signal);
			const payload: EnvironmentIBLBakeWorkerTaskPayload = {
				type: "prefilter-mip",
				envMap: envPayload,
				baseWidth,
				baseHeight,
				maxMipLevels: totalMipLevels,
				level,
			};

			const task = globalWorkerScheduler
				.schedule<
					EnvironmentIBLBakeWorkerTaskResult,
					EnvironmentIBLBakeWorkerTaskPayload
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
							"Environment IBL worker returned an invalid response"
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
	options: EnvironmentIBLBakeOptions,
	onMipComplete: (level: number) => void
): Texture {
	return prefilterEnvMapCPU(envMap, options.signal ?? null, (level) => {
		onMipComplete(level);
	});
}

function canUseWorkerAcceleration(options: EnvironmentIBLBakeOptions): boolean {
	return (
		options.acceleration === "worker" ||
		(options.acceleration !== "cpu" &&
			options.acceleration !== "webgpu" &&
			Platform.hasWorker())
	);
}

function canUseWebGPUAcceleration(options: EnvironmentIBLBakeOptions): boolean {
	if (options.acceleration === "webgpu") {
		return true;
	}
	if (options.acceleration === "cpu" || options.acceleration === "worker") {
		return false;
	}
	return !!options.webgpuSource;
}

async function prefilterEnvMapOnWebGPU(
	envMap: Texture,
	options: EnvironmentIBLBakeOptions,
	onMipComplete: (level: number) => void
): Promise<Texture> {
	if (!options.webgpuSource) {
		throw new Error(
			"WebGPU acceleration was requested for environment IBL baking, but no webgpuSource was provided."
		);
	}
	return prefilterEnvMapWithWebGPU(
		envMap,
		options.webgpuSource,
		options.signal ?? null,
		onMipComplete
	);
}

async function prefilterEnvMap(
	envMap: Texture,
	options: EnvironmentIBLBakeOptions,
	onMipComplete: (level: number) => void
): Promise<Texture> {
	if (canUseWebGPUAcceleration(options)) {
		try {
			return await prefilterEnvMapOnWebGPU(envMap, options, onMipComplete);
		} catch (error) {
			if (options.acceleration === "webgpu") {
				throw error;
			}
		}
	}

	if (!canUseWorkerAcceleration(options)) {
		if (options.acceleration === "worker") {
			throw new Error(
				"Worker acceleration was requested for environment IBL baking, but Worker API is unavailable."
			);
		}
		return prefilterEnvMapOnCPU(envMap, options, onMipComplete);
	}

	try {
		return await prefilterEnvMapWithWorkers(envMap, options, onMipComplete);
	} catch (error) {
		if (options.acceleration === "worker") {
			throw error;
		}
		return prefilterEnvMapOnCPU(envMap, options, onMipComplete);
	}
}

export async function bakeEnvironmentIBLFromEnvironmentMap(
	envMap: Texture,
	options: EnvironmentIBLBakeOptions = {}
): Promise<BakedEnvironmentIBL> {
	assertBakeNotAborted(options.signal);
	const totalMipLevels = ENVIRONMENT_IBL_MAX_MIP_LEVELS;
	const totalProgress = totalMipLevels + 2;
	let completed = 0;

	const sh = projectEquirectTextureToSH(envMap, options.signal ?? null);
	completed++;
	emitProgress(options, {
		phase: "project-sh",
		completed,
		total: totalProgress,
	});

	const prefiltered = await prefilterEnvMap(envMap, options, (level) => {
		completed++;
		emitProgress(options, {
			phase: "prefilter",
			completed,
			total: totalProgress,
			detail: `mip ${level + 1}/${totalMipLevels}`,
		});
	});

	assertBakeNotAborted(options.signal);
	completed++;
	emitProgress(options, {
		phase: "finalize",
		completed,
		total: totalProgress,
	});
	return {
		sh,
		prefilteredMap: prefiltered,
	};
}
