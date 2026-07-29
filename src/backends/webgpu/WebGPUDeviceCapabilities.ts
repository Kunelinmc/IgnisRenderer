/// <reference types="@webgpu/types" />
import type { RenderBackendDebugInfo } from "../IRenderBackend";

import {
	WEBGPU_DEFERRED_COLOR_BYTES_PER_SAMPLE,
	WEBGPU_DEFERRED_COLOR_TARGET_COUNT,
	WEBGPU_DEFERRED_STORAGE_TEXTURE_COUNT,
	WEBGPU_MRT_COLOR_BYTES_PER_SAMPLE,
	WEBGPU_MRT_COLOR_TARGET_COUNT,
	WEBGPU_REQUIRED_FRAGMENT_SAMPLED_TEXTURE_COUNT,
	WEBGPU_REQUIRED_FRAGMENT_STORAGE_BUFFER_COUNT,
	WEBGPU_SCENE_REQUIRED_FRAGMENT_SAMPLER_COUNT,
} from "./constants";

const WEBGPU_DEBUG_LIMIT_KEYS = [
	"maxTextureDimension2D",
	"maxTextureArrayLayers",
	"maxBindGroups",
	"maxBindingsPerBindGroup",
	"maxBufferSize",
	"maxStorageBufferBindingSize",
	"maxUniformBufferBindingSize",
	"maxSampledTexturesPerShaderStage",
	"maxSamplersPerShaderStage",
	"maxStorageBuffersPerShaderStage",
	"maxStorageTexturesPerShaderStage",
	"maxColorAttachments",
	"maxColorAttachmentBytesPerSample",
] as const;

const WEBGPU_OPTIONAL_DEVICE_FEATURES = [
	"timestamp-query",
	"indirect-first-instance",
] as const satisfies readonly GPUFeatureName[];

const WEBGPU_REQUIRED_DEVICE_LIMITS = [
	{
		name: "maxSampledTexturesPerShaderStage",
		minimum: WEBGPU_REQUIRED_FRAGMENT_SAMPLED_TEXTURE_COUNT,
		description: "WebGPU pipeline sampled texture count",
	},
	{
		name: "maxSamplersPerShaderStage",
		minimum: WEBGPU_SCENE_REQUIRED_FRAGMENT_SAMPLER_COUNT,
		description: "scene pipeline sampler count",
	},
	{
		name: "maxStorageBuffersPerShaderStage",
		minimum: WEBGPU_REQUIRED_FRAGMENT_STORAGE_BUFFER_COUNT,
		description: "WebGPU pipeline storage buffer count",
	},
] as const satisfies readonly WebGPUMinimumLimit[];

const WEBGPU_COLOR_ATTACHMENT_LIMIT_TIERS = [
	WEBGPU_DEFERRED_COLOR_TARGET_COUNT,
	WEBGPU_MRT_COLOR_TARGET_COUNT,
] as const;

const WEBGPU_COLOR_ATTACHMENT_BYTE_LIMIT_TIERS = [
	WEBGPU_DEFERRED_COLOR_BYTES_PER_SAMPLE,
	WEBGPU_MRT_COLOR_BYTES_PER_SAMPLE,
] as const;

type WebGPURequiredLimitName =
	| "maxSampledTexturesPerShaderStage"
	| "maxSamplersPerShaderStage"
	| "maxStorageBuffersPerShaderStage";

interface WebGPUMinimumLimit {
	name: WebGPURequiredLimitName;
	minimum: number;
	description: string;
}

type WebGPUAdapterInfoLike = Partial<GPUAdapterInfo> & {
	readonly isFallbackAdapter?: boolean;
};

/** @internal Builds the backend diagnostic snapshot for one WebGPU device. */
export function createWebGPUDebugInfo(
	adapter: GPUAdapter,
	device: GPUDevice,
): RenderBackendDebugInfo {
	const adapterInfo = resolveWebGPUAdapterInfo(adapter, device);
	const raw = collectWebGPUAdapterRaw(adapterInfo);
	const deviceInfo =
		adapterInfo || Object.keys(raw).length > 0
			? {
					vendor: normalizeDebugString(adapterInfo?.vendor),
					architecture: normalizeDebugString(adapterInfo?.architecture),
					device: normalizeDebugString(adapterInfo?.device),
					description: normalizeDebugString(adapterInfo?.description),
					isFallbackAdapter:
						typeof adapterInfo?.isFallbackAdapter === "boolean"
							? adapterInfo.isFallbackAdapter
							: undefined,
					raw: Object.keys(raw).length > 0 ? raw : undefined,
				}
			: undefined;

	return {
		backend: "webgpu",
		api: "webgpu",
		available: true,
		device: deviceInfo,
		limits: collectWebGPULimits(adapter, device),
		features: collectWebGPUFeatures(device.features),
	};
}

/** @internal Selects the device limits requested by `WebGPUBackend`. */
export function createWebGPURequiredLimits(limits: GPUSupportedLimits): Record<string, number> {
	assertWebGPUMinimumLimits(limits, "WebGPU adapter");

	const requiredLimits: Record<string, number> = {
		maxSampledTexturesPerShaderStage: WEBGPU_REQUIRED_FRAGMENT_SAMPLED_TEXTURE_COUNT,
		maxSamplersPerShaderStage: WEBGPU_SCENE_REQUIRED_FRAGMENT_SAMPLER_COUNT,
	};
	const maxColorAttachments = selectHighestSupportedLimitTier(
		limits.maxColorAttachments,
		WEBGPU_COLOR_ATTACHMENT_LIMIT_TIERS,
	);
	const maxColorAttachmentBytesPerSample = selectHighestSupportedLimitTier(
		limits.maxColorAttachmentBytesPerSample,
		WEBGPU_COLOR_ATTACHMENT_BYTE_LIMIT_TIERS,
	);

	if (maxColorAttachments !== undefined) {
		requiredLimits.maxColorAttachments = maxColorAttachments;
	}
	if (maxColorAttachmentBytesPerSample !== undefined) {
		requiredLimits.maxColorAttachmentBytesPerSample = maxColorAttachmentBytesPerSample;
	}
	if (limits.maxStorageTexturesPerShaderStage >= WEBGPU_DEFERRED_STORAGE_TEXTURE_COUNT) {
		requiredLimits.maxStorageTexturesPerShaderStage = WEBGPU_DEFERRED_STORAGE_TEXTURE_COUNT;
	}
	if (limits.maxTextureDimension2D > 0) {
		requiredLimits.maxTextureDimension2D = limits.maxTextureDimension2D;
	}
	if (limits.maxStorageBuffersPerShaderStage >= WEBGPU_REQUIRED_FRAGMENT_STORAGE_BUFFER_COUNT) {
		requiredLimits.maxStorageBuffersPerShaderStage = limits.maxStorageBuffersPerShaderStage;
	}

	return requiredLimits;
}

/** @internal Selects optional WebGPU features supported by the adapter. */
export function selectSupportedWebGPUFeatures(adapter: GPUAdapter): GPUFeatureName[] {
	if (typeof adapter.features?.has !== "function") {
		return [];
	}
	return WEBGPU_OPTIONAL_DEVICE_FEATURES.filter((feature) => adapter.features.has(feature));
}

/** @internal Validates the minimum limits required by WebGPU rendering. */
export function assertWebGPUMinimumLimits(limits: GPUSupportedLimits, owner: string): void {
	for (const { name, minimum, description } of WEBGPU_REQUIRED_DEVICE_LIMITS) {
		const available = limits[name];
		if (available < minimum) {
			throw new Error(
				`${owner} ${name} (${available}) is below required ${description} (${minimum}).`,
			);
		}
	}
}

function resolveWebGPUAdapterInfo(
	adapter: GPUAdapter,
	device: GPUDevice,
): WebGPUAdapterInfoLike | null {
	const deviceInfo = (device as { adapterInfo?: WebGPUAdapterInfoLike }).adapterInfo;
	if (deviceInfo) {
		return deviceInfo;
	}
	return (adapter as { info?: WebGPUAdapterInfoLike }).info ?? null;
}

function collectWebGPUAdapterRaw(
	info: WebGPUAdapterInfoLike | null,
): Record<string, string | number | boolean> {
	if (!info) {
		return {};
	}
	const raw: Record<string, string | number | boolean> = {};
	for (const key of [
		"vendor",
		"architecture",
		"device",
		"description",
		"isFallbackAdapter",
		"subgroupMinSize",
		"subgroupMaxSize",
	] as const) {
		const value = info[key];
		if (
			(typeof value === "string" && value.length > 0) ||
			typeof value === "number" ||
			typeof value === "boolean"
		) {
			raw[key] = value;
		}
	}
	return raw;
}

function collectWebGPULimits(adapter: GPUAdapter, device: GPUDevice): Record<string, number> {
	const limits: Record<string, number> = {};
	for (const key of WEBGPU_DEBUG_LIMIT_KEYS) {
		const value = readNumericLimit(device.limits, key) ?? readNumericLimit(adapter.limits, key);
		if (typeof value === "number") {
			limits[key] = value;
		}
	}
	return limits;
}

function readNumericLimit(limits: unknown, key: string): number | undefined {
	const value = (limits as Record<string, unknown> | null | undefined)?.[key];
	return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function collectWebGPUFeatures(features: unknown): readonly string[] {
	if (!features || typeof (features as Iterable<string>)[Symbol.iterator] !== "function") {
		return [];
	}
	try {
		return Array.from(features as Iterable<string>, (feature) => String(feature)).sort();
	} catch {
		return [];
	}
}

function normalizeDebugString(value: unknown): string | undefined {
	return typeof value === "string" && value.length > 0 ? value : undefined;
}

function selectHighestSupportedLimitTier(
	available: number,
	tiers: readonly number[],
): number | undefined {
	return tiers.find((tier) => available >= tier);
}
