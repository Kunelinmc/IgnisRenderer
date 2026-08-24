import type { ICommandBuffer, ICommandEncoder } from "../ICommandEncoder";
import type {
	BufferReadbackResult,
	ComputeBindingSchemaEntry,
	ComputeBindingType,
	ComputeDispatch2D,
	ComputeDispatchDimensions,
	ComputeDispatchGroupOverride,
	ComputeDispatchOptions,
	ComputeDispatchTicket,
	ComputeExtraBindGroup,
	ComputeKernelDescriptor,
	ComputeResolvedBindingSchemaEntry,
	ComputeResolvedWorkgroupSize,
	IComputeKernel,
	IComputeRuntime,
	ReadBufferOptions,
	ReadTextureOptions,
	TextureReadbackResult,
	WriteTextureSize,
} from "../IComputeRuntime";
import {
	getTextureFormatBytesPerRow,
	getTextureFormatInfo,
	TextureFormat,
} from "../../core/TextureFormat";
import {
	type BindingEntry,
	type BindingResource,
	type BufferDesc,
	type ComputePipelineDesc,
	type IBindingGroup,
	type IComputePipeline,
	type IRenderBuffer,
	type IRenderTexture,
	type ISampler,
	type IShaderModule,
	type SamplerDesc,
	type ShaderModuleDesc,
	type TextureDataLayout,
	type TextureDesc,
} from "../types";
import {
	resolveWebGPUComputeFacade,
	type IWebGPUComputeFacade,
	type WebGPUComputeFacadeSource,
} from "./ComputeFacade";
import { float16BitsToFloat32 } from "../../foundation/Float16";
import { alignTo } from "./texture";
import { getWebGPUResourceHandle } from "./WebGPUResourceHandle";
import { getWebGPUTexture, tryGetWebGPUBuffer, tryGetWebGPUTexture } from "./WebGPUResourceAccess";

export type {
	BufferReadbackResult,
	ComputeBindingSchemaEntry,
	ComputeBindingType,
	ComputeDispatch2D,
	ComputeDispatchDimensions,
	ComputeDispatchGroupOverride,
	ComputeDispatchOptions,
	ComputeDispatchTicket,
	ComputeExtraBindGroup,
	ComputeKernelDescriptor,
	ComputeResolvedBindingSchemaEntry,
	ComputeResolvedWorkgroupSize,
	IComputeKernel,
	IComputeRuntime,
	ReadBufferOptions,
	ReadTextureOptions,
	TextureReadbackResult,
	WriteTextureSize,
} from "../IComputeRuntime";

const DEFAULT_KERNEL_ENTRY_POINT = "csMain";
const DEFAULT_SHADER_LANGUAGE = "wgsl";
const DEFAULT_SHADER_SOURCE_KIND: ShaderModuleDesc["sourceKind"] = "unknown";
const WEBGPU_MAP_MODE_READ =
	(globalThis as typeof globalThis & {
		GPUMapMode?: { READ?: number };
	}).GPUMapMode?.READ ?? 0x0001;
const WEBGPU_BUFFER_USAGE_COPY_DST =
	(globalThis as typeof globalThis & {
		GPUBufferUsage?: { COPY_DST?: number };
	}).GPUBufferUsage?.COPY_DST ?? 0x0008;
const WEBGPU_BUFFER_USAGE_MAP_READ =
	(globalThis as typeof globalThis & {
		GPUBufferUsage?: { MAP_READ?: number };
	}).GPUBufferUsage?.MAP_READ ?? 0x0001;

type OwnedResourceKind =
	| "buffer"
	| "texture"
	| "sampler"
	| "shaderModule"
	| "computePipeline"
	| "unknown";

interface ComputeRuntimeResourceKindStats {
	buffer: number;
	texture: number;
	sampler: number;
	shaderModule: number;
	computePipeline: number;
	unknown: number;
}

export interface ComputeRuntimeResourceStats {
	destroyed: boolean;
	kernelCount: number;
	ownedResourceCount: number;
	activeResourceCount: number;
	destroyRequestedResourceCount: number;
	inflightReferenceCount: number;
	byKind: ComputeRuntimeResourceKindStats;
}

interface WebGPUComputeContext {
	device: GPUDevice;
	queue: GPUQueue;
}

interface OwnedResourceRecord {
	label: string;
	kind: OwnedResourceKind;
	target: object;
	proxy: object;
	inflightRefs: number;
	destroyRequested: boolean;
	destroyed: boolean;
}

type NormalizedComputeBindingSchemaEntry = ComputeResolvedBindingSchemaEntry;
type NormalizedWorkgroupSize = ComputeResolvedWorkgroupSize;

interface NormalizedDispatchDimensions {
	x: number;
	y: number;
	z: number;
}

interface KernelResourceSet {
	module: IShaderModule;
	pipeline: IComputePipeline;
	moduleRecord: OwnedResourceRecord;
	pipelineRecord: OwnedResourceRecord;
}

export class ComputeRuntime implements IComputeRuntime {
	private _computeFacade: IWebGPUComputeFacade;
	private _destroyed = false;
	private _ownedResources = new Set<OwnedResourceRecord>();
	private _ownedResourceByObject = new WeakMap<object, OwnedResourceRecord>();
	private _kernels = new Set<ComputeKernel>();
	private _textureMetadata = new WeakMap<
		object,
		{
			width: number;
			height: number;
			format: TextureFormat;
		}
	>();

	constructor(source: WebGPUComputeFacadeSource) {
		this._computeFacade = resolveWebGPUComputeFacade(source);
		resolveWebGPUComputeContext(this._computeFacade);
	}

	public createBuffer(desc: BufferDesc): IRenderBuffer {
		this._assertAlive("create buffers");
		const resource = this._computeFacade.createBuffer(desc);
		return this._wrapOwnedResource(
			resource,
			desc.label ?? "ComputeRuntimeBuffer",
			"buffer"
		);
	}

	public createTexture(desc: TextureDesc): IRenderTexture {
		this._assertAlive("create textures");
		const resource = this._computeFacade.createTexture(desc);
		const wrapped = this._wrapOwnedResource(
			resource,
			desc.label ?? "ComputeRuntimeTexture",
			"texture"
		);
		this._textureMetadata.set(wrapped as unknown as object, {
			width: Math.max(1, Math.floor(desc.width)),
			height: Math.max(1, Math.floor(desc.height)),
			format: desc.format,
		});
		return wrapped;
	}

	public createSampler(desc: SamplerDesc): ISampler {
		this._assertAlive("create samplers");
		const resource = this._computeFacade.createSampler(desc);
		return this._wrapOwnedResource(
			resource,
			desc.label ?? "ComputeRuntimeSampler",
			"sampler"
		);
	}

	public writeBuffer(
		buffer: IRenderBuffer,
		data: BufferSource,
		offset: number = 0
	): void {
		this._assertAlive("write buffers");
		this._computeFacade.writeBuffer(buffer, data, offset);
	}

	public writeTexture(
		texture: IRenderTexture,
		data: BufferSource,
		layout: TextureDataLayout,
		size: WriteTextureSize
	): void {
		this._assertAlive("write textures");
		const bytesPerRow = layout.bytesPerRow;
		if (!Number.isFinite(bytesPerRow) || (bytesPerRow as number) <= 0) {
			throw new Error(
				"ComputeRuntime.writeTexture() requires layout.bytesPerRow to be a positive number."
			);
		}
		this._computeFacade.writeTexture(
			texture,
			data,
			{
				offset: layout.offset ?? 0,
				bytesPerRow,
				rowsPerImage: layout.rowsPerImage ?? size.height,
				mipLevel: Math.max(0, Math.floor(layout.mipLevel ?? 0)),
			},
			{
				width: Math.max(1, Math.floor(size.width)),
				height: Math.max(1, Math.floor(size.height)),
				depthOrArrayLayers: Math.max(
					1,
					Math.floor(size.depthOrArrayLayers ?? 1),
				),
			},
		);
	}

	public async createKernel(
		descriptor: ComputeKernelDescriptor
	): Promise<ComputeKernel> {
		this._assertAlive("create compute kernels");
		const schema = normalizeBindingSchema(descriptor.bindings);
		const workgroupSize = normalizeWorkgroupSize(descriptor.workgroupSize);
		const label = descriptor.label ?? "ComputeKernel";

		const module = await this._computeFacade.createShaderModule({
			label: `${label}Module`,
			code: descriptor.code,
			language: descriptor.language ?? DEFAULT_SHADER_LANGUAGE,
			stage: "compute",
			sourceKind: descriptor.sourceKind ?? DEFAULT_SHADER_SOURCE_KIND,
		});
		this._assertAlive("finalize compute kernels");

		let pipeline: IComputePipeline | null = null;
		try {
			const pipelineDesc: ComputePipelineDesc = {
				label: `${label}Pipeline`,
				compute: {
					module,
					entryPoint:
						descriptor.entryPoint?.trim() || DEFAULT_KERNEL_ENTRY_POINT,
				},
			};
			pipeline = await this._computeFacade.createComputePipeline(pipelineDesc);
		} catch (error) {
			this._destroySafely(module, `${label}Module`);
			throw error;
		}

		const moduleRecord = this._registerOwnedResource(
			module as unknown as object,
			`${label}Module`,
			"shaderModule"
		);
		const pipelineRecord = this._registerOwnedResource(
			pipeline as unknown as object,
			`${label}Pipeline`,
			"computePipeline"
		);
		const kernelResources: KernelResourceSet = {
			module,
			pipeline,
			moduleRecord,
			pipelineRecord,
		};
		const kernel = new ComputeKernel(
			this,
			label,
			schema,
			workgroupSize,
			kernelResources
		);
		this._kernels.add(kernel);
		return kernel;
	}

	public async readBuffer(options: ReadBufferOptions): Promise<BufferReadbackResult> {
		this._assertAlive("read buffers");
		const offset = assertNonNegativeInteger(
			options.offset ?? 0,
			"readBuffer.offset"
		);
		const requestedSize = options.size ?? Math.max(0, options.buffer.size - offset);
		const size = assertNonNegativeInteger(requestedSize, "readBuffer.size");
		if (size <= 0) {
			return createBufferReadbackResult(new Uint8Array(0));
		}
		const sourceBuffer = resolveGPUBufferHandle(options.buffer);
		const context = resolveWebGPUComputeContext(this._computeFacade);
		const readbackBuffer = context.device.createBuffer({
			label: "ComputeRuntimeReadBuffer",
			size: Math.max(4, size),
			usage: WEBGPU_BUFFER_USAGE_COPY_DST | WEBGPU_BUFFER_USAGE_MAP_READ,
		});

		try {
			const encoder = context.device.createCommandEncoder({
				label: "ComputeRuntimeReadBufferEncoder",
			});
			encoder.copyBufferToBuffer(
				sourceBuffer,
				offset,
				readbackBuffer,
				0,
				size
			);
			context.queue.submit([encoder.finish()]);
			await readbackBuffer.mapAsync(WEBGPU_MAP_MODE_READ, 0, size);
			const mapped = readbackBuffer.getMappedRange(0, size);
			const bytes = new Uint8Array(mapped.slice(0));
			readbackBuffer.unmap();
			return createBufferReadbackResult(bytes);
		} finally {
			try {
				readbackBuffer.destroy();
			} catch {
				// ignore cleanup failures
			}
		}
	}

	public async readTexture(
		options: ReadTextureOptions
	): Promise<TextureReadbackResult> {
		this._assertAlive("read textures");
		const textureObject = options.texture as unknown as object;
		const metadata = this._textureMetadata.get(textureObject) ?? null;
		const width = assertPositiveInteger(
			options.width ?? options.texture.width ?? metadata?.width ?? 1,
			"readTexture.width"
		);
		const height = assertPositiveInteger(
			options.height ?? options.texture.height ?? metadata?.height ?? 1,
			"readTexture.height"
		);
		const format = options.format ?? metadata?.format ?? TextureFormat.RGBA8Unorm;
		const info = getTextureFormatInfo(format);
		const bytesPerPixel = options.bytesPerPixel ?? info.bytesPerBlock;
		const unalignedBytesPerRow =
			options.bytesPerPixel !== undefined ?
				width * bytesPerPixel
			:	getTextureFormatBytesPerRow(format, width);
		const bytesPerRow = alignTo(unalignedBytesPerRow, 256);
		const readbackSize = bytesPerRow * height;
		const mipLevel = assertNonNegativeInteger(
			options.mipLevel ?? 0,
			"readTexture.mipLevel"
		);
		const sourceTexture = getWebGPUTexture(options.texture).texture;
		const context = resolveWebGPUComputeContext(this._computeFacade);
		const readbackBuffer = context.device.createBuffer({
			label: "ComputeRuntimeReadTextureBuffer",
			size: Math.max(4, readbackSize),
			usage: WEBGPU_BUFFER_USAGE_COPY_DST | WEBGPU_BUFFER_USAGE_MAP_READ,
		});

		try {
			const encoder = context.device.createCommandEncoder({
				label: "ComputeRuntimeReadTextureEncoder",
			});
			encoder.copyTextureToBuffer(
				{
					texture: sourceTexture,
					mipLevel,
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
			context.queue.submit([encoder.finish()]);
			await readbackBuffer.mapAsync(WEBGPU_MAP_MODE_READ, 0, readbackSize);
			const mapped = readbackBuffer.getMappedRange(0, readbackSize);
			const bytes = new Uint8Array(mapped.slice(0));
			readbackBuffer.unmap();
			return createTextureReadbackResult({
				bytes,
				width,
				height,
				format,
				bytesPerPixel,
				bytesPerRow,
			});
		} finally {
			try {
				readbackBuffer.destroy();
			} catch {
				// ignore cleanup failures
			}
		}
	}

	public getResourceStats(): ComputeRuntimeResourceStats {
		const byKind: ComputeRuntimeResourceKindStats = {
			buffer: 0,
			texture: 0,
			sampler: 0,
			shaderModule: 0,
			computePipeline: 0,
			unknown: 0,
		};
		let destroyRequestedResourceCount = 0;
		let inflightReferenceCount = 0;

		for (const record of this._ownedResources) {
			byKind[record.kind]++;
			inflightReferenceCount += Math.max(0, record.inflightRefs);
			if (record.destroyRequested) {
				destroyRequestedResourceCount++;
			}
		}

		const ownedResourceCount = this._ownedResources.size;
		return {
			destroyed: this._destroyed,
			kernelCount: this._kernels.size,
			ownedResourceCount,
			activeResourceCount: Math.max(
				0,
				ownedResourceCount - destroyRequestedResourceCount
			),
			destroyRequestedResourceCount,
			inflightReferenceCount,
			byKind,
		};
	}

	public destroy(): void {
		if (this._destroyed) {
			return;
		}
		this._destroyed = true;
		for (const kernel of Array.from(this._kernels)) {
			this._destroyKernel(kernel);
		}
		for (const record of Array.from(this._ownedResources)) {
			this._requestOwnedResourceDestroy(record);
		}
	}

	public _dispatchKernel(
		kernel: ComputeKernel,
		options: ComputeDispatchOptions
	): ComputeDispatchTicket {
		this._assertAlive("dispatch compute kernels");
		kernel._assertAlive("dispatch");

		const dimensions = resolveDispatchDimensions(
			options.dispatch,
			options.dispatch2D,
			kernel.workgroupSize
		);
		const groupZeroEntries = resolveGroupZeroEntries(
			kernel.bindings,
			options.resources,
			options.overrideEntries
		);
		const extraBindGroups = normalizeExtraBindGroups(options.extraBindGroups);
		const bindGroup = this._computeFacade.createBindingGroup({
			pipeline: kernel.pipeline,
			layoutIndex: 0,
			entries: groupZeroEntries,
			label: options.label ? `${options.label}BindGroup` : `${kernel.label}BindGroup`,
		});

		const retained = new Set<OwnedResourceRecord>();
		this._retainOwnedResourceRecord(kernel.resources.moduleRecord, retained);
		this._retainOwnedResourceRecord(kernel.resources.pipelineRecord, retained);
		for (const entry of groupZeroEntries) {
			this._retainOwnedResource(entry.resource, retained);
		}

		const encoder = this._computeFacade.createCommandEncoder();
		recordKernelDispatch(
			encoder,
			options.label ?? kernel.label,
			kernel.pipeline,
			bindGroup,
			extraBindGroups,
			dimensions
		);
		const commandBuffer = encoder.finish();
		let ticketDone: Promise<void>;
		try {
			this._submitCommands([commandBuffer]);
			ticketDone = resolveWebGPUComputeContext(
				this._computeFacade,
			).queue.onSubmittedWorkDone();
		} catch (error) {
			this._destroySafely(bindGroup, `${kernel.label} bind group`);
			this._releaseRetainedOwnedResources(retained);
			throw error;
		}

		const done = ticketDone.finally(() => {
			this._destroySafely(bindGroup, `${kernel.label} bind group`);
			this._releaseRetainedOwnedResources(retained);
		});
		return {
			done,
		};
	}

	public _destroyKernel(kernel: ComputeKernel): void {
		if (!this._kernels.has(kernel)) {
			return;
		}
		this._kernels.delete(kernel);
		kernel._markDestroyedFromRuntime();
		this._requestOwnedResourceDestroy(kernel.resources.pipelineRecord);
		this._requestOwnedResourceDestroy(kernel.resources.moduleRecord);
	}

	private _assertAlive(operation: string): void {
		if (this._destroyed) {
			throw new Error(`ComputeRuntime is destroyed; cannot ${operation}.`);
		}
	}

	private _submitCommands(commands: ICommandBuffer[]): void {
		this._computeFacade.submit(commands);
	}

	private _wrapOwnedResource<TResource extends object>(
		resource: TResource,
		label: string,
		kind: OwnedResourceKind
	): TResource {
		const record = this._registerOwnedResource(resource, label, kind);
		const runtime = this;
		const proxy = new Proxy(resource, {
			get(target, prop, receiver) {
				if (prop === "destroy") {
					return () => {
						runtime._requestOwnedResourceDestroy(record);
					};
				}
				const value = Reflect.get(target, prop, receiver);
				return typeof value === "function" ? value.bind(target) : value;
			},
			set(target, prop, value, receiver) {
				return Reflect.set(target, prop, value, receiver);
			},
			has(target, prop) {
				if (prop === "destroy") {
					return true;
				}
				return Reflect.has(target, prop);
			},
			ownKeys(target) {
				const keys = Reflect.ownKeys(target);
				if (!keys.includes("destroy")) {
					keys.push("destroy");
				}
				return keys;
			},
			getOwnPropertyDescriptor(target, prop) {
				if (prop === "destroy") {
					return {
						configurable: true,
						enumerable: true,
						value: () => {
							runtime._requestOwnedResourceDestroy(record);
						},
						writable: false,
					};
				}
				return Reflect.getOwnPropertyDescriptor(target, prop);
			},
		});
		record.proxy = proxy;
		this._ownedResourceByObject.set(proxy, record);
		return proxy;
	}

	private _registerOwnedResource(
		resource: object,
		label: string,
		kind: OwnedResourceKind = "unknown"
	): OwnedResourceRecord {
		const existing = this._ownedResourceByObject.get(resource);
		if (existing) {
			if (existing.kind === "unknown" && kind !== "unknown") {
				existing.kind = kind;
			}
			return existing;
		}
		const record: OwnedResourceRecord = {
			label,
			kind,
			target: resource,
			proxy: resource,
			inflightRefs: 0,
			destroyRequested: false,
			destroyed: false,
		};
		this._ownedResources.add(record);
		this._ownedResourceByObject.set(resource, record);
		return record;
	}

	private _retainOwnedResource(
		resource: unknown,
		retained: Set<OwnedResourceRecord>
	): void {
		if (!resource || typeof resource !== "object") {
			return;
		}
		const record = this._ownedResourceByObject.get(resource);
		if (!record) {
			return;
		}
		this._retainOwnedResourceRecord(record, retained);
	}

	private _retainOwnedResourceRecord(
		record: OwnedResourceRecord,
		retained: Set<OwnedResourceRecord>
	): void {
		if (record.destroyed) {
			throw new Error(
				`ComputeRuntime resource "${record.label}" was already destroyed.`
			);
		}
		if (retained.has(record)) {
			return;
		}
		record.inflightRefs++;
		retained.add(record);
	}

	private _releaseRetainedOwnedResources(retained: Set<OwnedResourceRecord>): void {
		for (const record of retained) {
			record.inflightRefs = Math.max(0, record.inflightRefs - 1);
			if (record.inflightRefs === 0 && record.destroyRequested) {
				this._destroyOwnedResourceRecord(record);
			}
		}
	}

	private _requestOwnedResourceDestroy(record: OwnedResourceRecord): void {
		if (record.destroyed || record.destroyRequested) {
			return;
		}
		record.destroyRequested = true;
		if (record.inflightRefs > 0) {
			return;
		}
		this._destroyOwnedResourceRecord(record);
	}

	private _destroyOwnedResourceRecord(record: OwnedResourceRecord): void {
		if (record.destroyed) {
			return;
		}
		record.destroyed = true;
		this._ownedResources.delete(record);
		this._ownedResourceByObject.delete(record.proxy);
		this._ownedResourceByObject.delete(record.target);
		this._destroySafely(record.target, record.label);
	}

	private _destroySafely(resource: unknown, label: string): void {
		const destroyFn = (resource as { destroy?: () => void } | null)?.destroy;
		if (typeof destroyFn !== "function") {
			return;
		}
		try {
			destroyFn.call(resource);
		} catch (error) {
			const detail =
				error instanceof Error ? error.message : String(error);
			throw new Error(`Failed to destroy ${label}: ${detail}`);
		}
	}
}

export class ComputeKernel implements IComputeKernel {
	private _destroyed = false;

	constructor(
		private _runtime: ComputeRuntime,
		public readonly label: string,
		public readonly bindings: ReadonlyArray<NormalizedComputeBindingSchemaEntry>,
		public readonly workgroupSize: NormalizedWorkgroupSize,
		public readonly resources: KernelResourceSet
	) {}

	public dispatch(options: ComputeDispatchOptions): ComputeDispatchTicket {
		this._assertAlive("dispatch");
		return this._runtime._dispatchKernel(this, options);
	}

	public destroy(): void {
		if (this._destroyed) {
			return;
		}
		this._runtime._destroyKernel(this);
	}

	public _markDestroyedFromRuntime(): void {
		this._destroyed = true;
	}

	public _assertAlive(operation: string): void {
		if (this._destroyed) {
			throw new Error(`ComputeKernel "${this.label}" is destroyed; cannot ${operation}.`);
		}
	}

	public get pipeline(): IComputePipeline {
		return this.resources.pipeline;
	}
}

function resolveWebGPUComputeContext(
	source: WebGPUComputeFacadeSource
): WebGPUComputeContext {
	if (!source || typeof source !== "object") {
		throw new Error(
			"ComputeRuntime requires a webgpuSource that exposes an initialized GPU device and queue."
		);
	}
	const candidate = source as {
		device?: GPUDevice | null;
		queue?: GPUQueue | null;
	};
	const device = candidate.device;
	const queue = candidate.queue ?? candidate.device?.queue;
	if (
		device &&
		queue &&
		typeof device.createCommandEncoder === "function" &&
		typeof queue.submit === "function" &&
		typeof queue.writeTexture === "function" &&
		typeof queue.onSubmittedWorkDone === "function"
	) {
		return { device, queue };
	}

	throw new Error(
		"ComputeRuntime requires a webgpuSource that exposes an initialized GPU device and queue."
	);
}

function normalizeBindingSchema(
	bindings: ComputeBindingSchemaEntry[]
): NormalizedComputeBindingSchemaEntry[] {
	if (!Array.isArray(bindings) || bindings.length <= 0) {
		throw new Error(
			"ComputeKernelDescriptor.bindings must include at least one binding schema entry."
		);
	}
	const byKey = new Set<string>();
	const byBinding = new Set<number>();
	const normalized: NormalizedComputeBindingSchemaEntry[] = [];

	for (const entry of bindings) {
		const key = entry?.key?.trim();
		if (!key) {
			throw new Error("Compute binding schema key must be a non-empty string.");
		}
		if (byKey.has(key)) {
			throw new Error(`Compute binding schema has duplicate key "${key}".`);
		}
		byKey.add(key);

		const binding = assertNonNegativeInteger(
			entry.binding,
			`bindings["${key}"].binding`
		);
		if (byBinding.has(binding)) {
			throw new Error(
				`Compute binding schema has duplicate binding index ${binding}.`
			);
		}
		byBinding.add(binding);

		if (
			entry.type !== "buffer" &&
			entry.type !== "texture" &&
			entry.type !== "sampler"
		) {
			throw new Error(
				`Compute binding schema "${key}" has unsupported type "${String(entry.type)}".`
			);
		}

		normalized.push({
			key,
			binding,
			type: entry.type,
			optional: !!entry.optional,
		});
	}

	normalized.sort((left, right) => left.binding - right.binding);
	return normalized;
}

function normalizeWorkgroupSize(size: {
	x: number;
	y?: number;
	z?: number;
}): NormalizedWorkgroupSize {
	return {
		x: assertPositiveInteger(size.x, "workgroupSize.x"),
		y: assertPositiveInteger(size.y ?? 1, "workgroupSize.y"),
		z: assertPositiveInteger(size.z ?? 1, "workgroupSize.z"),
	};
}

function resolveDispatchDimensions(
	dispatch: ComputeDispatchDimensions | undefined,
	dispatch2D: ComputeDispatch2D | undefined,
	workgroupSize: NormalizedWorkgroupSize
): NormalizedDispatchDimensions {
	if (dispatch && dispatch2D) {
		throw new Error(
			"Compute dispatch options cannot include both dispatch and dispatch2D."
		);
	}
	if (!dispatch && !dispatch2D) {
		throw new Error(
			"Compute dispatch options require either dispatch or dispatch2D."
		);
	}
	if (dispatch2D) {
		const width = assertPositiveInteger(dispatch2D.width, "dispatch2D.width");
		const height = assertPositiveInteger(dispatch2D.height, "dispatch2D.height");
		const depth = assertPositiveInteger(dispatch2D.depth ?? 1, "dispatch2D.depth");
		return {
			x: Math.max(1, Math.ceil(width / workgroupSize.x)),
			y: Math.max(1, Math.ceil(height / workgroupSize.y)),
			z: depth,
		};
	}
	return {
		x: assertPositiveInteger((dispatch as ComputeDispatchDimensions).x, "dispatch.x"),
		y: assertPositiveInteger(
			(dispatch as ComputeDispatchDimensions).y ?? 1,
			"dispatch.y"
		),
		z: assertPositiveInteger(
			(dispatch as ComputeDispatchDimensions).z ?? 1,
			"dispatch.z"
		),
	};
}

function resolveGroupZeroEntries(
	schema: ReadonlyArray<NormalizedComputeBindingSchemaEntry>,
	resources: Record<string, BindingResource>,
	overrideEntries?: ComputeDispatchGroupOverride[]
): BindingEntry[] {
	const expectedKeys = new Set(schema.map((entry) => entry.key));
	for (const key of Object.keys(resources ?? {})) {
		if (!expectedKeys.has(key)) {
			throw new Error(
				`Compute dispatch received unknown resource key "${key}".`
			);
		}
	}

	const overrideByBinding = new Map<number, BindingResource>();
	for (const override of overrideEntries ?? []) {
		const binding = assertNonNegativeInteger(
			override.binding,
			"overrideEntries.binding"
		);
		if (overrideByBinding.has(binding)) {
			throw new Error(
				`Compute dispatch overrideEntries has duplicate binding ${binding}.`
			);
		}
		const schemaEntry = schema.find((entry) => entry.binding === binding);
		if (!schemaEntry) {
			throw new Error(
				`Compute dispatch overrideEntries contains unknown binding ${binding}.`
			);
		}
		validateBindingResourceType(
			schemaEntry.key,
			schemaEntry.type,
			override.resource
		);
		overrideByBinding.set(binding, override.resource);
	}

	const entries: BindingEntry[] = [];
	for (const entry of schema) {
		const resource =
			overrideByBinding.get(entry.binding) ?? resources[entry.key];
		if (!resource) {
			if (entry.optional) {
				continue;
			}
			throw new Error(
				`Compute dispatch is missing required resource "${entry.key}".`
			);
		}
		validateBindingResourceType(entry.key, entry.type, resource);
		entries.push({
			binding: entry.binding,
			resource,
		});
	}
	return entries;
}

function normalizeExtraBindGroups(
	extraBindGroups: ComputeExtraBindGroup[] | undefined
): ComputeExtraBindGroup[] {
	if (!extraBindGroups || extraBindGroups.length <= 0) {
		return [];
	}
	const byIndex = new Set<number>();
	const normalized: ComputeExtraBindGroup[] = [];

	for (const entry of extraBindGroups) {
		const index = assertNonNegativeInteger(entry.index, "extraBindGroups.index");
		if (index === 0) {
			throw new Error(
				"extraBindGroups cannot target index 0 because group 0 is managed by kernel schema."
			);
		}
		if (byIndex.has(index)) {
			throw new Error(
				`extraBindGroups has duplicate index ${index}.`
			);
		}
		byIndex.add(index);
		normalized.push({
			index,
			group: entry.group,
		});
	}

	normalized.sort((left, right) => left.index - right.index);
	return normalized;
}

function recordKernelDispatch(
	encoder: ICommandEncoder,
	label: string,
	pipeline: IComputePipeline,
	bindGroup: IBindingGroup,
	extraBindGroups: ComputeExtraBindGroup[],
	dimensions: NormalizedDispatchDimensions
): void {
	encoder.beginComputePass({
		label: `${label}Pass`,
	});
	encoder.setComputePipeline(pipeline);
	encoder.setBindingGroup(0, bindGroup);
	for (const extra of extraBindGroups) {
		encoder.setBindingGroup(extra.index, extra.group);
	}
	encoder.dispatchWorkgroups(dimensions.x, dimensions.y, dimensions.z);
	encoder.endComputePass();
}

function createBufferReadbackResult(bytes: Uint8Array): BufferReadbackResult {
	return {
		bytes,
		byteLength: bytes.byteLength,
		toFloat32: () => bytesToFloat32Array(bytes),
	};
}

function createTextureReadbackResult(input: {
	bytes: Uint8Array;
	width: number;
	height: number;
	format: TextureFormat;
	bytesPerPixel: number;
	bytesPerRow: number;
}): TextureReadbackResult {
	return {
		bytes: input.bytes,
		width: input.width,
		height: input.height,
		format: input.format,
		bytesPerPixel: input.bytesPerPixel,
		bytesPerRow: input.bytesPerRow,
		toFloat32: () => bytesToFloat32Array(input.bytes),
		toRGBAFloat32: () => decodeTextureReadbackToRGBAFloat32(input),
		toNormalizedRGBA8Float32: () => decodeNormalizedRGBA8Readback(input),
	};
}

function decodeTextureReadbackToRGBAFloat32(input: {
	bytes: Uint8Array;
	width: number;
	height: number;
	format: TextureFormat;
	bytesPerPixel: number;
	bytesPerRow: number;
}): Float32Array {
	validateTextureReadbackBytes(input);
	const info = getTextureFormatInfo(input.format);
	if (
		info.isCompressed ||
		info.hasDepth ||
		info.hasStencil ||
		info.componentType === "uint" ||
		info.componentType === "sint" ||
		info.componentType === "ufloat" ||
		info.componentType === "mixed"
	) {
		throw new Error(
			`toRGBAFloat32() does not support texture format "${input.format}".`
		);
	}
	const bytesPerComponent = info.bytesPerBlock / info.channelCount;
	if (!Number.isInteger(bytesPerComponent)) {
		throw new Error(
			`toRGBAFloat32() cannot decode packed texture format "${input.format}".`
		);
	}
	return decodeNumericTextureReadbackToRGBA(input, bytesPerComponent);
}

function decodeNormalizedRGBA8Readback(input: {
	bytes: Uint8Array;
	width: number;
	height: number;
	format: TextureFormat;
	bytesPerPixel: number;
	bytesPerRow: number;
}): Float32Array {
	const info = getTextureFormatInfo(input.format);
	if (info.componentType !== "unorm" || info.bytesPerBlock !== info.channelCount) {
		throw new Error(
			"toNormalizedRGBA8Float32() is only supported for 8-bit unorm readback formats."
		);
	}
	if (input.bytesPerPixel !== info.bytesPerBlock) {
		throw new Error(
			`toNormalizedRGBA8Float32() requires ${info.bytesPerBlock}-byte pixels, received ${input.bytesPerPixel}.`
		);
	}
	validateTextureReadbackBytes(input);
	const output = new Float32Array(input.width * input.height * 4);
	const isBgra = input.format === TextureFormat.BGRA8Unorm;
	for (let y = 0; y < input.height; y++) {
		const srcRowOffset = y * input.bytesPerRow;
		const dstRowOffset = y * input.width * 4;
		for (let x = 0; x < input.width; x++) {
			const srcOffset = srcRowOffset + x * info.channelCount;
			const dstOffset = dstRowOffset + x * 4;
			if (isBgra) {
				output[dstOffset] = input.bytes[srcOffset + 2] / 255;
				output[dstOffset + 1] = input.bytes[srcOffset + 1] / 255;
				output[dstOffset + 2] = input.bytes[srcOffset] / 255;
			} else {
				output[dstOffset] = input.bytes[srcOffset] / 255;
				output[dstOffset + 1] =
					info.channelCount > 1 ? input.bytes[srcOffset + 1] / 255 : 0;
				output[dstOffset + 2] =
					info.channelCount > 2 ? input.bytes[srcOffset + 2] / 255 : 0;
			}
			output[dstOffset + 3] =
				info.channelCount > 3 ? input.bytes[srcOffset + 3] / 255 : 1;
		}
	}
	return output;
}

function decodeNumericTextureReadbackToRGBA(input: {
	bytes: Uint8Array;
	width: number;
	height: number;
	format: TextureFormat;
	bytesPerRow: number;
}, bytesPerComponent: number): Float32Array {
	const info = getTextureFormatInfo(input.format);
	const output = new Float32Array(input.width * input.height * 4);
	const view = new DataView(
		input.bytes.buffer,
		input.bytes.byteOffset,
		input.bytes.byteLength
	);
	const isBgra =
		input.format === TextureFormat.BGRA8Unorm ||
		input.format === TextureFormat.BGRA8UnormSrgb;
	for (let y = 0; y < input.height; y++) {
		const srcRowOffset = y * input.bytesPerRow;
		const dstRowOffset = y * input.width * 4;
		for (let x = 0; x < input.width; x++) {
			const srcOffset = srcRowOffset + x * info.bytesPerBlock;
			const dstOffset = dstRowOffset + x * 4;
			for (let channel = 0; channel < 4; channel++) {
				const srcChannel =
					isBgra && channel === 0 ? 2
					: isBgra && channel === 2 ? 0
					: channel;
				output[dstOffset + channel] =
					srcChannel < info.channelCount ?
						readNumericComponent(
							view,
							srcOffset + srcChannel * bytesPerComponent,
							bytesPerComponent,
							info.componentType
						)
					:	channel === 3 ? 1
					:	0;
			}
		}
	}
	return output;
}

function readNumericComponent(
	view: DataView,
	offset: number,
	bytesPerComponent: number,
	componentType: string
): number {
	switch (componentType) {
		case "unorm":
			if (bytesPerComponent === 1) return view.getUint8(offset) / 255;
			if (bytesPerComponent === 2) return view.getUint16(offset, true) / 65535;
			return view.getUint32(offset, true) / 0xffffffff;
		case "snorm":
			if (bytesPerComponent === 1) return Math.max(-1, view.getInt8(offset) / 127);
			if (bytesPerComponent === 2) {
				return Math.max(-1, view.getInt16(offset, true) / 32767);
			}
			return Math.max(-1, view.getInt32(offset, true) / 0x7fffffff);
		case "float":
			if (bytesPerComponent === 2) {
				return float16BitsToFloat32(view.getUint16(offset, true));
			}
			if (bytesPerComponent === 4) {
				return view.getFloat32(offset, true);
			}
			break;
		default:
			break;
	}
	throw new Error(
		`Unsupported numeric texture component layout: ${componentType}/${bytesPerComponent}.`
	);
}

function decodeRGBA16FloatReadback(input: {
	bytes: Uint8Array;
	width: number;
	height: number;
	bytesPerRow: number;
}): Float32Array {
	const output = new Float32Array(input.width * input.height * 4);
	const view = new DataView(
		input.bytes.buffer,
		input.bytes.byteOffset,
		input.bytes.byteLength
	);
	for (let y = 0; y < input.height; y++) {
		const srcRowOffset = y * input.bytesPerRow;
		const dstRowOffset = y * input.width * 4;
		for (let x = 0; x < input.width; x++) {
			const srcOffset = srcRowOffset + x * 8;
			const dstOffset = dstRowOffset + x * 4;
			output[dstOffset] = float16BitsToFloat32(
				view.getUint16(srcOffset, true)
			);
			output[dstOffset + 1] = float16BitsToFloat32(
				view.getUint16(srcOffset + 2, true)
			);
			output[dstOffset + 2] = float16BitsToFloat32(
				view.getUint16(srcOffset + 4, true)
			);
			output[dstOffset + 3] = float16BitsToFloat32(
				view.getUint16(srcOffset + 6, true)
			);
		}
	}
	return output;
}

function validateTextureReadbackBytes(input: {
	bytes: Uint8Array;
	height: number;
	bytesPerRow: number;
}): void {
	const minByteLength = input.bytesPerRow * input.height;
	if (input.bytes.byteLength < minByteLength) {
		throw new Error(
			`Texture readback byte length ${input.bytes.byteLength} is smaller than expected ${minByteLength}.`
		);
	}
}

function bytesToFloat32Array(bytes: Uint8Array): Float32Array {
	const alignedByteLength = bytes.byteLength - (bytes.byteLength % 4);
	if (alignedByteLength <= 0) {
		return new Float32Array(0);
	}
	const start = bytes.byteOffset;
	const end = start + alignedByteLength;
	const sliced = bytes.buffer.slice(start, end);
	return new Float32Array(sliced);
}

function resolveTextureBytesPerPixel(format: TextureFormat): number {
	return getTextureFormatInfo(format).bytesPerBlock;
}

function resolveGPUBufferHandle(resource: unknown): GPUBuffer {
	const fromWrapped = tryGetWebGPUBuffer(resource);
	if (fromWrapped) {
		return fromWrapped;
	}
	if (isLikelyGPUBufferHandle(resource)) {
		return resource as GPUBuffer;
	}
	throw new Error("Expected a GPU buffer-backed resource.");
}

function validateBindingResourceType(
	key: string,
	expectedType: ComputeBindingType,
	resource: BindingResource
): void {
	const actualType = classifyBindingResource(resource);
	if (actualType === expectedType) {
		return;
	}
	throw new Error(
		`Compute resource "${key}" expects "${expectedType}" but received "${actualType}".`
	);
}

function classifyBindingResource(resource: unknown): ComputeBindingType | "unknown" {
	if (isTextureBindingResource(resource)) {
		return "texture";
	}
	if (isBufferBindingResource(resource)) {
		return "buffer";
	}
	if (isSamplerBindingResource(resource)) {
		return "sampler";
	}
	return "unknown";
}

function isBufferBindingResource(resource: unknown): boolean {
	if (!resource || typeof resource !== "object") {
		return false;
	}
	if (tryGetWebGPUBuffer(resource)) {
		return true;
	}
	const bufferBinding = resource as { buffer?: unknown };
	if (bufferBinding.buffer && isLikelyGPUBufferHandle(bufferBinding.buffer)) {
		return true;
	}
	const tag = getObjectTag(resource);
	if (tag === "[object GPUBuffer]") {
		return true;
	}
	if (
		typeof (resource as { size?: unknown }).size === "number" &&
		typeof (resource as { destroy?: unknown }).destroy === "function"
	) {
		return true;
	}
	return false;
}

function isTextureBindingResource(resource: unknown): boolean {
	if (!resource || typeof resource !== "object") {
		return false;
	}
	if (tryGetWebGPUTexture(resource)) {
		return true;
	}
	const tag = getObjectTag(resource);
	if (tag === "[object GPUTexture]" || tag === "[object GPUTextureView]") {
		return true;
	}
	if (typeof (resource as { createView?: unknown }).createView === "function") {
		return true;
	}
	const textureLike = resource as {
		width?: unknown;
		height?: unknown;
		destroy?: unknown;
	};
	if (
		typeof textureLike.width === "number" &&
		typeof textureLike.height === "number" &&
		typeof textureLike.destroy === "function"
	) {
		return true;
	}
	return false;
}

function isSamplerBindingResource(resource: unknown): boolean {
	if (!resource || typeof resource !== "object") {
		return false;
	}
	if (isBufferBindingResource(resource) || isTextureBindingResource(resource)) {
		return false;
	}
	const tag = getObjectTag(resource);
	if (tag === "[object GPUSampler]") {
		return true;
	}
	const handle = getWebGPUResourceHandle(resource);
	if (handle) {
		if (isLikelyGPUBufferHandle(handle) || isLikelyGPUTextureHandle(handle)) {
			return false;
		}
		return true;
	}
	return false;
}

function isLikelyGPUBufferHandle(value: unknown): boolean {
	if (!value || typeof value !== "object") {
		return false;
	}
	const candidate = value as {
		destroy?: unknown;
		mapAsync?: unknown;
		getMappedRange?: unknown;
		createView?: unknown;
	};
	if (typeof candidate.createView === "function") {
		return false;
	}
	if (
		typeof candidate.mapAsync === "function" ||
		typeof candidate.getMappedRange === "function"
	) {
		return true;
	}
	return typeof candidate.destroy === "function" && "size" in candidate;
}

function isLikelyGPUTextureHandle(value: unknown): boolean {
	if (!value || typeof value !== "object") {
		return false;
	}
	return typeof (value as { createView?: unknown }).createView === "function";
}

function getObjectTag(value: unknown): string {
	return Object.prototype.toString.call(value);
}

function assertPositiveInteger(value: number, name: string): number {
	if (!Number.isInteger(value) || value <= 0) {
		throw new Error(`${name} must be a positive integer, received ${value}.`);
	}
	return value;
}

function assertNonNegativeInteger(value: number, name: string): number {
	if (!Number.isInteger(value) || value < 0) {
		throw new Error(`${name} must be a non-negative integer, received ${value}.`);
	}
	return value;
}
