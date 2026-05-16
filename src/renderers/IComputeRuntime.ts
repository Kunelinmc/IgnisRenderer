import type {
	BindingResource,
	BufferDesc,
	IBindingGroup,
	IRenderBuffer,
	IRenderTexture,
	ISampler,
	SamplerDesc,
	ShaderModuleDesc,
	TextureDataLayout,
	TextureDesc,
	TextureFormat,
} from "./types";

export type ComputeBindingType = "buffer" | "texture" | "sampler";

export interface ComputeBindingSchemaEntry {
	key: string;
	binding: number;
	type: ComputeBindingType;
	optional?: boolean;
}

export interface ComputeResolvedBindingSchemaEntry
	extends ComputeBindingSchemaEntry {
	optional: boolean;
}

export interface ComputeWorkgroupSize {
	x: number;
	y?: number;
	z?: number;
}

export interface ComputeResolvedWorkgroupSize {
	x: number;
	y: number;
	z: number;
}

export interface ComputeKernelDescriptor {
	label?: string;
	code: string;
	entryPoint?: string;
	language?: ShaderModuleDesc["language"];
	sourceKind?: ShaderModuleDesc["sourceKind"];
	bindings: ComputeBindingSchemaEntry[];
	workgroupSize: ComputeWorkgroupSize;
}

export interface ComputeDispatchGroupOverride {
	binding: number;
	resource: BindingResource;
}

export interface ComputeDispatchDimensions {
	x: number;
	y?: number;
	z?: number;
}

export interface ComputeDispatch2D {
	width: number;
	height: number;
	depth?: number;
}

export interface ComputeExtraBindGroup {
	index: number;
	group: IBindingGroup;
}

export interface ComputeDispatchOptions {
	label?: string;
	resources: Record<string, BindingResource>;
	dispatch?: ComputeDispatchDimensions;
	dispatch2D?: ComputeDispatch2D;
	overrideEntries?: ComputeDispatchGroupOverride[];
	extraBindGroups?: ComputeExtraBindGroup[];
}

export interface ComputeDispatchTicket {
	done: Promise<void>;
}

export interface BufferReadbackResult {
	bytes: Uint8Array;
	byteLength: number;
	toFloat32(): Float32Array;
}

export interface TextureReadbackResult {
	bytes: Uint8Array;
	width: number;
	height: number;
	format: TextureFormat;
	bytesPerPixel: number;
	bytesPerRow: number;
	toFloat32(): Float32Array;
	toRGBAFloat32(): Float32Array;
	toNormalizedRGBA8Float32(): Float32Array;
}

export interface ReadBufferOptions {
	buffer: IRenderBuffer;
	size?: number;
	offset?: number;
}

export interface ReadTextureOptions {
	texture: IRenderTexture;
	width?: number;
	height?: number;
	mipLevel?: number;
	format?: TextureFormat;
	bytesPerPixel?: number;
}

export interface WriteTextureSize {
	width: number;
	height: number;
	depthOrArrayLayers?: number;
}

/**
 * Runtime-agnostic compute kernel contract.
 */
export interface IComputeKernel {
	readonly label: string;
	readonly bindings: ReadonlyArray<ComputeResolvedBindingSchemaEntry>;
	readonly workgroupSize: ComputeResolvedWorkgroupSize;
	dispatch(options: ComputeDispatchOptions): ComputeDispatchTicket;
	destroy(): void;
}

/**
 * Runtime-agnostic compute API contract.
 */
export interface IComputeRuntime {
	createBuffer(desc: BufferDesc): IRenderBuffer;
	createTexture(desc: TextureDesc): IRenderTexture;
	createSampler(desc: SamplerDesc): ISampler;
	writeBuffer(buffer: IRenderBuffer, data: BufferSource, offset?: number): void;
	writeTexture(
		texture: IRenderTexture,
		data: BufferSource,
		layout: TextureDataLayout,
		size: WriteTextureSize
	): void;
	createKernel(descriptor: ComputeKernelDescriptor): Promise<IComputeKernel>;
	readBuffer(options: ReadBufferOptions): Promise<BufferReadbackResult>;
	readTexture(options: ReadTextureOptions): Promise<TextureReadbackResult>;
	destroy(): void;
}
