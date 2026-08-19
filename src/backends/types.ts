/**
 * Universal Graphics Terminology for IgnisRenderer
 */
import type { ShaderSourceSegmentMap } from "../shaders/runtime/types";

export enum BufferUsage {
	Vertex = 1 << 0,
	Index = 1 << 1,
	Uniform = 1 << 2,
	Storage = 1 << 3,
	CopySrc = 1 << 4,
	CopyDst = 1 << 5,
	MapRead = 1 << 6,
	MapWrite = 1 << 7,
	Indirect = 1 << 8,
}

export enum TextureUsage {
	CopySrc = 1 << 0,
	CopyDst = 1 << 1,
	TextureBinding = 1 << 2,
	StorageBinding = 1 << 3,
	RenderAttachment = 1 << 4,
	ComputeStorage = 1 << 5,
}

export enum TextureFormat {
	R8Unorm = "r8unorm",
	R8Snorm = "r8snorm",
	R8Uint = "r8uint",
	R8Sint = "r8sint",
	R16Unorm = "r16unorm",
	R16Snorm = "r16snorm",
	R16Uint = "r16uint",
	R16Sint = "r16sint",
	R16Float = "r16float",
	RG8Unorm = "rg8unorm",
	RG8Snorm = "rg8snorm",
	RG8Uint = "rg8uint",
	RG8Sint = "rg8sint",
	R32Uint = "r32uint",
	R32Sint = "r32sint",
	R32Float = "r32float",
	RG16Unorm = "rg16unorm",
	RG16Snorm = "rg16snorm",
	RG16Uint = "rg16uint",
	RG16Sint = "rg16sint",
	RG16Float = "rg16float",
	RGBA8Unorm = "rgba8unorm",
	RGBA8UnormSrgb = "rgba8unorm-srgb",
	RGBA8Snorm = "rgba8snorm",
	RGBA8Uint = "rgba8uint",
	RGBA8Sint = "rgba8sint",
	BGRA8Unorm = "bgra8unorm",
	BGRA8UnormSrgb = "bgra8unorm-srgb",
	RGB9E5UFloat = "rgb9e5ufloat",
	RGB10A2Uint = "rgb10a2uint",
	RGB10A2Unorm = "rgb10a2unorm",
	RG11B10UFloat = "rg11b10ufloat",
	RG32Uint = "rg32uint",
	RG32Sint = "rg32sint",
	RG32Float = "rg32float",
	RGBA16Unorm = "rgba16unorm",
	RGBA16Snorm = "rgba16snorm",
	RGBA16Uint = "rgba16uint",
	RGBA16Sint = "rgba16sint",
	RGBA16Float = "rgba16float",
	RGBA32Uint = "rgba32uint",
	RGBA32Sint = "rgba32sint",
	RGBA32Float = "rgba32float",
	Stencil8 = "stencil8",
	Depth16Unorm = "depth16unorm",
	Depth24Plus = "depth24plus",
	Depth32Float = "depth32float",
	Depth24PlusStencil8 = "depth24plus-stencil8",
	Depth32FloatStencil8 = "depth32float-stencil8",
	BC1RGBAUnorm = "bc1-rgba-unorm",
	BC1RGBAUnormSrgb = "bc1-rgba-unorm-srgb",
	BC2RGBAUnorm = "bc2-rgba-unorm",
	BC2RGBAUnormSrgb = "bc2-rgba-unorm-srgb",
	BC3RGBAUnorm = "bc3-rgba-unorm",
	BC3RGBAUnormSrgb = "bc3-rgba-unorm-srgb",
	BC4RUnorm = "bc4-r-unorm",
	BC4RSnorm = "bc4-r-snorm",
	BC5RGUnorm = "bc5-rg-unorm",
	BC5RGSnorm = "bc5-rg-snorm",
	BC6HRGBUFloat = "bc6h-rgb-ufloat",
	BC6HRGBFloat = "bc6h-rgb-float",
	BC7RGBAUnorm = "bc7-rgba-unorm",
	BC7RGBAUnormSrgb = "bc7-rgba-unorm-srgb",
	ETC2RGB8Unorm = "etc2-rgb8unorm",
	ETC2RGB8UnormSrgb = "etc2-rgb8unorm-srgb",
	ETC2RGB8A1Unorm = "etc2-rgb8a1unorm",
	ETC2RGB8A1UnormSrgb = "etc2-rgb8a1unorm-srgb",
	ETC2RGBA8Unorm = "etc2-rgba8unorm",
	ETC2RGBA8UnormSrgb = "etc2-rgba8unorm-srgb",
	EACR11Unorm = "eac-r11unorm",
	EACR11Snorm = "eac-r11snorm",
	EACRG11Unorm = "eac-rg11unorm",
	EACRG11Snorm = "eac-rg11snorm",
	ASTC4x4Unorm = "astc-4x4-unorm",
	ASTC4x4UnormSrgb = "astc-4x4-unorm-srgb",
	ASTC5x4Unorm = "astc-5x4-unorm",
	ASTC5x4UnormSrgb = "astc-5x4-unorm-srgb",
	ASTC5x5Unorm = "astc-5x5-unorm",
	ASTC5x5UnormSrgb = "astc-5x5-unorm-srgb",
	ASTC6x5Unorm = "astc-6x5-unorm",
	ASTC6x5UnormSrgb = "astc-6x5-unorm-srgb",
	ASTC6x6Unorm = "astc-6x6-unorm",
	ASTC6x6UnormSrgb = "astc-6x6-unorm-srgb",
	ASTC8x5Unorm = "astc-8x5-unorm",
	ASTC8x5UnormSrgb = "astc-8x5-unorm-srgb",
	ASTC8x6Unorm = "astc-8x6-unorm",
	ASTC8x6UnormSrgb = "astc-8x6-unorm-srgb",
	ASTC8x8Unorm = "astc-8x8-unorm",
	ASTC8x8UnormSrgb = "astc-8x8-unorm-srgb",
	ASTC10x5Unorm = "astc-10x5-unorm",
	ASTC10x5UnormSrgb = "astc-10x5-unorm-srgb",
	ASTC10x6Unorm = "astc-10x6-unorm",
	ASTC10x6UnormSrgb = "astc-10x6-unorm-srgb",
	ASTC10x8Unorm = "astc-10x8-unorm",
	ASTC10x8UnormSrgb = "astc-10x8-unorm-srgb",
	ASTC10x10Unorm = "astc-10x10-unorm",
	ASTC10x10UnormSrgb = "astc-10x10-unorm-srgb",
	ASTC12x10Unorm = "astc-12x10-unorm",
	ASTC12x10UnormSrgb = "astc-12x10-unorm-srgb",
	ASTC12x12Unorm = "astc-12x12-unorm",
	ASTC12x12UnormSrgb = "astc-12x12-unorm-srgb",
}

export enum PrimitiveTopology {
	TriangleList = "triangle-list",
	TriangleStrip = "triangle-strip",
	PointList = "point-list",
	LineList = "line-list",
}

export type IndexFormat = "uint16" | "uint32";

export enum AddressMode {
	Repeat = "repeat",
	MirrorRepeat = "mirror-repeat",
	ClampToEdge = "clamp-to-edge",
}

export enum FilterMode {
	Nearest = "nearest",
	Linear = "linear",
}

export enum BufferBindingType {
	Uniform = "uniform",
	Storage = "storage",
	ReadOnlyStorage = "read-only-storage",
}

export enum SamplerBindingType {
	Filtering = "filtering",
	NonFiltering = "non-filtering",
	Comparison = "comparison",
}

export interface SamplerDesc {
	addressModeU?: AddressMode;
	addressModeV?: AddressMode;
	magFilter?: FilterMode;
	minFilter?: FilterMode;
	mipmapFilter?: FilterMode;
	compare?:
		| "less"
		| "always"
		| "never"
		| "equal"
		| "less-equal"
		| "greater"
		| "greater-equal";
	label?: string;
}

export interface BufferDesc {
	size: number;
	usage: BufferUsage;
	mappedAtCreation?: boolean;
	initialData?: BufferSource;
	label?: string;
}

export interface TextureDesc {
	width: number;
	height: number;
	depthOrArrayLayers?: number;
	dimension?: "1d" | "2d" | "3d";
	sampleCount?: number;
	format: TextureFormat;
	usage: TextureUsage;
	mipLevelCount?: number;
	viewFormats?: TextureFormat[];
	label?: string;
}

export interface ShaderModuleDesc {
	code: string;
	/** Optional source map used to map compiler diagnostics back to source segments */
	sourceMap?: ShaderSourceSegmentMap | null;
	/** Optional precomputed hash/fingerprint for shader module caching */
	codeHash?: string;
	/** Optional directive-stage fingerprint used for module cache partitioning */
	directiveFingerprint?: string;
	/** Optional language tag used for shader runtime validation/injection */
	language?: "glsl" | "wgsl";
	/** Optional stage tag used for shader runtime validation/injection */
	stage?: "vertex" | "fragment" | "compute" | "unknown";
	/** Optional entry point used for shader runtime validation/injection */
	entryPoint?: string;
	/** Optional source kind tag used for diagnostics/fallback policy */
	sourceKind?:
		| "builtin-scene"
		| "builtin-environment"
		| "builtin-present"
		| "postprocess"
		| "clustered"
		| "shadow"
		| "particle"
		| "decal"
		| "custom-material"
		| "unknown";
	label?: string;
	/** Optional variant key for diagnostics and warmup reporting */
	variantKey?: string;
	/** Optional material identifier for diagnostics and warmup reporting */
	materialId?: string;
	logCompilationInfo?: boolean;
}

export interface PipelineDesc {
	layout?: any;
	vertex: {
		module: IShaderModule;
		entryPoint: string;
		buffers?: VertexBufferLayout[];
	};
	fragment?: {
		module: IShaderModule;
		entryPoint: string;
		targets: ColorTargetState[];
	};
	primitive?: {
		topology?: PrimitiveTopology;
		cullMode?: "none" | "front" | "back";
		frontFace?: "ccw" | "cw";
	};
	depthStencil?: {
		format: TextureFormat;
		depthWriteEnabled: boolean;
		depthCompare:
			| "less"
			| "always"
			| "never"
			| "equal"
			| "less-equal"
			| "greater"
			| "greater-equal";
	};
	sampleCount?: number;
	label?: string;
}

export interface ComputePipelineDesc {
	compute: {
		module: IShaderModule;
		entryPoint: string;
	};
	layout?: any;
	label?: string;
}

export interface TextureDataLayout {
	offset?: number;
	bytesPerRow?: number;
	rowsPerImage?: number;
	mipLevel?: number;
}

export interface VertexBufferLayout {
	arrayStride: number;
	stepMode?: "vertex" | "instance";
	attributes: VertexAttribute[];
}

export interface VertexAttribute {
	format: VertexFormat;
	offset: number;
	shaderLocation: number;
}

export type VertexFormat =
	| "float32"
	| "float32x2"
	| "float32x3"
	| "float32x4"
	| "uint32"
	| "uint32x2"
	| "uint32x3"
	| "uint32x4"
	| "float16x2"
	| "snorm16x4"
	| "unorm16x4"
	| "unorm8x4";

export type BackendResourceHandle = unknown;

export interface ColorTargetState {
	format: TextureFormat;
	blend?: any;
	writeMask?: number;
}

export interface BindingGroupDesc {
	layout?: any;
	pipeline?: IRenderPipeline;
	layoutIndex?: number;
	entries: BindingEntry[];
	label?: string;
	/** @internal Set to false for owner-cached groups with unique resources. */
	cache?: boolean;
}

export type BindingResource =
	| IRenderBuffer
	| IRenderTexture
	| ISampler
	| GPUBindingResource
	| GPUTexture
	| GPUBuffer
	| {
			_gpuResource?: BackendResourceHandle;
	  };

export interface BindingEntry {
	binding: number;
	resource: BindingResource;
}

export interface IShaderModule {
	readonly label?: string;
}

export interface IRenderBuffer {
	readonly size: number;
	destroy(): void;
	unmap?(): void;
	/** Internal backend resource handle */
	_gpuResource?: BackendResourceHandle;
	_cpuData?: ArrayBuffer;
}

export interface IRenderTexture {
	readonly width: number;
	readonly height: number;
	readonly requestedFormat?: TextureFormat;
	readonly format?: TextureFormat;
	readonly formatFallbackReason?: string;
	destroy(): void;
	/** Internal backend resource handle */
	_gpuResource?: BackendResourceHandle;
	_cpuPixels?: Uint8ClampedArray;
}

export interface ISampler {
	readonly label?: string;
	_gpuResource?: BackendResourceHandle;
}

export interface IRenderPipeline {
	readonly label?: string;
	_gpuResource?: BackendResourceHandle;
}

export interface IBindingGroup {
	readonly label?: string;
	_gpuResource?: BackendResourceHandle;
}

export interface IComputePipeline {
	readonly label?: string;
	_gpuResource?: BackendResourceHandle;
}
