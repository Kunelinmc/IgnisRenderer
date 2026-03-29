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
	RGBA8Unorm = "rgba8unorm",
	BGRA8Unorm = "bgra8unorm",
	RGBA16Float = "rgba16float",
	Depth24Plus = "depth24plus",
	Depth32Float = "depth32float",
	Depth24PlusStencil8 = "depth24plus-stencil8",
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
	/** Optional language tag used for shader runtime validation/injection */
	language?: "glsl" | "wgsl";
	/** Optional stage tag used for shader runtime validation/injection */
	stage?: "vertex" | "fragment" | "compute" | "unknown";
	/** Optional entry point used for shader runtime validation/injection */
	entryPoint?: string;
	/** Optional source kind tag used for diagnostics/fallback policy */
	sourceKind?:
		| "builtin-scene"
		| "builtin-skybox"
		| "builtin-present"
		| "postprocess"
		| "clustered"
		| "shadow"
		| "particle"
		| "custom-material"
		| "unknown";
	label?: string;
	/** Optional variant key for diagnostics and warmup reporting */
	variantKey?: string;
	/** Optional material identifier for diagnostics and warmup reporting */
	materialId?: string;
	logCompilationInfo?: boolean;
	/** Optional software implementation of the shader */
	softwareDelegate?: Function;
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
