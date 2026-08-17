import type {
	CustomRenderPassResourceFacade,
} from "../../rendering/CustomRenderTargets";
import type {
	ReadTextureOptions,
	TextureReadbackResult,
	WriteTextureSize,
} from "../IComputeRuntime";
import type { ICommandEncoder } from "../ICommandEncoder";
import type {
	IRenderTexture,
	TextureDataLayout,
} from "../types";
import type {
	BackendExtensionAvailability,
	RenderBackendExtensionKey,
} from "../BackendExtensions";

export const WEBGL_AUXILIARY_RASTER_EXTENSION_ID =
	"webgl.auxiliary-raster";

export type WebGLAuxiliaryRasterFramePolicy =
	| "between-passes"
	| "idle-only";

export type WebGLAuxiliaryRasterContextLossPolicy =
	| "reject"
	| "retain-pending";

export interface WebGLAuxiliaryRasterRequirements {
	/** WebGL extensions that must all be present in the execution generation. */
	readonly requiredExtensions?: readonly string[];
	/** Extension groups for which at least one member of each group must exist. */
	readonly alternativeExtensionGroups?: readonly (readonly string[])[];
}

export interface WebGLAuxiliaryRasterAvailabilityOptions
	extends WebGLAuxiliaryRasterRequirements {
	readonly contextLossPolicy?: WebGLAuxiliaryRasterContextLossPolicy;
}

export type WebGLAuxiliaryUniformScalarType = "f32" | "i32" | "u32";
export type WebGLAuxiliaryUniformVectorType =
	| "vec2f"
	| "vec3f"
	| "vec4f"
	| "vec2i"
	| "vec3i"
	| "vec4i"
	| "vec2u"
	| "vec3u"
	| "vec4u";
export type WebGLAuxiliaryUniformMatrixType = "mat3x3f" | "mat4x4f";
export type WebGLAuxiliaryUniformType =
	| WebGLAuxiliaryUniformScalarType
	| WebGLAuxiliaryUniformVectorType
	| WebGLAuxiliaryUniformMatrixType;

export interface WebGLAuxiliaryUniform {
	readonly name: string;
	readonly type: WebGLAuxiliaryUniformType;
	/** Scalars accept one number; arrays and aggregate values use flat data. */
	readonly value: number | ArrayLike<number>;
}

export interface IWebGLAuxiliaryRasterEncoder extends ICommandEncoder {
	/** Sets the viewport for subsequent draw calls in the active render pass. */
	setViewport(x: number, y: number, width: number, height: number): void;
	/** Writes named scalar, vector, matrix, or flat uniform-array values. */
	setUniforms(uniforms: readonly WebGLAuxiliaryUniform[]): void;
}

export interface WebGLAuxiliaryRasterResourceFacade
	extends CustomRenderPassResourceFacade {
	/** Uploads one tightly packed 2D texture mip owned by the active scope. */
	writeTexture(
		texture: IRenderTexture,
		data: BufferSource,
		layout: TextureDataLayout,
		size: WriteTextureSize,
	): void;
	/** Reads a scope-owned color texture into backend-independent CPU data. */
	readTexture(options: ReadTextureOptions): Promise<TextureReadbackResult>;
}

export interface WebGLAuxiliaryRasterContext {
	readonly generation: number;
	readonly signal: AbortSignal;
	readonly encoder: IWebGLAuxiliaryRasterEncoder;
	readonly resources: WebGLAuxiliaryRasterResourceFacade;
}

export interface WebGLAuxiliaryRasterRequest<T>
	extends WebGLAuxiliaryRasterRequirements {
	readonly label: string;
	readonly framePolicy?: WebGLAuxiliaryRasterFramePolicy;
	readonly contextLossPolicy?: WebGLAuxiliaryRasterContextLossPolicy;
	readonly signal?: AbortSignal | null;
	readonly task: (
		context: WebGLAuxiliaryRasterContext,
	) => T | Promise<T>;
}

export interface IWebGLAuxiliaryRasterFacade {
	/** Reports whether a request with the supplied policies can be accepted. */
	getAvailability(
		options?: WebGLAuxiliaryRasterAvailabilityOptions,
	): BackendExtensionAvailability;
	/** Schedules one isolated raster task against the active context generation. */
	execute<T>(request: WebGLAuxiliaryRasterRequest<T>): Promise<T>;
}

export const WEBGL_AUXILIARY_RASTER_EXTENSION: RenderBackendExtensionKey<
	IWebGLAuxiliaryRasterFacade
> = {
	id: WEBGL_AUXILIARY_RASTER_EXTENSION_ID,
};
