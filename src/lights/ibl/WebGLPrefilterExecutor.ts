import type { Texture, TextureData } from "../../core/Texture";
import { float32ToFloat16Bits } from "../../foundation/Float16";
import {
	AddressMode,
	BufferUsage,
	FilterMode,
	PrimitiveTopology,
	TextureFormat,
	TextureUsage,
} from "../../backends/types";
import type { IWebGLAuxiliaryRasterFacade } from
	"../../backends/webgl/WebGLAuxiliaryRaster";
import { ShaderSource } from "../../shaders/ShaderSource";
import {
	assertIBLPrefilterSourceRevision,
	type IBLPrefilterExecutionRequest,
	type IBLPrefilterExecutorAvailability,
	type IBLPrefilterExecutorLike,
	type IBLPrefilterMipData,
} from "./IBLPrefilterExecutor";

const REQUIRED_EXTENSIONS = ["EXT_color_buffer_float"] as const;
const FLOAT_LINEAR_ALTERNATIVES = [
	"OES_texture_float_linear",
	"OES_texture_half_float_linear",
] as const;
const GPU_MAX_SAMPLE_COUNT = 256;
const GPU_MIN_SAMPLE_COUNT = 48;

/** Lighting-owned WebGL executor backed by generic auxiliary raster work. */
export class WebGLPrefilterExecutor implements IBLPrefilterExecutorLike {
	public readonly id = "webgl" as const;
	private readonly _raster: IWebGLAuxiliaryRasterFacade;

	public constructor(raster: IWebGLAuxiliaryRasterFacade) {
		this._raster = raster;
	}

	public getAvailability(): IBLPrefilterExecutorAvailability {
		return this._raster.getAvailability({
			contextLossPolicy: "retain-pending",
			requiredExtensions: REQUIRED_EXTENSIONS,
			alternativeExtensionGroups: [FLOAT_LINEAR_ALTERNATIVES],
		});
	}

	public execute(
		request: IBLPrefilterExecutionRequest,
	): Promise<IBLPrefilterMipData[]> {
		assertIBLPrefilterSourceRevision(
			request.envMap,
			request.sourceRevision,
		);
		return this._raster.execute({
			label: "ibl-prefilter",
			framePolicy: "between-passes",
			contextLossPolicy: "retain-pending",
			signal: request.signal,
			requiredExtensions: REQUIRED_EXTENSIONS,
			alternativeExtensionGroups: [FLOAT_LINEAR_ALTERNATIVES],
			task: async ({ encoder, resources, signal }) => {
				assertIBLPrefilterSourceRevision(
					request.envMap,
					request.sourceRevision,
				);
				assertNotAborted(signal);
				const sourceFormat = request.envMap.colorSpace === "sRGB" ?
					TextureFormat.RGBA8Unorm : TextureFormat.RGBA16Float;
				const sourceTexture = resources.createTexture({
					width: request.envMap.width,
					height: request.envMap.height,
					format: sourceFormat,
					usage: TextureUsage.TextureBinding | TextureUsage.CopyDst,
					mipLevelCount: resolveSourceMipLevelCount(request.envMap),
					label: "IBLPrefilterSource",
				});
				const sourceMipLevelCount = uploadSourceTexture(
					resources,
					sourceTexture,
					request.envMap,
					sourceFormat,
				);
				const sampler = resources.createSampler({
					addressModeU: AddressMode.Repeat,
					addressModeV: AddressMode.ClampToEdge,
					magFilter: FilterMode.Linear,
					minFilter: FilterMode.Linear,
					mipmapFilter: FilterMode.Linear,
					label: "IBLPrefilterSampler",
				});
				const vertex = await resources.createShaderModule({
					stage: "vertex",
					language: "glsl",
					code: (await ShaderSource.load("webgl.part.environmentVertex"))
						.source.code,
					sourceKind: "builtin-environment",
					label: "IBLPrefilterVertex",
				});
				const fragment = await resources.createShaderModule({
					stage: "fragment",
					language: "glsl",
					code: (await ShaderSource.load("webgl.part.iblPrefilterFragment"))
						.source.code,
					sourceKind: "builtin-environment",
					label: "IBLPrefilterFragment",
				});
				const pipeline = await resources.createRenderPipeline({
					vertex: { module: vertex, entryPoint: "main", buffers: [] },
					fragment: {
						module: fragment,
						entryPoint: "main",
						targets: [{ format: TextureFormat.RGBA16Float }],
					},
					primitive: {
						topology: PrimitiveTopology.TriangleList,
						cullMode: "none",
					},
					label: "IBLPrefilterPipeline",
				});
				const bindingGroup = resources.createBindingGroup({
					pipeline,
					layoutIndex: 0,
					entries: [
						{ binding: 0, resource: sourceTexture },
						{ binding: 0, resource: sampler },
					],
					label: "IBLPrefilterBindings",
				});
				const result: IBLPrefilterMipData[] = [];
				for (const mip of request.plan.mipLevels) {
					assertNotAborted(signal);
					const output = resources.createTexture({
						width: mip.width,
						height: mip.height,
						format: TextureFormat.RGBA16Float,
						usage: TextureUsage.RenderAttachment | TextureUsage.CopySrc,
						label: `IBLPrefilterOutput_mip${mip.level}`,
					});
					encoder.beginRenderPass({
						colorAttachments: [{
							view: output,
							loadOp: "clear",
							storeOp: "store",
							clearValue: { r: 0, g: 0, b: 0, a: 1 },
						}],
					});
					encoder.setPipeline(pipeline);
					encoder.setBindingGroup(0, bindingGroup);
					encoder.setViewport(0, 0, mip.width, mip.height);
					encoder.setUniforms([
						{ name: "uEnvironmentMap", type: "i32", value: 0 },
						{ name: "uOutputSize", type: "vec2f", value: [mip.width, mip.height] },
						{ name: "uSourceSize", type: "vec2f", value: [request.envMap.width, request.envMap.height] },
						{ name: "uRoughness", type: "f32", value: mip.roughness },
						{ name: "uSampleCount", type: "i32", value: resolveSampleCount(mip.roughness) },
						{ name: "uSourceIsLinear", type: "i32", value: request.envMap.colorSpace === "sRGB" ? 0 : 1 },
						{ name: "uSourceMipLevelCount", type: "i32", value: sourceMipLevelCount },
					]);
					encoder.draw(3);
					encoder.endRenderPass();
					const readback = await resources.readTexture({
						texture: output,
						width: mip.width,
						height: mip.height,
						format: TextureFormat.RGBA16Float,
					});
					const data = readback.toRGBAFloat32();
					for (let index = 3; index < data.length; index += 4) data[index] = 1;
					result.push({ ...mip, data });
					request.onMipComplete?.(mip.level);
				}
				return result;
			},
		});
	}
}

function uploadSourceTexture(
	resources: import("../../backends/webgl/WebGLAuxiliaryRaster").WebGLAuxiliaryRasterResourceFacade,
	texture: import("../../backends/types").IRenderTexture,
	envMap: Texture,
	format: TextureFormat,
): number {
	const count = resolveSourceMipLevelCount(envMap);
	for (let level = 0; level < count; level++) {
		const descriptor = envMap.getMipLevelDescriptor(level);
		const width = Math.max(1, descriptor?.width ?? envMap.width >> level);
		const height = Math.max(1, descriptor?.height ?? envMap.height >> level);
		const source = descriptor?.data ?? envMap.mipmaps[level] ?? envMap.data;
		if (!source) {
			throw new Error(`WebGL IBL prefilter source mip ${level} has no pixel data.`);
		}
		resources.writeTexture(
			texture,
			(format === TextureFormat.RGBA16Float ?
				toRGBA16F(source, width, height) : toRGBA8(source, width, height)
			) as unknown as BufferSource,
			{ mipLevel: level },
			{ width, height, depthOrArrayLayers: 1 },
		);
	}
	return count;
}

function resolveSourceMipLevelCount(envMap: Texture): number {
	const declared = Math.max(1, envMap.levels.length, envMap.mipmaps.length);
	const natural = Math.floor(
		Math.log2(Math.max(1, envMap.width, envMap.height)),
	) + 1;
	return Math.min(declared, natural);
}

function toRGBA16F(source: TextureData, width: number, height: number): Uint16Array {
	const result = new Uint16Array(Math.max(1, width * height * 4));
	for (let index = 0; index < result.length; index++) {
		const fallback = (index & 3) === 3 ? 1 : 0;
		const raw = source[index];
		const value = raw === undefined ? fallback :
			source instanceof Float32Array ? raw : raw / 255;
		result[index] = float32ToFloat16Bits(value);
	}
	return result;
}

function toRGBA8(source: TextureData, width: number, height: number): Uint8Array {
	const result = new Uint8Array(Math.max(1, width * height * 4));
	for (let index = 0; index < result.length; index++) {
		const fallback = (index & 3) === 3 ? 255 : 0;
		const raw = source[index];
		result[index] = raw === undefined ? fallback :
			source instanceof Float32Array ?
				Math.round(Math.max(0, Math.min(1, raw)) * 255) : raw;
	}
	return result;
}

function resolveSampleCount(roughness: number): number {
	const clamped = Math.max(0, Math.min(1, roughness));
	return Math.max(
		GPU_MIN_SAMPLE_COUNT,
		Math.min(
			GPU_MAX_SAMPLE_COUNT,
			Math.floor(
				GPU_MAX_SAMPLE_COUNT +
					(GPU_MIN_SAMPLE_COUNT - GPU_MAX_SAMPLE_COUNT) * clamped,
			),
		),
	);
}

function assertNotAborted(signal?: AbortSignal | null): void {
	if (!signal?.aborted) return;
	const error = new Error("IBL prefilter was aborted", { cause: signal.reason });
	error.name = "AbortError";
	throw error;
}
