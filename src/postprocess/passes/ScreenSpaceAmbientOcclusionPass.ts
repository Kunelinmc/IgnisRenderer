import { CameraType } from "../../cameras/Camera";
import type { OrthographicCamera } from "../../cameras/OrthographicCamera";
import type { IVector3 } from "../../maths/types";
import { Vector3 } from "../../maths/Vector3";
import {
	type FrameAttachments,
	type FrameContext,
} from "../../pipeline/types";
import type { ICommandEncoder } from "../../backends/ICommandEncoder";
import {
	BufferUsage,
	type IComputePipeline,
	type IRenderBuffer,
	type IRenderTexture,
	type IShaderModule,
} from "../../backends/types";
import {
	WEBGPU_2D_COMPUTE_WORKGROUP_SIZE as WORKGROUP_SIZE,
} from "../../backends/webgpu/constants";
import {
	type WebGPUPostProcessFrameTargets,
	type WebGPUPostProcessServices,
} from "../../backends/webgpu/WebGPUPostProcessContracts";
import type {
	WebGLProgramCompiler,
	WebGLProgramSlot,
} from "../../backends/webgl/WebGLProgramCompiler";
import { ceilDiv } from "../../maths/Misc";
import { ShaderSource } from "../../shaders/ShaderSource";
import {
	PostProcessPass,
	type PostProcessPassConfig,
	type PostProcessPassResolveRequest,
} from "../PostProcessPass";
import type { PostProcessScheduleEntry } from "../ordering";
import {
	POST_PROCESS_COLOR_ATTACHMENT_WRITE,
	POST_PROCESS_CPU_READ,
	POST_PROCESS_SAMPLED_READ,
	POST_PROCESS_STORAGE_WRITE,
	SOFTWARE_IN_PLACE_EXECUTION,
	WEBGL_VERSIONED_EXECUTION,
	WEBGPU_VERSIONED_EXECUTION,
} from "../executionDeclarations";
import type {
	PostProcessExecutionDeclaration,
	PostProcessPassImplementation,
	PostProcessPassRequest,
	PostProcessPassResult,
	PostProcessResourceAccessor,
	PostProcessTransientDescriptor,
} from "../types";
import {
	forEachSoftwareDirtyRect,
	type IncrementalDirtyRect,
} from "./ScreenPassShared";

const SSAO_NOISE_SIZE = 4;
const SSAO_SOFTWARE_MAX_SAMPLES = 48;
export const SCREEN_SPACE_AMBIENT_OCCLUSION_PASS_ID = "ssao";
export const SCREEN_SPACE_AMBIENT_OCCLUSION_PASS_ORDER = {
	id: SCREEN_SPACE_AMBIENT_OCCLUSION_PASS_ID,
	placement: "spatial",
	order: 100,
	incremental: {
		firstPass: "ssao",
		grade: "standard",
		inflationRadius: 8,
	},
} as const satisfies PostProcessScheduleEntry;
const SSAO_RAW_TRANSIENT_ID = "ssao:raw";
const SSAO_BLUR_TRANSIENT_ID = "ssao:blur";

export interface SSAOOptions {
	/** Ambient-occlusion sample count, rounded and clamped to backend limits. */
	samples?: number;
	/** View-space sampling radius. Larger values capture wider contact shadows. */
	radius?: number;
	/** Depth bias that suppresses self-occlusion acne near flat surfaces. */
	bias?: number;
	/** Multiplier for the darkening applied by ambient occlusion. */
	intensity?: number;
	/** Internal AO buffer scale divisor. Higher values improve speed. */
	downsample?: number;
	/** Bilateral blur radius in pixels for smoothing noisy AO. */
	blurRadius?: number;
	/** Depth edge sharpness for the bilateral blur. Higher values preserve edges. */
	blurSharpness?: number;
	/** Allows backend-specific experimental SSAO options. */
	[key: string]: unknown;
}

export const DEFAULT_SSAO_OPTIONS: Required<
	Pick<
		SSAOOptions,
		| "samples"
		| "radius"
		| "bias"
		| "intensity"
		| "downsample"
		| "blurRadius"
		| "blurSharpness"
	>
> = {
	samples: 16,
	radius: 8,
	bias: 0.1,
	intensity: 1,
	downsample: 2,
	blurRadius: 2,
	blurSharpness: 8,
};

export type ResolvedSSAOOptions = Required<
	Pick<
		SSAOOptions,
		| "samples"
		| "radius"
		| "bias"
		| "intensity"
		| "downsample"
		| "blurRadius"
		| "blurSharpness"
	>
>;

/** @internal Software context supplied to the built-in SSAO implementation. */
export interface SoftwareSSAOContext {
	readonly attachments: FrameAttachments;
	readonly dirtyRects: readonly IncrementalDirtyRect[];
	readonly resources: PostProcessResourceAccessor<ArrayBufferView>;
}

/** @internal WebGPU context supplied to the built-in SSAO implementation. */
export interface WebGPUSSAOContext {
	readonly encoder?: ICommandEncoder;
	readonly targets?: WebGPUPostProcessFrameTargets;
	readonly shared: WebGPUPostProcessServices;
	readonly resources: PostProcessResourceAccessor<IRenderTexture>;
}

/** @internal WebGL context supplied to the built-in SSAO implementation. */
export interface WebGLSSAOContext {
	readonly gl: WebGL2RenderingContext;
	readonly programCompiler: WebGLProgramCompiler;
	readonly fullscreenVao: WebGLVertexArrayObject | null;
	readonly postFramebuffer: WebGLFramebuffer | null;
	readonly sceneColorTexture: WebGLTexture | null;
	readonly width: number;
	readonly height: number;
	readonly resources: PostProcessResourceAccessor<WebGLTexture>;
	getSourceTexture(): WebGLTexture | null;
	bindColorTarget(texture: WebGLTexture): void;
	drawFullscreen(
		width: number,
		height: number,
		frameContext: FrameContext | null
	): void;
}

interface WebGLSSAORawProgram {
	readonly program: WebGLProgram;
	readonly uniforms: {
		readonly normalMap: WebGLUniformLocation | null;
		readonly depthMap: WebGLUniformLocation | null;
		readonly invSize: WebGLUniformLocation | null;
		readonly gtao: WebGLUniformLocation | null;
		readonly blurProj: WebGLUniformLocation | null;
		readonly pass: WebGLUniformLocation | null;
		readonly cameraPosition: WebGLUniformLocation | null;
		readonly basisRight: WebGLUniformLocation | null;
		readonly basisUp: WebGLUniformLocation | null;
		readonly basisBackward: WebGLUniformLocation | null;
	};
}

interface WebGLSSAOBlurProgram {
	readonly program: WebGLProgram;
	readonly uniforms: {
		readonly sourceMap: WebGLUniformLocation | null;
		readonly depthMap: WebGLUniformLocation | null;
		readonly invSize: WebGLUniformLocation | null;
		readonly blurProj: WebGLUniformLocation | null;
		readonly pass: WebGLUniformLocation | null;
	};
}

interface WebGLSSAOCombineProgram {
	readonly program: WebGLProgram;
	readonly uniforms: {
		readonly sceneColor: WebGLUniformLocation | null;
		readonly aoMap: WebGLUniformLocation | null;
		readonly invSize: WebGLUniformLocation | null;
	};
}

interface WebGPUSSAOResources {
	shared: WebGPUPostProcessServices;
	module: IShaderModule | null;
	rawPipeline: IComputePipeline | null;
	combinePipeline: IComputePipeline | null;
	params: IRenderBuffer | null;
	frameIndex: number;
}

function finiteClamped(
	value: unknown,
	fallback: number,
	min: number,
	max: number
): number {
	const resolved =
		typeof value === "number" && Number.isFinite(value) ? value : fallback;
	return Math.min(max, Math.max(min, resolved));
}

/**
 * Resolves SSAO options with the same numeric ranges for every backend.
 *
 * @param options User-provided SSAO options.
 * @returns Fully resolved SSAO options.
 * @sideEffects None.
 */
export function resolveSSAOOptions(
	options?: SSAOOptions | null
): ResolvedSSAOOptions {
	return {
		samples: Math.round(
			finiteClamped(options?.samples, DEFAULT_SSAO_OPTIONS.samples, 4, 48)
		),
		radius: finiteClamped(
			options?.radius,
			DEFAULT_SSAO_OPTIONS.radius,
			1,
			Number.POSITIVE_INFINITY
		),
		bias: finiteClamped(
			options?.bias,
			DEFAULT_SSAO_OPTIONS.bias,
			1e-4,
			Number.POSITIVE_INFINITY
		),
		intensity: finiteClamped(
			options?.intensity,
			DEFAULT_SSAO_OPTIONS.intensity,
			0,
			Number.POSITIVE_INFINITY
		),
		downsample: resolveSSAODownsample(options?.downsample),
		blurRadius: Math.round(
			finiteClamped(
				options?.blurRadius,
				DEFAULT_SSAO_OPTIONS.blurRadius,
				1,
				4
			)
		),
		blurSharpness: finiteClamped(
			options?.blurSharpness,
			DEFAULT_SSAO_OPTIONS.blurSharpness,
			1e-3,
			Number.POSITIVE_INFINITY
		),
	};
}

/**
 * Resolves the SSAO downsample factor shared by frame target allocation.
 *
 * @param value User-provided downsample factor.
 * @returns Integer factor clamped to `[1, 8]`.
 * @sideEffects None.
 */
export function resolveSSAODownsample(value: unknown): number {
	if (typeof value !== "number" || !Number.isFinite(value)) {
		return DEFAULT_SSAO_OPTIONS.downsample;
	}
	return Math.min(8, Math.max(1, Math.floor(value)));
}

/**
 * Creates packed SSAO shader parameters.
 *
 * @param width Full-resolution target width.
 * @param height Full-resolution target height.
 * @param aoWidth Ambient occlusion target width.
 * @param aoHeight Ambient occlusion target height.
 * @param options Resolved SSAO options.
 * @param camera Current frame camera.
 * @param blurDirX Horizontal blur direction.
 * @param blurDirY Vertical blur direction.
 * @param frameJitter Temporal noise phase.
 * @returns Sixteen float parameters expected by the SSAO kernels.
 * @sideEffects None.
 */
export function createSSAOKernelParams(
	width: number,
	height: number,
	aoWidth: number,
	aoHeight: number,
	options: ResolvedSSAOOptions,
	camera: FrameContext["viewCamera"],
	blurDirX: number,
	blurDirY: number,
	frameJitter: number
): Float32Array {
	const isOrthographic = camera.type === CameraType.Orthographic;
	const tanHalfFov =
		isOrthographic ? 0 : Math.tan((camera.fov * Math.PI) / 360);
	const aspect =
		camera.aspectRatio || Math.max(width, 1) / Math.max(height, 1);
	return new Float32Array([
		1 / Math.max(width, 1),
		1 / Math.max(height, 1),
		1 / Math.max(aoWidth, 1),
		1 / Math.max(aoHeight, 1),
		options.radius,
		options.bias,
		options.intensity,
		options.samples,
		options.blurRadius,
		options.blurSharpness,
		tanHalfFov,
		aspect,
		blurDirX,
		blurDirY,
		isOrthographic ? 1 : 0,
		frameJitter,
	]);
}

/**
 * CPU implementation of the cross-backend SSAO pass.
 */
/** @internal Software implementation for the built-in SSAO pass. */
export class SoftwareScreenSpaceAmbientOcclusionImplementation
	implements PostProcessPassImplementation<SoftwareSSAOContext>
{
	public readonly id = "ssao:software";
	public describeExecution() {
		return {
			...SOFTWARE_IN_PLACE_EXECUTION,
			gBuffer: (["depth", "normal"] as const).map((semantic) => ({
				semantic,
				...POST_PROCESS_CPU_READ,
			})),
		} satisfies PostProcessExecutionDeclaration;
	}
	private _kernel: IVector3[] = [];
	private _noise: IVector3[] = [];
	private _aoBuffer: Float32Array | null = null;
	private _blurTemp: Float32Array | null = null;

	constructor() {
		this._initKernel();
	}

	public execute(
		request: PostProcessPassRequest,
		context: SoftwareSSAOContext | undefined
	): PostProcessPassResult {
		if (!context) {
			return { ran: false };
		}
		return this._runSSAOKernel(request, context);
	}

	private _runSSAOKernel(
		request: PostProcessPassRequest,
		context: SoftwareSSAOContext
	): PostProcessPassResult {
		const frameContext = request.frameContext;
		const pixels = context.resources.color.input;
		const depthBuffer = context.attachments.depthBuffer;
		const normalBuffer = context.attachments.normalBuffer;
		if (!(pixels instanceof Float32Array) || !depthBuffer || !normalBuffer) {
			return { ran: false };
		}
		const dirtyRects = context.dirtyRects;
		if (dirtyRects.length === 0) {
			return { ran: false };
		}

		const width = context.attachments.width;
		const height = context.attachments.height;
		const options = resolveSSAOOptions(request.options as SSAOOptions);
		const pixelCount = width * height;
		if (!this._aoBuffer || this._aoBuffer.length !== pixelCount) {
			this._aoBuffer = new Float32Array(pixelCount);
		}
		const aoBuffer = this._aoBuffer;
		const camera = frameContext.viewCamera;
		const projection = camera.projectionMatrix.elements;

		forEachSoftwareDirtyRect(dirtyRects, (rect) => {
			for (let y = rect.minY; y <= rect.maxY; y++) {
				for (let x = rect.minX; x <= rect.maxX; x++) {
					const idx = y * width + x;
					const originDepth = depthBuffer[idx];
					if (originDepth === Infinity || originDepth <= 0) {
						aoBuffer[idx] = 1;
						continue;
					}

					const ndcX = (x / width) * 2 - 1;
					const ndcY = 1 - (y / height) * 2;
					const posView = reconstructViewPos(ndcX, ndcY, originDepth, camera);

					const nIdx = idx * 3;
					const normal = {
						x: normalBuffer[nIdx],
						y: normalBuffer[nIdx + 1],
						z: normalBuffer[nIdx + 2],
					};
					const noiseIdx =
						(y % SSAO_NOISE_SIZE) * SSAO_NOISE_SIZE +
						(x % SSAO_NOISE_SIZE);
					const randomVec = this._noise[noiseIdx];
					const randomNormalDot = Vector3.dot(randomVec, normal);
					const tangent = {
						x: randomVec.x - normal.x * randomNormalDot,
						y: randomVec.y - normal.y * randomNormalDot,
						z: randomVec.z - normal.z * randomNormalDot,
					};
					const tangentLen = Math.hypot(tangent.x, tangent.y, tangent.z) || 1;
					tangent.x /= tangentLen;
					tangent.y /= tangentLen;
					tangent.z /= tangentLen;

					const bitangent = Vector3.cross(normal, tangent);
					const tbn = [
						[tangent.x, bitangent.x, normal.x],
						[tangent.y, bitangent.y, normal.y],
						[tangent.z, bitangent.z, normal.z],
					];

					let occlusion = 0;
					for (let i = 0; i < options.samples; i++) {
						const sample = this._kernel[i];
						const sampleViewOffset = {
							x:
								tbn[0][0] * sample.x +
								tbn[0][1] * sample.y +
								tbn[0][2] * sample.z,
							y:
								tbn[1][0] * sample.x +
								tbn[1][1] * sample.y +
								tbn[1][2] * sample.z,
							z:
								tbn[2][0] * sample.x +
								tbn[2][1] * sample.y +
								tbn[2][2] * sample.z,
						};
						const samplePos = {
							x: posView.x + sampleViewOffset.x * options.radius,
							y: posView.y + sampleViewOffset.y * options.radius,
							z: posView.z + sampleViewOffset.z * options.radius,
						};

						let screenX: number;
						let screenY: number;
						if (camera.type === CameraType.Orthographic) {
							const sx = projection[0][0] * samplePos.x + projection[0][3];
							const sy = projection[1][1] * samplePos.y + projection[1][3];
							screenX = Math.round((sx * 0.5 + 0.5) * width - 0.5);
							screenY = Math.round((0.5 - sy * 0.5) * height - 0.5);
						} else {
							const offsetNDC = {
								x:
									(projection[0][0] * samplePos.x +
										projection[0][2] * samplePos.z) /
									-samplePos.z,
								y:
									(projection[1][1] * samplePos.y +
										projection[1][2] * samplePos.z) /
									-samplePos.z,
							};
							screenX = Math.round((offsetNDC.x * 0.5 + 0.5) * width - 0.5);
							screenY = Math.round((0.5 - offsetNDC.y * 0.5) * height - 0.5);
						}

						if (
							screenX >= 0 &&
							screenX < width &&
							screenY >= 0 &&
							screenY < height
						) {
							const sampleDepth = depthBuffer[screenY * width + screenX];
							const samplePosDepth = -samplePos.z;
							const rangeCheck =
								Math.abs(originDepth - sampleDepth) < options.radius ? 1 : 0;
							occlusion +=
								(sampleDepth <= samplePosDepth - options.bias ? 1 : 0) *
								rangeCheck;
						}
					}

					aoBuffer[idx] =
						1 - (occlusion / Math.max(options.samples, 1)) * options.intensity;
				}
			}
		});

		this._blur(aoBuffer, width, height, dirtyRects, options.blurRadius);
		forEachSoftwareDirtyRect(dirtyRects, (rect) => {
			for (let y = rect.minY; y <= rect.maxY; y++) {
				const row = y * width;
				for (let x = rect.minX; x <= rect.maxX; x++) {
					const i = row + x;
					const factor = aoBuffer[i];
					const idx = i << 2;
					pixels[idx] *= factor;
					pixels[idx + 1] *= factor;
					pixels[idx + 2] *= factor;
				}
			}
		});
		return { ran: true };
	}

	private _initKernel(): void {
		for (let i = 0; i < SSAO_SOFTWARE_MAX_SAMPLES; i++) {
			const sample = {
				x: Math.random() * 2 - 1,
				y: Math.random() * 2 - 1,
				z: Math.random(),
			};
			const len = Math.hypot(sample.x, sample.y, sample.z) || 1;
			sample.x /= len;
			sample.y /= len;
			sample.z /= len;

			let scale = i / SSAO_SOFTWARE_MAX_SAMPLES;
			scale = 0.1 + scale * scale * (1 - 0.1);
			sample.x *= scale;
			sample.y *= scale;
			sample.z *= scale;
			this._kernel.push(sample);
		}

		for (let i = 0; i < SSAO_NOISE_SIZE * SSAO_NOISE_SIZE; i++) {
			const noise = {
				x: Math.random() * 2 - 1,
				y: Math.random() * 2 - 1,
				z: 0,
			};
			const len = Math.hypot(noise.x, noise.y, noise.z) || 1;
			this._noise.push({ x: noise.x / len, y: noise.y / len, z: 0 });
		}
	}

	private _blur(
		buffer: Float32Array,
		width: number,
		height: number,
		dirtyRects: readonly IncrementalDirtyRect[],
		blurRadius: number
	): void {
		if (!this._blurTemp || this._blurTemp.length !== buffer.length) {
			this._blurTemp = new Float32Array(buffer.length);
		}
		const temp = this._blurTemp;
		temp.set(buffer);
		const radius = Math.max(1, Math.min(4, Math.round(blurRadius)));
		forEachSoftwareDirtyRect(dirtyRects, (rect) => {
			for (let y = rect.minY; y <= rect.maxY; y++) {
				for (let x = rect.minX; x <= rect.maxX; x++) {
					let sum = 0;
					let count = 0;
					for (let dy = -radius; dy <= radius; dy++) {
						for (let dx = -radius; dx <= radius; dx++) {
							const nx = x + dx;
							const ny = y + dy;
							if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
								sum += temp[ny * width + nx];
								count++;
							}
						}
					}
					buffer[y * width + x] = sum / Math.max(count, 1);
				}
			}
		});
	}
}

/**
 * WebGPU implementation of the cross-backend SSAO pass.
 */
/** @internal WebGPU implementation for the built-in SSAO pass. */
export class WebGPUScreenSpaceAmbientOcclusionImplementation
	implements PostProcessPassImplementation<WebGPUSSAOContext>
{
	public readonly id = "ssao:webgpu";
	public describeExecution(request: PostProcessPassResolveRequest<ResolvedSSAOOptions>) {
		const options = resolveSSAOOptions(request.options);
		const scale = 1 / options.downsample;
		return {
			...WEBGPU_VERSIONED_EXECUTION,
			gBuffer: (["depth", "normal"] as const).map((semantic) => ({
				semantic,
				...POST_PROCESS_SAMPLED_READ,
			})),
			transients: [
				{
					id: SSAO_RAW_TRANSIENT_ID,
					widthScale: scale,
					heightScale: scale,
					format: "rgba16float",
				},
				{
					id: SSAO_BLUR_TRANSIENT_ID,
					widthScale: scale,
					heightScale: scale,
					format: "rgba16float",
				},
			].map((descriptor) => ({
				descriptor,
				uses: [
					POST_PROCESS_STORAGE_WRITE,
					POST_PROCESS_SAMPLED_READ,
				],
			})),
		} satisfies PostProcessExecutionDeclaration;
	}
	private _resources = new Map<WebGPUPostProcessServices, WebGPUSSAOResources>();

	public async warmup(context: WebGPUSSAOContext | undefined): Promise<void> {
		if (context) {
			await this._ensureResources(context.shared);
		}
	}

	public async execute(
		request: PostProcessPassRequest,
		context: WebGPUSSAOContext | undefined
	): Promise<PostProcessPassResult> {
		if (!context?.encoder || !context.targets) {
			return { ran: false };
		}
		const ran = await this._runSSAOKernel(request, context);
		return ran ? { ran: true } : { ran: false };
	}

	public invalidate(): void {
		for (const resources of this._resources.values()) {
			resources.shared.invalidateBindingsByPrefix("ssao-");
		}
	}

	public destroy(): void {
		for (const resources of this._resources.values()) {
			resources.shared.destroyManagedResource(
				resources.rawPipeline,
				"SSAO raw pipeline"
			);
			resources.shared.destroyManagedResource(
				resources.combinePipeline,
				"SSAO combine pipeline"
			);
			resources.shared.destroyManagedResource(
				resources.module,
				"SSAO shader module"
			);
			resources.shared.destroyManagedResource(
				resources.params,
				"SSAO params buffer"
			);
			resources.shared.invalidateBindingsByPrefix("ssao-");
			resources.module = null;
			resources.rawPipeline = null;
			resources.combinePipeline = null;
			resources.params = null;
		}
		this._resources.clear();
	}

	private async _runSSAOKernel(
		request: PostProcessPassRequest,
		context: WebGPUSSAOContext
	): Promise<boolean> {
		const resources = await this._ensureResources(context.shared);
		const aoRaw = context.resources.getTransient(SSAO_RAW_TRANSIENT_ID);
		const aoBlur = context.resources.getTransient(SSAO_BLUR_TRANSIENT_ID);
		const depthTexture = context.resources.getGBuffer("depth");
		const normalTexture = context.resources.getGBuffer("normal");
		const input = context.resources.color.input;
		if (
			!context.encoder ||
			!context.targets ||
			!aoRaw ||
			!aoBlur ||
			!depthTexture ||
			!normalTexture ||
			!input ||
			!context.shared.sampler ||
			!resources.rawPipeline ||
			!resources.combinePipeline ||
			!resources.params
		) {
			return false;
		}

		const targets = context.targets;
		const options = resolveSSAOOptions(request.options as SSAOOptions);
		resources.frameIndex = (resources.frameIndex + 1) % 1024;
		context.shared.compute.writeBuffer(
			resources.params,
			createSSAOKernelParams(
				input.width,
				input.height,
				aoRaw.width,
				aoRaw.height,
				options,
				request.frameContext.viewCamera,
				0,
				0,
				resources.frameIndex / 1024
			) as unknown as BufferSource
		);
		let binding = context.shared.getCachedBindGroup(
			"ssao-raw",
			resources.rawPipeline,
			[
				{ binding: 0, resource: normalTexture },
				{ binding: 1, resource: depthTexture },
				{ binding: 2, resource: context.shared.sampler },
				{ binding: 3, resource: resources.params },
				{ binding: 4, resource: aoRaw },
			],
			"WebGPUSSAO_RawBinding"
		);
		context.encoder.beginComputePass({ label: "WebGPUSSAO_Raw" });
		context.encoder.setComputePipeline(resources.rawPipeline);
		context.encoder.setBindingGroup(0, binding);
		context.encoder.dispatchWorkgroups(
			ceilDiv(aoRaw.width, WORKGROUP_SIZE),
			ceilDiv(aoRaw.height, WORKGROUP_SIZE),
			1
		);
		context.encoder.endComputePass();

		await context.shared.getDenoiser().encode({
			scope: "ssao",
			encoder: context.encoder,
			source: aoRaw,
			scratch: aoBlur,
			output: aoRaw,
			depth: depthTexture,
			normal: normalTexture,
			sampler: context.shared.sampler,
			options: {
				mode: "fast",
				signal: "scalar",
				radius: options.blurRadius,
				depthPhi: options.blurSharpness,
				normalPhi: 16,
				valuePhi: 0,
				confidenceFloor: 1,
			},
		});

		const combineTarget = context.resources.color.output;
		if (!combineTarget) return false;
		binding = context.shared.getCachedBindGroup(
			`ssao-combine-${combineTarget === targets.postPing ? "ping" : "pong"}`,
			resources.combinePipeline,
			[
				{ binding: 0, resource: input },
				{ binding: 1, resource: aoRaw },
				{ binding: 2, resource: context.shared.sampler },
				{ binding: 3, resource: resources.params },
				{ binding: 4, resource: combineTarget },
			],
			"WebGPUSSAO_CombineBinding"
		);
		context.encoder.beginComputePass({ label: "WebGPUSSAO_Combine" });
		context.encoder.setComputePipeline(resources.combinePipeline);
		context.encoder.setBindingGroup(0, binding);
		context.encoder.dispatchWorkgroups(
			ceilDiv(combineTarget.width, WORKGROUP_SIZE),
			ceilDiv(combineTarget.height, WORKGROUP_SIZE),
			1
		);
		context.encoder.endComputePass();
		return true;
	}

	private async _ensureResources(
		shared: WebGPUPostProcessServices
	): Promise<WebGPUSSAOResources> {
		let resources = this._resources.get(shared);
		if (!resources) {
			resources = {
				shared,
				module: null,
				rawPipeline: null,
				combinePipeline: null,
				params: null,
				frameIndex: 0,
			};
			this._resources.set(shared, resources);
		}
		await shared.ensureCommonResources();
		await shared.getDenoiser().ensureResources();
		if (!resources.module) {
			const shader = await ShaderSource.load("webgpu.postprocess.ssao.composite");
			resources.module = await shared.compute.createShaderModule({
				label: "WebGPUSSAOShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!resources.rawPipeline) {
			resources.rawPipeline = await shared.compute.createComputePipeline({
				label: "WebGPUSSAORawPipeline",
				compute: { module: resources.module, entryPoint: "csRaw" },
			});
		}
		if (!resources.combinePipeline) {
			resources.combinePipeline = await shared.compute.createComputePipeline({
				label: "WebGPUSSAOCombinePipeline",
				compute: { module: resources.module, entryPoint: "csCombine" },
			});
		}
		if (!resources.params) {
			resources.params = shared.compute.createBuffer({
				label: "WebGPUSSAOParams",
				size: 16 * 4,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
		}
		return resources;
	}
}

/**
 * WebGL implementation of the cross-backend SSAO pass.
 */
/** @internal WebGL implementation for the built-in SSAO pass. */
export class WebGLScreenSpaceAmbientOcclusionImplementation
	implements PostProcessPassImplementation<WebGLSSAOContext>
{
	public readonly id = "ssao:webgl";
	public describeExecution(request: PostProcessPassResolveRequest<ResolvedSSAOOptions>) {
		const options = resolveSSAOOptions(request.options);
		const scale = 1 / options.downsample;
		return {
			...WEBGL_VERSIONED_EXECUTION,
			gBuffer: (["depth", "normal"] as const).map((semantic) => ({
				semantic,
				...POST_PROCESS_SAMPLED_READ,
			})),
			transients: [
				{
					id: SSAO_RAW_TRANSIENT_ID,
					widthScale: scale,
					heightScale: scale,
					format: "rgba16float",
				},
				{
					id: SSAO_BLUR_TRANSIENT_ID,
					widthScale: scale,
					heightScale: scale,
					format: "rgba16float",
				},
			].map((descriptor) => ({
				descriptor,
				uses: [POST_PROCESS_COLOR_ATTACHMENT_WRITE],
			})),
		} satisfies PostProcessExecutionDeclaration;
	}
	private _programCompiler: WebGLProgramCompiler | null = null;
	private _rawProgramSlot: WebGLProgramSlot<WebGLSSAORawProgram> | null = null;
	private _blurProgramSlot: WebGLProgramSlot<WebGLSSAOBlurProgram> | null = null;
	private _combineProgramSlot: WebGLProgramSlot<WebGLSSAOCombineProgram> | null = null;
	private _frameIndex = 0;

	public warmup(context: WebGLSSAOContext | undefined): void {
		if (!context) {
			return;
		}
		this._ensureProgramSlots(context.programCompiler);
		this._rawProgramSlot!.warmup();
		this._blurProgramSlot!.warmup();
		this._combineProgramSlot!.warmup();
	}

	public execute(
		request: PostProcessPassRequest,
		context: WebGLSSAOContext | undefined
	): PostProcessPassResult {
		if (!context) {
			return { ran: false };
		}
		return this._runSSAOKernel(request, context);
	}

	private _runSSAOKernel(
		request: PostProcessPassRequest,
		context: WebGLSSAOContext
	): PostProcessPassResult {
		const depthTexture = context.resources.getGBuffer("depth");
		const normalTexture = context.resources.getGBuffer("normal");
		const ssaoRawTexture = context.resources.getTransient(SSAO_RAW_TRANSIENT_ID);
		const ssaoBlurTexture = context.resources.getTransient(SSAO_BLUR_TRANSIENT_ID);
		const sourceTexture = context.resources.color.input;
		const targetTexture = context.resources.color.output;
		if (
			!context.postFramebuffer ||
			!context.sceneColorTexture ||
			!depthTexture ||
			!normalTexture ||
			!ssaoRawTexture ||
			!ssaoBlurTexture ||
			!sourceTexture ||
			!targetTexture ||
			!context.fullscreenVao
		) {
			return { ran: false };
		}
		const gl = context.gl;
		this._ensureProgramSlots(context.programCompiler);
		const rawProgram = this._rawProgramSlot!.tryGet();
		const blurProgram = this._blurProgramSlot!.tryGet();
		const combineProgram = this._combineProgramSlot!.tryGet();
		if (!rawProgram || !blurProgram || !combineProgram) {
			return { ran: false };
		}
		const options = resolveSSAOOptions(request.options as SSAOOptions);
		const aoWidth = Math.max(
			1,
			Math.floor(context.width / options.downsample)
		);
		const aoHeight = Math.max(
			1,
			Math.floor(context.height / options.downsample)
		);
		const params = createSSAOKernelParams(
			context.width,
			context.height,
			aoWidth,
			aoHeight,
			options,
			request.frameContext.viewCamera,
			1,
			0,
			this._nextFrameJitter()
		);
		const view = request.frameContext.viewCamera.viewMatrix.elements;
		const cameraPosition = request.frameContext.viewCamera.getWorldPosition();

		gl.bindFramebuffer(gl.FRAMEBUFFER, context.postFramebuffer);
		gl.bindVertexArray(context.fullscreenVao);
		gl.disable(gl.CULL_FACE);
		gl.disable(gl.DEPTH_TEST);
		gl.disable(gl.BLEND);

		context.bindColorTarget(ssaoRawTexture);
		gl.viewport(0, 0, aoWidth, aoHeight);
		gl.useProgram(rawProgram.program);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, normalTexture);
		gl.activeTexture(gl.TEXTURE1);
		gl.bindTexture(gl.TEXTURE_2D, depthTexture);
		if (rawProgram.uniforms.normalMap) gl.uniform1i(rawProgram.uniforms.normalMap, 0);
		if (rawProgram.uniforms.depthMap) gl.uniform1i(rawProgram.uniforms.depthMap, 1);
		if (rawProgram.uniforms.invSize) {
			gl.uniform4f(
				rawProgram.uniforms.invSize,
				params[0],
				params[1],
				params[2],
				params[3]
			);
		}
		if (rawProgram.uniforms.gtao) {
			gl.uniform4f(
				rawProgram.uniforms.gtao,
				params[4],
				params[5],
				params[6],
				params[7]
			);
		}
		if (rawProgram.uniforms.blurProj) {
			gl.uniform4f(
				rawProgram.uniforms.blurProj,
				params[8],
				params[9],
				params[10],
				params[11]
			);
		}
		if (rawProgram.uniforms.pass) {
			gl.uniform4f(
				rawProgram.uniforms.pass,
				params[12],
				params[13],
				params[14],
				params[15]
			);
		}
		if (rawProgram.uniforms.cameraPosition) {
			gl.uniform3f(
				rawProgram.uniforms.cameraPosition,
				cameraPosition.x,
				cameraPosition.y,
				cameraPosition.z
			);
		}
		if (rawProgram.uniforms.basisRight) {
			gl.uniform3f(
				rawProgram.uniforms.basisRight,
				view[0][0],
				view[0][1],
				view[0][2]
			);
		}
		if (rawProgram.uniforms.basisUp) {
			gl.uniform3f(rawProgram.uniforms.basisUp, view[1][0], view[1][1], view[1][2]);
		}
		if (rawProgram.uniforms.basisBackward) {
			gl.uniform3f(
				rawProgram.uniforms.basisBackward,
				view[2][0],
				view[2][1],
				view[2][2]
			);
		}
		context.drawFullscreen(aoWidth, aoHeight, request.frameContext);

		context.bindColorTarget(ssaoBlurTexture);
		gl.viewport(0, 0, aoWidth, aoHeight);
		gl.useProgram(blurProgram.program);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, ssaoRawTexture);
		gl.activeTexture(gl.TEXTURE1);
		gl.bindTexture(gl.TEXTURE_2D, depthTexture);
		if (blurProgram.uniforms.sourceMap) gl.uniform1i(blurProgram.uniforms.sourceMap, 0);
		if (blurProgram.uniforms.depthMap) gl.uniform1i(blurProgram.uniforms.depthMap, 1);
		if (blurProgram.uniforms.invSize) {
			gl.uniform4f(
				blurProgram.uniforms.invSize,
				params[0],
				params[1],
				params[2],
				params[3]
			);
		}
		if (blurProgram.uniforms.blurProj) {
			gl.uniform4f(
				blurProgram.uniforms.blurProj,
				params[8],
				params[9],
				params[10],
				params[11]
			);
		}
		if (blurProgram.uniforms.pass) {
			gl.uniform4f(blurProgram.uniforms.pass, 1, 0, params[14], params[15]);
		}
		context.drawFullscreen(aoWidth, aoHeight, request.frameContext);

		context.bindColorTarget(ssaoRawTexture);
		gl.viewport(0, 0, aoWidth, aoHeight);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, ssaoBlurTexture);
		gl.activeTexture(gl.TEXTURE1);
		gl.bindTexture(gl.TEXTURE_2D, depthTexture);
		if (blurProgram.uniforms.pass) {
			gl.uniform4f(blurProgram.uniforms.pass, 0, 1, params[14], params[15]);
		}
		context.drawFullscreen(context.width, context.height, request.frameContext);

		context.bindColorTarget(targetTexture);
		gl.viewport(0, 0, context.width, context.height);
		gl.useProgram(combineProgram.program);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, sourceTexture);
		gl.activeTexture(gl.TEXTURE1);
		gl.bindTexture(gl.TEXTURE_2D, ssaoRawTexture);
		if (combineProgram.uniforms.sceneColor) {
			gl.uniform1i(combineProgram.uniforms.sceneColor, 0);
		}
		if (combineProgram.uniforms.aoMap) {
			gl.uniform1i(combineProgram.uniforms.aoMap, 1);
		}
		if (combineProgram.uniforms.invSize) {
			gl.uniform4f(
				combineProgram.uniforms.invSize,
				params[0],
				params[1],
				params[2],
				params[3]
			);
		}
		context.drawFullscreen(context.width, context.height, request.frameContext);
		gl.bindVertexArray(null);
		return { ran: true };
	}

	public destroy(): void {
		this._rawProgramSlot?.destroy();
		this._blurProgramSlot?.destroy();
		this._combineProgramSlot?.destroy();
		this._rawProgramSlot = null;
		this._blurProgramSlot = null;
		this._combineProgramSlot = null;
		this._programCompiler = null;
		this._frameIndex = 0;
	}

	/** @internal Resets the pass-owned temporal noise phase after invalidation. */
	public invalidate(): void {
		this._frameIndex = 0;
	}

	private _nextFrameJitter(): number {
		this._frameIndex = (this._frameIndex + 1) % 1024;
		return this._frameIndex / 1024;
	}

	private _ensureProgramSlots(compiler: WebGLProgramCompiler): void {
		if (this._programCompiler === compiler) {
			return;
		}
		this.destroy();
		this._programCompiler = compiler;
		const vertex = () => ShaderSource.get("webgl.part.presentVertex.raw");
		this._rawProgramSlot = compiler.createSlot({
			label: "WebGLSSAORawProgram",
			vertex,
			fragment: () => ShaderSource.get("webgl.part.ssaoRawFragment.raw"),
			reflect: (gl, program) => ({
				program,
				uniforms: {
					normalMap: gl.getUniformLocation(program, "uNormalMap"),
					depthMap: gl.getUniformLocation(program, "uDepthMap"),
					invSize: gl.getUniformLocation(program, "uInvSize"),
					gtao: gl.getUniformLocation(program, "uGTAO"),
					blurProj: gl.getUniformLocation(program, "uBlurProj"),
					pass: gl.getUniformLocation(program, "uPass"),
					cameraPosition: gl.getUniformLocation(program, "uCameraPosition"),
					basisRight: gl.getUniformLocation(program, "uBasisRight"),
					basisUp: gl.getUniformLocation(program, "uBasisUp"),
					basisBackward: gl.getUniformLocation(program, "uBasisBackward"),
				},
			}),
		});
		this._blurProgramSlot = compiler.createSlot({
			label: "WebGLSSAOBlurProgram",
			vertex,
			fragment: () => ShaderSource.get("webgl.part.ssaoBlurFragment.raw"),
			reflect: (gl, program) => ({
				program,
				uniforms: {
					sourceMap: gl.getUniformLocation(program, "uSourceMap"),
					depthMap: gl.getUniformLocation(program, "uDepthMap"),
					invSize: gl.getUniformLocation(program, "uInvSize"),
					blurProj: gl.getUniformLocation(program, "uBlurProj"),
					pass: gl.getUniformLocation(program, "uPass"),
				},
			}),
		});
		this._combineProgramSlot = compiler.createSlot({
			label: "WebGLSSAOCombineProgram",
			vertex,
			fragment: () =>
				ShaderSource.get("webgl.part.ssaoCombineFragment.raw"),
			reflect: (gl, program) => ({
				program,
				uniforms: {
					sceneColor: gl.getUniformLocation(program, "uSceneColor"),
					aoMap: gl.getUniformLocation(program, "uAoMap"),
					invSize: gl.getUniformLocation(program, "uInvSize"),
				},
			}),
		});
	}
}

export interface ScreenSpaceAmbientOcclusionPassConfig
	extends Omit<
		PostProcessPassConfig<SSAOOptions>,
		| "id"
		| "builtIn"
		| "label"
		| "placement"
		| "order"
		| "implementations"
	> {}

/**
 * Stateful logical SSAO pass shared by all rendering backends.
 */
export class ScreenSpaceAmbientOcclusionPass extends PostProcessPass<
	SSAOOptions,
	ResolvedSSAOOptions
> {
	constructor(config: ScreenSpaceAmbientOcclusionPassConfig = {}) {
		super({
			...config,
			id: SCREEN_SPACE_AMBIENT_OCCLUSION_PASS_ORDER.id,
			schedule: {
				placement:
					config.schedule?.placement ??
					SCREEN_SPACE_AMBIENT_OCCLUSION_PASS_ORDER.placement,
				order: config.schedule?.order ?? SCREEN_SPACE_AMBIENT_OCCLUSION_PASS_ORDER.order,
				incremental:
					config.schedule?.incremental ??
					SCREEN_SPACE_AMBIENT_OCCLUSION_PASS_ORDER.incremental,
			},
			label: "SSAO",
			colorContract: config.colorContract ?? {
				input: "scene-linear-hdr",
				output: "scene-linear-hdr",
			},
			implementations: {
				software: () => new SoftwareScreenSpaceAmbientOcclusionImplementation(),
				webgpu: () => new WebGPUScreenSpaceAmbientOcclusionImplementation(),
				webgl: () => new WebGLScreenSpaceAmbientOcclusionImplementation(),
			},
		});
	}

	public override normalizeOptions(): ResolvedSSAOOptions {
		return resolveSSAOOptions(this.getRawOptions());
	}
}

function reconstructViewPos(
	ndcX: number,
	ndcY: number,
	zView: number,
	camera: FrameContext["viewCamera"]
): IVector3 {
	if (camera.type === CameraType.Orthographic) {
		const orthoCam = camera as OrthographicCamera;
		const bounds = orthoCam.getBounds();
		const xView =
			((ndcX + 1) * 0.5) * (bounds.right - bounds.left) + bounds.left;
		const yView =
			((ndcY + 1) * 0.5) * (bounds.top - bounds.bottom) + bounds.bottom;
		return { x: xView, y: yView, z: -zView };
	}

	const fovRad = (camera.fov * Math.PI) / 180;
	const tanHalfFov = Math.tan(fovRad * 0.5);
	const aspect = camera.aspectRatio;
	return {
		x: ndcX * aspect * tanHalfFov * zView,
		y: ndcY * tanHalfFov * zView,
		z: -zView,
	};
}
