import type { ICommandEncoder } from "../../backends/ICommandEncoder";
import {
	BufferUsage,
	TextureFormat,
	TextureUsage,
	type IComputePipeline,
	type IRenderBuffer,
	type IRenderTexture,
	type IShaderModule,
} from "../../backends/types";
import {
	type WebGPUPostProcessFrameTargets,
	type WebGPUPostProcessServices,
} from "../../backends/webgpu/WebGPUPostProcessContracts";
import {
	WEBGPU_2D_COMPUTE_WORKGROUP_SIZE as WORKGROUP_SIZE,
} from "../../backends/webgpu/constants";
import type {
	WebGLProgramCompiler,
	WebGLProgramSlot,
} from "../../backends/webgl/WebGLProgramCompiler";
import { clamp } from "../../maths/Common";
import { ceilDiv, finiteOr } from "../../maths/Misc";
import {
	ShaderSource,
	type WebGPUPostProcessShaderPart,
} from "../../shaders/ShaderSource";
import {
	PostProcessPass,
	type PostProcessPassConfig,
} from "../PostProcessPass";
import type { PostProcessScheduleEntry } from "../ordering";
import {
	WEBGL_VERSIONED_EXECUTION,
	WEBGPU_VERSIONED_EXECUTION,
} from "../executionDeclarations";
import type {
	PostProcessPassImplementation,
	PostProcessPassRequest,
	PostProcessPassResult,
	PostProcessResourceAccessor,
} from "../types";

export const BLOOM_PASS_ID = "bloom";
export const BLOOM_PASS_ORDER = {
	id: BLOOM_PASS_ID,
	placement: "hdr",
	order: 500,
	incremental: {
		firstPass: "bloom",
		grade: "standard",
		inflationRadius: 48,
	},
} as const satisfies PostProcessScheduleEntry;

export interface BloomOptions {
	/** Luminance threshold above which pixels contribute to bloom. */
	threshold?: number;
	/** Soft threshold width that smooths the transition into bloom. */
	softKnee?: number;
	/** Final bloom contribution mixed back into the HDR scene color. */
	intensity?: number;
	/** Single-pass blur radius used by the WebGL bloom path. */
	radius?: number;
	/** Number of downsample mip passes (1-8). Higher values produce wider bloom. */
	mipPasses?: number;
	/** Tent-filter radius used during upsample (default 1). */
	filterRadius?: number;
	/** Allows backend-specific experimental bloom options. */
	[key: string]: unknown;
}

export const DEFAULT_BLOOM_OPTIONS: Required<
	Pick<
		BloomOptions,
		"threshold" | "softKnee" | "intensity" | "radius" | "mipPasses" | "filterRadius"
	>
> = {
	threshold: 1,
	softKnee: 0.5,
	intensity: 0.8,
	radius: 1,
	mipPasses: 5,
	filterRadius: 1,
};

/** @internal WebGPU context supplied to the built-in bloom implementation. */
export interface WebGPUBloomContext {
	readonly encoder?: ICommandEncoder;
	readonly targets?: WebGPUPostProcessFrameTargets;
	readonly shared: WebGPUPostProcessServices;
	readonly resources: PostProcessResourceAccessor<IRenderTexture>;
}

/** @internal WebGL context supplied to the built-in bloom implementation. */
export interface WebGLBloomContext {
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
	drawFullscreen(): void;
}

interface WebGLBloomProgram {
	readonly program: WebGLProgram;
	readonly uniforms: {
		readonly sourceMap: WebGLUniformLocation | null;
		readonly texelSize: WebGLUniformLocation | null;
		readonly bloomParams: WebGLUniformLocation | null;
	};
}

interface WebGPUBloomResources {
	shared: WebGPUPostProcessServices;
	downsampleModule: IShaderModule | null;
	blurHModule: IShaderModule | null;
	blurVModule: IShaderModule | null;
	upsampleModule: IShaderModule | null;
	compositeModule: IShaderModule | null;
	downsamplePipeline: IComputePipeline | null;
	blurHPipeline: IComputePipeline | null;
	blurVPipeline: IComputePipeline | null;
	upsamplePipeline: IComputePipeline | null;
	compositePipeline: IComputePipeline | null;
	downsampleParams: IRenderBuffer | null;
	blurParams: IRenderBuffer | null;
	upsampleParams: IRenderBuffer | null;
	compositeParams: IRenderBuffer | null;
	mipTextures: Array<[IRenderTexture, IRenderTexture]>;
	mipWidth: number;
	mipHeight: number;
	mipCount: number;
}

/**
 * WebGPU implementation of the HDR bloom pass.
 */
/** @internal WebGPU implementation for the built-in bloom pass. */
export class WebGPUBloomImplementation
	implements PostProcessPassImplementation<WebGPUBloomContext, BloomOptions>
{
	public readonly id = "bloom:webgpu";
	public describeExecution() {
		return WEBGPU_VERSIONED_EXECUTION;
	}
	private _resources = new Map<WebGPUPostProcessServices, WebGPUBloomResources>();

	public async warmup(context: WebGPUBloomContext | undefined): Promise<void> {
		if (context) {
			await this._ensureResources(context.shared);
		}
	}

	public async execute(
		request: PostProcessPassRequest<BloomOptions>,
		context: WebGPUBloomContext | undefined
	): Promise<PostProcessPassResult> {
		if (!context?.encoder || !context.targets) {
			return { ran: false };
		}
		const ran = await this._runBloomKernel(request, context);
		return ran ? { ran: true } : { ran: false };
	}

	public invalidate(): void {
		for (const resources of this._resources.values()) {
			this._destroyMipTextures(resources);
		}
	}

	public destroy(): void {
		for (const resources of this._resources.values()) {
			this._destroyMipTextures(resources);
			const shared = resources.shared;
			shared.destroyManagedResource(
				resources.downsamplePipeline,
				"bloom downsample pipeline"
			);
			shared.destroyManagedResource(
				resources.blurHPipeline,
				"bloom horizontal blur pipeline"
			);
			shared.destroyManagedResource(
				resources.blurVPipeline,
				"bloom vertical blur pipeline"
			);
			shared.destroyManagedResource(
				resources.upsamplePipeline,
				"bloom upsample pipeline"
			);
			shared.destroyManagedResource(
				resources.compositePipeline,
				"bloom composite pipeline"
			);
			shared.destroyManagedResource(
				resources.downsampleModule,
				"bloom downsample shader module"
			);
			shared.destroyManagedResource(
				resources.blurHModule,
				"bloom horizontal blur shader module"
			);
			shared.destroyManagedResource(
				resources.blurVModule,
				"bloom vertical blur shader module"
			);
			shared.destroyManagedResource(
				resources.upsampleModule,
				"bloom upsample shader module"
			);
			shared.destroyManagedResource(
				resources.compositeModule,
				"bloom composite shader module"
			);
			shared.destroyManagedResource(
				resources.downsampleParams,
				"bloom downsample params buffer"
			);
			shared.destroyManagedResource(
				resources.blurParams,
				"bloom blur params buffer"
			);
			shared.destroyManagedResource(
				resources.upsampleParams,
				"bloom upsample params buffer"
			);
			shared.destroyManagedResource(
				resources.compositeParams,
				"bloom composite params buffer"
			);
			shared.invalidateBindingsByPrefix("bloom-");
		}
		this._resources.clear();
	}

	private async _runBloomKernel(
		request: PostProcessPassRequest<BloomOptions>,
		context: WebGPUBloomContext
	): Promise<boolean> {
		const resources = await this._ensureResources(context.shared);
		if (
			!context.encoder ||
			!context.targets ||
			!context.shared.sampler ||
			!resources.downsamplePipeline ||
			!resources.blurHPipeline ||
			!resources.blurVPipeline ||
			!resources.upsamplePipeline ||
			!resources.compositePipeline ||
			!resources.downsampleParams ||
			!resources.blurParams ||
			!resources.upsampleParams ||
			!resources.compositeParams
		) {
			return false;
		}

		const targets = context.targets;
		const input = context.resources.color.input;
		if (!input) return false;
		const options = request.options ?? {};
		const threshold = Math.max(
			0,
			finiteOr(options.threshold, DEFAULT_BLOOM_OPTIONS.threshold)
		);
		const softKnee = Math.max(
			1e-4,
			finiteOr(options.softKnee, DEFAULT_BLOOM_OPTIONS.softKnee)
		);
		const intensity = Math.max(
			0,
			finiteOr(options.intensity, DEFAULT_BLOOM_OPTIONS.intensity)
		);
		const filterRadius = clamp(
			finiteOr(options.filterRadius, DEFAULT_BLOOM_OPTIONS.filterRadius),
			0.5,
			4
		);
		const requestedMips = clamp(
			Math.round(finiteOr(options.mipPasses, DEFAULT_BLOOM_OPTIONS.mipPasses)),
			1,
			8
		);

		const srcW = input.width;
		const srcH = input.height;
		this._ensureMipTextures(resources, srcW, srcH, requestedMips);
		if (resources.mipCount === 0) {
			return false;
		}
		const mips = resources.mipTextures;
		const dsMipIndex = resources.mipCount - 1;
		const dsDst = mips[dsMipIndex][0];
		context.shared.compute.writeBuffer(
			resources.downsampleParams,
			new Float32Array([
				1 / Math.max(srcW, 1),
				1 / Math.max(srcH, 1),
				threshold,
				softKnee,
			])
		);
		let binding = context.shared.getCachedBindGroup(
			"bloom-ds-0",
			resources.downsamplePipeline,
			[
				{ binding: 0, resource: input },
				{ binding: 1, resource: context.shared.sampler },
				{ binding: 2, resource: resources.downsampleParams },
				{ binding: 3, resource: dsDst },
			],
			"WebGPUBloom_Downsample0"
		);
		context.encoder.beginComputePass({ label: "WebGPUBloom_Downsample0" });
		context.encoder.setComputePipeline(resources.downsamplePipeline);
		context.encoder.setBindingGroup(0, binding);
		context.encoder.dispatchWorkgroups(
			ceilDiv(dsDst.width, WORKGROUP_SIZE),
			ceilDiv(dsDst.height, WORKGROUP_SIZE),
			1
		);
		context.encoder.endComputePass();

		for (let i = dsMipIndex - 1; i >= 0; i--) {
			const src = mips[i + 1][0];
			const dst = mips[i][0];
			context.shared.compute.writeBuffer(
				resources.downsampleParams,
				new Float32Array([
					1 / Math.max(src.width, 1),
					1 / Math.max(src.height, 1),
					-1,
					1e-4,
				])
			);
			binding = context.shared.getCachedBindGroup(
				`bloom-ds-${dsMipIndex - i}`,
				resources.downsamplePipeline,
				[
					{ binding: 0, resource: src },
					{ binding: 1, resource: context.shared.sampler },
					{ binding: 2, resource: resources.downsampleParams },
					{ binding: 3, resource: dst },
				],
				`WebGPUBloom_Downsample${dsMipIndex - i}`
			);
			context.encoder.beginComputePass({
				label: `WebGPUBloom_Downsample${dsMipIndex - i}`,
			});
			context.encoder.setComputePipeline(resources.downsamplePipeline);
			context.encoder.setBindingGroup(0, binding);
			context.encoder.dispatchWorkgroups(
				ceilDiv(dst.width, WORKGROUP_SIZE),
				ceilDiv(dst.height, WORKGROUP_SIZE),
				1
			);
			context.encoder.endComputePass();
		}

		for (let i = 0; i < resources.mipCount; i++) {
			const texA = mips[i][0];
			const texB = mips[i][1];
			const invW = 1 / Math.max(texA.width, 1);
			const invH = 1 / Math.max(texA.height, 1);
			context.shared.compute.writeBuffer(
				resources.blurParams,
				new Float32Array([invW, invH, 1, 0])
			);
			binding = context.shared.getCachedBindGroup(
				`bloom-blurH-${i}`,
				resources.blurHPipeline,
				[
					{ binding: 0, resource: texA },
					{ binding: 1, resource: context.shared.sampler },
					{ binding: 2, resource: resources.blurParams },
					{ binding: 3, resource: texB },
				],
				`WebGPUBloom_BlurH_${i}`
			);
			context.encoder.beginComputePass({ label: `WebGPUBloom_BlurH_${i}` });
			context.encoder.setComputePipeline(resources.blurHPipeline);
			context.encoder.setBindingGroup(0, binding);
			context.encoder.dispatchWorkgroups(
				ceilDiv(texB.width, WORKGROUP_SIZE),
				ceilDiv(texB.height, WORKGROUP_SIZE),
				1
			);
			context.encoder.endComputePass();
			context.shared.compute.writeBuffer(
				resources.blurParams,
				new Float32Array([invW, invH, 0, 1])
			);
			binding = context.shared.getCachedBindGroup(
				`bloom-blurV-${i}`,
				resources.blurVPipeline,
				[
					{ binding: 0, resource: texB },
					{ binding: 1, resource: context.shared.sampler },
					{ binding: 2, resource: resources.blurParams },
					{ binding: 3, resource: texA },
				],
				`WebGPUBloom_BlurV_${i}`
			);
			context.encoder.beginComputePass({ label: `WebGPUBloom_BlurV_${i}` });
			context.encoder.setComputePipeline(resources.blurVPipeline);
			context.encoder.setBindingGroup(0, binding);
			context.encoder.dispatchWorkgroups(
				ceilDiv(texA.width, WORKGROUP_SIZE),
				ceilDiv(texA.height, WORKGROUP_SIZE),
				1
			);
			context.encoder.endComputePass();
		}

		for (let i = 1; i < resources.mipCount; i++) {
			const smallerMip = mips[i - 1][0];
			const currentMip = mips[i][0];
			const dstMip = mips[i][1];
			context.shared.compute.writeBuffer(
				resources.upsampleParams,
				new Float32Array([
					1 / Math.max(smallerMip.width, 1),
					1 / Math.max(smallerMip.height, 1),
					filterRadius,
					0,
				])
			);
			binding = context.shared.getCachedBindGroup(
				`bloom-up-${i}`,
				resources.upsamplePipeline,
				[
					{ binding: 0, resource: smallerMip },
					{ binding: 1, resource: currentMip },
					{ binding: 2, resource: context.shared.sampler },
					{ binding: 3, resource: resources.upsampleParams },
					{ binding: 4, resource: dstMip },
				],
				`WebGPUBloom_Upsample_${i}`
			);
			context.encoder.beginComputePass({ label: `WebGPUBloom_Upsample_${i}` });
			context.encoder.setComputePipeline(resources.upsamplePipeline);
			context.encoder.setBindingGroup(0, binding);
			context.encoder.dispatchWorkgroups(
				ceilDiv(dstMip.width, WORKGROUP_SIZE),
				ceilDiv(dstMip.height, WORKGROUP_SIZE),
				1
			);
			context.encoder.endComputePass();
			mips[i] = [dstMip, currentMip];
		}

		const bloomResult = mips[resources.mipCount - 1][0];
		const target = context.resources.color.output;
		if (!target) return false;
		context.shared.compute.writeBuffer(
			resources.compositeParams,
			new Float32Array([
				1 / Math.max(target.width, 1),
				1 / Math.max(target.height, 1),
				intensity,
				0,
			])
		);
		binding = context.shared.getCachedBindGroup(
			`bloom-comp-${target === targets.postPing ? "ping" : "pong"}`,
			resources.compositePipeline,
			[
				{ binding: 0, resource: input },
				{ binding: 1, resource: bloomResult },
				{ binding: 2, resource: context.shared.sampler },
				{ binding: 3, resource: resources.compositeParams },
				{ binding: 4, resource: target },
			],
			"WebGPUBloom_Composite"
		);
		context.encoder.beginComputePass({ label: "WebGPUBloom_Composite" });
		context.encoder.setComputePipeline(resources.compositePipeline);
		context.encoder.setBindingGroup(0, binding);
		context.encoder.dispatchWorkgroups(
			ceilDiv(target.width, WORKGROUP_SIZE),
			ceilDiv(target.height, WORKGROUP_SIZE),
			1
		);
		context.encoder.endComputePass();
		return true;
	}

	private async _ensureResources(
		shared: WebGPUPostProcessServices
	): Promise<WebGPUBloomResources> {
		let resources = this._resources.get(shared);
		if (!resources) {
			resources = {
				shared,
				downsampleModule: null,
				blurHModule: null,
				blurVModule: null,
				upsampleModule: null,
				compositeModule: null,
				downsamplePipeline: null,
				blurHPipeline: null,
				blurVPipeline: null,
				upsamplePipeline: null,
				compositePipeline: null,
				downsampleParams: null,
				blurParams: null,
				upsampleParams: null,
				compositeParams: null,
				mipTextures: [],
				mipWidth: 0,
				mipHeight: 0,
				mipCount: 0,
			};
			this._resources.set(shared, resources);
		}
		await shared.ensureCommonResources();
		if (!resources.downsampleModule) {
			resources.downsampleModule = await this._createModule(
				shared,
				"bloomDownsample",
				"WebGPUBloomDownsampleShader"
			);
		}
		if (!resources.blurHModule) {
			resources.blurHModule = await this._createModule(
				shared,
				"bloomBlurH",
				"WebGPUBloomBlurHShader"
			);
		}
		if (!resources.blurVModule) {
			resources.blurVModule = await this._createModule(
				shared,
				"bloomBlurV",
				"WebGPUBloomBlurVShader"
			);
		}
		if (!resources.upsampleModule) {
			resources.upsampleModule = await this._createModule(
				shared,
				"bloomUpsample",
				"WebGPUBloomUpsampleShader"
			);
		}
		if (!resources.compositeModule) {
			resources.compositeModule = await this._createModule(
				shared,
				"bloomComposite",
				"WebGPUBloomCompositeShader"
			);
		}
		resources.downsamplePipeline ??= await shared.compute.createComputePipeline({
			label: "WebGPUBloomDownsamplePipeline",
			compute: { module: resources.downsampleModule, entryPoint: "csMain" },
		});
		resources.blurHPipeline ??= await shared.compute.createComputePipeline({
			label: "WebGPUBloomBlurHPipeline",
			compute: { module: resources.blurHModule, entryPoint: "csMain" },
		});
		resources.blurVPipeline ??= await shared.compute.createComputePipeline({
			label: "WebGPUBloomBlurVPipeline",
			compute: { module: resources.blurVModule, entryPoint: "csMain" },
		});
		resources.upsamplePipeline ??= await shared.compute.createComputePipeline({
			label: "WebGPUBloomUpsamplePipeline",
			compute: { module: resources.upsampleModule, entryPoint: "csMain" },
		});
		resources.compositePipeline ??= await shared.compute.createComputePipeline({
			label: "WebGPUBloomCompositePipeline",
			compute: { module: resources.compositeModule, entryPoint: "csMain" },
		});
		resources.downsampleParams ??= shared.compute.createBuffer({
			label: "WebGPUBloomDownsampleParams",
			size: 4 * 4,
			usage: BufferUsage.Uniform | BufferUsage.CopyDst,
		});
		resources.blurParams ??= shared.compute.createBuffer({
			label: "WebGPUBloomBlurParams",
			size: 4 * 4,
			usage: BufferUsage.Uniform | BufferUsage.CopyDst,
		});
		resources.upsampleParams ??= shared.compute.createBuffer({
			label: "WebGPUBloomUpsampleParams",
			size: 4 * 4,
			usage: BufferUsage.Uniform | BufferUsage.CopyDst,
		});
		resources.compositeParams ??= shared.compute.createBuffer({
			label: "WebGPUBloomCompositeParams",
			size: 4 * 4,
			usage: BufferUsage.Uniform | BufferUsage.CopyDst,
		});
		return resources;
	}

	private async _createModule(
		shared: WebGPUPostProcessServices,
		part: WebGPUPostProcessShaderPart,
		label: string
	): Promise<IShaderModule> {
		const shader = await ShaderSource.load(
			`webgpu.postprocess.${part}`
		);
		return shared.compute.createShaderModule({
			label,
			code: shader.source.code,
			sourceMap: shader.source.sourceMap,
			language: "wgsl",
			stage: "compute",
			sourceKind: "postprocess",
		});
	}

	private _ensureMipTextures(
		resources: WebGPUBloomResources,
		srcWidth: number,
		srcHeight: number,
		requestedMips: number
	): void {
		const halfW = Math.max(1, Math.floor(srcWidth / 2));
		const halfH = Math.max(1, Math.floor(srcHeight / 2));
		const maxPossibleMips = Math.floor(Math.log2(Math.max(halfW, halfH))) + 1;
		const mipCount = Math.min(requestedMips, maxPossibleMips);
		if (
			resources.mipWidth === halfW &&
			resources.mipHeight === halfH &&
			resources.mipCount === mipCount &&
			resources.mipTextures.length === mipCount
		) {
			return;
		}
		this._destroyMipTextures(resources);
		resources.mipWidth = halfW;
		resources.mipHeight = halfH;
		resources.mipCount = mipCount;
		for (let i = 0; i < mipCount; i++) {
			const level = mipCount - 1 - i;
			const width = Math.max(1, halfW >> level);
			const height = Math.max(1, halfH >> level);
			const texA = resources.shared.compute.createTexture({
				width,
				height,
				format: TextureFormat.RGBA16Float,
				usage:
					TextureUsage.TextureBinding |
					TextureUsage.StorageBinding |
					TextureUsage.ComputeStorage,
				label: `WebGPUBloomMip${i}_A_${width}x${height}`,
			});
			const texB = resources.shared.compute.createTexture({
				width,
				height,
				format: TextureFormat.RGBA16Float,
				usage:
					TextureUsage.TextureBinding |
					TextureUsage.StorageBinding |
					TextureUsage.ComputeStorage,
				label: `WebGPUBloomMip${i}_B_${width}x${height}`,
			});
			resources.mipTextures.push([texA, texB]);
		}
	}

	private _destroyMipTextures(resources: WebGPUBloomResources): void {
		for (const [texA, texB] of resources.mipTextures) {
			texA.destroy();
			texB.destroy();
		}
		resources.mipTextures = [];
		resources.mipWidth = 0;
		resources.mipHeight = 0;
		resources.mipCount = 0;
		resources.shared.invalidateBindingsByPrefix("bloom-");
	}
}

/**
 * WebGL implementation of the HDR bloom pass.
 */
/** @internal WebGL implementation for the built-in bloom pass. */
export class WebGLBloomImplementation
	implements PostProcessPassImplementation<WebGLBloomContext, BloomOptions>
{
	public readonly id = "bloom:webgl";
	public describeExecution() {
		return WEBGL_VERSIONED_EXECUTION;
	}
	private _programCompiler: WebGLProgramCompiler | null = null;
	private _programSlot: WebGLProgramSlot<WebGLBloomProgram> | null = null;

	public warmup(context: WebGLBloomContext | undefined): void {
		if (context) {
			this._getProgramSlot(context.programCompiler).warmup();
		}
	}

	public execute(
		request: PostProcessPassRequest<BloomOptions>,
		context: WebGLBloomContext | undefined
	): PostProcessPassResult {
		if (!context) {
			return { ran: false };
		}
		if (
			!context.postFramebuffer ||
			!context.sceneColorTexture ||
			!context.fullscreenVao
		) {
			return { ran: false };
		}
		const sourceTexture = context.resources.color.input;
		const targetTexture = context.resources.color.output;
		if (!sourceTexture || !targetTexture) {
			return { ran: false };
		}

		const options = request.options ?? {};
		const threshold = Math.max(
			0,
			finiteOr(options.threshold, DEFAULT_BLOOM_OPTIONS.threshold)
		);
		const softKnee = Math.max(
			1e-4,
			finiteOr(options.softKnee, DEFAULT_BLOOM_OPTIONS.softKnee)
		);
		const intensity = Math.max(
			0,
			finiteOr(options.intensity, DEFAULT_BLOOM_OPTIONS.intensity)
		);
		const radius = clamp(
			finiteOr(options.radius, DEFAULT_BLOOM_OPTIONS.radius),
			0.5,
			4
		);

		const gl = context.gl;
		const bloomProgram = this._getProgramSlot(
			context.programCompiler
		).tryGet();
		if (!bloomProgram) {
			return { ran: false };
		}
		gl.bindFramebuffer(gl.FRAMEBUFFER, context.postFramebuffer);
		context.bindColorTarget(targetTexture);
		gl.viewport(0, 0, context.width, context.height);
		gl.useProgram(bloomProgram.program);
		gl.bindVertexArray(context.fullscreenVao);
		gl.disable(gl.CULL_FACE);
		gl.disable(gl.DEPTH_TEST);
		gl.disable(gl.BLEND);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, sourceTexture);
		if (bloomProgram.uniforms.sourceMap) {
			gl.uniform1i(bloomProgram.uniforms.sourceMap, 0);
		}
		if (bloomProgram.uniforms.texelSize) {
			gl.uniform2f(
				bloomProgram.uniforms.texelSize,
				1 / Math.max(1, context.width),
				1 / Math.max(1, context.height)
			);
		}
		if (bloomProgram.uniforms.bloomParams) {
			gl.uniform4f(
				bloomProgram.uniforms.bloomParams,
				threshold,
				softKnee,
				intensity,
				radius
			);
		}
		context.drawFullscreen();
		gl.bindVertexArray(null);
		return { ran: true };
	}

	public destroy(): void {
		this._programSlot?.destroy();
		this._programSlot = null;
		this._programCompiler = null;
	}

	private _getProgramSlot(
		compiler: WebGLProgramCompiler
	): WebGLProgramSlot<WebGLBloomProgram> {
		if (this._programCompiler !== compiler) {
			this._programSlot?.destroy();
			this._programCompiler = compiler;
			this._programSlot = compiler.createSlot({
				label: "WebGLBloomProgram",
				vertex: () => ShaderSource.get("webgl.part.presentVertex").source.code,
				fragment: () => ShaderSource.get("webgl.part.bloomFragment").source.code,
				reflect: (gl, program) => ({
					program,
					uniforms: {
						sourceMap: gl.getUniformLocation(program, "uSourceMap"),
						texelSize: gl.getUniformLocation(program, "uTexelSize"),
						bloomParams: gl.getUniformLocation(program, "uBloomParams"),
					},
				}),
			});
		}
		return this._programSlot!;
	}
}

export interface BloomPassConfig
	extends Omit<
		PostProcessPassConfig<BloomOptions>,
		| "id"
		| "builtIn"
		| "label"
		| "placement"
		| "order"
		| "implementations"
	> {}

/**
 * Stateful logical bloom pass shared by WebGPU and WebGL backends.
 */
export class BloomPass extends PostProcessPass<BloomOptions, BloomOptions> {
	public constructor(config: BloomPassConfig = {}) {
		super({
			...config,
			id: BLOOM_PASS_ORDER.id,
			schedule: {
				placement: config.schedule?.placement ?? BLOOM_PASS_ORDER.placement,
				order: config.schedule?.order ?? BLOOM_PASS_ORDER.order,
				incremental: config.schedule?.incremental ?? BLOOM_PASS_ORDER.incremental,
			},
			label: "bloom",
			colorContract: config.colorContract ?? {
				input: "scene-linear-hdr",
				output: "scene-linear-hdr",
			},
			implementations: {
				webgpu: () => new WebGPUBloomImplementation(),
				webgl: () => new WebGLBloomImplementation(),
			},
		});
	}

	public override normalizeOptions(): BloomOptions {
		return {
			...DEFAULT_BLOOM_OPTIONS,
			...this.getRawOptions(),
		};
	}
}
