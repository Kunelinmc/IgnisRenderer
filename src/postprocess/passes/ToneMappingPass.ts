import { clamp } from "../../maths/Common";
import { ceilDiv } from "../../maths/Misc";
import { type IComputePipeline, type IShaderModule } from "../../backends/types";
import {
	WEBGPU_2D_COMPUTE_WORKGROUP_SIZE as WEBGPU_WORKGROUP_SIZE,
} from "../../backends/webgpu/constants";
import type { PostProcessSharedContext } from "../../backends/webgpu/postprocess/PostProcessSharedContext";
import type {
	WebGLProgramCompiler,
	WebGLProgramSlot,
} from "../../backends/webgl/WebGLProgramCompiler";
import { ShaderSource } from "../../shaders/ShaderSource";
import type { PostProcessIncrementalMetadata } from "../../pipeline/incremental";
import { PostProcessPass, type PostProcessPassConfig } from "../PostProcessPass";
import type { PostProcessScheduleEntry } from "../ordering";
import { createPostProcessExecutionDeclaration } from "../executionDeclarations";
import type {
	PostProcessPassImplementation,
	PostProcessPassRequest,
	PostProcessPassResult,
} from "../types";
import {
	bindWebGLPostTarget,
	forEachSoftwareDirtyRect,
	resolveSoftwareDirtyRects,
	resolveWebGLTarget,
	resolveWebGPUTarget,
	type EmptyOptions,
	type SoftwareBuiltinPostProcessContext,
	type WebGLScreenPostProcessContext,
	type WebGPURuntimePostProcessContext,
} from "./ScreenPassShared";

export const TONE_MAPPING_PASS_ID = "tonemap";
export const TONE_MAPPING_PASS_INCREMENTAL = {
	firstPass: TONE_MAPPING_PASS_ID,
	grade: "light",
	inflationRadius: 0,
} as const satisfies PostProcessIncrementalMetadata;

interface WebGLToneMappingProgram {
	readonly program: WebGLProgram;
	readonly uniforms: {
		readonly sourceMap: WebGLUniformLocation | null;
	};
}
export const TONE_MAPPING_PASS_ORDER = {
	id: TONE_MAPPING_PASS_ID,
	placement: "hdr",
	order: 600,
	incremental: TONE_MAPPING_PASS_INCREMENTAL,
} as const satisfies PostProcessScheduleEntry;
export type WebGPUToneMappingContext = WebGPURuntimePostProcessContext;
export type WebGLToneMappingContext = WebGLScreenPostProcessContext;

/** @internal Software implementation for the built-in tone mapping pass. */
export class SoftwareToneMappingImplementation
	implements PostProcessPassImplementation<SoftwareBuiltinPostProcessContext>
{
	public readonly id = "tonemap:software";
	public describeExecution() {
		return createPostProcessExecutionDeclaration("software");
	}

	public execute(
		request: PostProcessPassRequest,
		_context: SoftwareBuiltinPostProcessContext | undefined
	): PostProcessPassResult {
		const pixels = request.frameContext.attachments.pixels;
		if (!pixels || pixels.length === 0) {
			return { ran: false };
		}
		const dirtyRects = resolveSoftwareDirtyRects(request.frameContext);
		const width = request.frameContext.attachments.width;
		forEachSoftwareDirtyRect(dirtyRects, (rect) => {
			for (let y = rect.minY; y <= rect.maxY; y++) {
				const row = y * width;
				for (let x = rect.minX; x <= rect.maxX; x++) {
					const index = (row + x) << 2;
					const red = applyAcesToneMap(pixels[index] / 255);
					const green = applyAcesToneMap(pixels[index + 1] / 255);
					const blue = applyAcesToneMap(pixels[index + 2] / 255);
					pixels[index] = Math.round(red * 255);
					pixels[index + 1] = Math.round(green * 255);
					pixels[index + 2] = Math.round(blue * 255);
				}
			}
		});
		return { ran: true };
	}
}
interface WebGPUToneMappingResources {
	shared: PostProcessSharedContext;
	module: IShaderModule | null;
	pipeline: IComputePipeline | null;
}
/** @internal WebGPU implementation for the built-in tone mapping pass. */
export class WebGPUToneMappingImplementation
	implements PostProcessPassImplementation<WebGPUToneMappingContext, EmptyOptions>
{
	public readonly id = "tonemap:webgpu";
	public describeExecution() {
		return createPostProcessExecutionDeclaration("webgpu");
	}
	private _resources =
		new Map<PostProcessSharedContext, WebGPUToneMappingResources>();

	public async warmup(
		context: WebGPUToneMappingContext | undefined
	): Promise<void> {
		if (context) {
			await this._ensureResources(context.shared);
		}
	}

	public async execute(
		_request: PostProcessPassRequest<EmptyOptions>,
		context: WebGPUToneMappingContext | undefined
	): Promise<PostProcessPassResult> {
		if (!context?.encoder || !context.targets) {
			return { ran: false };
		}
		const ran = await this._runToneMappingKernel(context);
		return ran ? { ran: true } : { ran: false };
	}

	public invalidate(): void {
		for (const resources of this._resources.values()) {
			resources.shared.invalidateBindingsByPrefix("tonemap-");
		}
	}

	public destroy(): void {
		for (const resources of this._resources.values()) {
			resources.shared.destroyManagedResource(
				resources.pipeline,
				"tone mapping pipeline"
			);
			resources.shared.destroyManagedResource(
				resources.module,
				"tone mapping shader module"
			);
			resources.shared.invalidateBindingsByPrefix("tonemap-");
			resources.module = null;
			resources.pipeline = null;
		}
		this._resources.clear();
	}

	private async _runToneMappingKernel(
		context: WebGPUToneMappingContext
	): Promise<boolean> {
		const resources = await this._ensureResources(context.shared);
		if (!context.encoder || !context.targets || !resources.pipeline) {
			return false;
		}
		const targets = context.targets;
		const target = resolveWebGPUTarget(context);
		const input = context.resources.color.input;
		if (!input) return false;
		const binding = context.shared.getCachedBindGroup(
			`tonemap-${target === targets.postPing ? "ping" : "pong"}`,
			resources.pipeline,
			[
				{ binding: 0, resource: input },
				{ binding: 1, resource: target },
			],
			"WebGPUToneMapping_Binding"
		);
		context.encoder.beginComputePass({ label: "WebGPUToneMapping" });
		context.encoder.setComputePipeline(resources.pipeline);
		context.encoder.setBindingGroup(0, binding);
		context.encoder.dispatchWorkgroups(
			ceilDiv(target.width, WEBGPU_WORKGROUP_SIZE),
			ceilDiv(target.height, WEBGPU_WORKGROUP_SIZE),
			1
		);
		context.encoder.endComputePass();
		return true;
	}

	private async _ensureResources(
		shared: PostProcessSharedContext
	): Promise<WebGPUToneMappingResources> {
		let resources = this._resources.get(shared);
		if (!resources) {
			resources = {
				shared,
				module: null,
				pipeline: null,
			};
			this._resources.set(shared, resources);
		}
		if (!resources.module) {
			const shader = await ShaderSource.load(
				"webgpu.postprocess.toneMapping.composite"
			);
			resources.module = await shared.compute.createShaderModule({
				label: "WebGPUToneMappingShader",
				code: shader.code,
				sourceMap: shader.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!resources.pipeline) {
			resources.pipeline = await shared.compute.createComputePipeline({
				label: "WebGPUToneMappingPipeline",
				compute: { module: resources.module, entryPoint: "csMain" },
			});
		}
		return resources;
	}
}
/** @internal WebGL implementation for the built-in tone mapping pass. */
export class WebGLToneMappingImplementation
	implements PostProcessPassImplementation<WebGLToneMappingContext, EmptyOptions>
{
	public readonly id = "tonemap:webgl";
	public describeExecution() {
		return createPostProcessExecutionDeclaration("webgl");
	}
	private _programCompiler: WebGLProgramCompiler | null = null;
	private _programSlot: WebGLProgramSlot<WebGLToneMappingProgram> | null = null;

	public warmup(context: WebGLToneMappingContext | undefined): void {
		if (context) {
			this._getProgramSlot(context.programCompiler).warmup();
		}
	}

	public execute(
		_request: PostProcessPassRequest<EmptyOptions>,
		context: WebGLToneMappingContext | undefined
	): PostProcessPassResult {
		const target = resolveWebGLTarget(context);
		if (!target) {
			return { ran: false };
		}
		const gl = context.gl;
		const program = this._getProgramSlot(context.programCompiler).tryGet();
		if (!program) {
			return { ran: false };
		}
		bindWebGLPostTarget(context, program.program, target.texture);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, target.source);
		if (program.uniforms.sourceMap) {
			gl.uniform1i(program.uniforms.sourceMap, 0);
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
	): WebGLProgramSlot<WebGLToneMappingProgram> {
		if (this._programCompiler !== compiler) {
			this._programSlot?.destroy();
			this._programCompiler = compiler;
			this._programSlot = compiler.createSlot({
				label: "WebGLToneMappingProgram",
				vertex: () => ShaderSource.get("webgl.part.presentVertex.raw"),
				fragment: () =>
					ShaderSource.get("webgl.part.toneMappingFragment.raw"),
				reflect: (gl, webglProgram) => ({
					program: webglProgram,
					uniforms: {
						sourceMap: gl.getUniformLocation(
							webglProgram,
							"uSourceMap"
						),
					},
				}),
			});
		}
		return this._programSlot!;
	}
}
/**
 * Stateful logical tone mapping pass.
 */
export class ToneMappingPass extends PostProcessPass<EmptyOptions, EmptyOptions> {
	public constructor(
		config: Omit<
			PostProcessPassConfig<EmptyOptions>,
			| "id"
			| "builtIn"
			| "warningLabel"
			| "implementations"
		> = {}
	) {
		super({
			...config,
			id: TONE_MAPPING_PASS_ORDER.id,
			schedule: {
				placement: config.schedule?.placement ?? TONE_MAPPING_PASS_ORDER.placement,
				order: config.schedule?.order ?? TONE_MAPPING_PASS_ORDER.order,
				incremental: config.schedule?.incremental ?? TONE_MAPPING_PASS_ORDER.incremental,
			},
			builtIn: true,
			warningLabel: "tone mapping",
			implementations: {
				software: () => new SoftwareToneMappingImplementation(),
				webgpu: () => new WebGPUToneMappingImplementation(),
				webgl: () => new WebGLToneMappingImplementation(),
			},
		});
	}
}
function applyAcesToneMap(value: number): number {
	const a = 2.51;
	const b = 0.03;
	const c = 2.43;
	const d = 0.59;
	const e = 0.14;
	const mapped = (value * (a * value + b)) / (value * (c * value + d) + e);
	return clamp(mapped, 0, 1);
}
