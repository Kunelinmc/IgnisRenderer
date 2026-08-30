import type { Vec3Tuple } from "../../maths/Vector3";
import { ceilDiv } from "../../maths/Misc";
import {
	BufferUsage,
	type IComputePipeline,
	type IRenderBuffer,
	type IShaderModule,
} from "../../backends/types";
import {
	WEBGPU_2D_COMPUTE_WORKGROUP_SIZE as WEBGPU_WORKGROUP_SIZE,
} from "../../backends/webgpu/constants";
import type { WebGPUPostProcessServices } from "../../backends/webgpu/WebGPUPostProcessContracts";
import type {
	WebGLProgramCompiler,
	WebGLProgramSlot,
} from "../../backends/webgl/WebGLProgramCompiler";
import { ShaderSource } from "../../shaders/ShaderSource";
import type { PostProcessIncrementalMetadata } from "../../pipeline/incremental";
import { PostProcessPass, type PostProcessPassConfig } from "../PostProcessPass";
import type { PostProcessScheduleEntry } from "../ordering";
import {
	SOFTWARE_IN_PLACE_EXECUTION,
	WEBGL_VERSIONED_EXECUTION,
	WEBGPU_VERSIONED_EXECUTION,
} from "../executionDeclarations";
import type {
	PostProcessPassImplementation,
	PostProcessPassRequest,
	PostProcessPassResult,
} from "../types";
import type { IRenderBackend } from "../../backends/IRenderBackend";
import type { DisplayOutputState } from "../../rendering/DisplayOutput";
import {
	bindWebGLPostTarget,
	forEachSoftwareDirtyRect,
	resolveWebGLTarget,
	resolveWebGPUTarget,
	type EmptyOptions,
	type SoftwareBuiltinPostProcessContext,
	type WebGLScreenPostProcessContext,
	type WebGPURuntimePostProcessContext,
} from "./ScreenPassShared";

const COLOR_EPSILON = 1e-6;

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
		readonly exposure: WebGLUniformLocation | null;
		readonly hdrHeadroom: WebGLUniformLocation | null;
		readonly hdrEnabled: WebGLUniformLocation | null;
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

/** @internal Reference ACES fitted mapping shared by CPU presentation. */
export function applyACESToneMapping(
	color: Readonly<Vec3Tuple>,
	exposure: number,
	out: Vec3Tuple = [0, 0, 0],
): Vec3Tuple {
	for (let index = 0; index < 3; index++) {
		const value = color[index];
		const exposed = Math.max(0, value * exposure);
		out[index] = Math.min(1, Math.max(0,
			(exposed * (2.51 * exposed + 0.03)) /
			(exposed * (2.43 * exposed + 0.59) + 0.14),
		));
	}
	return out;
}

/**
 * Applies the hue-preserving HDR shoulder used by presentation shaders.
 *
 * @internal Shared by CPU presentation and numerical contract tests.
 */
export function applyHDRSoftShoulder(
	color: Readonly<Vec3Tuple>,
	exposure: number,
	hdrHeadroom: number,
	out: Vec3Tuple = [0, 0, 0],
): Vec3Tuple {
	out[0] = Math.max(0, color[0] * exposure);
	out[1] = Math.max(0, color[1] * exposure);
	out[2] = Math.max(0, color[2] * exposure);
	const peak = Math.max(out[0], out[1], out[2]);
	if (peak <= 1) return out;
	if (hdrHeadroom <= 1.0001) {
		out[0] = Math.min(out[0], 1);
		out[1] = Math.min(out[1], 1);
		out[2] = Math.min(out[2], 1);
		return out;
	}
	const mappedPeak = 1 + (hdrHeadroom - 1) *
		(1 - Math.exp(-(peak - 1) / (hdrHeadroom - 1)));
	const scale = mappedPeak / Math.max(peak, COLOR_EPSILON);
	out[0] *= scale;
	out[1] *= scale;
	out[2] *= scale;
	return out;
}

/** @internal Software implementation for the built-in tone mapping pass. */
export class SoftwareToneMappingImplementation
	implements PostProcessPassImplementation<SoftwareBuiltinPostProcessContext>
{
	public readonly id = "tonemap:software";
	private readonly _mappedColor: Vec3Tuple = [0, 0, 0];
	private readonly _inputColor: Vec3Tuple = [0, 0, 0];

	public describeExecution() {
		return SOFTWARE_IN_PLACE_EXECUTION;
	}

	public execute(
		request: PostProcessPassRequest,
		context: SoftwareBuiltinPostProcessContext | undefined
	): PostProcessPassResult {
		const pixels = context?.resources.color.input;
		if (!context || !(pixels instanceof Float32Array) || pixels.length === 0) {
			return { ran: false };
		}
		const dirtyRects = context.dirtyRects;
		const width = request.frameContext.attachments.width;
		const display = context.displayOutput;
		const exposure = display?.requested.exposure ?? 1;
		const hdr = display?.activeDynamicRange === "hdr";
		const headroom = display?.requested.hdrHeadroom ?? 4;
		forEachSoftwareDirtyRect(dirtyRects, (rect) => {
			for (let y = rect.minY; y <= rect.maxY; y++) {
				const row = y * width;
				for (let x = rect.minX; x <= rect.maxX; x++) {
					const index = (row + x) << 2;
					const alpha = Math.min(1, Math.max(0, pixels[index + 3]));
					const inverseAlpha = alpha > COLOR_EPSILON ? 1 / alpha : 0;
					this._inputColor[0] = pixels[index] * inverseAlpha;
					this._inputColor[1] = pixels[index + 1] * inverseAlpha;
					this._inputColor[2] = pixels[index + 2] * inverseAlpha;
					const mapped = hdr ?
						applyHDRSoftShoulder(
							this._inputColor,
							exposure,
							headroom,
							this._mappedColor,
						)
					:	applyACESToneMapping(
							this._inputColor,
							exposure,
							this._mappedColor,
						);
					pixels[index] = mapped[0] * alpha;
					pixels[index + 1] = mapped[1] * alpha;
					pixels[index + 2] = mapped[2] * alpha;
				}
			}
		});
		return { ran: true };
	}
}
interface WebGPUToneMappingResources {
	shared: WebGPUPostProcessServices;
	module: IShaderModule | null;
	pipeline: IComputePipeline | null;
	params: IRenderBuffer | null;
	paramData: Float32Array<ArrayBuffer>;
}
/** @internal WebGPU implementation for the built-in tone mapping pass. */
export class WebGPUToneMappingImplementation
	implements PostProcessPassImplementation<WebGPUToneMappingContext, EmptyOptions>
{
	public readonly id = "tonemap:webgpu";
	public describeExecution() {
		return WEBGPU_VERSIONED_EXECUTION;
	}
	private _resources =
		new Map<WebGPUPostProcessServices, WebGPUToneMappingResources>();

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
			resources.shared.destroyManagedResource(
				resources.params,
				"tone mapping params buffer",
			);
			resources.shared.invalidateBindingsByPrefix("tonemap-");
			resources.module = null;
			resources.pipeline = null;
			resources.params = null;
		}
		this._resources.clear();
	}

	private async _runToneMappingKernel(
		context: WebGPUToneMappingContext
	): Promise<boolean> {
		const resources = await this._ensureResources(context.shared);
		if (
			!context.encoder ||
			!context.targets ||
			!resources.pipeline ||
			!resources.params
		) {
			return false;
		}
		const display = context.shared.getDisplayOutputState?.();
		resources.paramData[0] = display?.requested.exposure ?? 1;
		resources.paramData[1] = display?.requested.hdrHeadroom ?? 4;
		resources.paramData[2] =
			display?.activeDynamicRange === "hdr" ? 1 : 0;
		resources.paramData[3] = 0;
		context.shared.compute.writeBuffer(resources.params, resources.paramData);
		const targets = context.targets;
		const target = resolveWebGPUTarget(context);
		const input = context.resources.color.input;
		if (!input) return false;
		const binding = context.shared.getCachedBindGroup(
			`tonemap-${target === targets.postPing ? "ping" : "pong"}`,
			resources.pipeline,
			[
				{ binding: 0, resource: input },
				{ binding: 1, resource: resources.params },
				{ binding: 2, resource: target },
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
		shared: WebGPUPostProcessServices
	): Promise<WebGPUToneMappingResources> {
		let resources = this._resources.get(shared);
		if (!resources) {
			resources = {
				shared,
				module: null,
				pipeline: null,
				params: null,
				paramData: new Float32Array(4),
			};
			this._resources.set(shared, resources);
		}
		if (!resources.module) {
			const shader = await ShaderSource.load(
				"webgpu.postprocess.toneMapping"
			);
			resources.module = await shared.compute.createShaderModule({
				label: "WebGPUToneMappingShader",
				code: shader.source.code,
				sourceMap: shader.source.sourceMap,
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
		if (!resources.params) {
			resources.params = shared.compute.createBuffer({
				label: "WebGPUToneMappingParams",
				size: 16,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
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
	private readonly _backend: IRenderBackend | undefined;

	public constructor(backend?: IRenderBackend) {
		this._backend = backend;
	}
	public describeExecution() {
		return WEBGL_VERSIONED_EXECUTION;
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
		const display = resolveBackendDisplayOutput(this._backend);
		if (program.uniforms.exposure) {
			gl.uniform1f(program.uniforms.exposure, display?.requested.exposure ?? 1);
		}
		if (program.uniforms.hdrHeadroom) {
			gl.uniform1f(
				program.uniforms.hdrHeadroom,
				display?.requested.hdrHeadroom ?? 4,
			);
		}
		if (program.uniforms.hdrEnabled) {
			gl.uniform1f(
				program.uniforms.hdrEnabled,
				display?.activeDynamicRange === "hdr" ? 1 : 0,
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
	): WebGLProgramSlot<WebGLToneMappingProgram> {
		if (this._programCompiler !== compiler) {
			this._programSlot?.destroy();
			this._programCompiler = compiler;
			this._programSlot = compiler.createSlot({
				label: "WebGLToneMappingProgram",
				vertex: () => ShaderSource.get("webgl.part.presentVertex").source.code,
				fragment: () =>
					ShaderSource.get("webgl.part.toneMappingFragment").source.code,
				reflect: (gl, webglProgram) => ({
					program: webglProgram,
					uniforms: {
						sourceMap: gl.getUniformLocation(
							webglProgram,
							"uSourceMap"
						),
						exposure: gl.getUniformLocation(
							webglProgram,
							"uExposure",
						),
						hdrHeadroom: gl.getUniformLocation(
							webglProgram,
							"uHdrHeadroom",
						),
						hdrEnabled: gl.getUniformLocation(
							webglProgram,
							"uHdrEnabled",
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
			| "label"
			| "implementations"
		> = {}
	) {
		super({
			...config,
			alphaContract: "premultiplied",
			id: TONE_MAPPING_PASS_ORDER.id,
			schedule: {
				placement: config.schedule?.placement ?? TONE_MAPPING_PASS_ORDER.placement,
				order: config.schedule?.order ?? TONE_MAPPING_PASS_ORDER.order,
				incremental: config.schedule?.incremental ?? TONE_MAPPING_PASS_ORDER.incremental,
			},
			builtIn: true,
			label: "tone mapping",
			colorContract: config.colorContract ?? {
				input: "scene-linear-hdr",
				output: "display-linear",
			},
			implementations: {
				software: () => new SoftwareToneMappingImplementation(),
				webgpu: () => new WebGPUToneMappingImplementation(),
				webgl: (backend) => new WebGLToneMappingImplementation(backend),
			},
		});
	}
}
function resolveBackendDisplayOutput(
	backend?: IRenderBackend,
): DisplayOutputState | null {
	if (!backend) return null;
	const getter = (backend as Partial<IRenderBackend>).getDisplayOutputState;
	return typeof getter === "function" ?
		getter.call(backend) : null;
}
