import type { ICommandEncoder } from "../ICommandEncoder";
import {
	BufferUsage,
	TextureFormat,
	type IBindingGroup,
	type IComputePipeline,
	type IRenderBuffer,
	type IRenderTexture,
	type ISampler,
	type IShaderModule,
} from "../types";
import { ceilDiv, finiteOr } from "../../maths/Misc";
import { ShaderSource } from "../../shaders/ShaderSource";
import type { IWebGPUComputeFacade } from "./ComputeFacade";
import { WEBGPU_2D_COMPUTE_WORKGROUP_SIZE as WORKGROUP_SIZE } from "./constants";

const DENOISE_PARAM_FLOATS = 16;
const DENOISE_TILE_HALO = 8;
const MAX_DENOISE_RADIUS = 4;
const MAX_DENOISE_STEP_WIDTH = 4;
const QUALITY_STEPS = [1, 2, 4] as const;

export type WebGPUDenoiseMode = "fast" | "quality";
export type WebGPUDenoiseSignal = "radiance-confidence" | "scalar";

/**
 * Internal tuning controls for the shared WebGPU denoiser.
 *
 * @internal Owned by WebGPU post-process implementations.
 */
export interface WebGPUDenoiseOptions {
	readonly mode?: WebGPUDenoiseMode;
	readonly signal?: WebGPUDenoiseSignal;
	readonly radius?: number;
	readonly depthPhi?: number;
	readonly normalPhi?: number;
	readonly valuePhi?: number;
	readonly confidenceFloor?: number;
}

/** @internal Fully resolved shared WebGPU denoiser options. */
export interface ResolvedWebGPUDenoiseOptions {
	readonly mode: WebGPUDenoiseMode;
	readonly signal: WebGPUDenoiseSignal;
	readonly radius: number;
	readonly depthPhi: number;
	readonly normalPhi: number;
	readonly valuePhi: number;
	readonly confidenceFloor: number;
}

export const DEFAULT_WEBGPU_DENOISE_OPTIONS:
	Readonly<ResolvedWebGPUDenoiseOptions> = {
		mode: "fast",
		signal: "radiance-confidence",
		radius: 2,
		depthPhi: 24,
		normalPhi: 16,
		valuePhi: 2,
		confidenceFloor: 0.05,
	};

/**
 * One command-recording request for the shared WebGPU denoiser.
 *
 * `output` may alias `source`; `scratch` must be distinct from both.
 *
 * @internal Owned by WebGPU post-process implementations.
 */
export interface WebGPUDenoiseRequest {
	readonly scope: string;
	readonly encoder: ICommandEncoder;
	readonly source: IRenderTexture;
	readonly scratch: IRenderTexture;
	readonly output: IRenderTexture;
	readonly depth: IRenderTexture;
	readonly normal: IRenderTexture;
	readonly sampler: ISampler;
	readonly options?: WebGPUDenoiseOptions;
}

/** @internal Result of recording one shared WebGPU denoise operation. */
export interface WebGPUDenoiseResult {
	readonly texture: IRenderTexture;
	readonly mode: WebGPUDenoiseMode;
	readonly dispatchCount: number;
}

interface CachedBinding {
	group: IBindingGroup;
	resources: readonly unknown[];
}

interface ScopeResources {
	readonly params: Array<IRenderBuffer | null>;
	readonly paramData: Float32Array[];
	lastEncoder: ICommandEncoder | null;
	invocationIndex: number;
}

/**
 * Resolves and clamps shared WebGPU denoiser options.
 *
 * @param options Caller-provided internal denoiser options.
 * @returns Stable options suitable for parameter packing and dispatch.
 * @sideEffects None.
 * @internal
 */
export function resolveWebGPUDenoiseOptions(
	options?: WebGPUDenoiseOptions | null
): ResolvedWebGPUDenoiseOptions {
	const mode =
		options?.mode === "quality" ?
			"quality"
		:	DEFAULT_WEBGPU_DENOISE_OPTIONS.mode;
	const signal =
		options?.signal === "scalar" ?
			"scalar"
		:	DEFAULT_WEBGPU_DENOISE_OPTIONS.signal;
	return {
		mode,
		signal,
		radius:
			mode === "quality" ?
				2
			:	clampInteger(
					options?.radius,
					DEFAULT_WEBGPU_DENOISE_OPTIONS.radius,
					1,
					MAX_DENOISE_RADIUS
				),
		depthPhi: nonNegativeFiniteOr(
			options?.depthPhi,
			DEFAULT_WEBGPU_DENOISE_OPTIONS.depthPhi
		),
		normalPhi: nonNegativeFiniteOr(
			options?.normalPhi,
			DEFAULT_WEBGPU_DENOISE_OPTIONS.normalPhi
		),
		valuePhi: nonNegativeFiniteOr(
			options?.valuePhi,
			DEFAULT_WEBGPU_DENOISE_OPTIONS.valuePhi
		),
		confidenceFloor: clamp(
			finiteOr(
				options?.confidenceFloor,
				DEFAULT_WEBGPU_DENOISE_OPTIONS.confidenceFloor
			),
			0,
			1
		),
	};
}

/**
 * Packs one denoise dispatch parameter block into a pre-allocated array.
 *
 * @param target Sixteen-float destination array.
 * @param width Signal width.
 * @param height Signal height.
 * @param stepWidth Pixel step used by this scale.
 * @param options Resolved denoiser options.
 * @returns The supplied destination array.
 * @sideEffects Overwrites `target`.
 * @internal
 */
export function writeWebGPUDenoiseParams(
	target: Float32Array,
	width: number,
	height: number,
	stepWidth: number,
	options: ResolvedWebGPUDenoiseOptions
): Float32Array {
	if (target.length !== DENOISE_PARAM_FLOATS) {
		throw new Error(
			`WebGPU denoise parameter target must contain ` +
				`${DENOISE_PARAM_FLOATS} floats; received ${target.length}.`
		);
	}
	const radius = clampInteger(
		options.radius,
		DEFAULT_WEBGPU_DENOISE_OPTIONS.radius,
		1,
		MAX_DENOISE_RADIUS
	);
	// Keep the dilated filter footprint inside the shader's fixed tile halo.
	const maximumStepWidth = Math.min(
		MAX_DENOISE_STEP_WIDTH,
		Math.max(1, Math.floor(DENOISE_TILE_HALO / radius))
	);
	target[0] = 1 / Math.max(finiteOr(width, 1), 1);
	target[1] = 1 / Math.max(finiteOr(height, 1), 1);
	target[2] = radius;
	target[3] = clampInteger(stepWidth, 1, 1, maximumStepWidth);
	target[4] = nonNegativeFiniteOr(
		options.depthPhi,
		DEFAULT_WEBGPU_DENOISE_OPTIONS.depthPhi
	);
	target[5] = nonNegativeFiniteOr(
		options.normalPhi,
		DEFAULT_WEBGPU_DENOISE_OPTIONS.normalPhi
	);
	target[6] = nonNegativeFiniteOr(
		options.valuePhi,
		DEFAULT_WEBGPU_DENOISE_OPTIONS.valuePhi
	);
	target[7] = clamp(
		finiteOr(
			options.confidenceFloor,
			DEFAULT_WEBGPU_DENOISE_OPTIONS.confidenceFloor
		),
		0,
		1
	);
	target[8] = options.signal === "scalar" ? 1 : 0;
	target[9] = options.mode === "quality" ? 1 : 0;
	target[10] = 0;
	target[11] = 0;
	target[12] = 0;
	target[13] = 0;
	target[14] = 0;
	target[15] = 0;
	return target;
}

/** @internal Device-lifetime shared WebGPU edge-aware denoiser. */
export class WebGPUDenoiser {
	private _module: IShaderModule | null = null;
	private _horizontalPipeline: IComputePipeline | null = null;
	private _verticalPipeline: IComputePipeline | null = null;
	private readonly _bindings = new Map<string, CachedBinding>();
	private readonly _scopes = new Map<string, ScopeResources>();

	constructor(private readonly _compute: IWebGPUComputeFacade) {}

	public async ensureResources(): Promise<void> {
		if (!this._module) {
			const shader = await ShaderSource.load(
				"webgpu.postprocess.denoise"
			);
			this._module = await this._compute.createShaderModule({
				label: "WebGPUDenoiseShader",
				code: shader.source.code,
				sourceMap: shader.source.sourceMap,
				language: "wgsl",
				stage: "compute",
				sourceKind: "postprocess",
			});
		}
		if (!this._horizontalPipeline) {
			this._horizontalPipeline =
				await this._compute.createComputePipeline({
					label: "WebGPUDenoiseHorizontalPipeline",
					compute: {
						module: this._module,
						entryPoint: "csDenoiseHorizontal",
					},
				});
		}
		if (!this._verticalPipeline) {
			this._verticalPipeline =
				await this._compute.createComputePipeline({
					label: "WebGPUDenoiseVerticalPipeline",
					compute: {
						module: this._module,
						entryPoint: "csDenoiseVertical",
					},
				});
		}
	}

	/**
	 * Records edge-aware denoise dispatches into the supplied encoder.
	 *
	 * Repeated calls with one `scope` and encoder use separate parameter lanes
	 * so queued writes cannot alter earlier dispatches.
	 */
	public async encode(
		request: WebGPUDenoiseRequest
	): Promise<WebGPUDenoiseResult> {
		const options = resolveWebGPUDenoiseOptions(request.options);
		this._validateRequest(request);
		await this.ensureResources();
		if (!this._horizontalPipeline || !this._verticalPipeline) {
			throw new Error(
				`WebGPU denoiser resources are incomplete for "${request.scope}".`
			);
		}
		const scope = this._getScopeResources(request.scope);
		if (scope.lastEncoder === request.encoder) {
			scope.invocationIndex++;
		} else {
			scope.lastEncoder = request.encoder;
			scope.invocationIndex = 0;
		}

		const steps =
			options.mode === "quality" ? QUALITY_STEPS : ([1] as const);
		let source = request.source;
		let dispatchCount = 0;
		for (let index = 0; index < steps.length; index++) {
			const paramIndex = scope.invocationIndex * QUALITY_STEPS.length + index;
			const params = this._ensureParamBuffer(
				request.scope,
				scope,
				paramIndex
			);
			this._compute.writeBuffer(
				params,
				writeWebGPUDenoiseParams(
					scope.paramData[paramIndex],
					request.output.width,
					request.output.height,
					steps[index],
					options
				) as unknown as BufferSource
			);
			this._dispatch(
				request,
				source,
				request.scratch,
				params,
				this._horizontalPipeline,
				"horizontal",
				scope.invocationIndex,
				index,
				steps[index],
				options.mode
			);
			this._dispatch(
				request,
				request.scratch,
				request.output,
				params,
				this._verticalPipeline,
				"vertical",
				scope.invocationIndex,
				index,
				steps[index],
				options.mode
			);
			source = request.output;
			dispatchCount += 2;
		}
		return {
			texture: request.output,
			mode: options.mode,
			dispatchCount,
		};
	}

	public invalidateBindings(): void {
		for (const binding of this._bindings.values()) {
			this._destroy(binding.group);
		}
		this._bindings.clear();
		for (const scope of this._scopes.values()) {
			scope.lastEncoder = null;
			scope.invocationIndex = 0;
		}
	}

	public invalidateShaderResources(): void {
		this.invalidateBindings();
		this._destroy(this._horizontalPipeline);
		this._destroy(this._verticalPipeline);
		this._destroy(this._module);
		this._horizontalPipeline = null;
		this._verticalPipeline = null;
		this._module = null;
		for (const scope of this._scopes.values()) {
			for (const params of scope.params) {
				this._destroy(params);
			}
		}
		this._scopes.clear();
	}

	public destroy(): void {
		this.invalidateShaderResources();
	}

	private _dispatch(
		request: WebGPUDenoiseRequest,
		source: IRenderTexture,
		output: IRenderTexture,
		params: IRenderBuffer,
		pipeline: IComputePipeline,
		axis: "horizontal" | "vertical",
		invocation: number,
		iteration: number,
		stepWidth: number,
		mode: WebGPUDenoiseMode
	): void {
		const binding = this._getBinding(
			`${request.scope}:${invocation}:${axis}:${iteration}`,
			pipeline,
			[
				{ binding: 0, resource: source },
				{ binding: 1, resource: request.normal },
				{ binding: 2, resource: request.depth },
				{ binding: 3, resource: request.sampler },
				{ binding: 4, resource: params },
				{ binding: 5, resource: output },
			],
			`WebGPUDenoise_${request.scope}_${axis}_${iteration}`
		);
		const axisLabel = axis === "horizontal" ? "H" : "V";
		request.encoder.beginComputePass({
			label:
				`WebGPUDenoise_${request.scope}_${mode}_${axisLabel}_` +
				`${stepWidth}`,
		});
		request.encoder.setComputePipeline(pipeline);
		request.encoder.setBindingGroup(0, binding);
		request.encoder.dispatchWorkgroups(
			ceilDiv(output.width, WORKGROUP_SIZE),
			ceilDiv(output.height, WORKGROUP_SIZE),
			1
		);
		request.encoder.endComputePass();
	}

	private _getScopeResources(scope: string): ScopeResources {
		let resources = this._scopes.get(scope);
		if (!resources) {
			resources = {
				params: [],
				paramData: [],
				lastEncoder: null,
				invocationIndex: 0,
			};
			this._scopes.set(scope, resources);
		}
		return resources;
	}

	private _ensureParamBuffer(
		scopeId: string,
		scope: ScopeResources,
		index: number
	): IRenderBuffer {
		if (!scope.params[index]) {
			scope.params[index] = this._compute.createBuffer({
				label: `WebGPUDenoiseParams_${scopeId}_${index}`,
				size: DENOISE_PARAM_FLOATS * Float32Array.BYTES_PER_ELEMENT,
				usage: BufferUsage.Uniform | BufferUsage.CopyDst,
			});
			scope.paramData[index] = new Float32Array(DENOISE_PARAM_FLOATS);
		}
		return scope.params[index]!;
	}

	private _getBinding(
		key: string,
		pipeline: IComputePipeline,
		entries: Array<{ binding: number; resource: unknown }>,
		label: string
	): IBindingGroup {
		const resources = entries.map((entry) => entry.resource);
		const cached = this._bindings.get(key);
		if (
			cached &&
			cached.resources.length === resources.length &&
			cached.resources.every(
				(resource, index) => resource === resources[index]
			)
		) {
			return cached.group;
		}
		if (cached) {
			this._destroy(cached.group);
		}
		const group = this._compute.createBindingGroup({
			pipeline,
			layoutIndex: 0,
			entries: entries as Array<{ binding: number; resource: any }>,
			label,
		});
		this._bindings.set(key, { group, resources });
		return group;
	}

	private _validateRequest(request: WebGPUDenoiseRequest): void {
		if (request.scope.trim().length === 0) {
			throw new Error("WebGPU denoiser scope must not be empty.");
		}
		if (
			request.scratch === request.source ||
			request.scratch === request.output
		) {
			throw new Error(
				`WebGPU denoiser scope "${request.scope}" requires a scratch ` +
					"texture distinct from source and output."
			);
		}
		const width = request.source.width;
		const height = request.source.height;
		for (const [name, texture] of [
			["source", request.source],
			["scratch", request.scratch],
			["output", request.output],
		] as const) {
			if (texture.width !== width || texture.height !== height) {
				throw new Error(
					`WebGPU denoiser scope "${request.scope}" ${name} texture ` +
						`size ${texture.width}x${texture.height} does not match ` +
						`${width}x${height}.`
				);
			}
			const format = texture.format ?? texture.requestedFormat;
			if (format && format !== TextureFormat.RGBA16Float) {
				throw new Error(
					`WebGPU denoiser scope "${request.scope}" ${name} texture ` +
						`must use rgba16float; received ${format}.`
				);
			}
		}
	}

	private _destroy(resource: unknown): void {
		try {
			(resource as { destroy?: () => void } | null)?.destroy?.();
		} catch {
			/* Device loss may already release resources. */
		}
	}
}

function clampInteger(
	value: unknown,
	fallback: number,
	minimum: number,
	maximum: number
): number {
	return clamp(Math.floor(finiteOr(value, fallback)), minimum, maximum);
}

function nonNegativeFiniteOr(value: unknown, fallback: number): number {
	return Math.max(0, finiteOr(value, fallback));
}

function clamp(value: number, minimum: number, maximum: number): number {
	return Math.min(maximum, Math.max(minimum, value));
}
