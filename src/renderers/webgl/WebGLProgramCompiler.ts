import { Logger } from "../../foundation/Logger";
import {
	DEFAULT_SHADER_DIRECTIVE_PROFILE_REGISTRY,
	ShaderBackendCompileStage,
	ShaderCompileError,
	createInlineShaderSourceMap,
	parseWebGLShaderInfoLog,
	type ShaderCompilerMessage,
	type ShaderProcessResult,
	type ShaderRuntime,
	type ShaderSourceSegmentMap,
} from "../../shaders/runtime";

export interface WebGLProgramResource {
	readonly program: WebGLProgram;
}

export interface WebGLShaderCompileMetadata {
	readonly sourceMap?: ShaderSourceSegmentMap | null;
	readonly variantKey?: string;
	readonly materialId?: string;
	readonly sourceKind?: "custom-material" | "unknown";
}

export interface WebGLShaderSourceDescriptor {
	readonly code: string;
	readonly metadata?: WebGLShaderCompileMetadata;
}

export interface WebGLProgramDescriptor<
	TProgram extends WebGLProgramResource,
> {
	readonly label: string;
	readonly vertex: () => string | WebGLShaderSourceDescriptor;
	readonly fragment: () => string | WebGLShaderSourceDescriptor;
	reflect(gl: WebGL2RenderingContext, program: WebGLProgram): TProgram;
}

export interface WebGLProgramSlot<
	TProgram extends WebGLProgramResource,
> {
	readonly label: string;
	get(): TProgram;
	tryGet(): TProgram | null;
	warmup(): WebGLProgramWarmupHandle;
	invalidate(): void;
	destroy(): void;
}

export type WebGLProgramCompileState = "idle" | "pending" | "ready" | "failed";

export interface WebGLProgramWarmupHandle {
	readonly label: string;
	isComplete(): boolean;
	finalize(): void;
}

export interface WebGLProgramCompilerOptions {
	readonly validatePrograms?: boolean;
	readonly onProgramCompilePending?: () => void;
	readonly warn?: (key: string, message: string) => void;
}

interface WebGLParallelShaderCompileExtension {
	readonly COMPLETION_STATUS_KHR: number;
}

interface WebGLPendingShaderCompile {
	readonly shader: WebGLShader;
	readonly stage: "vertex" | "fragment";
	readonly label: string;
	readonly sourceKind: "custom-material" | "unknown";
	readonly variantKey?: string;
	readonly materialId?: string;
	readonly code: string;
	readonly sourceMap: ShaderSourceSegmentMap | null;
}

interface WebGLPendingProgramCompile {
	readonly label: string;
	readonly vertex: WebGLPendingShaderCompile;
	readonly fragment: WebGLPendingShaderCompile;
	readonly program: WebGLProgram;
	readonly startedFrame: number;
	readonly generation: number;
	readonly labelGeneration: number;
	status: "pending" | "ready" | "failed";
	error: unknown;
}

interface WebGLProgramSlotOwner {
	readonly label: string;
	invalidateFromCompiler(): void;
	destroyFromCompiler(): void;
}

const WEBGL_FALLBACK_READY_FRAME_DELAY = 2;
const WEBGL_FALLBACK_FINALIZE_BUDGET_PER_FRAME = 1;

/**
 * Compiles and tracks WebGL programs independently from concrete render passes.
 *
 * @internal WebGL backend infrastructure. Program definitions should be owned by
 * the subsystem that executes them.
 */
export class WebGLProgramCompiler {
	private readonly _gl: WebGL2RenderingContext;
	private readonly _shaderRuntime: ShaderRuntime | null;
	private readonly _shaderCompileStage: ShaderBackendCompileStage | null;
	private readonly _parallelShaderCompile: WebGLParallelShaderCompileExtension | null;
	private readonly _validatePrograms: boolean;
	private readonly _onProgramCompilePending: (() => void) | null;
	private readonly _warnCallback: ((key: string, message: string) => void) | null;
	private _disposeShaderRuntimeListener: (() => void) | null = null;
	private readonly _pendingProgramCompiles = new Map<
		string,
		WebGLPendingProgramCompile
	>();
	private readonly _precompiledPrograms = new Map<string, WebGLProgram>();
	private readonly _slots = new Map<string, WebGLProgramSlotOwner>();
	private readonly _labelGenerations = new Map<string, number>();
	private readonly _warmupHandleLog: WebGLProgramWarmupHandle[] = [];
	private readonly _invalidateListeners = new Set<() => void>();
	private _compileFrameIndex = 0;
	private _fallbackFinalizesThisFrame = 0;
	private _lastPendingNotificationFrame = -1;
	private _generation = 0;
	private _destroyed = false;

	public constructor(
		gl: WebGL2RenderingContext,
		shaderRuntime?: ShaderRuntime,
		shaderCompileStage?: ShaderBackendCompileStage,
		options: WebGLProgramCompilerOptions = {}
	) {
		this._gl = gl;
		this._shaderRuntime = shaderRuntime ?? null;
		this._shaderCompileStage =
			shaderCompileStage ??
			(this._shaderRuntime ?
				new ShaderBackendCompileStage({
					backend: "webgl",
					runtime: this._shaderRuntime,
					profiles: DEFAULT_SHADER_DIRECTIVE_PROFILE_REGISTRY,
					mode: this._shaderRuntime.getMode(),
				})
			: null);
		this._parallelShaderCompile = resolveParallelShaderCompileExtension(gl);
		this._validatePrograms = options.validatePrograms === true;
		this._onProgramCompilePending = options.onProgramCompilePending ?? null;
		this._warnCallback = options.warn ?? null;
		if (this._shaderRuntime) {
			this._disposeShaderRuntimeListener = this._shaderRuntime.onDidChange(() => {
				this.invalidate();
			});
		}
	}

	public createSlot<TProgram extends WebGLProgramResource>(
		descriptor: WebGLProgramDescriptor<TProgram>
	): WebGLProgramSlot<TProgram> {
		this._assertAlive();
		if (this._slots.has(descriptor.label)) {
			throw new Error(
				`WebGL program slot "${descriptor.label}" is already registered.`
			);
		}
		const slot = new WebGLProgramSlotImpl(this, descriptor);
		this._slots.set(descriptor.label, slot);
		return slot;
	}

	/** @internal WebGL context used to reflect implementation-owned programs. */
	public get context(): WebGL2RenderingContext {
		return this._gl;
	}

	public beginFrame(): void {
		this._compileFrameIndex++;
		this._fallbackFinalizesThisFrame = 0;
	}

	public getCompileState(label: string): WebGLProgramCompileState {
		const pending = this._pendingProgramCompiles.get(label);
		if (pending) {
			return pending.status;
		}
		if (this._precompiledPrograms.has(label)) {
			return "ready";
		}
		return "idle";
	}

	public markWarmupHandles(): number {
		return this._warmupHandleLog.length;
	}

	public collectWarmupHandlesSince(mark: number): WebGLProgramWarmupHandle[] {
		const start = Math.max(0, Math.min(mark, this._warmupHandleLog.length));
		const handles = this._warmupHandleLog.slice(start);
		this._warmupHandleLog.length = start;
		return handles;
	}

	/** @internal Records already-compiled backend program warmup work. */
	public createCompletedWarmupHandle(label: string): WebGLProgramWarmupHandle {
		const generation = this._generation;
		const labelGeneration = this._getLabelGeneration(label);
		return this._recordWarmupHandle({
			label,
			isComplete: () => {
				this._assertWarmupGeneration(
					generation,
					labelGeneration,
					label
				);
				return true;
			},
			finalize: () => {
				this._assertWarmupGeneration(
					generation,
					labelGeneration,
					label
				);
			},
		});
	}

	/** @internal Used by backend-owned dynamic program caches. */
	public createProgram(
		vertexSource: string,
		fragmentSource: string,
		label: string,
		vertexMetadata?: WebGLShaderCompileMetadata,
		fragmentMetadata?: WebGLShaderCompileMetadata
	): WebGLProgram {
		this._assertAlive();
		const precompiled = this._precompiledPrograms.get(label);
		if (precompiled) {
			this._precompiledPrograms.delete(label);
			return precompiled;
		}
		const pending = this._pendingProgramCompiles.get(label);
		return this._finalizeProgramCompile(
			pending ??
				this._beginProgramCompile(
					vertexSource,
					fragmentSource,
					label,
					vertexMetadata,
					fragmentMetadata
				)
		);
	}

	/** @internal Used by backend-owned dynamic program caches. */
	public tryCreateProgram(
		vertexSource: string,
		fragmentSource: string,
		label: string,
		vertexMetadata?: WebGLShaderCompileMetadata,
		fragmentMetadata?: WebGLShaderCompileMetadata
	): WebGLProgram | null {
		this._assertAlive();
		const precompiled = this._precompiledPrograms.get(label);
		if (precompiled) {
			this._precompiledPrograms.delete(label);
			return precompiled;
		}
		const pending =
			this._pendingProgramCompiles.get(label) ??
			this._beginProgramCompile(
				vertexSource,
				fragmentSource,
				label,
				vertexMetadata,
				fragmentMetadata
			);
		const program = this._tryFinalizeProgramCompile(pending);
		if (!program) {
			this._notifyProgramCompilePending();
		}
		return program;
	}

	/** @internal Used by backend-owned dynamic program caches. */
	public warmupProgram(
		label: string,
		vertexSource: string,
		fragmentSource: string,
		finalizeReadyProgram: () => void,
		vertexMetadata?: WebGLShaderCompileMetadata,
		fragmentMetadata?: WebGLShaderCompileMetadata,
		handleCompileError?: (error: unknown) => void
	): WebGLProgramWarmupHandle {
		this._assertAlive();
		const generation = this._generation;
		const labelGeneration = this._getLabelGeneration(label);
		if (this._precompiledPrograms.has(label)) {
			return this._recordWarmupHandle({
				label,
				isComplete: () => {
					this._assertWarmupGeneration(
						generation,
						labelGeneration,
						label
					);
					return true;
				},
				finalize: () => {
					this._assertWarmupGeneration(
						generation,
						labelGeneration,
						label
					);
					finalizeReadyProgram();
				},
			});
		}
		const pending =
			this._pendingProgramCompiles.get(label) ??
			this._beginProgramCompile(
				vertexSource,
				fragmentSource,
				label,
				vertexMetadata,
				fragmentMetadata
			);
		return this._recordWarmupHandle({
			label,
			isComplete: () => {
				this._assertWarmupGeneration(
					generation,
					labelGeneration,
					label
				);
				return this._isProgramCompileComplete(pending);
			},
			finalize: () => {
				this._assertWarmupGeneration(
					generation,
					labelGeneration,
					label
				);
				try {
					if (pending.status !== "ready") {
						this._precompiledPrograms.set(
							label,
							this._finalizeProgramCompile(pending)
						);
					}
					finalizeReadyProgram();
				} catch (error) {
					if (handleCompileError) {
						handleCompileError(error);
						return;
					}
					throw error;
				}
			},
		});
	}

	public onDidInvalidate(listener: () => void): () => void {
		this._invalidateListeners.add(listener);
		return () => this._invalidateListeners.delete(listener);
	}

	public invalidate(): void {
		if (this._destroyed) {
			return;
		}
		this._generation++;
		this._disposePendingAndPrecompiledPrograms();
		for (const slot of this._slots.values()) {
			slot.invalidateFromCompiler();
		}
		for (const listener of this._invalidateListeners) {
			listener();
		}
	}

	public destroy(): void {
		if (this._destroyed) {
			return;
		}
		this._destroyed = true;
		this._generation++;
		this._disposeShaderRuntimeListener?.();
		this._disposeShaderRuntimeListener = null;
		this._disposePendingAndPrecompiledPrograms();
		for (const slot of Array.from(this._slots.values())) {
			slot.destroyFromCompiler();
		}
		this._slots.clear();
		this._invalidateListeners.clear();
	}

	/** @internal Called by a slot when its reflected resource is released. */
	public releaseSlot(label: string, slot: WebGLProgramSlotOwner): void {
		if (this._slots.get(label) === slot) {
			this._slots.delete(label);
		}
		this._incrementLabelGeneration(label);
		this._disposeLabelCompilation(label);
	}

	/** @internal Called by a slot to invalidate its program generation. */
	public invalidateSlot(label: string, slot: WebGLProgramSlotOwner): void {
		this._assertAlive();
		if (this._slots.get(label) !== slot) {
			return;
		}
		this._incrementLabelGeneration(label);
		this._disposeLabelCompilation(label);
		slot.invalidateFromCompiler();
	}

	private _beginProgramCompile(
		vertexSource: string,
		fragmentSource: string,
		label: string,
		vertexMetadata?: WebGLShaderCompileMetadata,
		fragmentMetadata?: WebGLShaderCompileMetadata
	): WebGLPendingProgramCompile {
		const gl = this._gl;
		let vertexShader: WebGLPendingShaderCompile | null = null;
		let fragmentShader: WebGLPendingShaderCompile | null = null;
		let program: WebGLProgram | null = null;
		try {
			vertexShader = this._beginShaderCompile(
				gl.VERTEX_SHADER,
				vertexSource,
				`${label}:vertex`,
				vertexMetadata
			);
			fragmentShader = this._beginShaderCompile(
				gl.FRAGMENT_SHADER,
				fragmentSource,
				`${label}:fragment`,
				fragmentMetadata
			);
			program = gl.createProgram();
			if (!program) {
				throw new Error(`Failed to create WebGL program (${label})`);
			}
			gl.attachShader(program, vertexShader.shader);
			gl.attachShader(program, fragmentShader.shader);
			gl.linkProgram(program);
			const pending: WebGLPendingProgramCompile = {
				label,
				vertex: vertexShader,
				fragment: fragmentShader,
				program,
				startedFrame: this._compileFrameIndex,
				generation: this._generation,
				labelGeneration: this._getLabelGeneration(label),
				status: "pending",
				error: null,
			};
			this._pendingProgramCompiles.set(label, pending);
			return pending;
		} catch (error) {
			if (program) this._gl.deleteProgram(program);
			if (vertexShader) this._gl.deleteShader(vertexShader.shader);
			if (fragmentShader) this._gl.deleteShader(fragmentShader.shader);
			this._pendingProgramCompiles.delete(label);
			throw error;
		}
	}

	private _tryFinalizeProgramCompile(
		pending: WebGLPendingProgramCompile
	): WebGLProgram | null {
		if (pending.status === "ready") return pending.program;
		if (pending.status === "failed") throw pending.error;
		if (!this._canFinalizeProgramCompile(pending)) return null;
		return this._finalizeProgramCompile(pending);
	}

	private _finalizeProgramCompile(
		pending: WebGLPendingProgramCompile
	): WebGLProgram {
		if (
			pending.generation !== this._generation ||
			pending.labelGeneration !== this._getLabelGeneration(pending.label)
		) {
			throw new Error(
				`WebGL program "${pending.label}" was invalidated during compilation.`
			);
		}
		if (pending.status === "ready") return pending.program;
		if (pending.status === "failed") throw pending.error;
		const gl = this._gl;
		try {
			this._finalizeShaderCompile(pending.vertex);
			this._finalizeShaderCompile(pending.fragment);
			const linked = !!gl.getProgramParameter(pending.program, gl.LINK_STATUS);
			if (!linked) {
				const log =
					gl.getProgramInfoLog(pending.program) || "No program link log";
				const messages = parseWebGLShaderInfoLog(log);
				throw new ShaderCompileError({
					backend: "webgl",
					language: "glsl",
					stage: "unknown",
					label: pending.label,
					sourceKind: pending.vertex.sourceKind,
					variantKey:
						pending.vertex.variantKey ?? pending.fragment.variantKey,
					materialId:
						pending.vertex.materialId ?? pending.fragment.materialId,
					code: `${pending.vertex.code}\n\n${pending.fragment.code}`,
					sourceMap: null,
					messages:
						messages.length > 0 ?
							messages
						: [this._toCompilerMessage(log)],
					rawLog: log,
				});
			}
			this._validateProgramIfRequested(pending.program, pending.label);
			pending.status = "ready";
			return pending.program;
		} catch (error) {
			pending.status = "failed";
			pending.error = error;
			gl.deleteProgram(pending.program);
			throw error;
		} finally {
			gl.deleteShader(pending.vertex.shader);
			gl.deleteShader(pending.fragment.shader);
			this._pendingProgramCompiles.delete(pending.label);
		}
	}

	private _canFinalizeProgramCompile(
		pending: WebGLPendingProgramCompile
	): boolean {
		if (this._parallelShaderCompile) {
			return this._isProgramCompileComplete(pending);
		}
		const frameAge = this._compileFrameIndex - pending.startedFrame;
		if (frameAge < WEBGL_FALLBACK_READY_FRAME_DELAY) return false;
		if (
			this._fallbackFinalizesThisFrame >=
			WEBGL_FALLBACK_FINALIZE_BUDGET_PER_FRAME
		) {
			return false;
		}
		this._fallbackFinalizesThisFrame++;
		return true;
	}

	private _isProgramCompileComplete(
		pending: WebGLPendingProgramCompile
	): boolean {
		if (
			pending.generation !== this._generation ||
			pending.labelGeneration !== this._getLabelGeneration(pending.label)
		) {
			return true;
		}
		if (pending.status !== "pending" || !this._parallelShaderCompile) {
			return true;
		}
		return !!this._gl.getProgramParameter(
			pending.program,
			this._parallelShaderCompile.COMPLETION_STATUS_KHR
		);
	}

	private _beginShaderCompile(
		type: number,
		source: string,
		label: string,
		metadata?: WebGLShaderCompileMetadata
	): WebGLPendingShaderCompile {
		const stage = type === this._gl.VERTEX_SHADER ? "vertex" : "fragment";
		const sourceKind =
			metadata?.sourceKind ??
			(label.startsWith("WebGLShaderMaterialProgram_") ?
				"custom-material"
			: "unknown");
		const processed = this._processShaderSource(
			source,
			stage,
			sourceKind,
			label,
			metadata?.sourceMap
		);
		if (processed.hasErrors) {
			this._reportShaderRuntimeDiagnostics(label, processed);
		}
		const shader = this._gl.createShader(type);
		if (!shader) {
			throw new Error(`Failed to create WebGL shader (${label})`);
		}
		this._gl.shaderSource(shader, processed.code);
		this._gl.compileShader(shader);
		return {
			shader,
			stage,
			label,
			sourceKind,
			variantKey: metadata?.variantKey,
			materialId: metadata?.materialId,
			code: processed.code,
			sourceMap: processed.sourceMap,
		};
	}

	private _finalizeShaderCompile(shader: WebGLPendingShaderCompile): void {
		const compiled = !!this._gl.getShaderParameter(
			shader.shader,
			this._gl.COMPILE_STATUS
		);
		if (compiled) return;
		const log =
			this._gl.getShaderInfoLog(shader.shader) || "No shader compile log";
		const parsed = parseWebGLShaderInfoLog(log);
		throw new ShaderCompileError({
			backend: "webgl",
			language: "glsl",
			stage: shader.stage,
			label: shader.label,
			sourceKind: shader.sourceKind,
			variantKey: shader.variantKey,
			materialId: shader.materialId,
			code: shader.code,
			sourceMap: shader.sourceMap,
			messages: parsed.length > 0 ? parsed : [this._toCompilerMessage(log)],
			rawLog: log,
		});
	}

	private _processShaderSource(
		source: string,
		stage: "vertex" | "fragment",
		sourceKind: "custom-material" | "unknown",
		label: string,
		sourceMap?: ShaderSourceSegmentMap | null
	): ShaderProcessResult {
		const directiveSourcePath =
			sourceMap?.segments[0]?.sourcePath ?? label ?? "<webgl-shader>";
		if (this._shaderCompileStage) {
			return this._shaderCompileStage.compile({
				code: source,
				language: "glsl",
				stage,
				entryPoint: "main",
				label,
				sourceKind,
				sourceMap: sourceMap ?? null,
				directiveSourcePath,
			});
		}
		if (!this._shaderRuntime) {
			const effectiveSourceMap =
				sourceMap ?? createInlineShaderSourceMap(source, label, "source");
			return {
				code: source,
				sourceMap: effectiveSourceMap,
				composite: { code: source, sourceMap: effectiveSourceMap },
				diagnostics: [],
				hasErrors: false,
				fromCache: false,
			};
		}
		return this._shaderRuntime.process({
			code: source,
			language: "glsl",
			stage,
			entryPoint: "main",
			label,
			sourceKind,
			sourceMap: sourceMap ?? null,
			directiveSourcePath,
		});
	}

	private _validateProgramIfRequested(
		program: WebGLProgram,
		label: string
	): void {
		if (!this._validatePrograms) return;
		this._gl.validateProgram(program);
		if (this._gl.getProgramParameter(program, this._gl.VALIDATE_STATUS) !== false) {
			return;
		}
		this._warn(
			`webgl-program-validate-${label}`,
			`WebGL program validation reported issues (${label}): ` +
				`${this._gl.getProgramInfoLog(program) || "no log"}`
		);
	}

	private _disposeLabelCompilation(label: string): void {
		const pending = this._pendingProgramCompiles.get(label);
		if (pending) {
			this._gl.deleteShader(pending.vertex.shader);
			this._gl.deleteShader(pending.fragment.shader);
			this._gl.deleteProgram(pending.program);
			this._pendingProgramCompiles.delete(label);
		}
		const precompiled = this._precompiledPrograms.get(label);
		if (precompiled) {
			this._gl.deleteProgram(precompiled);
			this._precompiledPrograms.delete(label);
		}
	}

	private _disposePendingAndPrecompiledPrograms(): void {
		for (const label of Array.from(this._pendingProgramCompiles.keys())) {
			this._disposeLabelCompilation(label);
		}
		for (const program of this._precompiledPrograms.values()) {
			this._gl.deleteProgram(program);
		}
		this._precompiledPrograms.clear();
		this._warmupHandleLog.length = 0;
	}

	private _recordWarmupHandle(
		handle: WebGLProgramWarmupHandle
	): WebGLProgramWarmupHandle {
		this._warmupHandleLog.push(handle);
		return handle;
	}

	private _notifyProgramCompilePending(): void {
		if (
			this._lastPendingNotificationFrame === this._compileFrameIndex ||
			!this._onProgramCompilePending
		) {
			return;
		}
		this._lastPendingNotificationFrame = this._compileFrameIndex;
		this._onProgramCompilePending();
	}

	private _assertWarmupGeneration(
		generation: number,
		labelGeneration: number,
		label: string
	): void {
		if (
			generation !== this._generation ||
			labelGeneration !== this._getLabelGeneration(label) ||
			this._destroyed
		) {
			throw new Error(
				`WebGL warmup handle "${label}" became stale after program invalidation.`
			);
		}
	}

	private _getLabelGeneration(label: string): number {
		return this._labelGenerations.get(label) ?? 0;
	}

	private _incrementLabelGeneration(label: string): void {
		this._labelGenerations.set(label, this._getLabelGeneration(label) + 1);
	}

	private _assertAlive(): void {
		if (this._destroyed) {
			throw new Error("WebGL program compiler has been destroyed.");
		}
	}

	private _reportShaderRuntimeDiagnostics(
		label: string,
		result: ShaderProcessResult
	): void {
		for (const diagnostic of result.diagnostics) {
			this._warn(
				`webgl-shader-runtime-${diagnostic.severity}-${diagnostic.code}-${label}`,
				`WebGL shader runtime ${diagnostic.severity} [${label}] ` +
					`${diagnostic.code}: ${diagnostic.message}`
			);
		}
	}

	private _warn(key: string, message: string): void {
		this._warnCallback?.(key, message);
		Logger.warn(`[${key}] ${message}`, {
			scope: "WebGLProgramCompiler",
			onceKey: key,
		});
	}

	private _toCompilerMessage(log: string): ShaderCompilerMessage {
		return { type: "error", message: log, raw: log };
	}
}

class WebGLProgramSlotImpl<TProgram extends WebGLProgramResource>
	implements WebGLProgramSlot<TProgram>, WebGLProgramSlotOwner
{
	public readonly label: string;
	private readonly _compiler: WebGLProgramCompiler;
	private readonly _descriptor: WebGLProgramDescriptor<TProgram>;
	private _resource: TProgram | null = null;
	private _destroyed = false;

	public constructor(
		compiler: WebGLProgramCompiler,
		descriptor: WebGLProgramDescriptor<TProgram>
	) {
		this._compiler = compiler;
		this._descriptor = descriptor;
		this.label = descriptor.label;
	}

	public get(): TProgram {
		this._assertAlive();
		if (!this._resource) {
			const vertex = resolveShaderSource(this._descriptor.vertex());
			const fragment = resolveShaderSource(this._descriptor.fragment());
			const program = this._compiler.createProgram(
				vertex.code,
				fragment.code,
				this.label,
				vertex.metadata,
				fragment.metadata
			);
			this._resource = this._descriptor.reflect(this._compiler.context, program);
		}
		return this._resource;
	}

	public tryGet(): TProgram | null {
		this._assertAlive();
		if (this._resource) return this._resource;
		const vertex = resolveShaderSource(this._descriptor.vertex());
		const fragment = resolveShaderSource(this._descriptor.fragment());
		const program = this._compiler.tryCreateProgram(
			vertex.code,
			fragment.code,
			this.label,
			vertex.metadata,
			fragment.metadata
		);
		if (!program) return null;
		this._resource = this._descriptor.reflect(this._compiler.context, program);
		return this._resource;
	}

	public warmup(): WebGLProgramWarmupHandle {
		this._assertAlive();
		if (this._resource) {
			return this._compiler.createCompletedWarmupHandle(this.label);
		}
		const vertex = resolveShaderSource(this._descriptor.vertex());
		const fragment = resolveShaderSource(this._descriptor.fragment());
		return this._compiler.warmupProgram(
			this.label,
			vertex.code,
			fragment.code,
			() => {
				this.get();
			},
			vertex.metadata,
			fragment.metadata
		);
	}

	public invalidate(): void {
		if (this._destroyed) return;
		this._compiler.invalidateSlot(this.label, this);
	}

	public destroy(): void {
		if (this._destroyed) return;
		this._destroyed = true;
		this._deleteResource();
		this._compiler.releaseSlot(this.label, this);
	}

	public invalidateFromCompiler(): void {
		if (this._destroyed) return;
		this._deleteResource();
	}

	public destroyFromCompiler(): void {
		if (this._destroyed) return;
		this._destroyed = true;
		this._deleteResource();
	}

	private _deleteResource(): void {
		if (!this._resource) return;
		this._compiler.context.deleteProgram(this._resource.program);
		this._resource = null;
	}

	private _assertAlive(): void {
		if (this._destroyed) {
			throw new Error(`WebGL program slot "${this.label}" has been destroyed.`);
		}
	}
}

function resolveShaderSource(
	source: string | WebGLShaderSourceDescriptor
): WebGLShaderSourceDescriptor {
	return typeof source === "string" ? { code: source } : source;
}

function resolveParallelShaderCompileExtension(
	gl: WebGL2RenderingContext
): WebGLParallelShaderCompileExtension | null {
	if (typeof gl.getExtension !== "function") return null;
	try {
		return gl.getExtension(
			"KHR_parallel_shader_compile"
		) as WebGLParallelShaderCompileExtension | null;
	} catch {
		return null;
	}
}
