import type { Texture } from "../core/Texture";
import { Material, type MaterialParams, ShadingModel } from "./Material";

export type ShaderTargetMode = "single" | "mrt" | "deferred";
export type ShaderStageKind =
	| "vertex"
	| "fragment-single"
	| "fragment-mrt"
	| "fragment-deferred"
	| "fragment-depth";
export type ShaderChunkLanguage = "wgsl" | "glsl";
export type ShaderChunkBackend = "webgpu" | "webgl";
export type ShaderChunkStage = "vertex" | "fragment" | "fragment-depth";

export type ShaderChunk = {
	language: ShaderChunkLanguage;
	stage: ShaderChunkStage;
	code: string;
	backend?: ShaderChunkBackend;
	mode?: ShaderTargetMode;
};

export type ShaderMaterialGLSLToWGSL = (
	source: string,
	stage: ShaderStageKind
) => string;

export interface ShaderMaterialTextureBinding {
	name: string;
	texture: Texture | null;
	slot?: number;
	uvSet?: 0 | 1 | 2 | 3;
	linear?: boolean;
	webglUniform?: string;
}

export interface ResolvedShaderMaterialTextureBinding {
	name: string;
	texture: Texture | null;
	slot: number;
	uvSet: 0 | 1 | 2 | 3;
	linear: boolean;
	webglUniform: string;
}

export interface ShaderMaterialParams extends MaterialParams {
	vertexEntryPoint?: string;
	fragmentSingleEntryPoint?: string;
	fragmentMRTEntryPoint?: string;
	/** WebGPU deferred G-buffer fragment entry point. Defaults to `fsMainDeferred`. */
	fragmentDeferredEntryPoint?: string;
	depthFragmentEntryPoint?: string;
	depthFragmentCode?: string;
	/** Opts this material into WebGPU deferred lighting when a deferred chunk exists. */
	deferredLighting?: boolean;
	glslToWgsl?: ShaderMaterialGLSLToWGSL;
	chunks?: ShaderChunk[];
	textureBindings?: ShaderMaterialTextureBinding[];
}

export interface ResolvedWebGPUShaderProgram {
	vertexCode: string;
	fragmentCode: string;
	vertexEntryPoint: string;
	fragmentEntryPoint: string;
}

export interface ResolvedWebGPUDepthPrepassProgram {
	vertexCode: string;
	fragmentCode: string;
	vertexEntryPoint: string;
	fragmentEntryPoint: string;
}

export interface ResolvedWebGLShaderProgram {
	vertexCode: string;
	fragmentCode: string;
}

export interface ShaderProgramResolveOptions {
	enableRuntimeInjects?: boolean;
}

type ShaderChunkKey =
	| "webgpu:wgsl:vertex"
	| "webgpu:wgsl:fragment-single"
	| "webgpu:wgsl:fragment-mrt"
	| "webgpu:wgsl:fragment-deferred"
	| "webgpu:wgsl:fragment-depth"
	| "webgpu:glsl:vertex"
	| "webgpu:glsl:fragment-single"
	| "webgpu:glsl:fragment-mrt"
	| "webgpu:glsl:fragment-deferred"
	| "webgpu:glsl:fragment-depth"
	| "webgl:glsl:vertex"
	| "webgl:glsl:fragment-single"
	| "webgl:glsl:fragment-mrt";

interface ShaderMaterialTextureBindingRecord {
	name: string;
	texture: Texture | null;
	explicitSlot: number | null;
	uvSet: 0 | 1 | 2 | 3;
	linearOverride: boolean | null;
	webglUniform: string;
}

const SHADER_CHUNK_ORDER: readonly ShaderChunkKey[] = [
	"webgpu:wgsl:vertex",
	"webgpu:wgsl:fragment-single",
	"webgpu:wgsl:fragment-mrt",
	"webgpu:wgsl:fragment-deferred",
	"webgpu:wgsl:fragment-depth",
	"webgpu:glsl:vertex",
	"webgpu:glsl:fragment-single",
	"webgpu:glsl:fragment-mrt",
	"webgpu:glsl:fragment-deferred",
	"webgpu:glsl:fragment-depth",
	"webgl:glsl:vertex",
	"webgl:glsl:fragment-single",
	"webgl:glsl:fragment-mrt",
];

const SHADER_MATERIAL_TEXTURE_SLOT_COUNT = 14;
export const SHADER_MATERIAL_MAX_TEXTURE_SLOTS =
	SHADER_MATERIAL_TEXTURE_SLOT_COUNT;

let SHADER_MATERIAL_ID = 1;

export class ShaderMaterial extends Material {
	public readonly shaderId: number;
	public vertexEntryPoint: string;
	public fragmentSingleEntryPoint: string;
	public fragmentMRTEntryPoint: string;
	/** WebGPU deferred G-buffer fragment entry point. */
	public fragmentDeferredEntryPoint: string;
	public depthFragmentEntryPoint: string;
	/** Whether this material may route through WebGPU deferred lighting. */
	public deferredLighting: boolean;

	private _chunks: Map<ShaderChunkKey, string>;
	private _textureBindings: Map<string, ShaderMaterialTextureBindingRecord>;
	private _glslToWgsl: ShaderMaterialGLSLToWGSL | null;
	private _shaderRevision: number;

	constructor(params: ShaderMaterialParams = {}) {
		super({ ...params, shading: params.shading ?? ShadingModel.Flat });
		this.type = "Shader";
		this.shaderId = SHADER_MATERIAL_ID++;
		this.vertexEntryPoint = params.vertexEntryPoint ?? "vsMain";
		this.fragmentSingleEntryPoint =
			params.fragmentSingleEntryPoint ?? "fsMainSingle";
		this.fragmentMRTEntryPoint = params.fragmentMRTEntryPoint ?? "fsMain";
		this.fragmentDeferredEntryPoint =
			params.fragmentDeferredEntryPoint ?? "fsMainDeferred";
		this.depthFragmentEntryPoint =
			params.depthFragmentEntryPoint ?? "fsMainDepth";
		this.deferredLighting = params.deferredLighting === true;
		this._chunks = new Map<ShaderChunkKey, string>();
		this._textureBindings = new Map();
		this._glslToWgsl = params.glslToWgsl ?? null;
		this._shaderRevision = 0;

		if (Array.isArray(params.chunks) && params.chunks.length > 0) {
			this.setChunks(params.chunks);
		}
		if (
			typeof params.depthFragmentCode === "string" &&
			params.depthFragmentCode.trim().length > 0
		) {
			this.upsertChunk({
				backend: "webgpu",
				language: "wgsl",
				stage: "fragment-depth",
				code: params.depthFragmentCode,
			});
		}

		if (Array.isArray(params.textureBindings) && params.textureBindings.length > 0) {
			this.setTextureBindings(params.textureBindings);
		}
	}

	public get shaderRevision(): number {
		return this._shaderRevision;
	}

	public get chunks(): ShaderChunk[] {
		const chunks: ShaderChunk[] = [];
		for (const key of SHADER_CHUNK_ORDER) {
			const code = this._chunks.get(key);
			if (!code) {
				continue;
			}
			chunks.push(this._chunkFromKey(key, code));
		}
		return chunks;
	}

	public get glslToWgsl(): ShaderMaterialGLSLToWGSL | null {
		return this._glslToWgsl;
	}

	public setChunks(chunks: ShaderChunk[]): void {
		const nextChunks = new Map<ShaderChunkKey, string>();
		for (const chunk of chunks) {
			const normalized = this._normalizeChunk(chunk);
			if (!normalized) {
				continue;
			}
			nextChunks.set(normalized.key, normalized.code);
		}
		if (!this._didChunkMapChange(nextChunks)) {
			return;
		}
		this._chunks = nextChunks;
		this._touchShaderRevision();
	}

	public upsertChunk(chunk: ShaderChunk): void {
		const normalized = this._normalizeChunk(chunk);
		if (!normalized) {
			return;
		}
		const current = this._chunks.get(normalized.key) ?? null;
		if (current === normalized.code) {
			return;
		}
		this._chunks.set(normalized.key, normalized.code);
		this._touchShaderRevision();
	}

	public removeChunk(chunk: Omit<ShaderChunk, "code">): boolean {
		const key = this._resolveChunkKey(chunk);
		if (!key) {
			return false;
		}
		const removed = this._chunks.delete(key);
		if (removed) {
			this._touchShaderRevision();
		}
		return removed;
	}

	public setGLSLToWGSL(transpiler: ShaderMaterialGLSLToWGSL | null): void {
		if (this._glslToWgsl === transpiler) {
			return;
		}
		this._glslToWgsl = transpiler;
		this._touchShaderRevision();
	}

	public setTextureBindings(bindings: ShaderMaterialTextureBinding[]): void {
		const next = new Map<string, ShaderMaterialTextureBindingRecord>();
		for (const binding of bindings) {
			const normalized = this._normalizeTextureBinding(binding);
			next.set(normalized.name, normalized);
		}
		this._validateTextureBindingSlots(next);
		if (!this._didTextureBindingMapChange(next)) {
			return;
		}
		this._textureBindings = next;
		this._touchShaderRevision();
	}

	public setTextureBinding(binding: ShaderMaterialTextureBinding): void {
		const normalized = this._normalizeTextureBinding(binding);
		const next = new Map(this._textureBindings);
		next.set(normalized.name, normalized);
		this._validateTextureBindingSlots(next);
		if (!this._didTextureBindingMapChange(next)) {
			return;
		}
		this._textureBindings = next;
		this._touchShaderRevision();
	}

	public removeTextureBinding(name: string): boolean {
		const key = this._normalizeTextureBindingName(name);
		if (key.length <= 0) {
			return false;
		}
		const removed = this._textureBindings.delete(key);
		if (removed) {
			this._touchShaderRevision();
		}
		return removed;
	}

	public clearTextureBindings(): void {
		if (this._textureBindings.size <= 0) {
			return;
		}
		this._textureBindings.clear();
		this._touchShaderRevision();
	}

	public getTextureBindings(): ResolvedShaderMaterialTextureBinding[] {
		return this._resolveTextureBindings();
	}

	public getWebGPUCacheKey(): string {
		return [
			this.shaderId,
			this._shaderRevision,
			this.vertexEntryPoint,
			this.fragmentSingleEntryPoint,
			this.fragmentMRTEntryPoint,
			this.fragmentDeferredEntryPoint,
			this.depthFragmentEntryPoint,
			this.deferredLighting ? 1 : 0,
		].join(":");
	}

	public getWebGLCacheKey(): string {
		return [this.shaderId, this._shaderRevision].join(":");
	}

	public resolveWebGPUProgram(
		mode: ShaderTargetMode,
		options: ShaderProgramResolveOptions = {}
	): ResolvedWebGPUShaderProgram {
		const vertexCode = this._resolveWebGPUStageCode("vertex", mode);
		const fragmentStage =
			mode === "deferred" ? "fragment-deferred"
			: mode === "mrt" ? "fragment-mrt"
			: "fragment-single";
		const rawFragmentCode = this._resolveWebGPUStageCode(fragmentStage, mode);
		const fragmentCode = this._decorateFragmentSource(
			rawFragmentCode,
			"wgsl",
			options.enableRuntimeInjects === true
		);
		return {
			vertexCode,
			fragmentCode,
			vertexEntryPoint: this.vertexEntryPoint,
			fragmentEntryPoint:
				mode === "deferred" ?
					this.fragmentDeferredEntryPoint
				: mode === "mrt" ?
					this.fragmentMRTEntryPoint
				:	this.fragmentSingleEntryPoint,
		};
	}

	/**
	 * Returns whether this material has opted into the WebGPU deferred lighting
	 * shader contract and provides an explicit deferred fragment chunk.
	 *
	 * @returns `true` when `resolveWebGPUProgram("deferred")` is expected to
	 * resolve without falling back to the legacy MRT or single-target fragment.
	 * @sideEffects None.
	 */
	public hasWebGPUDeferredProgram(): boolean {
		return (
			this.deferredLighting &&
			(
				this._chunks.has("webgpu:wgsl:fragment-deferred") ||
				this._chunks.has("webgpu:glsl:fragment-deferred")
			)
		);
	}

	public resolveWebGPUDepthPrepassProgram(
		mode: ShaderTargetMode,
		options: ShaderProgramResolveOptions = {}
	): ResolvedWebGPUDepthPrepassProgram | null {
		const vertexCode = this._resolveWebGPUStageCode("vertex", mode);
		const rawDepthCode = this._resolveWebGPUDepthStageCode(mode);
		if (!rawDepthCode) {
			return null;
		}
		const fragmentCode = this._decorateFragmentSource(
			rawDepthCode,
			"wgsl",
			options.enableRuntimeInjects === true
		);
		return {
			vertexCode,
			fragmentCode,
			vertexEntryPoint: this.vertexEntryPoint,
			fragmentEntryPoint: this.depthFragmentEntryPoint,
		};
	}

	public resolveWebGLProgram(
		modeOrOptions: ShaderTargetMode | ShaderProgramResolveOptions = "single",
		options?: ShaderProgramResolveOptions
	): ResolvedWebGLShaderProgram {
		const { mode, resolveOptions } = this._resolveWebGLProgramArgs(
			modeOrOptions,
			options
		);
		const vertexCode =
			this._chunks.get("webgl:glsl:vertex") ??
			this._chunks.get("webgpu:glsl:vertex") ??
			null;
		const rawFragmentCode = this._resolveWebGLStageCode(mode);

		if (
			typeof vertexCode !== "string" ||
			vertexCode.trim().length === 0 ||
			typeof rawFragmentCode !== "string" ||
			rawFragmentCode.trim().length === 0
		) {
			throw new Error(
				`ShaderMaterial ${this.name} is missing WebGL GLSL source; ` +
					"provide webgl GLSL vertex and fragment-single chunks " +
					"(optionally fragment-mrt for MRT), or webgpu GLSL " +
					"vertex/fragment-single chunks as fallback"
			);
		}

		return {
			vertexCode,
			fragmentCode: this._decorateFragmentSource(
				rawFragmentCode,
				"glsl",
				resolveOptions.enableRuntimeInjects === true
			),
		};
	}

	private _resolveWebGPUStageCode(
		stage: ShaderStageKind,
		mode: ShaderTargetMode
	): string {
		const wgslSource = this._getChunkStageSource("wgsl", stage, mode);
		if (wgslSource) {
			return wgslSource;
		}

		const glslSource = this._getChunkStageSource("glsl", stage, mode);
		if (!glslSource) {
			throw new Error(
				`ShaderMaterial ${this.name} is missing ${stage} shader source for ${mode} mode`
			);
		}

		if (!this._glslToWgsl) {
			throw new Error(
				`ShaderMaterial ${this.name} has GLSL source but no glslToWgsl transpiler; provide webgpu WGSL chunks directly or call setGLSLToWGSL()`
			);
		}

		const transpiled = this._glslToWgsl(glslSource, stage);
		if (typeof transpiled !== "string" || transpiled.trim().length === 0) {
			throw new Error(
				`ShaderMaterial ${this.name} glslToWgsl transpiler returned empty output for ${stage}`
			);
		}
		return transpiled;
	}

	private _resolveWebGPUDepthStageCode(mode: ShaderTargetMode): string | null {
		const wgslSource = this._getChunkStageSource(
			"wgsl",
			"fragment-depth",
			mode
		);
		if (wgslSource) {
			return wgslSource;
		}

		const glslSource = this._getChunkStageSource(
			"glsl",
			"fragment-depth",
			mode
		);
		if (!glslSource) {
			return null;
		}

		if (!this._glslToWgsl) {
			throw new Error(
				`ShaderMaterial ${this.name} has GLSL depth fragment source but no glslToWgsl transpiler; provide webgpu WGSL depth fragment chunk or call setGLSLToWGSL()`
			);
		}

		const transpiled = this._glslToWgsl(glslSource, "fragment-depth");
		if (typeof transpiled !== "string" || transpiled.trim().length === 0) {
			throw new Error(
				`ShaderMaterial ${this.name} glslToWgsl transpiler returned empty output for fragment-depth`
			);
		}
		return transpiled;
	}

	private _chunkFromKey(key: ShaderChunkKey, code: string): ShaderChunk {
		switch (key) {
			case "webgpu:wgsl:vertex":
				return {
					backend: "webgpu",
					language: "wgsl",
					stage: "vertex",
					code,
				};
			case "webgpu:wgsl:fragment-single":
				return {
					backend: "webgpu",
					language: "wgsl",
					stage: "fragment",
					mode: "single",
					code,
				};
			case "webgpu:wgsl:fragment-mrt":
				return {
					backend: "webgpu",
					language: "wgsl",
					stage: "fragment",
					mode: "mrt",
					code,
				};
			case "webgpu:wgsl:fragment-deferred":
				return {
					backend: "webgpu",
					language: "wgsl",
					stage: "fragment",
					mode: "deferred",
					code,
				};
			case "webgpu:wgsl:fragment-depth":
				return {
					backend: "webgpu",
					language: "wgsl",
					stage: "fragment-depth",
					code,
				};
			case "webgpu:glsl:vertex":
				return {
					backend: "webgpu",
					language: "glsl",
					stage: "vertex",
					code,
				};
			case "webgpu:glsl:fragment-single":
				return {
					backend: "webgpu",
					language: "glsl",
					stage: "fragment",
					mode: "single",
					code,
				};
			case "webgpu:glsl:fragment-mrt":
				return {
					backend: "webgpu",
					language: "glsl",
					stage: "fragment",
					mode: "mrt",
					code,
				};
			case "webgpu:glsl:fragment-deferred":
				return {
					backend: "webgpu",
					language: "glsl",
					stage: "fragment",
					mode: "deferred",
					code,
				};
			case "webgpu:glsl:fragment-depth":
				return {
					backend: "webgpu",
					language: "glsl",
					stage: "fragment-depth",
					code,
				};
			case "webgl:glsl:vertex":
				return {
					backend: "webgl",
					language: "glsl",
					stage: "vertex",
					code,
				};
			case "webgl:glsl:fragment-single":
				return {
					backend: "webgl",
					language: "glsl",
					stage: "fragment",
					mode: "single",
					code,
				};
			case "webgl:glsl:fragment-mrt":
			default:
				return {
					backend: "webgl",
					language: "glsl",
					stage: "fragment",
					mode: "mrt",
					code,
				};
		}
	}

	private _normalizeChunk(
		chunk: ShaderChunk
	): { key: ShaderChunkKey; code: string } | null {
		const key = this._resolveChunkKey(chunk);
		if (!key) {
			return null;
		}
		if (typeof chunk.code !== "string") {
			throw new Error("Shader chunk code must be a string.");
		}
		return {
			key,
			code: chunk.code,
		};
	}

	private _resolveChunkKey(chunk: Omit<ShaderChunk, "code">): ShaderChunkKey | null {
		const backend = chunk.backend ?? "webgpu";
		const language = chunk.language;
		const stage = chunk.stage;
		const mode = chunk.mode ?? "single";

		if (backend === "webgl") {
			if (language !== "glsl") {
				throw new Error("WebGL shader chunks must use GLSL.");
			}
			if (stage === "vertex") {
				return "webgl:glsl:vertex";
			}
			if (stage === "fragment-depth") {
				throw new Error(
					"WebGL shader chunks do not support fragment-depth stage."
				);
			}
			if (mode !== "single" && mode !== "mrt") {
				throw new Error(`Unsupported shader chunk mode "${mode}".`);
			}
			return `webgl:glsl:fragment-${mode}` as ShaderChunkKey;
		}

		if (backend !== "webgpu") {
			throw new Error(`Unsupported shader chunk backend "${backend}".`);
		}
		if (language !== "wgsl" && language !== "glsl") {
			throw new Error(`Unsupported shader chunk language "${language}".`);
		}
		if (stage === "vertex") {
			return `webgpu:${language}:vertex` as ShaderChunkKey;
		}
		if (stage === "fragment-depth") {
			return `webgpu:${language}:fragment-depth` as ShaderChunkKey;
		}
		if (mode !== "single" && mode !== "mrt" && mode !== "deferred") {
			throw new Error(`Unsupported shader chunk mode "${mode}".`);
		}
		return `webgpu:${language}:fragment-${mode}` as ShaderChunkKey;
	}

	private _getChunkStageSource(
		language: ShaderChunkLanguage,
		stage: ShaderStageKind,
		mode: ShaderTargetMode
	): string | null {
		switch (stage) {
			case "vertex":
				return this._chunks.get(`webgpu:${language}:vertex`) ?? null;
			case "fragment-single":
				return this._chunks.get(`webgpu:${language}:fragment-single`) ?? null;
			case "fragment-mrt":
				return (
					mode === "mrt" ?
						(this._chunks.get(`webgpu:${language}:fragment-mrt`) ??
							this._chunks.get(`webgpu:${language}:fragment-single`) ??
							null)
					:	(this._chunks.get(`webgpu:${language}:fragment-single`) ??
							null)
				);
			case "fragment-deferred":
				return (
					mode === "deferred" ?
						this._chunks.get(`webgpu:${language}:fragment-deferred`) ??
						null
					:	null
				);
			case "fragment-depth":
				return this._chunks.get(`webgpu:${language}:fragment-depth`) ?? null;
			default:
				return null;
		}
	}

	private _resolveWebGLProgramArgs(
		modeOrOptions: ShaderTargetMode | ShaderProgramResolveOptions,
		options?: ShaderProgramResolveOptions
	): { mode: ShaderTargetMode; resolveOptions: ShaderProgramResolveOptions } {
		if (typeof modeOrOptions === "string") {
			if (modeOrOptions !== "single" && modeOrOptions !== "mrt") {
				throw new Error(`Unsupported WebGL shader target mode "${modeOrOptions}".`);
			}
			return {
				mode: modeOrOptions,
				resolveOptions: options ?? {},
			};
		}
		return {
			mode: "single",
			resolveOptions: modeOrOptions ?? {},
		};
	}

	private _resolveWebGLStageCode(mode: ShaderTargetMode): string | null {
		if (mode === "mrt") {
			return (
				this._chunks.get("webgl:glsl:fragment-mrt") ??
				this._chunks.get("webgl:glsl:fragment-single") ??
				this._chunks.get("webgpu:glsl:fragment-mrt") ??
				this._chunks.get("webgpu:glsl:fragment-single") ??
				null
			);
		}
		return (
			this._chunks.get("webgl:glsl:fragment-single") ??
			this._chunks.get("webgpu:glsl:fragment-single") ??
			null
		);
	}

	private _normalizeTextureBinding(
		binding: ShaderMaterialTextureBinding
	): ShaderMaterialTextureBindingRecord {
		const name = this._normalizeTextureBindingName(binding.name);
		if (name.length <= 0) {
			throw new Error("ShaderMaterial texture binding name must be non-empty.");
		}

		const explicitSlot = this._normalizeTextureBindingSlot(binding.slot);
		const uvSet = this._normalizeTextureBindingUVSet(binding.uvSet);
		const linearOverride =
			typeof binding.linear === "boolean" ? binding.linear : null;
		const webglUniform =
			this._normalizeWebGLUniformName(binding.webglUniform) ??
			`uShaderTex_${this._toIdentifierToken(name)}`;
		return {
			name,
			texture: binding.texture ?? null,
			explicitSlot,
			uvSet,
			linearOverride,
			webglUniform,
		};
	}

	private _normalizeTextureBindingName(name: string): string {
		if (typeof name !== "string") {
			return "";
		}
		return name.trim();
	}

	private _normalizeTextureBindingSlot(slot: number | undefined): number | null {
		if (slot === undefined || slot === null) {
			return null;
		}
		if (!Number.isFinite(slot)) {
			throw new Error(`ShaderMaterial texture slot "${slot}" is not finite.`);
		}
		const resolved = Math.floor(slot);
		if (resolved < 0 || resolved >= SHADER_MATERIAL_TEXTURE_SLOT_COUNT) {
			throw new Error(
				`ShaderMaterial texture slot ${resolved} is out of range [0, ${SHADER_MATERIAL_TEXTURE_SLOT_COUNT - 1}].`
			);
		}
		return resolved;
	}

	private _normalizeTextureBindingUVSet(
		uvSet: number | undefined
	): 0 | 1 | 2 | 3 {
		if (typeof uvSet !== "number" || !Number.isFinite(uvSet)) {
			return 0;
		}
		return Math.max(0, Math.min(3, Math.floor(uvSet))) as 0 | 1 | 2 | 3;
	}

	private _normalizeWebGLUniformName(name: string | undefined): string | null {
		if (typeof name !== "string") {
			return null;
		}
		const trimmed = name.trim();
		if (trimmed.length <= 0) {
			return null;
		}
		if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(trimmed)) {
			throw new Error(
				`Invalid WebGL uniform name "${name}". Expected GLSL identifier.`
			);
		}
		return trimmed;
	}

	private _resolveTextureBindings(): ResolvedShaderMaterialTextureBinding[] {
		if (this._textureBindings.size <= 0) {
			return [];
		}

		const explicitSlots = new Set<number>();
		for (const binding of this._textureBindings.values()) {
			if (binding.explicitSlot !== null) {
				if (explicitSlots.has(binding.explicitSlot)) {
					throw new Error(
						`ShaderMaterial has duplicate texture slot ${binding.explicitSlot}.`
					);
				}
				explicitSlots.add(binding.explicitSlot);
			}
		}

		let autoSlotCursor = 0;
		const resolved: ResolvedShaderMaterialTextureBinding[] = [];
		for (const binding of this._textureBindings.values()) {
			let slot = binding.explicitSlot;
			if (slot === null) {
				while (
					autoSlotCursor < SHADER_MATERIAL_TEXTURE_SLOT_COUNT &&
					explicitSlots.has(autoSlotCursor)
				) {
					autoSlotCursor++;
				}
				if (autoSlotCursor >= SHADER_MATERIAL_TEXTURE_SLOT_COUNT) {
					throw new Error(
						`ShaderMaterial texture bindings exceed ${SHADER_MATERIAL_TEXTURE_SLOT_COUNT} slots.`
					);
				}
				slot = autoSlotCursor++;
				explicitSlots.add(slot);
			}

			const linear =
				binding.linearOverride ??
				(binding.texture?.colorSpace === "Linear" ||
					binding.texture?.colorSpace === "HDR");
			resolved.push({
				name: binding.name,
				texture: binding.texture,
				slot,
				uvSet: binding.uvSet,
				linear,
				webglUniform: binding.webglUniform,
			});
		}
		return resolved;
	}

	private _validateTextureBindingSlots(
		bindings: Map<string, ShaderMaterialTextureBindingRecord>
	): void {
		let unresolvedCount = 0;
		const explicitSlots = new Set<number>();
		for (const binding of bindings.values()) {
			if (binding.explicitSlot === null) {
				unresolvedCount++;
				continue;
			}
			if (explicitSlots.has(binding.explicitSlot)) {
				throw new Error(
					`ShaderMaterial has duplicate texture slot ${binding.explicitSlot}.`
				);
			}
			explicitSlots.add(binding.explicitSlot);
		}
		if (explicitSlots.size + unresolvedCount > SHADER_MATERIAL_TEXTURE_SLOT_COUNT) {
			throw new Error(
				`ShaderMaterial texture bindings exceed ${SHADER_MATERIAL_TEXTURE_SLOT_COUNT} slots.`
			);
		}
	}

	private _buildTextureBindingInjectBlock(language: ShaderChunkLanguage): string {
		const bindings = this._resolveTextureBindings();
		if (bindings.length <= 0) {
			return "";
		}
		return bindings
			.map((binding) => this._buildTextureInjectDirective(language, binding))
			.join("\n");
	}

	private _buildTextureInjectDirective(
		_language: ShaderChunkLanguage,
		binding: ResolvedShaderMaterialTextureBinding
	): string {
		const escapedName = this._escapeInjectString(binding.name);
		const escapedUniform = this._escapeInjectString(binding.webglUniform);
		return (
			`#inject <ignis/material/texture-binding>(` +
			`name="${escapedName}", ` +
			`slot=${binding.slot}, ` +
			`uv=${binding.uvSet}, ` +
			`linear=${binding.linear ? "true" : "false"}, ` +
			`uniform="${escapedUniform}")`
		);
	}

	private _decorateFragmentSource(
		code: string,
		language: ShaderChunkLanguage,
		enableRuntimeInjects: boolean
	): string {
		if (!enableRuntimeInjects) {
			return code;
		}
		const injectBlock = this._buildTextureBindingInjectBlock(language);
		if (injectBlock.length <= 0) {
			return code;
		}

		if (language === "glsl") {
			const versionMatch = /^(\s*#version[^\n]*\n?)/.exec(code);
			if (versionMatch) {
				const prefix = versionMatch[1];
				return `${prefix}${injectBlock}\n${code.slice(prefix.length)}`;
			}
		}
		return `${injectBlock}\n${code}`;
	}

	private _toIdentifierToken(value: string): string {
		const sanitized = value.replace(/[^A-Za-z0-9_]/g, "_");
		if (/^[A-Za-z_]/.test(sanitized)) {
			return sanitized;
		}
		return `x_${sanitized}`;
	}

	private _escapeInjectString(value: string): string {
		return value.replace(/\\/g, "\\\\").replace(/"/g, '\\"');
	}

	private _didChunkMapChange(nextChunks: Map<ShaderChunkKey, string>): boolean {
		if (nextChunks.size !== this._chunks.size) {
			return true;
		}
		for (const [key, code] of nextChunks) {
			if (this._chunks.get(key) !== code) {
				return true;
			}
		}
		return false;
	}

	private _didTextureBindingMapChange(
		next: Map<string, ShaderMaterialTextureBindingRecord>
	): boolean {
		if (next.size !== this._textureBindings.size) {
			return true;
		}
		for (const [key, value] of next) {
			const current = this._textureBindings.get(key);
			if (!current) {
				return true;
			}
			if (
				current.texture !== value.texture ||
				current.explicitSlot !== value.explicitSlot ||
				current.uvSet !== value.uvSet ||
				current.linearOverride !== value.linearOverride ||
				current.webglUniform !== value.webglUniform
			) {
				return true;
			}
		}
		return false;
	}

	private _touchShaderRevision(): void {
		this._shaderRevision++;
	}
}
