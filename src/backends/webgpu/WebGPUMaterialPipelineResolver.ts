import {
	AlphaMode,
	type CullMode,
	type Material,
} from "../../materials/Material";
import {
	ShaderMaterial,
	type ResolvedWebGPUDepthPrepassProgram,
	type ResolvedWebGPUShaderProgram,
	type ShaderTargetMode,
} from "../../materials/ShaderMaterial";
import {
	isMaterialTransparentPass,
	materialUsesTransmission,
} from "../../materials/transparency";
import type {
	WebGPUMaterialUniformData,
	WebGPUShadingFamily,
} from "./types";
import {
	createShaderMaterialSourceBlocks,
	SHADER_MATERIAL_SOURCE_ABI_REVISION,
} from "../../shaders/ShaderMaterialSource";
import type { ShaderGeneratedSourceBlock } from "../../shaders/runtime";
import { WEBGPU_MODEL_BINDING_SHADER_UNIFORMS } from "./constants";

function createWebGPUShaderMaterialSourceBlocks(
	material: ShaderMaterial,
	stage: "vertex" | "fragment",
	source: string,
): ShaderGeneratedSourceBlock[] {
	return createShaderMaterialSourceBlocks({
		material,
		language: "wgsl",
		stage,
		source,
		wgslUniformBinding: WEBGPU_MODEL_BINDING_SHADER_UNIFORMS,
	});
}

/** @internal Immutable shader-runtime inputs used by material pipeline resolution. */
export interface WebGPUShaderRuntimeView {
	readonly revision: number;
	readonly mode: "strict" | "warn" | "silent";
	readonly directiveCacheTag: string;
}

/** @internal Reads the explicit runtime view with a compatibility fallback for test hosts. */
export function readWebGPUShaderRuntimeView(source: {
	getShaderRuntimeView?: () => WebGPUShaderRuntimeView;
	getShaderDirectiveCacheTag?: () => string;
	shaderRuntime?: {
		revision?: number;
		getMode?: () => "strict" | "warn" | "silent";
	};
}): WebGPUShaderRuntimeView {
	if (typeof source.getShaderRuntimeView === "function") {
		return source.getShaderRuntimeView();
	}
	const directiveCacheTag = source.getShaderDirectiveCacheTag?.() ?? "none";
	return {
		revision:
			typeof source.shaderRuntime?.revision === "number"
				? source.shaderRuntime.revision
				: 0,
		mode: source.shaderRuntime?.getMode?.() ?? "strict",
		directiveCacheTag,
	};
}

export interface WebGPUCustomMaterialProgramDescriptor {
	readonly kind: "custom";
	readonly cacheKey: string;
	readonly regularProgram: ResolvedWebGPUShaderProgram | null;
	readonly depthPrepassProgram: ResolvedWebGPUDepthPrepassProgram | null;
	readonly generatedSourceBlocks: {
		readonly vertex: readonly ShaderGeneratedSourceBlock[];
		readonly fragment: readonly ShaderGeneratedSourceBlock[];
	};
}

export interface WebGPUBuiltinMaterialProgramDescriptor {
	readonly kind: "builtin";
}

export type WebGPUMaterialProgramDescriptor =
	| WebGPUBuiltinMaterialProgramDescriptor
	| WebGPUCustomMaterialProgramDescriptor;

/** @internal Immutable material policy consumed by WebGPU pipeline resources. */
export interface WebGPUMaterialPipelineState {
	readonly materialRevision: number;
	readonly pipelineKey: string;
	readonly shaderCacheKey: string;
	readonly cullMode: CullMode;
	readonly depthWrite: boolean;
	readonly alphaMode: AlphaMode;
	readonly transparent: boolean;
	readonly usesTransmission: boolean;
	readonly wireframe: boolean;
	readonly shadingFamily: WebGPUShadingFamily;
	readonly shaderRuntime: WebGPUShaderRuntimeView;
	readonly diagnostic: {
		readonly materialName: string;
		readonly shaderId: number | null;
		readonly fallbackReason: string | null;
	};
	readonly program: WebGPUMaterialProgramDescriptor;
}

export type WebGPUMaterialPipelinePurpose = "scene" | "early-z";

interface CachedPipelineState {
	readonly key: string;
	readonly state: WebGPUMaterialPipelineState;
}

/**
 * Resolves backend-neutral material objects into immutable WebGPU pipeline policy.
 *
 * @internal Owned by `WebGPUFrameServiceOwner`; applications use materials through
 * scene draw packets.
 */
export class WebGPUMaterialPipelineResolver {
	private _entries = new WeakMap<Material, CachedPipelineState[]>();

	/**
	 * Resolves built-in depth eligibility and primitive policy without a GPU material snapshot.
	 * @internal Owned by WebGPU draw preparation; applications submit materials through scenes.
	 * @param material Built-in material; the caller must exclude `ShaderMaterial`.
	 * @param runtime Current shader-runtime fingerprint used for depth pipeline identity.
	 * @returns Immutable policy shared by compatible depth draws.
	 */
	public resolveBuiltinDepth(
		material: Material,
		runtime: WebGPUShaderRuntimeView,
	): WebGPUMaterialPipelineState {
		const revision = material._getRevisionInternal();
		// Preparation can await GPU work after the once-per-frame revision refresh.
		// Read depth policy directly so mid-preparation authoring changes remain observable.
		const cullMode = material.cullMode;
		const depthWrite = material.depthWrite;
		const alphaMode = material.alphaMode;
		const transparent = isMaterialTransparentPass(material);
		const usesTransmission = materialUsesTransmission(material);
		const key = `builtin-depth|${cullMode}|${depthWrite}|${alphaMode}|` +
			`${transparent}|${usesTransmission}|${runtime.revision}|${runtime.mode}|` +
			runtime.directiveCacheTag;
		const entries = this._entries.get(material);
		const cached = entries?.find((entry) => entry.key === key);
		if (cached) return cached.state;
		const state: WebGPUMaterialPipelineState = {
			materialRevision: revision,
			pipelineKey: "builtin-depth",
			shaderCacheKey: `builtin-depth|runtime:${runtime.revision}|` +
				`directive:${runtime.directiveCacheTag}`,
			cullMode,
			depthWrite,
			alphaMode,
			transparent,
			usesTransmission,
			// Early-Z uses solid geometry; authored wireframe is a separate color overlay.
			wireframe: false,
			// The standalone program has no lighting family; retain the common state type.
			shadingFamily: "unlit",
			shaderRuntime: runtime,
			diagnostic: { materialName: material.name, shaderId: null, fallbackReason: null },
			program: { kind: "builtin" },
		};
		const nextEntries = entries ?? [];
		nextEntries.push({ key, state });
		while (nextEntries.length > 16) nextEntries.shift();
		if (!entries) this._entries.set(material, nextEntries);
		return state;
	}

	public resolve(
		material: Material,
		materialData: WebGPUMaterialUniformData,
		wireframe: boolean,
		targetMode: ShaderTargetMode,
		purpose: WebGPUMaterialPipelinePurpose,
		runtime: WebGPUShaderRuntimeView,
	): WebGPUMaterialPipelineState {
		const revision = material._getRevisionInternal();
		const isMask = material.alphaMode === AlphaMode.Mask;
		const key = [
			revision,
			materialData.pipelineKey,
			materialData.shadingFamily,
			wireframe ? 1 : 0,
			targetMode,
			purpose,
			isMask ? 1 : 0,
			runtime.revision,
			runtime.directiveCacheTag,
			SHADER_MATERIAL_SOURCE_ABI_REVISION,
		].join("|");
		const entries = this._entries.get(material);
		const cached = entries?.find((entry) => entry.key === key);
		if (cached) return cached.state;

		let fallbackReason: string | null = null;
		let program: WebGPUMaterialProgramDescriptor;
		try {
			program = this._resolveProgram(
				material,
				targetMode,
				purpose,
				isMask,
			);
		} catch (error) {
			if (purpose !== "scene" || runtime.mode !== "warn") throw error;
			program = { kind: "builtin" };
			fallbackReason = String(error);
		}
		const shaderCacheKey = program.kind === "custom"
			? `shader:${program.cacheKey}|family:${materialData.shadingFamily}|runtime:${runtime.revision}|directive:${runtime.directiveCacheTag}`
			: `builtin-scene|family:${materialData.shadingFamily}|runtime:${runtime.revision}|directive:${runtime.directiveCacheTag}`;
		const state: WebGPUMaterialPipelineState = {
			materialRevision: revision,
			pipelineKey: materialData.pipelineKey,
			shaderCacheKey,
			cullMode: material.cullMode,
			depthWrite: material.depthWrite,
			alphaMode: material.alphaMode,
			transparent: isMaterialTransparentPass(material),
			usesTransmission: materialUsesTransmission(material),
			wireframe,
			shadingFamily: materialData.shadingFamily,
			shaderRuntime: runtime,
			diagnostic: {
				materialName: material.name,
				shaderId: material instanceof ShaderMaterial ? material.shaderId : null,
				fallbackReason,
			},
			program,
		};
		const nextEntries = entries ?? [];
		nextEntries.push({ key, state });
		while (nextEntries.length > 16) nextEntries.shift();
		if (!entries) this._entries.set(material, nextEntries);
		return state;
	}

	public clear(): void {
		this._entries = new WeakMap();
	}

	private _resolveProgram(
		material: Material,
		targetMode: ShaderTargetMode,
		purpose: WebGPUMaterialPipelinePurpose,
		isMask: boolean,
	): WebGPUMaterialProgramDescriptor {
		if (!(material instanceof ShaderMaterial)) return { kind: "builtin" };
		const cacheKey =
			`${material.getWebGPUCacheKey()}|abi:${SHADER_MATERIAL_SOURCE_ABI_REVISION}`;
		if (purpose === "early-z" && isMask) {
			const depthPrepassProgram = material.resolveWebGPUDepthPrepassProgram(
				targetMode,
			);
			return {
				kind: "custom",
				cacheKey,
				regularProgram: null,
				depthPrepassProgram,
				generatedSourceBlocks: {
					vertex:
						depthPrepassProgram ?
							createWebGPUShaderMaterialSourceBlocks(
								material,
								"vertex",
								depthPrepassProgram.vertexCode,
							)
							: [],
					fragment:
						depthPrepassProgram ?
							createWebGPUShaderMaterialSourceBlocks(
								material,
								"fragment",
								depthPrepassProgram.fragmentCode,
							)
							: [],
				},
			};
		}
		const regularProgram = material.resolveWebGPUProgram(targetMode);
		return {
			kind: "custom",
			cacheKey,
			regularProgram,
			depthPrepassProgram: null,
			generatedSourceBlocks: {
				vertex: createWebGPUShaderMaterialSourceBlocks(
					material,
					"vertex",
					regularProgram.vertexCode,
				),
				fragment: createWebGPUShaderMaterialSourceBlocks(
					material,
					"fragment",
					regularProgram.fragmentCode,
				),
			},
		};
	}
}
