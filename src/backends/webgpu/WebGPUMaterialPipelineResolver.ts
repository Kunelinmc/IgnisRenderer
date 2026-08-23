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
import type { WebGPUMaterialUniformData } from "./types";

/** @internal Immutable shader-runtime inputs used by material pipeline resolution. */
export interface WebGPUShaderRuntimeView {
	readonly revision: number;
	readonly mode: "strict" | "warn" | "silent";
	readonly directiveCacheTag: string;
	readonly supportsRuntimeInjects: boolean;
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
		supportsRuntimeInjects: directiveCacheTag !== "none",
	};
}

export interface WebGPUCustomMaterialProgramDescriptor {
	readonly kind: "custom";
	readonly cacheKey: string;
	readonly regularProgram: ResolvedWebGPUShaderProgram | null;
	readonly depthPrepassProgram: ResolvedWebGPUDepthPrepassProgram | null;
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
	readonly shaderRuntime: WebGPUShaderRuntimeView;
	readonly diagnostic: {
		readonly materialName: string;
		readonly shaderId: number | null;
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
			wireframe ? 1 : 0,
			targetMode,
			purpose,
			isMask ? 1 : 0,
			runtime.revision,
			runtime.directiveCacheTag,
			runtime.supportsRuntimeInjects ? 1 : 0,
		].join("|");
		const entries = this._entries.get(material);
		const cached = entries?.find((entry) => entry.key === key);
		if (cached) return cached.state;

		const program = this._resolveProgram(
			material,
			targetMode,
			purpose,
			isMask,
			runtime.supportsRuntimeInjects,
		);
		const shaderCacheKey = program.kind === "custom"
			? `shader:${program.cacheKey}|runtime:${runtime.revision}|directive:${runtime.directiveCacheTag}`
			: `builtin-scene|runtime:${runtime.revision}|directive:${runtime.directiveCacheTag}`;
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
			shaderRuntime: runtime,
			diagnostic: {
				materialName: material.name,
				shaderId: material instanceof ShaderMaterial ? material.shaderId : null,
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
		enableRuntimeInjects: boolean,
	): WebGPUMaterialProgramDescriptor {
		if (!(material instanceof ShaderMaterial)) return { kind: "builtin" };
		const cacheKey = material.getWebGPUCacheKey();
		if (purpose === "early-z" && isMask) {
			return {
				kind: "custom",
				cacheKey,
				regularProgram: null,
				depthPrepassProgram: material.resolveWebGPUDepthPrepassProgram(
					targetMode,
					{ enableRuntimeInjects },
				),
			};
		}
		return {
			kind: "custom",
			cacheKey,
			regularProgram: material.resolveWebGPUProgram(targetMode, {
				enableRuntimeInjects,
			}),
			depthPrepassProgram: null,
		};
	}
}
