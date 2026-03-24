import { Material, type MaterialParams, ShadingModel } from "./Material";

export type ShaderTargetMode = "single" | "mrt";
export type ShaderStageKind = "vertex" | "fragment-single" | "fragment-mrt";

export interface ShaderMaterialProgram {
	vertex: string;
	fragmentSingle: string;
	fragmentMRT?: string;
}

export interface ShaderMaterialWebGLProgram {
	vertex: string;
	fragment: string;
}

export type ShaderMaterialGLSLToWGSL = (
	source: string,
	stage: ShaderStageKind
) => string;

export interface ShaderMaterialParams extends MaterialParams {
	vertexEntryPoint?: string;
	fragmentSingleEntryPoint?: string;
	fragmentMRTEntryPoint?: string;
	webgpuWGSL?: ShaderMaterialProgram | null;
	webgpuGLSL?: ShaderMaterialProgram | null;
	webglGLSL?: ShaderMaterialWebGLProgram | null;
	glslToWgsl?: ShaderMaterialGLSLToWGSL;
}

export interface ResolvedWebGPUShaderProgram {
	vertexCode: string;
	fragmentCode: string;
	vertexEntryPoint: string;
	fragmentEntryPoint: string;
}

export interface ResolvedWebGLShaderProgram {
	vertexCode: string;
	fragmentCode: string;
}

let SHADER_MATERIAL_ID = 1;

export class ShaderMaterial extends Material {
	public readonly shaderId: number;
	public vertexEntryPoint: string;
	public fragmentSingleEntryPoint: string;
	public fragmentMRTEntryPoint: string;

	private _webgpuWGSL: ShaderMaterialProgram | null;
	private _webgpuGLSL: ShaderMaterialProgram | null;
	private _webglGLSL: ShaderMaterialWebGLProgram | null;
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
		this._webgpuWGSL = null;
		this._webgpuGLSL = null;
		this._webglGLSL = null;
		this._glslToWgsl = params.glslToWgsl ?? null;
		this._shaderRevision = 0;

		if (params.webgpuWGSL) {
			this.setWebGPUWGSL(params.webgpuWGSL);
		}
		if (params.webgpuGLSL) {
			this.setWebGPUGLSL(params.webgpuGLSL);
		}
		if (params.webglGLSL) {
			this.setWebGLGLSL(params.webglGLSL);
		}
	}

	public get shaderRevision(): number {
		return this._shaderRevision;
	}

	public get webgpuWGSL(): ShaderMaterialProgram | null {
		return this._webgpuWGSL ? { ...this._webgpuWGSL } : null;
	}

	public get webgpuGLSL(): ShaderMaterialProgram | null {
		return this._webgpuGLSL ? { ...this._webgpuGLSL } : null;
	}

	public get glslToWgsl(): ShaderMaterialGLSLToWGSL | null {
		return this._glslToWgsl;
	}

	public get webglGLSL(): ShaderMaterialWebGLProgram | null {
		return this._webglGLSL ? { ...this._webglGLSL } : null;
	}

	public setWebGPUWGSL(program: ShaderMaterialProgram | null): void {
		this._webgpuWGSL = program ? { ...program } : null;
		this._shaderRevision++;
	}

	public setWebGPUGLSL(program: ShaderMaterialProgram | null): void {
		this._webgpuGLSL = program ? { ...program } : null;
		this._shaderRevision++;
	}

	public setWebGLGLSL(program: ShaderMaterialWebGLProgram | null): void {
		this._webglGLSL = program ? { ...program } : null;
		this._shaderRevision++;
	}

	public setGLSLToWGSL(transpiler: ShaderMaterialGLSLToWGSL | null): void {
		this._glslToWgsl = transpiler;
		this._shaderRevision++;
	}

	public getWebGPUCacheKey(): string {
		return [
			this.shaderId,
			this._shaderRevision,
			this.vertexEntryPoint,
			this.fragmentSingleEntryPoint,
			this.fragmentMRTEntryPoint,
		].join(":");
	}

	public getWebGLCacheKey(): string {
		return [this.shaderId, this._shaderRevision].join(":");
	}

	public resolveWebGPUProgram(
		mode: ShaderTargetMode
	): ResolvedWebGPUShaderProgram {
		const vertexCode = this._resolveStageCode("vertex", mode);
		const fragmentStage = mode === "mrt" ? "fragment-mrt" : "fragment-single";
		const fragmentCode = this._resolveStageCode(fragmentStage, mode);
		return {
			vertexCode,
			fragmentCode,
			vertexEntryPoint: this.vertexEntryPoint,
			fragmentEntryPoint:
				mode === "mrt" ?
					this.fragmentMRTEntryPoint
				:	this.fragmentSingleEntryPoint,
		};
	}

	public resolveWebGLProgram(): ResolvedWebGLShaderProgram {
		const vertexCode =
			this._webglGLSL?.vertex ?? this._webgpuGLSL?.vertex ?? null;
		const fragmentCode =
			this._webglGLSL?.fragment ??
			this._webgpuGLSL?.fragmentSingle ??
			this._webgpuGLSL?.fragmentMRT ??
			null;

		if (
			typeof vertexCode !== "string" ||
			vertexCode.trim().length === 0 ||
			typeof fragmentCode !== "string" ||
			fragmentCode.trim().length === 0
		) {
			throw new Error(
				`ShaderMaterial ${this.name} is missing WebGL GLSL source; ` +
					"call setWebGLGLSL() or provide webgpuGLSL " +
					"vertex/fragmentSingle fallback"
			);
		}

		return {
			vertexCode,
			fragmentCode,
		};
	}

	private _resolveStageCode(
		stage: ShaderStageKind,
		mode: ShaderTargetMode
	): string {
		const wgslSource = this._getProgramStageSource(
			this._webgpuWGSL,
			stage,
			mode
		);
		if (wgslSource) {
			return wgslSource;
		}

		const glslSource = this._getProgramStageSource(
			this._webgpuGLSL,
			stage,
			mode
		);
		if (!glslSource) {
			throw new Error(
				`ShaderMaterial ${this.name} is missing ${stage} shader source for ${mode} mode`
			);
		}

		if (!this._glslToWgsl) {
			throw new Error(
				`ShaderMaterial ${this.name} has GLSL source but no glslToWgsl transpiler; provide webgpuWGSL directly or call setGLSLToWGSL()`
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

	private _getProgramStageSource(
		program: ShaderMaterialProgram | null,
		stage: ShaderStageKind,
		mode: ShaderTargetMode
	): string | null {
		if (!program) {
			return null;
		}

		switch (stage) {
			case "vertex":
				return program.vertex ?? null;
			case "fragment-single":
				return program.fragmentSingle ?? null;
			case "fragment-mrt":
				return mode === "mrt" ?
						(program.fragmentMRT ?? program.fragmentSingle ?? null)
					:	(program.fragmentSingle ?? null);
			default:
				return null;
		}
	}
}
