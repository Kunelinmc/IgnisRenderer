import {
	ShaderCompileError,
	type ShaderBackendCompileStage,
	type ShaderBackendCompileResult,
	type ShaderCompilerMessage,
	type ShaderProcessResult,
} from "../../shaders/runtime";
import { Logger } from "../../foundation/Logger";
import type { ShaderModuleDesc } from "../types";

export class WebGPUShaderModuleCompiler {
	constructor(private _shaderCompileStage: ShaderBackendCompileStage) {}

	public async processShaderSource(
		desc: ShaderModuleDesc
	): Promise<ShaderBackendCompileResult> {
		const sanitizedCode = this._stripUtf8BomCharacters(desc.code, desc.label);
		const directiveSourcePath =
			desc.sourceMap?.segments[0]?.sourcePath ?? desc.label ?? "<webgpu-shader>";
		return this._shaderCompileStage.compileAsync({
			code: sanitizedCode,
			language: desc.language ?? "wgsl",
			stage: desc.stage ?? "unknown",
			entryPoint: desc.entryPoint,
			label: desc.label,
			sourceKind: desc.sourceKind ?? "unknown",
			sourceMap: desc.sourceMap ?? null,
			directiveSourcePath,
		});
	}

	public createShaderModuleError(error: unknown, desc: ShaderModuleDesc): Error {
		if (error instanceof ShaderCompileError) {
			return error;
		}
		const compilerMessage: ShaderCompilerMessage = {
			type: "error",
			message: String(error),
		};
		return new ShaderCompileError({
			backend: "webgpu",
			language: desc.language ?? "wgsl",
			stage: desc.stage ?? "unknown",
			label: desc.label,
			sourceKind: desc.sourceKind ?? "unknown",
			variantKey: desc.variantKey,
			materialId: desc.materialId,
			code: desc.code,
			sourceMap: desc.sourceMap ?? null,
			messages: [compilerMessage],
			cause: error,
		});
	}

	public reportShaderRuntimeDiagnostics(
		desc: ShaderModuleDesc,
		result: ShaderProcessResult
	): void {
		const keyPrefix = desc.label && desc.label.length > 0 ? desc.label : "unnamed";
		for (const diagnostic of result.diagnostics) {
			const locationSuffix =
				diagnostic.sourcePath && typeof diagnostic.line === "number"
					? ` (${diagnostic.sourcePath}:${diagnostic.line}:${diagnostic.column ?? 1})`
					: "";
			const key =
				`webgpu-shader-runtime-${diagnostic.severity}` +
				`-${diagnostic.code}-${keyPrefix}` +
				`-${diagnostic.sourcePath ?? ""}-${diagnostic.line ?? ""}-${diagnostic.column ?? ""}`;
			Logger.warn(
				`[${key}] WebGPU shader runtime ${diagnostic.severity} [${keyPrefix}] ` +
					`${diagnostic.code}: ${diagnostic.message}${locationSuffix}`,
				{ scope: "WebGPUBackend", onceKey: key }
			);
		}
	}

	private _stripUtf8BomCharacters(code: string, label?: string): string {
		if (!code.includes("\uFEFF")) {
			return code;
		}
		const shaderLabel = label && label.length > 0 ? label : "unnamed";
		const key = `webgpu-shader-bom:${shaderLabel}`;
		Logger.warn(
			`[${key}] WebGPU shader source [${shaderLabel}] contained UTF-8 BOM characters; stripping before compilation.`,
			{ scope: "WebGPUBackend", onceKey: key }
		);
		return code.replace(/\uFEFF/g, "");
	}
}
