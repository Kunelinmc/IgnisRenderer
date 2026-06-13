import type { CSGDiagnostic } from "../csg/types";
import { mapShaderGeneratedLocation } from "../shaders/runtime/sourceMap";
import type {
	ShaderLanguage,
	ShaderSourceKind,
	ShaderSourceSegmentKind,
	ShaderSourceSegmentMap,
	ShaderStage,
} from "../shaders/runtime/types";

export type ShaderCompilerMessageType = "error" | "warning" | "info";
export type ShaderCompilerBackend = "webgpu" | "webgl" | "unknown";

export interface ShaderCompilerMessage {
	type: ShaderCompilerMessageType;
	message: string;
	line?: number;
	column?: number;
	length?: number;
	raw?: string;
}

export interface ShaderMappedCompilerMessage extends ShaderCompilerMessage {
	sourcePath?: string;
	sourceLine?: number;
	sourceColumn?: number;
	segmentKind?: ShaderSourceSegmentKind;
	segmentLabel?: string;
	snippet?: string;
}

export interface ShaderCompileErrorInit {
	backend: ShaderCompilerBackend;
	language: ShaderLanguage;
	stage: ShaderStage;
	label?: string;
	sourceKind?: ShaderSourceKind;
	variantKey?: string;
	materialId?: string;
	code: string;
	sourceMap?: ShaderSourceSegmentMap | null;
	messages: ShaderCompilerMessage[];
	rawLog?: string;
	cause?: unknown;
}

export class ShaderCompileError extends Error {
	public readonly backend: ShaderCompilerBackend;
	public readonly language: ShaderLanguage;
	public readonly stage: ShaderStage;
	public readonly label: string | null;
	public readonly sourceKind: ShaderSourceKind;
	public readonly variantKey: string | null;
	public readonly materialId: string | null;
	public readonly code: string;
	public readonly sourceMap: ShaderSourceSegmentMap | null;
	public readonly messages: ShaderMappedCompilerMessage[];
	public readonly rawLog: string | null;

	public constructor(init: ShaderCompileErrorInit) {
		const mappedMessages = mapShaderCompilerMessages(
			init.messages,
			init.code,
			init.sourceMap
		);
		const first =
			mappedMessages.find((message) => message.type === "error") ??
			mappedMessages[0];
		const label = init.label ?? "unnamed-shader";
		super(
			buildCompileErrorMessage(
				init.backend,
				label,
				init.language,
				init.stage,
				first
			)
		);
		this.name = "ShaderCompileError";
		this.backend = init.backend;
		this.language = init.language;
		this.stage = init.stage;
		this.label = init.label ?? null;
		this.sourceKind = init.sourceKind ?? "unknown";
		this.variantKey = init.variantKey ?? null;
		this.materialId = init.materialId ?? null;
		this.code = init.code;
		this.sourceMap = init.sourceMap ?? null;
		this.messages = mappedMessages;
		this.rawLog = init.rawLog ?? null;
		(this as { cause?: unknown }).cause = init.cause;
	}
}

/** @internal Used by the shader directive expression parser. */
export class ConditionParseError extends Error {
	public readonly column: number;

	public constructor(message: string, column: number) {
		super(message);
		this.name = "ConditionParseError";
		this.column = Math.max(1, Math.floor(column));
	}
}

/** @internal Used by the CSG solver to preserve structured diagnostics. */
export class CSGBuildError extends Error {
	public readonly diagnostic: CSGDiagnostic;

	public constructor(diagnostic: CSGDiagnostic) {
		super(diagnostic.message);
		this.diagnostic = diagnostic;
	}
}

/** @internal Used by the WebGPU backend lifecycle. */
export class WebGPUShaderModuleCreationInvalidatedError extends Error {
	public constructor(label?: string) {
		const shaderLabel = label && label.length > 0 ? label : "unnamed";
		super(
			`WebGPU shader module creation was invalidated by backend lifecycle reset [${shaderLabel}].`
		);
		this.name = "WebGPUShaderModuleCreationInvalidatedError";
	}
}

/** @internal Used by the WebGPU backend lifecycle. */
export class WebGPUPipelineCreationInvalidatedError extends Error {
	public constructor(label?: string) {
		const pipelineLabel = label && label.length > 0 ? label : "unnamed";
		super(
			`WebGPU pipeline creation was invalidated by backend lifecycle reset [${pipelineLabel}].`
		);
		this.name = "WebGPUPipelineCreationInvalidatedError";
	}
}

export function mapShaderCompilerMessages(
	messages: readonly ShaderCompilerMessage[],
	code: string,
	sourceMap: ShaderSourceSegmentMap | null | undefined
): ShaderMappedCompilerMessage[] {
	return messages.map((message) => {
		const normalizedLine = normalizeLine(message.line);
		const normalizedColumn = normalizeColumn(message.column);
		const mapped =
			normalizedLine ?
				mapShaderGeneratedLocation(sourceMap, normalizedLine, normalizedColumn)
			:	null;
		return {
			...message,
			line: normalizedLine ?? undefined,
			column: normalizedColumn,
			sourcePath: mapped?.sourcePath,
			sourceLine: mapped?.sourceLine,
			sourceColumn: mapped?.sourceColumn,
			segmentKind: mapped?.kind,
			segmentLabel: mapped?.label,
			snippet:
				normalizedLine ?
					extractShaderLineSnippet(code, normalizedLine)
				:	undefined,
		};
	});
}

function buildCompileErrorMessage(
	backend: ShaderCompilerBackend,
	label: string,
	language: ShaderLanguage,
	stage: ShaderStage,
	firstMessage: ShaderMappedCompilerMessage | undefined
): string {
	const base = `Shader compile failed [${backend}] ${label} (${language}/${stage}).`;
	if (!firstMessage) {
		return base;
	}
	const sourceLocation =
		firstMessage.sourcePath && typeof firstMessage.sourceLine === "number" ?
			` ${firstMessage.sourcePath}:${firstMessage.sourceLine}:${firstMessage.sourceColumn ?? 1}`
		:	"";
	return `${base} ${firstMessage.message}${sourceLocation}`;
}

function extractShaderLineSnippet(code: string, line: number): string {
	if (code.length <= 0) {
		return "";
	}
	const lines = code.split(/\r?\n/g);
	if (line < 1 || line > lines.length) {
		return "";
	}
	return lines[line - 1];
}

function normalizeLine(value: number | undefined): number | null {
	if (typeof value !== "number" || !Number.isFinite(value)) {
		return null;
	}
	const normalized = Math.floor(value);
	return normalized >= 1 ? normalized : 1;
}

function normalizeColumn(value: number | undefined): number {
	if (typeof value !== "number" || !Number.isFinite(value)) {
		return 1;
	}
	return Math.max(1, Math.floor(value));
}
