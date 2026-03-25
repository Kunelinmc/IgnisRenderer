import type {
	ShaderLanguage,
	ShaderSourceKind,
	ShaderSourceSegmentKind,
	ShaderSourceSegmentMap,
	ShaderStage,
} from "./types";
import { mapShaderGeneratedLocation } from "./sourceMap";

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
			column: normalizedColumn ?? undefined,
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

export function parseWebGLShaderInfoLog(log: string): ShaderCompilerMessage[] {
	const normalized = typeof log === "string" ? log : String(log ?? "");
	const lines = normalized
		.split(/\r?\n/g)
		.map((line) => line.trim())
		.filter((line) => line.length > 0);
	if (lines.length <= 0) {
		return [];
	}
	const messages: ShaderCompilerMessage[] = [];
	for (const line of lines) {
		const message = parseWebGLShaderLogLine(line);
		if (message) {
			messages.push(message);
		}
	}
	if (messages.length <= 0) {
		messages.push({
			type: "error",
			message: normalized,
			raw: normalized,
		});
	}
	return messages;
}

export function normalizeWebGPUCompilationMessages(
	messages: readonly {
		type?: string;
		message?: string;
		lineNum?: number;
		linePos?: number;
		length?: number;
	}[]
): ShaderCompilerMessage[] {
	return messages.map((message) => ({
		type: normalizeMessageType(message.type),
		message: typeof message.message === "string" ? message.message : "Unknown",
		line: normalizeLine(message.lineNum) ?? undefined,
		column: normalizeColumn(message.linePos) ?? undefined,
		length:
			typeof message.length === "number" && Number.isFinite(message.length) ?
				Math.max(0, Math.floor(message.length))
			:	undefined,
		raw: typeof message.message === "string" ? message.message : undefined,
	}));
}

export function formatShaderCompilerMessages(
	messages: readonly ShaderMappedCompilerMessage[]
): string {
	return messages
		.map((message) => {
			const tag = message.type.toUpperCase();
			const generatedLocation =
				typeof message.line === "number" ?
					`L${message.line}:${message.column ?? 1}`
				:	"Ln/a";
			const sourceLocation =
				message.sourcePath && typeof message.sourceLine === "number" ?
					`${message.sourcePath}:${message.sourceLine}:${message.sourceColumn ?? 1}`
				:	"source:n/a";
			return `[${tag}] ${message.message} (${generatedLocation}, ${sourceLocation})`;
		})
		.join("\n");
}

function parseWebGLShaderLogLine(line: string): ShaderCompilerMessage | null {
	const webglStyle =
		/^(ERROR|WARNING)\s*:\s*\d+\s*:\s*(\d+)(?:\s*:\s*(\d+))?\s*:\s*(.+)$/i.exec(
			line
		);
	if (webglStyle) {
		return {
			type: normalizeMessageType(webglStyle[1]),
			line: normalizeLine(Number.parseInt(webglStyle[2], 10)) ?? undefined,
			column: normalizeColumn(Number.parseInt(webglStyle[3] ?? "1", 10)) ?? 1,
			message: webglStyle[4].trim(),
			raw: line,
		};
	}

	const glslCompilerStyle =
		/^(\d+)\((\d+)\)\s*:\s*(error|warning)[^:]*:\s*(.+)$/i.exec(line);
	if (glslCompilerStyle) {
		return {
			type: normalizeMessageType(glslCompilerStyle[3]),
			line:
				normalizeLine(Number.parseInt(glslCompilerStyle[2], 10)) ?? undefined,
			column: 1,
			message: glslCompilerStyle[4].trim(),
			raw: line,
		};
	}

	const fallbackLine = /^\s*(error|warning)\s*:\s*(.+)$/i.exec(line);
	if (fallbackLine) {
		return {
			type: normalizeMessageType(fallbackLine[1]),
			message: fallbackLine[2].trim(),
			raw: line,
		};
	}

	return {
		type: "error",
		message: line,
		raw: line,
	};
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

function normalizeMessageType(type: string | undefined): ShaderCompilerMessageType {
	if (typeof type !== "string") {
		return "info";
	}
	const normalized = type.trim().toLowerCase();
	if (normalized === "error") {
		return "error";
	}
	if (normalized === "warning" || normalized === "warn") {
		return "warning";
	}
	return "info";
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
