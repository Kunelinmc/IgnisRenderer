import {
	mapShaderCompilerMessages,
	type ShaderCompilerMessage,
	type ShaderCompilerMessageType,
	type ShaderMappedCompilerMessage,
} from "../../foundation/Error";

export {
	mapShaderCompilerMessages,
	ShaderCompileError,
} from "../../foundation/Error";
export type {
	ShaderCompileErrorInit,
	ShaderCompilerBackend,
	ShaderCompilerMessage,
	ShaderCompilerMessageType,
	ShaderMappedCompilerMessage,
} from "../../foundation/Error";

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
