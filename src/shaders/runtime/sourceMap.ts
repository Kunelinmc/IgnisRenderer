import type {
	CompositeShaderSource,
	ShaderSourceSegment,
	ShaderSourceSegmentKind,
	ShaderSourceSegmentMap,
} from "./types";

const GENERATED_SOURCE_PATH = "<generated>";
export const SOURCE_MAP_SCHEMA_VERSION = 2;

interface LineOrigin {
	sourcePath: string;
	sourceLine: number;
	kind: ShaderSourceSegmentKind;
	label?: string;
}

export interface CompositeShaderSourcePart {
	code: string;
	sourceMap?: ShaderSourceSegmentMap | null;
	sourcePath?: string;
	kind?: ShaderSourceSegmentKind;
	label?: string;
}

export interface ShaderMappedLocation {
	generatedLine: number;
	generatedColumn: number;
	sourcePath: string;
	sourceLine: number;
	sourceColumn: number;
	kind: ShaderSourceSegmentKind;
	label?: string;
}

export function countSourceLines(code: string): number {
	if (code.length <= 0) {
		return 1;
	}
	let lineCount = 1;
	for (let i = 0; i < code.length; i++) {
		if (code.charCodeAt(i) === 10) {
			lineCount++;
		}
	}
	return lineCount;
}

export function createInlineShaderSourceMap(
	code: string,
	sourcePath: string,
	kind: ShaderSourceSegmentKind = "source",
	label?: string
): ShaderSourceSegmentMap {
	const lineCount = countSourceLines(code);
	return {
		schemaVersion: SOURCE_MAP_SCHEMA_VERSION,
		lineCount,
		segments: [
			{
				generatedLineStart: 1,
				generatedLineEnd: lineCount,
				sourcePath,
				sourceLineStart: 1,
				sourceLineEnd: lineCount,
				kind,
				label,
			},
		],
	};
}

export function createInlineCompositeShaderSource(
	code: string,
	sourcePath: string,
	kind: ShaderSourceSegmentKind = "source",
	label?: string
): CompositeShaderSource {
	return {
		code,
		sourceMap: createInlineShaderSourceMap(code, sourcePath, kind, label),
	};
}

export function sliceCompositeShaderSource(
	composite: CompositeShaderSource,
	startLine: number,
	endLine?: number
): CompositeShaderSource {
	const codeLines = composite.code.split(/\r?\n/g);
	const maxLine = Math.max(1, codeLines.length);
	const from = clampLine(startLine, maxLine);
	const to = clampLine(endLine ?? maxLine, maxLine);
	if (to < from) {
		return createInlineCompositeShaderSource(
			"",
			GENERATED_SOURCE_PATH,
			"generated"
		);
	}
	const selectedLines = codeLines.slice(from - 1, to);
	const selectedCode = selectedLines.join("\n");
	const origins = expandSourceMapToLineOrigins(
		composite.sourceMap,
		codeLines.length,
		GENERATED_SOURCE_PATH,
		"source"
	).slice(from - 1, to);
	return {
		code: selectedCode,
		sourceMap: compressLineOriginsToSourceMap(origins),
	};
}

export function composeCompositeShaderSources(
	parts: readonly CompositeShaderSourcePart[],
	separator: string = "\n\n",
	separatorKind: ShaderSourceSegmentKind = "generated"
): CompositeShaderSource {
	if (parts.length <= 0) {
		return createInlineCompositeShaderSource("", GENERATED_SOURCE_PATH, "generated");
	}

	const codeParts: string[] = [];
	const assignedOrigins = new Map<number, LineOrigin>();
	const separatorNewlineCount = countNewlineCharacters(separator);
	let currentLine = 1;
	for (let i = 0; i < parts.length; i++) {
		const part = parts[i];
		if (i > 0) {
			currentLine += separatorNewlineCount;
			codeParts.push(separator);
		}
		codeParts.push(part.code);
		const partOrigins = resolveCompositePartOrigins(part, i);
		writePartOrigins(assignedOrigins, partOrigins, currentLine);
		currentLine += Math.max(0, partOrigins.length - 1);
	}

	const code = codeParts.join("");
	const totalLineCount = Math.max(countSourceLines(code), currentLine, 1);
	const lineOrigins = buildFilledOrigins(
		assignedOrigins,
		totalLineCount,
		separatorKind
	);

	return {
		code,
		sourceMap: compressLineOriginsToSourceMap(lineOrigins),
	};
}

export function expandSourceMapToLineOrigins(
	sourceMap: ShaderSourceSegmentMap | null | undefined,
	lineCount: number,
	fallbackSourcePath: string,
	fallbackKind: ShaderSourceSegmentKind = "source"
): LineOrigin[] {
	const effectiveLineCount = Math.max(1, Math.floor(lineCount));
	const origins = createDefaultLineOrigins(
		effectiveLineCount,
		fallbackSourcePath,
		fallbackKind
	);
	if (!sourceMap || !Array.isArray(sourceMap.segments)) {
		return origins;
	}

	for (const segment of sourceMap.segments) {
		const start = clampLine(segment.generatedLineStart, effectiveLineCount);
		const end = clampLine(segment.generatedLineEnd, effectiveLineCount);
		if (end < start) {
			continue;
		}
		for (let generatedLine = start; generatedLine <= end; generatedLine++) {
			const offset = generatedLine - start;
			const sourceLine = clampLine(
				segment.sourceLineStart + offset,
				Math.max(segment.sourceLineEnd, segment.sourceLineStart)
			);
			origins[generatedLine - 1] = {
				sourcePath: segment.sourcePath || fallbackSourcePath,
				sourceLine,
				kind: segment.kind ?? fallbackKind,
				label: segment.label,
			};
		}
	}
	return origins;
}

export function mapShaderGeneratedLocation(
	sourceMap: ShaderSourceSegmentMap | null | undefined,
	generatedLine: number,
	generatedColumn: number = 1
): ShaderMappedLocation | null {
	if (!sourceMap || !Array.isArray(sourceMap.segments)) {
		return null;
	}
	const normalizedLine = Math.max(1, Math.floor(generatedLine));
	const normalizedColumn = Math.max(1, Math.floor(generatedColumn));
	for (const segment of sourceMap.segments) {
		if (
			normalizedLine < segment.generatedLineStart ||
			normalizedLine > segment.generatedLineEnd
		) {
			continue;
		}
		if (
			!hasColumnSpans(segment) ||
			segment.generatedLineStart !== segment.generatedLineEnd
		) {
			continue;
		}
		const generatedColumnStart = Math.max(1, segment.generatedColumnStart ?? 1);
		const generatedColumnEnd = Math.max(
			generatedColumnStart,
			segment.generatedColumnEnd ?? generatedColumnStart
		);
		if (
			normalizedColumn < generatedColumnStart ||
			normalizedColumn > generatedColumnEnd
		) {
			continue;
		}
		const sourceColumnStart = Math.max(1, segment.sourceColumnStart ?? 1);
		const sourceColumnEnd = Math.max(
			sourceColumnStart,
			segment.sourceColumnEnd ?? sourceColumnStart
		);
		return {
			generatedLine: normalizedLine,
			generatedColumn: normalizedColumn,
			sourcePath: segment.sourcePath,
			sourceLine: Math.max(
				1,
				Math.min(segment.sourceLineEnd, segment.sourceLineStart)
			),
			sourceColumn: mapColumnOffset(
				normalizedColumn,
				generatedColumnStart,
				generatedColumnEnd,
				sourceColumnStart,
				sourceColumnEnd
			),
			kind: segment.kind,
			label: segment.label,
		};
	}
	for (const segment of sourceMap.segments) {
		if (
			normalizedLine < segment.generatedLineStart ||
			normalizedLine > segment.generatedLineEnd
		) {
			continue;
		}
		const offset = normalizedLine - segment.generatedLineStart;
		return {
			generatedLine: normalizedLine,
			generatedColumn: normalizedColumn,
			sourcePath: segment.sourcePath,
			sourceLine: Math.max(
				1,
				Math.min(segment.sourceLineEnd, segment.sourceLineStart + offset)
			),
			sourceColumn: normalizedColumn,
			kind: segment.kind,
			label: segment.label,
		};
	}
	return null;
}

export function compressLineOriginsToSourceMap(
	lineOrigins: readonly LineOrigin[]
): ShaderSourceSegmentMap {
	const lineCount = Math.max(1, lineOrigins.length);
	const segments: ShaderSourceSegment[] = [];
	const safeOrigins =
		lineOrigins.length > 0 ?
			lineOrigins
		:	[
				{
					sourcePath: GENERATED_SOURCE_PATH,
					sourceLine: 1,
					kind: "generated" as ShaderSourceSegmentKind,
				},
			];

	let activeStart = 1;
	let activeOrigin = safeOrigins[0];
	for (let line = 2; line <= safeOrigins.length; line++) {
		const nextOrigin = safeOrigins[line - 1];
		const isSequentialSourceLine =
			nextOrigin.sourceLine === activeOrigin.sourceLine + (line - activeStart);
		const sameBucket =
			nextOrigin.sourcePath === activeOrigin.sourcePath &&
			nextOrigin.kind === activeOrigin.kind &&
			nextOrigin.label === activeOrigin.label &&
			isSequentialSourceLine;
		if (sameBucket) {
			continue;
		}
		segments.push({
			generatedLineStart: activeStart,
			generatedLineEnd: line - 1,
			sourcePath: activeOrigin.sourcePath,
			sourceLineStart: activeOrigin.sourceLine,
			sourceLineEnd: activeOrigin.sourceLine + (line - activeStart - 1),
			kind: activeOrigin.kind,
			label: activeOrigin.label,
		});
		activeStart = line;
		activeOrigin = nextOrigin;
	}

	const lastLine = safeOrigins.length;
	segments.push({
		generatedLineStart: activeStart,
		generatedLineEnd: lastLine,
		sourcePath: activeOrigin.sourcePath,
		sourceLineStart: activeOrigin.sourceLine,
		sourceLineEnd: activeOrigin.sourceLine + (lastLine - activeStart),
		kind: activeOrigin.kind,
		label: activeOrigin.label,
	});

	return {
		schemaVersion: SOURCE_MAP_SCHEMA_VERSION,
		lineCount,
		segments,
	};
}

function resolveCompositePartOrigins(
	part: CompositeShaderSourcePart,
	partIndex: number
): LineOrigin[] {
	const fallbackPath = part.sourcePath ?? `<inline:${partIndex}>`;
	const fallbackKind = part.kind ?? "source";
	const partLineCount = countSourceLines(part.code);
	return expandSourceMapToLineOrigins(
		part.sourceMap,
		partLineCount,
		fallbackPath,
		fallbackKind
	);
}

function writePartOrigins(
	target: Map<number, LineOrigin>,
	partOrigins: readonly LineOrigin[],
	startLine: number
): void {
	for (let i = 0; i < partOrigins.length; i++) {
		target.set(startLine + i, partOrigins[i]);
	}
}

function buildFilledOrigins(
	assignedOrigins: ReadonlyMap<number, LineOrigin>,
	lineCount: number,
	separatorKind: ShaderSourceSegmentKind
): LineOrigin[] {
	const origins: LineOrigin[] = [];
	for (let line = 1; line <= lineCount; line++) {
		const assigned = assignedOrigins.get(line);
		if (assigned) {
			origins.push(assigned);
			continue;
		}
		origins.push({
			sourcePath: GENERATED_SOURCE_PATH,
			sourceLine: line,
			kind: separatorKind,
		});
	}
	return origins;
}

function createDefaultLineOrigins(
	lineCount: number,
	sourcePath: string,
	kind: ShaderSourceSegmentKind
): LineOrigin[] {
	const origins: LineOrigin[] = [];
	for (let i = 0; i < lineCount; i++) {
		origins.push({
			sourcePath,
			sourceLine: i + 1,
			kind,
		});
	}
	return origins;
}

function clampLine(value: number, maxLine: number): number {
	if (!Number.isFinite(value)) {
		return 1;
	}
	const normalized = Math.floor(value);
	return Math.min(Math.max(normalized, 1), Math.max(1, Math.floor(maxLine)));
}

function countNewlineCharacters(value: string): number {
	if (value.length <= 0) {
		return 0;
	}
	let count = 0;
	for (let i = 0; i < value.length; i++) {
		if (value.charCodeAt(i) === 10) {
			count++;
		}
	}
	return count;
}

function hasColumnSpans(segment: ShaderSourceSegment): boolean {
	return (
		typeof segment.generatedColumnStart === "number" &&
		typeof segment.generatedColumnEnd === "number" &&
		typeof segment.sourceColumnStart === "number" &&
		typeof segment.sourceColumnEnd === "number"
	);
}

function mapColumnOffset(
	column: number,
	generatedColumnStart: number,
	generatedColumnEnd: number,
	sourceColumnStart: number,
	sourceColumnEnd: number
): number {
	if (generatedColumnEnd <= generatedColumnStart) {
		return sourceColumnStart;
	}
	const clampedGenerated = Math.min(
		Math.max(column, generatedColumnStart),
		generatedColumnEnd
	);
	const sourceOffset = clampedGenerated - generatedColumnStart;
	return Math.min(sourceColumnStart + sourceOffset, sourceColumnEnd);
}
