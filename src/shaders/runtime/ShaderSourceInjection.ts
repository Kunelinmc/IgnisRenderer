import {
	composeCompositeShaderSources,
	sliceCompositeShaderSource,
} from "./sourceMap";
import { normalizeLanguage } from "./runtimeShared";
import type {
	CompositeShaderSource,
	ShaderGLSLInjectionAnchor,
	ShaderInjectionAnchor,
	ShaderResolvedGLSLInjectionAnchors,
	ShaderResolvedWGSLInjectionAnchors,
	ShaderSourceSegmentMap,
	ShaderWGSLInjectionAnchor,
} from "./types";

export interface InjectionBlock {
	code: string;
	sourcePath: string;
	label: string;
	anchor: ShaderInjectionAnchor;
}

interface GLSLInsertionAnchors {
	afterVersion: number;
	afterPrecision: number;
	afterDefines: number;
	afterStruct: number;
	afterUniforms: number;
	beforeEntryPoint: number;
	endOfFile: number;
}

interface WGSLInsertionAnchors {
	afterEnable: number;
	afterAliases: number;
	afterStruct: number;
	afterBindings: number;
	beforeEntryPoint: number;
	endOfFile: number;
}

function countToken(line: string, token: string): number {
	let count = 0;
	for (let i = 0; i < line.length; i++) {
		if (line[i] === token) {
			count++;
		}
	}
	return count;
}

function countNewlinesUntilOffset(source: string, offset: number): number {
	if (source.length <= 0) {
		return 0;
	}
	const limit = Math.max(0, Math.min(source.length, Math.floor(offset)));
	let count = 0;
	for (let i = 0; i < limit; i++) {
		if (source.charCodeAt(i) === 10) {
			count++;
		}
	}
	return count;
}

function findLastStructEndLine(sourceLines: string[], pattern: RegExp): number {
	let lastStructEndLine = 0;
	let inStruct = false;
	let braceDepth = 0;
	for (let lineIndex = 0; lineIndex < sourceLines.length; lineIndex++) {
		const line = sourceLines[lineIndex];
		const lineNumber = lineIndex + 1;
		if (!inStruct && pattern.test(line)) {
			inStruct = true;
			braceDepth = countToken(line, "{") - countToken(line, "}");
			if (braceDepth <= 0 && /};?\s*$/.test(line)) {
				lastStructEndLine = lineNumber;
				inStruct = false;
				braceDepth = 0;
			}
			continue;
		}
		if (!inStruct) {
			continue;
		}
		braceDepth += countToken(line, "{");
		braceDepth -= countToken(line, "}");
		if (braceDepth <= 0 && /};?\s*$/.test(line)) {
			lastStructEndLine = lineNumber;
			inStruct = false;
			braceDepth = 0;
		}
	}
	return lastStructEndLine;
}

export function normalizeGLSLInjectionAnchor(
	anchor: ShaderInjectionAnchor | undefined
): ShaderGLSLInjectionAnchor {
	switch (anchor) {
		case "afterVersion":
		case "afterPrecision":
		case "afterDefines":
		case "afterStruct":
		case "afterUniforms":
		case "beforeEntryPoint":
		case "endOfFile":
			return anchor;
		default:
			return "afterVersion";
	}
}



export function normalizeWGSLInjectionAnchor(
	anchor: ShaderInjectionAnchor | undefined
): ShaderWGSLInjectionAnchor {
	switch (anchor) {
		case "afterEnable":
		case "afterAliases":
		case "afterStruct":
		case "afterBindings":
		case "beforeEntryPoint":
		case "endOfFile":
			return anchor;
		default:
			return "afterEnable";
	}
}

export function resolveGLSLInsertionAnchors(
	source: CompositeShaderSource
): GLSLInsertionAnchors {
	const sourceLines = source.code.split(/\r?\n/g);
	const lineCount = Math.max(1, sourceLines.length);
	let versionLine = 0;
	let lastPrecisionLine = 0;
	let entryPointLine = 0;
	let lastDefineLine = 0;
	let lastUniformLine = 0;
	for (let lineIndex = 0; lineIndex < sourceLines.length; lineIndex++) {
		const line = sourceLines[lineIndex];
		const lineNumber = lineIndex + 1;
		if (versionLine <= 0 && /^\s*#version\b/.test(line)) {
			versionLine = lineNumber;
			continue;
		}
		if (entryPointLine <= 0 && /\bvoid\s+main\s*\(/.test(line)) {
			entryPointLine = lineNumber;
		}
		if (/^\s*precision\b[^;]*;/.test(line)) {
			if (entryPointLine <= 0 || lineNumber < entryPointLine) {
				lastPrecisionLine = lineNumber;
			}
		}
		if (/^\s*#define\b/.test(line)) {
			if (entryPointLine <= 0 || lineNumber < entryPointLine) {
				lastDefineLine = lineNumber;
			}
		}
		if (/^\s*uniform\b/.test(line)) {
			if (entryPointLine <= 0 || lineNumber < entryPointLine) {
				lastUniformLine = lineNumber;
			}
		}
	}

	const lastStructEndLine = findLastStructEndLine(sourceLines, /^\s*struct\b/);
	const afterVersionLine = versionLine > 0 ? versionLine + 1 : 1;
	const afterPrecisionLine =
		lastPrecisionLine > 0 ? lastPrecisionLine + 1 : afterVersionLine;
	const afterDefinesLine =
		lastDefineLine > 0 ? Math.max(lastDefineLine + 1, afterPrecisionLine) : afterPrecisionLine;
	const afterStructLine =
		lastStructEndLine > 0 ? Math.max(lastStructEndLine + 1, afterDefinesLine) : afterDefinesLine;
	const afterUniformsLine =
		lastUniformLine > 0 ? Math.max(lastUniformLine + 1, afterStructLine) : afterStructLine;
	const beforeEntryPointLine = entryPointLine > 0 ? entryPointLine : lineCount + 1;
	return {
		afterVersion: clampInjectionLine(afterVersionLine, lineCount),
		afterPrecision: clampInjectionLine(afterPrecisionLine, lineCount),
		afterDefines: clampInjectionLine(afterDefinesLine, lineCount),
		afterStruct: clampInjectionLine(afterStructLine, lineCount),
		afterUniforms: clampInjectionLine(afterUniformsLine, lineCount),
		beforeEntryPoint: clampInjectionLine(beforeEntryPointLine, lineCount),
		endOfFile: lineCount + 1,
	};
}

export function resolveWGSLInsertionAnchors(
	source: CompositeShaderSource
): WGSLInsertionAnchors {
	const sourceLines = source.code.split(/\r?\n/g);
	const lineCount = Math.max(1, sourceLines.length);
	let lastEnableLine = 0;
	let lastAliasLine = 0;
	let lastBindingEndLine = 0;
	let entryPointLine = 0;
	for (let lineIndex = 0; lineIndex < sourceLines.length; lineIndex++) {
		const line = sourceLines[lineIndex];
		const lineNumber = lineIndex + 1;
		if (/^\s*enable\b/.test(line)) {
			lastEnableLine = lineNumber;
		}
		if (/^\s*alias\b/.test(line)) {
			lastAliasLine = lineNumber;
		}
		if (entryPointLine <= 0 && /@\s*(vertex|fragment|compute)\b/.test(line)) {
			entryPointLine = lineNumber;
		}
	}
	if (entryPointLine <= 0) {
		for (let lineIndex = 0; lineIndex < sourceLines.length; lineIndex++) {
			if (/\bfn\s+[A-Za-z_][A-Za-z0-9_]*\s*\(/.test(sourceLines[lineIndex])) {
				entryPointLine = lineIndex + 1;
				break;
			}
		}
	}
	const preEntryLines =
		entryPointLine > 0 ? sourceLines.slice(0, entryPointLine - 1) : sourceLines;
	const preEntrySource = preEntryLines.join("\n");
	const bindingDeclarationPattern =
		/(?:@\s*[A-Za-z_][A-Za-z0-9_]*(?:\s*\([^)]*\))?\s*)*var(?:<[^>]+>)?\s+[A-Za-z_][A-Za-z0-9_]*\s*:[^;]*;/gms;
	let bindingMatch: RegExpExecArray | null = bindingDeclarationPattern.exec(
		preEntrySource
	);
	while (bindingMatch) {
		const declaration = bindingMatch[0];
		if (
			/@group\s*\([^)]*\)/.test(declaration) &&
			/@binding\s*\([^)]*\)/.test(declaration)
		) {
			const declarationEnd = (bindingMatch.index ?? 0) + declaration.length;
			lastBindingEndLine = Math.max(
				lastBindingEndLine,
				1 + countNewlinesUntilOffset(preEntrySource, declarationEnd)
			);
		}
		bindingMatch = bindingDeclarationPattern.exec(preEntrySource);
	}
	const lastStructEndLine = findLastStructEndLine(
		preEntryLines,
		/^\s*struct\b/
	);
	const afterEnableLine = lastEnableLine > 0 ? lastEnableLine + 1 : 1;
	const afterAliasesLine =
		lastAliasLine > 0 ? Math.max(lastAliasLine + 1, afterEnableLine) : afterEnableLine;
	const afterStructLine =
		lastStructEndLine > 0 ? Math.max(lastStructEndLine + 1, afterAliasesLine) : afterAliasesLine;
	const afterBindingsLine =
		lastBindingEndLine > 0 ?
			Math.max(lastBindingEndLine + 1, afterStructLine)
		:	afterStructLine;
	const beforeEntryPointLine = entryPointLine > 0 ? entryPointLine : lineCount + 1;
	return {
		afterEnable: clampInjectionLine(afterEnableLine, lineCount),
		afterAliases: clampInjectionLine(afterAliasesLine, lineCount),
		afterStruct: clampInjectionLine(afterStructLine, lineCount),
		afterBindings: clampInjectionLine(afterBindingsLine, lineCount),
		beforeEntryPoint: clampInjectionLine(beforeEntryPointLine, lineCount),
		endOfFile: lineCount + 1,
	};
}

function clampInjectionLine(line: number, lineCount: number): number {
	const normalized = Number.isFinite(line) ? Math.floor(line) : 1;
	return Math.min(Math.max(normalized, 1), lineCount + 1);
}

function resolveGLSLInsertionLine(
	anchor: ShaderGLSLInjectionAnchor,
	anchors: GLSLInsertionAnchors
): number {
	switch (anchor) {
		case "afterPrecision":
			return anchors.afterPrecision;
		case "afterDefines":
			return anchors.afterDefines;
		case "afterStruct":
			return anchors.afterStruct;
		case "afterUniforms":
			return anchors.afterUniforms;
		case "beforeEntryPoint":
			return anchors.beforeEntryPoint;
		case "endOfFile":
			return anchors.endOfFile;
		case "afterVersion":
		default:
			return anchors.afterVersion;
	}
}

function resolveWGSLInsertionLine(
	anchor: ShaderWGSLInjectionAnchor,
	anchors: WGSLInsertionAnchors
): number {
	switch (anchor) {
		case "afterAliases":
			return anchors.afterAliases;
		case "afterStruct":
			return anchors.afterStruct;
		case "afterBindings":
			return anchors.afterBindings;
		case "beforeEntryPoint":
			return anchors.beforeEntryPoint;
		case "endOfFile":
			return anchors.endOfFile;
		case "afterEnable":
		default:
			return anchors.afterEnable;
	}
}



function injectBlocksAtLines(
	source: CompositeShaderSource,
	insertions: Map<number, InjectionBlock[]>
): CompositeShaderSource {
	if (insertions.size <= 0) {
		return source;
	}
	const sourceLines = source.code.split(/\r?\n/g);
	const lineCount = Math.max(1, sourceLines.length);
	const sourcePath = source.sourceMap.segments[0]?.sourcePath ?? "<shader>";
	const insertionLines = [...insertions.keys()].sort((left, right) => left - right);
	const parts: {
		code: string;
		sourceMap: ShaderSourceSegmentMap;
		sourcePath: string;
		kind: "source" | "define-block";
	}[] = [];
	let cursorLine = 1;
	for (const insertionLine of insertionLines) {
		if (insertionLine > cursorLine) {
			const sourceSlice = sliceCompositeShaderSource(
				source,
				cursorLine,
				insertionLine - 1
			);
			parts.push({
				code: sourceSlice.code,
				sourceMap: sourceSlice.sourceMap,
				sourcePath: sourceSlice.sourceMap.segments[0]?.sourcePath ?? sourcePath,
				kind: "source",
			});
		}
		const bucket = insertions.get(insertionLine) ?? [];
		const injection = composeCompositeShaderSources(
			bucket.map((block) => ({
				code: block.code,
				sourcePath: block.sourcePath,
				kind: "define-block" as const,
				label: block.label,
			})),
			"\n\n"
		);
		parts.push({
			code: injection.code,
			sourceMap: injection.sourceMap,
			sourcePath: "<runtime:injection>",
			kind: "define-block",
		});
		cursorLine = insertionLine;
	}
	if (cursorLine <= lineCount) {
		const sourceSlice = sliceCompositeShaderSource(source, cursorLine, lineCount);
		parts.push({
			code: sourceSlice.code,
			sourceMap: sourceSlice.sourceMap,
			sourcePath: sourceSlice.sourceMap.segments[0]?.sourcePath ?? sourcePath,
			kind: "source",
		});
	}
	return composeCompositeShaderSources(parts, "\n");
}

export function injectGLSLSource(
	source: CompositeShaderSource,
	blocks: InjectionBlock[]
): CompositeShaderSource {
	if (blocks.length <= 0) {
		return source;
	}
	const sourceLines = source.code.split(/\r?\n/g);
	const lineCount = Math.max(1, sourceLines.length);
	const anchors = resolveGLSLInsertionAnchors(source);
	const insertions = new Map<number, InjectionBlock[]>();
	for (const block of blocks) {
		const anchor = normalizeGLSLInjectionAnchor(block.anchor);
		const insertionLine = clampInjectionLine(
			resolveGLSLInsertionLine(anchor, anchors),
			lineCount
		);
		const bucket = insertions.get(insertionLine);
		if (bucket) {
			bucket.push(block);
			continue;
		}
		insertions.set(insertionLine, [block]);
	}
	return injectBlocksAtLines(source, insertions);
}

export function injectWGSLSource(
	source: CompositeShaderSource,
	blocks: InjectionBlock[]
): CompositeShaderSource {
	if (blocks.length <= 0) {
		return source;
	}
	const sourceLines = source.code.split(/\r?\n/g);
	const lineCount = Math.max(1, sourceLines.length);
	const anchors = resolveWGSLInsertionAnchors(source);
	const insertions = new Map<number, InjectionBlock[]>();
	for (const block of blocks) {
		const anchor = normalizeWGSLInjectionAnchor(block.anchor);
		const insertionLine = clampInjectionLine(
			resolveWGSLInsertionLine(anchor, anchors),
			lineCount
		);
		const bucket = insertions.get(insertionLine);
		if (bucket) {
			bucket.push(block);
			continue;
		}
		insertions.set(insertionLine, [block]);
	}
	return injectBlocksAtLines(source, insertions);
}

export function normalizeInjectionAnchorForLanguage(
	language: "wgsl" | "glsl",
	anchor: ShaderInjectionAnchor | undefined
): ShaderInjectionAnchor {
	return language === "wgsl" ?
		normalizeWGSLInjectionAnchor(anchor)
	:	normalizeGLSLInjectionAnchor(anchor);
}
