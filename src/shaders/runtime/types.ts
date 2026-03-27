export type ShaderRuntimeMode = "strict" | "warn";

export type ShaderLanguage = "glsl" | "wgsl";
export type ShaderStage = "vertex" | "fragment" | "compute" | "unknown";

export type ShaderSourceKind =
	| "builtin-scene"
	| "builtin-skybox"
	| "builtin-present"
	| "postprocess"
	| "clustered"
	| "shadow"
	| "particle"
	| "custom-material"
	| "unknown";

export type ShaderDiagnosticSeverity = "warning" | "error";

export interface ShaderDiagnosticPosition {
	line: number;
	column: number;
}

export interface ShaderDiagnosticRange {
	start: ShaderDiagnosticPosition;
	end: ShaderDiagnosticPosition;
}

export interface ShaderDiagnostic {
	ruleId: string;
	code: string;
	severity: ShaderDiagnosticSeverity;
	message: string;
	line?: number;
	column?: number;
	sourcePath?: string;
	range?: ShaderDiagnosticRange;
}

export type ShaderSourceSegmentKind =
	| "source"
	| "template"
	| "include"
	| "define-block"
	| "generated";

export interface ShaderSourceSegment {
	generatedLineStart: number;
	generatedLineEnd: number;
	sourcePath: string;
	sourceLineStart: number;
	sourceLineEnd: number;
	kind: ShaderSourceSegmentKind;
	label?: string;
}

export interface ShaderSourceSegmentMap {
	lineCount: number;
	segments: ShaderSourceSegment[];
}

export interface CompositeShaderSource {
	code: string;
	sourceMap: ShaderSourceSegmentMap;
}

export interface ShaderProcessRequest {
	code: string;
	language: ShaderLanguage;
	stage?: ShaderStage;
	entryPoint?: string;
	label?: string;
	sourceKind?: ShaderSourceKind;
	sourceMap?: ShaderSourceSegmentMap | null;
}

export interface ShaderProcessResult {
	code: string;
	sourceMap: ShaderSourceSegmentMap;
	composite: CompositeShaderSource;
	diagnostics: ShaderDiagnostic[];
	hasErrors: boolean;
	fromCache: boolean;
}

export interface ShaderRuleContext {
	mode: ShaderRuntimeMode;
	language: ShaderLanguage;
	stage: ShaderStage;
	entryPoint: string | null;
	label: string | null;
	sourceKind: ShaderSourceKind;
	source: string;
}

export interface ShaderRuleInjection {
	header?: string;
	functions?: string;
	symbols?: string[];
	headerAnchor?: ShaderGLSLInjectionAnchor;
	functionsAnchor?: ShaderGLSLInjectionAnchor;
}

export type ShaderGLSLInjectionAnchor =
	| "afterVersion"
	| "afterPrecision"
	| "beforeEntryPoint"
	| "endOfFile";

export interface ShaderRule {
	id: string;
	priority?: number;
	symbols?: string[];
	match?: (context: ShaderRuleContext) => boolean;
	validate?: (context: ShaderRuleContext) => ShaderDiagnostic[];
	inject?: (context: ShaderRuleContext) => ShaderRuleInjection | null;
}
