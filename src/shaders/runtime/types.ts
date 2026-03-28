export type ShaderRuntimeMode = "strict" | "warn" | "silent";

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
	sourceHash?: string;
	diagnosticFilter?: ShaderDiagnosticFilter;
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

export type ShaderDiagnosticFilter = (diagnostic: ShaderDiagnostic) => boolean;

export interface ShaderRuleInjection {
	header?: string;
	functions?: string;
	symbols?: string[];
	headerAnchor?: ShaderInjectionAnchor;
	functionsAnchor?: ShaderInjectionAnchor;
}

export type ShaderGLSLInjectionAnchor =
	| "afterVersion"
	| "afterPrecision"
	| "afterDefines"
	| "afterStruct"
	| "afterUniforms"
	| "beforeEntryPoint"
	| "endOfFile";

export type ShaderWGSLInjectionAnchor =
	| "afterEnable"
	| "afterAliases"
	| "afterStruct"
	| "afterBindings"
	| "beforeEntryPoint"
	| "endOfFile";

export type ShaderInjectionAnchor =
	| ShaderGLSLInjectionAnchor
	| ShaderWGSLInjectionAnchor;

export type ShaderRuleMatchResult = boolean | Promise<boolean>;
export type ShaderRuleValidateResult =
	| ShaderDiagnostic[]
	| null
	| undefined
	| Promise<ShaderDiagnostic[] | null | undefined>;
export type ShaderRuleInjectResult =
	| ShaderRuleInjection
	| null
	| undefined
	| Promise<ShaderRuleInjection | null | undefined>;

export interface ShaderRule {
	id: string;
	description?: string;
	priority?: number;
	symbols?: string[];
	dependsOn?: string[];
	match?: (context: ShaderRuleContext) => ShaderRuleMatchResult;
	validate?: (context: ShaderRuleContext) => ShaderRuleValidateResult;
	inject?: (context: ShaderRuleContext) => ShaderRuleInjectResult;
}

export type ShaderRuntimeCacheKind = "sync" | "async";

export interface ShaderRuntimeCacheStats {
	hits: number;
	misses: number;
	evictions: number;
	invalidations: number;
	size: number;
	limit: number;
}

export interface ShaderRuntimeCacheStatsSnapshot {
	sync: ShaderRuntimeCacheStats;
	async: ShaderRuntimeCacheStats;
}

export type ShaderRuntimeChangeAction =
	| "mode"
	| "register-rule"
	| "update-rule"
	| "unregister-rule"
	| "clear-rules"
	| "invalidate-cache";

export interface ShaderRuntimeChangeEvent {
	revision: number;
	action: ShaderRuntimeChangeAction;
	ruleIds: string[];
}

export interface ShaderResolvedGLSLInjectionAnchors {
	language: "glsl";
	lineCount: number;
	anchors: Record<ShaderGLSLInjectionAnchor, number>;
}

export interface ShaderResolvedWGSLInjectionAnchors {
	language: "wgsl";
	lineCount: number;
	anchors: Record<ShaderWGSLInjectionAnchor, number>;
}

export type ShaderResolvedInjectionAnchors =
	| ShaderResolvedGLSLInjectionAnchors
	| ShaderResolvedWGSLInjectionAnchors;
