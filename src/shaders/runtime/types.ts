export type ShaderRuntimeMode = "strict" | "warn" | "silent";

export type ShaderLanguage = "glsl" | "wgsl";
export type ShaderStage = "vertex" | "fragment" | "compute" | "unknown";

export type ShaderSourceKind =
	| "builtin-scene"
	| "builtin-environment"
	| "builtin-present"
	| "postprocess"
	| "clustered"
	| "shadow"
	| "decal"
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
	generatedColumnStart?: number;
	generatedColumnEnd?: number;
	sourcePath: string;
	sourceLineStart: number;
	sourceLineEnd: number;
	sourceColumnStart?: number;
	sourceColumnEnd?: number;
	kind: ShaderSourceSegmentKind;
	label?: string;
}

export interface ShaderSourceSegmentMap {
	schemaVersion?: number;
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
	enableDirectives?: boolean;
	directiveSourcePath?: string;
}

export interface ShaderProcessResult {
	code: string;
	sourceMap: ShaderSourceSegmentMap;
	composite: CompositeShaderSource;
	diagnostics: ShaderDiagnostic[];
	hasErrors: boolean;
	fromCache: boolean;
}

export interface ShaderDirectivePreprocessResult {
	code: string;
	sourceMap: ShaderSourceSegmentMap;
	composite: CompositeShaderSource;
	diagnostics: ShaderDiagnostic[];
	hasErrors: boolean;
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

export type ShaderInjectionArgValue = string | number | boolean;

export interface ShaderIncludeModule {
	language: ShaderLanguage;
	id: string;
	code: string;
	sourcePath?: string;
}

export interface ShaderInjectionScriptContext extends ShaderRuleContext {}

export interface ShaderInjectionScript {
	id: string;
	language?: ShaderLanguage;
	description?: string;
	symbols?: string[];
	run: (
		args: Readonly<Record<string, ShaderInjectionArgValue>>,
		context: ShaderInjectionScriptContext
	) => ShaderRuleInjectResult;
}

export type ShaderBackendId = "webgpu" | "webgl" | "software";

export interface ShaderDirectiveProfile {
	id: string;
	backend: ShaderBackendId;
	revision: number;
	includeModules: ShaderIncludeModule[];
	injectionScripts: ShaderInjectionScript[];
}

export type ShaderDirectiveProfileRegistry = Record<
	ShaderBackendId,
	ShaderDirectiveProfile
>;

export interface ShaderDirectiveHookContext {
	backend: ShaderBackendId;
	language: ShaderLanguage;
	stage: ShaderStage;
	sourceKind: ShaderSourceKind;
	label: string | null;
	directiveSourcePath: string;
}

export interface ShaderDirectiveHookResult {
	token: string;
	includeModules?: ShaderIncludeModule[];
	injectionScripts?: ShaderInjectionScript[];
}

export type ShaderDirectiveCompileHook = (
	context: Readonly<ShaderDirectiveHookContext>
) =>
	| ShaderDirectiveHookResult
	| null
	| Promise<ShaderDirectiveHookResult | null>;

export interface ShaderDirectiveStageRequest {
	code: string;
	sourceMap?: ShaderSourceSegmentMap | null;
	language: ShaderLanguage;
	stage?: ShaderStage;
	entryPoint?: string;
	label?: string;
	sourceKind?: ShaderSourceKind;
	directiveSourcePath?: string;
}

export interface ShaderDirectiveStageResult {
	code: string;
	sourceMap: ShaderSourceSegmentMap;
	composite: CompositeShaderSource;
	diagnostics: ShaderDiagnostic[];
	hasErrors: boolean;
	directiveFingerprint: string;
}

export interface ShaderBackendCompileResult extends ShaderProcessResult {
	directiveFingerprint: string;
	directiveDiagnostics: ShaderDiagnostic[];
	backendDiagnostics: ShaderDiagnostic[];
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

export interface ShaderRuleTransformOutput {
	code: string;
	sourceMap?: ShaderSourceSegmentMap | null;
	diagnostics?: ShaderDiagnostic[];
}

export type ShaderRuleTransformResolved =
	| string
	| ShaderRuleTransformOutput
	| null
	| undefined;

export type ShaderRuleTransformResult =
	| ShaderRuleTransformResolved
	| Promise<ShaderRuleTransformResolved>;

export interface ShaderRuleReplacePatch {
	pattern: string | RegExp;
	replacement: string;
	replaceAll?: boolean;
}

export interface ShaderRuleReplaceOutput {
	patches: ShaderRuleReplacePatch[];
	diagnostics?: ShaderDiagnostic[];
}

export type ShaderRuleReplaceResolved =
	| ShaderRuleReplacePatch[]
	| ShaderRuleReplaceOutput
	| null
	| undefined;

export type ShaderRuleReplaceResult =
	| ShaderRuleReplaceResolved
	| Promise<ShaderRuleReplaceResolved>;

export interface ShaderRule {
	id: string;
	description?: string;
	priority?: number;
	symbols?: string[];
	dependsOn?: string[];
	match?: (context: ShaderRuleContext) => ShaderRuleMatchResult;
	transform?: (context: ShaderRuleContext) => ShaderRuleTransformResult;
	replace?: (context: ShaderRuleContext) => ShaderRuleReplaceResult;
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
	| "register-include-module"
	| "update-include-module"
	| "unregister-include-module"
	| "clear-include-modules"
	| "register-injection-script"
	| "update-injection-script"
	| "unregister-injection-script"
	| "clear-injection-scripts"
	| "invalidate-cache";

export interface ShaderRuntimeChangeEvent {
	revision: number;
	action: ShaderRuntimeChangeAction;
	ruleIds: string[];
	includeModuleIds?: string[];
	injectionScriptIds?: string[];
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
