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

/** Describes one validated `#inject` argument. */
export type ShaderInjectionArgumentDefinition =
	| {
			type: "string";
			required?: boolean;
			default?: string;
	  }
	| {
			type: "number" | "integer";
			required?: boolean;
			default?: number;
			min?: number;
			max?: number;
	  }
	| {
			type: "boolean";
			required?: boolean;
			default?: boolean;
	  }
	| {
			type: "enum";
			required?: boolean;
			default?: string;
			values: readonly string[];
	  };

/** Declares the accepted arguments for one injection script. */
export type ShaderInjectionArgumentSchema = Readonly<
	Record<string, ShaderInjectionArgumentDefinition>
>;

type ShaderInjectionArgumentValueForDefinition<
	Definition extends ShaderInjectionArgumentDefinition,
> = Definition["type"] extends "number" | "integer" ? number
	: Definition["type"] extends "boolean" ? boolean
	: string;

/** Resolves schema definitions to the values received by an injection script. */
export type ShaderInjectionArguments<
	Schema extends ShaderInjectionArgumentSchema,
> = {
	readonly [Key in keyof Schema]:
		| ShaderInjectionArgumentValueForDefinition<Schema[Key]>
		| undefined;
};

export interface ShaderIncludeModule {
	language: ShaderLanguage;
	id: string;
	code: string;
	sourcePath?: string;
}

export interface ShaderInjectionScriptContext extends ShaderRuleContext {}

export interface ShaderInjectionScript<
	Schema extends ShaderInjectionArgumentSchema = ShaderInjectionArgumentSchema,
> {
	id: string;
	language?: ShaderLanguage;
	description?: string;
	symbols?: readonly string[];
	arguments: Schema;
	validateArguments?: (
		args: ShaderInjectionArguments<Schema>,
		context: ShaderInjectionScriptContext,
	) => string | readonly string[] | null | undefined;
	run: (
		args: ShaderInjectionArguments<Schema>,
		context: ShaderInjectionScriptContext
	) => ShaderRuleInjectResult;
}

export type ShaderBackendId = "webgpu" | "webgl" | "software";

/** @internal Backend-owned shader directive compilation environment. */
export interface ShaderDirectiveProfile {
	readonly id: string;
	readonly backend: ShaderBackendId;
	readonly fingerprint: string;
	readonly includeModules: readonly ShaderIncludeModule[];
	readonly injectionScripts: readonly ShaderInjectionScript[];
}

/** @internal Backend-applicable directive feature composition unit. */
export interface ShaderDirectiveFeaturePack {
	readonly id: string;
	readonly backend: ShaderBackendId;
	readonly revision: number;
	readonly includeModules: readonly ShaderIncludeModule[];
	readonly injectionScripts: readonly ShaderInjectionScript[];
}

/** @internal Prepared device-independent directive profile base. */
export interface ShaderDirectiveProfileBase {
	readonly id: string;
	readonly backend: ShaderBackendId;
	readonly packs: readonly ShaderDirectiveFeaturePack[];
}

/** @internal Backend-instance directive profile overlay. */
export interface ShaderDirectiveProfileOverlay {
	readonly id: string;
	readonly backend: ShaderBackendId;
	readonly includeModules: readonly ShaderIncludeModule[];
}

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
