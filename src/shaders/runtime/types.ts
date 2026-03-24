export type ShaderRuntimeMode = "strict" | "warn";

export type ShaderLanguage = "glsl" | "wgsl";
export type ShaderStage = "vertex" | "fragment" | "compute" | "unknown";

export type ShaderSourceKind =
	| "builtin-scene"
	| "builtin-skybox"
	| "builtin-present"
	| "postprocess"
	| "shadow"
	| "particle"
	| "custom-material"
	| "unknown";

export type ShaderDiagnosticSeverity = "warning" | "error";

export interface ShaderDiagnostic {
	ruleId: string;
	code: string;
	severity: ShaderDiagnosticSeverity;
	message: string;
}

export interface ShaderProcessRequest {
	code: string;
	language: ShaderLanguage;
	stage?: ShaderStage;
	entryPoint?: string;
	label?: string;
	sourceKind?: ShaderSourceKind;
}

export interface ShaderProcessResult {
	code: string;
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
}

export interface ShaderRule {
	id: string;
	priority?: number;
	symbols?: string[];
	match?: (context: ShaderRuleContext) => boolean;
	validate?: (context: ShaderRuleContext) => ShaderDiagnostic[];
	inject?: (context: ShaderRuleContext) => ShaderRuleInjection | null;
}
