import { ConditionParseError } from "../../foundation/Error";
import { DirectiveConditionParser } from "./DirectiveConditionParser";
import {
	compressLineOriginsToSourceMap,
	createInlineCompositeShaderSource,
	expandSourceMapToLineOrigins,
} from "./sourceMap";
import {
	injectGLSLSource,
	injectWGSLSource,
	normalizeInjectionAnchorForLanguage,
	resolveGLSLInsertionAnchors,
	resolveWGSLInsertionAnchors,
	type InjectionBlock,
} from "./ShaderSourceInjection";
import {
	createPointRange,
	isIdentifierPartCharacter,
	isIdentifierStartCharacter,
	isWhitespaceCharacter,
	isPromiseLike,
	normalizeInjectionBlock,
	normalizeLanguage,
	normalizeSourceKind,
	normalizeStage,
} from "./runtimeShared";
import type {
	CompositeShaderSource,
	ShaderDiagnostic,
	ShaderDiagnosticSeverity,
	ShaderInjectionArgValue,
	ShaderInjectionArgumentDefinition,
	ShaderInjectionArguments,
	ShaderInjectionArgumentSchema,
	ShaderInjectionScript,
	ShaderInjectionScriptContext,
	ShaderLanguage,
	ShaderProcessRequest,
	ShaderResolvedInjectionAnchors,
	ShaderRuleContext,
	ShaderRuleInjection,
	ShaderRuntimeMode,
} from "./types";

interface LineOrigin {
	sourcePath: string;
	sourceLine: number;
	kind: "source" | "template" | "include" | "define-block" | "generated";
	label?: string;
}

interface PreprocessContext {
	request: ShaderProcessRequest;
	contextTemplate: Omit<ShaderRuleContext, "source">;
	mode: ShaderRuntimeMode;
	language: ShaderLanguage;
	sourcePath: string;
	diagnostics: ShaderDiagnostic[];
	macros: Map<string, MacroDefinition>;
	expandedModules: Set<string>;
	processingStack: string[];
}

interface DirectiveLineScanState {
	inBlockComment: boolean;
	stringQuote: '"' | "'" | null;
	escape: boolean;
}

interface DirectiveLine {
	name: string;
	body: string;
	column: number;
	raw: string;
}

type IncludeSpecifierKind = "angle" | "quote";

interface IncludeSpecifier {
	kind: IncludeSpecifierKind;
	path: string;
}

interface InjectInvocation {
	id: string;
	args: Record<string, ShaderInjectionArgValue>;
}

type MacroDefinitionKind = "object" | "function";

interface BaseMacroDefinition {
	kind: MacroDefinitionKind;
	name: string;
	replacement: string;
	sourcePath: string;
	sourceLine: number;
}

interface ObjectMacroDefinition extends BaseMacroDefinition {
	kind: "object";
}

interface FunctionMacroDefinition extends BaseMacroDefinition {
	kind: "function";
	params: string[];
}

type MacroDefinition = ObjectMacroDefinition | FunctionMacroDefinition;

export interface RegisteredIncludeModule {
	id: string;
	canonicalId: string;
	code: string;
	sourcePath: string;
}

interface ConditionalBranchState {
	parentActive: boolean;
	branchTaken: boolean;
	currentActive: boolean;
	elseSeen: boolean;
	sourcePath: string;
	sourceLine: number;
	column: number;
}

interface DirectivePreprocessorOptions {
	mode: ShaderRuntimeMode;
	includeModulesByLanguage: Map<ShaderLanguage, Map<string, RegisteredIncludeModule>>;
	injectionScripts: Map<string, ShaderInjectionScript>;
}

const DIRECTIVE_MAX_MACRO_EXPANSION_DEPTH = 32;
const DIRECTIVE_CONDITIONAL_NAMES = new Set([
	"if",
	"ifdef",
	"ifndef",
	"elif",
	"else",
	"endif",
]);

export interface PreprocessResult {
	composite: CompositeShaderSource;
	diagnostics: ShaderDiagnostic[];
}

export class DirectivePreprocessor {
	private _mode: ShaderRuntimeMode;
	private _includeModulesByLanguage: Map<
		ShaderLanguage,
		Map<string, RegisteredIncludeModule>
	>;
	private _injectionScripts: Map<string, ShaderInjectionScript>;

	public constructor(options: DirectivePreprocessorOptions) {
		this._mode = options.mode;
		this._includeModulesByLanguage = options.includeModulesByLanguage;
		this._injectionScripts = options.injectionScripts;
	}

	public setMode(mode: ShaderRuntimeMode): void {
		this._mode = mode;
	}

	public resolveInjectionAnchors(
		request: ShaderProcessRequest
	): ShaderResolvedInjectionAnchors {
		const sourcePath = this.resolveRequestSourcePath(request);
		const initialComposite =
			request.sourceMap ?
				{
					code: request.code,
					sourceMap: request.sourceMap,
				}
			:	createInlineCompositeShaderSource(request.code, sourcePath, "source");
		const baseComposite =
			request.enableDirectives === false ?
				initialComposite
			:	this.preprocessSync(request, initialComposite).composite;
		if (normalizeLanguage(request.language) === "glsl") {
			const anchors = resolveGLSLInsertionAnchors(baseComposite);
			return {
				language: "glsl",
				lineCount: Math.max(1, baseComposite.code.split(/\r?\n/g).length),
				anchors,
			};
		}
		const anchors = resolveWGSLInsertionAnchors(baseComposite);
		return {
			language: "wgsl",
			lineCount: Math.max(1, baseComposite.code.split(/\r?\n/g).length),
			anchors,
		};
	}

	public resolveRequestSourcePath(request: ShaderProcessRequest): string {
		const explicitDirectivePath =
			typeof request.directiveSourcePath === "string" ?
				request.directiveSourcePath.trim()
			:	"";
		if (explicitDirectivePath.length > 0) {
			return explicitDirectivePath;
		}
		const sourceMapPath = request.sourceMap?.segments?.[0]?.sourcePath;
		if (typeof sourceMapPath === "string" && sourceMapPath.length > 0) {
			return sourceMapPath;
		}
		const normalizedLanguage = normalizeLanguage(request.language);
		return (
			request.label ??
			`<runtime:${normalizedLanguage}:${normalizeStage(request.stage)}:${normalizeSourceKind(
				request.sourceKind
			)}>`
		);
	}

	private _createPreprocessContext(
		request: ShaderProcessRequest,
		sourcePath: string
	): PreprocessContext {
		const language = normalizeLanguage(request.language);
		return {
			request,
			mode: this._mode,
			language,
			sourcePath,
			contextTemplate: {
				mode: this._mode,
				language,
				stage: normalizeStage(request.stage),
				entryPoint: request.entryPoint ?? null,
				label: request.label ?? null,
				sourceKind: normalizeSourceKind(request.sourceKind),
			},
			diagnostics: [],
			macros: new Map(),
			expandedModules: new Set(),
			processingStack: [],
		};
	}

	public preprocessSync(
		request: ShaderProcessRequest,
		initialComposite: CompositeShaderSource
	): PreprocessResult {
		if (request.enableDirectives === false) {
			return {
				composite: initialComposite,
				diagnostics: [],
			};
		}
		const sourcePath =
			initialComposite.sourceMap.segments[0]?.sourcePath ??
			this.resolveRequestSourcePath(request);
		const preprocessContext = this._createPreprocessContext(request, sourcePath);
		const expanded = this._expandDirectiveComposite(
			initialComposite,
			this._canonicalizeModulePathSafe(sourcePath),
			preprocessContext
		);
		const macroExpanded = this._expandMacrosInComposite(expanded, preprocessContext);
		const injected = this._resolveDirectiveInjectsSync(
			macroExpanded,
			preprocessContext
		);
		return {
			composite: injected,
			diagnostics: [...preprocessContext.diagnostics],
		};
	}

	public async preprocessAsync(
		request: ShaderProcessRequest,
		initialComposite: CompositeShaderSource
	): Promise<PreprocessResult> {
		if (request.enableDirectives === false) {
			return {
				composite: initialComposite,
				diagnostics: [],
			};
		}
		const sourcePath =
			initialComposite.sourceMap.segments[0]?.sourcePath ??
			this.resolveRequestSourcePath(request);
		const preprocessContext = this._createPreprocessContext(request, sourcePath);
		const expanded = this._expandDirectiveComposite(
			initialComposite,
			this._canonicalizeModulePathSafe(sourcePath),
			preprocessContext
		);
		const macroExpanded = this._expandMacrosInComposite(expanded, preprocessContext);
		const injected = await this._resolveDirectiveInjectsAsync(
			macroExpanded,
			preprocessContext
		);
		return {
			composite: injected,
			diagnostics: [...preprocessContext.diagnostics],
		};
	}

	private _splitCompositeLines(composite: CompositeShaderSource): {
		lines: string[];
		origins: LineOrigin[];
	} {
		const lines = composite.code.split(/\r?\n/g);
		const fallbackPath = composite.sourceMap.segments[0]?.sourcePath ?? "<generated>";
		const origins = expandSourceMapToLineOrigins(
			composite.sourceMap,
			lines.length,
			fallbackPath,
			"source"
		) as LineOrigin[];
		return { lines, origins };
	}

	private _composeLinesToComposite(
		lines: string[],
		origins: LineOrigin[]
	): CompositeShaderSource {
		const effectiveLines = lines.length > 0 ? lines : [""];
		const effectiveOrigins =
			origins.length > 0 ?
				origins
			:	[
					{
						sourcePath: "<generated>",
						sourceLine: 1,
						kind: "generated" as const,
					},
				];
		return {
			code: effectiveLines.join("\n"),
			sourceMap: compressLineOriginsToSourceMap(
				effectiveOrigins as unknown as Array<{
					sourcePath: string;
					sourceLine: number;
					kind: "source" | "template" | "include" | "define-block" | "generated";
					label?: string;
				}>
			),
		};
	}

	private _scanDirectiveFromLine(
		line: string,
		state: DirectiveLineScanState
	): DirectiveLine | null {
		let index = 0;
		while (index < line.length && (line[index] === " " || line[index] === "\t")) {
			index++;
		}
		if (index >= line.length || line[index] !== "#") {
			this._updateDirectiveStateFromLine(line, state);
			return null;
		}
		if (state.inBlockComment || state.stringQuote) {
			this._updateDirectiveStateFromLine(line, state);
			return null;
		}
		const raw = line.slice(index + 1).trim();
		const match = /^([A-Za-z_][A-Za-z0-9_]*)(?:\s+(.*))?$/.exec(raw);
		this._updateDirectiveStateFromLine(line, state);
		if (!match) {
			return null;
		}
		return {
			name: match[1].toLowerCase(),
			body: (match[2] ?? "").trim(),
			column: index + 1,
			raw: line.trim(),
		};
	}

	private _updateDirectiveStateFromLine(
		line: string,
		state: DirectiveLineScanState
	): void {
		for (let i = 0; i < line.length; i++) {
			const char = line[i];
			const next = i + 1 < line.length ? line[i + 1] : "";
			if (state.inBlockComment) {
				if (char === "*" && next === "/") {
					state.inBlockComment = false;
					i++;
				}
				continue;
			}
			if (state.stringQuote) {
				if (state.escape) {
					state.escape = false;
					continue;
				}
				if (char === "\\") {
					state.escape = true;
					continue;
				}
				if (char === state.stringQuote) {
					state.stringQuote = null;
					continue;
				}
				continue;
			}
			if (char === "/" && next === "/") {
				break;
			}
			if (char === "/" && next === "*") {
				state.inBlockComment = true;
				i++;
				continue;
			}
			if (char === "\"" || char === "'") {
				state.stringQuote = char as '"' | "'";
				state.escape = false;
			}
		}
		if (state.stringQuote && !state.inBlockComment) {
			state.escape = false;
		}
	}

	private _expandDirectiveComposite(
		composite: CompositeShaderSource,
		modulePath: string,
		preprocessContext: PreprocessContext
	): CompositeShaderSource {
		const { lines, origins } = this._splitCompositeLines(composite);
		const outputLines: string[] = [];
		const outputOrigins: LineOrigin[] = [];
		const directiveState: DirectiveLineScanState = {
			inBlockComment: false,
			stringQuote: null,
			escape: false,
		};
		const conditionalStack: ConditionalBranchState[] = [];
		const firstVersionLine = this._findFirstGLSLVersionLine(lines);
		const isBranchActive = (): boolean =>
			conditionalStack.length <= 0 ?
				true
			:	conditionalStack[conditionalStack.length - 1].currentActive;

		for (let index = 0; index < lines.length; index++) {
			const lineNumber = index + 1;
			const line = lines[index];
			const origin = origins[index] ?? {
				sourcePath: modulePath,
				sourceLine: lineNumber,
				kind: "source",
			};
			const directive = this._scanDirectiveFromLine(line, directiveState);
			if (!directive) {
				if (isBranchActive()) {
					outputLines.push(line);
					outputOrigins.push(origin);
				}
				continue;
			}
			if (DIRECTIVE_CONDITIONAL_NAMES.has(directive.name)) {
				this._applyConditionalDirective(
					directive,
					origin,
					preprocessContext,
					conditionalStack
				);
				continue;
			}
			if (!isBranchActive()) {
				continue;
			}
			if (
				preprocessContext.language === "glsl" &&
				firstVersionLine > 0 &&
				lineNumber < firstVersionLine &&
				(directive.name === "include" || directive.name === "import")
			) {
				this._pushDirectiveDiagnostic(
					preprocessContext,
					"directive-include-before-version",
					`Directive "#${directive.name}" appears before "#version" and was skipped.`,
					origin.sourcePath,
					origin.sourceLine,
					directive.column
				);
				continue;
			}
			switch (directive.name) {
				case "include":
				case "import": {
					const specifier = this._parseIncludeSpecifier(
						directive,
						origin,
						preprocessContext
					);
					if (!specifier) {
						continue;
					}
					if (directive.name === "import" && specifier.kind !== "angle") {
						this._pushDirectiveDiagnostic(
							preprocessContext,
							"directive-import-invalid-path",
							`Directive "#import" only supports angle-bracket paths.`,
							origin.sourcePath,
							origin.sourceLine,
							directive.column
						);
						continue;
					}
					const includeComposite = this._resolveIncludeComposite(
						specifier,
						modulePath,
						preprocessContext,
						origin
					);
					if (!includeComposite) {
						continue;
					}
					const includeLines = includeComposite.code.split(/\r?\n/g);
					const includeOrigins = expandSourceMapToLineOrigins(
						includeComposite.sourceMap,
						includeLines.length,
						includeComposite.sourceMap.segments[0]?.sourcePath ?? modulePath,
						"include"
					) as LineOrigin[];
					for (let includeIndex = 0; includeIndex < includeLines.length; includeIndex++) {
						outputLines.push(includeLines[includeIndex]);
						outputOrigins.push(
							includeOrigins[includeIndex] ?? {
								sourcePath: modulePath,
								sourceLine: lineNumber,
								kind: "include",
							}
						);
					}
					continue;
				}
				case "define": {
					const macro = this._parseMacroDefinition(
						directive,
						origin,
						preprocessContext
					);
					if (!macro) {
						continue;
					}
					if (preprocessContext.macros.has(macro.name)) {
						this._pushDirectiveDiagnostic(
							preprocessContext,
							"directive-define-redefined",
							`Macro "${macro.name}" was redefined; latest definition wins.`,
							origin.sourcePath,
							origin.sourceLine,
							directive.column,
							"warning"
						);
					}
						preprocessContext.macros.set(macro.name, macro);
						continue;
					}
					case "undef": {
						const macroName = this._parseSingleDirectiveIdentifier(
							directive,
							origin,
							preprocessContext,
							"directive-undef-invalid"
						);
						if (!macroName) {
							continue;
						}
						preprocessContext.macros.delete(macroName);
						continue;
					}
					case "inject":
						outputLines.push(line);
						outputOrigins.push(origin);
						continue;
					default:
						outputLines.push(line);
						outputOrigins.push(origin);
				}
			}
		if (conditionalStack.length > 0) {
			for (const state of conditionalStack) {
				this._pushDirectiveDiagnostic(
					preprocessContext,
					"directive-conditional-unterminated",
					`Directive conditional block starting at "${state.sourcePath}:${state.sourceLine}" was not terminated with "#endif".`,
					state.sourcePath,
					state.sourceLine,
					state.column
				);
			}
		}

			return this._composeLinesToComposite(outputLines, outputOrigins);
		}

	private _applyConditionalDirective(
		directive: DirectiveLine,
		origin: LineOrigin,
		preprocessContext: PreprocessContext,
		stack: ConditionalBranchState[]
	): void {
		switch (directive.name) {
			case "if": {
				const parentActive =
					stack.length <= 0 ? true : stack[stack.length - 1].currentActive;
				const branchValue =
					parentActive ?
						this._evaluateDirectiveConditionExpression(
							directive.body,
							directive,
							origin,
							preprocessContext
						)
					:	false;
				stack.push({
					parentActive,
					branchTaken: parentActive && branchValue,
					currentActive: parentActive && branchValue,
					elseSeen: false,
					sourcePath: origin.sourcePath,
					sourceLine: origin.sourceLine,
					column: directive.column,
				});
				return;
			}
			case "ifdef":
			case "ifndef": {
				const parentActive =
					stack.length <= 0 ? true : stack[stack.length - 1].currentActive;
				if (!parentActive) {
					stack.push({
						parentActive,
						branchTaken: false,
						currentActive: false,
						elseSeen: false,
						sourcePath: origin.sourcePath,
						sourceLine: origin.sourceLine,
						column: directive.column,
					});
					return;
				}
				const identifier = this._parseSingleDirectiveIdentifier(
					directive,
					origin,
					preprocessContext,
					"directive-conditional-invalid-identifier"
				);
				const branchValue =
					identifier ?
						directive.name === "ifdef" ?
							preprocessContext.macros.has(identifier)
						:	!preprocessContext.macros.has(identifier)
					:	false;
				stack.push({
					parentActive,
					branchTaken: parentActive && branchValue,
					currentActive: parentActive && branchValue,
					elseSeen: false,
					sourcePath: origin.sourcePath,
					sourceLine: origin.sourceLine,
					column: directive.column,
				});
				return;
			}
			case "elif": {
				if (stack.length <= 0) {
					this._pushDirectiveDiagnostic(
						preprocessContext,
						"directive-conditional-elif-without-if",
						`Directive "#elif" must follow "#if", "#ifdef", or "#ifndef".`,
						origin.sourcePath,
						origin.sourceLine,
						directive.column
					);
					return;
				}
				const branch = stack[stack.length - 1];
				if (branch.elseSeen) {
					this._pushDirectiveDiagnostic(
						preprocessContext,
						"directive-conditional-elif-after-else",
						`Directive "#elif" cannot appear after "#else" in the same conditional block.`,
						origin.sourcePath,
						origin.sourceLine,
						directive.column
					);
					branch.currentActive = false;
					return;
				}
				if (!branch.parentActive || branch.branchTaken) {
					branch.currentActive = false;
					return;
				}
				const value = this._evaluateDirectiveConditionExpression(
					directive.body,
					directive,
					origin,
					preprocessContext
				);
				branch.currentActive = branch.parentActive && value;
				if (branch.currentActive) {
					branch.branchTaken = true;
				}
				return;
			}
			case "else": {
				if (stack.length <= 0) {
					this._pushDirectiveDiagnostic(
						preprocessContext,
						"directive-conditional-else-without-if",
						`Directive "#else" must follow "#if", "#ifdef", or "#ifndef".`,
						origin.sourcePath,
						origin.sourceLine,
						directive.column
					);
					return;
				}
				const branch = stack[stack.length - 1];
				if (branch.elseSeen) {
					this._pushDirectiveDiagnostic(
						preprocessContext,
						"directive-conditional-else-duplicate",
						`Directive "#else" can appear only once per conditional block.`,
						origin.sourcePath,
						origin.sourceLine,
						directive.column
					);
					branch.currentActive = false;
					return;
				}
				branch.elseSeen = true;
				branch.currentActive = branch.parentActive && !branch.branchTaken;
				if (branch.currentActive) {
					branch.branchTaken = true;
				}
				return;
			}
			case "endif": {
				if (stack.length <= 0) {
					this._pushDirectiveDiagnostic(
						preprocessContext,
						"directive-conditional-endif-without-if",
						`Directive "#endif" must follow "#if", "#ifdef", or "#ifndef".`,
						origin.sourcePath,
						origin.sourceLine,
						directive.column
					);
					return;
				}
				stack.pop();
				return;
			}
		}
	}

	private _parseSingleDirectiveIdentifier(
		directive: DirectiveLine,
		origin: LineOrigin,
		preprocessContext: PreprocessContext,
		errorCode: string
	): string | null {
		const body = directive.body.trim();
		if (/^[A-Za-z_][A-Za-z0-9_]*$/.test(body)) {
			return body;
		}
		this._pushDirectiveDiagnostic(
			preprocessContext,
			errorCode,
			`Directive "#${directive.name}" expects a single macro identifier.`,
			origin.sourcePath,
			origin.sourceLine,
			directive.column
		);
		return null;
	}

	private _evaluateDirectiveConditionExpression(
		expression: string,
		directive: DirectiveLine,
		origin: LineOrigin,
		preprocessContext: PreprocessContext
	): boolean {
		const trimmedExpression = expression.trim();
		if (trimmedExpression.length <= 0) {
			this._pushDirectiveDiagnostic(
				preprocessContext,
				"directive-conditional-expression-invalid",
				`Directive "#${directive.name}" requires a condition expression.`,
				origin.sourcePath,
				origin.sourceLine,
				directive.column
			);
			return false;
		}
		const resolverStack = new Set<string>();
		const baseColumn = directive.column + directive.name.length + 2;
		try {
			const parser = new DirectiveConditionParser({
				expression: trimmedExpression,
				baseColumn,
				isDefined: (identifier) => preprocessContext.macros.has(identifier),
				resolveIdentifier: (identifier) =>
					this._resolveDirectiveConditionIdentifier(
						identifier,
						preprocessContext,
						origin,
						baseColumn,
						resolverStack
					),
			});
			return parser.parse() !== 0n;
		} catch (error) {
			const message =
				error instanceof Error ? error.message : "Invalid directive condition expression.";
			const column =
				error instanceof ConditionParseError ? error.column : directive.column;
			this._pushDirectiveDiagnostic(
				preprocessContext,
				"directive-conditional-expression-invalid",
				message,
				origin.sourcePath,
				origin.sourceLine,
				column
			);
			return false;
		}
	}

	private _resolveDirectiveConditionIdentifier(
		identifier: string,
		preprocessContext: PreprocessContext,
		origin: LineOrigin,
		column: number,
		resolverStack: Set<string>
	): bigint {
		if (resolverStack.has(identifier)) {
			return 0n;
		}
		const macro = preprocessContext.macros.get(identifier);
		if (!macro) {
			return 0n;
		}
		if (macro.kind === "function") {
			return 0n;
		}
		const expanded = this._expandMacroText(
			macro.replacement,
			preprocessContext,
			origin.sourcePath,
			origin.sourceLine,
			1
		).trim();
		if (expanded.length <= 0) {
			return 0n;
		}
		resolverStack.add(identifier);
		try {
			const parser = new DirectiveConditionParser({
				expression: expanded,
				baseColumn: column,
				isDefined: (name) => preprocessContext.macros.has(name),
				resolveIdentifier: (name) =>
					this._resolveDirectiveConditionIdentifier(
						name,
						preprocessContext,
						origin,
						column,
						resolverStack
					),
			});
			return parser.parse();
		} finally {
			resolverStack.delete(identifier);
		}
	}

	private _findFirstGLSLVersionLine(lines: string[]): number {
		for (let i = 0; i < lines.length; i++) {
			const line = lines[i].trim();
			if (line.length <= 0) {
				continue;
			}
			if (line.startsWith("#version")) {
				return i + 1;
			}
			return 0;
		}
		return 0;
	}

	private _parseIncludeSpecifier(
		directive: DirectiveLine,
		origin: LineOrigin,
		preprocessContext: PreprocessContext
	): IncludeSpecifier | null {
		const body = directive.body.trim();
		if (body.length <= 0) {
			this._pushDirectiveDiagnostic(
				preprocessContext,
				"directive-include-empty",
				`Directive "#${directive.name}" requires a module path.`,
				origin.sourcePath,
				origin.sourceLine,
				directive.column
			);
			return null;
		}
		if (body.startsWith("<") && body.endsWith(">")) {
			return {
				kind: "angle",
				path: body.slice(1, -1).trim(),
			};
		}
		if (body.startsWith("\"") && body.endsWith("\"")) {
			return {
				kind: "quote",
				path: body.slice(1, -1),
			};
		}
		this._pushDirectiveDiagnostic(
			preprocessContext,
			"directive-include-invalid-path",
			`Directive "#${directive.name}" expects <path> or "path".`,
			origin.sourcePath,
			origin.sourceLine,
			directive.column
		);
		return null;
	}

	private _resolveIncludeComposite(
		specifier: IncludeSpecifier,
		currentModulePath: string,
		preprocessContext: PreprocessContext,
		origin: LineOrigin
	): CompositeShaderSource | null {
		const canonicalModuleId = this._resolveIncludeModuleId(
			specifier,
			currentModulePath,
			preprocessContext.language
		);
		if (!canonicalModuleId) {
			this._pushDirectiveDiagnostic(
				preprocessContext,
				"directive-include-invalid-target",
				`Include target "${specifier.path}" is invalid.`,
				origin.sourcePath,
				origin.sourceLine,
				1
			);
			return null;
		}
		if (preprocessContext.processingStack.includes(canonicalModuleId)) {
			const chain = [...preprocessContext.processingStack, canonicalModuleId].join(
				" -> "
			);
			this._pushDirectiveDiagnostic(
				preprocessContext,
				"directive-include-cycle",
				`Detected cyclic include/import chain: ${chain}.`,
				origin.sourcePath,
				origin.sourceLine,
				1
			);
			return null;
		}
		if (preprocessContext.expandedModules.has(canonicalModuleId)) {
			return null;
		}
		const module = this._resolveRegisteredIncludeModule(
			preprocessContext.language,
			canonicalModuleId
		);
		if (!module) {
			this._pushDirectiveDiagnostic(
				preprocessContext,
				"directive-include-not-found",
				`Include module "${canonicalModuleId}" was not registered for ${preprocessContext.language}.`,
				origin.sourcePath,
				origin.sourceLine,
				1
			);
			return null;
		}
		preprocessContext.processingStack.push(canonicalModuleId);
		preprocessContext.expandedModules.add(canonicalModuleId);
		const expanded = this._expandDirectiveComposite(
			createInlineCompositeShaderSource(
				module.code,
				module.sourcePath,
				"include"
			),
			canonicalModuleId,
			preprocessContext
		);
		preprocessContext.processingStack.pop();
		return expanded;
	}

	private _resolveIncludeModuleId(
		specifier: IncludeSpecifier,
		currentModulePath: string,
		language: ShaderLanguage
	): string | null {
		const normalizedPath = specifier.path.replace(/\\/g, "/").trim();
		if (normalizedPath.length <= 0) {
			return null;
		}
		if (specifier.kind === "angle") {
			const canonical = this._canonicalizeModulePathSafe(normalizedPath);
			if (!canonical) {
				return null;
			}
			if (this._resolveRegisteredIncludeModule(language, canonical)) {
				return canonical;
			}
			const withExtension = this._withLanguageDefaultExtension(
				canonical,
				language
			);
			if (this._resolveRegisteredIncludeModule(language, withExtension)) {
				return withExtension;
			}
			return withExtension;
		}
		const joined = this._joinModulePath(currentModulePath, normalizedPath);
		const canonical = this._canonicalizeModulePathSafe(joined);
		if (!canonical) {
			return null;
		}
		if (this._resolveRegisteredIncludeModule(language, canonical)) {
			return canonical;
		}
		const withExtension = this._withLanguageDefaultExtension(canonical, language);
		if (this._resolveRegisteredIncludeModule(language, withExtension)) {
			return withExtension;
		}
		return withExtension;
	}

	private _resolveRegisteredIncludeModule(
		language: ShaderLanguage,
		moduleId: string
	): RegisteredIncludeModule | null {
		const modules = this._includeModulesByLanguage.get(language);
		if (!modules) {
			return null;
		}
		return modules.get(moduleId) ?? null;
	}

	private _withLanguageDefaultExtension(
		moduleId: string,
		language: ShaderLanguage
	): string {
		const slashIndex = moduleId.lastIndexOf("/");
		const fileName = slashIndex >= 0 ? moduleId.slice(slashIndex + 1) : moduleId;
		if (fileName.includes(".")) {
			return moduleId;
		}
		return `${moduleId}.${language === "wgsl" ? "wgsl" : "glsl"}`;
	}

	private _joinModulePath(baseModulePath: string, relativePath: string): string {
		const base = baseModulePath.replace(/\\/g, "/");
		const slashIndex = base.lastIndexOf("/");
		const directory = slashIndex >= 0 ? base.slice(0, slashIndex) : "";
		return directory.length > 0 ? `${directory}/${relativePath}` : relativePath;
	}

	private _canonicalizeModulePathSafe(value: string): string {
		try {
			return this._canonicalizeModulePath(value);
		} catch (error) {
			return value
				.replace(/\\/g, "/")
				.replace(/^\/+/, "")
				.replace(/\/{2,}/g, "/")
				.trim();
		}
	}

	private _canonicalizeModulePath(value: string): string {
		const normalized = value.replace(/\\/g, "/").trim();
		if (normalized.length <= 0) {
			throw new Error("Shader module path cannot be empty.");
		}
		const segments: string[] = [];
		for (const rawSegment of normalized.split("/")) {
			const segment = rawSegment.trim();
			if (segment.length <= 0 || segment === ".") {
				continue;
			}
			if (segment === "..") {
				if (segments.length <= 0) {
					throw new Error(
						`Shader module path "${value}" escapes outside include root.`
					);
				}
				segments.pop();
				continue;
			}
			segments.push(segment);
		}
		if (segments.length <= 0) {
			throw new Error(`Shader module path "${value}" resolved to an empty path.`);
		}
		return segments.join("/");
	}

	private _formatIncludeModuleEventId(
		language: ShaderLanguage,
		moduleId: string
	): string {
		return `${language}:${moduleId}`;
	}

	private _pushDirectiveDiagnostic(
		preprocessContext: PreprocessContext,
		code: string,
		message: string,
		sourcePath: string,
		line: number,
		column: number,
		overrideSeverity?: ShaderDiagnosticSeverity
	): void {
		preprocessContext.diagnostics.push({
			ruleId: "ignis/directive-runtime",
			code,
			severity:
				overrideSeverity ?? this._resolveDirectiveSeverityByMode(preprocessContext.mode),
			message,
			sourcePath,
			line: Math.max(1, Math.floor(line)),
			column: Math.max(1, Math.floor(column)),
			range: createPointRange(
				Math.max(1, Math.floor(line)),
				Math.max(1, Math.floor(column))
			),
		});
	}

	private _resolveDirectiveSeverityByMode(
		mode: ShaderRuntimeMode
	): ShaderDiagnosticSeverity {
		return mode === "strict" ? "error" : "warning";
	}

	private _parseMacroDefinition(
		directive: DirectiveLine,
		origin: LineOrigin,
		preprocessContext: PreprocessContext
	): MacroDefinition | null {
		const body = directive.body;
		if (body.length <= 0) {
			this._pushDirectiveDiagnostic(
				preprocessContext,
				"directive-define-invalid",
				`Directive "#define" requires a macro name.`,
				origin.sourcePath,
				origin.sourceLine,
				directive.column
			);
			return null;
		}
		const match = /^([A-Za-z_][A-Za-z0-9_]*)(.*)$/.exec(body);
		if (!match) {
			this._pushDirectiveDiagnostic(
				preprocessContext,
				"directive-define-invalid",
				`Directive "#define" has invalid syntax.`,
				origin.sourcePath,
				origin.sourceLine,
				directive.column
			);
			return null;
		}
		const macroName = match[1];
		const remainder = match[2] ?? "";
		if (remainder.startsWith("(")) {
			const closeIndex = remainder.indexOf(")");
			if (closeIndex < 0) {
				this._pushDirectiveDiagnostic(
					preprocessContext,
					"directive-define-function-invalid",
					`Function macro "${macroName}" is missing closing ")".`,
					origin.sourcePath,
					origin.sourceLine,
					directive.column
				);
				return null;
			}
			const parameterList = remainder.slice(1, closeIndex).trim();
			const params =
				parameterList.length <= 0 ?
					[]
				:	parameterList
						.split(",")
						.map((parameter) => parameter.trim())
						.filter((parameter) => parameter.length > 0);
			for (const parameter of params) {
				if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(parameter)) {
					this._pushDirectiveDiagnostic(
						preprocessContext,
						"directive-define-function-param-invalid",
						`Function macro "${macroName}" has invalid parameter "${parameter}".`,
						origin.sourcePath,
						origin.sourceLine,
						directive.column
					);
					return null;
				}
			}
			const replacement = remainder.slice(closeIndex + 1).trimStart();
			return {
				kind: "function",
				name: macroName,
				params,
				replacement,
				sourcePath: origin.sourcePath,
				sourceLine: origin.sourceLine,
			};
		}
		return {
			kind: "object",
			name: macroName,
			replacement: remainder.trimStart(),
			sourcePath: origin.sourcePath,
			sourceLine: origin.sourceLine,
		};
	}

	private _expandMacrosInComposite(
		composite: CompositeShaderSource,
		preprocessContext: PreprocessContext
	): CompositeShaderSource {
		if (preprocessContext.macros.size <= 0) {
			return composite;
		}
		const { lines, origins } = this._splitCompositeLines(composite);
		const outputLines: string[] = [];
		const macroState: DirectiveLineScanState = {
			inBlockComment: false,
			stringQuote: null,
			escape: false,
		};
		for (let index = 0; index < lines.length; index++) {
			const origin = origins[index] ?? {
				sourcePath: preprocessContext.sourcePath,
				sourceLine: index + 1,
				kind: "source",
			};
			outputLines.push(
				this._expandMacrosInLine(
					lines[index],
					macroState,
					preprocessContext,
					origin.sourcePath,
					origin.sourceLine
				)
			);
		}
		return this._composeLinesToComposite(outputLines, origins);
	}

	private _expandMacrosInLine(
		line: string,
		state: DirectiveLineScanState,
		preprocessContext: PreprocessContext,
		sourcePath: string,
		sourceLine: number
	): string {
		let output = "";
		let index = 0;
		while (index < line.length) {
			const char = line[index];
			const next = index + 1 < line.length ? line[index + 1] : "";
			if (state.inBlockComment) {
				output += char;
				if (char === "*" && next === "/") {
					output += "/";
					state.inBlockComment = false;
					index += 2;
					continue;
				}
				index++;
				continue;
			}
			if (state.stringQuote) {
				output += char;
				if (state.escape) {
					state.escape = false;
					index++;
					continue;
				}
				if (char === "\\") {
					state.escape = true;
					index++;
					continue;
				}
				if (char === state.stringQuote) {
					state.stringQuote = null;
					index++;
					continue;
				}
				index++;
				continue;
			}
			if (char === "/" && next === "/") {
				output += line.slice(index);
				break;
			}
			if (char === "/" && next === "*") {
				output += "/*";
				state.inBlockComment = true;
				index += 2;
				continue;
			}
			if (char === "\"" || char === "'") {
				output += char;
				state.stringQuote = char as '"' | "'";
				state.escape = false;
				index++;
				continue;
			}
			if (isIdentifierStartCharacter(char)) {
				let end = index + 1;
				while (
					end < line.length &&
					isIdentifierPartCharacter(line[end])
				) {
					end++;
				}
				const token = line.slice(index, end);
				const macro = preprocessContext.macros.get(token);
				if (!macro) {
					output += token;
					index = end;
					continue;
				}
				if (macro.kind === "object") {
					output += this._expandMacroText(
						macro.replacement,
						preprocessContext,
						sourcePath,
						sourceLine,
						1
					);
					index = end;
					continue;
				}
				const invocation = this._parseFunctionMacroInvocation(line, end);
				if (!invocation) {
					output += token;
					index = end;
					continue;
				}
				if (invocation.args.length !== macro.params.length) {
					this._pushDirectiveDiagnostic(
						preprocessContext,
						"directive-macro-arg-count",
						`Macro "${macro.name}" expected ${macro.params.length} argument(s) but got ${invocation.args.length}.`,
						sourcePath,
						sourceLine,
						index + 1,
						"warning"
					);
				}
				const substituted = this._substituteFunctionMacro(
					macro,
					invocation.args.map((argument) =>
						this._expandMacroText(
							argument.trim(),
							preprocessContext,
							sourcePath,
							sourceLine,
							1
						)
					)
				);
				output += this._expandMacroText(
					substituted,
					preprocessContext,
					sourcePath,
					sourceLine,
					1
				);
				index = invocation.endIndex + 1;
				continue;
			}
			output += char;
			index++;
		}
		return output;
	}

	private _expandMacroText(
		text: string,
		preprocessContext: PreprocessContext,
		sourcePath: string,
		sourceLine: number,
		depth: number
	): string {
		if (text.length <= 0 || preprocessContext.macros.size <= 0) {
			return text;
		}
		if (depth > DIRECTIVE_MAX_MACRO_EXPANSION_DEPTH) {
			this._pushDirectiveDiagnostic(
				preprocessContext,
				"directive-macro-depth-limit",
				`Macro expansion exceeded maximum depth (${DIRECTIVE_MAX_MACRO_EXPANSION_DEPTH}).`,
				sourcePath,
				sourceLine,
				1,
				"warning"
			);
			return text;
		}
		let output = "";
		let index = 0;
		while (index < text.length) {
			const char = text[index];
			if (!isIdentifierStartCharacter(char)) {
				output += char;
				index++;
				continue;
			}
			let end = index + 1;
			while (end < text.length && isIdentifierPartCharacter(text[end])) {
				end++;
			}
			const token = text.slice(index, end);
			const macro = preprocessContext.macros.get(token);
			if (!macro) {
				output += token;
				index = end;
				continue;
			}
			if (macro.kind === "object") {
				output += this._expandMacroText(
					macro.replacement,
					preprocessContext,
					sourcePath,
					sourceLine,
					depth + 1
				);
				index = end;
				continue;
			}
			output += token;
			index = end;
		}
		return output;
	}

	private _parseFunctionMacroInvocation(
		line: string,
		identifierEndIndex: number
	): { args: string[]; endIndex: number } | null {
		let index = identifierEndIndex;
		while (index < line.length && isWhitespaceCharacter(line[index])) {
			index++;
		}
		if (index >= line.length || line[index] !== "(") {
			return null;
		}
		let depth = 1;
		let cursor = index + 1;
		let current = "";
		const args: string[] = [];
		let stringQuote: '"' | "'" | null = null;
		let escape = false;
		while (cursor < line.length) {
			const char = line[cursor];
			if (stringQuote) {
				current += char;
				if (escape) {
					escape = false;
					cursor++;
					continue;
				}
				if (char === "\\") {
					escape = true;
					cursor++;
					continue;
				}
				if (char === stringQuote) {
					stringQuote = null;
				}
				cursor++;
				continue;
			}
			if (char === "\"" || char === "'") {
				current += char;
				stringQuote = char as '"' | "'";
				escape = false;
				cursor++;
				continue;
			}
			if (char === "(") {
				depth++;
				current += char;
				cursor++;
				continue;
			}
			if (char === ")") {
				depth--;
				if (depth === 0) {
					if (current.trim().length > 0 || args.length > 0) {
						args.push(current.trim());
					}
					return {
						args,
						endIndex: cursor,
					};
				}
				current += char;
				cursor++;
				continue;
			}
			if (char === "," && depth === 1) {
				args.push(current.trim());
				current = "";
				cursor++;
				continue;
			}
			current += char;
			cursor++;
		}
		return null;
	}

	private _substituteFunctionMacro(
		macro: FunctionMacroDefinition,
		args: string[]
	): string {
		const parameterMap = new Map<string, string>();
		for (let index = 0; index < macro.params.length; index++) {
			parameterMap.set(macro.params[index], args[index] ?? "");
		}
		let output = "";
		let cursor = 0;
		while (cursor < macro.replacement.length) {
			const char = macro.replacement[cursor];
			if (!isIdentifierStartCharacter(char)) {
				output += char;
				cursor++;
				continue;
			}
			let end = cursor + 1;
			while (
				end < macro.replacement.length &&
				isIdentifierPartCharacter(macro.replacement[end])
			) {
				end++;
			}
			const token = macro.replacement.slice(cursor, end);
			output += parameterMap.get(token) ?? token;
			cursor = end;
		}
		return output;
	}

	private _resolveDirectiveInjectsSync(
		composite: CompositeShaderSource,
		preprocessContext: PreprocessContext
	): CompositeShaderSource {
		const { lines, origins } = this._splitCompositeLines(composite);
		const outputLines: string[] = [];
		const outputOrigins: LineOrigin[] = [];
		const headerBlocks: InjectionBlock[] = [];
		const functionBlocks: InjectionBlock[] = [];
		const directiveState: DirectiveLineScanState = {
			inBlockComment: false,
			stringQuote: null,
			escape: false,
		};
		for (let index = 0; index < lines.length; index++) {
			const line = lines[index];
			const origin = origins[index] ?? {
				sourcePath: preprocessContext.sourcePath,
				sourceLine: index + 1,
				kind: "source",
			};
			const directive = this._scanDirectiveFromLine(line, directiveState);
			if (!directive || directive.name !== "inject") {
				outputLines.push(line);
				outputOrigins.push(origin);
				continue;
			}
			const invocation = this._parseInjectInvocation(
				directive,
				preprocessContext,
				origin
			);
			if (!invocation) {
				continue;
			}
			const script = this._injectionScripts.get(invocation.id);
			if (!script) {
				this._pushDirectiveDiagnostic(
					preprocessContext,
					"directive-inject-not-found",
					`Injection script "${invocation.id}" was not registered.`,
					origin.sourcePath,
					origin.sourceLine,
					directive.column
				);
				continue;
			}
			if (script.language && script.language !== preprocessContext.language) {
				this._pushDirectiveDiagnostic(
					preprocessContext,
					"directive-inject-language-mismatch",
					`Injection script "${invocation.id}" does not support ${preprocessContext.language}.`,
					origin.sourcePath,
					origin.sourceLine,
					directive.column
				);
				continue;
			}
			const scriptContext = this._createInjectionScriptContext(
				preprocessContext,
				composite.code
			);
			const args = this._validateInjectionArguments(
				script,
				invocation.args,
				scriptContext,
				preprocessContext,
				origin,
				directive.column,
			);
			if (!args) {
				continue;
			}
			const injection = script.run(args, scriptContext);
			if (isPromiseLike(injection)) {
				throw new Error(
					`Injection script "${script.id}" returned a Promise during process(). Use processAsync().`
				);
			}
			this._appendDirectiveInjectionBlocks(
				preprocessContext,
				script,
				injection,
				headerBlocks,
				functionBlocks
			);
		}
		const baseComposite = this._composeLinesToComposite(outputLines, outputOrigins);
		const mergedBlocks = [...headerBlocks, ...functionBlocks];
		if (mergedBlocks.length <= 0) {
			return baseComposite;
		}
		return preprocessContext.language === "wgsl" ?
				injectWGSLSource(baseComposite, mergedBlocks)
			:	injectGLSLSource(baseComposite, mergedBlocks);
	}

	private async _resolveDirectiveInjectsAsync(
		composite: CompositeShaderSource,
		preprocessContext: PreprocessContext
	): Promise<CompositeShaderSource> {
		const { lines, origins } = this._splitCompositeLines(composite);
		const outputLines: string[] = [];
		const outputOrigins: LineOrigin[] = [];
		const headerBlocks: InjectionBlock[] = [];
		const functionBlocks: InjectionBlock[] = [];
		const directiveState: DirectiveLineScanState = {
			inBlockComment: false,
			stringQuote: null,
			escape: false,
		};
		for (let index = 0; index < lines.length; index++) {
			const line = lines[index];
			const origin = origins[index] ?? {
				sourcePath: preprocessContext.sourcePath,
				sourceLine: index + 1,
				kind: "source",
			};
			const directive = this._scanDirectiveFromLine(line, directiveState);
			if (!directive || directive.name !== "inject") {
				outputLines.push(line);
				outputOrigins.push(origin);
				continue;
			}
			const invocation = this._parseInjectInvocation(
				directive,
				preprocessContext,
				origin
			);
			if (!invocation) {
				continue;
			}
			const script = this._injectionScripts.get(invocation.id);
			if (!script) {
				this._pushDirectiveDiagnostic(
					preprocessContext,
					"directive-inject-not-found",
					`Injection script "${invocation.id}" was not registered.`,
					origin.sourcePath,
					origin.sourceLine,
					directive.column
				);
				continue;
			}
			if (script.language && script.language !== preprocessContext.language) {
				this._pushDirectiveDiagnostic(
					preprocessContext,
					"directive-inject-language-mismatch",
					`Injection script "${invocation.id}" does not support ${preprocessContext.language}.`,
					origin.sourcePath,
					origin.sourceLine,
					directive.column
				);
				continue;
			}
			const scriptContext = this._createInjectionScriptContext(
				preprocessContext,
				composite.code
			);
			const args = this._validateInjectionArguments(
				script,
				invocation.args,
				scriptContext,
				preprocessContext,
				origin,
				directive.column,
			);
			if (!args) {
				continue;
			}
			const injection = await script.run(args, scriptContext);
			this._appendDirectiveInjectionBlocks(
				preprocessContext,
				script,
				injection,
				headerBlocks,
				functionBlocks
			);
		}
		const baseComposite = this._composeLinesToComposite(outputLines, outputOrigins);
		const mergedBlocks = [...headerBlocks, ...functionBlocks];
		if (mergedBlocks.length <= 0) {
			return baseComposite;
		}
		return preprocessContext.language === "wgsl" ?
				injectWGSLSource(baseComposite, mergedBlocks)
			:	injectGLSLSource(baseComposite, mergedBlocks);
	}

	private _appendDirectiveInjectionBlocks(
		preprocessContext: PreprocessContext,
		script: ShaderInjectionScript,
		injection: ShaderRuleInjection | null | undefined,
		headers: InjectionBlock[],
		functions: InjectionBlock[]
	): void {
		if (!injection) {
			return;
		}
		const header = normalizeInjectionBlock(injection.header);
		if (header.length > 0) {
			headers.push({
				code: header,
				sourcePath: `<directive:inject:${script.id}:header>`,
				label: `directive-inject:${script.id}:header`,
				anchor: normalizeInjectionAnchorForLanguage(
					preprocessContext.language,
					injection.headerAnchor
				),
			});
		}
		const functionsBlock = normalizeInjectionBlock(injection.functions);
		if (functionsBlock.length > 0) {
			functions.push({
				code: functionsBlock,
				sourcePath: `<directive:inject:${script.id}:functions>`,
				label: `directive-inject:${script.id}:functions`,
				anchor: normalizeInjectionAnchorForLanguage(
					preprocessContext.language,
					injection.functionsAnchor
				),
			});
		}
	}

	private _validateInjectionArguments(
		script: ShaderInjectionScript,
		rawArgs: Readonly<Record<string, ShaderInjectionArgValue>>,
		scriptContext: ShaderInjectionScriptContext,
		preprocessContext: PreprocessContext,
		origin: LineOrigin,
		column: number,
	): ShaderInjectionArguments<ShaderInjectionArgumentSchema> | null {
		const schema = script.arguments;
		const resolved: Record<string, ShaderInjectionArgValue | undefined> = {};
		let valid = true;
		for (const name of Object.keys(rawArgs)) {
			if (Object.prototype.hasOwnProperty.call(schema, name)) {
				continue;
			}
			valid = false;
			this._pushInjectionArgumentDiagnostic(
				preprocessContext,
				"directive-inject-argument-unknown",
				`Injection script "${script.id}" does not declare argument "${name}".`,
				origin,
				column,
			);
		}
		for (const [name, definition] of Object.entries(schema)) {
			const value = rawArgs[name];
			if (value === undefined) {
				if ("default" in definition && definition.default !== undefined) {
					resolved[name] = definition.default;
					continue;
				}
				resolved[name] = undefined;
				if (definition.required === true) {
					valid = false;
					this._pushInjectionArgumentDiagnostic(
						preprocessContext,
						"directive-inject-argument-missing",
						`Injection script "${script.id}" requires argument "${name}".`,
						origin,
						column,
					);
				}
				continue;
			}
			const message = this._validateInjectionArgumentValue(
				name,
				value,
				definition,
			);
			if (message) {
				valid = false;
				this._pushInjectionArgumentDiagnostic(
					preprocessContext,
					"directive-inject-argument-invalid",
					`Injection script "${script.id}" ${message}`,
					origin,
					column,
				);
				continue;
			}
			resolved[name] = value;
		}
		if (!valid) {
			return null;
		}
		const typedArgs = resolved as ShaderInjectionArguments<ShaderInjectionArgumentSchema>;
		const validation = script.validateArguments?.(typedArgs, scriptContext);
		const messages =
			typeof validation === "string" ? [validation]
			: Array.isArray(validation) ? validation
			: [];
		if (messages.length <= 0) {
			return typedArgs;
		}
		for (const message of messages) {
			this._pushInjectionArgumentDiagnostic(
				preprocessContext,
				"directive-inject-validation-failed",
				`Injection script "${script.id}" ${message}`,
				origin,
				column,
			);
		}
		return null;
	}

	private _validateInjectionArgumentValue(
		name: string,
		value: ShaderInjectionArgValue,
		definition: ShaderInjectionArgumentDefinition,
	): string | null {
		switch (definition.type) {
			case "string":
				return typeof value === "string" ?
					null
				: `argument "${name}" must be a string.`;
			case "boolean":
				return typeof value === "boolean" ?
					null
				: `argument "${name}" must be a boolean.`;
			case "enum":
				return typeof value === "string" && definition.values.includes(value) ?
					null
				: `argument "${name}" must be one of ${definition.values
						.map((entry) => `"${entry}"`)
						.join(", ")}.`;
			case "number":
			case "integer":
				if (
					typeof value !== "number" ||
					!Number.isFinite(value) ||
					(definition.type === "integer" && !Number.isInteger(value))
				) {
					return `argument "${name}" must be ${definition.type === "integer" ? "an integer" : "a finite number"}.`;
				}
				if (definition.min !== undefined && value < definition.min) {
					return `argument "${name}" must be at least ${definition.min}.`;
				}
				if (definition.max !== undefined && value > definition.max) {
					return `argument "${name}" must be at most ${definition.max}.`;
				}
				return null;
		}
	}

	private _pushInjectionArgumentDiagnostic(
		preprocessContext: PreprocessContext,
		code: string,
		message: string,
		origin: LineOrigin,
		column: number,
	): void {
		if (preprocessContext.mode === "silent") {
			return;
		}
		this._pushDirectiveDiagnostic(
			preprocessContext,
			code,
			message,
			origin.sourcePath,
			origin.sourceLine,
			column,
		);
	}

	private _parseInjectInvocation(
		directive: DirectiveLine,
		preprocessContext: PreprocessContext,
		origin: LineOrigin
	): InjectInvocation | null {
		const body = directive.body.trim();
		const match =
			/^<([^>]+)>\s*(?:\((.*)\))?$/.exec(body) ??
			/^([A-Za-z_][A-Za-z0-9_\/\.-]*)\s*(?:\((.*)\))?$/.exec(body);
		if (!match) {
			this._pushDirectiveDiagnostic(
				preprocessContext,
				"directive-inject-invalid",
				`Directive "#inject" expects <script-id>(key=value, ...).`,
				origin.sourcePath,
				origin.sourceLine,
				directive.column
			);
			return null;
		}
		const id = match[1].trim();
		const argsPayload = (match[2] ?? "").trim();
		const args: Record<string, ShaderInjectionArgValue> = {};
		if (argsPayload.length > 0) {
			for (const part of this._splitInjectArguments(argsPayload)) {
				const equalIndex = part.indexOf("=");
				if (equalIndex <= 0) {
					this._pushDirectiveDiagnostic(
						preprocessContext,
						"directive-inject-arg-invalid",
						`Invalid inject argument "${part}". Expected key=value.`,
						origin.sourcePath,
						origin.sourceLine,
						directive.column,
						"warning"
					);
					continue;
				}
				const key = part.slice(0, equalIndex).trim();
				if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(key)) {
					this._pushDirectiveDiagnostic(
						preprocessContext,
						"directive-inject-arg-key-invalid",
						`Invalid inject argument key "${key}".`,
						origin.sourcePath,
						origin.sourceLine,
						directive.column,
						"warning"
					);
					continue;
				}
				const valueRaw = part.slice(equalIndex + 1).trim();
				args[key] = this._parseInjectArgumentValue(valueRaw);
			}
		}
		return {
			id,
			args,
		};
	}

	private _splitInjectArguments(payload: string): string[] {
		const parts: string[] = [];
		let current = "";
		let stringQuote: '"' | "'" | null = null;
		let escape = false;
		let depth = 0;
		for (let index = 0; index < payload.length; index++) {
			const char = payload[index];
			if (stringQuote) {
				current += char;
				if (escape) {
					escape = false;
					continue;
				}
				if (char === "\\") {
					escape = true;
					continue;
				}
				if (char === stringQuote) {
					stringQuote = null;
				}
				continue;
			}
			if (char === "\"" || char === "'") {
				current += char;
				stringQuote = char as '"' | "'";
				escape = false;
				continue;
			}
			if (char === "(") {
				depth++;
				current += char;
				continue;
			}
			if (char === ")") {
				depth = Math.max(0, depth - 1);
				current += char;
				continue;
			}
			if (char === "," && depth === 0) {
				const value = current.trim();
				if (value.length > 0) {
					parts.push(value);
				}
				current = "";
				continue;
			}
			current += char;
		}
		const tail = current.trim();
		if (tail.length > 0) {
			parts.push(tail);
		}
		return parts;
	}

	private _parseInjectArgumentValue(
		rawValue: string
	): ShaderInjectionArgValue {
		const trimmed = rawValue.trim();
		if (
			(trimmed.startsWith("\"") && trimmed.endsWith("\"")) ||
			(trimmed.startsWith("'") && trimmed.endsWith("'"))
		) {
			return trimmed.slice(1, -1);
		}
		if (trimmed === "true") {
			return true;
		}
		if (trimmed === "false") {
			return false;
		}
		const numeric = Number(trimmed);
		if (Number.isFinite(numeric) && trimmed.length > 0) {
			return numeric;
		}
		return trimmed;
	}

	private _createInjectionScriptContext(
		preprocessContext: PreprocessContext,
		source: string
	): ShaderInjectionScriptContext {
		return {
			...preprocessContext.contextTemplate,
			source,
		};
	}
}
