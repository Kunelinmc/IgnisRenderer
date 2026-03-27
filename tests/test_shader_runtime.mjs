import assert from "node:assert/strict";
import {
	createInlineShaderSourceMap,
	mapShaderCompilerMessages,
	parseWebGLShaderInfoLog,
	SHADER_RUNTIME_RESERVED_RULE_PREFIX,
	ShaderRuntime,
} from "../src/shaders/runtime/index.ts";

const GLSL_SOURCE = `#version 300 es
precision highp float;
void main() {
	gl_Position = vec4(0.0);
}
`;

const WGSL_SOURCE = `
@vertex
fn vsMain() -> @builtin(position) vec4<f32> {
	return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
`;

function testReservedPrefixProtection() {
	const runtime = new ShaderRuntime({ mode: "warn" });
	assert.throws(
		() =>
			runtime.registerRule({
				id: `${SHADER_RUNTIME_RESERVED_RULE_PREFIX}user-rule`,
			}),
		/reserved prefix/
	);
}

function testGLSLInjectionOrderAndLocation() {
	const runtime = new ShaderRuntime({ mode: "warn" });
	runtime.registerRule({
		id: "user/low-priority",
		priority: 1,
		inject() {
			return {
				header: "// low header",
				functions: "float lowFn() { return 1.0; }",
			};
		},
	});
	runtime.registerRule({
		id: "user/high-priority",
		priority: 10,
		inject() {
			return {
				header: "// high header",
				functions: "float highFn() { return 2.0; }",
			};
		},
	});

	const processed = runtime.process({
		code: GLSL_SOURCE,
		language: "glsl",
		stage: "vertex",
		entryPoint: "main",
		label: "GLSLInjectionTest",
		sourceKind: "custom-material",
	});
	assert.equal(processed.hasErrors, false);
	assert.equal(processed.fromCache, false);
	assert.ok(processed.code.includes("#version 300 es\n// high header"));
	assert.ok(
		processed.code.indexOf("// high header") <
			processed.code.indexOf("// low header")
	);
	assert.ok(
		processed.code.indexOf("float highFn") <
			processed.code.indexOf("float lowFn")
	);
}

function testWGSLInjectionLocation() {
	const runtime = new ShaderRuntime({ mode: "warn" });
	runtime.registerRule({
		id: "user/wgsl-header",
		priority: 2,
		inject() {
			return {
				header: "const USER_DEFINE: f32 = 1.0;",
			};
		},
	});

	const processed = runtime.process({
		code: WGSL_SOURCE,
		language: "wgsl",
		stage: "vertex",
		entryPoint: "vsMain",
		label: "WGSLInjectionTest",
		sourceKind: "custom-material",
	});
	assert.equal(processed.hasErrors, false);
	assert.ok(processed.code.trimStart().startsWith("const USER_DEFINE"));
	assert.ok(processed.code.includes("@vertex"));
}

function testGLSLInjectionAnchors() {
	const runtime = new ShaderRuntime({ mode: "warn" });
	runtime.registerRule({
		id: "user/after-precision",
		priority: 10,
		inject() {
			return {
				header: "// after precision anchor",
				headerAnchor: "afterPrecision",
			};
		},
	});
	runtime.registerRule({
		id: "user/before-entry-point",
		priority: 9,
		inject() {
			return {
				functions: "float beforeMainFn() { return 42.0; }",
				functionsAnchor: "beforeEntryPoint",
			};
		},
	});
	runtime.registerRule({
		id: "user/end-of-file",
		priority: 8,
		inject() {
			return {
				functions: "// end of file anchor",
				functionsAnchor: "endOfFile",
			};
		},
	});
	const processed = runtime.process({
		code: GLSL_SOURCE,
		language: "glsl",
		stage: "vertex",
		entryPoint: "main",
		label: "GLSLAnchors",
		sourceKind: "custom-material",
	});
	assert.equal(processed.hasErrors, false);
	assert.ok(
		processed.code.includes(
			"precision highp float;\n// after precision anchor"
		)
	);
	assert.ok(
		processed.code.indexOf("float beforeMainFn") <
			processed.code.indexOf("void main()")
	);
	assert.ok(processed.code.trimEnd().endsWith("// end of file anchor"));
}

function testBuiltInValidationRules() {
	const runtime = new ShaderRuntime({ mode: "warn" });
	const processed = runtime.process({
		code: "void main() { return; } __UNRESOLVED__",
		language: "glsl",
		stage: "fragment",
		entryPoint: "main",
		label: "PlaceholderTest",
		sourceKind: "custom-material",
	});
	assert.equal(processed.hasErrors, true);
	const placeholderDiagnostic = processed.diagnostics.find(
		(diagnostic) => diagnostic.code === "placeholder-not-resolved"
	);
	assert.ok(placeholderDiagnostic);
	assert.equal(placeholderDiagnostic.sourcePath, "PlaceholderTest");
	assert.equal(placeholderDiagnostic.line, 1);
	assert.ok((placeholderDiagnostic.column ?? 0) > 1);
	assert.ok(placeholderDiagnostic.range);

	const unbalanced = runtime.process({
		code: "void main() {",
		language: "glsl",
		stage: "fragment",
		entryPoint: "main",
		label: "BracketBalanceTest",
		sourceKind: "custom-material",
	});
	assert.equal(unbalanced.hasErrors, true);
	const unbalancedDiagnostic = unbalanced.diagnostics.find(
		(diagnostic) => diagnostic.code === "unbalanced-brackets"
	);
	assert.ok(unbalancedDiagnostic);
	assert.equal(unbalancedDiagnostic.sourcePath, "BracketBalanceTest");
	assert.equal(unbalancedDiagnostic.line, 1);
	assert.ok((unbalancedDiagnostic.column ?? 0) > 1);
	assert.ok(unbalancedDiagnostic.range);
}

function testWGSLAndGLSLEntryPointChecks() {
	const runtime = new ShaderRuntime({ mode: "warn" });
	const wgslMissingEntry = runtime.process({
		code: WGSL_SOURCE,
		language: "wgsl",
		stage: "vertex",
		entryPoint: "unknownEntry",
		label: "WGSLEntryPointTest",
		sourceKind: "custom-material",
	});
	assert.equal(wgslMissingEntry.hasErrors, true);
	assert.ok(
		wgslMissingEntry.diagnostics.some(
			(diagnostic) => diagnostic.code === "missing-entry-point"
		)
	);

	const glslMissingMain = runtime.process({
		code: "void helper() {}",
		language: "glsl",
		stage: "vertex",
		entryPoint: "main",
		label: "GLSLEntryPointTest",
		sourceKind: "custom-material",
	});
	assert.equal(glslMissingMain.hasErrors, true);
	assert.ok(
		glslMissingMain.diagnostics.some(
			(diagnostic) => diagnostic.code === "missing-main"
		)
	);
}

function testCacheAndRevisionInvalidation() {
	const runtime = new ShaderRuntime({ mode: "warn" });
	const request = {
		code: GLSL_SOURCE,
		language: "glsl",
		stage: "vertex",
		entryPoint: "main",
		label: "CacheTest",
		sourceKind: "custom-material",
	};

	const first = runtime.process(request);
	const second = runtime.process(request);
	assert.equal(first.fromCache, false);
	assert.equal(second.fromCache, true);

	runtime.registerRule({
		id: "user/revision-bump",
		priority: 3,
		inject() {
			return {
				header: "// revision bump",
			};
		},
	});

	const third = runtime.process(request);
	assert.equal(third.fromCache, false);
	assert.ok(third.code.includes("// revision bump"));
}

function testCacheKeyIncludesLabel() {
	const runtime = new ShaderRuntime({ mode: "warn" });
	runtime.registerRule({
		id: "user/label-sensitive",
		priority: 10,
		inject(context) {
			if (context.label === "LabelA") {
				return { header: "// LabelA only" };
			}
			return { header: "// LabelB only" };
		},
	});
	const requestA = {
		code: GLSL_SOURCE,
		language: "glsl",
		stage: "vertex",
		entryPoint: "main",
		label: "LabelA",
		sourceKind: "custom-material",
	};
	const requestB = {
		...requestA,
		label: "LabelB",
	};
	const resultA = runtime.process(requestA);
	assert.equal(resultA.fromCache, false);
	assert.equal(resultA.code.includes("// LabelA only"), true);
	const resultB = runtime.process(requestB);
	assert.equal(resultB.fromCache, false);
	assert.equal(resultB.code.includes("// LabelB only"), true);
	assert.equal(resultB.code.includes("// LabelA only"), false);
	const cachedB = runtime.process(requestB);
	assert.equal(cachedB.fromCache, true);
}

function testManualCacheInvalidation() {
	const runtime = new ShaderRuntime({ mode: "warn" });
	const request = {
		code: GLSL_SOURCE,
		language: "glsl",
		stage: "vertex",
		entryPoint: "main",
		label: "ManualCacheInvalidation",
		sourceKind: "custom-material",
	};
	runtime.process(request);
	const cached = runtime.process(request);
	assert.equal(cached.fromCache, true);
	const clearedCount = runtime.invalidateProcessCache();
	assert.ok(clearedCount > 0);
	const afterInvalidation = runtime.process(request);
	assert.equal(afterInvalidation.fromCache, false);
}

function testCachedResultDefensiveCopy() {
	const runtime = new ShaderRuntime({ mode: "warn" });
	const request = {
		code: GLSL_SOURCE,
		language: "glsl",
		stage: "vertex",
		entryPoint: "main",
		label: "CacheIsolation",
		sourceKind: "custom-material",
	};
	runtime.process(request);
	const cached = runtime.process(request);
	cached.diagnostics.push({
		ruleId: "mutated",
		code: "mutated",
		severity: "error",
		message: "mutated",
	});
	cached.sourceMap.segments.push({
		generatedLineStart: 999,
		generatedLineEnd: 999,
		sourcePath: "mutated",
		sourceLineStart: 999,
		sourceLineEnd: 999,
		kind: "generated",
	});
	const third = runtime.process(request);
	assert.equal(third.fromCache, true);
	assert.equal(third.diagnostics.length, 0);
	assert.equal(
		third.sourceMap.segments.some((segment) => segment.sourcePath === "mutated"),
		false
	);
}

function testReservedSymbolConflictByMode() {
	const warnRuntime = new ShaderRuntime({ mode: "warn" });
	warnRuntime.registerRule({
		id: "user/reserved-symbol-warn",
		symbols: ["ignis_runtime_reserved_symbol"],
		inject() {
			return {
				header: "// should not be injected",
			};
		},
	});
	const warnResult = warnRuntime.process({
		code: GLSL_SOURCE,
		language: "glsl",
		stage: "vertex",
		entryPoint: "main",
		label: "ReservedSymbolWarn",
		sourceKind: "custom-material",
	});
	assert.equal(warnResult.hasErrors, false);
	assert.ok(
		warnResult.diagnostics.some(
			(diagnostic) => diagnostic.code === "reserved-symbol-conflict"
		)
	);
	assert.equal(warnResult.code.includes("// should not be injected"), false);

	const strictRuntime = new ShaderRuntime({ mode: "strict" });
	strictRuntime.registerRule({
		id: "user/reserved-symbol-strict",
		symbols: ["ignis_runtime_reserved_symbol"],
		inject() {
			return {
				header: "// strict should throw",
			};
		},
	});
	assert.throws(
		() =>
			strictRuntime.process({
				code: GLSL_SOURCE,
				language: "glsl",
				stage: "vertex",
				entryPoint: "main",
				label: "ReservedSymbolStrict",
				sourceKind: "custom-material",
			}),
		/ShaderRuntime validation failed/
	);
}

function testProcessCarriesSourceMap() {
	const runtime = new ShaderRuntime({ mode: "warn" });
	const sourceMap = createInlineShaderSourceMap(
		GLSL_SOURCE,
		"./parts/testShader.glsl",
		"source"
	);
	const processed = runtime.process({
		code: GLSL_SOURCE,
		language: "glsl",
		stage: "vertex",
		entryPoint: "main",
		label: "SourceMapCarry",
		sourceKind: "custom-material",
		sourceMap,
	});
	assert.ok(processed.sourceMap.segments.length > 0);
	const mapped = mapShaderCompilerMessages(
		[{ type: "error", message: "synthetic", line: 2, column: 1 }],
		processed.code,
		processed.sourceMap
	);
	assert.equal(mapped[0].sourcePath, "./parts/testShader.glsl");
	assert.equal(mapped[0].sourceLine, 2);
}

function testDiagnosticsMapToSourceWithRange() {
	const runtime = new ShaderRuntime({ mode: "warn" });
	const source = `#version 300 es
precision highp float;
__UNRESOLVED__;
void main() {
	gl_Position = vec4(0.0);
}
`;
	const sourceMap = createInlineShaderSourceMap(
		source,
		"./parts/diagnosticShader.glsl",
		"source"
	);
	const processed = runtime.process({
		code: source,
		language: "glsl",
		stage: "vertex",
		entryPoint: "main",
		label: "DiagnosticMapping",
		sourceKind: "custom-material",
		sourceMap,
	});
	const diagnostic = processed.diagnostics.find(
		(entry) => entry.code === "placeholder-not-resolved"
	);
	assert.ok(diagnostic);
	assert.equal(diagnostic.sourcePath, "./parts/diagnosticShader.glsl");
	assert.equal(diagnostic.line, 3);
	assert.equal(diagnostic.column, 1);
	assert.ok(diagnostic.range);
	assert.equal(diagnostic.range.start.line, 3);
	assert.equal(diagnostic.range.start.column, 1);
	assert.equal(diagnostic.range.end.line, 3);
}

function testWebGLInfoLogParsing() {
	const parsed = parseWebGLShaderInfoLog(
		`ERROR: 0:12: syntax error\n0(8) : warning C0000: dead code`
	);
	assert.equal(parsed.length, 2);
	assert.equal(parsed[0].type, "error");
	assert.equal(parsed[0].line, 12);
	assert.equal(parsed[1].type, "warning");
	assert.equal(parsed[1].line, 8);
}

function run() {
	testReservedPrefixProtection();
	testGLSLInjectionOrderAndLocation();
	testWGSLInjectionLocation();
	testGLSLInjectionAnchors();
	testBuiltInValidationRules();
	testWGSLAndGLSLEntryPointChecks();
	testCacheAndRevisionInvalidation();
	testCacheKeyIncludesLabel();
	testManualCacheInvalidation();
	testCachedResultDefensiveCopy();
	testReservedSymbolConflictByMode();
	testProcessCarriesSourceMap();
	testDiagnosticsMapToSourceWithRange();
	testWebGLInfoLogParsing();
	console.log("ShaderRuntime tests passed");
}

run();
