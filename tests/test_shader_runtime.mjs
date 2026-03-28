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

const GLSL_EXTENDED_ANCHOR_SOURCE = `#version 300 es
precision highp float;
#define USE_LIGHTING 1
struct LightData {
	vec3 direction;
};
uniform vec4 uColor;
void main() {
	gl_Position = vec4(0.0);
}
`;

const WGSL_ANCHOR_SOURCE = `enable f16;
alias Scalar = f32;
struct Payload {
	value: f32,
};
@group(0) @binding(0) var<uniform> uniforms: Payload;

@vertex
fn vsMain() -> @builtin(position) vec4<f32> {
	return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
`;

function hashStringFNV1a(value) {
	let hash = 0x811c9dc5;
	for (let i = 0; i < value.length; i++) {
		hash ^= value.charCodeAt(i);
		hash = Math.imul(hash, 0x01000193);
	}
	return (hash >>> 0).toString(16);
}

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

function testProcessRejectsAsyncRuleInSyncPath() {
	const runtime = new ShaderRuntime({ mode: "warn" });
	runtime.registerRule({
		id: "user/async-sync-reject",
		async validate() {
			return [];
		},
	});
	assert.throws(
		() =>
			runtime.process({
				code: GLSL_SOURCE,
				language: "glsl",
				stage: "vertex",
				entryPoint: "main",
				label: "AsyncSyncReject",
				sourceKind: "custom-material",
			}),
		/processAsync/
	);
}

async function testProcessAsyncSupportsAsyncRulesAndInFlightDedup() {
	const runtime = new ShaderRuntime({ mode: "warn" });
	let injectCallCount = 0;
	runtime.registerRule({
		id: "user/async-inject",
		async inject() {
			injectCallCount++;
			await new Promise((resolve) => setTimeout(resolve, 10));
			return {
				header: "// async injected",
			};
		},
	});
	const request = {
		code: GLSL_SOURCE,
		language: "glsl",
		stage: "vertex",
		entryPoint: "main",
		label: "AsyncDedup",
		sourceKind: "custom-material",
	};
	const [first, second] = await Promise.all([
		runtime.processAsync(request),
		runtime.processAsync(request),
	]);
	assert.equal(first.code.includes("// async injected"), true);
	assert.equal(second.code.includes("// async injected"), true);
	assert.equal(first.fromCache, false);
	assert.equal(second.fromCache, true);
	assert.equal(injectCallCount, 1);
	const third = await runtime.processAsync(request);
	assert.equal(third.fromCache, true);
}

function testRuleDependenciesAndOrdering() {
	const runtime = new ShaderRuntime({ mode: "warn" });
	runtime.registerRule({
		id: "user/base",
		priority: 1,
	});
	runtime.registerRule({
		id: "user/middle",
		priority: 999,
		dependsOn: ["user/base"],
	});
	runtime.registerRule({
		id: "user/top",
		priority: -5,
		dependsOn: ["user/middle"],
	});
	const orderedIds = runtime.listRules().map((rule) => rule.id);
	assert.ok(orderedIds.indexOf("user/base") < orderedIds.indexOf("user/middle"));
	assert.ok(orderedIds.indexOf("user/middle") < orderedIds.indexOf("user/top"));

	assert.throws(
		() =>
			runtime.registerRule({
				id: "user/missing-dependency",
				dependsOn: ["user/not-exist"],
			}),
		/missing rule/
	);

	const cycleRuntime = new ShaderRuntime({ mode: "warn" });
	cycleRuntime.registerRule({
		id: "user/cycle-a",
	});
	cycleRuntime.registerRule({
		id: "user/cycle-b",
		dependsOn: ["user/cycle-a"],
	});
	assert.throws(
		() =>
			cycleRuntime.registerRule({
				id: "user/cycle-a",
				dependsOn: ["user/cycle-b"],
			}),
		/cycle/i
	);
}

function testUserSymbolConflictsStaticAndDynamic() {
	const staticRuntime = new ShaderRuntime({ mode: "warn" });
	staticRuntime.registerRule({
		id: "user/static-a",
		symbols: ["sharedSymbol"],
	});
	assert.throws(
		() =>
			staticRuntime.registerRule({
				id: "user/static-b",
				symbols: ["sharedSymbol"],
			}),
		/conflicts/
	);

	const dynamicRuntime = new ShaderRuntime({ mode: "warn" });
	dynamicRuntime.registerRule({
		id: "user/owner",
		symbols: ["ownedSymbol"],
	});
	dynamicRuntime.registerRule({
		id: "user/conflict",
		inject() {
			return {
				header: "// should be skipped",
				symbols: ["ownedSymbol"],
			};
		},
	});
	const result = dynamicRuntime.process({
		code: GLSL_SOURCE,
		language: "glsl",
		stage: "vertex",
		entryPoint: "main",
		label: "DynamicSymbolConflict",
		sourceKind: "custom-material",
	});
	assert.equal(result.code.includes("// should be skipped"), false);
	assert.ok(
		result.diagnostics.some(
			(diagnostic) => diagnostic.code === "user-symbol-conflict"
		)
	);
}

function testExtendedAnchorsAndDryRun() {
	const runtime = new ShaderRuntime({ mode: "warn" });
	runtime.registerRule({
		id: "user/after-defines",
		inject() {
			return {
				header: "// after defines",
				headerAnchor: "afterDefines",
			};
		},
	});
	runtime.registerRule({
		id: "user/after-struct",
		inject() {
			return {
				header: "// after struct",
				headerAnchor: "afterStruct",
			};
		},
	});
	runtime.registerRule({
		id: "user/after-uniforms",
		inject() {
			return {
				header: "// after uniforms",
				headerAnchor: "afterUniforms",
			};
		},
	});
	const glsl = runtime.process({
		code: GLSL_EXTENDED_ANCHOR_SOURCE,
		language: "glsl",
		stage: "vertex",
		entryPoint: "main",
		label: "GLSLExtendedAnchors",
		sourceKind: "custom-material",
	});
	assert.ok(glsl.code.includes("#define USE_LIGHTING 1\n// after defines"));
	assert.ok(glsl.code.includes("};\n// after struct"));
	assert.ok(glsl.code.includes("uniform vec4 uColor;\n// after uniforms"));
	const glslAnchors = runtime.resolveInjectionAnchors({
		code: GLSL_EXTENDED_ANCHOR_SOURCE,
		language: "glsl",
	});
	assert.ok(glslAnchors.anchors.afterDefines >= glslAnchors.anchors.afterPrecision);
	assert.ok(glslAnchors.anchors.afterUniforms >= glslAnchors.anchors.afterStruct);

	const wgslRuntime = new ShaderRuntime({ mode: "warn" });
	wgslRuntime.registerRule({
		id: "user/wgsl-anchor",
		inject() {
			return {
				header: "// wgsl after bindings",
				headerAnchor: "afterBindings",
			};
		},
	});
	const wgsl = wgslRuntime.process({
		code: WGSL_ANCHOR_SOURCE,
		language: "wgsl",
		stage: "vertex",
		entryPoint: "vsMain",
		label: "WGSLAnchors",
		sourceKind: "custom-material",
	});
	assert.ok(wgsl.code.includes("@group(0) @binding(0) var<uniform> uniforms: Payload;\n// wgsl after bindings"));
	const wgslAnchors = wgslRuntime.resolveInjectionAnchors({
		code: WGSL_ANCHOR_SOURCE,
		language: "wgsl",
	});
	assert.ok(wgslAnchors.anchors.afterAliases >= wgslAnchors.anchors.afterEnable);
	assert.ok(
		wgslAnchors.anchors.afterBindings >= wgslAnchors.anchors.afterStruct
	);
}

function testSilentModeAndDiagnosticFilters() {
	const silentRuntime = new ShaderRuntime({ mode: "silent" });
	silentRuntime.registerRule({
		id: "user/silent-diagnostic",
		validate() {
			return [
				{
					ruleId: "user/silent-diagnostic",
					code: "silent-error",
					severity: "error",
					message: "silent mode should suppress this",
				},
			];
		},
	});
	const silentResult = silentRuntime.process({
		code: GLSL_SOURCE,
		language: "glsl",
		stage: "vertex",
		entryPoint: "main",
		label: "SilentMode",
		sourceKind: "custom-material",
	});
	assert.equal(silentResult.diagnostics.length, 0);
	assert.equal(silentResult.hasErrors, false);

	const runtime = new ShaderRuntime({ mode: "warn" });
	runtime.registerRule({
		id: "user/filter-test",
		validate() {
			return [
				{
					ruleId: "user/filter-test",
					code: "filter-a",
					severity: "error",
					message: "a",
				},
				{
					ruleId: "user/filter-test",
					code: "filter-b",
					severity: "error",
					message: "b",
				},
			];
		},
	});
	const disposeGlobalFilter = runtime.filterDiagnostics(
		(diagnostic) => diagnostic.code !== "filter-a"
	);
	const filtered = runtime.process({
		code: GLSL_SOURCE,
		language: "glsl",
		stage: "vertex",
		entryPoint: "main",
		label: "FilterLayering",
		sourceKind: "custom-material",
		diagnosticFilter: (diagnostic) => diagnostic.code !== "filter-b",
	});
	assert.equal(filtered.diagnostics.length, 0);
	assert.equal(filtered.hasErrors, false);
	disposeGlobalFilter();
	const afterDispose = runtime.process({
		code: GLSL_SOURCE,
		language: "glsl",
		stage: "vertex",
		entryPoint: "main",
		label: "FilterLayering",
		sourceKind: "custom-material",
	});
	assert.equal(afterDispose.diagnostics.length, 2);
}

function testRuleScopedInvalidationAndCacheStats() {
	const runtime = new ShaderRuntime({ mode: "warn", cacheLimit: 2 });
	runtime.registerRule({
		id: "user/invalidate-a",
		match(context) {
			return context.label === "A";
		},
		inject() {
			return { header: "// A-v1" };
		},
	});
	runtime.registerRule({
		id: "user/invalidate-b",
		match(context) {
			return context.label === "B";
		},
		inject() {
			return { header: "// B-v1" };
		},
	});
	const requestA = {
		code: GLSL_SOURCE,
		language: "glsl",
		stage: "vertex",
		entryPoint: "main",
		label: "A",
		sourceKind: "custom-material",
	};
	const requestB = { ...requestA, label: "B" };
	runtime.process(requestA);
	runtime.process(requestA);
	runtime.process(requestB);
	runtime.process(requestB);

	runtime.registerRule({
		id: "user/invalidate-a",
		match(context) {
			return context.label === "A";
		},
		inject() {
			return { header: "// A-v2" };
		},
	});
	const afterA = runtime.process(requestA);
	const afterB = runtime.process(requestB);
	assert.equal(afterA.fromCache, false);
	assert.equal(afterA.code.includes("// A-v2"), true);
	assert.equal(afterB.fromCache, true);

	runtime.registerRule({
		id: "user/invalidate-c",
		match(context) {
			return context.label === "C";
		},
		inject() {
			return { header: "// C-v1" };
		},
	});
	const afterBWithNewRule = runtime.process(requestB);
	assert.equal(afterBWithNewRule.fromCache, true);
	const requestC = { ...requestA, label: "C" };
	const afterC = runtime.process(requestC);
	assert.equal(afterC.fromCache, false);
	assert.equal(afterC.code.includes("// C-v1"), true);

	const stats = runtime.getCacheStats("sync");
	assert.ok(stats.hits > 0);
	assert.ok(stats.misses > 0);
	assert.ok(stats.invalidations > 0);

	runtime.resetCacheStats("sync");
	const resetStats = runtime.getCacheStats("sync");
	assert.equal(resetStats.hits, 0);
	assert.equal(resetStats.misses, 0);
	assert.equal(resetStats.evictions, 0);
	assert.equal(resetStats.invalidations, 0);
}

function testChangeEventsAndSourceHashValidation() {
	const runtime = new ShaderRuntime({ mode: "warn" });
	const events = [];
	let legacyCalls = 0;
	runtime.onDidChange((event) => events.push(event));
	runtime.onDidChange(() => {
		legacyCalls++;
	});
	runtime.registerRule({ id: "user/event-rule" });
	const event = events[events.length - 1];
	assert.equal(event.action, "register-rule");
	assert.deepEqual(event.ruleIds, ["user/event-rule"]);
	assert.ok(event.revision >= 2);
	assert.equal(legacyCalls > 0, true);

	const hash = hashStringFNV1a(GLSL_SOURCE);
	const withHash = runtime.process({
		code: GLSL_SOURCE,
		language: "glsl",
		stage: "vertex",
		entryPoint: "main",
		label: "HashValid",
		sourceKind: "custom-material",
		sourceHash: hash,
	});
	assert.equal(withHash.hasErrors, false);
	assert.throws(
		() =>
			runtime.process({
				code: GLSL_SOURCE,
				language: "glsl",
				stage: "vertex",
				entryPoint: "main",
				label: "HashInvalid",
				sourceKind: "custom-material",
				sourceHash: "deadbeef",
			}),
		/sourceHash mismatch/
	);
}

async function run() {
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
	testProcessRejectsAsyncRuleInSyncPath();
	await testProcessAsyncSupportsAsyncRulesAndInFlightDedup();
	testRuleDependenciesAndOrdering();
	testUserSymbolConflictsStaticAndDynamic();
	testExtendedAnchorsAndDryRun();
	testSilentModeAndDiagnosticFilters();
	testRuleScopedInvalidationAndCacheStats();
	testChangeEventsAndSourceHashValidation();
	console.log("ShaderRuntime tests passed");
}

run().catch((error) => {
	console.error(error);
	process.exit(1);
});
