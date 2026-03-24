import assert from "node:assert/strict";
import {
	SHADER_RUNTIME_RESERVED_RULE_PREFIX,
	ShaderRuntime,
} from "../src/renderers/shaders/index.ts";

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
	assert.ok(
		processed.diagnostics.some(
			(diagnostic) => diagnostic.code === "placeholder-not-resolved"
		)
	);

	const unbalanced = runtime.process({
		code: "void main() {",
		language: "glsl",
		stage: "fragment",
		entryPoint: "main",
		label: "BracketBalanceTest",
		sourceKind: "custom-material",
	});
	assert.equal(unbalanced.hasErrors, true);
	assert.ok(
		unbalanced.diagnostics.some(
			(diagnostic) => diagnostic.code === "unbalanced-brackets"
		)
	);
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

function run() {
	testReservedPrefixProtection();
	testGLSLInjectionOrderAndLocation();
	testWGSLInjectionLocation();
	testBuiltInValidationRules();
	testWGSLAndGLSLEntryPointChecks();
	testCacheAndRevisionInvalidation();
	testReservedSymbolConflictByMode();
	console.log("ShaderRuntime tests passed");
}

run();
