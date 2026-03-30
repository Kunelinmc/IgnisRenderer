import assert from "node:assert/strict";
import {
	assertShaderDirectiveProfileRegistryComplete,
	DEFAULT_SHADER_DIRECTIVE_PROFILE_REGISTRY,
	ShaderBackendCompileStage,
	ShaderRuntime,
} from "../src/shaders/runtime/index.ts";

function createStage(options = {}) {
	const runtime = options.runtime ?? new ShaderRuntime({ mode: "warn" });
	const warnings = [];
	const stage = new ShaderBackendCompileStage({
		backend: "webgl",
		runtime,
		profiles:
			options.profiles ?? DEFAULT_SHADER_DIRECTIVE_PROFILE_REGISTRY,
		hook: options.hook ?? null,
		mode: options.mode ?? "warn",
		warn:
			options.warn ??
			((key, message) => {
				warnings.push({ key, message });
			}),
	});
	return { stage, runtime, warnings };
}

function compileSceneVertex(stage, code, extra = {}) {
	return stage.compile({
		code,
		language: "glsl",
		stage: "vertex",
		entryPoint: "main",
		label: extra.label ?? "DirectivePipelineV2",
		sourceKind: extra.sourceKind ?? "builtin-scene",
		directiveSourcePath:
			extra.directiveSourcePath ?? "./parts/testDirectivePipeline.glsl",
	});
}

function testProfileCompletenessRequiresSoftwareProfile() {
	const incompleteProfiles = {
		...DEFAULT_SHADER_DIRECTIVE_PROFILE_REGISTRY,
	};
	delete incompleteProfiles.software;
	assert.throws(
		() => assertShaderDirectiveProfileRegistryComplete(incompleteProfiles),
		/software/
	);
	assert.throws(
		() =>
			createStage({
				profiles: incompleteProfiles,
			}),
		/software/
	);
}

function testStageABBoundaryNoDuplicateDirectiveDiagnostics() {
	const { stage } = createStage();
	const result = compileSceneVertex(
		stage,
		`#version 300 es
precision highp float;
#inject <missing-script>()
void main() {
	gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
}`
	);
	const directiveErrors = result.directiveDiagnostics.filter(
		(diagnostic) => diagnostic.code === "directive-inject-not-found"
	);
	assert.equal(directiveErrors.length, 1);
	assert.equal(
		result.backendDiagnostics.some((diagnostic) =>
			diagnostic.code.startsWith("directive-")
		),
		false
	);
	assert.equal(
		result.diagnostics.filter(
			(diagnostic) => diagnostic.code === "directive-inject-not-found"
		).length,
		1
	);
}

function testHookMissingTokenStrictWarnBehavior() {
	const code = `#version 300 es
precision highp float;
#import <hook/dynamic>
void main() {
	float value = hookDynamic();
	gl_Position = vec4(value, 0.0, 0.0, 1.0);
}`;

	const warnStage = createStage({
		mode: "warn",
		hook() {
			return {
				includeModules: [
					{
						language: "glsl",
						id: "hook/dynamic.glsl",
						code: "float hookDynamic() { return 1.0; }",
					},
				],
			};
		},
	});
	const warnResult = compileSceneVertex(warnStage.stage, code, {
		label: "MissingTokenWarn",
	});
	assert.ok(
		warnStage.warnings.some((warning) =>
			warning.key.includes("hook-token-invalid")
		)
	);
	assert.ok(
		warnResult.directiveDiagnostics.some(
			(diagnostic) => diagnostic.code === "directive-include-not-found"
		)
	);

	const strictStage = createStage({
		mode: "strict",
		hook() {
			return {
				includeModules: [
					{
						language: "glsl",
						id: "hook/dynamic.glsl",
						code: "float hookDynamic() { return 1.0; }",
					},
				],
			};
		},
	});
	assert.throws(
		() =>
			compileSceneVertex(strictStage.stage, code, {
				label: "MissingTokenStrict",
			}),
		/token/i
	);
}

function testHookTokenCollisionFallsBackToBasePatch() {
	const { stage, warnings } = createStage({
		hook(context) {
			if (context.label === "CollisionA") {
				return {
					token: "collision-token",
					includeModules: [
						{
							language: "glsl",
							id: "collision/mod.glsl",
							code: "float collisionValue() { return 1.0; }",
						},
					],
				};
			}
			return {
				token: "collision-token",
				includeModules: [
					{
						language: "glsl",
						id: "collision/mod.glsl",
						code: "float collisionValue() { return 2.0; }",
					},
				],
			};
		},
	});

	const code = `#version 300 es
precision highp float;
#import <collision/mod>
void main() {
	float value = collisionValue();
	gl_Position = vec4(value, 0.0, 0.0, 1.0);
}`;
	const first = compileSceneVertex(stage, code, {
		label: "CollisionA",
	});
	assert.equal(
		first.directiveDiagnostics.some(
			(diagnostic) => diagnostic.code === "directive-include-not-found"
		),
		false
	);
	const second = compileSceneVertex(stage, code, {
		label: "CollisionB",
	});
	assert.ok(
		warnings.some((warning) => warning.key.includes("hook-token-collision"))
	);
	assert.ok(
		second.directiveDiagnostics.some(
			(diagnostic) => diagnostic.code === "directive-include-not-found"
		)
	);
}

function testDirectiveFingerprintChangeTriggersCacheMiss() {
	let dynamicToken = "token-a";
	let dynamicValue = "1.0";
	const { stage } = createStage({
		hook() {
			return {
				token: dynamicToken,
				includeModules: [
					{
						language: "glsl",
						id: "dynamic/value.glsl",
						code: `float dynamicValue() { return ${dynamicValue}; }`,
					},
				],
			};
		},
	});
	const code = `#version 300 es
precision highp float;
#import <dynamic/value>
void main() {
	float value = dynamicValue();
	gl_Position = vec4(value, 0.0, 0.0, 1.0);
}`;
	const first = compileSceneVertex(stage, code, {
		label: "FingerprintA",
	});
	const second = compileSceneVertex(stage, code, {
		label: "FingerprintA",
	});
	assert.equal(first.fromCache, false);
	assert.equal(second.fromCache, true);
	assert.equal(first.directiveFingerprint, second.directiveFingerprint);

	dynamicToken = "token-b";
	dynamicValue = "2.0";
	const third = compileSceneVertex(stage, code, {
		label: "FingerprintA",
	});
	assert.equal(third.fromCache, false);
	assert.notEqual(third.directiveFingerprint, first.directiveFingerprint);
	assert.ok(third.code.includes("return 2.0;"));
}

function testAsyncHookFallbackByMode() {
	const code = `#version 300 es
precision highp float;
void main() {
	gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
}`;
	const warnStage = createStage({
		mode: "warn",
		hook: async () => null,
	});
	const warnResult = compileSceneVertex(warnStage.stage, code, {
		label: "AsyncHookWarn",
	});
	assert.equal(warnResult.hasErrors, false);
	assert.ok(
		warnStage.warnings.some((warning) =>
			warning.key.includes("hook-async-sync-path")
		)
	);

	const strictStage = createStage({
		mode: "strict",
		hook: async () => null,
	});
	assert.throws(
		() =>
			compileSceneVertex(strictStage.stage, code, {
				label: "AsyncHookStrict",
			}),
		/Promise/
	);
}

function run() {
	testProfileCompletenessRequiresSoftwareProfile();
	testStageABBoundaryNoDuplicateDirectiveDiagnostics();
	testHookMissingTokenStrictWarnBehavior();
	testHookTokenCollisionFallsBackToBasePatch();
	testDirectiveFingerprintChangeTriggersCacheMiss();
	testAsyncHookFallbackByMode();
	console.log("Shader directive pipeline v2 tests passed");
}

run();
