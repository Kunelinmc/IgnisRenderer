import assert from "node:assert/strict";
import {
	composeShaderDirectiveProfile,
	createInlineShaderSourceMap,
	SOURCE_MAP_SCHEMA_VERSION,
	ShaderBackendCompileStage,
	ShaderRuntime,
} from "../../../src/shaders/runtime/index.ts";
import { WEBGL_TEST_PROFILE } from "./shaderDirectiveTestProfiles.mjs";
import { Logger } from "../../../src/foundation/Logger.ts";
import {
	MAX_DIRECTIONAL_LIGHTS,
	MAX_LOCAL_LIGHT_PROBES,
	MAX_POINT_LIGHTS,
	MAX_REFLECTION_PROBES,
	MAX_SPOT_LIGHTS,
} from "../../../src/backends/constants.ts";

function createStage(options = {}) {
	const runtime = options.runtime ?? new ShaderRuntime({ mode: "warn" });
	const warnings = [];
	Logger.configure({
		level: "warn",
		sink: {
			warn: (...args) => {
				const rendered = args
					.map((arg) => (typeof arg === "string" ? arg : JSON.stringify(arg)))
					.join(" ");
				warnings.push(rendered);
			},
		},
		resetOnceKeys: true,
	});
	const stage = new ShaderBackendCompileStage({
		runtime,
		profile: options.profile ?? WEBGL_TEST_PROFILE,
		hook: options.hook ?? null,
		mode: options.mode ?? "warn",
	});
	return { stage, runtime, warnings };
}

function compileSceneVertex(stage, code, extra = {}) {
	return stage.compile({
		code,
		sourceMap: extra.sourceMap ?? null,
		language: "glsl",
		stage: "vertex",
		entryPoint: "main",
		label: extra.label ?? "DirectivePipelineV2",
		sourceKind: extra.sourceKind ?? "builtin-scene",
		directiveSourcePath:
			extra.directiveSourcePath ?? "./parts/testDirectivePipeline.glsl",
		generatedSourceBlocks: extra.generatedSourceBlocks,
	});
}

function compileSceneVertexAsync(stage, code, extra = {}) {
	return stage.compileAsync({
		code,
		sourceMap: extra.sourceMap ?? null,
		language: "glsl",
		stage: "vertex",
		entryPoint: "main",
		label: extra.label ?? "DirectivePipelineV2Async",
		sourceKind: extra.sourceKind ?? "builtin-scene",
		directiveSourcePath:
			extra.directiveSourcePath ?? "./parts/testDirectivePipelineAsync.glsl",
		generatedSourceBlocks: extra.generatedSourceBlocks,
	});
}

function testProfileCompositionValidationAndFingerprint() {
	const base = {
		id: "test/base",
		backend: "webgl",
		revision: 1,
		includeModules: [],
	};
	const overlay = {
		id: "test/overlay",
		backend: "webgl",
		includeModules: [],
	};
	const first = composeShaderDirectiveProfile(base, overlay);
	const second = composeShaderDirectiveProfile(base, overlay);
	assert.equal(first.fingerprint, second.fingerprint);
	const changed = composeShaderDirectiveProfile(base, {
		...overlay,
		includeModules: [
			{
				language: "glsl",
				id: "test/module",
				code: "const float TEST_VALUE = 1.0;",
			},
		],
	});
	assert.notEqual(first.fingerprint, changed.fingerprint);
	assert.throws(
		() =>
			composeShaderDirectiveProfile(base, {
				...overlay,
				backend: "webgpu",
			}),
		/does not match/
	);
	assert.throws(
		() =>
			composeShaderDirectiveProfile(
				{
					...base,
					includeModules: [
						{ language: "glsl", id: "same", code: "a" },
						{ language: "glsl", id: "same", code: "b" },
					],
				},
				overlay,
			),
		/Duplicate shader directive include module/
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
		warnStage.warnings.some((warning) => warning.includes("hook-token-invalid"))
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
		warnings.some((warning) => warning.includes("hook-token-collision"))
	);
	assert.ok(
		second.directiveDiagnostics.some(
			(diagnostic) => diagnostic.code === "directive-include-not-found"
		)
	);
}

function testHookCannotOverrideProfileModules() {
	const code = `#version 300 es
precision highp float;
#import <ignis/color/srgb>
out vec4 fragColor;
void main() { fragColor = vec4(linearToSrgb(vec3(1.0)), 1.0); }`;
	const warn = createStage({
		hook: () => ({
			token: "override-profile-module",
			includeModules: [
				{
					language: "glsl",
					id: "ignis/color/srgb.glsl",
					code: "vec3 linearToSrgb(vec3 c) { return vec3(0.0); }",
				},
			],
		}),
	});
	const result = warn.stage.compile({
		code,
		language: "glsl",
		stage: "fragment",
		entryPoint: "main",
		directiveSourcePath: "./hookCollision.glsl",
	});
	assert.ok(result.code.includes("1.055"));
	assert.ok(warn.warnings.some((message) => message.includes("hook-profile-collision")));
	const strict = createStage({
		mode: "strict",
		hook: () => ({
			token: "override-profile-module",
			includeModules: [
				{
					language: "glsl",
					id: "ignis/color/srgb.glsl",
					code: "",
				},
			],
		}),
	});
	assert.throws(
		() =>
			strict.stage.compile({
				code,
				language: "glsl",
				stage: "fragment",
				entryPoint: "main",
				directiveSourcePath: "./hookCollisionStrict.glsl",
			}),
		/attempted to replace profile/,
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

function testSourceMapSchemaVersionNormalization() {
	const { stage } = createStage();
	const code = `#version 300 es
precision highp float;
void main() {
	gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
}`;
	const sourceMapV2 = createInlineShaderSourceMap(
		code,
		"./parts/schemaCache.glsl",
		"source"
	);
	const sourceMapV1 = {
		...sourceMapV2,
		schemaVersion: 1,
		segments: sourceMapV2.segments.map((segment) => ({ ...segment })),
	};
	const first = compileSceneVertex(stage, code, {
		label: "SchemaVersionCache",
		sourceMap: sourceMapV1,
	});
	const second = compileSceneVertex(stage, code, {
		label: "SchemaVersionCache",
		sourceMap: sourceMapV1,
	});
	const third = compileSceneVertex(stage, code, {
		label: "SchemaVersionCache",
		sourceMap: sourceMapV2,
	});
	assert.equal(first.fromCache, false);
	assert.equal(second.fromCache, true);
	assert.equal(third.fromCache, true);
	assert.equal(third.sourceMap.schemaVersion, SOURCE_MAP_SCHEMA_VERSION);
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
		warnStage.warnings.some((warning) => warning.includes("hook-async-sync-path"))
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

function testStaticLumaIncludeExpandsForGLSL() {
	const { stage } = createStage();
	const result = stage.compile({
		code: `#version 300 es
precision highp float;
#import <ignis/postprocess/luma-common>
out vec4 fragColor;
void main() {
	float lum = ignisLuma(
		vec3(1.0, 1.0, 1.0),
		IGNIS_LUMA_WEIGHTS_BT709,
		false
	);
	fragColor = vec4(vec3(lum), 1.0);
}`,
		language: "glsl",
		stage: "fragment",
		entryPoint: "main",
		label: "LumaInjectionGLSL",
		sourceKind: "builtin-scene",
		directiveSourcePath: "./parts/taaFragment.glsl",
	});
	assert.equal(result.hasErrors, false);
	assert.ok(result.code.includes("vec3(0.2126, 0.7152, 0.0722)"));
	assert.equal(result.code.includes("IGNIS_LUMA_WEIGHTS_BT709"), false);
}

async function testGeneratedSourceBlocksComposeBetweenStages() {
	const { stage } = createStage();
	const code = `#version 300 es
precision highp float;
void main() {
	gl_Position = vec4(GENERATED_VALUE, 0.0, 0.0, 1.0);
}`;
	const generatedSourceBlocks = [{
		code: "const float GENERATED_VALUE = 1.0;",
		sourcePath: "<generated:test:value>",
		label: "test-generated-value",
		anchor: "afterPrecision",
	}];
	const syncResult = compileSceneVertex(stage, code, {
		label: "GeneratedSourceBlocks",
		generatedSourceBlocks,
	});
	const asyncResult = await compileSceneVertexAsync(stage, code, {
		label: "GeneratedSourceBlocks",
		generatedSourceBlocks,
	});
	assert.equal(syncResult.code, asyncResult.code);
	assert.ok(syncResult.code.includes("const float GENERATED_VALUE = 1.0;"));
	assert.ok(
		syncResult.sourceMap.segments.some(
			(segment) =>
				segment.kind === "generated" &&
				segment.sourcePath === "<generated:test:value>",
		),
	);
	const changed = compileSceneVertex(stage, code, {
		label: "GeneratedSourceBlocks",
		generatedSourceBlocks: [{
			...generatedSourceBlocks[0],
			code: "const float GENERATED_VALUE = 2.0;",
		}],
	});
	assert.ok(changed.code.includes("const float GENERATED_VALUE = 2.0;"));
	assert.equal(changed.code.includes("GENERATED_VALUE = 1.0"), false);
}

function testWebGLLightLimitConstantsExpandFromRuntimeProfile() {
	const { stage } = createStage({
		mode: "strict",
		runtime: new ShaderRuntime({ mode: "strict" }),
	});
	const result = stage.compile({
		code: `#version 300 es
precision highp float;
#import <ignis/webgl/constants>
const int MAX_DIRECTIONAL_LIGHTS = __WEBGL_MAX_DIRECTIONAL_LIGHTS__;
const int MAX_POINT_LIGHTS = __WEBGL_MAX_POINT_LIGHTS__;
const int MAX_SPOT_LIGHTS = __WEBGL_MAX_SPOT_LIGHTS__;
const int MAX_LOCAL_LIGHT_PROBES = __WEBGL_MAX_LOCAL_LIGHT_PROBES__;
const int MAX_REFLECTION_PROBES = __WEBGL_MAX_REFLECTION_PROBES__;
out vec4 fragColor;
void main() {
	fragColor = vec4(float(
		MAX_DIRECTIONAL_LIGHTS +
		MAX_POINT_LIGHTS +
		MAX_SPOT_LIGHTS +
		MAX_LOCAL_LIGHT_PROBES +
		MAX_REFLECTION_PROBES
	));
}`,
		language: "glsl",
		stage: "fragment",
		entryPoint: "main",
		label: "WebGLLightLimitConstants",
		sourceKind: "builtin-scene",
		directiveSourcePath: "./parts/sceneFragment.glsl",
	});
	assert.equal(result.hasErrors, false);
	assert.ok(
		result.code.includes(
			`const int MAX_DIRECTIONAL_LIGHTS = ${MAX_DIRECTIONAL_LIGHTS};`
		)
	);
	assert.ok(
		result.code.includes(
			`const int MAX_POINT_LIGHTS = ${MAX_POINT_LIGHTS};`
		)
	);
	assert.ok(
		result.code.includes(
			`const int MAX_SPOT_LIGHTS = ${MAX_SPOT_LIGHTS};`
		)
	);
	assert.ok(
		result.code.includes(
			`const int MAX_LOCAL_LIGHT_PROBES = ${MAX_LOCAL_LIGHT_PROBES};`
		)
	);
	assert.ok(
		result.code.includes(
			`const int MAX_REFLECTION_PROBES = ${MAX_REFLECTION_PROBES};`
		)
	);
	assert.equal(result.code.includes("__WEBGL_MAX_DIRECTIONAL_LIGHTS__"), false);
}

function testWebGLAnimationIncludeExpandsFromRuntimeProfile() {
	const { stage } = createStage({
		mode: "strict",
		runtime: new ShaderRuntime({ mode: "strict" }),
	});
	const result = stage.compile({
		code: `#version 300 es
precision highp float;
#import <ignis/webgl/animation>
void main() { gl_Position = vec4(0.0); }`,
		language: "glsl",
		stage: "vertex",
		entryPoint: "main",
		label: "WebGLAnimationABI",
		sourceKind: "custom-material",
		directiveSourcePath: "./custom/animatedVertex.glsl",
	});
	assert.equal(result.hasErrors, false);
	assert.ok(result.code.includes("uniform highp sampler2D uAnimationPayload;"));
	assert.ok(result.code.includes("uniform highp sampler2D uMorphPositionDeltas;"));
	assert.ok(result.code.includes("uniform highp sampler2D uMorphNormalDeltas;"));
	assert.ok(result.code.includes("uniform ivec4 uAnimationCounts;"));
	assert.ok(result.code.includes("uniform ivec4 uAnimationOffsets;"));
	assert.ok(result.code.includes("uniform ivec4 uAnimationTextureWidths;"));
	assert.equal(result.code.includes("uIgnisAnimationPayload"), false);
	assert.ok(result.code.includes("IgnisAnimationVertex ignisApplyAnimationVertex("));
	assert.ok(result.code.includes("vec3 ignisApplyAnimationPosition("));
}

async function testCompileAsyncUsesAsyncDirectivePreprocessPath() {
	const { stage } = createStage({
		hook: async () => ({
			token: "async-directive-patch",
			injectionScripts: [
				{
					id: "hook/async-directive-script",
					language: "glsl",
					arguments: {},
					async run() {
						return {
							header: "const float ASYNC_DIRECTIVE_VALUE = 2.0;",
						};
					},
				},
			],
		}),
	});
	const result = await compileSceneVertexAsync(
		stage,
		`#version 300 es
precision highp float;
#inject <hook/async-directive-script>()
void main() {
	gl_Position = vec4(ASYNC_DIRECTIVE_VALUE, 0.0, 0.0, 1.0);
}`
	);
	assert.equal(result.hasErrors, false);
	assert.ok(result.code.includes("ASYNC_DIRECTIVE_VALUE"));
}

async function testCompileAsyncUsesAsyncRuntimeProcessPath() {
	const runtime = new ShaderRuntime({ mode: "warn" });
	runtime.registerRule({
		id: "user/async-runtime-rule",
		async inject() {
			return {
				header: "const float ASYNC_RUNTIME_VALUE = 3.0;",
			};
		},
	});
	const { stage } = createStage({
		runtime,
	});
	const result = await compileSceneVertexAsync(
		stage,
		`#version 300 es
precision highp float;
void main() {
	gl_Position = vec4(ASYNC_RUNTIME_VALUE, 0.0, 0.0, 1.0);
}`
	);
	assert.equal(result.hasErrors, false);
	assert.ok(result.code.includes("ASYNC_RUNTIME_VALUE"));
}

async function run() {
	try {
		testProfileCompositionValidationAndFingerprint();
		testStageABBoundaryNoDuplicateDirectiveDiagnostics();
		testHookMissingTokenStrictWarnBehavior();
		testHookTokenCollisionFallsBackToBasePatch();
		testHookCannotOverrideProfileModules();
		testDirectiveFingerprintChangeTriggersCacheMiss();
		testSourceMapSchemaVersionNormalization();
		testAsyncHookFallbackByMode();
		testStaticLumaIncludeExpandsForGLSL();
		testWebGLLightLimitConstantsExpandFromRuntimeProfile();
		testWebGLAnimationIncludeExpandsFromRuntimeProfile();
		await testCompileAsyncUsesAsyncDirectivePreprocessPath();
		await testCompileAsyncUsesAsyncRuntimeProcessPath();
		await testGeneratedSourceBlocksComposeBetweenStages();
		console.log("Shader directive pipeline v2 tests passed");
	} finally {
		Logger.reset();
	}
}

run().catch((error) => {
	console.error(error);
	process.exit(1);
});
