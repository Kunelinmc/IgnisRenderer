import assert from "node:assert/strict";import { AlphaMode, Material } from "../../../src/materials/Material.ts";import { ShaderMaterial } from "../../../src/materials/ShaderMaterial.ts";import { WebGLProgramCompiler } from "../../../src/backends/webgl/WebGLProgramCompiler.ts";import { getWebGLSceneVariantKey } from "../../../src/backends/webgl/WebGLSceneProgramVariants.ts";import { ShaderCompileError, ShaderRuntime } from "../../../src/shaders/runtime/index.ts";import { createProgramLibrary, createTestBuiltinSceneVariant, prepareTestBuiltinSceneVariant, createCompilerSlot, createProgramCompileFailGL, createProgramCaptureGL, createSelectiveCompileFailGL, CUSTOM_WEBGL_VERTEX, CUSTOM_WEBGL_FRAGMENT, CUSTOM_WEBGL_FRAGMENT_MRT, CUSTOM_WEBGL_FRAGMENT_DEPTH, runWebGLBackendFile } from "../../helpers/webgl-backend.mjs";

function testProgramLibraryCompileErrorMessage() {
	const library = createProgramLibrary(createProgramCompileFailGL(), () => {});
	assert.throws(
		() => library.getSceneProgram(),
		(error) => {
			assert.ok(error instanceof ShaderCompileError);
			assert.equal(error.backend, "webgl");
			assert.equal(error.stage, "vertex");
			assert.match(error.message, /Shader compile failed \[webgl\]/);
			return true;
		}
	);
}

function testProgramLibraryCompileErrorMapsSourceLine() {
	const gl = createProgramCompileFailGL();
	gl.getShaderInfoLog = () => "ERROR: 0:4: syntax error";
	const library = createProgramLibrary(gl, () => {});
	assert.throws(
		() => library.getSceneProgram(),
		(error) => {
			assert.ok(error instanceof ShaderCompileError);
			assert.equal(error.messages[0].line, 4);
			assert.equal(error.messages[0].sourceLine, 4);
			assert.ok(String(error.messages[0].sourcePath).includes("sceneVertex"));
			return true;
		}
	);
}

function testProgramLibraryShaderMaterialCustomProgram() {
	const warnings = [];
	const gl = createProgramCaptureGL();
	const library = createProgramLibrary(gl, (key, message) =>
		warnings.push({ key, message })
	);
	const material = new ShaderMaterial({
		name: "CustomWebGLShader",
		chunks: [
			{
				backend: "webgl",
				language: "glsl",
				stage: "vertex",
				code: CUSTOM_WEBGL_VERTEX,
			},
			{
				backend: "webgl",
				language: "glsl",
				stage: "fragment",
				code: CUSTOM_WEBGL_FRAGMENT,
			},
		],
	});

	const builtin = library.getSceneProgram();
	const customA = library.getSceneProgram(material);
	const customB = library.getSceneProgram(material);

	assert.notStrictEqual(customA, builtin);
	assert.strictEqual(customA, customB);
	assert.equal(gl.programCount, 2);
	assert.ok(
		gl.shaderSources.some((entry) => entry.source === CUSTOM_WEBGL_VERTEX)
	);
	assert.ok(
		gl.shaderSources.some((entry) => entry.source === CUSTOM_WEBGL_FRAGMENT)
	);
	assert.equal(warnings.length, 0);
}

async function testProgramLibraryCachesBuiltinSceneVariants() {
	const noMapVariant = createTestBuiltinSceneVariant();
	const baseMapVariant = createTestBuiltinSceneVariant({
		material: { baseMap: true },
	});
	const materialGBufferVariant = createTestBuiltinSceneVariant({
		output: "mrt",
		materialGBuffer: true,
	});
	await prepareTestBuiltinSceneVariant(noMapVariant);
	await prepareTestBuiltinSceneVariant(baseMapVariant);
	await prepareTestBuiltinSceneVariant(materialGBufferVariant);

	const gl = createProgramCaptureGL();
	const library = createProgramLibrary(gl, () => {});
	const first = library.getSceneProgram(undefined, "single", noMapVariant);
	const second = library.getSceneProgram(new Material(), "single", noMapVariant);
	const withBaseMap = library.getSceneProgram(
		new Material(),
		"single",
		baseMapVariant
	);
	const withMaterialGBuffer = library.getSceneProgram(
		new Material(),
		"mrt",
		materialGBufferVariant
	);

	assert.strictEqual(first, second);
	assert.notStrictEqual(first, withBaseMap);
	assert.equal(first.colorOutputCount, 1);
	assert.equal(withBaseMap.colorOutputCount, 1);
	assert.equal(withMaterialGBuffer.colorOutputCount, 5);
	assert.equal(gl.programCount, 3);
	assert.ok(
		gl.shaderSources.some(
			(entry) =>
				entry.type === gl.FRAGMENT_SHADER &&
				entry.source.includes("uniform sampler2D uBaseMap;")
		)
	);
	assert.ok(
		gl.shaderSources.some(
			(entry) =>
				entry.type === gl.FRAGMENT_SHADER &&
				!entry.source.includes("uniform sampler2D uBaseMap;")
		)
	);
	assert.notEqual(
		getWebGLSceneVariantKey(noMapVariant),
		getWebGLSceneVariantKey(baseMapVariant)
	);
}

async function testProgramLibraryShaderMaterialIgnoresBuiltinVariant() {
	const variant = createTestBuiltinSceneVariant({
		material: { baseMap: true },
	});
	await prepareTestBuiltinSceneVariant(variant);

	const gl = createProgramCaptureGL();
	const library = createProgramLibrary(gl, () => {});
	const material = new ShaderMaterial({
		name: "VariantIgnoredCustomWebGLShader",
		chunks: [
			{
				backend: "webgl",
				language: "glsl",
				stage: "vertex",
				code: CUSTOM_WEBGL_VERTEX,
			},
			{
				backend: "webgl",
				language: "glsl",
				stage: "fragment",
				code: CUSTOM_WEBGL_FRAGMENT,
			},
		],
	});

	const custom = library.getSceneProgram(material, "single", variant);

	assert.ok(custom);
	assert.equal(gl.programCount, 1);
	assert.ok(
		gl.shaderSources.some((entry) => entry.source === CUSTOM_WEBGL_FRAGMENT)
	);
	assert.ok(
		!gl.shaderSources.some((entry) =>
			entry.source.includes("uniform sampler2D uBaseMap;")
		)
	);
}

function testProgramLibraryShaderMaterialCachesPerSceneTargetMode() {
	const gl = createProgramCaptureGL();
	const library = createProgramLibrary(gl, () => {});
	const material = new ShaderMaterial({
		name: "ModeAwareCustomWebGLShader",
		chunks: [
			{
				backend: "webgl",
				language: "glsl",
				stage: "vertex",
				code: CUSTOM_WEBGL_VERTEX,
			},
			{
				backend: "webgl",
				language: "glsl",
				stage: "fragment",
				mode: "single",
				code: CUSTOM_WEBGL_FRAGMENT,
			},
			{
				backend: "webgl",
				language: "glsl",
				stage: "fragment",
				mode: "mrt",
				code: CUSTOM_WEBGL_FRAGMENT_MRT,
			},
		],
	});

	const singleA = library.getSceneProgram(material, "single");
	const singleB = library.getSceneProgram(material, "single");
	const mrtA = library.getSceneProgram(material, "mrt");
	const mrtB = library.getSceneProgram(material, "mrt");

	assert.strictEqual(singleA, singleB);
	assert.strictEqual(mrtA, mrtB);
	assert.notStrictEqual(singleA, mrtA);
	assert.equal(singleA.colorOutputCount, 1);
	assert.equal(mrtA.colorOutputCount, 3);
	assert.equal(gl.programCount, 2);
	assert.ok(
		gl.shaderSources.some((entry) => entry.source === CUSTOM_WEBGL_FRAGMENT)
	);
	assert.ok(
		gl.shaderSources.some((entry) => entry.source === CUSTOM_WEBGL_FRAGMENT_MRT)
	);
}

function testProgramLibraryBuiltinDepthPrepassProgram() {
	const gl = createProgramCaptureGL();
	const library = createProgramLibrary(gl, () => {});

	const depthProgramA = library.getSceneDepthPrepassProgram(new Material());
	const depthProgramB = library.getSceneDepthPrepassProgram(new Material());

	assert.ok(depthProgramA);
	assert.strictEqual(depthProgramA, depthProgramB);
	assert.equal(gl.programCount, 1);
	assert.ok(
		gl.shaderSources.some((entry) =>
			entry.source.includes("uBaseColor.a")
		)
	);
	assert.ok(
		gl.shaderSources.some((entry) =>
			entry.source.includes("texture(uBaseMap")
		)
	);
	assert.ok(
		gl.shaderSources.some((entry) => entry.source.includes("discard"))
	);
}

function testProgramLibraryShaderMaterialDepthPrepassProgram() {
	const gl = createProgramCaptureGL();
	const library = createProgramLibrary(gl, () => {});
	const material = new ShaderMaterial({
		name: "DepthCustomWebGLShader",
		chunks: [
			{
				backend: "webgl",
				language: "glsl",
				stage: "vertex",
				code: CUSTOM_WEBGL_VERTEX,
			},
			{
				backend: "webgl",
				language: "glsl",
				stage: "fragment-depth",
				code: CUSTOM_WEBGL_FRAGMENT_DEPTH,
			},
		],
	});

	const depthA = library.getSceneDepthPrepassProgram(material);
	const depthB = library.getSceneDepthPrepassProgram(material);

	assert.ok(depthA);
	assert.strictEqual(depthA, depthB);
	assert.equal(gl.programCount, 1);
	assert.ok(
		gl.shaderSources.some((entry) => entry.source === CUSTOM_WEBGL_FRAGMENT_DEPTH)
	);
}

function testProgramLibraryShaderMaterialDepthPrepassMissingSourceDiagnostics() {
	const warnings = [];
	const gl = createProgramCaptureGL();
	const library = createProgramLibrary(gl, (key, message) =>
		warnings.push({ key, message })
	);
	const nonMaskMaterial = new ShaderMaterial({
		name: "NoDepthNonMask",
		chunks: [
			{
				backend: "webgl",
				language: "glsl",
				stage: "vertex",
				code: CUSTOM_WEBGL_VERTEX,
			},
		],
	});
	const maskMaterial = new ShaderMaterial({
		name: "NoDepthMask",
		alphaMode: AlphaMode.Mask,
		chunks: [
			{
				backend: "webgl",
				language: "glsl",
				stage: "vertex",
				code: CUSTOM_WEBGL_VERTEX,
			},
		],
	});

	assert.equal(library.getSceneDepthPrepassProgram(nonMaskMaterial), null);
	assert.equal(warnings.length, 0);
	assert.equal(library.getSceneDepthPrepassProgram(maskMaterial), null);
	assert.equal(library.getSceneDepthPrepassProgram(maskMaterial), null);
	assert.equal(
		warnings.filter((warning) =>
			warning.key.startsWith(
				"webgl-shader-material-depth-prepass-missing-source-"
			)
		).length,
		1
	);
}

function testProgramLibraryShaderMaterialMissingSourceFallsBack() {
	const warnings = [];
	const gl = createProgramCaptureGL();
	const library = createProgramLibrary(gl, (key, message) =>
		warnings.push({ key, message })
	);
	const material = new ShaderMaterial({
		name: "NoWebGLShader",
	});

	const builtin = library.getSceneProgram();
	const resolved = library.getSceneProgram(material);

	assert.strictEqual(resolved, builtin);
	assert.ok(
		warnings.some((warning) =>
			warning.key.startsWith("webgl-shader-material-missing-source-")
		)
	);
}

function testProgramLibraryWarnModeFallsBackOnCustomCompileFailure() {
	const warnings = [];
	const runtime = new ShaderRuntime({ mode: "warn" });
	const gl = createSelectiveCompileFailGL("FORCE_CUSTOM_FAIL");
	const library = createProgramLibrary(
		gl,
		(key, message) => warnings.push({ key, message }),
		runtime
	);
	const material = new ShaderMaterial({
		name: "WarnFallbackCustomMaterial",
		chunks: [
			{
				backend: "webgl",
				language: "glsl",
				stage: "vertex",
				code: `${CUSTOM_WEBGL_VERTEX}\n// FORCE_CUSTOM_FAIL`,
			},
			{
				backend: "webgl",
				language: "glsl",
				stage: "fragment",
				code: `${CUSTOM_WEBGL_FRAGMENT}\n// FORCE_CUSTOM_FAIL`,
			},
		],
	});

	const builtin = library.getSceneProgram();
	const resolved = library.getSceneProgram(material);

	assert.strictEqual(resolved, builtin);
	assert.ok(
		warnings.some((warning) =>
			warning.key.startsWith("webgl-shader-material-compile-failed-")
		)
	);
}

function testProgramLibraryRuntimeRevisionInvalidatesCustomCache() {
	const runtime = new ShaderRuntime({ mode: "warn" });
	const gl = createProgramCaptureGL();
	const library = createProgramLibrary(gl, () => {}, runtime);
	const material = new ShaderMaterial({
		name: "RevisionInvalidateMaterial",
		chunks: [
			{
				backend: "webgl",
				language: "glsl",
				stage: "vertex",
				code: CUSTOM_WEBGL_VERTEX,
			},
			{
				backend: "webgl",
				language: "glsl",
				stage: "fragment",
				code: CUSTOM_WEBGL_FRAGMENT,
			},
		],
	});

	const first = library.getSceneProgram(material);
	const initialProgramCount = gl.programCount;
	assert.ok(first);
	assert.equal(initialProgramCount, 1);

	runtime.registerRule({
		id: "user/runtime-revision-invalidate",
		priority: 1,
		inject() {
			return {
				header: "// runtime revision invalidation",
			};
		},
	});

	const second = library.getSceneProgram(material);
	assert.ok(second);
	assert.equal(gl.programCount, 2);
}

function testProgramOwnershipSeparatesPostProcessAndBackendPrograms() {
	const gl = createProgramCaptureGL();
	const compiler = new WebGLProgramCompiler(gl);
	const library = createProgramLibrary(gl, () => {});

	const motionBlurProgram = createCompilerSlot(
		compiler,
		"WebGLMotionBlurProgram"
	).get();
	const dofProgram = createCompilerSlot(compiler, "WebGLDOFProgram").get();
	const oitResolveProgram = library.getOITResolveProgram();

	assert.ok(motionBlurProgram.program);
	assert.ok(dofProgram.program);
	assert.ok(oitResolveProgram.program);
	assert.equal(gl.programCount, 3);
}

await runWebGLBackendFile([
	testProgramLibraryCompileErrorMessage,
	testProgramLibraryCompileErrorMapsSourceLine,
	testProgramLibraryShaderMaterialCustomProgram,
	testProgramLibraryCachesBuiltinSceneVariants,
	testProgramLibraryShaderMaterialIgnoresBuiltinVariant,
	testProgramLibraryShaderMaterialCachesPerSceneTargetMode,
	testProgramLibraryBuiltinDepthPrepassProgram,
	testProgramLibraryShaderMaterialDepthPrepassProgram,
	testProgramLibraryShaderMaterialDepthPrepassMissingSourceDiagnostics,
	testProgramLibraryShaderMaterialMissingSourceFallsBack,
	testProgramLibraryWarnModeFallsBackOnCustomCompileFailure,
	testProgramLibraryRuntimeRevisionInvalidatesCustomCache,
	testProgramOwnershipSeparatesPostProcessAndBackendPrograms,
], "WebGL program library tests");
