import assert from "node:assert/strict";import { AlphaMode, Material } from "../../../src/materials/Material.ts";import { ShaderMaterial } from "../../../src/materials/ShaderMaterial.ts";import { WebGLProgramCompiler } from "../../../src/backends/webgl/WebGLProgramCompiler.ts";import { getWebGLSceneVariantKey } from "../../../src/backends/webgl/WebGLSceneProgramVariants.ts";import { ShaderCompileError, ShaderRuntime } from "../../../src/shaders/runtime/index.ts";import { ShaderSource } from "../../../src/shaders/ShaderSource.ts";import { PROGRAM_LIBRARY_SCENE_LIMITS, createSceneProgramRepository, createTestBuiltinSceneVariant, prepareTestBuiltinSceneVariant, createCompilerSlot, createProgramCompileFailGL, createProgramCaptureGL, createSelectiveCompileFailGL, CUSTOM_WEBGL_VERTEX, CUSTOM_WEBGL_FRAGMENT, CUSTOM_WEBGL_FRAGMENT_MRT, CUSTOM_WEBGL_FRAGMENT_DEPTH, runWebGLBackendFile } from "../../helpers/webgl-backend.mjs";
import { WebGLProgramPreparationError } from "../../../src/foundation/Error.ts";
import { createWebGLShaderMaterialFallbackVariant } from "../../../src/backends/webgl/WebGLSceneProgramVariants.ts";

function testUnpreparedExactVariantFailsWithoutFallbackProgram() {
	const gl = createProgramCaptureGL();
	const repository = createSceneProgramRepository(gl, () => {});
	const variant = createTestBuiltinSceneVariant({
		oit: true,
		scene: {
			shadows: true,
			shadowTransmittance: false,
			clusteredLighting: true,
		},
		material: { model: "legacy", baseMap: true },
	});
	const variantKey = ShaderSource.getIdentity("webgl.scene", {
		specialization: variant,
	});
	assert.throws(
		() => repository.getSceneProgram(undefined, "single", variant),
		(error) => {
			assert.ok(error instanceof WebGLProgramPreparationError);
			assert.equal(error.code, "webgl-scene-program-source-unprepared");
			assert.equal(error.variantKey, variantKey);
			return true;
		},
	);
	assert.equal(gl.programCount, 0);
}

function testSceneProgramRepositoryCompileErrorMessage() {
	const library = createSceneProgramRepository(createProgramCompileFailGL(), () => {});
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

function testSceneProgramRepositoryCompileErrorMapsSourceLine() {
	const gl = createProgramCompileFailGL();
	gl.getShaderInfoLog = () => "ERROR: 0:4: syntax error";
	const library = createSceneProgramRepository(gl, () => {});
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

function testSceneProgramRepositoryShaderMaterialCustomProgram() {
	const warnings = [];
	const gl = createProgramCaptureGL();
	const library = createSceneProgramRepository(gl, (key, message) =>
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

	const builtin = library.getSceneProgram(
		undefined,
		"single",
		createWebGLShaderMaterialFallbackVariant("single"),
	);
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

function testSceneProgramRepositoryPropagatesSamplerOverflowInWarnMode() {
	const gl = createProgramCaptureGL();
	gl.MAX_TEXTURE_IMAGE_UNITS = 0x8872;
	gl.getParameter = (parameter) =>
		parameter === gl.MAX_TEXTURE_IMAGE_UNITS ? 8 : 0;
	const material = new ShaderMaterial({
		name: "SamplerOverflowMaterial",
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
		textureBindings: Array.from({ length: 9 }, (_, index) => ({
			name: `texture-${index}`,
			texture: null,
			webglUniform: `uTexture${index}`,
		})),
	});
	const library = createSceneProgramRepository(
		gl,
		() => {},
		new ShaderRuntime({ mode: "warn" }),
	);

	assert.throws(
		() => library.getSceneProgram(material),
		(error) => {
			assert.equal(error?.code, "material-texture-unit-overflow");
			assert.match(error.message, /required=9/);
			assert.match(error.message, /available=8/);
			return true;
		},
	);
}

async function testSceneProgramRepositoryCachesBuiltinSceneVariants() {
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
	const library = createSceneProgramRepository(gl, () => {});
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
		ShaderSource.getIdentity("webgl.scene", { specialization: noMapVariant }),
		ShaderSource.getIdentity("webgl.scene", { specialization: baseMapVariant })
	);
}

async function testNoShadowPBRVariantDeclaresFallbackBeforeLighting() {
	const variant = createTestBuiltinSceneVariant({
		output: "mrt",
		material: { model: "pbr" },
	});
	await prepareTestBuiltinSceneVariant(variant);
	const source = ShaderSource.get("webgl.scene", {
		specialization: variant,
	}).stages.fragment.code;
	const fallbackIndex = source.indexOf(
		"vec3 sampleDirectionalShadowVisibility(int index"
	);
	const lightingIndex = source.indexOf("vec3 shadow = sampleDirectionalShadowVisibility(");

	assert.ok(fallbackIndex >= 0);
	assert.ok(lightingIndex > fallbackIndex);
}

async function testShadowVariantWithoutTransmittanceKeepsShadowUniforms() {
	const variant = createTestBuiltinSceneVariant({
		output: "mrt",
		scene: {
			shadows: true,
			shadowTransmittance: false,
		},
		material: {
			model: "pbr",
			baseMap: true,
		},
	});
	await prepareTestBuiltinSceneVariant(variant);
	const source = ShaderSource.get("webgl.scene", {
		specialization: variant,
	}).stages.fragment.code;

	assert.ok(
		source.includes("uniform vec4 uParticleShadowVolumeSliceParams[4];"),
	);
	assert.ok(
		source.includes(
			"uniform mat4 uDirShadowCascadeViewProjection[MAX_DIRECTIONAL_LIGHTS * 4];",
		),
	);
	assert.ok(!source.includes("uniform sampler2D uShadowTransmittanceAtlas;"));
}

async function testSceneProgramRepositoryShaderMaterialIgnoresBuiltinVariant() {
	const variant = createTestBuiltinSceneVariant({
		material: { baseMap: true },
	});
	await prepareTestBuiltinSceneVariant(variant);

	const gl = createProgramCaptureGL();
	const library = createSceneProgramRepository(gl, () => {});
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

function testSceneProgramRepositoryShaderMaterialCachesPerSceneTargetMode() {
	const gl = createProgramCaptureGL();
	const library = createSceneProgramRepository(gl, () => {});
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

function testSceneProgramRepositoryBuiltinDepthPrepassProgram() {
	const gl = createProgramCaptureGL();
	const library = createSceneProgramRepository(gl, () => {});

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

function testSceneProgramRepositoryShaderMaterialDepthPrepassProgram() {
	const gl = createProgramCaptureGL();
	const library = createSceneProgramRepository(gl, () => {});
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

function testSceneProgramRepositoryShaderMaterialDepthPrepassMissingSourceDiagnostics() {
	const warnings = [];
	const gl = createProgramCaptureGL();
	const library = createSceneProgramRepository(gl, (key, message) =>
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

function testSceneProgramRepositoryShaderMaterialMissingSourceFallsBack() {
	const warnings = [];
	const gl = createProgramCaptureGL();
	const library = createSceneProgramRepository(gl, (key, message) =>
		warnings.push({ key, message })
	);
	const material = new ShaderMaterial({
		name: "NoWebGLShader",
	});

	const builtin = library.getSceneProgram(
		undefined,
		"single",
		createWebGLShaderMaterialFallbackVariant("single"),
	);
	const resolved = library.getSceneProgram(material);

	assert.strictEqual(resolved, builtin);
	assert.equal(resolved.samplerLayout.required, 0);
	assert.ok(
		warnings.some((warning) =>
			warning.key.startsWith("webgl-shader-material-missing-source-")
		)
	);
}

function testSceneProgramRepositoryWarnModeFallsBackOnCustomCompileFailure() {
	const warnings = [];
	const runtime = new ShaderRuntime({ mode: "warn" });
	const gl = createSelectiveCompileFailGL("FORCE_CUSTOM_FAIL");
	const library = createSceneProgramRepository(
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

	const builtin = library.getSceneProgram(
		undefined,
		"single",
		createWebGLShaderMaterialFallbackVariant("single"),
	);
	const resolved = library.getSceneProgram(material);

	assert.strictEqual(resolved, builtin);
	assert.ok(
		warnings.some((warning) =>
			warning.key.startsWith("webgl-shader-material-compile-failed-")
		)
	);
}

function testSceneProgramRepositoryRuntimeRevisionInvalidatesCustomCache() {
	const runtime = new ShaderRuntime({ mode: "warn" });
	const gl = createProgramCaptureGL();
	const library = createSceneProgramRepository(gl, () => {}, runtime);
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
	const library = createSceneProgramRepository(gl, () => {});

	const motionBlurProgram = createCompilerSlot(
		compiler,
		"WebGLMotionBlurProgram"
	).get();
	const dofProgram = createCompilerSlot(compiler, "WebGLDOFProgram").get();
	const sceneProgram = library.getSceneProgram();

	assert.ok(motionBlurProgram.program);
	assert.ok(dofProgram.program);
	assert.ok(sceneProgram.program);
	assert.equal(gl.programCount, 3);
}

async function testOpaqueBaseMapPreparesNormalizedDepthPrepassVariant() {
	ShaderSource.clearCache("webgl");
	const variant = createTestBuiltinSceneVariant({
		material: {
			alphaMask: false,
			baseMap: true,
		},
	});
	const depthVariant = {
		alphaMask: false,
		baseMap: false,
		skinProfile: "static",
		morphPosition: false,
	};
	const gl = createProgramCaptureGL();
	const library = createSceneProgramRepository(gl, () => {});

	await library.prepareBuiltinSceneVariants([variant]);

	assert.equal(
		ShaderSource.has("webgl.scene.depth", {
			specialization: depthVariant,
		}),
		true,
	);
	assert.ok(
		library.getSceneDepthPrepassProgram(
			new Material(),
			"single",
			depthVariant,
		),
	);
}

await runWebGLBackendFile([
	testUnpreparedExactVariantFailsWithoutFallbackProgram,
	testSceneProgramRepositoryCompileErrorMessage,
	testSceneProgramRepositoryCompileErrorMapsSourceLine,
	testSceneProgramRepositoryShaderMaterialCustomProgram,
	testSceneProgramRepositoryPropagatesSamplerOverflowInWarnMode,
	testSceneProgramRepositoryCachesBuiltinSceneVariants,
	testNoShadowPBRVariantDeclaresFallbackBeforeLighting,
	testShadowVariantWithoutTransmittanceKeepsShadowUniforms,
	testSceneProgramRepositoryShaderMaterialIgnoresBuiltinVariant,
	testSceneProgramRepositoryShaderMaterialCachesPerSceneTargetMode,
	testSceneProgramRepositoryBuiltinDepthPrepassProgram,
	testSceneProgramRepositoryShaderMaterialDepthPrepassProgram,
	testSceneProgramRepositoryShaderMaterialDepthPrepassMissingSourceDiagnostics,
	testSceneProgramRepositoryShaderMaterialMissingSourceFallsBack,
	testSceneProgramRepositoryWarnModeFallsBackOnCustomCompileFailure,
	testSceneProgramRepositoryRuntimeRevisionInvalidatesCustomCache,
	testProgramOwnershipSeparatesPostProcessAndBackendPrograms,
	testOpaqueBaseMapPreparesNormalizedDepthPrepassVariant,
], "WebGL scene program repository tests");
