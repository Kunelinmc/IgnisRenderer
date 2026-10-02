import assert from "node:assert/strict";import { AlphaMode, Material } from "../../../src/materials/Material.ts";import { ShaderMaterial } from "../../../src/materials/ShaderMaterial.ts";import { WebGLProgramCompiler } from "../../../src/backends/webgl/WebGLProgramCompiler.ts";import { getWebGLSceneVariantKey } from "../../../src/backends/webgl/WebGLSceneProgramVariants.ts";import { ShaderCompileError, ShaderRuntime } from "../../../src/shaders/runtime/index.ts";import { ShaderSource } from "../../../src/shaders/ShaderSource.ts";import { PROGRAM_LIBRARY_SCENE_LIMITS, createSceneProgramRepository, createTestBuiltinSceneVariant, prepareTestBuiltinSceneVariant, createCompilerSlot, createProgramCompileFailGL, createProgramCaptureGL, createSelectiveCompileFailGL, CUSTOM_WEBGL_VERTEX, CUSTOM_WEBGL_FRAGMENT, CUSTOM_WEBGL_FRAGMENT_MRT, CUSTOM_WEBGL_FRAGMENT_DEPTH, runWebGLBackendFile } from "../../helpers/webgl-backend.mjs";
import { WebGLProgramPreparationError } from "../../../src/foundation/Error.ts";
import { createWebGLShaderMaterialFallbackVariant } from "../../../src/backends/webgl/WebGLSceneProgramVariants.ts";
import { ShaderBackendCompileStage } from "../../../src/shaders/runtime/index.ts";
import { WEBGL_TEST_PROFILE } from "../shaders/shaderDirectiveTestProfiles.mjs";
import { WebGLSceneProgramRepository } from "../../../src/backends/webgl/WebGLSceneProgramRepository.ts";
import { WebGLSceneRuntime } from "../../../src/backends/webgl/WebGLSceneRuntime.ts";
import { createTestDrawPacket } from "../helpers/drawPacket.mjs";

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
		material: { model: "phong", baseMap: true },
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

async function testCachedSceneProgramAvoidsRepeatedSourceResolution() {
	const variant = createTestBuiltinSceneVariant();
	await prepareTestBuiltinSceneVariant(variant);
	const gl = createProgramCaptureGL();
	const repository = createSceneProgramRepository(gl, () => {});
	const first = repository.getSceneProgram(undefined, "single", variant);
	const originalResolve = ShaderSource._resolve;
	let resolutions = 0;
	ShaderSource._resolve = function (...args) {
		resolutions++;
		return originalResolve.apply(this, args);
	};
	try {
		for (let index = 0; index < 20; index++) {
			assert.strictEqual(
				repository.getSceneProgram(undefined, "single", structuredClone(variant)),
				first,
			);
		}
		assert.equal(resolutions, 0, "Warm program lookup must reuse its source identity");
		assert.equal(gl.programCount, 1);
	} finally {
		ShaderSource._resolve = originalResolve;
		repository.destroy();
	}
}

async function testCachedSceneProgramObservesSourceInvalidation() {
	const variant = createTestBuiltinSceneVariant();
	await prepareTestBuiltinSceneVariant(variant);
	const repository = createSceneProgramRepository(createProgramCaptureGL(), () => {});
	const first = repository.getSceneProgram(undefined, "single", variant);
	const getProgram = () => repository.getSceneProgram(undefined, "single", variant);
	try {
		ShaderSource.clearCache("webgpu");
		assert.strictEqual(getProgram(), first);
		ShaderSource.clearCache("webgl");
		assert.throws(getProgram, WebGLProgramPreparationError);
		await prepareTestBuiltinSceneVariant(variant);
		assert.strictEqual(getProgram(), first);
		ShaderSource.configure({});
		assert.throws(getProgram, WebGLProgramPreparationError);
		await prepareTestBuiltinSceneVariant(variant);
		assert.strictEqual(getProgram(), first);
		variant.material.baseMap = true;
		assert.throws(getProgram, WebGLProgramPreparationError);
		await prepareTestBuiltinSceneVariant(variant);
		assert.notStrictEqual(getProgram(), first);
	} finally {
		repository.destroy();
		ShaderSource.resetConfiguration();
	}
}

async function testCachedBuiltinProgramObservesCompilerFingerprintChanges() {
	const variant = createTestBuiltinSceneVariant();
	await prepareTestBuiltinSceneVariant(variant);
	const runtime = new ShaderRuntime({ mode: "strict" });
	const stage = new ShaderBackendCompileStage({
		runtime, profile: WEBGL_TEST_PROFILE, mode: "strict",
	});
	const gl = createProgramCaptureGL();
	const repository = createSceneProgramRepository(gl, () => {}, runtime, stage);
	const getProgram = () => repository.getSceneProgram(undefined, "single", variant);
	try {
		const first = getProgram();
		assert.strictEqual(getProgram(), first);
		runtime.registerRule({
			id: "user/builtin-cache-revision",
			inject: () => ({ header: "// builtin runtime revision changed" }),
		});
		const second = getProgram();
		assert.notStrictEqual(second, first);
		assert.strictEqual(getProgram(), second);
		// Vary the compiler-provided fingerprint while retaining real compilation.
		const fingerprint = stage.getCacheFingerprintTag();
		stage.getCacheFingerprintTag = () => `${fingerprint}|changed`;
		const third = getProgram();
		assert.notStrictEqual(third, second);
		assert.strictEqual(getProgram(), third);
		assert.equal(gl.programCount, 3);
	} finally {
		repository.destroy();
	}
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

async function testOpaquePBRMRTVariantDeclaresAlphaUniform() {
	const variant = createTestBuiltinSceneVariant({
		output: "mrt",
		skinProfile: "skin4",
		scene: { shadows: true },
		material: {
			model: "pbr",
			alphaMask: false,
		},
	});
	await prepareTestBuiltinSceneVariant(variant);
	const source = ShaderSource.get("webgl.scene", {
		specialization: variant,
	}).stages.fragment.code;

	assert.ok(source.includes("uniform IgnisMaterialCommon"));
	assert.ok(source.includes("vec4 ignisAlpha;"));
	assert.ok(source.includes("float finalAlpha = uAlpha.z > 0.5"));
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

async function testIssuePlannedSceneProgramCompilesStartsAheadOfFirstDraw() {
	const variant = createTestBuiltinSceneVariant();
	await prepareTestBuiltinSceneVariant(variant);

	const gl = createProgramCaptureGL();
	const completionStatus = 0x91b1;
	const getProgramParameter = gl.getProgramParameter.bind(gl);
	let ready = false;
	let completionPolls = 0;
	gl.getExtension = (name) => name === "KHR_parallel_shader_compile"
		? { COMPLETION_STATUS_KHR: completionStatus } : null;
	gl.getProgramParameter = (program, parameter) => {
		if (parameter === completionStatus) {
			completionPolls++;
			return ready;
		}
		assert.equal(ready, true, "status checks must wait for asynchronous completion");
		return getProgramParameter(program, parameter);
	};
	const library = createSceneProgramRepository(gl, () => {});
	const plan = {
		lightState: null,
		sceneVariants: new Map([[getWebGLSceneVariantKey(variant), variant]]),
		depthVariants: new Map(),
	};

	// Browser tasks must run before preparation resolves an incomplete compile.
	let settled = false;
	const preparing = Promise.resolve(library.issuePlannedSceneProgramCompiles(plan))
		.then((issued) => { settled = true; return issued; });
	await Promise.resolve();
	assert.equal(settled, false, "frame preparation must wait for pending compilation");
	setTimeout(() => { ready = true; }, 0);
	assert.equal(await preparing, 1);
	assert.ok(completionPolls > 0);
	assert.equal(gl.programCount, 1);

	// Draw-time resolution consumes the in-flight compile: no second program.
	assert.ok(library.getSceneProgram(undefined, "single", variant));
	assert.equal(gl.programCount, 1);

	// A resolved variant must not be re-issued on later frames.
	const pollsBeforeCacheHit = completionPolls;
	assert.equal(await library.issuePlannedSceneProgramCompiles(plan), 0);
	assert.equal(completionPolls, pollsBeforeCacheHit);
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

async function testScenePreparationDoesNotWaitForUnrelatedPrograms() {
	const variant = createTestBuiltinSceneVariant();
	await prepareTestBuiltinSceneVariant(variant);
	for (const cached of [false, true]) {
		const gl = createProgramCaptureGL();
		const completionStatus = 0x91b1;
		const getProgramParameter = gl.getProgramParameter.bind(gl);
		let polls = 0;
		gl.getExtension = () => ({ COMPLETION_STATUS_KHR: completionStatus });
		gl.getProgramParameter = (program, parameter) => {
			if (parameter !== completionStatus) return getProgramParameter(program, parameter);
			polls++;
			return false;
		};
		const compiler = new WebGLProgramCompiler(gl);
		const repository = new WebGLSceneProgramRepository({ compiler });
		if (cached) repository.getSceneProgram(undefined, "single", variant);
		const optional = createCompilerSlot(compiler, "unrelated-effect");
		assert.equal(optional.tryGet(), null);
		const pollsBefore = polls;
		const controller = new AbortController();
		const originalTimeout = globalThis.setTimeout;
		let timers = 0;
		try {
			globalThis.setTimeout = (callback) => {
				timers++;
				queueMicrotask(() => { controller.abort(); callback(); });
				return 0;
			};
			const plan = {
				lightState: null,
				sceneVariants: new Map(cached ? [[getWebGLSceneVariantKey(variant), variant]] : []),
				depthVariants: new Map(),
			};
			const result = await repository.issuePlannedSceneProgramCompiles(plan, controller.signal)
				.catch((error) => error.name);
			assert.equal(result, 0, "unrelated compilation must not delay empty or cached plans");
			assert.equal(polls, pollsBefore);
			assert.equal(timers, 0);
		} finally {
			globalThis.setTimeout = originalTimeout;
			repository.destroy();
			compiler.destroy();
		}
	}
}

async function testScenePreparationWaitsForPreviouslyIssuedSceneAndDepthPrograms() {
	const variant = createTestBuiltinSceneVariant();
	await prepareTestBuiltinSceneVariant(variant);
	const depth = { alphaMask: false, baseMap: false, skinProfile: "static", morphPosition: false };
	const gl = createProgramCaptureGL();
	const completionStatus = 0x91b1;
	const getProgramParameter = gl.getProgramParameter.bind(gl);
	let ready = false;
	const polledPrograms = [];
	gl.getExtension = () => ({ COMPLETION_STATUS_KHR: completionStatus });
	gl.getProgramParameter = (program, parameter) => {
		if (parameter !== completionStatus) return getProgramParameter(program, parameter);
		polledPrograms.push(program.id);
		return ready && program.id !== 3;
	};
	const compiler = new WebGLProgramCompiler(gl);
	const repository = new WebGLSceneProgramRepository({ compiler });
	repository.warmupSceneProgram(undefined, "single", variant);
	repository.warmupSceneDepthPrepassProgram(undefined, "single", depth);
	assert.equal(createCompilerSlot(compiler, "unrelated-effect").tryGet(), null);
	polledPrograms.length = 0;
	const controller = new AbortController();
	const originalTimeout = globalThis.setTimeout;
	let timers = 0;
	try {
		globalThis.setTimeout = (callback) => {
			timers++;
			queueMicrotask(() => {
				ready = true;
				if (timers > 1) controller.abort();
				callback();
			});
			return 0;
		};
		const plan = {
			lightState: null,
			sceneVariants: new Map([[getWebGLSceneVariantKey(variant), variant]]),
			depthVariants: new Map([["depth", depth]]),
		};
		const result = await repository.issuePlannedSceneProgramCompiles(plan, controller.signal)
			.catch((error) => error.name);
		assert.equal(result, 0, "already-issued required programs must complete without reissuance");
		assert.equal(gl.programCount, 3);
		assert.equal(timers, 1);
		assert.deepEqual(polledPrograms, [1, 2, 1, 2]);
	} finally {
		globalThis.setTimeout = originalTimeout;
		repository.destroy();
		compiler.destroy();
	}
}

async function testColdSourcePreparationYieldsBetweenVariantsAndSkipsWarmWork() {
	ShaderSource.clearCache("webgl");
	const variants = [
		createTestBuiltinSceneVariant({ material: { model: "pbr", baseMap: true } }),
		createTestBuiltinSceneVariant({ material: { model: "pbr", normalMap: true } }),
	];
	const repository = createSceneProgramRepository(createProgramCaptureGL(), () => {});
	const progress = [];
	const scheduler = { async yieldIfNeeded() {
		progress.push(variants.map((specialization) =>
			ShaderSource.has("webgl.scene", { specialization })));
	} };
	await repository.prepareBuiltinSceneVariants(variants, undefined, scheduler);
	assert.deepEqual(progress, [[true, false], [true, true]]);
	await repository.prepareBuiltinSceneVariants(variants, undefined, scheduler);
	assert.equal(progress.length, 2, "prepared sources must skip scheduling");
	repository.destroy();
}

async function testSourcePreparationCancellationStopsLaterVariants() {
	ShaderSource.clearCache("webgl");
	const variants = [createTestBuiltinSceneVariant(), createTestBuiltinSceneVariant({ oit: true })];
	const repository = createSceneProgramRepository(createProgramCaptureGL(), () => {});
	const controller = new AbortController();
	await assert.rejects(repository.prepareBuiltinSceneVariants(variants, controller.signal, {
		async yieldIfNeeded() { controller.abort(new Error("cancelled source preparation")); },
	}), /cancelled source preparation/);
	assert.equal(ShaderSource.has("webgl.scene", { specialization: variants[0] }), true);
	assert.equal(ShaderSource.has("webgl.scene", { specialization: variants[1] }), false);
	repository.destroy();
}

async function testProgramIssuanceYieldsAndStopsOnInterruption() {
	const variants = [createTestBuiltinSceneVariant(), createTestBuiltinSceneVariant({ oit: true })];
	for (const variant of variants) await prepareTestBuiltinSceneVariant(variant);
	const plan = {
		lightState: null,
		sceneVariants: new Map(variants.map((v) => [getWebGLSceneVariantKey(v), v])),
		depthVariants: new Map(),
	};
	for (const interruption of [null, "abort", "invalidate"]) {
		const gl = createProgramCaptureGL();
		const compiler = new WebGLProgramCompiler(gl);
		const repository = new WebGLSceneProgramRepository({ compiler });
		const controller = new AbortController();
		const progress = [];
		const scheduler = { async yieldIfNeeded() {
			await new Promise((resolve) => setTimeout(resolve, 0));
			progress.push(gl.programCount);
			if (interruption === "abort") controller.abort(new Error("cancelled issuance"));
			if (interruption === "invalidate") compiler.invalidate();
		} };
		const issuing = repository.issuePlannedSceneProgramCompiles(plan, controller.signal, scheduler);
		if (interruption) {
			await assert.rejects(issuing, /cancelled issuance|invalidated/);
			assert.deepEqual(progress, [1], "must stop before issuing another program");
		} else {
			assert.equal(await issuing, 2);
			assert.deepEqual(progress, [1, 2]);
			assert.equal(await repository.issuePlannedSceneProgramCompiles(plan, undefined, scheduler), 0);
			assert.deepEqual(progress, [1, 2], "in-flight programs must skip scheduling");
		}
		repository.destroy();
		compiler.destroy();
	}
}

async function testFramePreparationSharesSourceAndIssuanceBudget() {
	ShaderSource.clearCache("webgl");
	const gl = createProgramCaptureGL();
	const compiler = new WebGLProgramCompiler(gl);
	const repository = new WebGLSceneProgramRepository({ compiler });
	const runtime = new WebGLSceneRuntime({ scenePrograms: repository, deps: {} });
	const context = { scene: {
		opaquePackets: [createTestDrawPacket({ material: new Material() })],
		transparentPackets: [], lights: [],
	}, features: {} };
	const originalPrepare = ShaderSource.prepare;
	const originalNow = performance.now;
	const originalTimeout = globalThis.setTimeout;
	const originalLink = gl.linkProgram;
	let now = 0;
	let sources = 0;
	const yieldAtPrograms = [];
	try {
		performance.now = () => now;
		ShaderSource.prepare = async (...args) => {
			const result = await originalPrepare.apply(ShaderSource, args);
			if (sources++ === 0) now += 3;
			return result;
		};
		gl.linkProgram = (...args) => {
			originalLink.apply(gl, args);
			now += 2;
		};
		globalThis.setTimeout = (callback) => {
			yieldAtPrograms.push(gl.programCount);
			queueMicrotask(callback);
			return 0;
		};
		await runtime.prepareSceneProgramSources(context);
		// 3 ms preparing sources + 2 ms issuing the first program crosses the frame budget.
		assert.equal(yieldAtPrograms[0], 1);
		const coldSources = sources;
		const coldPrograms = gl.programCount;
		const coldYields = yieldAtPrograms.length;
		await runtime.prepareSceneProgramSources(context);
		assert.equal(sources, coldSources);
		assert.equal(gl.programCount, coldPrograms);
		assert.equal(yieldAtPrograms.length, coldYields);
	} finally {
		ShaderSource.prepare = originalPrepare;
		performance.now = originalNow;
		globalThis.setTimeout = originalTimeout;
		repository.destroy();
		compiler.destroy();
	}
}

await runWebGLBackendFile([
	testUnpreparedExactVariantFailsWithoutFallbackProgram,
	testSceneProgramRepositoryCompileErrorMessage,
	testSceneProgramRepositoryCompileErrorMapsSourceLine,
	testSceneProgramRepositoryShaderMaterialCustomProgram,
	testSceneProgramRepositoryPropagatesSamplerOverflowInWarnMode,
	testSceneProgramRepositoryCachesBuiltinSceneVariants,
	testCachedSceneProgramAvoidsRepeatedSourceResolution,
	testCachedBuiltinProgramObservesCompilerFingerprintChanges,
	testNoShadowPBRVariantDeclaresFallbackBeforeLighting,
	testOpaquePBRMRTVariantDeclaresAlphaUniform,
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
	testIssuePlannedSceneProgramCompilesStartsAheadOfFirstDraw,
	testCachedSceneProgramObservesSourceInvalidation,
	testColdSourcePreparationYieldsBetweenVariantsAndSkipsWarmWork,
	testSourcePreparationCancellationStopsLaterVariants,
	testProgramIssuanceYieldsAndStopsOnInterruption,
	testScenePreparationDoesNotWaitForUnrelatedPrograms,
	testScenePreparationWaitsForPreviouslyIssuedSceneAndDepthPrograms,
	testFramePreparationSharesSourceAndIssuanceBudget,
], "WebGL scene program repository tests");
