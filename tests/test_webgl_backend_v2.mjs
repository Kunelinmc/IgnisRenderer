import assert from "node:assert/strict";
import { AmbientLight } from "../src/lights/AmbientLight.ts";
import { DirectionalLight } from "../src/lights/DirectionalLight.ts";
import { LightProbe } from "../src/lights/LightProbe.ts";
import { PointLight } from "../src/lights/PointLight.ts";
import { ReflectionProbe } from "../src/lights/ReflectionProbe.ts";
import { SpotLight } from "../src/lights/SpotLight.ts";
import { ShadowMap, createShadowRenderSet } from "../src/lights/shadows/ShadowMapping.ts";
import { Material } from "../src/materials/Material.ts";
import { PBRMaterial } from "../src/materials/PBRMaterial.ts";
import { ShaderMaterial } from "../src/materials/ShaderMaterial.ts";
import { Matrix4 } from "../src/maths/Matrix4.ts";
import { SH } from "../src/maths/SH.ts";
import { Texture } from "../src/core/Texture.ts";
import { CubeTexture } from "../src/core/CubeTexture.ts";
import { Node } from "../src/core/Node.ts";
import { Scene } from "../src/core/Scene.ts";
import { collectWebGLLights } from "../src/renderers/webgl/WebGLLightCollector.ts";
import { WebGLProgramLibrary } from "../src/renderers/webgl/WebGLProgramLibrary.ts";
import { WebGLGeometryRegistry } from "../src/renderers/webgl/WebGLGeometryRegistry.ts";
import {
	bindWebGLShaderMaterialUniforms,
	drawWebGLPacket,
} from "../src/renderers/webgl/WebGLScenePass.ts";
import {
	WEBGL_MAX_DIRECTIONAL_LIGHTS,
	WEBGL_MAX_POINT_LIGHTS,
	WEBGL_MAX_SPOT_LIGHTS,
} from "../src/renderers/webgl/constants.ts";
import { createWebGLShaderSourceFactory } from "../src/shaders/webgl/WebGLShaderSourceFactory.ts";
import { WebGLBackend } from "../src/renderers/WebGLBackend.ts";
import { PARTICLE_SIM_DELTA_TIME_SECONDS_KEY } from "../src/pipeline/types.ts";
import { ShaderCompileError, ShaderRuntime } from "../src/shaders/runtime/index.ts";
import { createResolvedPostProcess } from "./helpers/postprocess.mjs";

const WEBGL_SHADER_SOURCE_FACTORY = createWebGLShaderSourceFactory();

function createTinyCubeTexture(mips = 1, value = 1) {
	const createFace = () => new Float32Array([value, value, value, 1]);
	const faceMipmaps = [];
	for (let level = 1; level < mips; level++) {
		faceMipmaps.push(
			Array.from({ length: 6 }, () => createFace())
		);
	}
	return new CubeTexture({
		faces: Array.from({ length: 6 }, () => createFace()),
		faceMipmaps,
		size: 1,
		colorSpace: "HDR",
	});
}

function createProgramLibrary(gl, warn, shaderRuntime, shaderCompileStage) {
	return new WebGLProgramLibrary(
		gl,
		warn,
		shaderRuntime,
		shaderCompileStage,
		WEBGL_SHADER_SOURCE_FACTORY
	);
}

function createProgramCompileFailGL() {
	return {
		VERTEX_SHADER: 0x8b31,
		FRAGMENT_SHADER: 0x8b30,
		COMPILE_STATUS: 0x8b81,
		LINK_STATUS: 0x8b82,
		VALIDATE_STATUS: 0x8b83,
		createShader(type) {
			return { type, compiled: false };
		},
		shaderSource(shader, source) {
			shader.source = source;
		},
		compileShader(shader) {
			shader.compiled = shader.type !== this.VERTEX_SHADER;
		},
		getShaderParameter(shader, parameter) {
			if (parameter === this.COMPILE_STATUS) {
				return shader.compiled;
			}
			return true;
		},
		getShaderInfoLog() {
			return "mock compile fail";
		},
		deleteShader() {},
		createProgram() {
			return {};
		},
		attachShader() {},
		linkProgram() {},
		getProgramParameter() {
			return true;
		},
		getProgramInfoLog() {
			return "";
		},
		deleteProgram() {},
		getUniformLocation() {
			return {};
		},
	};
}

function createGeometryTestGL() {
	return {
		TRIANGLES: 0x0004,
		LINES: 0x0001,
		POINTS: 0x0000,
		UNSIGNED_SHORT: 0x1403,
		UNSIGNED_INT: 0x1405,
		ARRAY_BUFFER: 0x8892,
		ELEMENT_ARRAY_BUFFER: 0x8893,
		STATIC_DRAW: 0x88e4,
		FLOAT: 0x1406,
		createVertexArray() {
			return {};
		},
		createBuffer() {
			return {};
		},
		deleteVertexArray() {},
		deleteBuffer() {},
		bindVertexArray() {},
		bindBuffer() {},
		bufferData() {},
		enableVertexAttribArray() {},
		vertexAttribPointer() {},
	};
}

function createRetryGeometryTestGL() {
	const gl = createGeometryTestGL();
	let createBufferCallCount = 0;
	return {
		...gl,
		createBuffer() {
			createBufferCallCount++;
			if (createBufferCallCount === 1) {
				return null;
			}
			return {};
		},
	};
}

function createGeometryCaptureGL() {
	const gl = createGeometryTestGL();
	const calls = {
		attributePointers: [],
		vertexData: null,
	};
	return {
		...gl,
		calls,
		bufferData(target, data) {
			if (target === this.ARRAY_BUFFER) {
				calls.vertexData = data;
			}
		},
		vertexAttribPointer(index, size, type, normalized, stride, offset) {
			calls.attributePointers.push({
				index,
				size,
				type,
				normalized,
				stride,
				offset,
			});
		},
	};
}

function createScenePassCaptureGL() {
	const calls = {
		activeTextures: [],
		boundTextures: [],
		uniform1i: [],
		uniform1ui: [],
		uniform1f: [],
		uniform2fv: [],
		uniform3fv: [],
		uniform4fv: [],
		uniform2iv: [],
		uniform3iv: [],
		uniform4iv: [],
		uniform2uiv: [],
		uniform3uiv: [],
		uniform4uiv: [],
		uniform2f: [],
		uniform4f: [],
		uniformMatrix4fv: [],
		depthMask: [],
	};
	return {
		TEXTURE0: 0x84c0,
		TEXTURE_2D: 0x0de1,
		calls,
		activeTexture(unit) {
			calls.activeTextures.push(unit);
		},
		bindTexture(target, texture) {
			calls.boundTextures.push({ target, texture });
		},
		bindVertexArray() {},
		uniformMatrix4fv(location, transpose, values) {
			calls.uniformMatrix4fv.push({
				location,
				transpose,
				values: Array.from(values),
			});
		},
		uniformMatrix3fv() {},
		uniform4fv(location, values) {
			calls.uniform4fv.push({ location, values: Array.from(values) });
		},
		uniform2fv(location, values) {
			calls.uniform2fv.push({ location, values: Array.from(values) });
		},
		uniform3fv(location, values) {
			calls.uniform3fv.push({ location, values: Array.from(values) });
		},
		uniform2iv(location, values) {
			calls.uniform2iv.push({ location, values: Array.from(values) });
		},
		uniform3iv(location, values) {
			calls.uniform3iv.push({ location, values: Array.from(values) });
		},
		uniform4iv(location, values) {
			calls.uniform4iv.push({ location, values: Array.from(values) });
		},
		uniform2uiv(location, values) {
			calls.uniform2uiv.push({ location, values: Array.from(values) });
		},
		uniform3uiv(location, values) {
			calls.uniform3uiv.push({ location, values: Array.from(values) });
		},
		uniform4uiv(location, values) {
			calls.uniform4uiv.push({ location, values: Array.from(values) });
		},
		uniform2f(location, x, y) {
			calls.uniform2f.push({ location, x, y });
		},
		uniform4f(location, x, y, z, w) {
			calls.uniform4f.push({ location, x, y, z, w });
		},
		uniform1i(location, value) {
			calls.uniform1i.push({ location, value });
		},
		uniform1ui(location, value) {
			calls.uniform1ui.push({ location, value });
		},
		uniform1f(location, value) {
			calls.uniform1f.push({ location, value });
		},
		depthMask(flag) {
			calls.depthMask.push(flag);
		},
		drawElements() {},
	};
}

function createProgramCaptureGL() {
	let programCount = 0;
	const shaderSources = [];
	return {
		VERTEX_SHADER: 0x8b31,
		FRAGMENT_SHADER: 0x8b30,
		COMPILE_STATUS: 0x8b81,
		LINK_STATUS: 0x8b82,
		VALIDATE_STATUS: 0x8b83,
		shaderSources,
		get programCount() {
			return programCount;
		},
		createShader(type) {
			return { type, compiled: true };
		},
		shaderSource(shader, source) {
			shader.source = source;
			shaderSources.push({ type: shader.type, source });
		},
		compileShader() {},
		getShaderParameter() {
			return true;
		},
		getShaderInfoLog() {
			return "";
		},
		deleteShader() {},
		createProgram() {
			programCount++;
			return { id: programCount };
		},
		attachShader() {},
		linkProgram() {},
		validateProgram() {},
		getProgramParameter(_program, parameter) {
			if (parameter === this.LINK_STATUS || parameter === this.VALIDATE_STATUS) {
				return true;
			}
			return true;
		},
		getProgramInfoLog() {
			return "";
		},
		deleteProgram() {},
		getUniformLocation() {
			return {};
		},
	};
}

function createSelectiveCompileFailGL(failPattern) {
	let programCount = 0;
	return {
		VERTEX_SHADER: 0x8b31,
		FRAGMENT_SHADER: 0x8b30,
		COMPILE_STATUS: 0x8b81,
		LINK_STATUS: 0x8b82,
		VALIDATE_STATUS: 0x8b83,
		get programCount() {
			return programCount;
		},
		createShader(type) {
			return { type, compiled: true };
		},
		shaderSource(shader, source) {
			shader.source = source;
		},
		compileShader(shader) {
			shader.compiled = !String(shader.source).includes(failPattern);
		},
		getShaderParameter(shader, parameter) {
			if (parameter === this.COMPILE_STATUS) {
				return shader.compiled;
			}
			return true;
		},
		getShaderInfoLog() {
			return "selective compile fail";
		},
		deleteShader() {},
		createProgram() {
			programCount++;
			return { id: programCount };
		},
		attachShader() {},
		linkProgram() {},
		validateProgram() {},
		getProgramParameter(_program, parameter) {
			if (parameter === this.LINK_STATUS || parameter === this.VALIDATE_STATUS) {
				return true;
			}
			return true;
		},
		getProgramInfoLog() {
			return "";
		},
		deleteProgram() {},
		getUniformLocation() {
			return {};
		},
	};
}

const CUSTOM_WEBGL_VERTEX = /* glsl */ `
#version 300 es
precision highp float;
layout(location = 0) in vec3 aPosition;
void main() {
	gl_Position = vec4(aPosition, 1.0);
}
`;

const CUSTOM_WEBGL_FRAGMENT = /* glsl */ `
#version 300 es
precision highp float;
out vec4 outColor;
void main() {
	outColor = vec4(0.0, 1.0, 0.0, 1.0);
}
`;

const CUSTOM_WEBGL_FRAGMENT_MRT = /* glsl */ `
#version 300 es
precision highp float;
layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outMotionDepth;
layout(location = 2) out vec4 outNormal;
void main() {
	outColor = vec4(0.0, 1.0, 0.0, 1.0);
	outMotionDepth = vec4(0.0, 0.0, 0.0, 1.0);
	outNormal = vec4(0.5, 0.5, 1.0, 1.0);
}
`;

function testLightCollectorLimitsAndWarnings() {
	const warnings = [];
	const warn = (key, message) => warnings.push({ key, message });
	const lights = [new AmbientLight()];

	for (let i = 0; i < WEBGL_MAX_DIRECTIONAL_LIGHTS + 2; i++) {
		lights.push(new DirectionalLight({ intensity: 1 + i * 0.1 }));
	}
	for (let i = 0; i < WEBGL_MAX_POINT_LIGHTS + 2; i++) {
		lights.push(new PointLight({ range: 100 + i }));
	}
	for (let i = 0; i < WEBGL_MAX_SPOT_LIGHTS + 2; i++) {
		lights.push(new SpotLight({ range: 100 + i }));
	}

	const state = collectWebGLLights(lights, true, warn);
	assert.equal(state.directionalLights.length, WEBGL_MAX_DIRECTIONAL_LIGHTS);
	assert.equal(state.pointLights.length, WEBGL_MAX_POINT_LIGHTS);
	assert.equal(state.spotLights.length, WEBGL_MAX_SPOT_LIGHTS);
	assert.ok(warnings.some((warning) => warning.key === "webgl-directional-light-limit"));
	assert.ok(warnings.some((warning) => warning.key === "webgl-point-light-limit"));
	assert.ok(warnings.some((warning) => warning.key === "webgl-spot-light-limit"));
}

function testLightProbeAmbientAndReflectionProbeSpecularCollection() {
	const warnings = [];
	const warn = (key, message) => warnings.push({ key, message });
	const sh = SH.empty();
	sh[0] = { r: 120, g: 60, b: 30 };
	const probeMap = new Texture(new Float32Array(4 * 2 * 4), 4, 2, "HDR");
	probeMap.mipmaps = [
		new Float32Array(4 * 2 * 4),
		new Float32Array(2 * 1 * 4),
	];
	const lightProbe = new LightProbe(sh);
	const reflectionProbe = new ReflectionProbe({
		prefilteredMap: probeMap,
		shape: "box",
	});

	const withoutSH = collectWebGLLights(
		[lightProbe, reflectionProbe],
		true,
		warn,
		false,
		undefined,
		false
	);
	assert.ok(withoutSH.ambientColor[0] > 0);
	assert.ok(withoutSH.ambientColor[1] > 0);
	assert.ok(withoutSH.ambientColor[2] > 0);
	assert.ok(withoutSH.envSpecularMap);
	assert.equal(withoutSH.reflectionProbeCount, 1);
	assert.equal(withoutSH.reflectionProbes.length, 1);
	assert.equal(
		warnings.some(
			(warning) => warning.key === "webgl-light-unsupported-lightProbe"
		),
		false
	);

	const withSH = collectWebGLLights(
		[lightProbe, reflectionProbe],
		true,
		warn,
		false,
		undefined,
		true
	);
	assert.equal(withSH.ambientColor[0], 0);
	assert.equal(withSH.ambientColor[1], 0);
	assert.equal(withSH.ambientColor[2], 0);
}

function testLightCollectorCollectsLocalizedLightProbes() {
	const warnings = [];
	const warn = (key, message) => warnings.push({ key, message });
	const globalProbe = new LightProbe(SH.empty());

	const localProbe = new LightProbe({
		sh: SH.empty(),
		shape: "box",
		halfExtents: { x: 2, y: 2, z: 2 },
		priority: 7,
	});
	localProbe.position.set(0, 0, 0);
	localProbe.updateWorldMatrix();
	localProbe.markRuntimeDirty();

	const state = collectWebGLLights(
		[globalProbe, localProbe],
		true,
		warn,
		false,
		undefined,
		true,
		null,
		false,
		{ x: 0, y: 0, z: 0 }
	);
	assert.equal(state.localLightProbeCount, 1);
	assert.equal(state.localLightProbes.length, 1);
	assert.equal(state.localLightProbes[0].priority, 7);
	assert.equal(state.localLightProbes[0].shape, 1);
	assert.equal(state.ambientColor[0], 0);
	assert.equal(warnings.length, 0);
}

function testLightCollectorSupportsCubeTextureEnvironmentMaps() {
	const warn = () => {};
	const cubeProbeMap = createTinyCubeTexture(3, 0.75);
	const cubeEnvironment = createTinyCubeTexture(2, 0.5);
	const reflectionProbe = new ReflectionProbe({
		prefilteredMap: cubeProbeMap,
		shape: "sphere",
	});

	const probeState = collectWebGLLights(
		[reflectionProbe],
		true,
		warn,
		false,
		undefined,
		false,
		cubeEnvironment
	);
	assert.ok(probeState.envSpecularMap);
	assert.notEqual(probeState.envSpecularMap, cubeProbeMap);
	assert.equal(probeState.envSpecularFallbackMap, null);
	assert.equal(probeState.envSpecularMap.width, 4);
	assert.equal(probeState.envSpecularMap.height, 2);
	assert.equal(probeState.reflectionProbeCount, 1);

	const environmentState = collectWebGLLights(
		[],
		true,
		warn,
		false,
		undefined,
		false,
		cubeEnvironment
	);
	assert.ok(environmentState.envSpecularMap);
	assert.notEqual(environmentState.envSpecularMap, cubeEnvironment);
	assert.equal(environmentState.envSpecularFallbackMap, null);
	assert.equal(environmentState.envSpecularMap.width, 4);
	assert.equal(environmentState.envSpecularMap.height, 2);
	assert.equal(environmentState.reflectionProbeCount, 0);
}

function testLightCollectorUsesParentedProbeCaptureOrigin() {
	const warn = () => {};
	const probeMap = createTinyCubeTexture(3, 0.75);
	const model = new Node();
	model.position.set(5, 0, 0);
	const probe = new ReflectionProbe({
		prefilteredMap: probeMap,
		shape: "box",
	});
	model.addChild(probe);
	probe.position.set(2, 0, 0);
	model.updateWorldMatrix();

	const state = collectWebGLLights([probe], true, warn);
	assert.equal(state.reflectionProbeCount, 1);
	assert.deepEqual(state.reflectionProbes[0].captureWorldPosition, [5, 0, 0]);
}

function testLightCollectorUsesProbeCaptureOriginWhenParentedToSceneRoot() {
	const warn = () => {};
	const probeMap = createTinyCubeTexture(3, 0.75);
	const scene = new Scene();
	const probe = new ReflectionProbe({
		prefilteredMap: probeMap,
		shape: "box",
	});
	scene.add(probe);
	probe.position.set(5, 0, 0);
	scene.updateWorldMatrices();

	const state = collectWebGLLights([probe], true, warn);
	assert.equal(state.reflectionProbeCount, 1);
	assert.deepEqual(state.reflectionProbes[0].captureWorldPosition, [5, 0, 0]);
}

function testLightCollectorDoesNotExposeEnvironmentSpecularFallbackMap() {
	const warnings = [];
	const warn = (key, message) => warnings.push({ key, message });
	const environment = createTinyCubeTexture(2, 0.5);
	const state = collectWebGLLights(
		[],
		true,
		warn,
		false,
		undefined,
		false,
		environment,
		false
	);
	assert.ok(state.envSpecularMap);
	assert.equal(state.envSpecularFallbackMap, null);
	assert.equal(state.reflectionProbeCount, 0);
	assert.equal(state.reflectionProbes.length, 0);
	assert.equal(
		warnings.some((warning) => warning.key.startsWith("webgl-environment-")),
		false
	);
}

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
	assert.equal(gl.programCount, 2);
	assert.ok(
		gl.shaderSources.some((entry) => entry.source === CUSTOM_WEBGL_FRAGMENT)
	);
	assert.ok(
		gl.shaderSources.some((entry) => entry.source === CUSTOM_WEBGL_FRAGMENT_MRT)
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

function testProgramLibraryCompilesMotionBlurAndDOFPrograms() {
	const gl = createProgramCaptureGL();
	const library = createProgramLibrary(gl, () => {});

	const motionBlurProgram = library.getMotionBlurProgram();
	const dofProgram = library.getDOFProgram();
	const oitResolveProgram = library.getOITResolveProgram();

	assert.ok(motionBlurProgram.program);
	assert.ok(dofProgram.program);
	assert.ok(oitResolveProgram.program);
	assert.equal(gl.programCount, 3);
}

function testLightCollectorShadowBias() {
	const light = new DirectionalLight();
	const shadowMap = new ShadowMap(1024, {
		shadowBias: 0.008,
		shadowSlopeBias: 0.03,
		shadowTexelBias: 1,
		shadowMaxBias: 0.05,
	});
	shadowMap.viewProjectionMatrix = Matrix4.identity();
	const state = collectWebGLLights(
		[light],
		true,
		() => {},
		true,
		new Map([[light, shadowMap]])
	);
	const shadow = state.directionalShadows[0];
	assert.ok(shadow.enabled);
	assert.ok(Math.abs(shadow.depthBias - (0.008 + 1 / 1024)) < 1e-6);
	assert.ok(Math.abs(shadow.slopeBias - 0.03) < 1e-6);
	assert.equal(shadow.shadowMapSize, 1024);
	assert.equal(shadow.pcssEnabled, false);
	assert.equal(shadow.pcssRadius, 0);
	assert.equal(shadow.shadowSamples, 16);
	assert.equal(shadow.shadowSearchSamples, 16);
}

function testLightCollectorPCSSShadowParams() {
	const light = new DirectionalLight();
	const shadowMap = new ShadowMap(1024, {
		shadowBias: 0.008,
		shadowSlopeBias: 0.03,
		shadowTexelBias: 1,
		shadowMaxBias: 0.05,
		shadowPCF: 1.25,
		shadowRadius: 5,
		shadowSamples: 24,
		shadowSearchSamples: 12,
	});
	shadowMap.viewProjectionMatrix = Matrix4.identity();
	const state = collectWebGLLights(
		[light],
		true,
		() => {},
		true,
		new Map([[light, shadowMap]])
	);
	const shadow = state.directionalShadows[0];
	assert.equal(shadow.pcfRadius, 1.25);
	assert.equal(shadow.pcssEnabled, true);
	assert.equal(shadow.pcssRadius, 5);
	assert.equal(shadow.shadowSamples, 24);
	assert.equal(shadow.shadowSearchSamples, 12);
}

function testLightCollectorDirectionalCSMShadowData() {
	const scene = new Scene();
	const light = new DirectionalLight();
	scene.add(light);
	scene.shadows.bind(
		light,
		scene.shadows.createCSM({
			size: 1024,
			blendRatio: 0.2,
			cascadeCounts: {
				directional: 4,
			},
		})
	);
	const shadowConfig = scene.shadows.getLegacyShadowConfig(light);
	assert.ok(shadowConfig);
	const renderSet = createShadowRenderSet(shadowConfig);
	for (let index = 0; index < renderSet.slices.length; index++) {
		const slice = renderSet.slices[index];
		slice.shadowMap.viewProjectionMatrix = Matrix4.identity();
		slice.splitNear = index * 10;
		slice.splitFar = (index + 1) * 10;
	}

	const state = collectWebGLLights(
		[light],
		true,
		() => {},
		true,
		new Map([[light, renderSet]])
	);
	const shadow = state.directionalShadows[0];
	assert.equal(shadow.enabled, true);
	assert.equal(shadow.strategyType, "csm");
	assert.equal(shadow.cascadeCount, 4);
	assert.equal(shadow.cascadeBlendRatio, 0.2);
	assert.equal(shadow.shadowMapBaseSize, 1024);
	assert.equal(shadow.shadowMapSize, 512);
	assert.ok(shadow.cascadeViewProjectionMatrices[3]);
	assert.deepEqual(shadow.cascadeSplits[0], [0, 10, 0, 0]);
	assert.deepEqual(shadow.cascadeSplits[1], [10, 20, 1, 0]);
	assert.deepEqual(shadow.cascadeSplits[2], [20, 30, 0, 1]);
	assert.deepEqual(shadow.cascadeSplits[3], [30, 40, 1, 1]);
}

function testSceneShaderBackLitShadowGuard() {
	const shader = WEBGL_SHADER_SOURCE_FACTORY.createSceneShaderSource({
		maxDirectionalLights: 4,
		maxPointLights: 4,
		maxSpotLights: 4,
	});
	assert.ok(shader.fragment.includes("dot(normal, lightDirection) <= 0.0"));
	assert.ok(shader.fragment.includes("uniform int uDoubleSided;"));
	assert.ok(shader.fragment.includes("if (uDoubleSided == 1 && dot(normal, viewDir) < 0.0)"));
}

function testSceneShaderUsesDecoupledShadowNormal() {
	const shader = WEBGL_SHADER_SOURCE_FACTORY.createSceneShaderSource({
		maxDirectionalLights: 4,
		maxPointLights: 4,
		maxSpotLights: 4,
	});
	assert.ok(shader.fragment.includes("vec3 shadowNormal = normal;"));
	assert.ok(
		/shadePBR\(\s*albedo,\s*normal,\s*shadowNormal,\s*viewDir,/.test(
			shader.fragment
		)
	);
	assert.ok(shader.fragment.includes("shadePhong(albedo, normal, shadowNormal, viewDir);"));
	assert.ok(/sampleDirectionalShadowVisibility\([\s\S]*shadowNormal/.test(shader.fragment));
	assert.ok(shader.fragment.includes("uDirShadowParamsC"));
	assert.ok(shader.fragment.includes("uDirShadowCascadeViewProjection"));
	assert.ok(shader.fragment.includes("uDirShadowCascadeSplits"));
	assert.ok(shader.fragment.includes("resolveDirectionalCascadeIndex"));
	assert.ok(shader.fragment.includes("uSpotShadowParamsC"));
	assert.ok(shader.fragment.includes("uParticleShadowVolumeAtlas"));
	assert.ok(shader.fragment.includes("uParticleShadowVolumeSliceParams"));
	assert.ok(shader.fragment.includes("sampleParticleShadowVolumeTransmittance"));
}

function testSceneShaderIncludesReflectionProbeUniforms() {
	const shader = WEBGL_SHADER_SOURCE_FACTORY.createSceneShaderSource({
		maxDirectionalLights: 4,
		maxPointLights: 4,
		maxSpotLights: 4,
	});
	assert.ok(shader.fragment.includes("uniform sampler2D uEnvSpecularMap;"));
	assert.ok(shader.fragment.includes("uniform sampler2D uBrdfLUT;"));
	assert.ok(shader.fragment.includes("uReflectionProbeCount"));
	assert.ok(shader.fragment.includes("uReflectionProbeWorldToProbeRow0"));
	assert.ok(shader.fragment.includes("computeReflectionProbeParallaxDirection"));
	assert.ok(shader.fragment.includes("computeReflectionProbeDepthOcclusion"));
	assert.ok(shader.fragment.includes("sampleEnvironmentSpecular"));
	assert.ok(shader.fragment.includes("uniform vec4 uTransmissionVolume;"));
	assert.ok(shader.fragment.includes("uniform vec4 uAttenuationColor;"));
	assert.ok(shader.fragment.includes("uniform vec4 uIridescence;"));
	assert.ok(shader.fragment.includes("uniform sampler2D uIridescenceMap;"));
	assert.ok(shader.fragment.includes("uniform sampler2D uIridescenceThicknessMap;"));
	assert.ok(shader.fragment.includes("float ior = max(uTransmissionVolume.x, 1.0);"));
	assert.ok(shader.fragment.includes("volumeAttenuation = exp(-absorb * thickness);"));
	assert.ok(shader.fragment.includes("refract(-viewDir, refractNormal, eta)"));
	assert.ok(shader.fragment.includes("resolveIridescenceFresnel"));
	assert.ok(shader.fragment.includes("diffuseFresnelWeight"));
}

function testSceneShaderIncludesLocalizedLightProbeUniforms() {
	const shader = WEBGL_SHADER_SOURCE_FACTORY.createSceneShaderSource({
		maxDirectionalLights: 4,
		maxPointLights: 4,
		maxSpotLights: 4,
	});
	assert.ok(shader.fragment.includes("uniform int uLocalLightProbeCount;"));
	assert.ok(shader.fragment.includes("uLocalLightProbeWorldToProbeRow0"));
	assert.ok(shader.fragment.includes("uLocalLightProbeCoeffs"));
	assert.ok(shader.fragment.includes("selectTopTwoLocalLightProbes"));
	assert.ok(shader.fragment.includes("sampleBlendedLocalLightProbeIrradiance"));
	assert.ok(shader.fragment.includes("sampleBlendedLocalLightProbeRadiance"));
}

function testSceneShaderFitsCommonWebGLTextureUnitLimit() {
	const shader = WEBGL_SHADER_SOURCE_FACTORY.createSceneShaderSource({
		maxDirectionalLights: 4,
		maxPointLights: 4,
		maxSpotLights: 4,
	});
	const samplerMatches = shader.fragment.match(/\buniform\s+sampler2D\b/g) ?? [];

	assert.equal(samplerMatches.length, 16);
	assert.ok(
		shader.fragment.includes(
			"uniform vec3 uSHAmbientCoeffs[SH_COEFFICIENT_COUNT];"
		)
	);
	assert.ok(!shader.fragment.includes("uniform sampler2D uSHAmbientCoeffs;"));
}

function testSceneShaderIncludesPBRTextureAndUV1Pipeline() {
	const shader = WEBGL_SHADER_SOURCE_FACTORY.createSceneShaderSource({
		maxDirectionalLights: 4,
		maxPointLights: 4,
		maxSpotLights: 4,
	});
	assert.ok(shader.vertex.includes("layout(location = 3) in vec2 aUv1;"));
	assert.ok(shader.vertex.includes("out vec2 vUv1;"));
	assert.ok(shader.fragment.includes("in vec2 vUv1;"));
	assert.ok(shader.fragment.includes("uniform sampler2D uMetallicRoughnessMap;"));
	assert.ok(shader.fragment.includes("uniform sampler2D uNormalMap;"));
	assert.ok(shader.fragment.includes("uniform sampler2D uOcclusionMap;"));
	assert.ok(shader.fragment.includes("uniform int uBaseMapUV;"));
	assert.ok(shader.fragment.includes("vec2 resolveUV(int uvSet) {"));
	assert.ok(shader.fragment.includes("uniform vec4 uBaseMapTransformA;"));
	assert.ok(shader.fragment.includes("uniform vec2 uBaseMapTransformB;"));
	assert.ok(shader.fragment.includes("vec2 resolveMappedUV("));
}

function testSceneShaderIncludesOITPassMode() {
	const shader = WEBGL_SHADER_SOURCE_FACTORY.createSceneShaderSource({
		maxDirectionalLights: 4,
		maxPointLights: 4,
		maxSpotLights: 4,
	});

	assert.ok(shader.fragment.includes("uniform int uOITPassMode;"));
	assert.ok(shader.fragment.includes("float resolveOITWeight("));
	assert.ok(shader.fragment.includes("if (uOITPassMode == 1)"));
	assert.ok(shader.fragment.includes("if (uOITPassMode == 2)"));
}

function testParticleShaderIncludesOITPassMode() {
	const shader = WEBGL_SHADER_SOURCE_FACTORY.getRawPart("particleFragment");

	assert.ok(shader.includes("uniform int uOITPassMode;"));
	assert.ok(shader.includes("float resolveParticleOITWeight("));
	assert.ok(shader.includes("if (uOITPassMode == 1)"));
	assert.ok(shader.includes("if (uOITPassMode == 2)"));
}

function testGeometryRegistryRejectsOutOfRangeIndices() {
	const warnings = [];
	const registry = new WebGLGeometryRegistry(createGeometryTestGL(), (k, m) =>
		warnings.push({ key: k, message: m })
	);

	const primitive = {
		id: "p0",
		geometry: {
			positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
			normals: null,
			uv0: null,
			indices: new Uint32Array([0, 1, 9]),
		},
		topology: "triangle-list",
		material: new Material(),
	};
	const packet = {
		id: "packet-0",
		primitive,
	};

	assert.equal(registry.getGeometry(packet), null);
	assert.ok(
		warnings.some((warning) => warning.key === "webgl-geometry-index-range-p0")
	);
}

function testGeometryRegistryRetriesAfterUploadAllocationFailure() {
	const warnings = [];
	const registry = new WebGLGeometryRegistry(createRetryGeometryTestGL(), (k, m) =>
		warnings.push({ key: k, message: m })
	);

	const primitive = {
		id: "p-retry",
		geometry: {
			positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
			normals: null,
			uv0: null,
			indices: new Uint32Array([0, 1, 2]),
		},
		topology: "triangle-list",
		material: new Material(),
	};
	const packet = {
		id: "packet-retry",
		primitive,
	};

	assert.equal(registry.getGeometry(packet), null);
	const retried = registry.getGeometry(packet);
	assert.ok(retried);
	assert.equal(retried?.indexCount, 3);
	assert.ok(
		warnings.some(
			(warning) => warning.key === "webgl-geometry-upload-failed-p-retry"
		)
	);
}

function testGeometryRegistryUploadsUV1Attribute() {
	const gl = createGeometryCaptureGL();
	const registry = new WebGLGeometryRegistry(gl, () => {});
	const primitive = {
		id: "p-uv1",
		geometry: {
			positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
			normals: new Float32Array([
				0, 0, 1,
				0, 0, 1,
				0, 0, 1,
			]),
			uv0: new Float32Array([0, 0, 1, 0, 0, 1]),
			uv1: new Float32Array([0.25, 0.5, 0.75, 0.5, 0.25, 0.9]),
			uv2: new Float32Array([0.125, 0.25, 0.375, 0.5, 0.625, 0.75]),
			uv3: new Float32Array([0.875, 0.75, 0.625, 0.5, 0.375, 0.25]),
			indices: new Uint32Array([0, 1, 2]),
		},
		topology: "triangle-list",
		material: new Material(),
	};
	const packet = {
		id: "packet-uv1",
		primitive,
	};

	const handle = registry.getGeometry(packet);
	assert.ok(handle);
	assert.ok(gl.calls.vertexData instanceof Float32Array);
	assert.equal(gl.calls.vertexData.length, 42);
	assert.equal(gl.calls.vertexData[8], 0.25);
	assert.equal(gl.calls.vertexData[9], 0.5);
	assert.equal(gl.calls.vertexData[10], 0.125);
	assert.equal(gl.calls.vertexData[11], 0.25);
	assert.equal(gl.calls.vertexData[12], 0.875);
	assert.equal(gl.calls.vertexData[13], 0.75);
	assert.equal(gl.calls.vertexData[22], 0.75);
	assert.equal(gl.calls.vertexData[23], 0.5);
	assert.equal(gl.calls.vertexData[24], 0.375);
	assert.equal(gl.calls.vertexData[25], 0.5);
	assert.equal(gl.calls.vertexData[26], 0.625);
	assert.equal(gl.calls.vertexData[27], 0.5);
	assert.ok(
		gl.calls.attributePointers.some(
			(call) =>
				call.index === 3 &&
				call.size === 2 &&
				call.stride === 56 &&
				call.offset === 32
		)
	);
	assert.ok(
		gl.calls.attributePointers.some(
			(call) =>
				call.index === 4 &&
				call.size === 2 &&
				call.stride === 56 &&
				call.offset === 40
		)
	);
	assert.ok(
		gl.calls.attributePointers.some(
			(call) =>
				call.index === 5 &&
				call.size === 2 &&
				call.stride === 56 &&
				call.offset === 48
		)
	);
}

function testDrawWebGLPacketBindsPBRTexturesAndUVSets() {
	const gl = createScenePassCaptureGL();
	const material = new PBRMaterial();
	const baseMap = { id: "base-map", linear: false };
	const normalMap = { id: "normal-map", linear: true };
	const metallicRoughnessMap = { id: "mr-map", linear: true };
	const emissiveMap = { id: "emissive-map", linear: false };
	const occlusionMap = { id: "occlusion-map", linear: true };
	const iridescenceMap = { id: "iridescence-map", linear: true };
	const iridescenceThicknessMap = {
		id: "iridescence-thickness-map",
		linear: true,
	};
	const anisotropyMap = { id: "anisotropy-map", linear: true };
	baseMap.repeat = { x: 0.5, y: 1.5 };
	baseMap.offset = { x: 0.25, y: -0.125 };
	baseMap.rotation = Math.PI / 6;
	normalMap.repeat = { x: 2, y: 0.5 };
	normalMap.offset = { x: -0.2, y: 0.3 };
	normalMap.rotation = -Math.PI / 4;
	metallicRoughnessMap.repeat = { x: 1.25, y: 0.75 };
	metallicRoughnessMap.offset = { x: 0.1, y: 0.2 };
	metallicRoughnessMap.rotation = Math.PI / 8;
	emissiveMap.repeat = { x: 0.8, y: 0.9 };
	emissiveMap.offset = { x: -0.05, y: 0.15 };
	emissiveMap.rotation = 0;
	occlusionMap.repeat = { x: 1.1, y: 1.2 };
	occlusionMap.offset = { x: 0.05, y: -0.1 };
	occlusionMap.rotation = Math.PI / 3;
	iridescenceMap.repeat = { x: 0.7, y: 1.4 };
	iridescenceMap.offset = { x: 0.12, y: -0.07 };
	iridescenceMap.rotation = Math.PI / 5;
	iridescenceThicknessMap.repeat = { x: 1.3, y: 0.6 };
	iridescenceThicknessMap.offset = { x: -0.11, y: 0.09 };
	iridescenceThicknessMap.rotation = -Math.PI / 7;
	anisotropyMap.repeat = { x: 0.9, y: 1.1 };
	anisotropyMap.offset = { x: 0.07, y: -0.03 };
	anisotropyMap.rotation = Math.PI / 9;
	material.map = baseMap;
	material.albedoMapUV = 2;
	material.normalMap = normalMap;
	material.normalMapUV = 3;
	material.normalScale = 0.35;
	material.metallicRoughnessMap = metallicRoughnessMap;
	material.metallicRoughnessMapUV = 2;
	material.emissiveMap = emissiveMap;
	material.emissiveMapUV = 3;
	material.occlusionMap = occlusionMap;
	material.occlusionMapUV = 2;
	material.occlusionStrength = 0.4;
	material.iridescenceFactor = 0.8;
	material.iridescenceMap = iridescenceMap;
	material.iridescenceMapUV = 3;
	material.iridescenceThicknessMap = iridescenceThicknessMap;
	material.iridescenceThicknessMapUV = 2;
	material.anisotropyStrength = 0.6;
	material.anisotropyRotation = Math.PI / 4;
	material.anisotropyMap = anisotropyMap;
	material.anisotropyMapUV = 3;
	const textureTable = new Map([
		[baseMap, { texture: { id: "base" }, isLinear: false }],
		[normalMap, { texture: { id: "normal" }, isLinear: true }],
		[metallicRoughnessMap, { texture: { id: "mr" }, isLinear: true }],
		[emissiveMap, { texture: { id: "emissive" }, isLinear: false }],
		[occlusionMap, { texture: { id: "occlusion" }, isLinear: true }],
		[iridescenceMap, { texture: { id: "iridescence" }, isLinear: true }],
		[
			iridescenceThicknessMap,
			{ texture: { id: "iridescence-thickness" }, isLinear: true },
		],
		[anisotropyMap, { texture: { id: "anisotropy" }, isLinear: true }],
	]);
	const sceneProgram = {
		program: {},
		uniforms: {
			model: null,
			normalMatrix: null,
			prevModel: null,
			shadingModel: null,
			baseColor: null,
			emissive: null,
			pbr: null,
			transmissionVolume: null,
			iridescence: "uIridescence",
			attenuationColor: null,
			anisotropy: "uAnisotropy",
			phong: null,
			alpha: null,
			baseMap: "uBaseMap",
			hasBaseMap: "uHasBaseMap",
			baseMapIsLinear: "uBaseMapIsLinear",
			baseMapUV: "uBaseMapUV",
			baseMapTransformA: "uBaseMapTransformA",
			baseMapTransformB: "uBaseMapTransformB",
			metallicRoughnessMap: "uMetallicRoughnessMap",
			hasMetallicRoughnessMap: "uHasMetallicRoughnessMap",
			metallicRoughnessMapUV: "uMetallicRoughnessMapUV",
			metallicRoughnessMapTransformA: "uMetallicRoughnessMapTransformA",
			metallicRoughnessMapTransformB: "uMetallicRoughnessMapTransformB",
			normalMap: "uNormalMap",
			hasNormalMap: "uHasNormalMap",
			normalMapUV: "uNormalMapUV",
			normalMapTransformA: "uNormalMapTransformA",
			normalMapTransformB: "uNormalMapTransformB",
			normalScale: "uNormalScale",
			emissiveMap: "uEmissiveMap",
			hasEmissiveMap: "uHasEmissiveMap",
			emissiveMapIsLinear: "uEmissiveMapIsLinear",
			emissiveMapUV: "uEmissiveMapUV",
			emissiveMapTransformA: "uEmissiveMapTransformA",
			emissiveMapTransformB: "uEmissiveMapTransformB",
			occlusionMap: "uOcclusionMap",
			hasOcclusionMap: "uHasOcclusionMap",
			occlusionMapUV: "uOcclusionMapUV",
			occlusionMapTransformA: "uOcclusionMapTransformA",
			occlusionMapTransformB: "uOcclusionMapTransformB",
			occlusionStrength: "uOcclusionStrength",
			iridescenceMap: "uIridescenceMap",
			hasIridescenceMap: "uHasIridescenceMap",
			iridescenceMapUV: "uIridescenceMapUV",
			iridescenceMapTransformA: "uIridescenceMapTransformA",
			iridescenceMapTransformB: "uIridescenceMapTransformB",
			iridescenceThicknessMap: "uIridescenceThicknessMap",
			hasIridescenceThicknessMap: "uHasIridescenceThicknessMap",
			iridescenceThicknessMapUV: "uIridescenceThicknessMapUV",
			iridescenceThicknessMapTransformA: "uIridescenceThicknessMapTransformA",
			iridescenceThicknessMapTransformB: "uIridescenceThicknessMapTransformB",
			hasAnisotropyMap: "uHasAnisotropyMap",
			anisotropyMapUV: "uAnisotropyMapUV",
			anisotropyMapTransformA: "uAnisotropyMapTransformA",
			anisotropyMapTransformB: "uAnisotropyMapTransformB",
			doubleSided: null,
			customSamplers: {},
		},
	};
	const host = {
		_gl: gl,
		_geometry: {
			getGeometry() {
				return {
					vao: {},
					topology: 4,
					indexCount: 3,
					indexType: 5123,
				};
			},
		},
		_textures: {
			getBaseColorTexture(texture) {
				return textureTable.get(texture) ?? { texture: null, isLinear: true };
			},
		},
		_modelMatrixCache: new Map(),
		_modelMatrixKeysThisFrame: new Set(),
		_setCullMode() {},
		_bindShaderMaterialTextures() {},
		_bindShaderMaterialUniforms() {},
	};
	const packet = {
		id: "packet-pbr-textures",
		meshInstance: { id: "mesh-0", skeleton: null },
		material,
		worldMatrix: Matrix4.identity(),
		normalMatrix: Matrix4.identity(),
	};

	drawWebGLPacket(host, sceneProgram, packet, false, {});
	const unitFor = (name) =>
		gl.calls.uniform1i.find((entry) => entry.location === name)?.value;
	assert.equal(unitFor("uBaseMap"), 0);
	assert.equal(unitFor("uNormalMap"), 8);
	assert.equal(unitFor("uMetallicRoughnessMap"), 9);
	assert.equal(unitFor("uEmissiveMap"), 10);
	assert.equal(unitFor("uOcclusionMap"), 11);
	assert.equal(unitFor("uIridescenceMap"), 12);
	assert.equal(unitFor("uIridescenceThicknessMap"), 15);
	assert.equal(unitFor("uBaseMapUV"), 2);
	assert.equal(unitFor("uNormalMapUV"), 3);
	assert.equal(unitFor("uMetallicRoughnessMapUV"), 2);
	assert.equal(unitFor("uEmissiveMapUV"), 3);
	assert.equal(unitFor("uOcclusionMapUV"), 2);
	assert.equal(unitFor("uIridescenceMapUV"), 3);
	assert.equal(unitFor("uIridescenceThicknessMapUV"), 2);
	assert.equal(unitFor("uAnisotropyMapUV"), 3);
	assert.equal(unitFor("uHasNormalMap"), 1);
	assert.equal(unitFor("uHasMetallicRoughnessMap"), 1);
	assert.equal(unitFor("uHasEmissiveMap"), 1);
	assert.equal(unitFor("uHasOcclusionMap"), 1);
	assert.equal(unitFor("uHasIridescenceMap"), 1);
	assert.equal(unitFor("uHasIridescenceThicknessMap"), 1);
	assert.equal(unitFor("uHasAnisotropyMap"), 0);
	assert.equal(unitFor("uBaseMapIsLinear"), 0);
	assert.equal(unitFor("uEmissiveMapIsLinear"), 0);
	assert.ok(
		gl.calls.uniform1f.some(
			(entry) =>
				entry.location === "uNormalScale" &&
				Math.abs(entry.value - 0.35) < 1e-6
		)
	);
	assert.ok(
		gl.calls.uniform1f.some(
			(entry) =>
				entry.location === "uOcclusionStrength" &&
				Math.abs(entry.value - 0.4) < 1e-6
		)
	);
	assert.ok(
		gl.calls.uniform4fv.some(
			(entry) =>
				entry.location === "uAnisotropy" &&
				Math.abs(entry.values[0] - 0.6) < 1e-6
		)
	);

	const transformAFor = (name) =>
		gl.calls.uniform4f.find((entry) => entry.location === name);
	const transformBFor = (name) =>
		gl.calls.uniform2f.find((entry) => entry.location === name);
	const assertUVTransform = (nameA, nameB, map) => {
		const transformA = transformAFor(nameA);
		const transformB = transformBFor(nameB);
		assert.ok(transformA);
		assert.ok(transformB);
		assert.ok(Math.abs(transformA.x - map.repeat.x) < 1e-6);
		assert.ok(Math.abs(transformA.y - map.repeat.y) < 1e-6);
		assert.ok(Math.abs(transformA.z - map.offset.x) < 1e-6);
		assert.ok(Math.abs(transformA.w - map.offset.y) < 1e-6);
		assert.ok(Math.abs(transformB.x - Math.cos(map.rotation)) < 1e-6);
		assert.ok(Math.abs(transformB.y - Math.sin(map.rotation)) < 1e-6);
	};
	assertUVTransform("uBaseMapTransformA", "uBaseMapTransformB", baseMap);
	assertUVTransform(
		"uMetallicRoughnessMapTransformA",
		"uMetallicRoughnessMapTransformB",
		metallicRoughnessMap
	);
	assertUVTransform("uNormalMapTransformA", "uNormalMapTransformB", normalMap);
	assertUVTransform("uEmissiveMapTransformA", "uEmissiveMapTransformB", emissiveMap);
	assertUVTransform(
		"uOcclusionMapTransformA",
		"uOcclusionMapTransformB",
		occlusionMap
	);
	assertUVTransform(
		"uIridescenceMapTransformA",
		"uIridescenceMapTransformB",
		iridescenceMap
	);
	assertUVTransform(
		"uIridescenceThicknessMapTransformA",
		"uIridescenceThicknessMapTransformB",
		iridescenceThicknessMap
	);
	assertUVTransform(
		"uAnisotropyMapTransformA",
		"uAnisotropyMapTransformB",
		anisotropyMap
	);
}

function testDrawWebGLPacketBindsAnisotropyMapWhenSharedSlotIsFree() {
	const gl = createScenePassCaptureGL();
	const material = new PBRMaterial();
	const anisotropyMap = { id: "anisotropy-map", linear: true };
	material.anisotropyStrength = 0.6;
	material.anisotropyMap = anisotropyMap;
	material.anisotropyMapUV = 3;
	const sceneProgram = {
		program: {},
		uniforms: {
			model: null,
			normalMatrix: null,
			prevModel: null,
			shadingModel: null,
			baseColor: null,
			emissive: null,
			pbr: null,
			transmissionVolume: null,
			iridescence: null,
			attenuationColor: null,
			anisotropy: null,
			phong: null,
			alpha: null,
			iridescenceThicknessMap: "uIridescenceThicknessMap",
			hasIridescenceThicknessMap: "uHasIridescenceThicknessMap",
			hasAnisotropyMap: "uHasAnisotropyMap",
			anisotropyMapUV: "uAnisotropyMapUV",
			anisotropyMapTransformA: null,
			anisotropyMapTransformB: null,
			doubleSided: null,
			customSamplers: {},
		},
	};
	const textureTable = new Map([
		[anisotropyMap, { texture: { id: "anisotropy" }, isLinear: true }],
	]);
	const host = {
		_gl: gl,
		_geometry: {
			getGeometry() {
				return {
					vao: {},
					topology: 4,
					indexCount: 3,
					indexType: 5123,
				};
			},
		},
		_textures: {
			getBaseColorTexture(texture) {
				return textureTable.get(texture) ?? { texture: null, isLinear: true };
			},
		},
		_modelMatrixCache: new Map(),
		_modelMatrixKeysThisFrame: new Set(),
		_setCullMode() {},
		_bindShaderMaterialTextures() {},
		_bindShaderMaterialUniforms() {},
	};
	const packet = {
		id: "packet-pbr-anisotropy",
		meshInstance: { id: "mesh-0", skeleton: null },
		material,
		worldMatrix: Matrix4.identity(),
		normalMatrix: Matrix4.identity(),
	};

	drawWebGLPacket(host, sceneProgram, packet, false, {});
	const textureUnit15Index = gl.calls.activeTextures.findIndex(
		(unit) => unit === gl.TEXTURE0 + 15
	);
	assert.notEqual(textureUnit15Index, -1);
	assert.equal(
		gl.calls.boundTextures[textureUnit15Index].texture.id,
		"anisotropy"
	);
	const unitFor = (name) =>
		gl.calls.uniform1i.find((entry) => entry.location === name)?.value;
	assert.equal(unitFor("uIridescenceThicknessMap"), 15);
	assert.equal(unitFor("uHasIridescenceThicknessMap"), 0);
	assert.equal(unitFor("uHasAnisotropyMap"), 1);
	assert.equal(unitFor("uAnisotropyMapUV"), 3);
}

function testDrawWebGLPacketAppliesMaterialDepthWriteState() {
	const gl = createScenePassCaptureGL();
	const sceneProgram = {
		program: {},
		uniforms: {},
	};
	const host = {
		_gl: gl,
		_geometry: {
			getGeometry() {
				return {
					vao: {},
					topology: 4,
					indexCount: 3,
					indexType: 5123,
				};
			},
		},
		_textures: {
			getBaseColorTexture() {
				return { texture: null, isLinear: true };
			},
		},
		_modelMatrixCache: new Map(),
		_modelMatrixKeysThisFrame: new Set(),
		_setCullMode() {},
		_bindShaderMaterialTextures() {},
		_bindShaderMaterialUniforms() {},
	};
	const material = new Material({
		depthWrite: false,
	});
	const packet = {
		id: "packet-depth-read",
		meshInstance: { id: "mesh-depth-read", skeleton: null },
		material,
		worldMatrix: Matrix4.identity(),
		normalMatrix: Matrix4.identity(),
	};

	drawWebGLPacket(host, sceneProgram, packet, false, {});
	assert.deepEqual(gl.calls.depthMask, [false]);

	material.depthWrite = true;
	drawWebGLPacket(host, sceneProgram, packet, false, {});
	assert.deepEqual(gl.calls.depthMask, [false, true]);
}

function testShaderMaterialCustomUniformBinding() {
	const gl = createScenePassCaptureGL();
	const material = new ShaderMaterial({
		uniformBindings: [
			{ name: "time", type: "f32", value: 1.5, webglUniform: "uTime" },
			{ name: "mode", type: "i32", value: 2, webglUniform: "uMode" },
			{ name: "flags", type: "u32", value: 3, webglUniform: "uFlags" },
			{ name: "uvScale", type: "vec2f", value: [4, 5], webglUniform: "uUVScale" },
			{ name: "normal", type: "vec3f", value: [6, 7, 8], webglUniform: "uNormal" },
			{ name: "tint", type: "vec4f", value: [1, 0.5, 0.25, 1], webglUniform: "uTint" },
			{ name: "offset", type: "vec2i", value: [9, 10], webglUniform: "uOffset" },
			{ name: "indices", type: "vec3i", value: [11, 12, 13], webglUniform: "uIndices" },
			{ name: "mask", type: "vec4i", value: [14, 15, 16, 17], webglUniform: "uMask" },
			{ name: "uoffset", type: "vec2u", value: [18, 19], webglUniform: "uUOffset" },
			{ name: "uindices", type: "vec3u", value: [20, 21, 22], webglUniform: "uUIndices" },
			{ name: "umask", type: "vec4u", value: [23, 24, 25, 26], webglUniform: "uUMask" },
			{
				name: "matrix",
				type: "mat4x4f",
				value: [
					[1, 2, 3, 4],
					[5, 6, 7, 8],
					[9, 10, 11, 12],
					[13, 14, 15, 16],
				],
				webglUniform: "uMatrix",
			},
			{ name: "unused", type: "f32", value: 99, webglUniform: "uUnused" },
		],
	});
	const customUniforms = {};
	for (const binding of material.getUniformBindings()) {
		customUniforms[binding.webglUniform] =
			binding.webglUniform === "uUnused" ? null : binding.webglUniform;
	}
	const sceneProgram = { uniforms: { customUniforms } };
	const host = { _gl: gl };

	bindWebGLShaderMaterialUniforms(host, sceneProgram, material);

	assert.deepEqual(gl.calls.uniform1f, [{ location: "uTime", value: 1.5 }]);
	assert.deepEqual(gl.calls.uniform1i, [{ location: "uMode", value: 2 }]);
	assert.deepEqual(gl.calls.uniform1ui, [{ location: "uFlags", value: 3 }]);
	assert.deepEqual(gl.calls.uniform2fv[0], {
		location: "uUVScale",
		values: [4, 5],
	});
	assert.deepEqual(gl.calls.uniform3fv[0], {
		location: "uNormal",
		values: [6, 7, 8],
	});
	assert.deepEqual(gl.calls.uniform4fv[0], {
		location: "uTint",
		values: [1, 0.5, 0.25, 1],
	});
	assert.deepEqual(gl.calls.uniform2iv[0].values, [9, 10]);
	assert.deepEqual(gl.calls.uniform3iv[0].values, [11, 12, 13]);
	assert.deepEqual(gl.calls.uniform4iv[0].values, [14, 15, 16, 17]);
	assert.deepEqual(gl.calls.uniform2uiv[0].values, [18, 19]);
	assert.deepEqual(gl.calls.uniform3uiv[0].values, [20, 21, 22]);
	assert.deepEqual(gl.calls.uniform4uiv[0].values, [23, 24, 25, 26]);
	assert.deepEqual(gl.calls.uniformMatrix4fv[0], {
		location: "uMatrix",
		transpose: false,
		values: [1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16],
	});
	assert.equal(gl.calls.uniform1f.length, 1);
}

function testWebGLBackendParticleDeltaTimeClamp() {
	const backend = new WebGLBackend();
	const transient = new Map([
		[PARTICLE_SIM_DELTA_TIME_SECONDS_KEY, 1000],
	]);
	const deltaTimeSeconds = backend._resolveParticleDeltaTime({ transient });
	assert.equal(deltaTimeSeconds, 0.5);
}

async function testWebGLBackendWarmupDelegatesToFrameExecutor() {
	const backend = new WebGLBackend();
	backend._frameExecutor = {
		warmup() {
			return {
				phase: "webgl-programs",
				total: 3,
				compiled: 2,
				skipped: 1,
				failed: 0,
				errors: [],
			};
		},
	};
	const report = await backend.warmup({
		camera: {},
		attachments: { width: 1, height: 1 },
		features: {
			enableLighting: true,
			enableSH: false,
			enableShadows: false,
			enableReflection: false,
			enableEnvironment: false,
			warnings: [],
		},
		postProcess: createResolvedPostProcess(),
		shadowMaps: new Map(),
		scene: {
			environment: null,
			particleSystems: [],
			opaquePackets: [],
			transparentPackets: [],
			shadowCasterPackets: [],
			reflectivePackets: [],
			decalPackets: [],
		},
		shCoeffs: [],
		shAmbientCoeffs: [],
		worldMatrix: Matrix4.identity(),
		transient: new Map(),
	});
	assert.equal(report.total, 3);
	assert.equal(report.compiled, 2);
	assert.equal(report.skipped, 1);
	assert.equal(report.failed, 0);
}

async function run() {
	await WEBGL_SHADER_SOURCE_FACTORY.prepareAll();
	testLightCollectorLimitsAndWarnings();
	testLightProbeAmbientAndReflectionProbeSpecularCollection();
	testLightCollectorCollectsLocalizedLightProbes();
	testLightCollectorSupportsCubeTextureEnvironmentMaps();
	testLightCollectorUsesParentedProbeCaptureOrigin();
	testLightCollectorUsesProbeCaptureOriginWhenParentedToSceneRoot();
	testLightCollectorDoesNotExposeEnvironmentSpecularFallbackMap();
	testProgramLibraryCompileErrorMessage();
	testProgramLibraryCompileErrorMapsSourceLine();
	testProgramLibraryShaderMaterialCustomProgram();
	testProgramLibraryShaderMaterialCachesPerSceneTargetMode();
	testProgramLibraryShaderMaterialMissingSourceFallsBack();
	testProgramLibraryWarnModeFallsBackOnCustomCompileFailure();
	testProgramLibraryRuntimeRevisionInvalidatesCustomCache();
	testProgramLibraryCompilesMotionBlurAndDOFPrograms();
	testLightCollectorShadowBias();
	testLightCollectorPCSSShadowParams();
	testLightCollectorDirectionalCSMShadowData();
	testSceneShaderBackLitShadowGuard();
	testSceneShaderUsesDecoupledShadowNormal();
	testSceneShaderIncludesReflectionProbeUniforms();
	testSceneShaderIncludesLocalizedLightProbeUniforms();
	testSceneShaderFitsCommonWebGLTextureUnitLimit();
	testSceneShaderIncludesPBRTextureAndUV1Pipeline();
	testSceneShaderIncludesOITPassMode();
	testParticleShaderIncludesOITPassMode();
	testGeometryRegistryRejectsOutOfRangeIndices();
	testGeometryRegistryRetriesAfterUploadAllocationFailure();
	testGeometryRegistryUploadsUV1Attribute();
	testDrawWebGLPacketBindsPBRTexturesAndUVSets();
	testDrawWebGLPacketBindsAnisotropyMapWhenSharedSlotIsFree();
	testDrawWebGLPacketAppliesMaterialDepthWriteState();
	testShaderMaterialCustomUniformBinding();
	testWebGLBackendParticleDeltaTimeClamp();
	await testWebGLBackendWarmupDelegatesToFrameExecutor();
	console.log("WebGL backend v2 unit tests passed");
}

await run();
