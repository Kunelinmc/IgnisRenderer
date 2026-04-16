import assert from "node:assert/strict";
import { AmbientLight } from "../src/lights/AmbientLight.ts";
import { DirectionalLight } from "../src/lights/DirectionalLight.ts";
import { LightProbe } from "../src/lights/LightProbe.ts";
import { PointLight } from "../src/lights/PointLight.ts";
import { ReflectionProbe } from "../src/lights/ReflectionProbe.ts";
import { SpotLight } from "../src/lights/SpotLight.ts";
import { ShadowMap } from "../src/lights/ShadowMapping.ts";
import { Material } from "../src/materials/Material.ts";
import { PBRMaterial } from "../src/materials/PBRMaterial.ts";
import { ShaderMaterial } from "../src/materials/ShaderMaterial.ts";
import { Matrix4 } from "../src/maths/Matrix4.ts";
import { SH } from "../src/maths/SH.ts";
import { Texture } from "../src/core/Texture.ts";
import { collectWebGLLights } from "../src/renderers/webgl/WebGLLightCollector.ts";
import { WebGLProgramLibrary } from "../src/renderers/webgl/WebGLProgramLibrary.ts";
import { WebGLGeometryRegistry } from "../src/renderers/webgl/WebGLGeometryRegistry.ts";
import { drawWebGLPacket } from "../src/renderers/webgl/WebGLScenePass.ts";
import { createWebGLShaderSourceFactory } from "../src/shaders/webgl/WebGLShaderSourceFactory.ts";
import { WebGLBackend } from "../src/renderers/WebGLBackend.ts";
import { PARTICLE_SIM_DELTA_TIME_SECONDS_KEY } from "../src/pipeline/types.ts";
import { ShaderCompileError, ShaderRuntime } from "../src/shaders/runtime/index.ts";

const WEBGL_SHADER_SOURCE_FACTORY = createWebGLShaderSourceFactory();

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
		uniform1f: [],
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
		uniformMatrix4fv() {},
		uniformMatrix3fv() {},
		uniform4fv() {},
		uniform4f() {},
		uniform1i(location, value) {
			calls.uniform1i.push({ location, value });
		},
		uniform1f(location, value) {
			calls.uniform1f.push({ location, value });
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

function testLightCollectorLimitsAndWarnings() {
	const warnings = [];
	const warn = (key, message) => warnings.push({ key, message });
	const lights = [new AmbientLight()];

	for (let i = 0; i < 6; i++) {
		lights.push(new DirectionalLight({ intensity: 1 + i * 0.1 }));
	}
	for (let i = 0; i < 6; i++) {
		lights.push(new PointLight({ range: 100 + i }));
	}
	for (let i = 0; i < 10; i++) {
		lights.push(new SpotLight({ range: 100 + i }));
	}

	const state = collectWebGLLights(lights, true, warn);
	assert.equal(state.directionalLights.length, 4);
	assert.equal(state.pointLights.length, 4);
	assert.equal(state.spotLights.length, 8);
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
	const lightProbe = new LightProbe(sh, 0.75);
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

	assert.ok(motionBlurProgram.program);
	assert.ok(dofProgram.program);
	assert.equal(gl.programCount, 2);
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
	assert.ok(shader.fragment.includes("uSpotShadowParamsC"));
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
	assert.ok(shader.fragment.includes("sampleEnvironmentSpecular"));
	assert.ok(shader.fragment.includes("uniform vec4 uTransmissionVolume;"));
	assert.ok(shader.fragment.includes("uniform vec4 uAttenuationColor;"));
	assert.ok(shader.fragment.includes("float ior = max(uTransmissionVolume.x, 1.0);"));
	assert.ok(shader.fragment.includes("volumeAttenuation = exp(-absorb * thickness);"));
	assert.ok(shader.fragment.includes("refract(-viewDir, refractNormal, eta)"));
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
	assert.equal(gl.calls.vertexData.length, 30);
	assert.equal(gl.calls.vertexData[8], 0.25);
	assert.equal(gl.calls.vertexData[9], 0.5);
	assert.equal(gl.calls.vertexData[18], 0.75);
	assert.equal(gl.calls.vertexData[19], 0.5);
	assert.ok(
		gl.calls.attributePointers.some(
			(call) =>
				call.index === 3 &&
				call.size === 2 &&
				call.stride === 40 &&
				call.offset === 32
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
	material.map = baseMap;
	material.albedoMapUV = 1;
	material.normalMap = normalMap;
	material.normalMapUV = 1;
	material.normalScale = 0.35;
	material.metallicRoughnessMap = metallicRoughnessMap;
	material.metallicRoughnessMapUV = 1;
	material.emissiveMap = emissiveMap;
	material.emissiveMapUV = 1;
	material.occlusionMap = occlusionMap;
	material.occlusionMapUV = 1;
	material.occlusionStrength = 0.4;
	const textureTable = new Map([
		[baseMap, { texture: { id: "base" }, isLinear: false }],
		[normalMap, { texture: { id: "normal" }, isLinear: true }],
		[metallicRoughnessMap, { texture: { id: "mr" }, isLinear: true }],
		[emissiveMap, { texture: { id: "emissive" }, isLinear: false }],
		[occlusionMap, { texture: { id: "occlusion" }, isLinear: true }],
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
			attenuationColor: null,
			phong: null,
			alpha: null,
			baseMap: "uBaseMap",
			hasBaseMap: "uHasBaseMap",
			baseMapIsLinear: "uBaseMapIsLinear",
			baseMapUV: "uBaseMapUV",
			metallicRoughnessMap: "uMetallicRoughnessMap",
			hasMetallicRoughnessMap: "uHasMetallicRoughnessMap",
			metallicRoughnessMapUV: "uMetallicRoughnessMapUV",
			normalMap: "uNormalMap",
			hasNormalMap: "uHasNormalMap",
			normalMapUV: "uNormalMapUV",
			normalScale: "uNormalScale",
			emissiveMap: "uEmissiveMap",
			hasEmissiveMap: "uHasEmissiveMap",
			emissiveMapIsLinear: "uEmissiveMapIsLinear",
			emissiveMapUV: "uEmissiveMapUV",
			occlusionMap: "uOcclusionMap",
			hasOcclusionMap: "uHasOcclusionMap",
			occlusionMapUV: "uOcclusionMapUV",
			occlusionStrength: "uOcclusionStrength",
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
	assert.equal(unitFor("uBaseMapUV"), 1);
	assert.equal(unitFor("uNormalMapUV"), 1);
	assert.equal(unitFor("uMetallicRoughnessMapUV"), 1);
	assert.equal(unitFor("uEmissiveMapUV"), 1);
	assert.equal(unitFor("uOcclusionMapUV"), 1);
	assert.equal(unitFor("uHasNormalMap"), 1);
	assert.equal(unitFor("uHasMetallicRoughnessMap"), 1);
	assert.equal(unitFor("uHasEmissiveMap"), 1);
	assert.equal(unitFor("uHasOcclusionMap"), 1);
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
			enableGamma: true,
			enableSH: false,
			enableShadows: false,
			enableReflection: false,
			enableSkybox: false,
			enableSSAO: false,
			enableTAA: false,
			enableSSR: false,
			enableVolumetric: false,
			enableFXAA: false,
			warnings: [],
		},
		shadowMaps: new Map(),
		scene: {
			skybox: null,
			particleSystems: [],
			opaquePackets: [],
			transparentPackets: [],
			shadowCasterPackets: [],
			reflectivePackets: [],
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
	testProgramLibraryCompileErrorMessage();
	testProgramLibraryCompileErrorMapsSourceLine();
	testProgramLibraryShaderMaterialCustomProgram();
	testProgramLibraryShaderMaterialMissingSourceFallsBack();
	testProgramLibraryWarnModeFallsBackOnCustomCompileFailure();
	testProgramLibraryRuntimeRevisionInvalidatesCustomCache();
	testProgramLibraryCompilesMotionBlurAndDOFPrograms();
	testLightCollectorShadowBias();
	testSceneShaderBackLitShadowGuard();
	testSceneShaderUsesDecoupledShadowNormal();
	testSceneShaderIncludesReflectionProbeUniforms();
	testSceneShaderIncludesPBRTextureAndUV1Pipeline();
	testGeometryRegistryRejectsOutOfRangeIndices();
	testGeometryRegistryRetriesAfterUploadAllocationFailure();
	testGeometryRegistryUploadsUV1Attribute();
	testDrawWebGLPacketBindsPBRTexturesAndUVSets();
	testWebGLBackendParticleDeltaTimeClamp();
	await testWebGLBackendWarmupDelegatesToFrameExecutor();
	console.log("WebGL backend v2 unit tests passed");
}

await run();
