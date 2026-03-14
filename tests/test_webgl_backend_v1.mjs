import assert from "node:assert/strict";
import { AmbientLight } from "../src/lights/AmbientLight.ts";
import { DirectionalLight } from "../src/lights/DirectionalLight.ts";
import { PointLight } from "../src/lights/PointLight.ts";
import { SpotLight } from "../src/lights/SpotLight.ts";
import { ShadowMap } from "../src/lights/ShadowMapping.ts";
import { Material } from "../src/materials/Material.ts";
import { ShaderMaterial } from "../src/materials/ShaderMaterial.ts";
import { Matrix4 } from "../src/maths/Matrix4.ts";
import { collectWebGLLights } from "../src/renderers/webgl/WebGLLightCollector.ts";
import { WebGLProgramLibrary } from "../src/renderers/webgl/WebGLProgramLibrary.ts";
import { WebGLGeometryRegistry } from "../src/renderers/webgl/WebGLGeometryRegistry.ts";
import { createWebGLSceneShaderSource } from "../src/shaders/webgl/sceneShader.ts";

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
	for (let i = 0; i < 6; i++) {
		lights.push(new SpotLight({ range: 100 + i }));
	}

	const state = collectWebGLLights(lights, true, warn);
	assert.equal(state.directionalLights.length, 4);
	assert.equal(state.pointLights.length, 4);
	assert.equal(state.spotLights.length, 4);
	assert.ok(warnings.some((warning) => warning.key === "webgl-directional-light-limit"));
	assert.ok(warnings.some((warning) => warning.key === "webgl-point-light-limit"));
	assert.ok(warnings.some((warning) => warning.key === "webgl-spot-light-limit"));
}

function testProgramLibraryCompileErrorMessage() {
	const library = new WebGLProgramLibrary(createProgramCompileFailGL(), () => {});
	assert.throws(
		() => library.getSceneProgram(),
		/WebGL shader compile failed \(WebGLSceneProgram:vertex\): mock compile fail/
	);
}

function testProgramLibraryShaderMaterialCustomProgram() {
	const warnings = [];
	const gl = createProgramCaptureGL();
	const library = new WebGLProgramLibrary(gl, (key, message) =>
		warnings.push({ key, message })
	);
	const material = new ShaderMaterial({
		name: "CustomWebGLShader",
		webglGLSL: {
			vertex: CUSTOM_WEBGL_VERTEX,
			fragment: CUSTOM_WEBGL_FRAGMENT,
		},
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
	const library = new WebGLProgramLibrary(gl, (key, message) =>
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
	assert.equal(shadow.shadowMapSize, 1024);
}

function testSceneShaderBackLitShadowGuard() {
	const shader = createWebGLSceneShaderSource({
		maxDirectionalLights: 4,
		maxPointLights: 4,
		maxSpotLights: 4,
	});
	assert.ok(shader.fragment.includes("dot(normal, lightDirection) <= 0.0"));
	assert.ok(shader.fragment.includes("uniform int uDoubleSided;"));
	assert.ok(shader.fragment.includes("if (uDoubleSided == 1 && dot(normal, viewDir) < 0.0)"));
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

function run() {
	testLightCollectorLimitsAndWarnings();
	testProgramLibraryCompileErrorMessage();
	testProgramLibraryShaderMaterialCustomProgram();
	testProgramLibraryShaderMaterialMissingSourceFallsBack();
	testLightCollectorShadowBias();
	testSceneShaderBackLitShadowGuard();
	testGeometryRegistryRejectsOutOfRangeIndices();
	console.log("WebGL backend v1 unit tests passed");
}

run();
