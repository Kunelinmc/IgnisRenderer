import assert from "node:assert/strict";
import { AmbientLight } from "../../../src/lights/AmbientLight.ts";
import { DirectionalLight } from "../../../src/lights/DirectionalLight.ts";
import { LightProbe } from "../../../src/lights/LightProbe.ts";
import { PointLight } from "../../../src/lights/PointLight.ts";
import { ReflectionProbe } from "../../../src/lights/ReflectionProbe.ts";
import { SpotLight } from "../../../src/lights/SpotLight.ts";
import { ShadowMap, createShadowRenderSet } from "../../../src/lights/shadows/ShadowMapping.ts";
import { AlphaMode, Material } from "../../../src/materials/Material.ts";
import { PBRMaterial } from "../../../src/materials/PBRMaterial.ts";
import { ShaderMaterial } from "../../../src/materials/ShaderMaterial.ts";
import { Matrix4 } from "../../../src/maths/Matrix4.ts";
import { SH } from "../../../src/maths/SH.ts";
import { Texture } from "../../../src/core/Texture.ts";
import { CubeTexture } from "../../../src/core/CubeTexture.ts";
import { Node } from "../../../src/core/Node.ts";
import { Scene } from "../../../src/core/Scene.ts";
import { collectWebGLLights } from "../../../src/backends/webgl/WebGLLightCollector.ts";
import { WebGLProgramCompiler } from "../../../src/backends/webgl/WebGLProgramCompiler.ts";
import { WebGLProgramWarmupQueue } from "../../../src/backends/webgl/WebGLProgramWarmupQueue.ts";
import { WebGLProgramLibrary } from "../../../src/backends/webgl/WebGLProgramLibrary.ts";
import {
	getWebGLSceneVariantKey,
} from "../../../src/backends/webgl/WebGLSceneProgramVariants.ts";
import { WebGLGeometryRegistry } from "../../../src/backends/webgl/WebGLGeometryRegistry.ts";
import {
	bindWebGLShaderMaterialUniforms,
	bindWebGLShaderMaterialTextures,
	drawWebGLPacket,
	renderWebGLEarlyZPrepass,
	renderWebGLPackets,
} from "../../../src/backends/webgl/WebGLScenePass.ts";
import { WebGLShadowPass } from "../../../src/backends/webgl/WebGLShadowPass.ts";
import {
	MAX_DIRECTIONAL_LIGHTS,
	MAX_POINT_LIGHTS,
	MAX_SPOT_LIGHTS,
} from "../../../src/backends/constants.ts";
import {
	ShaderSource,
	WEBGL_SHADER_PARTS,
} from "../../../src/shaders/ShaderSource.ts";
import { WebGLBackend } from "../../../src/backends/webgl/WebGLBackend.ts";
import { PARTICLE_SIM_DELTA_TIME_SECONDS_KEY } from "../../../src/pipeline/types.ts";
import { ShaderCompileError, ShaderRuntime } from "../../../src/shaders/runtime/index.ts";
import { createResolvedPostProcess } from "../../helpers/postprocess.mjs";

const TEST_SCENE_LIMITS = {
	maxDirectionalLights: 4,
	maxPointLights: 4,
	maxSpotLights: 4,
};

const PROGRAM_LIBRARY_SCENE_LIMITS = {
	maxDirectionalLights: MAX_DIRECTIONAL_LIGHTS,
	maxPointLights: MAX_POINT_LIGHTS,
	maxSpotLights: MAX_SPOT_LIGHTS,
};

function getTestSceneShader() {
	return ShaderSource.get("webgl.scene.raw", {
		limits: TEST_SCENE_LIMITS,
	});
}

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
	);
}

function createTestBuiltinSceneVariant(overrides = {}) {
	return {
		output: overrides.output ?? "single",
		materialGBuffer: overrides.materialGBuffer ?? false,
		oit: overrides.oit ?? false,
		scene: {
			shadows: false,
			shadowTransmittance: false,
			clusteredLighting: false,
			sh: false,
			localLightProbes: false,
			irradianceProbeGrid: false,
			reflectionProbes: false,
			environmentSpecular: false,
			...(overrides.scene ?? {}),
		},
		material: {
			model: "unlit",
			baseMap: false,
			metallicRoughnessMap: false,
			normalMap: false,
			emissiveMap: false,
			occlusionMap: false,
			iridescence: false,
			iridescenceMap: false,
			iridescenceThicknessMap: false,
			anisotropy: false,
			anisotropyMap: false,
			transmission: false,
			alphaMask: false,
			...(overrides.material ?? {}),
		},
	};
}

async function prepareTestBuiltinSceneVariant(variant) {
	await ShaderSource.prepareMany([
		{
			key: "webgl.scene.raw",
			params: { limits: PROGRAM_LIBRARY_SCENE_LIMITS, variant },
		},
		{
			key: "webgl.scene.composite",
			params: { limits: PROGRAM_LIBRARY_SCENE_LIMITS, variant },
		},
	]);
}

function createCompilerSlot(compiler, label, uniformNames = []) {
	return compiler.createSlot({
		label,
		vertex: () => CUSTOM_WEBGL_VERTEX,
		fragment: () => CUSTOM_WEBGL_FRAGMENT,
		reflect(gl, program) {
			return {
				program,
				uniforms: Object.fromEntries(
					uniformNames.map((name) => [name, gl.getUniformLocation(program, name)])
				),
			};
		},
	});
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
		uniform3f: [],
		uniform4f: [],
		uniformMatrix4fv: [],
		depthMask: [],
		depthFunc: [],
		colorMask: [],
		drawBuffers: [],
		enable: [],
		disable: [],
		scissor: [],
		drawElements: [],
	};
	return {
		FRAMEBUFFER: 0x8d40,
		COLOR_ATTACHMENT0: 0x8ce0,
		COLOR_ATTACHMENT1: 0x8ce1,
		COLOR_ATTACHMENT2: 0x8ce2,
		COLOR_ATTACHMENT3: 0x8ce3,
		COLOR_ATTACHMENT4: 0x8ce4,
		NONE: 0,
		TEXTURE0: 0x84c0,
		TEXTURE_2D: 0x0de1,
		TRIANGLES: 0x0004,
		DEPTH_TEST: 0x0b71,
		BLEND: 0x0be2,
		CULL_FACE: 0x0b44,
		SCISSOR_TEST: 0x0c11,
		LESS: 0x0201,
		LEQUAL: 0x0203,
		calls,
		bindFramebuffer() {},
		drawBuffers(buffers) {
			calls.drawBuffers.push([...buffers]);
		},
		colorMask(r, g, b, a) {
			calls.colorMask.push([r, g, b, a]);
		},
		enable(cap) {
			calls.enable.push(cap);
		},
		disable(cap) {
			calls.disable.push(cap);
		},
		depthFunc(func) {
			calls.depthFunc.push(func);
		},
		scissor(x, y, width, height) {
			calls.scissor.push({ x, y, width, height });
		},
		useProgram() {},
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
		uniform3f(location, x, y, z) {
			calls.uniform3f.push({ location, x, y, z });
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
		drawElements(mode, count, type, offset) {
			calls.drawElements.push({ mode, count, type, offset });
		},
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

function createProgramWarmupTrackingGL(options = {}) {
	let programCount = 0;
	const completionStatus = 0x91b1;
	const completeAfterPolls = options.completeAfterPolls ?? 0;
	const validateStatus = options.validateStatus ?? true;
	const calls = {
		compileShader: 0,
		linkProgram: 0,
		getShaderParameter: [],
		getProgramParameter: [],
		getUniformLocation: [],
		validateProgram: 0,
	};
	const extension = options.parallel ? {
		COMPLETION_STATUS_KHR: completionStatus,
	} : null;
	return {
		VERTEX_SHADER: 0x8b31,
		FRAGMENT_SHADER: 0x8b30,
		COMPILE_STATUS: 0x8b81,
		LINK_STATUS: 0x8b82,
		VALIDATE_STATUS: 0x8b83,
		calls,
		get programCount() {
			return programCount;
		},
		getExtension(name) {
			return name === "KHR_parallel_shader_compile" ? extension : null;
		},
		createShader(type) {
			return { type, compiled: true };
		},
		shaderSource(shader, source) {
			shader.source = source;
		},
		compileShader() {
			calls.compileShader++;
		},
		getShaderParameter(shader, parameter) {
			calls.getShaderParameter.push(parameter);
			if (parameter === this.COMPILE_STATUS) {
				return shader.compiled;
			}
			return true;
		},
		getShaderInfoLog() {
			return "";
		},
		deleteShader() {},
		createProgram() {
			programCount++;
			return { id: programCount, polls: 0 };
		},
		attachShader() {},
		linkProgram() {
			calls.linkProgram++;
		},
		validateProgram() {
			calls.validateProgram++;
		},
		getProgramParameter(program, parameter) {
			calls.getProgramParameter.push(parameter);
			if (parameter === completionStatus) {
				program.polls++;
				return program.polls > completeAfterPolls;
			}
			if (parameter === this.LINK_STATUS) {
				return true;
			}
			if (parameter === this.VALIDATE_STATUS) {
				return validateStatus;
			}
			return true;
		},
		getProgramInfoLog() {
			return "mock program info";
		},
		deleteProgram() {},
		getUniformLocation(_program, name) {
			calls.getUniformLocation.push(name);
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

const CUSTOM_WEBGL_FRAGMENT_DEPTH = /* glsl */ `
#version 300 es
precision highp float;
void main() {
}
`;

function createScenePassContext(overrides = {}) {
	return {
		viewCamera: {
			viewProjectionMatrix: Matrix4.identity(),
		},
		incremental: {
			enabled: false,
			forceFullFrame: false,
			dirtyRects: [],
		},
		attachments: {
			width: 64,
			height: 64,
		},
		scene: {
			opaquePackets: [],
			transparentPackets: [],
			spatialIndex: null,
		},
		...overrides,
	};
}

function createEarlyZScenePassHost(gl, options = {}) {
	const colorProgram =
		options.colorProgram ?? {
			program: { id: "color-program" },
			uniforms: {},
		};
	const depthProgram =
		options.depthProgram ?? {
			program: { id: "depth-program" },
			uniforms: {
				model: "uModel",
				normalMatrix: null,
				baseColor: "uBaseColor",
				alpha: "uAlpha",
				baseMap: "uBaseMap",
				hasBaseMap: "uHasBaseMap",
				baseMapUV: "uBaseMapUV",
				baseMapTransformA: "uBaseMapTransformA",
				baseMapTransformB: "uBaseMapTransformB",
				doubleSided: null,
				customSamplers: {},
				customUniforms: {},
			},
		};
	return {
		_gl: gl,
		_programs: {
			getSceneProgram() {
				return colorProgram;
			},
			getSceneDepthPrepassProgram() {
				return options.depthProgramAvailable === false ? null : depthProgram;
			},
		},
		_geometry: {
			getGeometry(packet) {
				return options.getGeometry?.(packet) ?? {
					vao: { id: `vao-${packet.id}` },
					topology: gl.TRIANGLES,
					indexCount: 3,
					indexType: 5123,
				};
			},
		},
		_textures: {
			getBaseColorTexture(texture) {
				return options.getBaseColorTexture?.(texture) ?? {
					texture: texture ? { id: "base-map" } : null,
					isLinear: false,
				};
			},
		},
		_sceneFramebuffer: { id: "scene-fbo" },
		_sceneNormalTexture: options.sceneNormalTexture ?? null,
		_oitPassMode: 0,
		_width: 64,
		_height: 64,
		_maxTextureImageUnits: 32,
		_modelMatrixCache: new Map(),
		_modelMatrixKeysThisFrame: new Set(),
		_prevViewProjection: null,
		_taaHistoryValid: false,
		_isIncrementalPartial(context) {
			return context.incremental?.enabled === true &&
				context.incremental.forceFullFrame !== true &&
				(context.incremental.dirtyRects?.length ?? 0) > 0;
		},
		_resolveDirtyRects(context) {
			return context.incremental?.dirtyRects?.length ?
				context.incremental.dirtyRects
			:	[{ x: 0, y: 0, width: 64, height: 64 }];
		},
		_resolvePacketsForRect(context, packets, rect) {
			return options.resolvePacketsForRect?.(context, packets, rect) ?? packets;
		},
		_setScissorRect(x, y, width, height) {
			gl.scissor(x, y, width, height);
		},
		_bindGlobalUniforms() {},
		_setCullMode() {},
		_drawPacket(sceneProgram, packet, transparentPass, context, drawOptions) {
			drawWebGLPacket(this, sceneProgram, packet, transparentPass, context, drawOptions);
		},
		_bindShaderMaterialTextures() {},
		_bindShaderMaterialUniforms() {},
	};
}

function createEarlyZPacket(id, material = new Material()) {
	return {
		id,
		meshInstance: { id: `mesh-${id}`, skeleton: null },
		material,
		worldMatrix: Matrix4.identity(),
		normalMatrix: Matrix4.identity(),
	};
}

function createShadowPassHost(gl, options = {}) {
	let cullModeCalls = 0;
	return {
		gl,
		programs: {
			getShadowDepthProgram() {
				return {
					program: { id: "shadow-depth" },
					uniforms: { mvp: "uMvp" },
				};
			},
			getShadowTransmittanceProgram() {
				return {
					program: { id: "shadow-transmittance" },
					uniforms: {
						mvp: "uMvp",
						transmittance: "uTransmittance",
					},
				};
			},
		},
		geometry: {
			getGeometry(packet) {
				return options.getGeometry?.(packet) ?? {
					vao: { id: `shadow-vao-${packet.id}` },
					topology: gl.TRIANGLES,
					indexCount: 3,
					indexType: 5123,
				};
			},
		},
		_setCullMode() {
			cullModeCalls++;
		},
		getLightState() {
			return null;
		},
		getSceneFramebuffer() {
			return null;
		},
		getViewportSize() {
			return { width: 64, height: 64 };
		},
		getMaxTextureSize() {
			return 4096;
		},
		get cullModeCalls() {
			return cullModeCalls;
		},
	};
}

function createShadowPacket(material = new Material()) {
	return {
		id: "shadow-packet",
		meshInstance: { id: "shadow-mesh", skeleton: null },
		material,
		worldMatrix: Matrix4.identity(),
	};
}

function testLightCollectorLimitsAndWarnings() {
	const warnings = [];
	const warn = (key, message) => warnings.push({ key, message });
	const lights = [new AmbientLight()];

	for (let i = 0; i < MAX_DIRECTIONAL_LIGHTS + 2; i++) {
		lights.push(new DirectionalLight({ intensity: 1 + i * 0.1 }));
	}
	for (let i = 0; i < MAX_POINT_LIGHTS + 2; i++) {
		lights.push(new PointLight({ range: 100 + i }));
	}
	for (let i = 0; i < MAX_SPOT_LIGHTS + 2; i++) {
		lights.push(new SpotLight({ range: 100 + i }));
	}

	const state = collectWebGLLights(lights, true, warn);
	assert.equal(state.directionalLights.length, MAX_DIRECTIONAL_LIGHTS);
	assert.equal(state.pointLights.length, MAX_POINT_LIGHTS);
	assert.equal(state.spotLights.length, MAX_SPOT_LIGHTS);
	assert.ok(warnings.some((warning) => warning.key === "webgl-directional-light-limit"));
	assert.ok(warnings.some((warning) => warning.key === "webgl-point-light-limit"));
	assert.ok(warnings.some((warning) => warning.key === "webgl-spot-light-limit"));
}

function testLightProbeAmbientAndReflectionProbeSpecularCollection() {
	const warnings = [];
	const warn = (key, message) => warnings.push({ key, message });
	const sh = SH.empty();
	sh[0] = { r: 120, g: 60, b: 30 };
	const probeMap = new Texture({
		data: new Float32Array(4 * 2 * 4),
		width: 4,
		height: 2,
		colorSpace: "HDR",
	});
	probeMap.mipmaps = [
		new Float32Array(4 * 2 * 4),
		new Float32Array(2 * 1 * 4),
	];
	const lightProbe = new LightProbe({ sh });
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
	const globalProbe = new LightProbe({ sh: SH.empty() });

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

function testProgramCompilerParallelWarmupDefersStatusQueries() {
	const gl = createProgramWarmupTrackingGL({
		parallel: true,
		completeAfterPolls: 2,
	});
	const compiler = new WebGLProgramCompiler(gl);
	const slot = createCompilerSlot(compiler, "WebGLFXAAProgram", [
		"uSourceMap",
		"uTexelSize",
	]);
	const handle = slot.warmup();

	assert.equal(gl.calls.linkProgram, 1);
	assert.equal(gl.calls.getShaderParameter.length, 0);
	assert.equal(
		gl.calls.getProgramParameter.includes(gl.LINK_STATUS),
		false
	);
	assert.equal(gl.calls.getUniformLocation.length, 0);
	assert.equal(handle.isComplete(), false);
	assert.equal(handle.isComplete(), false);
	assert.equal(gl.calls.getShaderParameter.length, 0);
	assert.equal(
		gl.calls.getProgramParameter.includes(gl.LINK_STATUS),
		false
	);

	assert.equal(handle.isComplete(), true);
	handle.finalize();

	assert.deepEqual(gl.calls.getShaderParameter, [
		gl.COMPILE_STATUS,
		gl.COMPILE_STATUS,
	]);
	assert.ok(gl.calls.getProgramParameter.includes(gl.LINK_STATUS));
	assert.ok(gl.calls.getUniformLocation.includes("uSourceMap"));
	assert.ok(gl.calls.getUniformLocation.includes("uTexelSize"));
}

function testProgramCompilerFallbackWarmupBatchesBeforeFinalize() {
	const gl = createProgramWarmupTrackingGL();
	const compiler = new WebGLProgramCompiler(gl);

	const fxaa = createCompilerSlot(compiler, "WebGLFXAAProgram").warmup();
	const present = createCompilerSlot(compiler, "WebGLPresentProgram").warmup();

	assert.equal(gl.calls.linkProgram, 2);
	assert.equal(gl.calls.getShaderParameter.length, 0);
	assert.equal(
		gl.calls.getProgramParameter.includes(gl.LINK_STATUS),
		false
	);

	fxaa.finalize();
	present.finalize();

	assert.equal(gl.calls.getShaderParameter.length, 4);
	assert.equal(
		gl.calls.getProgramParameter.filter((parameter) => parameter === gl.LINK_STATUS)
			.length,
		2
	);
}

function testProgramCompilerTryGetDefersFallbackFinalization() {
	const gl = createProgramWarmupTrackingGL();
	let pendingNotifications = 0;
	const compiler = new WebGLProgramCompiler(
		gl,
		undefined,
		undefined,
		{
			onProgramCompilePending: () => {
				pendingNotifications++;
			},
		},
	);
	const fxaaSlot = createCompilerSlot(compiler, "WebGLFXAAProgram", [
		"uSourceMap",
		"uTexelSize",
	]);
	const bloomSlot = createCompilerSlot(compiler, "WebGLBloomProgram", [
		"uBloomParams",
	]);

	compiler.beginFrame();
	assert.equal(fxaaSlot.tryGet(), null);
	assert.equal(bloomSlot.tryGet(), null);
	assert.equal(pendingNotifications, 1);
	assert.equal(gl.calls.linkProgram, 2);
	assert.equal(gl.calls.getShaderParameter.length, 0);
	assert.equal(
		gl.calls.getProgramParameter.includes(gl.LINK_STATUS),
		false
	);
	assert.equal(gl.calls.getUniformLocation.length, 0);

	compiler.beginFrame();
	assert.equal(fxaaSlot.tryGet(), null);
	assert.equal(bloomSlot.tryGet(), null);
	assert.equal(pendingNotifications, 2);
	assert.equal(gl.calls.getShaderParameter.length, 0);
	assert.equal(
		gl.calls.getProgramParameter.includes(gl.LINK_STATUS),
		false
	);

	compiler.beginFrame();
	const fxaa = fxaaSlot.tryGet();
	const bloomPending = bloomSlot.tryGet();
	assert.ok(fxaa);
	assert.equal(bloomPending, null);
	assert.equal(pendingNotifications, 3);
	assert.equal(gl.calls.getShaderParameter.length, 2);
	assert.equal(
		gl.calls.getProgramParameter.filter((parameter) => parameter === gl.LINK_STATUS)
			.length,
		1
	);
	assert.ok(gl.calls.getUniformLocation.includes("uSourceMap"));
	assert.ok(gl.calls.getUniformLocation.includes("uTexelSize"));
	assert.equal(
		gl.calls.getUniformLocation.includes("uBloomParams"),
		false
	);

	compiler.beginFrame();
	const bloom = bloomSlot.tryGet();
	assert.ok(bloom);
	assert.equal(pendingNotifications, 3);
	assert.equal(gl.calls.getShaderParameter.length, 4);
	assert.equal(
		gl.calls.getProgramParameter.filter((parameter) => parameter === gl.LINK_STATUS)
			.length,
		2
	);
	assert.ok(gl.calls.getUniformLocation.includes("uBloomParams"));
}

function testProgramCompilerValidationIsOptIn() {
	const gl = createProgramWarmupTrackingGL({
		validateStatus: false,
	});
	const compiler = new WebGLProgramCompiler(gl);

	createCompilerSlot(compiler, "WebGLFXAAProgram").get();

	assert.equal(gl.calls.validateProgram, 0);
	assert.equal(
		gl.calls.getProgramParameter.includes(gl.VALIDATE_STATUS),
		false
	);
}

function testProgramCompilerValidationWarnsWhenEnabled() {
	const warnings = [];
	const gl = createProgramWarmupTrackingGL({
		validateStatus: false,
	});
	const compiler = new WebGLProgramCompiler(
		gl,
		undefined,
		undefined,
		{
			validatePrograms: true,
			warn: (key, message) => warnings.push({ key, message }),
		},
	);

	createCompilerSlot(compiler, "WebGLFXAAProgram").get();

	assert.equal(gl.calls.validateProgram, 1);
	assert.ok(gl.calls.getProgramParameter.includes(gl.VALIDATE_STATUS));
	assert.ok(
		warnings.some((warning) =>
			warning.key.startsWith("webgl-program-validate-WebGLFXAAProgram")
		)
	);
}

function testProgramCompilerSlotLifecycleAndStaleWarmup() {
	const gl = createProgramWarmupTrackingGL({
		parallel: true,
		completeAfterPolls: 100,
	});
	let deletedPrograms = 0;
	gl.deleteProgram = () => {
		deletedPrograms++;
	};
	const compiler = new WebGLProgramCompiler(gl);
	let sourceRevision = 0;
	let sourceResolutions = 0;
	const createSlot = () => compiler.createSlot({
		label: "WebGLLifecycleProgram",
		vertex: () => {
			sourceResolutions++;
			return `${CUSTOM_WEBGL_VERTEX}\n// revision ${sourceRevision}`;
		},
		fragment: () => {
			sourceResolutions++;
			return `${CUSTOM_WEBGL_FRAGMENT}\n// revision ${sourceRevision}`;
		},
		reflect: (_gl, program) => ({ program }),
	});
	const slot = createSlot();

	assert.throws(createSlot, /already registered/);
	const staleHandle = slot.warmup();
	assert.equal(compiler.getCompileState(slot.label), "pending");
	sourceRevision++;
	slot.invalidate();
	assert.equal(compiler.getCompileState(slot.label), "idle");
	assert.throws(() => staleHandle.isComplete(), /became stale/);
	assert.throws(() => staleHandle.finalize(), /became stale/);
	assert.equal(deletedPrograms, 1);

	const first = slot.get();
	assert.ok(first.program);
	assert.equal(sourceResolutions, 4);
	sourceRevision++;
	slot.invalidate();
	const second = slot.get();
	assert.ok(second.program);
	assert.notEqual(second.program, first.program);
	assert.equal(sourceResolutions, 6);
	assert.equal(deletedPrograms, 2);

	slot.destroy();
	slot.destroy();
	assert.equal(deletedPrograms, 3);
	const replacement = createSlot();
	replacement.get();
	compiler.destroy();
	compiler.destroy();
	assert.equal(deletedPrograms, 4);
	assert.throws(() => replacement.get(), /destroyed/);
}

async function testProgramWarmupQueuePrioritizesCoreWork() {
	const events = [];
	const queue = new WebGLProgramWarmupQueue({
		waitForSlice: async () => {},
	});
	const yieldController = { yieldIfNeeded: async () => {} };
	const createHandle = (label) => ({
		label,
		isComplete: () => true,
		finalize: () => {
			events.push(`finalize:${label}`);
		},
	});

	queue.enqueue({
		label: "post",
		priority: "postprocess",
		action: () => {
			events.push("action:post");
			return [createHandle("post")];
		},
	});
	queue.enqueue({
		label: "core",
		priority: "core",
		action: () => {
			events.push("action:core");
			return [createHandle("core")];
		},
	});
	queue.enqueue({
		label: "optional",
		priority: "optional",
		action: () => {
			events.push("action:optional");
			return [createHandle("optional")];
		},
	});

	const result = await queue.run(yieldController);

	assert.deepEqual(events, [
		"action:core",
		"finalize:core",
		"action:optional",
		"finalize:optional",
		"action:post",
		"finalize:post",
	]);
	assert.equal(result.compiled, 3);
	assert.equal(result.failed, 0);
}

async function testProgramWarmupQueueFinalizesOneProgramPerSlice() {
	let slice = 0;
	const finalizedAt = [];
	const queue = new WebGLProgramWarmupQueue({
		waitForSlice: async () => {
			slice++;
		},
	});
	const yieldController = { yieldIfNeeded: async () => {} };

	queue.enqueue({
		label: "batch",
		priority: "core",
		action: () => ["a", "b", "c"].map((label) => ({
			label,
			isComplete: () => true,
			finalize: () => {
				finalizedAt.push(slice);
			},
		})),
	});

	const result = await queue.run(yieldController);

	assert.deepEqual(finalizedAt, [0, 1, 2]);
	assert.equal(result.compiled, 3);
	assert.equal(result.failed, 0);
}

async function testProgramWarmupQueueReportsStaleHandles() {
	const queue = new WebGLProgramWarmupQueue({
		waitForSlice: async () => {},
	});
	const yieldController = { yieldIfNeeded: async () => {} };

	queue.enqueue({
		label: "stale",
		priority: "core",
		action: () => [{
			label: "stale",
			isComplete: () => {
				throw new Error("stale handle");
			},
			finalize: () => {
				throw new Error("should not finalize");
			},
		}],
	});

	const result = await queue.run(yieldController);

	assert.equal(result.compiled, 0);
	assert.equal(result.failed, 1);
	assert.equal(result.errors[0].label, "stale");
	assert.match(String(result.errors[0].error), /stale handle/);
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
		scene.shadows.createCascaded({
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
	const shader = getTestSceneShader();
	const doubleSidedNormalFlip =
		"if (uDoubleSided == 1 && dot(normal, viewDir) < 0.0)";
	const firstFlipIndex = shader.fragment.indexOf(doubleSidedNormalFlip);
	const secondFlipIndex = shader.fragment.indexOf(
		doubleSidedNormalFlip,
		firstFlipIndex + doubleSidedNormalFlip.length
	);
	const normalMapIndex = shader.fragment.indexOf("normal = applyNormalMap(");
	assert.ok(shader.fragment.includes("dot(normal, lightDirection) <= 0.0"));
	assert.ok(shader.fragment.includes("uniform int uDoubleSided;"));
	assert.ok(firstFlipIndex >= 0);
	assert.ok(secondFlipIndex > normalMapIndex);
}

function testSceneShaderKeepsClusteredFragmentLightLimitPlaceholder() {
	const shader = getTestSceneShader();
	assert.ok(
		shader.fragment.includes(
			"__WEBGL_MAX_CLUSTER_LIGHTS_PER_FRAGMENT__"
		)
	);
}

function testSceneShaderUsesFlippedShadowNormal() {
	const shader = getTestSceneShader();
	const doubleSidedNormalFlip =
		"if (uDoubleSided == 1 && dot(normal, viewDir) < 0.0)";
	const firstFlipIndex = shader.fragment.indexOf(doubleSidedNormalFlip);
	const secondFlipIndex = shader.fragment.indexOf(
		doubleSidedNormalFlip,
		firstFlipIndex + doubleSidedNormalFlip.length
	);
	const shadowNormalIndex = shader.fragment.indexOf("vec3 shadowNormal = normal;");
	assert.ok(firstFlipIndex >= 0);
	assert.ok(secondFlipIndex > firstFlipIndex);
	assert.ok(shadowNormalIndex > secondFlipIndex);
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

function testShadowPassDisablesCullFaceForDepthAndTransmittance() {
	const gl = createScenePassCaptureGL();
	const host = createShadowPassHost(gl);
	const pass = new WebGLShadowPass(host);
	const material = new Material({
		doubleSided: false,
		cullMode: "front",
	});
	const packet = createShadowPacket(material);
	const depthProgram = {
		program: { id: "shadow-depth" },
		uniforms: { mvp: "uMvp" },
	};
	const transmittanceProgram = {
		program: { id: "shadow-transmittance" },
		uniforms: {
			mvp: "uMvp",
			transmittance: "uTransmittance",
		},
	};

	pass.drawShadowPacket(depthProgram, packet, Matrix4.identity());
	pass.drawShadowTransmittancePacket(
		transmittanceProgram,
		packet,
		Matrix4.identity()
	);

	assert.equal(host.cullModeCalls, 0);
	assert.deepEqual(gl.calls.disable, [gl.CULL_FACE, gl.CULL_FACE]);
	assert.equal(gl.calls.drawElements.length, 2);
}

function testSceneShaderIncludesReflectionProbeUniforms() {
	const shader = getTestSceneShader();
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
	const shader = getTestSceneShader();
	assert.ok(shader.fragment.includes("uniform int uLocalLightProbeCount;"));
	assert.ok(shader.fragment.includes("uLocalLightProbeWorldToProbeRow0"));
	assert.ok(shader.fragment.includes("uLocalLightProbeCoeffs"));
	assert.ok(shader.fragment.includes("selectTopTwoLocalLightProbes"));
	assert.ok(shader.fragment.includes("sampleBlendedLocalLightProbeIrradiance"));
	assert.ok(shader.fragment.includes("sampleBlendedLocalLightProbeRadiance"));
}

function testSceneShaderFitsCommonWebGLTextureUnitLimit() {
	const shader = getTestSceneShader();
	const samplerMatches = shader.fragment.match(/\buniform\s+sampler2D\b/g) ?? [];

	assert.equal(samplerMatches.length, 16);
	assert.ok(
		shader.fragment.includes(
			"uniform vec3 uSHAmbientCoeffs[SH_COEFFICIENT_COUNT];"
		)
	);
	assert.ok(!shader.fragment.includes("uniform sampler2D uSHAmbientCoeffs;"));
	assert.ok(!shader.fragment.includes("uIrradianceProbeGridCoeffs"));
	assert.ok(shader.fragment.includes("vec3 sampleDiffuseProbeIrradiance"));
	assert.ok(!shader.fragment.includes("sampleIrradianceProbeGridIrradiance"));
}

function testSceneShaderIncludesIrradianceProbeGridWhenEnabled() {
	const shader = ShaderSource.get("webgl.scene.raw", {
		limits: {
			...TEST_SCENE_LIMITS,
			enableIrradianceProbeGrid: true,
		},
	});
	const samplerMatches = shader.fragment.match(/\buniform\s+sampler2D\b/g) ?? [];

	assert.equal(samplerMatches.length, 17);
	assert.ok(shader.fragment.includes("uniform sampler2D uIrradianceProbeGridCoeffs;"));
	assert.ok(shader.fragment.includes("uIrradianceProbeGridWorldToGridRow0"));
	assert.ok(shader.fragment.includes("sampleIrradianceProbeGridIrradiance"));
	assert.ok(shader.fragment.includes("return mix(fallback, gridAmbientBase.rgb"));
}

function testSceneShaderIncludesPBRTextureAndUV1Pipeline() {
	const shader = getTestSceneShader();
	assert.ok(shader.vertex.includes("layout(location = 3) in vec2 aUv1;"));
	assert.ok(shader.vertex.includes("layout(location = 6) in vec4 aTangent;"));
	assert.ok(shader.vertex.includes("out vec2 vUv1;"));
	assert.ok(shader.vertex.includes("out vec4 vTangent;"));
	assert.ok(shader.fragment.includes("in vec2 vUv1;"));
	assert.ok(shader.fragment.includes("in vec4 vTangent;"));
	assert.ok(shader.fragment.includes("uniform sampler2D uMetallicRoughnessMap;"));
	assert.ok(shader.fragment.includes("uniform sampler2D uNormalMap;"));
	assert.ok(shader.fragment.includes("uniform sampler2D uOcclusionMap;"));
	assert.ok(shader.fragment.includes("uniform int uBaseMapUV;"));
	assert.ok(shader.fragment.includes("vec2 resolveUV(int uvSet) {"));
	assert.ok(shader.fragment.includes("uniform vec4 uBaseMapTransformA;"));
	assert.ok(shader.fragment.includes("uniform vec2 uBaseMapTransformB;"));
	assert.ok(shader.fragment.includes("vec2 resolveMappedUV("));
	assert.ok(shader.fragment.includes("bool resolveTangentFrame("));
	assert.ok(shader.fragment.includes("vec4 tangent,"));
}

function testSceneShaderIncludesOITPassMode() {
	const shader = getTestSceneShader();

	assert.ok(shader.fragment.includes("uniform int uOITPassMode;"));
	assert.ok(shader.fragment.includes("float resolveOITWeight("));
	assert.ok(shader.fragment.includes("if (uOITPassMode == 1)"));
	assert.ok(shader.fragment.includes("if (uOITPassMode == 2)"));
}

function testParticleShaderIncludesOITPassMode() {
	const shader = ShaderSource.get("webgl.part.particleFragment.raw");

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
			tangents: new Float32Array([
				1, 0, 0, 1,
				0, 1, 0, -1,
				0, 0, 1, 1,
			]),
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
	assert.equal(gl.calls.vertexData.length, 54);
	assert.equal(gl.calls.vertexData[8], 0.25);
	assert.equal(gl.calls.vertexData[9], 0.5);
	assert.equal(gl.calls.vertexData[10], 0.125);
	assert.equal(gl.calls.vertexData[11], 0.25);
	assert.equal(gl.calls.vertexData[12], 0.875);
	assert.equal(gl.calls.vertexData[13], 0.75);
	assert.deepEqual(Array.from(gl.calls.vertexData.slice(14, 18)), [1, 0, 0, 1]);
	assert.equal(gl.calls.vertexData[26], 0.75);
	assert.equal(gl.calls.vertexData[27], 0.5);
	assert.equal(gl.calls.vertexData[28], 0.375);
	assert.equal(gl.calls.vertexData[29], 0.5);
	assert.equal(gl.calls.vertexData[30], 0.625);
	assert.equal(gl.calls.vertexData[31], 0.5);
	assert.deepEqual(Array.from(gl.calls.vertexData.slice(32, 36)), [0, 1, 0, -1]);
	assert.ok(
		gl.calls.attributePointers.some(
			(call) =>
				call.index === 3 &&
				call.size === 2 &&
				call.stride === 72 &&
				call.offset === 32
		)
	);
	assert.ok(
		gl.calls.attributePointers.some(
			(call) =>
				call.index === 4 &&
				call.size === 2 &&
				call.stride === 72 &&
				call.offset === 40
		)
	);
	assert.ok(
		gl.calls.attributePointers.some(
			(call) =>
				call.index === 5 &&
				call.size === 2 &&
				call.stride === 72 &&
				call.offset === 48
		)
	);
	assert.ok(
		gl.calls.attributePointers.some(
			(call) =>
				call.index === 6 &&
				call.size === 4 &&
				call.stride === 72 &&
				call.offset === 56
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
	assert.deepEqual(gl.calls.depthFunc, [gl.LESS]);

	material.depthWrite = true;
	drawWebGLPacket(host, sceneProgram, packet, false, {});
	assert.deepEqual(gl.calls.depthMask, [false, true]);
	assert.deepEqual(gl.calls.depthFunc, [gl.LESS, gl.LESS]);
}

function testEarlyZPrepassUsesDepthOnlyStateAndDrivesColorLEQUAL() {
	const gl = createScenePassCaptureGL();
	const material = new Material();
	const packet = createEarlyZPacket("early-z", material);
	const context = createScenePassContext();
	const host = createEarlyZScenePassHost(gl);

	const prepassedIds = renderWebGLEarlyZPrepass(host, context, [packet]);

	assert.equal(prepassedIds.has(packet.id), true);
	assert.deepEqual(gl.calls.drawBuffers[0], [gl.NONE]);
	assert.deepEqual(gl.calls.colorMask[0], [false, false, false, false]);
	assert.deepEqual(
		gl.calls.colorMask[gl.calls.colorMask.length - 1],
		[true, true, true, true]
	);
	assert.deepEqual(
		gl.calls.drawBuffers[gl.calls.drawBuffers.length - 1],
		[gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]
	);
	assert.ok(gl.calls.disable.includes(gl.BLEND));
	assert.ok(gl.calls.depthMask.includes(true));

	renderWebGLPackets(host, context, [packet], false, {
		earlyZPacketIds: prepassedIds,
	});

	assert.ok(gl.calls.depthFunc.includes(gl.LEQUAL));
	assert.equal(gl.calls.depthMask.includes(false), true);
}

function testEarlyZPrepassSkipsDepthWriteDisabledPackets() {
	const gl = createScenePassCaptureGL();
	const material = new Material({
		depthWrite: false,
	});
	const packet = createEarlyZPacket("depth-read", material);
	const context = createScenePassContext();
	const host = createEarlyZScenePassHost(gl);

	const prepassedIds = renderWebGLEarlyZPrepass(host, context, [packet]);
	assert.equal(prepassedIds.size, 0);
	assert.equal(gl.calls.drawElements.length, 0);

	renderWebGLPackets(host, context, [packet], false, {
		earlyZPacketIds: prepassedIds,
	});
	assert.equal(gl.calls.depthFunc.includes(gl.LEQUAL), false);
	assert.equal(gl.calls.depthFunc.includes(gl.LESS), true);
	assert.equal(gl.calls.depthMask.includes(false), true);
}

function testBuiltInMaskDepthPrepassShaderContract() {
	const fragment = ShaderSource.get("webgl.part.sceneDepthPrepassFragment.raw");
	assert.ok(fragment.includes("uBaseColor.a"));
	assert.ok(fragment.includes("texture(uBaseMap"));
	assert.ok(fragment.includes("uAlpha.x"));
	assert.ok(fragment.includes("discard"));
}

function testEarlyZPrepassUsesDirtyRectPacketSelection() {
	const gl = createScenePassCaptureGL();
	const packetA = createEarlyZPacket("a");
	const packetB = createEarlyZPacket("b");
	const resolvedRects = [];
	const context = createScenePassContext({
		incremental: {
			enabled: true,
			forceFullFrame: false,
			dirtyRects: [
				{ x: 0, y: 0, width: 16, height: 16 },
				{ x: 32, y: 32, width: 16, height: 16 },
			],
		},
	});
	const host = createEarlyZScenePassHost(gl, {
		resolvePacketsForRect(_context, _packets, rect) {
			resolvedRects.push(rect);
			return rect.x === 0 ? [packetA] : [];
		},
	});

	const prepassedIds = renderWebGLEarlyZPrepass(host, context, [
		packetA,
		packetB,
	]);

	assert.deepEqual(resolvedRects, context.incremental.dirtyRects);
	assert.equal(prepassedIds.has(packetA.id), true);
	assert.equal(prepassedIds.has(packetB.id), false);
	assert.equal(gl.calls.scissor.length, 1);
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

function testShaderMaterialCustomTextureBinding() {
	const gl = createScenePassCaptureGL();
	const fakeTexture = { id: "test-tex", colorSpace: "Linear" };
	const material = new ShaderMaterial({
		textureBindings: [
			{ name: "customTex", texture: fakeTexture, webglUniform: "uCustomTex" }
		]
	});
	const customSamplers = {
		uCustomTex: "uCustomTex"
	};
	const sceneProgram = { uniforms: { customSamplers } };

	// Case 1: _maxTextureImageUnits is 32 (starts at WEBGL_TEXTURE_UNIT_CUSTOM_START = 17)
	{
		gl.calls.activeTextures = [];
		gl.calls.boundTextures = [];
		gl.calls.uniform1i = [];
		const host = {
			_gl: gl,
			_maxTextureImageUnits: 32,
			_textures: {
				getBaseColorTexture(texture) {
					return { texture };
				}
			}
		};
		bindWebGLShaderMaterialTextures(host, sceneProgram, material);
		assert.equal(gl.calls.activeTextures.length, 2);
		assert.equal(gl.calls.activeTextures[0], gl.TEXTURE0 + 17);
		assert.equal(gl.calls.activeTextures[1], gl.TEXTURE0 + 0);
		assert.deepEqual(gl.calls.boundTextures, [{ target: gl.TEXTURE_2D, texture: fakeTexture }]);
		assert.deepEqual(gl.calls.uniform1i, [{ location: "uCustomTex", value: 17 }]);
	}

	// Case 2: _maxTextureImageUnits is 16 (falls back to start at 8)
	{
		gl.calls.activeTextures = [];
		gl.calls.boundTextures = [];
		gl.calls.uniform1i = [];
		const host = {
			_gl: gl,
			_maxTextureImageUnits: 16,
			_textures: {
				getBaseColorTexture(texture) {
					return { texture };
				}
			}
		};
		bindWebGLShaderMaterialTextures(host, sceneProgram, material);
		assert.equal(gl.calls.activeTextures.length, 2);
		assert.equal(gl.calls.activeTextures[0], gl.TEXTURE0 + 8);
		assert.equal(gl.calls.activeTextures[1], gl.TEXTURE0 + 0);
		assert.deepEqual(gl.calls.boundTextures, [{ target: gl.TEXTURE_2D, texture: fakeTexture }]);
		assert.deepEqual(gl.calls.uniform1i, [{ location: "uCustomTex", value: 8 }]);
	}
}

function testSceneProgramDrawBuffersMatchFragmentOutputCount() {
	const gl = createScenePassCaptureGL();
	const material = new ShaderMaterial({
		uniformBindings: [],
	});
	const sceneProgram = {
		program: {},
		uniforms: {},
		targetMode: "single",
	};

	const packet = {
		id: "test-pkt",
		material,
		worldMatrix: Matrix4.identity(),
		normalMatrix: Matrix4.identity(),
		meshInstance: {
			id: "test-mesh",
			skeleton: null,
		},
	};

	const host = {
		_gl: gl,
		_geometry: {
			getGeometry() {
				return {
					vao: {},
					topology: gl.TRIANGLES,
					indexCount: 3,
					indexType: 5123,
				};
			},
		},
		_textures: {
			getBaseColorTexture() {
				return { texture: null, isLinear: false };
			},
		},
		_activeDrawBuffers: [gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1, gl.COLOR_ATTACHMENT2],
		_modelMatrixCache: new Map(),
		_modelMatrixKeysThisFrame: new Set(),
		_bindShaderMaterialTextures() {},
		_bindShaderMaterialUniforms() {},
		_setCullMode() {},
	};

	gl.calls.drawBuffers = [];
	drawWebGLPacket(host, sceneProgram, packet, false, {});

	// Should have changed draw buffers to [COLOR_ATTACHMENT0], then drawn, then restored back to original
	assert.deepEqual(gl.calls.drawBuffers, [
		[gl.COLOR_ATTACHMENT0],
		[gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1, gl.COLOR_ATTACHMENT2],
	]);

	host._activeDrawBuffers = [
		gl.COLOR_ATTACHMENT0,
		gl.COLOR_ATTACHMENT1,
		gl.COLOR_ATTACHMENT2,
		gl.COLOR_ATTACHMENT3,
		gl.COLOR_ATTACHMENT4,
	];
	gl.calls.drawBuffers = [];
	drawWebGLPacket(host, {
		program: {},
		uniforms: {},
		targetMode: "mrt",
		colorOutputCount: 3,
	}, packet, false, {});
	assert.deepEqual(gl.calls.drawBuffers, [
		[
			gl.COLOR_ATTACHMENT0,
			gl.COLOR_ATTACHMENT1,
			gl.COLOR_ATTACHMENT2,
		],
		[
			gl.COLOR_ATTACHMENT0,
			gl.COLOR_ATTACHMENT1,
			gl.COLOR_ATTACHMENT2,
			gl.COLOR_ATTACHMENT3,
			gl.COLOR_ATTACHMENT4,
		],
	]);

	gl.calls.drawBuffers = [];
	drawWebGLPacket(host, {
		program: {},
		uniforms: {},
		targetMode: "mrt",
		colorOutputCount: 5,
	}, packet, false, {});
	assert.deepEqual(gl.calls.drawBuffers, []);

	const drawError = new Error("draw failed");
	const originalDrawElements = gl.drawElements;
	gl.drawElements = () => {
		throw drawError;
	};
	assert.throws(
		() => drawWebGLPacket(host, {
			program: {},
			uniforms: {},
			targetMode: "mrt",
			colorOutputCount: 3,
		}, packet, false, {}),
		(error) => error === drawError,
	);
	assert.deepEqual(gl.calls.drawBuffers, [
		[
			gl.COLOR_ATTACHMENT0,
			gl.COLOR_ATTACHMENT1,
			gl.COLOR_ATTACHMENT2,
		],
		[
			gl.COLOR_ATTACHMENT0,
			gl.COLOR_ATTACHMENT1,
			gl.COLOR_ATTACHMENT2,
			gl.COLOR_ATTACHMENT3,
			gl.COLOR_ATTACHMENT4,
		],
	]);
	gl.drawElements = originalDrawElements;
}

function testWebGLBackendParticleDeltaTimeClamp() {
	const backend = new WebGLBackend();
	backend.attach({
		surface: { canvas: {} },
		events: { emit: () => {} },
	});
	const transient = new Map([
		[PARTICLE_SIM_DELTA_TIME_SECONDS_KEY, 1000],
	]);
	const deltaTimeSeconds = backend._resolveParticleDeltaTime({ transient });
	assert.equal(deltaTimeSeconds, 0.5);
}

async function testWebGLBackendWarmupDelegatesToCoordinator() {
	const backend = new WebGLBackend();
	backend.attach({
		surface: { canvas: {} },
		events: { emit: () => {} },
	});
	backend._frameServices = {
		warmupCoordinator: {
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
		},
	};
	const report = await backend.warmup({
		viewCamera: {},
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
	await ShaderSource.prepareMany([
		...WEBGL_SHADER_PARTS.flatMap((part) => [
			{ key: `webgl.part.${part}.raw` },
			{ key: `webgl.part.${part}.composite` },
		]),
		{ key: "webgl.scene.raw", params: { limits: TEST_SCENE_LIMITS } },
		{ key: "webgl.scene.composite", params: { limits: TEST_SCENE_LIMITS } },
		{
			key: "webgl.scene.raw",
			params: {
				limits: {
					maxDirectionalLights: MAX_DIRECTIONAL_LIGHTS,
					maxPointLights: MAX_POINT_LIGHTS,
					maxSpotLights: MAX_SPOT_LIGHTS,
				},
			},
		},
		{
			key: "webgl.scene.composite",
			params: {
				limits: {
					maxDirectionalLights: MAX_DIRECTIONAL_LIGHTS,
					maxPointLights: MAX_POINT_LIGHTS,
					maxSpotLights: MAX_SPOT_LIGHTS,
				},
			},
		},
		{
			key: "webgl.scene.raw",
			params: {
				limits: {
					maxDirectionalLights: MAX_DIRECTIONAL_LIGHTS,
					maxPointLights: MAX_POINT_LIGHTS,
					maxSpotLights: MAX_SPOT_LIGHTS,
					enableShadowTransmittance: true,
				},
			},
		},
		{
			key: "webgl.scene.composite",
			params: {
				limits: {
					maxDirectionalLights: MAX_DIRECTIONAL_LIGHTS,
					maxPointLights: MAX_POINT_LIGHTS,
					maxSpotLights: MAX_SPOT_LIGHTS,
					enableShadowTransmittance: true,
				},
			},
		},
		{
			key: "webgl.scene.raw",
			params: {
				limits: {
					...TEST_SCENE_LIMITS,
					enableShadowTransmittance: true,
				},
			},
		},
		{
			key: "webgl.scene.composite",
			params: {
				limits: {
					...TEST_SCENE_LIMITS,
					enableShadowTransmittance: true,
				},
			},
		},
		{
			key: "webgl.scene.raw",
			params: {
				limits: {
					...TEST_SCENE_LIMITS,
					enableIrradianceProbeGrid: true,
				},
			},
		},
		{
			key: "webgl.scene.composite",
			params: {
				limits: {
					...TEST_SCENE_LIMITS,
					enableIrradianceProbeGrid: true,
				},
			},
		},
	]);
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
	await testProgramLibraryCachesBuiltinSceneVariants();
	await testProgramLibraryShaderMaterialIgnoresBuiltinVariant();
	testProgramLibraryShaderMaterialCachesPerSceneTargetMode();
	testProgramLibraryBuiltinDepthPrepassProgram();
	testProgramLibraryShaderMaterialDepthPrepassProgram();
	testProgramLibraryShaderMaterialDepthPrepassMissingSourceDiagnostics();
	testProgramLibraryShaderMaterialMissingSourceFallsBack();
	testProgramLibraryWarnModeFallsBackOnCustomCompileFailure();
	testProgramLibraryRuntimeRevisionInvalidatesCustomCache();
	testProgramOwnershipSeparatesPostProcessAndBackendPrograms();
	testProgramCompilerParallelWarmupDefersStatusQueries();
	testProgramCompilerFallbackWarmupBatchesBeforeFinalize();
	testProgramCompilerTryGetDefersFallbackFinalization();
	testProgramCompilerValidationIsOptIn();
	testProgramCompilerValidationWarnsWhenEnabled();
	testProgramCompilerSlotLifecycleAndStaleWarmup();
	await testProgramWarmupQueuePrioritizesCoreWork();
	await testProgramWarmupQueueFinalizesOneProgramPerSlice();
	await testProgramWarmupQueueReportsStaleHandles();
	testLightCollectorShadowBias();
	testLightCollectorPCSSShadowParams();
	testLightCollectorDirectionalCSMShadowData();
	testSceneShaderBackLitShadowGuard();
	testSceneShaderKeepsClusteredFragmentLightLimitPlaceholder();
	testSceneShaderUsesFlippedShadowNormal();
	testShadowPassDisablesCullFaceForDepthAndTransmittance();
	testSceneShaderIncludesReflectionProbeUniforms();
	testSceneShaderIncludesLocalizedLightProbeUniforms();
	testSceneShaderFitsCommonWebGLTextureUnitLimit();
	testSceneShaderIncludesIrradianceProbeGridWhenEnabled();
	testSceneShaderIncludesPBRTextureAndUV1Pipeline();
	testSceneShaderIncludesOITPassMode();
	testParticleShaderIncludesOITPassMode();
	testGeometryRegistryRejectsOutOfRangeIndices();
	testGeometryRegistryRetriesAfterUploadAllocationFailure();
	testGeometryRegistryUploadsUV1Attribute();
	testDrawWebGLPacketBindsPBRTexturesAndUVSets();
	testDrawWebGLPacketBindsAnisotropyMapWhenSharedSlotIsFree();
	testDrawWebGLPacketAppliesMaterialDepthWriteState();
	testEarlyZPrepassUsesDepthOnlyStateAndDrivesColorLEQUAL();
	testEarlyZPrepassSkipsDepthWriteDisabledPackets();
	testBuiltInMaskDepthPrepassShaderContract();
	testEarlyZPrepassUsesDirtyRectPacketSelection();
	testShaderMaterialCustomUniformBinding();
	testShaderMaterialCustomTextureBinding();
	testSceneProgramDrawBuffersMatchFragmentOutputCount();
	testWebGLBackendParticleDeltaTimeClamp();
	await testWebGLBackendWarmupDelegatesToCoordinator();
	console.log("WebGL backend v2 unit tests passed");
}

await run();
