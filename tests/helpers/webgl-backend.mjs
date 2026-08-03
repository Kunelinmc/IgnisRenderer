import { Matrix4 } from "../../src/maths/Matrix4.ts";
import { Logger } from "../../src/foundation/Logger.ts";
import { CubeTexture } from "../../src/core/CubeTexture.ts";
import { Material } from "../../src/materials/Material.ts";
import { WebGLProgramLibrary } from "../../src/backends/webgl/WebGLProgramLibrary.ts";
import { drawWebGLPacket } from "../../src/backends/webgl/WebGLScenePass.ts";
import {
	MAX_DIRECTIONAL_LIGHTS,
	MAX_POINT_LIGHTS,
	MAX_SPOT_LIGHTS,
} from "../../src/backends/constants.ts";
import { ShaderSource, WEBGL_SHADER_PARTS } from "../../src/shaders/ShaderSource.ts";

export const TEST_SCENE_LIMITS = {
	maxDirectionalLights: 4,
	maxPointLights: 4,
	maxSpotLights: 4,
};

export const PROGRAM_LIBRARY_SCENE_LIMITS = {
	maxDirectionalLights: MAX_DIRECTIONAL_LIGHTS,
	maxPointLights: MAX_POINT_LIGHTS,
	maxSpotLights: MAX_SPOT_LIGHTS,
};

export function getTestSceneShader() {
	return ShaderSource.get("webgl.scene.raw", {
		limits: TEST_SCENE_LIMITS,
	});
}

export function createTinyCubeTexture(mips = 1, value = 1) {
	const createFace = () => new Float32Array([value, value, value, 1]);
	const faceMipmaps = [];
	for (let level = 1; level < mips; level++) {
		faceMipmaps.push(Array.from({ length: 6 }, () => createFace()));
	}
	return new CubeTexture({
		faces: Array.from({ length: 6 }, () => createFace()),
		faceMipmaps,
		size: 1,
		colorSpace: "HDR",
	});
}

export function createProgramLibrary(gl, warn, shaderRuntime, shaderCompileStage) {
	return new WebGLProgramLibrary(gl, warn, shaderRuntime, shaderCompileStage);
}

export function createTestBuiltinSceneVariant(overrides = {}) {
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

export async function prepareTestBuiltinSceneVariant(variant) {
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

export function createCompilerSlot(compiler, label, uniformNames = []) {
	return compiler.createSlot({
		label,
		vertex: () => CUSTOM_WEBGL_VERTEX,
		fragment: () => CUSTOM_WEBGL_FRAGMENT,
		reflect(gl, program) {
			return {
				program,
				uniforms: Object.fromEntries(
					uniformNames.map((name) => [name, gl.getUniformLocation(program, name)]),
				),
			};
		},
	});
}

export function createProgramCompileFailGL() {
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

export function createGeometryTestGL() {
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

export function createRetryGeometryTestGL() {
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

export function createGeometryCaptureGL() {
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

export function createScenePassCaptureGL() {
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

export function createShadowRasterCaptureGL() {
	const gl = createScenePassCaptureGL();
	let handle = 0;
	gl.DEPTH_ATTACHMENT = 0x8d00;
	gl.DEPTH_COMPONENT24 = 0x81a6;
	gl.DEPTH_COMPONENT = 0x1902;
	gl.UNSIGNED_INT = 0x1405;
	gl.RGBA8 = 0x8058;
	gl.RGBA = 0x1908;
	gl.UNSIGNED_BYTE = 0x1401;
	gl.TEXTURE_MIN_FILTER = 0x2801;
	gl.TEXTURE_MAG_FILTER = 0x2800;
	gl.TEXTURE_WRAP_S = 0x2802;
	gl.TEXTURE_WRAP_T = 0x2803;
	gl.NEAREST = 0x2600;
	gl.CLAMP_TO_EDGE = 0x812f;
	gl.FRAMEBUFFER_COMPLETE = 0x8cd5;
	gl.DEPTH_BUFFER_BIT = 0x0100;
	gl.COLOR_BUFFER_BIT = 0x4000;
	gl.ZERO = 0;
	gl.ONE = 1;
	gl.SRC_COLOR = 0x0300;
	gl.calls.viewport = [];
	gl.calls.deletedTextures = [];
	gl.calls.deletedFramebuffers = [];
	gl.createTexture = () => ({ id: `shadow-texture-${++handle}` });
	gl.createFramebuffer = () => ({ id: `shadow-framebuffer-${++handle}` });
	gl.deleteTexture = (texture) => gl.calls.deletedTextures.push(texture);
	gl.deleteFramebuffer = (framebuffer) => gl.calls.deletedFramebuffers.push(framebuffer);
	gl.texParameteri = () => {};
	gl.texImage2D = () => {};
	gl.framebufferTexture2D = () => {};
	gl.readBuffer = () => {};
	gl.checkFramebufferStatus = () => gl.FRAMEBUFFER_COMPLETE;
	gl.clearDepth = () => {};
	gl.clear = () => {};
	gl.blendFuncSeparate = () => {};
	gl.clearColor = () => {};
	gl.viewport = (x, y, width, height) => {
		gl.calls.viewport.push({ x, y, width, height });
	};
	return gl;
}

export function createProgramCaptureGL() {
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

export function createSelectiveCompileFailGL(failPattern) {
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

export function createProgramWarmupTrackingGL(options = {}) {
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
	const extension = options.parallel
		? {
				COMPLETION_STATUS_KHR: completionStatus,
			}
		: null;
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

export const CUSTOM_WEBGL_VERTEX = /* glsl */ `
#version 300 es
precision highp float;
layout(location = 0) in vec3 aPosition;
void main() {
	gl_Position = vec4(aPosition, 1.0);
}
`;

export const CUSTOM_WEBGL_FRAGMENT = /* glsl */ `
#version 300 es
precision highp float;
out vec4 outColor;
void main() {
	outColor = vec4(0.0, 1.0, 0.0, 1.0);
}
`;

export const CUSTOM_WEBGL_FRAGMENT_MRT = /* glsl */ `
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

export const CUSTOM_WEBGL_FRAGMENT_DEPTH = /* glsl */ `
#version 300 es
precision highp float;
void main() {
}
`;

export function createScenePassContext(overrides = {}) {
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

export function createEarlyZScenePassHost(gl, options = {}) {
	const colorProgram = options.colorProgram ?? {
		program: { id: "color-program" },
		uniforms: {},
	};
	const depthProgram = options.depthProgram ?? {
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
				return (
					options.getGeometry?.(packet) ?? {
						vao: { id: `vao-${packet.id}` },
						topology: gl.TRIANGLES,
						indexCount: 3,
						indexType: 5123,
					}
				);
			},
		},
		_textures: {
			getBaseColorTexture(texture) {
				return (
					options.getBaseColorTexture?.(texture) ?? {
						texture: texture ? { id: "base-map" } : null,
						isLinear: false,
					}
				);
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
			return (
				context.incremental?.enabled === true &&
				context.incremental.forceFullFrame !== true &&
				(context.incremental.dirtyRects?.length ?? 0) > 0
			);
		},
		_resolveDirtyRects(context) {
			return context.incremental?.dirtyRects?.length
				? context.incremental.dirtyRects
				: [{ x: 0, y: 0, width: 64, height: 64 }];
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

export function createEarlyZPacket(id, material = new Material()) {
	return {
		id,
		meshInstance: { id: `mesh-${id}`, skeleton: null },
		material,
		worldMatrix: Matrix4.identity(),
		normalMatrix: Matrix4.identity(),
	};
}

export function createShadowPassHost(gl, options = {}) {
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
				return (
					options.getGeometry?.(packet) ?? {
						vao: { id: `shadow-vao-${packet.id}` },
						topology: gl.TRIANGLES,
						indexCount: 3,
						indexType: 5123,
					}
				);
			},
		},
		maxTextureSize: 4096,
		get cullModeCalls() {
			return cullModeCalls;
		},
	};
}

export function createShadowRasterPlan({ casterPackets = [], transmitterPackets = [] } = {}) {
	return {
		atlasTileSize: 64,
		atlasWidth: 256,
		atlasHeight: 192,
		slices: [
			{
				kind: "directional",
				lightIndex: 0,
				cascadeIndex: 0,
				viewportX: 8,
				viewportY: 16,
				viewportWidth: 32,
				viewportHeight: 32,
				viewProjectionMatrix: Matrix4.identity(),
			},
		],
		sliceCount: 1,
		casterPackets,
		transmitterPackets,
		baselineFramebuffer: { id: "scene-framebuffer" },
		baselineViewportWidth: 320,
		baselineViewportHeight: 180,
	};
}

export function createShadowPacket(material = new Material()) {
	return {
		id: "shadow-packet",
		meshInstance: { id: "shadow-mesh", skeleton: null },
		material,
		worldMatrix: Matrix4.identity(),
	};
}

export async function runWebGLBackendFile(testCases, label) {
	Logger.reset();
	ShaderSource.clearCache("webgl");
	try {
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
		for (const testCase of testCases) {
			await testCase();
		}
	} finally {
		ShaderSource.clearCache("webgl");
		Logger.reset();
	}
	console.log(`${label} passed`);
}
