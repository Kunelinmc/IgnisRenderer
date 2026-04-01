import assert from "node:assert/strict";
import { WebGLFrameExecutor } from "../src/renderers/webgl/WebGLFrameExecutor.ts";

function createFXAATestGL() {
	const calls = [];
	return {
		calls,
		MAX_TEXTURE_SIZE: 0x0d33,
		MAX_RENDERBUFFER_SIZE: 0x84e8,
		FRAMEBUFFER: 0x8d40,
		COLOR_ATTACHMENT0: 0x8ce0,
		COLOR_ATTACHMENT1: 0x8ce1,
		TEXTURE_2D: 0x0de1,
		TEXTURE0: 0x84c0,
		TEXTURE1: 0x84c1,
		TEXTURE2: 0x84c2,
		TEXTURE3: 0x84c3,
		TRIANGLES: 0x0004,
		CULL_FACE: 0x0b44,
		DEPTH_TEST: 0x0b71,
		BLEND: 0x0be2,
		ONE: 1,
		SRC_ALPHA: 0x0302,
		ONE_MINUS_SRC_ALPHA: 0x0303,
		getParameter(parameter) {
			if (
				parameter === this.MAX_TEXTURE_SIZE ||
				parameter === this.MAX_RENDERBUFFER_SIZE
			) {
				return 4096;
			}
			return 0;
		},
		createVertexArray() {
			return {};
		},
		deleteVertexArray() {},
		deleteFramebuffer() {},
		deleteTexture() {},
		deleteRenderbuffer() {},
		bindFramebuffer(target, framebuffer) {
			calls.push({ name: "bindFramebuffer", target, framebuffer });
		},
		framebufferTexture2D(target, attachment, textarget, texture, level) {
			calls.push({
				name: "framebufferTexture2D",
				target,
				attachment,
				textarget,
				texture,
				level,
			});
		},
		drawBuffers(buffers) {
			calls.push({ name: "drawBuffers", buffers: [...buffers] });
		},
		viewport(x, y, width, height) {
			calls.push({ name: "viewport", x, y, width, height });
		},
		useProgram(program) {
			calls.push({ name: "useProgram", program });
		},
		bindVertexArray(vao) {
			calls.push({ name: "bindVertexArray", vao });
		},
		disable(cap) {
			calls.push({ name: "disable", cap });
		},
		enable(cap) {
			calls.push({ name: "enable", cap });
		},
		depthMask(flag) {
			calls.push({ name: "depthMask", flag });
		},
		blendFuncSeparate(srcRgb, dstRgb, srcAlpha, dstAlpha) {
			calls.push({
				name: "blendFuncSeparate",
				srcRgb,
				dstRgb,
				srcAlpha,
				dstAlpha,
			});
		},
		activeTexture(unit) {
			calls.push({ name: "activeTexture", unit });
		},
		bindTexture(target, texture) {
			calls.push({ name: "bindTexture", target, texture });
		},
		uniform1i(location, value) {
			calls.push({ name: "uniform1i", location, value });
		},
		uniform1f(location, value) {
			calls.push({ name: "uniform1f", location, value });
		},
		uniform2f(location, x, y) {
			calls.push({ name: "uniform2f", location, x, y });
		},
		uniform3f(location, x, y, z) {
			calls.push({ name: "uniform3f", location, x, y, z });
		},
		uniform4f(location, x, y, z, w) {
			calls.push({ name: "uniform4f", location, x, y, z, w });
		},
		uniform4fv(location, values) {
			calls.push({ name: "uniform4fv", location, values: Array.from(values) });
		},
		uniformMatrix4fv(location, transpose, values) {
			calls.push({
				name: "uniformMatrix4fv",
				location,
				transpose,
				values: Array.from(values),
			});
		},
		drawArrays(mode, first, count) {
			calls.push({ name: "drawArrays", mode, first, count });
		},
	};
}

function createFrameTargetTestGL(options = {}) {
	const frameStatuses =
		options.frameStatuses ?? [0x8cd5, 0x8cd5];
	let statusIndex = 0;
	let textureId = 0;
	let framebufferId = 0;
	let renderbufferId = 0;

	const createdTextures = [];
	const deletedTextures = [];
	const createdFramebuffers = [];
	const deletedFramebuffers = [];
	const createdRenderbuffers = [];
	const deletedRenderbuffers = [];
	const texImage2DCalls = [];

	return {
		MAX_TEXTURE_SIZE: 0x0d33,
		MAX_RENDERBUFFER_SIZE: 0x84e8,
		FRAMEBUFFER: 0x8d40,
		FRAMEBUFFER_COMPLETE: 0x8cd5,
		FRAMEBUFFER_INCOMPLETE_ATTACHMENT: 0x8cd6,
		RENDERBUFFER: 0x8d41,
		COLOR_ATTACHMENT0: 0x8ce0,
		COLOR_ATTACHMENT1: 0x8ce1,
		COLOR_ATTACHMENT2: 0x8ce2,
		DEPTH_ATTACHMENT: 0x8d00,
		DEPTH_COMPONENT24: 0x81a6,
		TEXTURE_2D: 0x0de1,
		TEXTURE_MIN_FILTER: 0x2801,
		TEXTURE_MAG_FILTER: 0x2800,
		TEXTURE_WRAP_S: 0x2802,
		TEXTURE_WRAP_T: 0x2803,
		LINEAR: 0x2601,
		CLAMP_TO_EDGE: 0x812f,
		RGBA: 0x1908,
		RGBA8: 0x8058,
		RGBA16F: 0x881a,
		HALF_FLOAT: 0x140b,
		UNSIGNED_BYTE: 0x1401,
		createVertexArray() {
			return {};
		},
		deleteVertexArray() {},
		getParameter(parameter) {
			if (
				parameter === this.MAX_TEXTURE_SIZE ||
				parameter === this.MAX_RENDERBUFFER_SIZE
			) {
				return 4096;
			}
			return 0;
		},
		getExtension(name) {
			if (name === "EXT_color_buffer_float") {
				return options.floatExtension ? {} : null;
			}
			return null;
		},
		createTexture() {
			const texture = { id: `tex-${++textureId}` };
			createdTextures.push(texture);
			return texture;
		},
		deleteTexture(texture) {
			deletedTextures.push(texture);
		},
		createFramebuffer() {
			const framebuffer = { id: `fbo-${++framebufferId}` };
			createdFramebuffers.push(framebuffer);
			return framebuffer;
		},
		deleteFramebuffer(framebuffer) {
			deletedFramebuffers.push(framebuffer);
		},
		createRenderbuffer() {
			const renderbuffer = { id: `rbo-${++renderbufferId}` };
			createdRenderbuffers.push(renderbuffer);
			return renderbuffer;
		},
		deleteRenderbuffer(renderbuffer) {
			deletedRenderbuffers.push(renderbuffer);
		},
		bindTexture() {},
		texParameteri() {},
		texImage2D(_target, _level, internalFormat, width, height, _border, _format, type) {
			texImage2DCalls.push({ internalFormat, type, width, height });
		},
		bindRenderbuffer() {},
		renderbufferStorage() {},
		bindFramebuffer() {},
		framebufferTexture2D() {},
		framebufferRenderbuffer() {},
		drawBuffers() {},
		checkFramebufferStatus() {
			const index = Math.min(statusIndex, frameStatuses.length - 1);
			statusIndex++;
			return frameStatuses[index];
		},
		createdTextures,
		deletedTextures,
		createdFramebuffers,
		deletedFramebuffers,
		createdRenderbuffers,
		deletedRenderbuffers,
		texImage2DCalls,
	};
}

function testFXAAPassUsesLatestPostSourceAndRebindsPostTarget() {
	const gl = createFXAATestGL();
	const executor = new WebGLFrameExecutor(gl, () => {});
	const sceneColor = { id: "scene-color" };
	const taaHistory = { id: "taa-history" };
	const postColor = { id: "post-color" };
	const postFramebuffer = { id: "post-fbo" };
	const fullscreenVao = { id: "fullscreen-vao" };

	executor._programs = {
		getFXAAProgram() {
			return {
				program: { id: "fxaa-program" },
				uniforms: {
					sourceMap: { id: "uSourceMap" },
					texelSize: { id: "uTexelSize" },
				},
			};
		},
	};
	executor._sceneColorTexture = sceneColor;
	executor._presentSourceTexture = taaHistory;
	executor._postColorTexture = postColor;
	executor._postFramebuffer = postFramebuffer;
	executor._fullscreenVao = fullscreenVao;
	executor._width = 1280;
	executor._height = 720;

	executor._applyFXAA();

	const attachmentWrite = gl.calls.find(
		(call) =>
			call.name === "framebufferTexture2D" &&
			call.attachment === gl.COLOR_ATTACHMENT0
	);
	assert.equal(attachmentWrite?.texture, postColor);

	const sourceBindCalls = gl.calls.filter(
		(call) => call.name === "bindTexture" && call.target === gl.TEXTURE_2D
	);
	const sourceBind = sourceBindCalls[sourceBindCalls.length - 1];
	assert.equal(sourceBind?.texture, taaHistory);

	assert.equal(executor._presentSourceTexture, postColor);
}

function testFrameTargetsFallbackToRGBA8MotionWithoutFloatExtension() {
	const warnings = [];
	const gl = createFrameTargetTestGL({ floatExtension: false });
	const executor = new WebGLFrameExecutor(gl, (key, message) =>
		warnings.push({ key, message })
	);

	executor._ensureFrameTargets(320, 180);

	assert.equal(
		gl.texImage2DCalls.some((call) => call.internalFormat === gl.RGBA16F),
		false
	);
	assert.ok(
		warnings.some(
			(warning) => warning.key === "webgl-motion-float-unsupported"
		)
	);
}

function testSceneFramebufferFailureCleansAllAllocatedTargets() {
	const gl = createFrameTargetTestGL({
		floatExtension: true,
		frameStatuses: [0x8cd6],
	});
	const executor = new WebGLFrameExecutor(gl, () => {});

	assert.throws(
		() => executor._ensureFrameTargets(256, 256),
		/WebGL scene framebuffer is incomplete/
	);
	assert.equal(gl.deletedTextures.length, gl.createdTextures.length);
	assert.equal(gl.deletedFramebuffers.length, gl.createdFramebuffers.length);
	assert.equal(gl.deletedRenderbuffers.length, gl.createdRenderbuffers.length);
}

function testEndFramePrunesStaleModelMatrixCache() {
	const gl = createFXAATestGL();
	const executor = new WebGLFrameExecutor(gl, () => {});

	executor._modelMatrixCache.set("keep", new Float32Array(16));
	executor._modelMatrixCache.set("drop", new Float32Array(16));
	executor._modelMatrixKeysThisFrame.add("keep");
	executor._presentedInFrame = true;
	executor.endFrame();

	assert.equal(executor._modelMatrixCache.has("keep"), true);
	assert.equal(executor._modelMatrixCache.has("drop"), false);
}

function testShadowSkinningWarningKeyIsStable() {
	const warnings = [];
	const gl = createFXAATestGL();
	const executor = new WebGLFrameExecutor(gl, (key, message) =>
		warnings.push({ key, message })
	);

	executor._drawShadowPacket(
		{ uniforms: { mvp: null } },
		{ meshInstance: { id: "mesh-a", skeleton: {} } },
		{}
	);

	assert.equal(warnings[0]?.key, "webgl-shadow-skinning-unsupported");
}

function testTransparentRenderPacketsConfiguresBlendAndDepthState() {
	const gl = createFXAATestGL();
	const executor = new WebGLFrameExecutor(gl, () => {});
	executor._sceneFramebuffer = { id: "scene-fbo" };
	executor._sceneNormalTexture = { id: "normal" };
	executor._programs = {
		getSceneProgram() {
			return {
				program: { id: "scene-program" },
				uniforms: {},
			};
		},
	};
	executor._bindGlobalUniforms = () => {};
	executor._drawPacket = () => {};

	executor._renderPackets(
		{
			camera: {
				viewProjectionMatrix: [
					[1, 0, 0, 0],
					[0, 1, 0, 0],
					[0, 0, 1, 0],
					[0, 0, 0, 1],
				],
			},
		},
		[{ material: {} }],
		true
	);

	const drawBuffersCall = gl.calls.find((call) => call.name === "drawBuffers");
	assert.deepEqual(drawBuffersCall?.buffers, [
		gl.COLOR_ATTACHMENT0,
		gl.COLOR_ATTACHMENT1,
	]);
	const depthMaskCalls = gl.calls.filter((call) => call.name === "depthMask");
	assert.equal(depthMaskCalls[0]?.flag, false);
	assert.equal(depthMaskCalls[depthMaskCalls.length - 1]?.flag, true);
	const blendCall = gl.calls.find((call) => call.name === "blendFuncSeparate");
	assert.equal(blendCall?.srcRgb, gl.SRC_ALPHA);
	assert.equal(blendCall?.dstRgb, gl.ONE_MINUS_SRC_ALPHA);
	assert.equal(blendCall?.srcAlpha, gl.ONE);
	assert.equal(blendCall?.dstAlpha, gl.ONE_MINUS_SRC_ALPHA);
	assert.ok(
		gl.calls.some((call) => call.name === "enable" && call.cap === gl.BLEND)
	);
	assert.ok(
		gl.calls.some((call) => call.name === "disable" && call.cap === gl.BLEND)
	);
}

function testTAAPassDetachesMotionAttachmentAndSanitizesOptions() {
	const gl = createFXAATestGL();
	const executor = new WebGLFrameExecutor(gl, () => {});
	executor._programs = {
		getTAAProgram() {
			return {
				program: { id: "taa-program" },
				uniforms: {
					sceneColor: "uSceneColor",
					historyMap: "uHistoryMap",
					motionMap: "uMotionMap",
					motionHistory: "uMotionHistory",
					texelSize: "uTexelSize",
					historyWeight: "uHistoryWeight",
					depthThreshold: "uDepthThreshold",
					motionFactor: "uMotionFactor",
					varianceClampGamma: "uVarianceClampGamma",
					sharpen: "uSharpen",
					historyValid: "uHistoryValid",
				},
			};
		},
	};
	executor._postFramebuffer = { id: "post-fbo" };
	executor._sceneColorTexture = { id: "scene-color" };
	executor._sceneMotionTexture = { id: "scene-motion" };
	executor._taaHistoryTextures = [{ id: "history-a" }, { id: "history-b" }];
	executor._taaMotionHistoryTextures = [{ id: "motion-a" }, { id: "motion-b" }];
	executor._fullscreenVao = { id: "fullscreen-vao" };
	executor._width = 1920;
	executor._height = 1080;

	executor._applyTAA({
		historyWeight: Number.POSITIVE_INFINITY,
		disocclusionDepthThreshold: Number.NaN,
		motionFactor: 1e9,
		varianceClampGamma: -5,
		sharpen: 4,
	});

	const attachment1Writes = gl.calls.filter(
		(call) =>
			call.name === "framebufferTexture2D" &&
			call.attachment === gl.COLOR_ATTACHMENT1
	);
	assert.equal(attachment1Writes[attachment1Writes.length - 1]?.texture, null);

	const drawBuffersCalls = gl.calls.filter((call) => call.name === "drawBuffers");
	assert.deepEqual(drawBuffersCalls[drawBuffersCalls.length - 1]?.buffers, [
		gl.COLOR_ATTACHMENT0,
	]);

	const uniform1fCalls = gl.calls.filter((call) => call.name === "uniform1f");
	const uniformMap = new Map(uniform1fCalls.map((call) => [call.location, call.value]));
	assert.equal(uniformMap.get("uHistoryWeight"), 0.9);
	assert.equal(uniformMap.get("uDepthThreshold"), 0.02);
	assert.equal(uniformMap.get("uMotionFactor"), 512);
	assert.equal(uniformMap.get("uVarianceClampGamma"), 0);
	assert.equal(uniformMap.get("uSharpen"), 2);
	assert.equal(uniformMap.get("uHistoryValid"), 0);
}

function testSSAOPassDetachesSecondaryAttachmentForDownsampleTargets() {
	const gl = createFXAATestGL();
	const executor = new WebGLFrameExecutor(gl, () => {});
	executor._programs = {
		getSSAORawProgram() {
			return { program: { id: "ssao-raw" }, uniforms: {} };
		},
		getSSAOBlurProgram() {
			return { program: { id: "ssao-blur" }, uniforms: {} };
		},
		getSSAOCombineProgram() {
			return { program: { id: "ssao-combine" }, uniforms: {} };
		},
	};
	executor._postFramebuffer = { id: "post-fbo" };
	executor._postColorTexture = { id: "post-color" };
	executor._sceneColorTexture = { id: "scene-color" };
	executor._sceneMotionTexture = { id: "scene-motion" };
	executor._sceneNormalTexture = { id: "scene-normal" };
	executor._ssaoRawTexture = { id: "ssao-raw" };
	executor._ssaoBlurTexture = { id: "ssao-blur" };
	executor._fullscreenVao = { id: "fullscreen-vao" };
	executor._width = 1280;
	executor._height = 720;
	executor._targetSSAODownsample = 2;

	const identity = [
		[1, 0, 0, 0],
		[0, 1, 0, 0],
		[0, 0, 1, 0],
		[0, 0, 0, 1],
	];
	const context = {
		camera: {
			type: "perspective",
			fov: 60,
			aspectRatio: 1280 / 720,
			viewMatrix: { elements: identity },
			getWorldPosition() {
				return { x: 0, y: 0, z: 5 };
			},
		},
	};

	executor._applySSAO(undefined, context);

	const detachCalls = gl.calls.filter(
		(call) =>
			call.name === "framebufferTexture2D" &&
			call.attachment === gl.COLOR_ATTACHMENT1 &&
			call.texture === null
	);
	assert.ok(detachCalls.length > 0);
}

function testGlobalUniformsBindLightProbeIBLTextures() {
	const gl = createFXAATestGL();
	const executor = new WebGLFrameExecutor(gl, () => {});
	const envTexture = { id: "env-specular" };
	const brdfTexture = { id: "brdf-lut" };
	const envProbeMap = {
		mipmaps: [new Float32Array(4), new Float32Array(4), new Float32Array(4)],
	};

	executor._textures = {
		getEnvironmentSpecularTexture(texture) {
			assert.equal(texture, envProbeMap);
			return { texture: envTexture, isLinear: true };
		},
		getBRDFLUTTexture(texture) {
			assert.ok(texture);
			return { texture: brdfTexture, isLinear: true };
		},
	};
	executor._lightState = {
		ambientColor: [0, 0, 0],
		directionalLights: [],
		directionalShadows: [],
		pointLights: [],
		spotLights: [],
		spotShadows: [],
		envSpecularMap: envProbeMap,
	};
	const sceneProgram = {
		uniforms: {
			envSpecularMap: "uEnvSpecularMap",
			hasEnvSpecularMap: "uHasEnvSpecularMap",
			envSpecularMapIsLinear: "uEnvSpecularMapIsLinear",
			envSpecularMaxMipLevel: "uEnvSpecularMaxMipLevel",
			brdfLUT: "uBrdfLUT",
		},
	};
	const context = {
		features: {
			enableLighting: true,
			enableShadows: false,
		},
	};

	executor._bindGlobalUniforms(sceneProgram, context);

	const uniform1i = new Map(
		gl.calls
			.filter((call) => call.name === "uniform1i")
			.map((call) => [call.location, call.value])
	);
	assert.equal(uniform1i.get("uEnvSpecularMap"), 2);
	assert.equal(uniform1i.get("uHasEnvSpecularMap"), 1);
	assert.equal(uniform1i.get("uEnvSpecularMapIsLinear"), 1);
	assert.equal(uniform1i.get("uBrdfLUT"), 3);

	const uniform1f = new Map(
		gl.calls
			.filter((call) => call.name === "uniform1f")
			.map((call) => [call.location, call.value])
	);
	assert.equal(uniform1f.get("uEnvSpecularMaxMipLevel"), 2);

	const activeTextureUnits = gl.calls
		.filter((call) => call.name === "activeTexture")
		.map((call) => call.unit);
	assert.ok(activeTextureUnits.includes(gl.TEXTURE2));
	assert.ok(activeTextureUnits.includes(gl.TEXTURE3));
}

function testGlobalUniformsSanitizeNonFiniteCameraAndLightValues() {
	const warnings = [];
	const gl = createFXAATestGL();
	const executor = new WebGLFrameExecutor(gl, (key, message) =>
		warnings.push({ key, message })
	);

	executor._lightState = {
		ambientColor: [0, 0, 0],
		directionalLights: [
			{
				direction: [Number.NaN, 1, 0],
				color: [Number.POSITIVE_INFINITY, 1, 0.5],
			},
		],
		directionalShadows: [
			{
				viewProjectionMatrix: [
					[1, 0, 0, 0],
					[0, Number.NaN, 0, 0],
					[0, 0, 1, 0],
					[0, 0, 0, 1],
				],
			},
		],
		pointLights: [],
		spotLights: [],
		spotShadows: [],
	};

	const sceneProgram = {
		uniforms: {
			viewProjection: "uViewProjection",
			viewMatrix: "uViewMatrix",
			cameraPosition: "uCameraPosition",
			dirLightDirection: "uDirLightDirection",
			dirLightColor: "uDirLightColor",
			dirShadowViewProjection: "uDirShadowViewProjection",
		},
	};
	const context = {
		camera: {
			viewProjectionMatrix: {
				elements: [
					[Number.NaN, 0, 0, 0],
					[0, 1, 0, 0],
					[0, 0, 1, 0],
					[0, 0, 0, 1],
				],
			},
			viewMatrix: {
				elements: [
					[1, 0, 0, 0],
					[0, Number.POSITIVE_INFINITY, 0, 0],
					[0, 0, 1, 0],
					[0, 0, 0, 1],
				],
			},
			getWorldPosition() {
				return { x: Number.NaN, y: Number.POSITIVE_INFINITY, z: 3 };
			},
		},
		features: {
			enableLighting: true,
			enableShadows: true,
		},
	};

	executor._bindGlobalUniforms(sceneProgram, context);

	const matrixUploads = gl.calls.filter(
		(call) =>
			call.name === "uniformMatrix4fv" &&
			(call.location === "uViewProjection" ||
				call.location === "uViewMatrix" ||
				call.location === "uDirShadowViewProjection")
	);
	for (const upload of matrixUploads) {
		assert.equal(upload.values.every(Number.isFinite), true);
	}

	const vecUploads = gl.calls.filter(
		(call) =>
			(call.name === "uniform3f" && call.location === "uCameraPosition") ||
			(call.name === "uniform4fv" &&
				(call.location === "uDirLightDirection" ||
					call.location === "uDirLightColor"))
	);
	for (const upload of vecUploads) {
		const values =
			upload.name === "uniform3f" ? [upload.x, upload.y, upload.z] : upload.values;
		assert.equal(values.every(Number.isFinite), true);
	}

	assert.ok(
		warnings.some(
			(warning) => warning.key === "webgl-camera-view-projection-invalid"
		)
	);
}

function run() {
	testFXAAPassUsesLatestPostSourceAndRebindsPostTarget();
	testFrameTargetsFallbackToRGBA8MotionWithoutFloatExtension();
	testSceneFramebufferFailureCleansAllAllocatedTargets();
	testEndFramePrunesStaleModelMatrixCache();
	testShadowSkinningWarningKeyIsStable();
	testTransparentRenderPacketsConfiguresBlendAndDepthState();
	testTAAPassDetachesMotionAttachmentAndSanitizesOptions();
	testSSAOPassDetachesSecondaryAttachmentForDownsampleTargets();
	testGlobalUniformsBindLightProbeIBLTextures();
	testGlobalUniformsSanitizeNonFiniteCameraAndLightValues();
	console.log("WebGL FXAA frame executor tests passed");
}

run();
