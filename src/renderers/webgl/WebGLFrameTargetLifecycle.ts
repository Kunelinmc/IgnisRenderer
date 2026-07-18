import { DEFAULT_SSAO_OPTIONS } from "../../postprocess/passes/ScreenSpaceAmbientOcclusionPass";
import { Logger } from "../../foundation/Logger";

export type WebGLFrameTargetFormat = "rgba16float" | "rgba8unorm";

export interface WebGLFrameTargetLifecycleHost {
	_gl: WebGL2RenderingContext;
	_maxTextureSize: number;
	_maxRenderbufferSize: number;
	_sceneFramebuffer: WebGLFramebuffer | null;
	_sceneColorTexture: WebGLTexture | null;
	_sceneColorFormat: WebGLFrameTargetFormat;
	_sceneMotionTexture: WebGLTexture | null;
	_sceneMotionFormat: WebGLFrameTargetFormat;
	_sceneNormalTexture: WebGLTexture | null;
	_sceneNormalFormat: WebGLFrameTargetFormat;
	_sceneAlbedoTexture: WebGLTexture | null;
	_sceneAlbedoFormat: WebGLFrameTargetFormat;
	_sceneSpecularTexture: WebGLTexture | null;
	_sceneSpecularFormat: WebGLFrameTargetFormat;
	_materialGBufferEnabled: boolean;
	_sceneDepthBuffer: WebGLRenderbuffer | null;
	_oitFramebuffer: WebGLFramebuffer | null;
	_oitAccumTexture: WebGLTexture | null;
	_oitRevealTexture: WebGLTexture | null;
	_postFramebuffer: WebGLFramebuffer | null;
	_postColorTexture: WebGLTexture | null;
	_postColorFormat: WebGLFrameTargetFormat;
	_ssaoRawTexture: WebGLTexture | null;
	_ssaoBlurTexture: WebGLTexture | null;
	_ssaoColorFormat: WebGLFrameTargetFormat;
	_presentSourceTexture: WebGLTexture | null;
	_targetWidth: number;
	_targetHeight: number;
	_targetSSAODownsample: number;
	_targetMaterialGBufferEnabled: boolean;
	_ssaoFrameIndex: number;
	_supportsFloatColorBuffer: boolean | null;
}

function createColorTexture(
	gl: WebGL2RenderingContext,
	width: number,
	height: number,
	internalFormat: number = gl.RGBA8,
	type: number = gl.UNSIGNED_BYTE
): WebGLTexture | null {
	const texture = gl.createTexture();
	if (!texture) {
		return null;
	}
	gl.bindTexture(gl.TEXTURE_2D, texture);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
	gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

	let format = gl.RGBA;
	if (internalFormat === gl.RGBA16F || internalFormat === gl.RGBA32F) {
		format = gl.RGBA;
	}

	gl.texImage2D(
		gl.TEXTURE_2D,
		0,
		internalFormat,
		width,
		height,
		0,
		format,
		type,
		null
	);
	gl.bindTexture(gl.TEXTURE_2D, null);
	return texture;
}

function resolveLimit(
	gl: WebGL2RenderingContext,
	parameter: number | undefined,
	fallback: number
): number {
	if (typeof parameter !== "number" || typeof gl.getParameter !== "function") {
		return fallback;
	}
	try {
		const value = gl.getParameter(parameter);
		return typeof value === "number" && Number.isFinite(value) ?
			Math.max(0, Math.floor(value))
		:	fallback;
	} catch {
		return fallback;
	}
}

export function bindWebGLPostSingleColorTarget(
	host: WebGLFrameTargetLifecycleHost,
	texture: WebGLTexture
): void {
	const gl = host._gl;
	gl.framebufferTexture2D(
		gl.FRAMEBUFFER,
		gl.COLOR_ATTACHMENT0,
		gl.TEXTURE_2D,
		texture,
		0
	);
	gl.framebufferTexture2D(
		gl.FRAMEBUFFER,
		gl.COLOR_ATTACHMENT1,
		gl.TEXTURE_2D,
		null,
		0
	);
	gl.framebufferTexture2D(
		gl.FRAMEBUFFER,
		gl.COLOR_ATTACHMENT2,
		gl.TEXTURE_2D,
		null,
		0
	);
	gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
}

export function bindWebGLOITSingleColorTarget(
	host: WebGLFrameTargetLifecycleHost,
	texture: WebGLTexture
): void {
	const gl = host._gl;
	gl.framebufferTexture2D(
		gl.FRAMEBUFFER,
		gl.COLOR_ATTACHMENT0,
		gl.TEXTURE_2D,
		texture,
		0
	);
	gl.framebufferTexture2D(
		gl.FRAMEBUFFER,
		gl.COLOR_ATTACHMENT1,
		gl.TEXTURE_2D,
		null,
		0
	);
	gl.framebufferTexture2D(
		gl.FRAMEBUFFER,
		gl.COLOR_ATTACHMENT2,
		gl.TEXTURE_2D,
		null,
		0
	);
	gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
}

export function resolveWebGLPostProcessTargetTexture(
	host: WebGLFrameTargetLifecycleHost,
	sourceTexture: WebGLTexture
): WebGLTexture | null {
	if (!host._sceneColorTexture || !host._postColorTexture) {
		return null;
	}
	if (sourceTexture === host._sceneColorTexture) {
		return host._postColorTexture;
	}
	if (sourceTexture === host._postColorTexture) {
		return host._sceneColorTexture;
	}
	return host._postColorTexture;
}

export function ensureWebGLFrameTargets(
	host: WebGLFrameTargetLifecycleHost,
	width: number,
	height: number,
	ssaoDownsample: number,
	materialGBufferRequested: boolean
): void {
	const supportsFloatColorBuffer = !!host._gl.getExtension(
		"EXT_color_buffer_float"
	);
	if (
		host._sceneFramebuffer &&
		host._sceneColorTexture &&
		host._sceneMotionTexture &&
		host._sceneNormalTexture &&
		(!materialGBufferRequested ||
			(host._sceneAlbedoTexture && host._sceneSpecularTexture) ||
			!host._materialGBufferEnabled) &&
		host._sceneDepthBuffer &&
		(!supportsFloatColorBuffer ||
			(host._oitFramebuffer &&
				host._oitAccumTexture &&
				host._oitRevealTexture)) &&
		host._postFramebuffer &&
		host._postColorTexture &&
		host._ssaoRawTexture &&
		host._ssaoBlurTexture &&
		host._targetWidth === width &&
		host._targetHeight === height &&
		host._targetSSAODownsample === ssaoDownsample &&
		host._targetMaterialGBufferEnabled === materialGBufferRequested
	) {
		return;
	}

	if (
		width > host._maxTextureSize ||
		height > host._maxTextureSize ||
		width > host._maxRenderbufferSize ||
		height > host._maxRenderbufferSize
	) {
		throw new Error(
			`WebGL frame size ${width}x${height} exceeds device limits ` +
				`(MAX_TEXTURE_SIZE=${host._maxTextureSize}, ` +
				`MAX_RENDERBUFFER_SIZE=${host._maxRenderbufferSize})`
		);
	}

	destroyWebGLFrameTargets(host);
	const gl = host._gl;
	const colorInternalFormat = supportsFloatColorBuffer ? gl.RGBA16F : gl.RGBA8;
	const colorType = supportsFloatColorBuffer ? gl.HALF_FLOAT : gl.UNSIGNED_BYTE;
	const motionInternalFormat = supportsFloatColorBuffer ? gl.RGBA16F : gl.RGBA8;
	const motionType = supportsFloatColorBuffer ? gl.HALF_FLOAT : gl.UNSIGNED_BYTE;
	const colorFormat: WebGLFrameTargetFormat =
		supportsFloatColorBuffer ? "rgba16float" : "rgba8unorm";
	const motionFormat: WebGLFrameTargetFormat =
		supportsFloatColorBuffer ? "rgba16float" : "rgba8unorm";
	const normalInternalFormat =
		materialGBufferRequested && supportsFloatColorBuffer ? gl.RGBA16F : gl.RGBA8;
	const normalType =
		materialGBufferRequested && supportsFloatColorBuffer ? gl.HALF_FLOAT : gl.UNSIGNED_BYTE;
	const normalFormat: WebGLFrameTargetFormat =
		materialGBufferRequested && supportsFloatColorBuffer ? "rgba16float" : "rgba8unorm";
	const materialGBufferSupported =
		materialGBufferRequested &&
		resolveLimit(gl, gl.MAX_DRAW_BUFFERS, 4) >= 5 &&
		resolveLimit(gl, gl.MAX_COLOR_ATTACHMENTS, 4) >= 5;
	if (materialGBufferRequested && !materialGBufferSupported) {
		const key = "webgl-gbuffer-material-semantics-unsupported";
		Logger.warn(
			`[${key}] WebGL material G-buffer requires at least five draw buffers and color attachments.`,
			{ scope: "WebGLFrameTargetLifecycle", onceKey: key }
		);
	}
	const aoWidth = Math.max(1, Math.floor(width / Math.max(ssaoDownsample, 1)));
	const aoHeight = Math.max(1, Math.floor(height / Math.max(ssaoDownsample, 1)));
	host._supportsFloatColorBuffer = supportsFloatColorBuffer;
	if (!supportsFloatColorBuffer) {
		const key = "webgl-hdr-float-unsupported";
		Logger.warn(
			`[${key}] EXT_color_buffer_float is unavailable; falling back to RGBA8 color, motion, and post-process attachments.`,
			{
				scope: "WebGLFrameTargetLifecycle",
				onceKey: key,
			}
		);
	}

	const sceneFramebuffer = gl.createFramebuffer();
	const sceneColorTexture = createColorTexture(
		gl,
		width,
		height,
		colorInternalFormat,
		colorType
	);
	const sceneMotionTexture = createColorTexture(
		gl,
		width,
		height,
		motionInternalFormat,
		motionType
	);
	const sceneNormalTexture = createColorTexture(
		gl,
		width,
		height,
		normalInternalFormat,
		normalType
	);
	const sceneAlbedoTexture = materialGBufferSupported ?
		createColorTexture(gl, width, height, gl.RGBA8, gl.UNSIGNED_BYTE)
	:	null;
	const sceneSpecularTexture = materialGBufferSupported ?
		createColorTexture(gl, width, height, colorInternalFormat, colorType)
	:	null;
	const sceneDepthBuffer = gl.createRenderbuffer();
	const oitFramebuffer = supportsFloatColorBuffer ? gl.createFramebuffer() : null;
	const oitAccumTexture =
		supportsFloatColorBuffer ?
			createColorTexture(
				gl,
				width,
				height,
				gl.RGBA16F,
				gl.HALF_FLOAT
			)
		:	null;
	const oitRevealTexture =
		supportsFloatColorBuffer ?
			createColorTexture(
				gl,
				width,
				height,
				gl.RGBA8,
				gl.UNSIGNED_BYTE
			)
		:	null;
	const postFramebuffer = gl.createFramebuffer();
	const postColorTexture = createColorTexture(
		gl,
		width,
		height,
		colorInternalFormat,
		colorType
	);
	const ssaoRawTexture = createColorTexture(
		gl,
		aoWidth,
		aoHeight,
		colorInternalFormat,
		colorType
	);
	const ssaoBlurTexture = createColorTexture(
		gl,
		aoWidth,
		aoHeight,
		colorInternalFormat,
		colorType
	);

	const cleanupAllocatedTargets = (): void => {
		if (sceneFramebuffer) {
			gl.deleteFramebuffer(sceneFramebuffer);
		}
		if (sceneColorTexture) {
			gl.deleteTexture(sceneColorTexture);
		}
		if (sceneMotionTexture) {
			gl.deleteTexture(sceneMotionTexture);
		}
		if (sceneNormalTexture) {
			gl.deleteTexture(sceneNormalTexture);
		}
		if (sceneAlbedoTexture) {
			gl.deleteTexture(sceneAlbedoTexture);
		}
		if (sceneSpecularTexture) {
			gl.deleteTexture(sceneSpecularTexture);
		}
		if (sceneDepthBuffer) {
			gl.deleteRenderbuffer(sceneDepthBuffer);
		}
		if (oitFramebuffer) {
			gl.deleteFramebuffer(oitFramebuffer);
		}
		if (oitAccumTexture) {
			gl.deleteTexture(oitAccumTexture);
		}
		if (oitRevealTexture) {
			gl.deleteTexture(oitRevealTexture);
		}
		if (postFramebuffer) {
			gl.deleteFramebuffer(postFramebuffer);
		}
		if (postColorTexture) {
			gl.deleteTexture(postColorTexture);
		}
		if (ssaoRawTexture) {
			gl.deleteTexture(ssaoRawTexture);
		}
		if (ssaoBlurTexture) {
			gl.deleteTexture(ssaoBlurTexture);
		}
	};

	if (
		!sceneFramebuffer ||
		!sceneColorTexture ||
		!sceneMotionTexture ||
		!sceneNormalTexture ||
		(materialGBufferSupported && (!sceneAlbedoTexture || !sceneSpecularTexture)) ||
		!sceneDepthBuffer ||
		(supportsFloatColorBuffer &&
			(!oitFramebuffer || !oitAccumTexture || !oitRevealTexture)) ||
		!postFramebuffer ||
		!postColorTexture ||
		!ssaoRawTexture ||
		!ssaoBlurTexture
	) {
		cleanupAllocatedTargets();
		throw new Error("Failed to create WebGL frame targets");
	}

	gl.bindRenderbuffer(gl.RENDERBUFFER, sceneDepthBuffer);
	gl.renderbufferStorage(
		gl.RENDERBUFFER,
		gl.DEPTH_COMPONENT24,
		width,
		height
	);
	gl.bindRenderbuffer(gl.RENDERBUFFER, null);

	gl.bindFramebuffer(gl.FRAMEBUFFER, sceneFramebuffer);
	gl.framebufferTexture2D(
		gl.FRAMEBUFFER,
		gl.COLOR_ATTACHMENT0,
		gl.TEXTURE_2D,
		sceneColorTexture,
		0
	);
	gl.framebufferTexture2D(
		gl.FRAMEBUFFER,
		gl.COLOR_ATTACHMENT1,
		gl.TEXTURE_2D,
		sceneMotionTexture,
		0
	);
	gl.framebufferTexture2D(
		gl.FRAMEBUFFER,
		gl.COLOR_ATTACHMENT2,
		gl.TEXTURE_2D,
		sceneNormalTexture,
		0
	);
	if (sceneAlbedoTexture && sceneSpecularTexture) {
		gl.framebufferTexture2D(
			gl.FRAMEBUFFER,
			gl.COLOR_ATTACHMENT3,
			gl.TEXTURE_2D,
			sceneAlbedoTexture,
			0
		);
		gl.framebufferTexture2D(
			gl.FRAMEBUFFER,
			gl.COLOR_ATTACHMENT4,
			gl.TEXTURE_2D,
			sceneSpecularTexture,
			0
		);
	}
	gl.framebufferRenderbuffer(
		gl.FRAMEBUFFER,
		gl.DEPTH_ATTACHMENT,
		gl.RENDERBUFFER,
		sceneDepthBuffer
	);
	gl.drawBuffers(sceneAlbedoTexture && sceneSpecularTexture ? [
		gl.COLOR_ATTACHMENT0,
		gl.COLOR_ATTACHMENT1,
		gl.COLOR_ATTACHMENT2,
		gl.COLOR_ATTACHMENT3,
		gl.COLOR_ATTACHMENT4,
	] : [
		gl.COLOR_ATTACHMENT0,
		gl.COLOR_ATTACHMENT1,
		gl.COLOR_ATTACHMENT2,
	]);
	let status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
	if (status !== gl.FRAMEBUFFER_COMPLETE) {
		gl.bindFramebuffer(gl.FRAMEBUFFER, null);
		cleanupAllocatedTargets();
		throw new Error(
			`WebGL scene framebuffer is incomplete (status=0x${status.toString(16)})`
		);
	}

	if (oitFramebuffer && oitAccumTexture && oitRevealTexture) {
		gl.bindFramebuffer(gl.FRAMEBUFFER, oitFramebuffer);
		gl.framebufferTexture2D(
			gl.FRAMEBUFFER,
			gl.COLOR_ATTACHMENT0,
			gl.TEXTURE_2D,
			oitAccumTexture,
			0
		);
		gl.framebufferTexture2D(
			gl.FRAMEBUFFER,
			gl.COLOR_ATTACHMENT1,
			gl.TEXTURE_2D,
			oitRevealTexture,
			0
		);
		gl.framebufferRenderbuffer(
			gl.FRAMEBUFFER,
			gl.DEPTH_ATTACHMENT,
			gl.RENDERBUFFER,
			sceneDepthBuffer
		);
		gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
		status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
		if (status !== gl.FRAMEBUFFER_COMPLETE) {
			gl.bindFramebuffer(gl.FRAMEBUFFER, null);
			cleanupAllocatedTargets();
			throw new Error(
				`WebGL OIT framebuffer is incomplete (status=0x${status.toString(16)})`
			);
		}
	}

	gl.bindFramebuffer(gl.FRAMEBUFFER, postFramebuffer);
	gl.framebufferTexture2D(
		gl.FRAMEBUFFER,
		gl.COLOR_ATTACHMENT0,
		gl.TEXTURE_2D,
		postColorTexture,
		0
	);
	status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
	gl.bindFramebuffer(gl.FRAMEBUFFER, null);
	if (status !== gl.FRAMEBUFFER_COMPLETE) {
		cleanupAllocatedTargets();
		throw new Error(
			`WebGL post framebuffer is incomplete (status=0x${status.toString(16)})`
		);
	}

	host._sceneFramebuffer = sceneFramebuffer;
	host._sceneColorTexture = sceneColorTexture;
	host._sceneColorFormat = colorFormat;
	host._sceneMotionTexture = sceneMotionTexture;
	host._sceneMotionFormat = motionFormat;
	host._sceneNormalTexture = sceneNormalTexture;
	host._sceneNormalFormat = normalFormat;
	host._sceneAlbedoTexture = sceneAlbedoTexture;
	host._sceneAlbedoFormat = "rgba8unorm";
	host._sceneSpecularTexture = sceneSpecularTexture;
	host._sceneSpecularFormat = colorFormat;
	host._materialGBufferEnabled = materialGBufferSupported;
	host._sceneDepthBuffer = sceneDepthBuffer;
	host._oitFramebuffer = oitFramebuffer;
	host._oitAccumTexture = oitAccumTexture;
	host._oitRevealTexture = oitRevealTexture;
	host._postFramebuffer = postFramebuffer;
	host._postColorTexture = postColorTexture;
	host._postColorFormat = colorFormat;
	host._ssaoRawTexture = ssaoRawTexture;
	host._ssaoBlurTexture = ssaoBlurTexture;
	host._ssaoColorFormat = colorFormat;
	host._presentSourceTexture = sceneColorTexture;
	host._targetWidth = width;
	host._targetHeight = height;
	host._targetSSAODownsample = ssaoDownsample;
	host._targetMaterialGBufferEnabled = materialGBufferRequested;
	host._ssaoFrameIndex = 0;
}

export function destroyWebGLFrameTargets(
	host: WebGLFrameTargetLifecycleHost
): void {
	const gl = host._gl;
	if (host._sceneFramebuffer) {
		gl.deleteFramebuffer(host._sceneFramebuffer);
		host._sceneFramebuffer = null;
	}
	if (host._sceneColorTexture) {
		gl.deleteTexture(host._sceneColorTexture);
		host._sceneColorTexture = null;
	}
	host._sceneColorFormat = "rgba8unorm";
	if (host._sceneMotionTexture) {
		gl.deleteTexture(host._sceneMotionTexture);
		host._sceneMotionTexture = null;
	}
	host._sceneMotionFormat = "rgba8unorm";
	if (host._sceneNormalTexture) {
		gl.deleteTexture(host._sceneNormalTexture);
		host._sceneNormalTexture = null;
	}
	host._sceneNormalFormat = "rgba8unorm";
	if (host._sceneAlbedoTexture) {
		gl.deleteTexture(host._sceneAlbedoTexture);
		host._sceneAlbedoTexture = null;
	}
	host._sceneAlbedoFormat = "rgba8unorm";
	if (host._sceneSpecularTexture) {
		gl.deleteTexture(host._sceneSpecularTexture);
		host._sceneSpecularTexture = null;
	}
	host._sceneSpecularFormat = "rgba8unorm";
	host._materialGBufferEnabled = false;
	if (host._sceneDepthBuffer) {
		gl.deleteRenderbuffer(host._sceneDepthBuffer);
		host._sceneDepthBuffer = null;
	}
	if (host._oitFramebuffer) {
		gl.deleteFramebuffer(host._oitFramebuffer);
		host._oitFramebuffer = null;
	}
	if (host._oitAccumTexture) {
		gl.deleteTexture(host._oitAccumTexture);
		host._oitAccumTexture = null;
	}
	if (host._oitRevealTexture) {
		gl.deleteTexture(host._oitRevealTexture);
		host._oitRevealTexture = null;
	}
	if (host._postFramebuffer) {
		gl.deleteFramebuffer(host._postFramebuffer);
		host._postFramebuffer = null;
	}
	if (host._postColorTexture) {
		gl.deleteTexture(host._postColorTexture);
		host._postColorTexture = null;
	}
	host._postColorFormat = "rgba8unorm";
	if (host._ssaoRawTexture) {
		gl.deleteTexture(host._ssaoRawTexture);
		host._ssaoRawTexture = null;
	}
	if (host._ssaoBlurTexture) {
		gl.deleteTexture(host._ssaoBlurTexture);
		host._ssaoBlurTexture = null;
	}
	host._ssaoColorFormat = "rgba8unorm";
	host._presentSourceTexture = null;
	host._targetWidth = 0;
	host._targetHeight = 0;
	host._targetSSAODownsample = DEFAULT_SSAO_OPTIONS.downsample;
	host._targetMaterialGBufferEnabled = false;
	host._ssaoFrameIndex = 0;
	host._supportsFloatColorBuffer = null;
}
