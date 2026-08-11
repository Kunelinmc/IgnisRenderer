import { WebGLCapabilityError } from "../../foundation/Error";

/** @internal Validates the non-negotiable WebGL internal HDR contract. */
export function assertWebGLHDRCapabilities(gl: WebGL2RenderingContext): void {
	if (!gl.getExtension("EXT_color_buffer_float")) {
		throw new WebGLCapabilityError("hdr-float-color-buffer-unavailable");
	}

	if (!probeRGBA16FFramebuffer(gl)) {
		throw new WebGLCapabilityError("hdr-float-color-buffer-unavailable");
	}

	if (
		!gl.getExtension("OES_texture_half_float_linear") &&
		!gl.getExtension("OES_texture_float_linear")
	) {
		throw new WebGLCapabilityError("hdr-float-linear-filtering-unavailable");
	}
}

function probeRGBA16FFramebuffer(gl: WebGL2RenderingContext): boolean {
	const texture = gl.createTexture();
	const framebuffer = gl.createFramebuffer();
	if (!texture || !framebuffer) {
		if (texture) gl.deleteTexture(texture);
		if (framebuffer) gl.deleteFramebuffer(framebuffer);
		return false;
	}

	let complete = false;
	try {
		gl.bindTexture(gl.TEXTURE_2D, texture);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
		gl.texImage2D(
			gl.TEXTURE_2D,
			0,
			gl.RGBA16F,
			1,
			1,
			0,
			gl.RGBA,
			gl.HALF_FLOAT,
			null,
		);
		gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
		gl.framebufferTexture2D(
			gl.FRAMEBUFFER,
			gl.COLOR_ATTACHMENT0,
			gl.TEXTURE_2D,
			texture,
			0,
		);
		complete = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
	} catch {
		complete = false;
	} finally {
		gl.bindFramebuffer(gl.FRAMEBUFFER, null);
		gl.bindTexture(gl.TEXTURE_2D, null);
		gl.deleteFramebuffer(framebuffer);
		gl.deleteTexture(texture);
	}
	return complete;
}
