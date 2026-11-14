import { finiteOr } from "../../maths/Misc";
import { Logger } from "../../foundation/Logger";
import { SH_COEFFICIENT_COUNT } from "./constants";
import type { WebGLLightState } from "./WebGLLightCollector";

function logWebGLProbeSHTexturesWarning(key: string, message: string): void {
	Logger.warn(`[${key}] ${message}`, {
		scope: "WebGLProbeSHTextures",
		onceKey: key,
	});
}

/**
 * Owns the localized probe SH coefficient texture sampled by scene shaders.
 *
 * Each texture is created lazily on first upload and reused across frames.
 * Upload failures degrade gracefully by returning `false` so callers can
 * disable the corresponding lighting feature for the current frame.
 * @internal Owned by WebGL lighting; applications author `LightProbe.sh`.
 */
export class WebGLProbeSHTextures {
	public localLightProbeSHTexture: WebGLTexture | null = null;
	public localLightProbeSHTextureWidth = SH_COEFFICIENT_COUNT;
	public localLightProbeSHTextureHeight = 1;

	private readonly _gl: WebGL2RenderingContext;

	constructor(gl: WebGL2RenderingContext) {
		this._gl = gl;
	}

	public uploadLocalLightProbeCoefficients(
		probes: WebGLLightState["localLightProbes"] | null | undefined
	): boolean {
		const gl = this._gl;
		const resolvedProbes = Array.isArray(probes) ? probes : [];
		const width = SH_COEFFICIENT_COUNT;
		const height = Math.max(1, resolvedProbes.length);
		const data = new Float32Array(width * height * 4);

		for (let probeIndex = 0; probeIndex < resolvedProbes.length; probeIndex++) {
			const probe = resolvedProbes[probeIndex];
			for (let coeffIndex = 0; coeffIndex < width; coeffIndex++) {
				const coeff = probe.sh[coeffIndex];
				const base = (probeIndex * width + coeffIndex) * 4;
				data[base] = finiteOr(coeff?.r, 0);
				data[base + 1] = finiteOr(coeff?.g, 0);
				data[base + 2] = finiteOr(coeff?.b, 0);
				data[base + 3] = 0;
			}
		}

		if (!this.localLightProbeSHTexture) {
			if (typeof gl.createTexture !== "function") {
				logWebGLProbeSHTexturesWarning(
					"webgl-local-light-probe-texture-create-unsupported",
					"WebGL context does not expose createTexture(); disabling local light probe SH for this frame."
				);
				return false;
			}
			this.localLightProbeSHTexture = gl.createTexture();
			if (!this.localLightProbeSHTexture) {
				logWebGLProbeSHTexturesWarning(
					"webgl-local-light-probe-texture-create-failed",
					"Failed to create WebGL local light probe texture; disabling local light probe SH for this frame."
				);
				return false;
			}
			this._configureNearestClampTexture(this.localLightProbeSHTexture);
		}

		gl.bindTexture(gl.TEXTURE_2D, this.localLightProbeSHTexture);
		try {
			const internalFormat =
				(
					gl as WebGL2RenderingContext & {
						RGBA32F?: number;
					}
				).RGBA32F ?? gl.RGBA;
			gl.texImage2D(
				gl.TEXTURE_2D,
				0,
				internalFormat,
				width,
				height,
				0,
				gl.RGBA,
				gl.FLOAT,
				data
			);
			this.localLightProbeSHTextureWidth = width;
			this.localLightProbeSHTextureHeight = height;
			return true;
		} catch (error) {
			logWebGLProbeSHTexturesWarning(
				"webgl-local-light-probe-texture-upload-failed",
				`WebGL local light probe texture upload failed; disabling local light probe SH for this frame (${String(error)})`
			);
			return false;
		}
	}

	public destroy(): void {
		const gl = this._gl;
		if (this.localLightProbeSHTexture) {
			gl.deleteTexture(this.localLightProbeSHTexture);
			this.localLightProbeSHTexture = null;
		}
	}

	private _configureNearestClampTexture(texture: WebGLTexture): void {
		const gl = this._gl;
		gl.bindTexture(gl.TEXTURE_2D, texture);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
	}
}
