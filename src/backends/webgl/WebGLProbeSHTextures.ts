import { finiteOr } from "../../maths/Misc";
import type { SHCoefficients } from "../../maths/types";
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
 * Owns the probe SH coefficient textures consumed by global scene uniforms.
 *
 * Each texture is created lazily on first upload and reused across frames.
 * Upload failures degrade gracefully by returning `false` so callers can
 * disable the corresponding lighting feature for the current frame.
 */
export class WebGLProbeSHTextures {
	public shAmbientTexture: WebGLTexture | null = null;
	public shAmbientTextureWidth = SH_COEFFICIENT_COUNT;
	public shAmbientTextureHeight = 1;
	public localLightProbeSHTexture: WebGLTexture | null = null;
	public localLightProbeSHTextureWidth = SH_COEFFICIENT_COUNT;
	public localLightProbeSHTextureHeight = 1;
	public irradianceProbeGridSHTexture: WebGLTexture | null = null;
	public irradianceProbeGridSHTextureWidth = SH_COEFFICIENT_COUNT;
	public irradianceProbeGridSHTextureHeight = 1;

	private readonly _gl: WebGL2RenderingContext;

	constructor(gl: WebGL2RenderingContext) {
		this._gl = gl;
	}

	public uploadSHAmbientCoefficients(
		coeffs: SHCoefficients | null | undefined
	): boolean {
		const gl = this._gl;
		const texelCount = SH_COEFFICIENT_COUNT;
		const data = new Float32Array(texelCount * 4);
		for (let i = 0; i < texelCount; i++) {
			const coeff = coeffs?.[i];
			const base = i * 4;
			data[base] = finiteOr(coeff?.r, 0);
			data[base + 1] = finiteOr(coeff?.g, 0);
			data[base + 2] = finiteOr(coeff?.b, 0);
			data[base + 3] = 0;
		}

		if (!this.shAmbientTexture) {
			if (typeof gl.createTexture !== "function") {
				logWebGLProbeSHTexturesWarning(
					"webgl-sh-ambient-texture-create-unsupported",
					"WebGL context does not expose createTexture(); disabling SH for this frame."
				);
				return false;
			}
			this.shAmbientTexture = gl.createTexture();
			if (!this.shAmbientTexture) {
				logWebGLProbeSHTexturesWarning(
					"webgl-sh-ambient-texture-create-failed",
					"Failed to create WebGL SH ambient texture; disabling SH for this frame."
				);
				return false;
			}
			this._configureNearestClampTexture(this.shAmbientTexture);
		}

		gl.bindTexture(gl.TEXTURE_2D, this.shAmbientTexture);
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
				SH_COEFFICIENT_COUNT,
				1,
				0,
				gl.RGBA,
				gl.FLOAT,
				data
			);
			this.shAmbientTextureWidth = SH_COEFFICIENT_COUNT;
			this.shAmbientTextureHeight = 1;
			return true;
		} catch (error) {
			logWebGLProbeSHTexturesWarning(
				"webgl-sh-ambient-texture-upload-failed",
				`WebGL SH ambient texture upload failed; disabling SH for this frame (${String(error)})`
			);
			return false;
		}
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

	public uploadIrradianceProbeGridCoefficients(
		grid: WebGLLightState["irradianceProbeGrid"] | null | undefined
	): boolean {
		if (!grid || grid.cellCount <= 0) {
			return false;
		}
		const gl = this._gl;
		const width = SH_COEFFICIENT_COUNT;
		const height = Math.max(1, Math.floor(grid.cellCount));
		const data = new Float32Array(width * height * 4);

		for (let cellIndex = 0; cellIndex < height; cellIndex++) {
			const cellSH = grid.sh[cellIndex];
			const valid = grid.validMask[cellIndex] ? 1 : 0;
			for (let coeffIndex = 0; coeffIndex < width; coeffIndex++) {
				const coeff = cellSH?.[coeffIndex];
				const base = (cellIndex * width + coeffIndex) * 4;
				data[base] = finiteOr(coeff?.r, 0);
				data[base + 1] = finiteOr(coeff?.g, 0);
				data[base + 2] = finiteOr(coeff?.b, 0);
				data[base + 3] = valid;
			}
		}

		if (!this.irradianceProbeGridSHTexture) {
			if (typeof gl.createTexture !== "function") {
				logWebGLProbeSHTexturesWarning(
					"webgl-irradiance-probe-grid-texture-create-unsupported",
					"WebGL context does not expose createTexture(); disabling irradiance probe grid for this frame."
				);
				return false;
			}
			this.irradianceProbeGridSHTexture = gl.createTexture();
			if (!this.irradianceProbeGridSHTexture) {
				logWebGLProbeSHTexturesWarning(
					"webgl-irradiance-probe-grid-texture-create-failed",
					"Failed to create WebGL irradiance probe grid texture; disabling the grid for this frame."
				);
				return false;
			}
			this._configureNearestClampTexture(this.irradianceProbeGridSHTexture);
		}

		gl.bindTexture(gl.TEXTURE_2D, this.irradianceProbeGridSHTexture);
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
			this.irradianceProbeGridSHTextureWidth = width;
			this.irradianceProbeGridSHTextureHeight = height;
			return true;
		} catch (error) {
			logWebGLProbeSHTexturesWarning(
				"webgl-irradiance-probe-grid-texture-upload-failed",
				`WebGL irradiance probe grid texture upload failed; disabling the grid for this frame (${String(error)})`
			);
			return false;
		}
	}

	public destroy(): void {
		const gl = this._gl;
		if (this.shAmbientTexture) {
			gl.deleteTexture(this.shAmbientTexture);
			this.shAmbientTexture = null;
		}
		if (this.localLightProbeSHTexture) {
			gl.deleteTexture(this.localLightProbeSHTexture);
			this.localLightProbeSHTexture = null;
		}
		if (this.irradianceProbeGridSHTexture) {
			gl.deleteTexture(this.irradianceProbeGridSHTexture);
			this.irradianceProbeGridSHTexture = null;
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
