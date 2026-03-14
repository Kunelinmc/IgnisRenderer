import {
	WEBGL_MAX_DIRECTIONAL_LIGHTS,
	WEBGL_MAX_POINT_LIGHTS,
	WEBGL_MAX_SPOT_LIGHTS,
} from "./constants";
import {
	WEBGL_COPY_FRAGMENT_SHADER,
	WEBGL_COPY_VERTEX_SHADER,
	WEBGL_FXAA_FRAGMENT_SHADER,
	WEBGL_FXAA_VERTEX_SHADER,
	WEBGL_PARTICLE_FRAGMENT_SHADER,
	WEBGL_PARTICLE_VERTEX_SHADER,
	WEBGL_POST_PROCESS_STUB_FRAGMENT_SHADER,
	WEBGL_PRESENT_FRAGMENT_SHADER,
	WEBGL_PRESENT_VERTEX_SHADER,
	WEBGL_SHADOW_DEPTH_FRAGMENT_SHADER,
	WEBGL_SHADOW_DEPTH_VERTEX_SHADER,
	WEBGL_SKYBOX_FRAGMENT_SHADER,
	WEBGL_SKYBOX_VERTEX_SHADER,
	WEBGL_TAA_FRAGMENT_SHADER,
} from "../../shaders/webgl/pipelineShaders";
import { createWebGLSceneShaderSource } from "../../shaders/webgl/sceneShader";

export interface WebGLShadowDepthProgram {
	program: WebGLProgram;
	uniforms: {
		mvp: WebGLUniformLocation | null;
	};
}

export interface WebGLCopyProgram {
	program: WebGLProgram;
	uniforms: {
		sourceMap: WebGLUniformLocation | null;
	};
}

export interface WebGLSSAOProgram {
	program: WebGLProgram;
	uniforms: {
		sceneColor: WebGLUniformLocation | null;
		depthMap: WebGLUniformLocation | null;
		normalMap: WebGLUniformLocation | null;
		texelSize: WebGLUniformLocation | null;
	};
}

export interface WebGLTAAProgram {
	program: WebGLProgram;
	uniforms: {
		sceneColor: WebGLUniformLocation | null;
		historyMap: WebGLUniformLocation | null;
		motionMap: WebGLUniformLocation | null;
		motionHistory: WebGLUniformLocation | null;
		texelSize: WebGLUniformLocation | null;
		historyWeight: WebGLUniformLocation | null;
		depthThreshold: WebGLUniformLocation | null;
		motionFactor: WebGLUniformLocation | null;
		varianceClampGamma: WebGLUniformLocation | null;
		sharpen: WebGLUniformLocation | null;
		historyValid: WebGLUniformLocation | null;
	};
}

export interface WebGLSSRProgram {
	program: WebGLProgram;
	uniforms: {
		sceneColor: WebGLUniformLocation | null;
		depthMap: WebGLUniformLocation | null;
		normalMap: WebGLUniformLocation | null;
		texelSize: WebGLUniformLocation | null;
	};
}

export interface WebGLVolumetricProgram {
	program: WebGLProgram;
	uniforms: {
		sceneColor: WebGLUniformLocation | null;
		depthMap: WebGLUniformLocation | null;
		texelSize: WebGLUniformLocation | null;
	};
}


export interface WebGLSceneProgram {
	program: WebGLProgram;
	uniforms: {
		model: WebGLUniformLocation | null;
		viewProjection: WebGLUniformLocation | null;
		normalMatrix: WebGLUniformLocation | null;
		cameraPosition: WebGLUniformLocation | null;
		ambientColor: WebGLUniformLocation | null;
		enableLighting: WebGLUniformLocation | null;
		shadingModel: WebGLUniformLocation | null;
		baseColor: WebGLUniformLocation | null;
		emissive: WebGLUniformLocation | null;
		pbr: WebGLUniformLocation | null;
		phong: WebGLUniformLocation | null;
		alpha: WebGLUniformLocation | null;
		baseMap: WebGLUniformLocation | null;
		hasBaseMap: WebGLUniformLocation | null;
		baseMapIsLinear: WebGLUniformLocation | null;
		dirLightCount: WebGLUniformLocation | null;
		dirLightDirection: WebGLUniformLocation | null;
		dirLightColor: WebGLUniformLocation | null;
		pointLightCount: WebGLUniformLocation | null;
		pointLightPositionRange: WebGLUniformLocation | null;
		pointLightColor: WebGLUniformLocation | null;
		spotLightCount: WebGLUniformLocation | null;
		spotLightPositionRange: WebGLUniformLocation | null;
		spotLightDirectionOuter: WebGLUniformLocation | null;
		spotLightColorInner: WebGLUniformLocation | null;
		taaJitter: WebGLUniformLocation | null;
		prevViewProjection: WebGLUniformLocation | null;
		prevModel: WebGLUniformLocation | null;
	};
}

export interface WebGLSkyboxProgram {
	program: WebGLProgram;
	uniforms: {
		skyboxMap: WebGLUniformLocation | null;
		skyboxBasisRight: WebGLUniformLocation | null;
		skyboxBasisUp: WebGLUniformLocation | null;
		skyboxBasisBackward: WebGLUniformLocation | null;
		skyboxIsOrthographic: WebGLUniformLocation | null;
		skyboxMapIsLinear: WebGLUniformLocation | null;
	};
}

export interface WebGLPresentProgram {
	program: WebGLProgram;
	uniforms: {
		sourceMap: WebGLUniformLocation | null;
		applyGamma: WebGLUniformLocation | null;
	};
}

export interface WebGLParticleProgram {
	program: WebGLProgram;
	uniforms: {
		viewProjection: WebGLUniformLocation | null;
		basisRight: WebGLUniformLocation | null;
		basisUp: WebGLUniformLocation | null;
		particleMap: WebGLUniformLocation | null;
		uvTransformA: WebGLUniformLocation | null;
		uvTransformB: WebGLUniformLocation | null;
		mapIsLinear: WebGLUniformLocation | null;
	};
}

export interface WebGLFXAAProgram {
	program: WebGLProgram;
	uniforms: {
		sourceMap: WebGLUniformLocation | null;
		texelSize: WebGLUniformLocation | null;
	};
}

type WarnFn = (key: string, message: string) => void;

const {
	vertex: SCENE_VERTEX_SHADER,
	fragment: SCENE_FRAGMENT_SHADER,
} = createWebGLSceneShaderSource({
	maxDirectionalLights: WEBGL_MAX_DIRECTIONAL_LIGHTS,
	maxPointLights: WEBGL_MAX_POINT_LIGHTS,
	maxSpotLights: WEBGL_MAX_SPOT_LIGHTS,
});


export class WebGLProgramLibrary {
	private _gl: WebGL2RenderingContext;
	private _warn: WarnFn;
	private _sceneProgram: WebGLSceneProgram | null = null;
	private _skyboxProgram: WebGLSkyboxProgram | null = null;
	private _presentProgram: WebGLPresentProgram | null = null;
	private _particleProgram: WebGLParticleProgram | null = null;
	private _fxaaProgram: WebGLFXAAProgram | null = null;

	private _shadowDepthProgram: WebGLShadowDepthProgram | null = null;
	private _copyProgram: WebGLCopyProgram | null = null;
	private _ssaoProgram: WebGLSSAOProgram | null = null;
	private _taaProgram: WebGLTAAProgram | null = null;
	private _ssrProgram: WebGLSSRProgram | null = null;
	private _volumetricProgram: WebGLVolumetricProgram | null = null;

	constructor(gl: WebGL2RenderingContext, warn: WarnFn) {
		this._gl = gl;
		this._warn = warn;
	}

	public getSceneProgram(): WebGLSceneProgram {
		if (this._sceneProgram) {
			return this._sceneProgram;
		}
		const program = this._createProgram(
			SCENE_VERTEX_SHADER,
			SCENE_FRAGMENT_SHADER,
			"WebGLSceneProgram"
		);
		this._sceneProgram = {
			program,
			uniforms: {
				model: this._gl.getUniformLocation(program, "uModel"),
				viewProjection: this._gl.getUniformLocation(program, "uViewProjection"),
				normalMatrix: this._gl.getUniformLocation(program, "uNormalMatrix"),
				cameraPosition: this._gl.getUniformLocation(program, "uCameraPosition"),
				ambientColor: this._gl.getUniformLocation(program, "uAmbientColor"),
				enableLighting: this._gl.getUniformLocation(program, "uEnableLighting"),
				shadingModel: this._gl.getUniformLocation(program, "uShadingModel"),
				baseColor: this._gl.getUniformLocation(program, "uBaseColor"),
				emissive: this._gl.getUniformLocation(program, "uEmissive"),
				pbr: this._gl.getUniformLocation(program, "uPBR"),
				phong: this._gl.getUniformLocation(program, "uPhong"),
				alpha: this._gl.getUniformLocation(program, "uAlpha"),
				baseMap: this._gl.getUniformLocation(program, "uBaseMap"),
				hasBaseMap: this._gl.getUniformLocation(program, "uHasBaseMap"),
				baseMapIsLinear: this._gl.getUniformLocation(
					program,
					"uBaseMapIsLinear"
				),
				dirLightCount: this._gl.getUniformLocation(program, "uDirLightCount"),
				dirLightDirection: this._gl.getUniformLocation(
					program,
					"uDirLightDirection"
				),
				dirLightColor: this._gl.getUniformLocation(program, "uDirLightColor"),
				pointLightCount: this._gl.getUniformLocation(
					program,
					"uPointLightCount"
				),
				pointLightPositionRange: this._gl.getUniformLocation(
					program,
					"uPointLightPositionRange"
				),
				pointLightColor: this._gl.getUniformLocation(
					program,
					"uPointLightColor"
				),
				spotLightCount: this._gl.getUniformLocation(program, "uSpotLightCount"),
				spotLightPositionRange: this._gl.getUniformLocation(
					program,
					"uSpotLightPositionRange"
				),
				spotLightDirectionOuter: this._gl.getUniformLocation(
					program,
					"uSpotLightDirectionOuter"
				),
				spotLightColorInner: this._gl.getUniformLocation(
					program,
					"uSpotLightColorInner"
				),
				taaJitter: this._gl.getUniformLocation(program, "uTaaJitter"),
				prevViewProjection: this._gl.getUniformLocation(
					program,
					"uPrevViewProjection"
				),
				prevModel: this._gl.getUniformLocation(program, "uPrevModel"),
			},
		};
		return this._sceneProgram;
	}

	public getSkyboxProgram(): WebGLSkyboxProgram {
		if (this._skyboxProgram) {
			return this._skyboxProgram;
		}
		const program = this._createProgram(
			WEBGL_SKYBOX_VERTEX_SHADER,
			WEBGL_SKYBOX_FRAGMENT_SHADER,
			"WebGLSkyboxProgram"
		);
		this._skyboxProgram = {
			program,
			uniforms: {
				skyboxMap: this._gl.getUniformLocation(program, "uSkyboxMap"),
				skyboxBasisRight: this._gl.getUniformLocation(
					program,
					"uSkyboxBasisRight"
				),
				skyboxBasisUp: this._gl.getUniformLocation(program, "uSkyboxBasisUp"),
				skyboxBasisBackward: this._gl.getUniformLocation(
					program,
					"uSkyboxBasisBackward"
				),
				skyboxIsOrthographic: this._gl.getUniformLocation(
					program,
					"uSkyboxIsOrthographic"
				),
				skyboxMapIsLinear: this._gl.getUniformLocation(
					program,
					"uSkyboxMapIsLinear"
				),
			},
		};
		return this._skyboxProgram;
	}

	public getPresentProgram(): WebGLPresentProgram {
		if (this._presentProgram) {
			return this._presentProgram;
		}
		const program = this._createProgram(
			WEBGL_PRESENT_VERTEX_SHADER,
			WEBGL_PRESENT_FRAGMENT_SHADER,
			"WebGLPresentProgram"
		);
		this._presentProgram = {
			program,
			uniforms: {
				sourceMap: this._gl.getUniformLocation(program, "uSourceMap"),
				applyGamma: this._gl.getUniformLocation(program, "uApplyGamma"),
			},
		};
		return this._presentProgram;
	}

	public getParticleProgram(): WebGLParticleProgram {
		if (this._particleProgram) {
			return this._particleProgram;
		}
		const program = this._createProgram(
			WEBGL_PARTICLE_VERTEX_SHADER,
			WEBGL_PARTICLE_FRAGMENT_SHADER,
			"WebGLParticleProgram"
		);
		this._particleProgram = {
			program,
			uniforms: {
				viewProjection: this._gl.getUniformLocation(program, "uViewProjection"),
				basisRight: this._gl.getUniformLocation(program, "uBasisRight"),
				basisUp: this._gl.getUniformLocation(program, "uBasisUp"),
				particleMap: this._gl.getUniformLocation(program, "uParticleMap"),
				uvTransformA: this._gl.getUniformLocation(program, "uUvTransformA"),
				uvTransformB: this._gl.getUniformLocation(program, "uUvTransformB"),
				mapIsLinear: this._gl.getUniformLocation(program, "uMapIsLinear"),
			},
		};
		return this._particleProgram;
	}

	public getFXAAProgram(): WebGLFXAAProgram {
		if (this._fxaaProgram) {
			return this._fxaaProgram;
		}
		const program = this._createProgram(
			WEBGL_FXAA_VERTEX_SHADER,
			WEBGL_FXAA_FRAGMENT_SHADER,
			"WebGLFXAAProgram"
		);
		this._fxaaProgram = {
			program,
			uniforms: {
				sourceMap: this._gl.getUniformLocation(program, "uSourceMap"),
				texelSize: this._gl.getUniformLocation(program, "uTexelSize"),
			},
		};
		return this._fxaaProgram;
	}

	public getShadowDepthProgram(): WebGLShadowDepthProgram {
		if (this._shadowDepthProgram) {
			return this._shadowDepthProgram;
		}
		const program = this._createProgram(
			WEBGL_SHADOW_DEPTH_VERTEX_SHADER,
			WEBGL_SHADOW_DEPTH_FRAGMENT_SHADER,
			"WebGLShadowDepthProgram"
		);
		this._shadowDepthProgram = {
			program,
			uniforms: {
				mvp: this._gl.getUniformLocation(program, "uMvp"),
			},
		};
		return this._shadowDepthProgram;
	}

	public getCopyProgram(): WebGLCopyProgram {
		if (this._copyProgram) {
			return this._copyProgram;
		}
		const program = this._createProgram(
			WEBGL_COPY_VERTEX_SHADER,
			WEBGL_COPY_FRAGMENT_SHADER,
			"WebGLCopyProgram"
		);
		this._copyProgram = {
			program,
			uniforms: {
				sourceMap: this._gl.getUniformLocation(program, "uSourceMap"),
			},
		};
		return this._copyProgram;
	}

	public getSSAOProgram(): WebGLSSAOProgram {
		if (this._ssaoProgram) {
			return this._ssaoProgram;
		}
		const program = this._createProgram(
			WEBGL_PRESENT_VERTEX_SHADER,
			WEBGL_POST_PROCESS_STUB_FRAGMENT_SHADER,
			"WebGLSSAOProgram"
		);
		this._ssaoProgram = {
			program,
			uniforms: {
				sceneColor: this._gl.getUniformLocation(program, "uSceneColor"),
				depthMap: this._gl.getUniformLocation(program, "uDepthMap"),
				normalMap: this._gl.getUniformLocation(program, "uNormalMap"),
				texelSize: this._gl.getUniformLocation(program, "uTexelSize"),
			},
		};
		return this._ssaoProgram;
	}

	public getTAAProgram(): WebGLTAAProgram {
		if (this._taaProgram) {
			return this._taaProgram;
		}
		const program = this._createProgram(
			WEBGL_PRESENT_VERTEX_SHADER,
			WEBGL_TAA_FRAGMENT_SHADER,
			"WebGLTAAProgram"
		);
		this._taaProgram = {
			program,
			uniforms: {
				sceneColor: this._gl.getUniformLocation(program, "uSceneColor"),
				historyMap: this._gl.getUniformLocation(program, "uHistoryMap"),
				motionMap: this._gl.getUniformLocation(program, "uMotionMap"),
				motionHistory: this._gl.getUniformLocation(program, "uMotionHistory"),
				texelSize: this._gl.getUniformLocation(program, "uTexelSize"),
				historyWeight: this._gl.getUniformLocation(program, "uHistoryWeight"),
				depthThreshold: this._gl.getUniformLocation(program, "uDepthThreshold"),
				motionFactor: this._gl.getUniformLocation(program, "uMotionFactor"),
				varianceClampGamma: this._gl.getUniformLocation(
					program,
					"uVarianceClampGamma"
				),
				sharpen: this._gl.getUniformLocation(program, "uSharpen"),
				historyValid: this._gl.getUniformLocation(program, "uHistoryValid"),
			},
		};
		return this._taaProgram;
	}

	public getSSRProgram(): WebGLSSRProgram {
		if (this._ssrProgram) {
			return this._ssrProgram;
		}
		const program = this._createProgram(
			WEBGL_PRESENT_VERTEX_SHADER,
			WEBGL_POST_PROCESS_STUB_FRAGMENT_SHADER,
			"WebGLSSRProgram"
		);
		this._ssrProgram = {
			program,
			uniforms: {
				sceneColor: this._gl.getUniformLocation(program, "uSceneColor"),
				depthMap: this._gl.getUniformLocation(program, "uDepthMap"),
				normalMap: this._gl.getUniformLocation(program, "uNormalMap"),
				texelSize: this._gl.getUniformLocation(program, "uTexelSize"),
			},
		};
		return this._ssrProgram;
	}

	public getVolumetricProgram(): WebGLVolumetricProgram {
		if (this._volumetricProgram) {
			return this._volumetricProgram;
		}
		const program = this._createProgram(
			WEBGL_PRESENT_VERTEX_SHADER,
			WEBGL_POST_PROCESS_STUB_FRAGMENT_SHADER,
			"WebGLVolumetricProgram"
		);
		this._volumetricProgram = {
			program,
			uniforms: {
				sceneColor: this._gl.getUniformLocation(program, "uSceneColor"),
				depthMap: this._gl.getUniformLocation(program, "uDepthMap"),
				texelSize: this._gl.getUniformLocation(program, "uTexelSize"),
			},
		};
		return this._volumetricProgram;
	}

	public destroy(): void {
		if (this._sceneProgram) {
			this._gl.deleteProgram(this._sceneProgram.program);
			this._sceneProgram = null;
		}
		if (this._skyboxProgram) {
			this._gl.deleteProgram(this._skyboxProgram.program);
			this._skyboxProgram = null;
		}
		if (this._presentProgram) {
			this._gl.deleteProgram(this._presentProgram.program);
			this._presentProgram = null;
		}
		if (this._particleProgram) {
			this._gl.deleteProgram(this._particleProgram.program);
			this._particleProgram = null;
		}
		if (this._fxaaProgram) {
			this._gl.deleteProgram(this._fxaaProgram.program);
			this._fxaaProgram = null;
		}
		if (this._shadowDepthProgram) {
			this._gl.deleteProgram(this._shadowDepthProgram.program);
			this._shadowDepthProgram = null;
		}
		if (this._copyProgram) {
			this._gl.deleteProgram(this._copyProgram.program);
			this._copyProgram = null;
		}
		if (this._ssaoProgram) {
			this._gl.deleteProgram(this._ssaoProgram.program);
			this._ssaoProgram = null;
		}
		if (this._taaProgram) {
			this._gl.deleteProgram(this._taaProgram.program);
			this._taaProgram = null;
		}
		if (this._ssrProgram) {
			this._gl.deleteProgram(this._ssrProgram.program);
			this._ssrProgram = null;
		}
		if (this._volumetricProgram) {
			this._gl.deleteProgram(this._volumetricProgram.program);
			this._volumetricProgram = null;
		}
	}

	private _createProgram(
		vertexSource: string,
		fragmentSource: string,
		label: string
	): WebGLProgram {
		const gl = this._gl;
		const vertexShader = this._compileShader(
			gl.VERTEX_SHADER,
			vertexSource,
			`${label}:vertex`
		);
		const fragmentShader = this._compileShader(
			gl.FRAGMENT_SHADER,
			fragmentSource,
			`${label}:fragment`
		);
		const program = gl.createProgram();
		if (!program) {
			gl.deleteShader(vertexShader);
			gl.deleteShader(fragmentShader);
			throw new Error(`Failed to create WebGL program (${label})`);
		}

		gl.attachShader(program, vertexShader);
		gl.attachShader(program, fragmentShader);
		gl.linkProgram(program);
		gl.deleteShader(vertexShader);
		gl.deleteShader(fragmentShader);

		const linked = !!gl.getProgramParameter(program, gl.LINK_STATUS);
		if (!linked) {
			const log = gl.getProgramInfoLog(program) || "No program link log";
			gl.deleteProgram(program);
			throw new Error(`WebGL program link failed (${label}): ${log}`);
		}

		const validateStatus = gl.getProgramParameter(program, gl.VALIDATE_STATUS);
		if (validateStatus === false) {
			this._warn(
				`webgl-program-validate-${label}`,
				`WebGL program validation reported issues (${label}): ${gl.getProgramInfoLog(program) || "no log"}`
			);
		}

		return program;
	}

	private _compileShader(
		type: number,
		source: string,
		label: string
	): WebGLShader {
		const gl = this._gl;
		const shader = gl.createShader(type);
		if (!shader) {
			throw new Error(`Failed to create WebGL shader (${label})`);
		}
		gl.shaderSource(shader, source);
		gl.compileShader(shader);
		const compiled = !!gl.getShaderParameter(shader, gl.COMPILE_STATUS);
		if (!compiled) {
			const log = gl.getShaderInfoLog(shader) || "No shader compile log";
			gl.deleteShader(shader);
			throw new Error(`WebGL shader compile failed (${label}): ${log}`);
		}
		return shader;
	}
}
