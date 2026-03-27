import {
	WEBGL_MAX_DIRECTIONAL_LIGHTS,
	WEBGL_MAX_POINT_LIGHTS,
	WEBGL_MAX_SPOT_LIGHTS,
} from "./constants";
import type { Material } from "../../materials/Material";
import { ShaderMaterial } from "../../materials/ShaderMaterial";
import {
	createInlineShaderSourceMap,
	mapShaderCompilerMessages,
	parseWebGLShaderInfoLog,
	ShaderCompileError,
	type ShaderCompilerMessage,
	type ShaderSourceSegmentMap,
	type ShaderProcessResult,
	type ShaderRuntime,
} from "../../shaders/runtime";
import {
	WEBGL_COPY_FRAGMENT_SHADER,
	WEBGL_COPY_VERTEX_SHADER,
	WEBGL_BLOOM_FRAGMENT_SHADER,
	WEBGL_DOF_FRAGMENT_SHADER,
	WEBGL_FXAA_FRAGMENT_SHADER,
	WEBGL_FXAA_VERTEX_SHADER,
	WEBGL_MOTION_BLUR_FRAGMENT_SHADER,
	WEBGL_PARTICLE_FRAGMENT_SHADER,
	WEBGL_PARTICLE_VERTEX_SHADER,
	WEBGL_POST_PROCESS_STUB_FRAGMENT_SHADER,
	WEBGL_PRESENT_FRAGMENT_SHADER,
	WEBGL_PRESENT_VERTEX_SHADER,
	WEBGL_SSAO_BLUR_FRAGMENT_SHADER,
	WEBGL_SSAO_COMBINE_FRAGMENT_SHADER,
	WEBGL_SSAO_RAW_FRAGMENT_SHADER,
	WEBGL_SHADOW_DEPTH_FRAGMENT_SHADER,
	WEBGL_SHADOW_DEPTH_VERTEX_SHADER,
	WEBGL_SKYBOX_FRAGMENT_SHADER,
	WEBGL_SKYBOX_VERTEX_SHADER,
	WEBGL_TAA_FRAGMENT_SHADER,
} from "../../shaders/webgl/pipelineShaders";
import {
	createWebGLSceneCompositeShaderSource,
	createWebGLSceneShaderSource,
} from "../../shaders/webgl/sceneShader";

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

export interface WebGLSSAORawProgram {
	program: WebGLProgram;
	uniforms: {
		normalMap: WebGLUniformLocation | null;
		depthMap: WebGLUniformLocation | null;
		invSize: WebGLUniformLocation | null;
		gtao: WebGLUniformLocation | null;
		blurProj: WebGLUniformLocation | null;
		pass: WebGLUniformLocation | null;
		cameraPosition: WebGLUniformLocation | null;
		basisRight: WebGLUniformLocation | null;
		basisUp: WebGLUniformLocation | null;
		basisBackward: WebGLUniformLocation | null;
	};
}

export interface WebGLSSAOBlurProgram {
	program: WebGLProgram;
	uniforms: {
		sourceMap: WebGLUniformLocation | null;
		depthMap: WebGLUniformLocation | null;
		invSize: WebGLUniformLocation | null;
		blurProj: WebGLUniformLocation | null;
		pass: WebGLUniformLocation | null;
	};
}

export interface WebGLSSAOCombineProgram {
	program: WebGLProgram;
	uniforms: {
		sceneColor: WebGLUniformLocation | null;
		aoMap: WebGLUniformLocation | null;
		invSize: WebGLUniformLocation | null;
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
		viewMatrix: WebGLUniformLocation | null;
		viewProjection: WebGLUniformLocation | null;
		normalMatrix: WebGLUniformLocation | null;
		cameraPosition: WebGLUniformLocation | null;
		ambientColor: WebGLUniformLocation | null;
		enableLighting: WebGLUniformLocation | null;
		enableShadows: WebGLUniformLocation | null;
		doubleSided: WebGLUniformLocation | null;
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
		shadowAtlas: WebGLUniformLocation | null;
		dirShadowViewProjection: WebGLUniformLocation | null;
		dirShadowParamsA: WebGLUniformLocation | null;
		dirShadowParamsB: WebGLUniformLocation | null;
		dirShadowParamsC: WebGLUniformLocation | null;
		spotShadowViewProjection: WebGLUniformLocation | null;
		spotShadowParamsA: WebGLUniformLocation | null;
		spotShadowParamsB: WebGLUniformLocation | null;
		spotShadowParamsC: WebGLUniformLocation | null;
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

export interface WebGLBloomProgram {
	program: WebGLProgram;
	uniforms: {
		sourceMap: WebGLUniformLocation | null;
		texelSize: WebGLUniformLocation | null;
		bloomParams: WebGLUniformLocation | null;
	};
}

export interface WebGLMotionBlurProgram {
	program: WebGLProgram;
	uniforms: {
		sourceMap: WebGLUniformLocation | null;
		motionDepthMap: WebGLUniformLocation | null;
		texelSize: WebGLUniformLocation | null;
		motionParams: WebGLUniformLocation | null;
		centerWeight: WebGLUniformLocation | null;
	};
}

export interface WebGLDOFProgram {
	program: WebGLProgram;
	uniforms: {
		sourceMap: WebGLUniformLocation | null;
		motionDepthMap: WebGLUniformLocation | null;
		texelSize: WebGLUniformLocation | null;
		focusParams: WebGLUniformLocation | null;
		dofParams: WebGLUniformLocation | null;
		chromaticAberration: WebGLUniformLocation | null;
	};
}

type WarnFn = (key: string, message: string) => void;

const SCENE_SHADER_SOURCE = createWebGLSceneShaderSource({
	maxDirectionalLights: WEBGL_MAX_DIRECTIONAL_LIGHTS,
	maxPointLights: WEBGL_MAX_POINT_LIGHTS,
	maxSpotLights: WEBGL_MAX_SPOT_LIGHTS,
});
const SCENE_SHADER_COMPOSITE_SOURCE = createWebGLSceneCompositeShaderSource({
	maxDirectionalLights: WEBGL_MAX_DIRECTIONAL_LIGHTS,
	maxPointLights: WEBGL_MAX_POINT_LIGHTS,
	maxSpotLights: WEBGL_MAX_SPOT_LIGHTS,
});

interface ShaderCompileMetadata {
	sourceMap?: ShaderSourceSegmentMap | null;
	variantKey?: string;
	materialId?: string;
	sourceKind?: "custom-material" | "unknown";
}


export class WebGLProgramLibrary {
	private _gl: WebGL2RenderingContext;
	private _warn: WarnFn;
	private _shaderRuntime: ShaderRuntime | null;
	private _disposeShaderRuntimeListener: (() => void) | null = null;
	private _sceneProgram: WebGLSceneProgram | null = null;
	private _customScenePrograms = new Map<string, WebGLSceneProgram>();
	private _skyboxProgram: WebGLSkyboxProgram | null = null;
	private _presentProgram: WebGLPresentProgram | null = null;
	private _particleProgram: WebGLParticleProgram | null = null;
	private _fxaaProgram: WebGLFXAAProgram | null = null;
	private _bloomProgram: WebGLBloomProgram | null = null;
	private _motionBlurProgram: WebGLMotionBlurProgram | null = null;
	private _dofProgram: WebGLDOFProgram | null = null;

	private _shadowDepthProgram: WebGLShadowDepthProgram | null = null;
	private _copyProgram: WebGLCopyProgram | null = null;
	private _ssaoRawProgram: WebGLSSAORawProgram | null = null;
	private _ssaoBlurProgram: WebGLSSAOBlurProgram | null = null;
	private _ssaoCombineProgram: WebGLSSAOCombineProgram | null = null;
	private _taaProgram: WebGLTAAProgram | null = null;
	private _ssrProgram: WebGLSSRProgram | null = null;
	private _volumetricProgram: WebGLVolumetricProgram | null = null;

	constructor(
		gl: WebGL2RenderingContext,
		warn: WarnFn,
		shaderRuntime?: ShaderRuntime
	) {
		this._gl = gl;
		this._warn = warn;
		this._shaderRuntime = shaderRuntime ?? null;
		if (this._shaderRuntime) {
			this._disposeShaderRuntimeListener = this._shaderRuntime.onDidChange(
				() => this._invalidateProgramCachesForShaderRuntime()
			);
		}
	}

	public getSceneProgram(material?: Material): WebGLSceneProgram {
		if (!(material instanceof ShaderMaterial)) {
			return this._getBuiltinSceneProgram();
		}

		const custom = this._getShaderMaterialSceneProgram(material);
		return custom ?? this._getBuiltinSceneProgram();
	}

	private _getBuiltinSceneProgram(): WebGLSceneProgram {
		if (!this._sceneProgram) {
			this._sceneProgram = this._createSceneProgram(
				SCENE_SHADER_SOURCE.vertex,
				SCENE_SHADER_SOURCE.fragment,
				"WebGLSceneProgram",
				{
					sourceMap: SCENE_SHADER_COMPOSITE_SOURCE.vertex.sourceMap,
					sourceKind: "unknown",
				},
				{
					sourceMap: SCENE_SHADER_COMPOSITE_SOURCE.fragment.sourceMap,
					sourceKind: "unknown",
				}
			);
		}
		return this._sceneProgram;
	}

	private _getShaderMaterialSceneProgram(
		material: ShaderMaterial
	): WebGLSceneProgram | null {
		const shaderKey =
			`${material.getWebGLCacheKey()}|runtime:${this._shaderRuntime?.revision ?? 0}`;
		const cached = this._customScenePrograms.get(shaderKey);
		if (cached) {
			return cached;
		}

		let source: { vertexCode: string; fragmentCode: string };
		try {
			source = material.resolveWebGLProgram();
		} catch (error) {
			this._warn(
				`webgl-shader-material-missing-source-${material.shaderId}`,
				`ShaderMaterial ${material.name} has no WebGL GLSL source; ` +
					`using built-in scene shader. ${String(error)}`
			);
			return null;
		}

		let sceneProgram: WebGLSceneProgram;
		try {
			sceneProgram = this._createSceneProgram(
				source.vertexCode,
				source.fragmentCode,
				`WebGLShaderMaterialProgram_${shaderKey}`,
				{
					sourceMap: createInlineShaderSourceMap(
						source.vertexCode,
						`<shader-material:${shaderKey}:vertex>`,
						"source"
					),
					variantKey: shaderKey,
					materialId: String(material.shaderId),
					sourceKind: "custom-material",
				},
				{
					sourceMap: createInlineShaderSourceMap(
						source.fragmentCode,
						`<shader-material:${shaderKey}:fragment>`,
						"source"
					),
					variantKey: shaderKey,
					materialId: String(material.shaderId),
					sourceKind: "custom-material",
				}
			);
		} catch (error) {
			if (!this._isWarnMode()) {
				throw error;
			}
			this._warn(
				`webgl-shader-material-compile-failed-${material.shaderId}`,
				`ShaderMaterial ${material.name} custom WebGL shader compile failed; ` +
					`using built-in scene shader. ${String(error)}`
			);
			return null;
		}
		this._customScenePrograms.set(shaderKey, sceneProgram);
		return sceneProgram;
	}

	private _createSceneProgram(
		vertexSource: string,
		fragmentSource: string,
		label: string,
		vertexMetadata?: ShaderCompileMetadata,
		fragmentMetadata?: ShaderCompileMetadata
	): WebGLSceneProgram {
		const program = this._createProgram(
			vertexSource,
			fragmentSource,
			label,
			vertexMetadata,
			fragmentMetadata
		);
		return {
			program,
			uniforms: {
				model: this._gl.getUniformLocation(program, "uModel"),
				viewMatrix: this._gl.getUniformLocation(program, "uViewMatrix"),
				viewProjection: this._gl.getUniformLocation(program, "uViewProjection"),
				normalMatrix: this._gl.getUniformLocation(program, "uNormalMatrix"),
				cameraPosition: this._gl.getUniformLocation(program, "uCameraPosition"),
				ambientColor: this._gl.getUniformLocation(program, "uAmbientColor"),
				enableLighting: this._gl.getUniformLocation(program, "uEnableLighting"),
				enableShadows: this._gl.getUniformLocation(program, "uEnableShadows"),
				doubleSided: this._gl.getUniformLocation(program, "uDoubleSided"),
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
				shadowAtlas: this._gl.getUniformLocation(program, "uShadowAtlas"),
				dirShadowViewProjection: this._gl.getUniformLocation(
					program,
					"uDirShadowViewProjection[0]"
				),
				dirShadowParamsA: this._gl.getUniformLocation(
					program,
					"uDirShadowParamsA[0]"
				),
				dirShadowParamsB: this._gl.getUniformLocation(
					program,
					"uDirShadowParamsB[0]"
				),
				dirShadowParamsC: this._gl.getUniformLocation(
					program,
					"uDirShadowParamsC[0]"
				),
				spotShadowViewProjection: this._gl.getUniformLocation(
					program,
					"uSpotShadowViewProjection[0]"
				),
				spotShadowParamsA: this._gl.getUniformLocation(
					program,
					"uSpotShadowParamsA[0]"
				),
				spotShadowParamsB: this._gl.getUniformLocation(
					program,
					"uSpotShadowParamsB[0]"
				),
				spotShadowParamsC: this._gl.getUniformLocation(
					program,
					"uSpotShadowParamsC[0]"
				),
				taaJitter: this._gl.getUniformLocation(program, "uTaaJitter"),
				prevViewProjection: this._gl.getUniformLocation(
					program,
					"uPrevViewProjection"
				),
				prevModel: this._gl.getUniformLocation(program, "uPrevModel"),
			},
		};
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

	public getBloomProgram(): WebGLBloomProgram {
		if (this._bloomProgram) {
			return this._bloomProgram;
		}
		const program = this._createProgram(
			WEBGL_PRESENT_VERTEX_SHADER,
			WEBGL_BLOOM_FRAGMENT_SHADER,
			"WebGLBloomProgram"
		);
		this._bloomProgram = {
			program,
			uniforms: {
				sourceMap: this._gl.getUniformLocation(program, "uSourceMap"),
				texelSize: this._gl.getUniformLocation(program, "uTexelSize"),
				bloomParams: this._gl.getUniformLocation(program, "uBloomParams"),
			},
		};
		return this._bloomProgram;
	}

	public getMotionBlurProgram(): WebGLMotionBlurProgram {
		if (this._motionBlurProgram) {
			return this._motionBlurProgram;
		}
		const program = this._createProgram(
			WEBGL_PRESENT_VERTEX_SHADER,
			WEBGL_MOTION_BLUR_FRAGMENT_SHADER,
			"WebGLMotionBlurProgram"
		);
		this._motionBlurProgram = {
			program,
			uniforms: {
				sourceMap: this._gl.getUniformLocation(program, "uSourceMap"),
				motionDepthMap: this._gl.getUniformLocation(program, "uMotionDepthMap"),
				texelSize: this._gl.getUniformLocation(program, "uTexelSize"),
				motionParams: this._gl.getUniformLocation(program, "uMotionParams"),
				centerWeight: this._gl.getUniformLocation(program, "uCenterWeight"),
			},
		};
		return this._motionBlurProgram;
	}

	public getDOFProgram(): WebGLDOFProgram {
		if (this._dofProgram) {
			return this._dofProgram;
		}
		const program = this._createProgram(
			WEBGL_PRESENT_VERTEX_SHADER,
			WEBGL_DOF_FRAGMENT_SHADER,
			"WebGLDOFProgram"
		);
		this._dofProgram = {
			program,
			uniforms: {
				sourceMap: this._gl.getUniformLocation(program, "uSourceMap"),
				motionDepthMap: this._gl.getUniformLocation(program, "uMotionDepthMap"),
				texelSize: this._gl.getUniformLocation(program, "uTexelSize"),
				focusParams: this._gl.getUniformLocation(program, "uFocusParams"),
				dofParams: this._gl.getUniformLocation(program, "uDOFParams"),
				chromaticAberration: this._gl.getUniformLocation(
					program,
					"uChromaticAberration"
				),
			},
		};
		return this._dofProgram;
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

	public getSSAORawProgram(): WebGLSSAORawProgram {
		if (this._ssaoRawProgram) {
			return this._ssaoRawProgram;
		}
		const program = this._createProgram(
			WEBGL_PRESENT_VERTEX_SHADER,
			WEBGL_SSAO_RAW_FRAGMENT_SHADER,
			"WebGLSSAORawProgram"
		);
		this._ssaoRawProgram = {
			program,
			uniforms: {
				normalMap: this._gl.getUniformLocation(program, "uNormalMap"),
				depthMap: this._gl.getUniformLocation(program, "uDepthMap"),
				invSize: this._gl.getUniformLocation(program, "uInvSize"),
				gtao: this._gl.getUniformLocation(program, "uGTAO"),
				blurProj: this._gl.getUniformLocation(program, "uBlurProj"),
				pass: this._gl.getUniformLocation(program, "uPass"),
				cameraPosition: this._gl.getUniformLocation(program, "uCameraPosition"),
				basisRight: this._gl.getUniformLocation(program, "uBasisRight"),
				basisUp: this._gl.getUniformLocation(program, "uBasisUp"),
				basisBackward: this._gl.getUniformLocation(program, "uBasisBackward"),
			},
		};
		return this._ssaoRawProgram;
	}

	public getSSAOBlurProgram(): WebGLSSAOBlurProgram {
		if (this._ssaoBlurProgram) {
			return this._ssaoBlurProgram;
		}
		const program = this._createProgram(
			WEBGL_PRESENT_VERTEX_SHADER,
			WEBGL_SSAO_BLUR_FRAGMENT_SHADER,
			"WebGLSSAOBlurProgram"
		);
		this._ssaoBlurProgram = {
			program,
			uniforms: {
				sourceMap: this._gl.getUniformLocation(program, "uSourceMap"),
				depthMap: this._gl.getUniformLocation(program, "uDepthMap"),
				invSize: this._gl.getUniformLocation(program, "uInvSize"),
				blurProj: this._gl.getUniformLocation(program, "uBlurProj"),
				pass: this._gl.getUniformLocation(program, "uPass"),
			},
		};
		return this._ssaoBlurProgram;
	}

	public getSSAOCombineProgram(): WebGLSSAOCombineProgram {
		if (this._ssaoCombineProgram) {
			return this._ssaoCombineProgram;
		}
		const program = this._createProgram(
			WEBGL_PRESENT_VERTEX_SHADER,
			WEBGL_SSAO_COMBINE_FRAGMENT_SHADER,
			"WebGLSSAOCombineProgram"
		);
		this._ssaoCombineProgram = {
			program,
			uniforms: {
				sceneColor: this._gl.getUniformLocation(program, "uSceneColor"),
				aoMap: this._gl.getUniformLocation(program, "uAoMap"),
				invSize: this._gl.getUniformLocation(program, "uInvSize"),
			},
		};
		return this._ssaoCombineProgram;
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
		this._disposeShaderRuntimeListener?.();
		this._disposeShaderRuntimeListener = null;
		this._disposePrograms();
	}

	private _createProgram(
		vertexSource: string,
		fragmentSource: string,
		label: string,
		vertexMetadata?: ShaderCompileMetadata,
		fragmentMetadata?: ShaderCompileMetadata
	): WebGLProgram {
		const gl = this._gl;
		const vertexShader = this._compileShader(
			gl.VERTEX_SHADER,
			vertexSource,
			`${label}:vertex`,
			vertexMetadata
		);
		const fragmentShader = this._compileShader(
			gl.FRAGMENT_SHADER,
			fragmentSource,
			`${label}:fragment`,
			fragmentMetadata
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
			const messages = parseWebGLShaderInfoLog(log);
			throw new ShaderCompileError({
				backend: "webgl",
				language: "glsl",
				stage: "unknown",
				label,
				sourceKind:
					vertexMetadata?.sourceKind ?? fragmentMetadata?.sourceKind ?? "unknown",
				variantKey: vertexMetadata?.variantKey ?? fragmentMetadata?.variantKey,
				materialId: vertexMetadata?.materialId ?? fragmentMetadata?.materialId,
				code: `${vertexSource}\n\n${fragmentSource}`,
				sourceMap: null,
				messages: messages.length > 0 ? messages : [this._toCompilerMessage(log)],
				rawLog: log,
			});
		}

		gl.validateProgram(program);
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
		label: string,
		metadata?: ShaderCompileMetadata
	): WebGLShader {
		const stage = type === this._gl.VERTEX_SHADER ? "vertex" : "fragment";
		const sourceKind =
			metadata?.sourceKind ??
			(label.startsWith("WebGLShaderMaterialProgram_") ?
				"custom-material"
			:	"unknown");
		const processed = this._processShaderSource(
			source,
			stage,
			sourceKind,
			label,
			metadata?.sourceMap
		);
		if (processed.hasErrors) {
			this._reportShaderRuntimeDiagnostics(label, processed);
		}

		const gl = this._gl;
		const shader = gl.createShader(type);
		if (!shader) {
			throw new Error(`Failed to create WebGL shader (${label})`);
		}
		gl.shaderSource(shader, processed.code);
		gl.compileShader(shader);
		const compiled = !!gl.getShaderParameter(shader, gl.COMPILE_STATUS);
		if (!compiled) {
			const log = gl.getShaderInfoLog(shader) || "No shader compile log";
			gl.deleteShader(shader);
			const parsed = parseWebGLShaderInfoLog(log);
			throw new ShaderCompileError({
				backend: "webgl",
				language: "glsl",
				stage,
				label,
				sourceKind,
				variantKey: metadata?.variantKey,
				materialId: metadata?.materialId,
				code: processed.code,
				sourceMap: processed.sourceMap,
				messages: parsed.length > 0 ? parsed : [this._toCompilerMessage(log)],
				rawLog: log,
			});
		}
		return shader;
	}

	private _isWarnMode(): boolean {
		return this._shaderRuntime?.getMode() === "warn";
	}

	private _processShaderSource(
		source: string,
		stage: "vertex" | "fragment",
		sourceKind: "custom-material" | "unknown",
		label: string,
		sourceMap?: ShaderSourceSegmentMap | null
	): ShaderProcessResult {
		if (!this._shaderRuntime) {
			const effectiveSourceMap =
				sourceMap ?? createInlineShaderSourceMap(source, label, "source");
			return {
				code: source,
				sourceMap: effectiveSourceMap,
				composite: {
					code: source,
					sourceMap: effectiveSourceMap,
				},
				diagnostics: [],
				hasErrors: false,
				fromCache: false,
			};
		}
		return this._shaderRuntime.process({
			code: source,
			language: "glsl",
			stage,
			entryPoint: "main",
			label,
			sourceKind,
			sourceMap: sourceMap ?? null,
		});
	}

	private _reportShaderRuntimeDiagnostics(
		label: string,
		result: ShaderProcessResult
	): void {
		for (const diagnostic of result.diagnostics) {
			this._warn(
				`webgl-shader-runtime-${diagnostic.severity}-${diagnostic.code}-${label}`,
				`WebGL shader runtime ${diagnostic.severity} [${label}] ` +
					`${diagnostic.code}: ${diagnostic.message}`
			);
		}
	}

	private _toCompilerMessage(log: string): ShaderCompilerMessage {
		return {
			type: "error",
			message: log,
			raw: log,
		};
	}

	private _invalidateProgramCachesForShaderRuntime(): void {
		this._disposePrograms();
	}

	private _disposePrograms(): void {
		if (this._sceneProgram) {
			this._gl.deleteProgram(this._sceneProgram.program);
			this._sceneProgram = null;
		}
		for (const sceneProgram of this._customScenePrograms.values()) {
			this._gl.deleteProgram(sceneProgram.program);
		}
		this._customScenePrograms.clear();
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
		if (this._bloomProgram) {
			this._gl.deleteProgram(this._bloomProgram.program);
			this._bloomProgram = null;
		}
		if (this._motionBlurProgram) {
			this._gl.deleteProgram(this._motionBlurProgram.program);
			this._motionBlurProgram = null;
		}
		if (this._dofProgram) {
			this._gl.deleteProgram(this._dofProgram.program);
			this._dofProgram = null;
		}
		if (this._shadowDepthProgram) {
			this._gl.deleteProgram(this._shadowDepthProgram.program);
			this._shadowDepthProgram = null;
		}
		if (this._copyProgram) {
			this._gl.deleteProgram(this._copyProgram.program);
			this._copyProgram = null;
		}
		if (this._ssaoRawProgram) {
			this._gl.deleteProgram(this._ssaoRawProgram.program);
			this._ssaoRawProgram = null;
		}
		if (this._ssaoBlurProgram) {
			this._gl.deleteProgram(this._ssaoBlurProgram.program);
			this._ssaoBlurProgram = null;
		}
		if (this._ssaoCombineProgram) {
			this._gl.deleteProgram(this._ssaoCombineProgram.program);
			this._ssaoCombineProgram = null;
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
}
