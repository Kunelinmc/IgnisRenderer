import {
	WEBGL_MAX_DIRECTIONAL_LIGHTS,
	WEBGL_MAX_POINT_LIGHTS,
	WEBGL_MAX_SPOT_LIGHTS,
} from "./constants";
import {
	createWebGLSceneUniforms,
	type WebGLSceneUniforms,
} from "./WebGLSceneProgramUniforms";
import type { Material } from "../../materials/Material";
import {
	ShaderMaterial,
	type ShaderTargetMode,
} from "../../materials/ShaderMaterial";
import {
	DEFAULT_SHADER_DIRECTIVE_PROFILE_REGISTRY,
	ShaderBackendCompileStage,
	createInlineShaderSourceMap,
	mapShaderCompilerMessages,
	parseWebGLShaderInfoLog,
	ShaderCompileError,
	type ShaderCompilerMessage,
	type ShaderProcessResult,
	type ShaderSourceSegmentMap,
	type ShaderRuntime,
} from "../../shaders/runtime";
import {
	createWebGLShaderSourceFactory,
	type WebGLShaderPart,
	type WebGLShaderSourceFactory,
} from "../../shaders/webgl/WebGLShaderSourceFactory";
import { Logger } from "../../foundation/Logger";

export interface WebGLShadowDepthProgram {
	program: WebGLProgram;
	uniforms: {
		mvp: WebGLUniformLocation | null;
	};
}

export interface WebGLShadowTransmittanceProgram {
	program: WebGLProgram;
	uniforms: {
		mvp: WebGLUniformLocation | null;
		transmittance: WebGLUniformLocation | null;
	};
}

export interface WebGLCopyProgram {
	program: WebGLProgram;
	uniforms: {
		sourceMap: WebGLUniformLocation | null;
	};
}

export interface WebGLOITResolveProgram {
	program: WebGLProgram;
	uniforms: {
		sceneColor: WebGLUniformLocation | null;
		oitAccumMap: WebGLUniformLocation | null;
		oitRevealMap: WebGLUniformLocation | null;
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

export interface WebGLFogProgram {
	program: WebGLProgram;
	uniforms: {
		sceneColor: WebGLUniformLocation | null;
		motionDepthMap: WebGLUniformLocation | null;
		fogParams0: WebGLUniformLocation | null;
		fogParams1: WebGLUniformLocation | null;
	};
}


export interface WebGLSceneProgram {
	program: WebGLProgram;
	uniforms: WebGLSceneUniforms;
}

export interface WebGLEnvironmentProgram {
	program: WebGLProgram;
	uniforms: {
		environmentMap: WebGLUniformLocation | null;
		environmentBasisRight: WebGLUniformLocation | null;
		environmentBasisUp: WebGLUniformLocation | null;
		environmentBasisBackward: WebGLUniformLocation | null;
		environmentIsOrthographic: WebGLUniformLocation | null;
		environmentMapIsLinear: WebGLUniformLocation | null;
		environmentBackgroundTint: WebGLUniformLocation | null;
		environmentBackgroundExposure: WebGLUniformLocation | null;
		environmentBackgroundStrength: WebGLUniformLocation | null;
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
		cameraPosition: WebGLUniformLocation | null;
		fogParams0: WebGLUniformLocation | null;
		fogParams1: WebGLUniformLocation | null;
		oitPassMode: WebGLUniformLocation | null;
	};
}

export interface WebGLFXAAProgram {
	program: WebGLProgram;
	uniforms: {
		sourceMap: WebGLUniformLocation | null;
		texelSize: WebGLUniformLocation | null;
	};
}

export interface WebGLToneMappingProgram {
	program: WebGLProgram;
	uniforms: {
		sourceMap: WebGLUniformLocation | null;
	};
}

export interface WebGLColorFilterProgram {
	program: WebGLProgram;
	uniforms: {
		sourceMap: WebGLUniformLocation | null;
		filterParams0: WebGLUniformLocation | null;
		filterParams1: WebGLUniformLocation | null;
	};
}

export interface WebGLInteractionOutlineProgram {
	program: WebGLProgram;
	uniforms: {
		sourceMap: WebGLUniformLocation | null;
		outlineColor: WebGLUniformLocation | null;
		outlineParams: WebGLUniformLocation | null;
		viewportSize: WebGLUniformLocation | null;
		circleCount: WebGLUniformLocation | null;
		circles: WebGLUniformLocation | null;
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

interface ShaderCompileMetadata {
	sourceMap?: ShaderSourceSegmentMap | null;
	variantKey?: string;
	materialId?: string;
	sourceKind?: "custom-material" | "unknown";
}

type WebGLProgramWarn = (key: string, message: string) => void;

const WEBGL_MIN_TEXTURE_UNITS_FOR_SHADOW_TRANSMITTANCE = 17;

function supportsShadowTransmittanceSampler(
	gl: WebGL2RenderingContext
): boolean {
	const textureUnitParameter = gl.MAX_TEXTURE_IMAGE_UNITS;
	if (
		!Number.isFinite(textureUnitParameter) ||
		typeof gl.getParameter !== "function"
	) {
		return false;
	}
	try {
		const textureUnits = gl.getParameter(textureUnitParameter);
		return (
			Number.isFinite(textureUnits) &&
			Math.floor(textureUnits) >=
				WEBGL_MIN_TEXTURE_UNITS_FOR_SHADOW_TRANSMITTANCE
		);
	} catch {
		return false;
	}
}

export class WebGLProgramLibrary {
	private _gl: WebGL2RenderingContext;
	private _shaderRuntime: ShaderRuntime | null;
	private _shaderCompileStage: ShaderBackendCompileStage | null;
	private _shaderSourceFactory: WebGLShaderSourceFactory;
	private _enableShadowTransmittanceSampling: boolean;
	private _warnCallback: WebGLProgramWarn | null = null;
	private _disposeShaderRuntimeListener: (() => void) | null = null;
	private _sceneProgram: WebGLSceneProgram | null = null;
	private _sceneProgramDirectiveTag: string = "";
	private _customScenePrograms = new Map<string, WebGLSceneProgram>();
	private _environmentProgram: WebGLEnvironmentProgram | null = null;
	private _presentProgram: WebGLPresentProgram | null = null;
	private _particleProgram: WebGLParticleProgram | null = null;
	private _toneMappingProgram: WebGLToneMappingProgram | null = null;
	private _colorFilterProgram: WebGLColorFilterProgram | null = null;
	private _fxaaProgram: WebGLFXAAProgram | null = null;
	private _interactionOutlineProgram: WebGLInteractionOutlineProgram | null = null;
	private _bloomProgram: WebGLBloomProgram | null = null;
	private _motionBlurProgram: WebGLMotionBlurProgram | null = null;
	private _dofProgram: WebGLDOFProgram | null = null;

	private _shadowDepthProgram: WebGLShadowDepthProgram | null = null;
	private _shadowTransmittanceProgram: WebGLShadowTransmittanceProgram | null = null;
	private _copyProgram: WebGLCopyProgram | null = null;
	private _oitResolveProgram: WebGLOITResolveProgram | null = null;
	private _ssaoRawProgram: WebGLSSAORawProgram | null = null;
	private _ssaoBlurProgram: WebGLSSAOBlurProgram | null = null;
	private _ssaoCombineProgram: WebGLSSAOCombineProgram | null = null;
	private _taaProgram: WebGLTAAProgram | null = null;
	private _ssrProgram: WebGLSSRProgram | null = null;
	private _volumetricProgram: WebGLVolumetricProgram | null = null;
	private _fogProgram: WebGLFogProgram | null = null;

	constructor(
		gl: WebGL2RenderingContext,
		warn: WebGLProgramWarn,
		shaderRuntime?: ShaderRuntime,
		shaderCompileStage?: ShaderBackendCompileStage,
		shaderSourceFactory?: WebGLShaderSourceFactory,
	);
	constructor(
		gl: WebGL2RenderingContext,
		shaderRuntime?: ShaderRuntime,
		shaderCompileStage?: ShaderBackendCompileStage,
		shaderSourceFactory?: WebGLShaderSourceFactory,
	);
	constructor(
		gl: WebGL2RenderingContext,
		shaderRuntimeOrWarn?: ShaderRuntime | WebGLProgramWarn,
		shaderCompileStageOrRuntime?: ShaderBackendCompileStage | ShaderRuntime,
		shaderSourceFactoryOrCompileStage?:
			| WebGLShaderSourceFactory
			| ShaderBackendCompileStage,
		shaderSourceFactoryMaybe?: WebGLShaderSourceFactory,
	) {
		this._gl = gl;
		this._enableShadowTransmittanceSampling =
			supportsShadowTransmittanceSampler(gl);
		let shaderRuntime: ShaderRuntime | null = null;
		let shaderCompileStage: ShaderBackendCompileStage | null = null;
		let shaderSourceFactory: WebGLShaderSourceFactory | undefined;
		if (typeof shaderRuntimeOrWarn === "function") {
			this._warnCallback = shaderRuntimeOrWarn;
			shaderRuntime =
				isShaderRuntime(shaderCompileStageOrRuntime) ?
					shaderCompileStageOrRuntime
				:	null;
			shaderCompileStage =
				shaderSourceFactoryOrCompileStage instanceof ShaderBackendCompileStage ?
					shaderSourceFactoryOrCompileStage
				:	null;
			shaderSourceFactory = shaderSourceFactoryMaybe;
		} else {
			shaderRuntime = shaderRuntimeOrWarn ?? null;
			shaderCompileStage =
				shaderCompileStageOrRuntime instanceof ShaderBackendCompileStage ?
					shaderCompileStageOrRuntime
				:	null;
			shaderSourceFactory =
				shaderSourceFactoryOrCompileStage &&
				!(shaderSourceFactoryOrCompileStage instanceof ShaderBackendCompileStage) ?
					shaderSourceFactoryOrCompileStage
				:	undefined;
		}
		this._shaderRuntime = shaderRuntime;
		this._shaderCompileStage = shaderCompileStage;
		this._shaderSourceFactory =
			shaderSourceFactory ?? createWebGLShaderSourceFactory();
		if (!this._shaderCompileStage && this._shaderRuntime) {
			this._shaderCompileStage = new ShaderBackendCompileStage({
				backend: "webgl",
				runtime: this._shaderRuntime,
				profiles: DEFAULT_SHADER_DIRECTIVE_PROFILE_REGISTRY,
				mode: this._shaderRuntime.getMode(),
			});
		}
		if (this._shaderRuntime) {
			this._disposeShaderRuntimeListener = this._shaderRuntime.onDidChange(() =>
				this._invalidateProgramCachesForShaderRuntime(),
			);
		}
	}

	public getSceneProgram(
		material?: Material,
		mode: ShaderTargetMode = "single"
	): WebGLSceneProgram {
		if (!(material instanceof ShaderMaterial)) {
			return this._getBuiltinSceneProgram();
		}

		const custom = this._getShaderMaterialSceneProgram(material, mode);
		return custom ?? this._getBuiltinSceneProgram();
	}

	private _getBuiltinSceneProgram(): WebGLSceneProgram {
		const directiveTag = this._shaderCompileStage?.getCacheFingerprintTag() ?? "";
		if (this._sceneProgram && this._sceneProgramDirectiveTag === directiveTag) {
			return this._sceneProgram;
		}
		if (this._sceneProgram && this._sceneProgramDirectiveTag !== directiveTag) {
			this._gl.deleteProgram(this._sceneProgram.program);
			this._sceneProgram = null;
		}
		if (!this._sceneProgram) {
			const sceneShaderSource = this._shaderSourceFactory.createSceneShaderSource({
				maxDirectionalLights: WEBGL_MAX_DIRECTIONAL_LIGHTS,
				maxPointLights: WEBGL_MAX_POINT_LIGHTS,
				maxSpotLights: WEBGL_MAX_SPOT_LIGHTS,
				enableShadowTransmittance: this._enableShadowTransmittanceSampling,
			});
			const sceneCompositeSource = this._shaderSourceFactory.createSceneCompositeShaderSource(
				{
					maxDirectionalLights: WEBGL_MAX_DIRECTIONAL_LIGHTS,
					maxPointLights: WEBGL_MAX_POINT_LIGHTS,
					maxSpotLights: WEBGL_MAX_SPOT_LIGHTS,
					enableShadowTransmittance: this._enableShadowTransmittanceSampling,
				},
			);
			this._sceneProgram = this._createSceneProgram(
				sceneShaderSource.vertex,
				sceneShaderSource.fragment,
				"WebGLSceneProgram",
				{
					sourceMap: sceneCompositeSource.vertex.sourceMap,
					sourceKind: "unknown",
				},
				{
					sourceMap: sceneCompositeSource.fragment.sourceMap,
					sourceKind: "unknown",
				},
			);
		}
		this._sceneProgramDirectiveTag =
			this._shaderCompileStage?.getCacheFingerprintTag() ?? directiveTag;
		return this._sceneProgram;
	}

	private _getShaderMaterialSceneProgram(
		material: ShaderMaterial,
		mode: ShaderTargetMode
	): WebGLSceneProgram | null {
		const initialDirectiveTag = this._shaderCompileStage?.getCacheFingerprintTag() ?? "none";
		const shaderKey =
			`${material.getWebGLCacheKey()}` +
			`|mode:${mode}` +
			`|runtime:${this._shaderRuntime?.revision ?? 0}` +
			`|directive:${initialDirectiveTag}`;
		const cached = this._customScenePrograms.get(shaderKey);
		if (cached) {
			return cached;
		}

		let source: { vertexCode: string; fragmentCode: string };
		let customSamplerUniforms: string[] = [];
		try {
			source = material.resolveWebGLProgram(mode, {
				enableRuntimeInjects: this._supportsRuntimeInjects(),
			});
			customSamplerUniforms = this._collectCustomSamplerUniforms(material);
		} catch (error) {
			const key = `webgl-shader-material-missing-source-${material.shaderId}`;
			const message =
				`ShaderMaterial ${material.name} has no WebGL GLSL source; ` +
				`using built-in scene shader. ${String(error)}`;
			this._warn(key, message);
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
						"source",
					),
					variantKey: shaderKey,
					materialId: String(material.shaderId),
					sourceKind: "custom-material",
				},
				{
					sourceMap: createInlineShaderSourceMap(
						source.fragmentCode,
						`<shader-material:${shaderKey}:fragment>`,
						"source",
					),
					variantKey: shaderKey,
					materialId: String(material.shaderId),
					sourceKind: "custom-material",
				},
				customSamplerUniforms,
			);
		} catch (error) {
			if (!this._isWarnMode()) {
				throw error;
			}
			const key = `webgl-shader-material-compile-failed-${material.shaderId}`;
			const message =
				`ShaderMaterial ${material.name} custom WebGL shader compile failed; ` +
				`using built-in scene shader. ${String(error)}`;
			this._warn(key, message);
			return null;
		}
		const finalDirectiveTag =
			this._shaderCompileStage?.getCacheFingerprintTag() ?? initialDirectiveTag;
		const finalShaderKey =
			`${material.getWebGLCacheKey()}` +
			`|mode:${mode}` +
			`|runtime:${this._shaderRuntime?.revision ?? 0}` +
			`|directive:${finalDirectiveTag}`;
		const existingFinal = this._customScenePrograms.get(finalShaderKey);
		if (existingFinal) {
			this._gl.deleteProgram(sceneProgram.program);
			return existingFinal;
		}
		this._customScenePrograms.set(finalShaderKey, sceneProgram);
		if (finalShaderKey !== shaderKey) {
			this._customScenePrograms.delete(shaderKey);
		}
		return sceneProgram;
	}

	private _createSceneProgram(
		vertexSource: string,
		fragmentSource: string,
		label: string,
		vertexMetadata?: ShaderCompileMetadata,
		fragmentMetadata?: ShaderCompileMetadata,
		customSamplerUniforms: string[] = [],
	): WebGLSceneProgram {
		const program = this._createProgram(
			vertexSource,
			fragmentSource,
			label,
			vertexMetadata,
			fragmentMetadata,
		);
		return {
			program,
			uniforms: createWebGLSceneUniforms(
				this._gl,
				program,
				customSamplerUniforms
			),
		};
	}

	public getEnvironmentProgram(): WebGLEnvironmentProgram {
		if (this._environmentProgram) {
			return this._environmentProgram;
		}
		const program = this._createProgram(
			this._shaderSource("environmentVertex"),
			this._shaderSource("environmentFragment"),
			"WebGLEnvironmentProgram",
		);
		this._environmentProgram = {
			program,
			uniforms: {
				environmentMap: this._gl.getUniformLocation(program, "uEnvironmentMap"),
				environmentBasisRight: this._gl.getUniformLocation(program, "uEnvironmentBasisRight"),
				environmentBasisUp: this._gl.getUniformLocation(program, "uEnvironmentBasisUp"),
				environmentBasisBackward: this._gl.getUniformLocation(program, "uEnvironmentBasisBackward"),
				environmentIsOrthographic: this._gl.getUniformLocation(program, "uEnvironmentIsOrthographic"),
				environmentMapIsLinear: this._gl.getUniformLocation(program, "uEnvironmentMapIsLinear"),
				environmentBackgroundTint: this._gl.getUniformLocation(
					program,
					"uEnvironmentBackgroundTint"
				),
				environmentBackgroundExposure: this._gl.getUniformLocation(
					program,
					"uEnvironmentBackgroundExposure"
				),
				environmentBackgroundStrength: this._gl.getUniformLocation(
					program,
					"uEnvironmentBackgroundStrength"
				),
			},
		};
		return this._environmentProgram;
	}

	public getPresentProgram(): WebGLPresentProgram {
		if (this._presentProgram) {
			return this._presentProgram;
		}
		const program = this._createProgram(
			this._shaderSource("presentVertex"),
			this._shaderSource("presentFragment"),
			"WebGLPresentProgram",
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
			this._shaderSource("particleVertex"),
			this._shaderSource("particleFragment"),
			"WebGLParticleProgram",
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
				cameraPosition: this._gl.getUniformLocation(program, "uCameraPosition"),
				fogParams0: this._gl.getUniformLocation(program, "uFogParams0"),
				fogParams1: this._gl.getUniformLocation(program, "uFogParams1"),
				oitPassMode: this._gl.getUniformLocation(program, "uOITPassMode"),
			},
		};
		return this._particleProgram;
	}

	public getFXAAProgram(): WebGLFXAAProgram {
		if (this._fxaaProgram) {
			return this._fxaaProgram;
		}
		const program = this._createProgram(
			this._shaderSource("presentVertex"),
			this._shaderSource("fxaaFragment"),
			"WebGLFXAAProgram",
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

	public getToneMappingProgram(): WebGLToneMappingProgram {
		if (this._toneMappingProgram) {
			return this._toneMappingProgram;
		}
		const program = this._createProgram(
			this._shaderSource("presentVertex"),
			this._shaderSource("toneMappingFragment"),
			"WebGLToneMappingProgram",
		);
		this._toneMappingProgram = {
			program,
			uniforms: {
				sourceMap: this._gl.getUniformLocation(program, "uSourceMap"),
			},
		};
		return this._toneMappingProgram;
	}

	public getInteractionOutlineProgram(): WebGLInteractionOutlineProgram {
		if (this._interactionOutlineProgram) {
			return this._interactionOutlineProgram;
		}
		const program = this._createProgram(
			this._shaderSource("presentVertex"),
			this._shaderSource("interactionOutlineFragment"),
			"WebGLInteractionOutlineProgram",
		);
		this._interactionOutlineProgram = {
			program,
			uniforms: {
				sourceMap: this._gl.getUniformLocation(program, "uSourceMap"),
				outlineColor: this._gl.getUniformLocation(program, "uOutlineColor"),
				outlineParams: this._gl.getUniformLocation(program, "uOutlineParams"),
				viewportSize: this._gl.getUniformLocation(program, "uViewportSize"),
				circleCount: this._gl.getUniformLocation(program, "uCircleCount"),
				circles: this._gl.getUniformLocation(program, "uCircles[0]"),
			},
		};
		return this._interactionOutlineProgram;
	}

	public getColorFilterProgram(): WebGLColorFilterProgram {
		if (this._colorFilterProgram) {
			return this._colorFilterProgram;
		}
		const program = this._createProgram(
			this._shaderSource("presentVertex"),
			this._shaderSource("colorFilterFragment"),
			"WebGLColorFilterProgram",
		);
		this._colorFilterProgram = {
			program,
			uniforms: {
				sourceMap: this._gl.getUniformLocation(program, "uSourceMap"),
				filterParams0: this._gl.getUniformLocation(program, "uFilterParams0"),
				filterParams1: this._gl.getUniformLocation(program, "uFilterParams1"),
			},
		};
		return this._colorFilterProgram;
	}

	public getBloomProgram(): WebGLBloomProgram {
		if (this._bloomProgram) {
			return this._bloomProgram;
		}
		const program = this._createProgram(
			this._shaderSource("presentVertex"),
			this._shaderSource("bloomFragment"),
			"WebGLBloomProgram",
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
			this._shaderSource("presentVertex"),
			this._shaderSource("motionBlurFragment"),
			"WebGLMotionBlurProgram",
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
			this._shaderSource("presentVertex"),
			this._shaderSource("dofFragment"),
			"WebGLDOFProgram",
		);
		this._dofProgram = {
			program,
			uniforms: {
				sourceMap: this._gl.getUniformLocation(program, "uSourceMap"),
				motionDepthMap: this._gl.getUniformLocation(program, "uMotionDepthMap"),
				texelSize: this._gl.getUniformLocation(program, "uTexelSize"),
				focusParams: this._gl.getUniformLocation(program, "uFocusParams"),
				dofParams: this._gl.getUniformLocation(program, "uDOFParams"),
				chromaticAberration: this._gl.getUniformLocation(program, "uChromaticAberration"),
			},
		};
		return this._dofProgram;
	}

	public getShadowDepthProgram(): WebGLShadowDepthProgram {
		if (this._shadowDepthProgram) {
			return this._shadowDepthProgram;
		}
		const program = this._createProgram(
			this._shaderSource("shadowDepthVertex"),
			this._shaderSource("shadowDepthFragment"),
			"WebGLShadowDepthProgram",
		);
		this._shadowDepthProgram = {
			program,
			uniforms: {
				mvp: this._gl.getUniformLocation(program, "uMvp"),
			},
		};
		return this._shadowDepthProgram;
	}

	public getShadowTransmittanceProgram(): WebGLShadowTransmittanceProgram {
		if (this._shadowTransmittanceProgram) {
			return this._shadowTransmittanceProgram;
		}
		const program = this._createProgram(
			this._shaderSource("shadowDepthVertex"),
			this._shaderSource("shadowTransmittanceFragment"),
			"WebGLShadowTransmittanceProgram",
		);
		this._shadowTransmittanceProgram = {
			program,
			uniforms: {
				mvp: this._gl.getUniformLocation(program, "uMvp"),
				transmittance: this._gl.getUniformLocation(program, "uTransmittance"),
			},
		};
		return this._shadowTransmittanceProgram;
	}

	public getCopyProgram(): WebGLCopyProgram {
		if (this._copyProgram) {
			return this._copyProgram;
		}
		const program = this._createProgram(
			this._shaderSource("presentVertex"),
			this._shaderSource("copyFragment"),
			"WebGLCopyProgram",
		);
		this._copyProgram = {
			program,
			uniforms: {
				sourceMap: this._gl.getUniformLocation(program, "uSourceMap"),
			},
		};
		return this._copyProgram;
	}

	public getOITResolveProgram(): WebGLOITResolveProgram {
		if (this._oitResolveProgram) {
			return this._oitResolveProgram;
		}
		const program = this._createProgram(
			this._shaderSource("presentVertex"),
			this._shaderSource("oitResolveFragment"),
			"WebGLOITResolveProgram",
		);
		this._oitResolveProgram = {
			program,
			uniforms: {
				sceneColor: this._gl.getUniformLocation(program, "uSceneColor"),
				oitAccumMap: this._gl.getUniformLocation(program, "uOITAccumMap"),
				oitRevealMap: this._gl.getUniformLocation(program, "uOITRevealMap"),
			},
		};
		return this._oitResolveProgram;
	}

	public getSSAORawProgram(): WebGLSSAORawProgram {
		if (this._ssaoRawProgram) {
			return this._ssaoRawProgram;
		}
		const program = this._createProgram(
			this._shaderSource("presentVertex"),
			this._shaderSource("ssaoRawFragment"),
			"WebGLSSAORawProgram",
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
			this._shaderSource("presentVertex"),
			this._shaderSource("ssaoBlurFragment"),
			"WebGLSSAOBlurProgram",
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
			this._shaderSource("presentVertex"),
			this._shaderSource("ssaoCombineFragment"),
			"WebGLSSAOCombineProgram",
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
			this._shaderSource("presentVertex"),
			this._shaderSource("taaFragment"),
			"WebGLTAAProgram",
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
				varianceClampGamma: this._gl.getUniformLocation(program, "uVarianceClampGamma"),
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
			this._shaderSource("presentVertex"),
			this._shaderSource("postProcessStubFragment"),
			"WebGLSSRProgram",
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
			this._shaderSource("presentVertex"),
			this._shaderSource("postProcessStubFragment"),
			"WebGLVolumetricProgram",
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

	public getFogProgram(): WebGLFogProgram {
		if (this._fogProgram) {
			return this._fogProgram;
		}
		const program = this._createProgram(
			this._shaderSource("presentVertex"),
			this._shaderSource("fogFragment"),
			"WebGLFogProgram",
		);
		this._fogProgram = {
			program,
			uniforms: {
				sceneColor: this._gl.getUniformLocation(program, "uSceneColor"),
				motionDepthMap: this._gl.getUniformLocation(program, "uMotionDepthMap"),
				fogParams0: this._gl.getUniformLocation(program, "uFogParams0"),
				fogParams1: this._gl.getUniformLocation(program, "uFogParams1"),
			},
		};
		return this._fogProgram;
	}

	private _shaderSource(part: WebGLShaderPart): string {
		return this._shaderSourceFactory.getRawPart(part);
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
		fragmentMetadata?: ShaderCompileMetadata,
	): WebGLProgram {
		const gl = this._gl;
		const vertexShader = this._compileShader(
			gl.VERTEX_SHADER,
			vertexSource,
			`${label}:vertex`,
			vertexMetadata,
		);
		const fragmentShader = this._compileShader(
			gl.FRAGMENT_SHADER,
			fragmentSource,
			`${label}:fragment`,
			fragmentMetadata,
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
				sourceKind: vertexMetadata?.sourceKind ?? fragmentMetadata?.sourceKind ?? "unknown",
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
			const key = `webgl-program-validate-${label}`;
			const message =
				`WebGL program validation reported issues (${label}): ` +
				`${gl.getProgramInfoLog(program) || "no log"}`;
			this._warn(key, message);
		}

		return program;
	}

	private _compileShader(
		type: number,
		source: string,
		label: string,
		metadata?: ShaderCompileMetadata,
	): WebGLShader {
		const stage = type === this._gl.VERTEX_SHADER ? "vertex" : "fragment";
		const sourceKind =
			metadata?.sourceKind ??
			(label.startsWith("WebGLShaderMaterialProgram_") ? "custom-material" : "unknown");
		const processed = this._processShaderSource(
			source,
			stage,
			sourceKind,
			label,
			metadata?.sourceMap,
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

	private _supportsRuntimeInjects(): boolean {
		return this._shaderCompileStage !== null || this._shaderRuntime !== null;
	}

	private _collectCustomSamplerUniforms(material: ShaderMaterial): string[] {
		const uniforms = new Set<string>();
		for (const binding of material.getTextureBindings()) {
			if (binding.webglUniform.trim().length <= 0) {
				continue;
			}
			uniforms.add(binding.webglUniform);
		}
		return [...uniforms];
	}

	private _processShaderSource(
		source: string,
		stage: "vertex" | "fragment",
		sourceKind: "custom-material" | "unknown",
		label: string,
		sourceMap?: ShaderSourceSegmentMap | null,
	): ShaderProcessResult {
		const directiveSourcePath = sourceMap?.segments[0]?.sourcePath ?? label ?? "<webgl-shader>";
		if (this._shaderCompileStage) {
			return this._shaderCompileStage.compile({
				code: source,
				language: "glsl",
				stage,
				entryPoint: "main",
				label,
				sourceKind,
				sourceMap: sourceMap ?? null,
				directiveSourcePath,
			});
		}
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
			directiveSourcePath,
		});
	}

	private _reportShaderRuntimeDiagnostics(label: string, result: ShaderProcessResult): void {
		for (const diagnostic of result.diagnostics) {
			const key =
				`webgl-shader-runtime-${diagnostic.severity}-` + `${diagnostic.code}-${label}`;
			const message =
				`WebGL shader runtime ${diagnostic.severity} [${label}] ` +
				`${diagnostic.code}: ${diagnostic.message}`;
			this._warn(key, message);
		}
	}

	private _warn(key: string, message: string): void {
		this._warnCallback?.(key, message);
		Logger.warn(`[${key}] ${message}`, {
			scope: "WebGLProgramLibrary",
			onceKey: key,
		});
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
		this._sceneProgramDirectiveTag = "";
		for (const sceneProgram of this._customScenePrograms.values()) {
			this._gl.deleteProgram(sceneProgram.program);
		}
		this._customScenePrograms.clear();
		if (this._environmentProgram) {
			this._gl.deleteProgram(this._environmentProgram.program);
			this._environmentProgram = null;
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
		if (this._toneMappingProgram) {
			this._gl.deleteProgram(this._toneMappingProgram.program);
			this._toneMappingProgram = null;
		}
		if (this._colorFilterProgram) {
			this._gl.deleteProgram(this._colorFilterProgram.program);
			this._colorFilterProgram = null;
		}
		if (this._interactionOutlineProgram) {
			this._gl.deleteProgram(this._interactionOutlineProgram.program);
			this._interactionOutlineProgram = null;
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
		if (this._shadowTransmittanceProgram) {
			this._gl.deleteProgram(this._shadowTransmittanceProgram.program);
			this._shadowTransmittanceProgram = null;
		}
		if (this._copyProgram) {
			this._gl.deleteProgram(this._copyProgram.program);
			this._copyProgram = null;
		}
		if (this._oitResolveProgram) {
			this._gl.deleteProgram(this._oitResolveProgram.program);
			this._oitResolveProgram = null;
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
		if (this._fogProgram) {
			this._gl.deleteProgram(this._fogProgram.program);
			this._fogProgram = null;
		}
	}
}

function isShaderRuntime(value: unknown): value is ShaderRuntime {
	return (
		typeof value === "object" &&
		value !== null &&
		"process" in value &&
		typeof (value as { process?: unknown }).process === "function" &&
		"onDidChange" in value &&
		typeof (value as { onDidChange?: unknown }).onDidChange === "function" &&
		"getMode" in value &&
		typeof (value as { getMode?: unknown }).getMode === "function"
	);
}
